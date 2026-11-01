"""engine/playtest/agent.py
Opaque-Box Blind Playtester Agent with Prefix-Cache-Aligned Append-Only Context,
Code-Merged Permanent Rulebook, and 5-Turn Overlap Episodic Compaction.

Enforces:
- Rule 1: Zero domain-specific assumptions or internal game hints (knows ONLY title + store description)
- Rule 2: Zero regex/keyword epistemic heuristics (all mood shifts, wishes, and stop decisions are structured JSON)
- Rule 4: Rationale-First (`deliberation` at index 0 on every schema, logged to observability)
- Rule 5: Fail-fast authentic error surfacing with zero silent fallbacks
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from engine.playtest.godot_bridge import TurnObservation
from engine.playtest.models import (
    IMPRESSION_PILLAR_LABELS,
    BlindStoreCard,
    PlaytestBugReport,
    PlaytestBudgetExceededError,
    PlaytestCompactedJournal,
    PlaytestExperienceReport,
    PlaytestFieldNoteUpdate,
    PlaytestImpressionUpdate,
    PlaytestMoodNote,
    PlaytestQuestionUpdate,
    PlaytestRulebookUpdate,
    PlaytestSessionConfig,
    PlaytestTokenTracker,
    PlaytestTurnDecision,
    normalize_screen_x,
    normalize_screen_y,
)
from engine.studio.models import (
    StudioAgentConfigurationError,
    StudioAgentError,
    _sanitize_gemini_json_schema,
    bounded_thinking_config,
    is_transient_gemini_error,
)

logger = logging.getLogger("BlindPlaytesterAgent")

UNCHANGED_FRAME_MSE_THRESHOLD = 0.0005


def _uses_gemini_3_defaults(model_name: str) -> bool:
    """Returns True for Gemini 3.x models (`gemini-3.5*`, `gemini-3.6*`, `gemini-3.8*`) where the SDK
    recommends omitting manual `temperature` overrides in favor of native model defaults."""
    m_low = (model_name or "").strip().lower()
    return m_low.startswith(("gemini-3.", "gemini-3-"))


def _frame_mean_luminance(path: Optional[Path]) -> float:
    """Computes mean grayscale luminance (0..255) on a downsampled thumbnail to detect dark modal blackouts."""
    if path is None or not path.is_file():
        return 0.0
    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize((64, 36))
            raw = gray.tobytes()
            return (sum(raw) / len(raw)) if raw else 0.0
    except Exception:
        return 0.0


def _playtest_non_thinking_config(model_name: str) -> Any:
    """Returns a fast per-turn ThinkingConfig tailored to `model_name`:
    - `flash-lite`: uses `ThinkingLevel.MINIMAL`.
    - `gemini-3.8-flash` (and 3.x non-lite): uses `ThinkingLevel.LOW` for fast multimodal spatial/UI acuity.
    All player reasoning is emitted directly in the Rationale-First `deliberation` JSON property (Rule 4).
    """
    try:
        from google.genai import types
        thinking_cls = getattr(types, "ThinkingConfig", None)
        if thinking_cls is None:
            return None
        m_low = (model_name or "").strip().lower()
        thinking_lvl = getattr(types, "ThinkingLevel", None)
        if thinking_lvl is not None:
            if "flash-lite" in m_low and hasattr(thinking_lvl, "MINIMAL"):
                return thinking_cls(thinking_level=thinking_lvl.MINIMAL)
            if m_low.startswith(("gemini-3.", "gemini-3-")) and hasattr(thinking_lvl, "LOW"):
                return thinking_cls(thinking_level=thinking_lvl.LOW)
        return thinking_cls(thinking_budget=0)
    except Exception:
        return None


def _playtest_report_thinking_config(model_name: str, budget: int = 2048) -> Any:
    """Returns a deep ThinkingConfig (`ThinkingLevel.HIGH` on `gemini-3.8-flash`, `MEDIUM` on `flash-lite`)
    for the once-per-session `PlaytestExperienceReport` synthesis across up to 5 HIGH-resolution screenshots.
    """
    try:
        from google.genai import types
        thinking_cls = getattr(types, "ThinkingConfig", None)
        if thinking_cls is None:
            return None
        m_low = (model_name or "").strip().lower()
        thinking_lvl = getattr(types, "ThinkingLevel", None)
        if thinking_lvl is not None:
            if "flash-lite" in m_low:
                if hasattr(thinking_lvl, "MEDIUM"):
                    return thinking_cls(thinking_level=thinking_lvl.MEDIUM)
                if hasattr(thinking_lvl, "MINIMAL"):
                    return thinking_cls(thinking_level=thinking_lvl.MINIMAL)
            if m_low.startswith(("gemini-3.", "gemini-3-")) and hasattr(thinking_lvl, "HIGH"):
                return thinking_cls(thinking_level=thinking_lvl.HIGH)
        return bounded_thinking_config(budget=budget)
    except Exception:
        return None


def _ensure_schema_fields_required(schema_dict: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """Ensures `fields` present in `schema_dict['properties']` are listed in `schema_dict['required']`
    for live Gemini structured generation while preserving Python model defaults for unit tests.
    """
    props = schema_dict.get("properties")
    if not isinstance(props, dict):
        return schema_dict
    req = list(schema_dict.get("required") or [])
    for f_name in fields:
        if f_name in props and f_name not in req:
            req.append(f_name)
    schema_dict["required"] = req
    return schema_dict


def _read_dotenv_api_key(dotenv_path: Optional[Path] = None) -> str:
    """Reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` directly from `<PROJECT_ROOT>/.env` when not exported into `os.environ`."""
    candidates: List[Path] = []
    if dotenv_path is not None:
        candidates.append(Path(dotenv_path))
    try:
        import config as _cfg
        candidates.append(Path(_cfg.PROJECT_ROOT) / ".env")
    except Exception:
        pass
    candidates.append(Path(__file__).resolve().parent.parent.parent / ".env")

    for env_file in candidates:
        if not env_file.is_file():
            continue
        try:
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                for var_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                    if line.startswith(f"{var_name}="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except OSError:
            continue
    return ""


def create_isolated_playtest_genai_client(
    api_key: Optional[str] = None,
    *,
    dotenv_path: Optional[Path] = None,
    timeout_seconds: float = 120.0,
) -> Any:
    """Creates a dedicated `google.genai.Client` for the Blind Playtester that is NOT wrapped
    by `install_studio_gemini_guardrails`, ensuring `.studio_paused.json` and build-pipeline
    circuit breakers never block or freeze independent playtesting sessions.
    """
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not resolved_key:
        try:
            import config as _cfg
            resolved_key = str(getattr(_cfg, "GEMINI_API_KEY", "") or "").strip()
        except Exception:
            resolved_key = ""
    if not resolved_key:
        resolved_key = _read_dotenv_api_key(dotenv_path=dotenv_path)
    if not resolved_key:
        raise StudioAgentConfigurationError(
            "Cannot start Blind Playtester: GEMINI_API_KEY is not configured in environment or .env (Rule 5)."
        )
    from google import genai
    timeout_ms = int(max(timeout_seconds, 10.0) * 1000)
    return genai.Client(api_key=resolved_key, http_options={"timeout": timeout_ms})


def _read_project_godot_name(checkpoint_dir: Path) -> str:
    proj_file = checkpoint_dir / "project.godot"
    if not proj_file.is_file():
        return ""
    try:
        txt = proj_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    m = re.search(r'^config/name\s*=\s*"([^"]+)"', txt, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_blind_store_card(
    checkpoint_dir: Path,
    *,
    checkpoint_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    slice_id: Optional[str] = None,
    title_override: Optional[str] = None,
    description_override: Optional[str] = None,
) -> BlindStoreCard:
    """Extracts strictly the public store title and short premise for a slice checkpoint.
    Explicitly ignores `deliberation`, `core_rules`, `gameplay_pillars`, `game_bible.json`,
    `requirements_spec.json`, `.tscn` scene files, and `.gd` scripts so the player starts blind.
    """
    c_dir = Path(checkpoint_dir).resolve()
    if not c_dir.is_dir():
        raise StudioAgentError(f"Checkpoint directory does not exist: {c_dir}")

    manifest_path = c_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except Exception as exc:
            raise StudioAgentError(f"Failed reading checkpoint manifest '{manifest_path}': {exc}") from exc

    cp_meta_path = c_dir / "checkpoint_metadata.json"
    cp_meta: Dict[str, Any] = {}
    if cp_meta_path.is_file():
        try:
            loaded_cp = json.loads(cp_meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded_cp, dict):
                cp_meta = loaded_cp
        except Exception:
            cp_meta = {}

    creative_dir = manifest.get("creative_direction") if isinstance(manifest.get("creative_direction"), dict) else {}

    raw_title = (
        (title_override.strip() if title_override and title_override.strip() else None)
        or manifest.get("title")
        or manifest.get("game_title")
        or creative_dir.get("title")
        or _read_project_godot_name(c_dir)
        or (checkpoint_id or cp_meta.get("checkpoint_id") or c_dir.name)
    )
    raw_desc = (
        (description_override.strip() if description_override and description_override.strip() else None)
        or manifest.get("narrative_premise")
        or manifest.get("store_description")
        or creative_dir.get("narrative_premise")
        or ""
    )
    resolved_cand = (
        candidate_id
        or manifest.get("candidate_id")
        or manifest.get("source_candidate_id")
        or cp_meta.get("candidate_id")
        or None
    )
    resolved_slice = (
        slice_id
        or manifest.get("slice_id")
        or manifest.get("completed_slice_id")
        or cp_meta.get("slice_id")
        or None
    )

    return BlindStoreCard(
        deliberation="Extracted strictly public title and short store description for opaque-box player session.",
        checkpoint_id=str(checkpoint_id or c_dir.name),
        candidate_id=str(resolved_cand) if resolved_cand else None,
        slice_id=str(resolved_slice) if resolved_slice else None,
        game_title=str(raw_title).strip(),
        game_description=str(raw_desc).strip(),
    )


def compute_frame_mse(path_a: Optional[Path], path_b: Optional[Path]) -> float:
    """Computes fast normalized [0.0..1.0] RGB mean-squared difference between two screenshots."""
    if path_a is None or path_b is None or not path_a.is_file() or not path_b.is_file():
        return 1.0
    try:
        with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
            ra = im_a.convert("RGB").resize((160, 90), Image.Resampling.BILINEAR)
            rb = im_b.convert("RGB").resize((160, 90), Image.Resampling.BILINEAR)
            ba = ra.tobytes()
            bb = rb.tobytes()
            if len(ba) != len(bb) or not ba:
                return 1.0
            total_sq = 0
            for x, y in zip(ba, bb):
                d = x - y
                total_sq += d * d
            return total_sq / float(len(ba) * 255 * 255)
    except Exception:
        return 1.0


def build_blind_player_system_prompt(store_card: BlindStoreCard) -> str:
    """Minimalist, byte-0-invariant player persona prompt with zero internal hints or leading emotional cues."""
    desc_line = store_card.game_description if store_card.game_description else "(No store description provided.)"
    return (
        f"You are a discerning human player sitting down to play '{store_card.game_title}' for the first time.\n"
        f"Store Page Description: {desc_line}\n\n"
        "You have no manual, walkthrough, or knowledge of how the game was built—only what you literally see rendered "
        "on the screen (including your white pointer arrow cursor and any in-game cursor or UI) and hear in the game audio. "
        "Treat the Store Page Description strictly as promotional box copy: NEVER parrot its adjectives or credit the game "
        "for aesthetics, polish, or mechanics you cannot directly verify in the pixels and audio. Figure out how to play "
        "naturally, experiment at a human pace, notice immediately when visuals, UI, or interactions look rough or broken, "
        "and form an unvarnished, candid impression of the actual player experience.\n\n"
        "CONTROLLER INTERFACE:\n"
        "- Mouse (Normalized 0..1000 Coordinates): Both X and Y use standard normalized 0..1000 image coordinates "
        "where (0, 0) is top-left, (500, 500) is center, and (1000, 1000) is bottom-right. Set `mouse_x` (0..1000) "
        "and `mouse_y` (0..1000) to the visual spot you want to point at; your hand sweeps the cursor smoothly there. "
        "Leave `mouse_x` and `mouse_y` null when keeping the cursor at its current position. Set `mouse_button_action` "
        "(`none`, `left_click`, `right_click`, `double_click`, `drag_left`, `hold_left`, `release_left`, `wheel_up`, `wheel_down`) "
        "for clicks, drags, or scrolling. For `drag_left`, if your cursor is not already on the object you want to grab, "
        "set `drag_from_x` and `drag_from_y` (0..1000) to where the drag starts and `mouse_x`/`mouse_y` (0..1000) to where you drag it.\n"
        "- Keyboard: `hold_keys` holds keys across `duration_ms` (350..2000ms); `tap_keys` taps keys briefly.\n"
        "- Visual detail: Set `inspect_high_res_next=true` early (e.g., on Turn 1 so Turn 2 renders in full detail) and "
        "whenever inspecting new UI panels, icons, sprite edges, or small text—do NOT praise graphics or UI as 'clean', "
        "'polished', or 'legible' before checking them at HIGH resolution.\n\n"
        "VISUAL & UI CRAFT HONESTY (LOOK WITH CRITICAL HUMAN EYES):\n"
        "- Never grade visuals or UI on a curve. Before describing `graphics_and_visuals` or UI overlays positively, actively "
        "scan the screen for universal rendering and layout defects:\n"
        "  1. Sprite & icon slicing / empty-slot glitches: sprites sliced mid-graphic or showing only a fractional corner, "
        "stray floating edge slivers bleeding in from adjacent sprite-sheet/atlas frames, empty/blank icon boxes inside open "
        "UI menus or hotbars where sibling slots show icons, or icons stretched, squished, or clipped inside UI slots/buttons.\n"
        "  2. UI layout & typography bugs: text spilling outside its panel/box background, labels chopped off mid-word or colliding "
        "with neighboring elements, overlapping HUD panels, or tiny/low-contrast unreadable text.\n"
        "  3. Placeholder shapes, texture seams & style clashes: crude flat solid-color geometric placeholder shapes (unshaded "
        "triangles, rectangles, or circles) standing in for world structures or icons alongside rendered artwork, harsh unblended "
        "rectangular texture boxes pasted inside irregular polygons, visible grid/tile seams, or large untextured flat terrain polygons.\n"
        "- Whenever any of these defects appear on screen, call them out plainly in `impression_updates` and log each distinct "
        "issue in `bug_reports` (`category='visual_glitch'` or `'input_ui_bug'`).\n\n"
        "PLAYER FIELD NOTES, QUESTIONS & RULEBOOK:\n"
        "- `turn_log_entry`, `field_note_updates`, `question_updates`, and `rulebook_updates` evaluate ONLY what ALREADY "
        "changed between the previous turn's screen/audio and the current screen/audio—NEVER what you expect or hope your "
        "upcoming `motor_action` will do.\n"
        "- `field_note_updates`: Jot brief notes on what you are looking at (screen layout, exact HUD numbers/counters, objects, "
        "UI indicators) when you notice something new or when an element's state visibly changes.\n"
        "- `question_updates`: Naturally ask questions as you explore ('What am I looking at?', 'What does X do?', "
        "'Why is Y like that?') with `status='open'`. Keep a question `status='open'` on the turn you test it, and update "
        "that `question_key` to `status='answered'` ONLY on a subsequent turn once comparing the previous screen vs current "
        "screen (or audio) proves the result. If later turns show counters or objects did NOT actually change, reopen the "
        "question (`status='open'`, `answer=null`) or revise the answer.\n"
        "- `rulebook_updates`: Your `motor_action` executes AFTER your turn decision—never record a confirmed rule on the "
        "same turn you try an action! Wait until the following turn and cite `observed_change_from_prior_turn` first. "
        "If an earlier Rulebook entry turns out to be wrong because numbers/objects stayed unchanged, set `action='retract'` "
        "to remove it (or overwrite it with 'does nothing / unresponsive'), log a `bug_report` if an on-screen prompt or "
        "control failed to work after 2-3 tries, and switch to a different interaction rather than repeating the same input.\n"
        "- `current_plan`: After evaluating what actually happened, state what you will test next across the coming steps.\n\n"
        "MOOD, EVOLVING IMPRESSIONS, BUGS, WISHES & STOPPING:\n"
        "- `current_mood` & `mood_shift`: Keep `current_mood` steady across routine exploration and ordinary clicks—do NOT "
        "flip back and forth on minor steps. Leave `mood_shift` null on almost all turns, populating it ONLY at genuine "
        "emotional turning points (e.g., delight at a real breakthrough, surprise, mounting confusion, boredom, or frustration when stuck).\n"
        "- `impression_updates`: Leave empty (`[]`) on most turns. Build and revise your candid impression across "
        "`graphics_and_visuals`, `gameplay_and_feel`, `sound_and_audio`, `intuitiveness_and_controls`, `goals_and_pacing`, "
        "and `world_and_atmosphere`. Early impressions are tentative—wait until HIGH-resolution inspection before praising "
        "visual/UI polish, revise any earlier note immediately if closer inspection or new screens reveal defects, and never repeat unchanged notes.\n"
        "- `bug_reports`: Log bugs, visual/sprite glitches, overflowing/truncated/overlapping UI, unresponsive controls, or broken "
        "behaviors as soon as you spot them, recording each distinct issue once without duplicating earlier bug entries.\n"
        "- `submit_wish`: File a concrete player wish whenever you hit UX/design friction or wish something worked better—never duplicate wishes.\n"
        "- `end_playtest`: Populate once you complete the available objectives/content or confirm after testing distinct ideas that you cannot progress any further."
    )


class BlindPlaytesterAgent:
    """Prefix-cache-aligned Blind Playtester cognitive loop."""

    def __init__(
        self,
        *,
        store_card: BlindStoreCard,
        config: Optional[PlaytestSessionConfig] = None,
        client: Optional[Any] = None,
        token_tracker: Optional[PlaytestTokenTracker] = None,
    ) -> None:
        self.store_card = store_card
        self.config = config or PlaytestSessionConfig()
        self.client = client if client is not None else create_isolated_playtest_genai_client()
        self.token_tracker = token_tracker or PlaytestTokenTracker(
            max_session_tokens=self.config.max_session_tokens
        )

        self.system_prompt: str = build_blind_player_system_prompt(self.store_card)
        raw_turn_schema = _sanitize_gemini_json_schema(copy.deepcopy(PlaytestTurnDecision.model_json_schema()))
        _ensure_schema_fields_required(
            raw_turn_schema,
            [
                "field_note_updates",
                "question_updates",
                "rulebook_updates",
                "bug_reports",
                "impression_updates",
                "current_mood",
            ],
        )
        t_props = raw_turn_schema.get("properties") or {}
        if isinstance(t_props.get("rulebook_updates"), dict) and isinstance(t_props["rulebook_updates"].get("items"), dict):
            _ensure_schema_fields_required(
                t_props["rulebook_updates"]["items"],
                ["observed_change_from_prior_turn", "action"],
            )
        if isinstance(t_props.get("question_updates"), dict) and isinstance(t_props["question_updates"].get("items"), dict):
            _ensure_schema_fields_required(
                t_props["question_updates"]["items"],
                ["observed_change_from_prior_turn", "answer", "status"],
            )
        self.turn_schema: Dict[str, Any] = raw_turn_schema
        self.compaction_schema: Dict[str, Any] = _sanitize_gemini_json_schema(
            copy.deepcopy(PlaytestCompactedJournal.model_json_schema())
        )
        self.report_schema: Dict[str, Any] = _ensure_schema_fields_required(
            _sanitize_gemini_json_schema(copy.deepcopy(PlaytestExperienceReport.model_json_schema())),
            [
                "visual_and_ui_defects_observed",
                "key_highlights",
                "key_pain_points",
                "bugs_and_glitches_observed",
                "unanswered_questions_and_mysteries",
            ],
        )

        # Permanent deterministic player notebook (NEVER summarized away across compactions)
        self.rulebook: Dict[str, str] = {}
        self.field_notes: Dict[str, str] = {}
        self.questions: Dict[str, Dict[str, Any]] = {}
        self.bugs_found: Dict[str, Dict[str, Any]] = {}
        self.impressions: Dict[str, str] = {}
        self.filed_wish_summaries: List[str] = []
        self.mood_history: List[Dict[str, Any]] = []
        # Episodic history summary (refreshed every compaction_interval_turns)
        self.compacted_journal: Optional[PlaytestCompactedJournal] = None
        self.compaction_count: int = 0
        self.covered_up_to_turn: int = 0

        # Current epoch frozen header + strictly append-only turn lines
        self.epoch_header_text: str = self._render_epoch_header_text()
        self.epoch_turn_lines: List[Tuple[int, str]] = []

        # Verbatim immediate continuity state
        self.current_plan: str = "Observe the opening screen and figure out how to start or interact."
        self.current_mood: str = "curious"
        self.inspect_high_res_next: bool = False
        self._has_seen_high_res_frame: bool = False
        self._opening_screenshot_path: Optional[Path] = None
        self._distinct_mid_screenshot: Optional[Tuple[int, float, Path]] = None
        self._session_screenshots: List[Tuple[int, Path, bool]] = []
        self.prev_observation: Optional[TurnObservation] = None
        self._last_click_spot: Optional[Tuple[float, float, str]] = None
        self._same_spot_click_streak: int = 0
        self._last_key_signature: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
        self._repeated_key_streak: int = 0

    def reset_memory(self) -> None:
        """Clears all player memory, rulebook, notes, questions, bugs, impressions, and turn tape for a clean Reset."""
        self.rulebook.clear()
        self.field_notes.clear()
        self.questions.clear()
        self.bugs_found.clear()
        self.impressions.clear()
        self.filed_wish_summaries.clear()
        self.mood_history.clear()
        self.compacted_journal = None
        self.compaction_count = 0
        self.covered_up_to_turn = 0
        self.epoch_header_text = self._render_epoch_header_text()
        self.epoch_turn_lines.clear()
        self.current_plan = "Observe the opening screen and figure out how to start or interact."
        self.current_mood = "curious"
        self.inspect_high_res_next = False
        self._has_seen_high_res_frame = False
        self._opening_screenshot_path = None
        self._distinct_mid_screenshot = None
        self._session_screenshots.clear()
        self.prev_observation = None
        self._last_click_spot = None
        self._same_spot_click_streak = 0
        self._last_key_signature = None
        self._repeated_key_streak = 0

    def _render_epoch_header_text(self) -> str:
        """Renders `[Block 1A — Permanent Notebook (Rulebook, Notes, Questions, Bugs, Impressions)]`
        + `[Block 1B — Compacted Episodic Journal]`.
        Frozen across all turns of the current epoch so byte-0 prefix caching stays 100% invariant.
        """
        lines: List[str] = ["=== PLAYER PERMANENT HOW-TO-PLAY RULEBOOK (CONFIRMED SO FAR) ==="]
        if not self.rulebook:
            lines.append("- (No controls or rules confirmed yet.)")
        else:
            for k, v in self.rulebook.items():
                lines.append(f"- {k}: {v}")

        lines.append("")
        lines.append("=== PLAYER FIELD NOTES (WHAT YOU HAVE OBSERVED ON SCREEN) ===")
        if not self.field_notes:
            lines.append("- (No screen observations recorded yet.)")
        else:
            for topic, obs_note in self.field_notes.items():
                lines.append(f"- {topic}: {obs_note}")

        lines.append("")
        lines.append("=== PLAYER QUESTIONS & DISCOVERED ANSWERS ===")
        if not self.questions:
            lines.append("- (No player questions logged yet.)")
        else:
            for q_key, q_entry in self.questions.items():
                q_txt = q_entry.get("question", "")
                if q_entry.get("status") == "answered" and q_entry.get("answer"):
                    lines.append(f"- [ANSWERED] ({q_key}) {q_txt} -> {q_entry['answer']}")
                else:
                    lines.append(f"- [OPEN] ({q_key}) {q_txt}")

        lines.append("")
        lines.append("=== BUGS & GLITCHES SPOTTED SO FAR ===")
        if not self.bugs_found:
            lines.append("- (No bugs or glitches logged yet.)")
        else:
            for b_key, b_entry in self.bugs_found.items():
                lines.append(
                    f"- [{b_entry.get('severity', 'major').upper()} | {b_entry.get('category', 'bug')}] "
                    f"({b_key}) {b_entry.get('summary', '')}: {b_entry.get('details', '')}"
                )

        lines.append("")
        lines.append("=== PLAYER RUNNING IMPRESSIONS (REVISE IF THINGS CHANGE — DO NOT REPEAT UNCHANGED) ===")
        if not self.impressions:
            lines.append("- (No experience pillar impressions recorded yet.)")
        else:
            for pillar, note in self.impressions.items():
                label = IMPRESSION_PILLAR_LABELS.get(pillar, pillar)
                lines.append(f"- {label} ({pillar}): {note}")

        lines.append("")
        lines.append("=== PLAYER COMPACTED EPISODIC MEMORY (OLDER TURNS) ===")
        if self.compacted_journal is None:
            lines.append("(Beginning of play session — no prior compacted episodes.)")
        else:
            j = self.compacted_journal
            if j.episodic_story_and_world_chronicle:
                lines.append("Chronicle:")
                for item in j.episodic_story_and_world_chronicle:
                    lines.append(f"  * {item}")
            if j.world_landmarks_map:
                lines.append("World & Landmarks Mental Map:")
                for item in j.world_landmarks_map:
                    lines.append(f"  * {item}")
            if j.failed_attempts_and_dead_ends:
                lines.append("Already Tried / Dead Ends:")
                for item in j.failed_attempts_and_dead_ends:
                    lines.append(f"  * {item}")
            if j.open_questions_and_hypotheses:
                lines.append("Open Questions & Things To Test:")
                for item in j.open_questions_and_hypotheses:
                    lines.append(f"  * {item}")
            if j.fun_and_aesthetic_impressions:
                lines.append(f"Running Player Impressions: {j.fun_and_aesthetic_impressions}")

        lines.append("")
        lines.append("=== RECENT APPEND-ONLY TURN LOG ===")
        return "\n".join(lines)

    def build_turn_contents(self, turn_index: int, obs: TurnObservation) -> Tuple[List[Any], Dict[str, Any]]:
        """Constructs `[EpochHeaderContent, ...AppendOnlyTurnLines, EphemeralTailContent]`.
        Guarantees `contents[:-1]` is strictly append-only with zero historical image/audio bytes.
        """
        from google.genai import types

        contents: List[Any] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self.epoch_header_text)],
            )
        ]
        for _, tape_line in self.epoch_turn_lines:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=tape_line)],
                )
            )

        # Determine visual & perceptual deltas against prior turn
        frame_mse = 1.0
        cursor_delta = 999.0
        if self.prev_observation is not None:
            frame_mse = compute_frame_mse(self.prev_observation.screenshot_path, obs.screenshot_path)
            dx = obs.cursor_x - self.prev_observation.cursor_x
            dy = obs.cursor_y - self.prev_observation.cursor_y
            cursor_delta = (dx * dx + dy * dy) ** 0.5

        screen_unchanged = (
            self.prev_observation is not None
            and frame_mse < UNCHANGED_FRAME_MSE_THRESHOLD
            and cursor_delta < 1.0
        )
        cur_audio_silent = bool(obs.audio_is_silent or obs.wav_path is None)
        audio_became_silent = (
            self.config.enable_audio_perception
            and self.prev_observation is not None
            and not bool(self.prev_observation.audio_is_silent or self.prev_observation.wav_path is None)
            and cur_audio_silent
        )

        norm_cx = normalize_screen_x(obs.cursor_x)
        norm_cy = normalize_screen_y(obs.cursor_y)

        # Build Ephemeral Tail (never appended into future prefix turns)
        tail_parts: List[Any] = []
        status_lines = [
            f"=== CURRENT MOMENT (Turn #{turn_index} | Game Time {obs.sim_time_sec:.2f}s) ===",
            f"Your Cursor Position: ({norm_cx:.0f}, {norm_cy:.0f}) in 0..1000 normalized screen coords ((0,0)=top-left, (500,500)=center, (1000,1000)=bottom-right)",
            f"Your Current Mood: {self.current_mood}",
            f"What You Were Actively Doing (Current Plan): {self.current_plan}",
        ]
        if self.rulebook:
            rb_brief = " | ".join(f"{k}: {v}" for k, v in list(self.rulebook.items())[-10:])
            status_lines.append(
                f"Your Confirmed Rulebook So Far (retract with action='retract' if later screens disprove an entry): {rb_brief}"
            )
        if self.field_notes:
            fn_brief = " | ".join(f"{k}: {v}" for k, v in list(self.field_notes.items())[-10:])
            status_lines.append(f"Your Field Notes So Far: {fn_brief}")
        if self.questions:
            open_qs = [
                f"[{k}] {q['question']}"
                for k, q in self.questions.items()
                if q.get("status") != "answered"
            ]
            ans_qs = [
                f"[{k}] {q['question']} -> {q.get('answer', '')}"
                for k, q in self.questions.items()
                if q.get("status") == "answered"
            ]
            if open_qs:
                status_lines.append(f"Open Questions You Are Trying To Answer: {' | '.join(open_qs[-8:])}")
            if ans_qs:
                status_lines.append(f"Questions Already Answered: {' | '.join(ans_qs[-6:])}")
        if self.bugs_found:
            bug_brief = "; ".join(
                f"[{b.get('category')}/{b.get('severity')}] ({k}) {b.get('summary')}"
                for k, b in list(self.bugs_found.items())[-10:]
            )
            status_lines.append(f"Bugs & Glitches Already Logged (do NOT duplicate): {bug_brief}")
        if self.impressions:
            imp_brief = " | ".join(f"{k}: {v}" for k, v in self.impressions.items())
            status_lines.append(f"Your Recorded Impressions So Far (only update if changed): {imp_brief}")
        if self.filed_wish_summaries:
            status_lines.append(
                f"Wishes You Already Filed (do NOT duplicate): {'; '.join(self.filed_wish_summaries[-10:])}"
            )
        if obs.interrupted_early:
            status_lines.append("Note: Your last action burst stopped early because the screen changed dramatically.")
        if screen_unchanged:
            status_lines.append(
                "Visual Delta Note: The screen is visually identical to your previous turn — your last input caused no visible screen change."
            )
        auto_high_res_for_streak = (
            self._same_spot_click_streak >= 3 or self._repeated_key_streak >= 3
        )
        auto_high_res_for_visual_verify = (
            not self._has_seen_high_res_frame and "graphics_and_visuals" in self.impressions
        )
        if auto_high_res_for_visual_verify and not screen_unchanged:
            status_lines.append(
                "Visual Acuity Note: You recorded an early `graphics_and_visuals` impression before seeing a HIGH-resolution "
                "frame. This turn's screen is now attached at HIGH resolution—closely inspect sprite edges for cut-offs or "
                "bleeding neighbor slivers, UI icon proportions, text spilling outside or clipping panel boxes, and harsh "
                "texture/polygon seams; if any defects are visible, immediately revise `graphics_and_visuals` and log `bug_reports`."
            )
        if self._same_spot_click_streak >= 3 and self._last_click_spot is not None:
            spot_x, spot_y, spot_act = self._last_click_spot
            if spot_act in ("left_click", "right_click", "double_click"):
                lead_clause = (
                    f"Action Continuity Note: You have clicked nearly the same spot ({spot_x:.0f}, {spot_y:.0f}) "
                    f"{self._same_spot_click_streak} turns in a row."
                )
            else:
                lead_clause = (
                    f"Action Continuity Note: You have repeated '{spot_act}' on nearly the same spot ({spot_x:.0f}, {spot_y:.0f}) "
                    f"{self._same_spot_click_streak} turns in a row."
                )
            escalation_clause = (
                f" STOP REPEATING '{spot_act}' AT ({spot_x:.0f}, {spot_y:.0f}): {self._same_spot_click_streak} consecutive attempts "
                "is conclusive—immediately try a different control, key, UI element, or location (or populate end_playtest "
                "with 'stuck_cannot_progress' if every distinct interaction is unresponsive)."
                if self._same_spot_click_streak >= 5
                else ""
            )
            status_lines.append(
                f"{lead_clause} Check carefully whether exact HUD numbers, counters, objects, or UI actually changed "
                "between Previous turn screen and Current screen—if nothing progressed, do NOT assume slow or hidden "
                "progress: retract/reopen any mistaken Rulebook or Answered Question entries, log a bug_report if an "
                f"on-screen prompt failed to respond, and try a different step.{escalation_clause}"
            )
        elif self._repeated_key_streak >= 4 and self._last_key_signature is not None:
            hk, tk = self._last_key_signature
            key_desc = ", ".join(list(hk) + list(tk))
            status_lines.append(
                f"Action Continuity Note: You have repeated the exact same key input ({key_desc}) "
                f"{self._repeated_key_streak} turns in a row. Check carefully whether counters, position, or UI actually "
                "changed—if nothing progressed, retract any mistaken assumptions and try a different control or interaction."
            )
        if audio_became_silent:
            status_lines.append("Perceptual Change Note: Game audio stopped and is now silent.")

        tail_parts.append(types.Part.from_text(text="\n".join(status_lines)))

        # Attach Previous Frame (LOW res) + Current Frame (MEDIUM/HIGH or deduplicated LOW)
        cur_bytes = obs.screenshot_path.read_bytes()
        if screen_unchanged:
            tail_parts.append(types.Part.from_text(text="Current screen (unchanged from previous turn):"))
            tail_parts.append(
                types.Part.from_bytes(
                    data=cur_bytes,
                    mime_type="image/png",
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                )
            )
            cur_res_label = "LOW_DEDUP"
        else:
            if self.prev_observation is not None and self.prev_observation.screenshot_path.is_file():
                prev_bytes = self.prev_observation.screenshot_path.read_bytes()
                tail_parts.append(types.Part.from_text(text="Previous turn screen (for motion/change comparison):"))
                tail_parts.append(
                    types.Part.from_bytes(
                        data=prev_bytes,
                        mime_type="image/png",
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                    )
                )
            use_high_res = bool(
                self.inspect_high_res_next or auto_high_res_for_streak or auto_high_res_for_visual_verify
            )
            cur_res = (
                types.MediaResolution.MEDIA_RESOLUTION_HIGH
                if use_high_res
                else types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
            )
            cur_res_label = "HIGH" if use_high_res else "MEDIUM"
            if use_high_res:
                self._has_seen_high_res_frame = True
            tail_parts.append(
                types.Part.from_text(
                    text=f"Current screen ({cur_res_label} resolution — white pointer arrow cursor is at ({norm_cx:.0f}, {norm_cy:.0f}) in 0..1000 coords):"
                )
            )
            tail_parts.append(
                types.Part.from_bytes(
                    data=cur_bytes,
                    mime_type="image/png",
                    media_resolution=cur_res,
                )
            )

        audio_attached = False
        if self.config.enable_audio_perception and not obs.audio_is_silent and obs.wav_path is not None and obs.wav_path.is_file():
            wav_bytes = obs.wav_path.read_bytes()
            if wav_bytes:
                tail_parts.append(types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"))
                audio_attached = True

        contents.append(types.Content(role="user", parts=tail_parts))
        meta = {
            "frame_mse": round(frame_mse, 6),
            "screen_unchanged": screen_unchanged,
            "image_resolution": cur_res_label,
            "audio_attached": audio_attached,
        }
        return contents, meta

    def decide_turn(
        self,
        turn_index: int,
        obs: TurnObservation,
    ) -> Tuple[PlaytestTurnDecision, str, Dict[str, Any]]:
        """Runs a single player perception-deliberation-action turn and updates append-only memory."""
        from google.genai import types

        self.token_tracker.check_budget()
        contents, perception_meta = self.build_turn_contents(turn_index, obs)

        gen_cfg_kwargs: Dict[str, Any] = {
            "system_instruction": self.system_prompt,
            "response_mime_type": "application/json",
            "response_schema": self.turn_schema,
            "thinking_config": _playtest_non_thinking_config(self.config.model_name),
        }
        if not _uses_gemini_3_defaults(self.config.model_name):
            gen_cfg_kwargs["temperature"] = 0.55
        gen_cfg = types.GenerateContentConfig(**gen_cfg_kwargs)

        resp = self._call_gemini_resilient(
            model=self.config.model_name,
            contents=contents,
            config=gen_cfg,
            label=f"BlindPlaytester.turn_{turn_index}",
        )
        tok_stats = self.token_tracker.record_response(resp)
        raw_text = getattr(resp, "text", None) or ""
        if not raw_text.strip():
            raise StudioAgentError(f"BlindPlaytester received empty response on turn {turn_index} (Rule 5).")

        decision = PlaytestTurnDecision.model_validate_json(raw_text)

        # 1. Deterministically merge permanent player notebook updates (never lost across compactions)
        self._apply_field_note_updates(decision.field_note_updates)
        self._apply_question_updates(turn_index, decision.question_updates)
        self._apply_rulebook_updates(decision.rulebook_updates)
        self._apply_bug_reports(turn_index, obs, decision.bug_reports)
        self._apply_impression_updates(decision.impression_updates)
        if decision.submit_wish is not None:
            w_headline = f"[{decision.submit_wish.category}] {' '.join(decision.submit_wish.summary.split())}"
            if w_headline not in self.filed_wish_summaries:
                self.filed_wish_summaries.append(w_headline)

        # 2. Update verbatim continuity fields & capture live mood transitions
        self.current_plan = decision.current_plan.strip()
        self.inspect_high_res_next = bool(decision.motor_action.inspect_high_res_next)

        prev_mood_norm = self.current_mood.strip().lower()
        cand_mood = ""
        cand_note = ""
        cand_delib = decision.deliberation
        is_explicit_shift = False
        if decision.mood_shift is not None:
            cand_mood = " ".join(decision.mood_shift.mood.split())
            cand_note = " ".join(decision.mood_shift.note.split())
            cand_delib = decision.mood_shift.deliberation
            is_explicit_shift = True
        elif decision.current_mood and decision.current_mood.strip():
            cand_mood = " ".join(decision.current_mood.split())
            cand_note = " ".join(decision.turn_log_entry.split()) or f"Player mood shifted to {cand_mood}."

        cand_norm = cand_mood.lower()
        if not cand_norm or cand_norm == prev_mood_norm:
            decision.mood_shift = None
        else:
            last_shift_turn = int(self.mood_history[-1]["turn_index"]) if self.mood_history else -999
            turns_since_last_shift = turn_index - last_shift_turn
            recent_mood_norms = {str(m["mood"]).strip().lower() for m in self.mood_history[-3:]}
            if self.mood_history:
                recent_mood_norms.add("curious")
            has_high_salience_signal = bool(
                decision.bug_reports or decision.submit_wish is not None or decision.end_playtest is not None
            )
            is_rapid_bounce_back = cand_norm in recent_mood_norms and turns_since_last_shift < 8
            is_too_frequent = (
                len(self.mood_history) >= 2
                and turns_since_last_shift < 5
                and not has_high_salience_signal
            )
            if is_rapid_bounce_back or is_too_frequent:
                decision.mood_shift = None
            else:
                self.current_mood = cand_mood
                final_note = cand_note or f"Player mood shifted to {cand_mood}."
                if not is_explicit_shift:
                    decision.mood_shift = PlaytestMoodNote(
                        deliberation=cand_delib,
                        mood=cand_mood,
                        note=final_note,
                    )
                else:
                    decision.mood_shift.mood = cand_mood
                    decision.mood_shift.note = final_note
                self.mood_history.append({
                    "turn_index": turn_index,
                    "sim_time_sec": obs.sim_time_sec,
                    "mood": cand_mood,
                    "note": final_note,
                })

        # Track consecutive repeated mouse-spot or key actions so ephemeral tail can break zombie loops
        act = decision.motor_action
        if act.mouse_button_action in (
            "left_click",
            "right_click",
            "double_click",
            "hold_left",
            "drag_left",
            "wheel_up",
            "wheel_down",
        ):
            click_x = act.mouse_x if act.mouse_x is not None else normalize_screen_x(obs.cursor_x)
            click_y = act.mouse_y if act.mouse_y is not None else normalize_screen_y(obs.cursor_y)
            if (
                self._last_click_spot is not None
                and act.mouse_button_action == self._last_click_spot[2]
                and ((click_x - self._last_click_spot[0]) ** 2 + (click_y - self._last_click_spot[1]) ** 2) ** 0.5 <= 30.0
            ):
                self._same_spot_click_streak += 1
            else:
                self._last_click_spot = (click_x, click_y, act.mouse_button_action)
                self._same_spot_click_streak = 1
            self._last_key_signature = None
            self._repeated_key_streak = 0
        else:
            self._last_click_spot = None
            self._same_spot_click_streak = 0
            key_sig = (
                tuple(sorted(k.strip().upper() for k in act.hold_keys if k.strip())),
                tuple(sorted(k.strip().upper() for k in act.tap_keys if k.strip())),
            )
            if key_sig[0] or key_sig[1]:
                if self._last_key_signature == key_sig:
                    self._repeated_key_streak += 1
                else:
                    self._last_key_signature = key_sig
                    self._repeated_key_streak = 1
            else:
                self._last_key_signature = None
                self._repeated_key_streak = 0

        # 3. Format compact single-line append-only turn tape entry & track distinct frames for report audit
        tape_line = self._format_append_tape_line(turn_index, obs, decision)
        self.epoch_turn_lines.append((turn_index, tape_line))
        if obs.screenshot_path is not None and obs.screenshot_path.is_file():
            is_high_res = perception_meta.get("image_resolution") == "HIGH"
            self._session_screenshots.append((turn_index, obs.screenshot_path, is_high_res))
            if self._opening_screenshot_path is None:
                self._opening_screenshot_path = obs.screenshot_path
            else:
                mse_vs_open = compute_frame_mse(self._opening_screenshot_path, obs.screenshot_path)
                if mse_vs_open >= 0.005 and (
                    self._distinct_mid_screenshot is None or mse_vs_open >= self._distinct_mid_screenshot[1]
                ):
                    self._distinct_mid_screenshot = (turn_index, mse_vs_open, obs.screenshot_path)
        self.prev_observation = obs

        combined_stats = {**tok_stats, **perception_meta}
        return decision, tape_line, combined_stats

    def should_compact(self) -> bool:
        return len(self.epoch_turn_lines) >= self.config.compaction_interval_turns

    def compact_journal(self) -> Tuple[PlaytestCompactedJournal, int, Dict[str, int]]:
        """Compacts older turns in `epoch_turn_lines` into `PlaytestCompactedJournal` while:
        1. Keeping `self.rulebook`, `self.field_notes`, `self.questions`, `self.bugs_found`, and `self.impressions`
           100% deterministic in `[Block 1A]` (never LLM-summarized), and
        2. Carrying the latest `compaction_overlap_turns` (default 5) raw lines verbatim into the
           new epoch's `[Block 2]` so immediate spatial/task momentum is never lost at the seam.
        """
        from google.genai import types

        self.token_tracker.check_budget()
        overlap = min(
            max(1, self.config.compaction_overlap_turns),
            max(1, len(self.epoch_turn_lines) - 1),
        )
        lines_to_compact = self.epoch_turn_lines[:-overlap]
        carryover_lines = list(self.epoch_turn_lines[-overlap:])
        if not lines_to_compact:
            lines_to_compact = list(self.epoch_turn_lines)
            carryover_lines = []

        up_to_turn = lines_to_compact[-1][0]

        prior_journal_json = (
            self.compacted_journal.model_dump_json(indent=2)
            if self.compacted_journal is not None
            else "(None yet — first compaction pass.)"
        )
        rulebook_lines = (
            "\n".join(f"- {k}: {v}" for k, v in self.rulebook.items())
            if self.rulebook
            else "(None confirmed yet.)"
        )
        impressions_lines = (
            "\n".join(
                f"- {IMPRESSION_PILLAR_LABELS.get(k, k)} ({k}): {v}"
                for k, v in self.impressions.items()
            )
            if self.impressions
            else "(None recorded yet.)"
        )
        questions_lines = (
            "\n".join(
                f"- [{q.get('status', 'open').upper()}] ({k}) {q.get('question', '')}"
                + (f" -> {q.get('answer')}" if q.get("answer") else "")
                for k, q in self.questions.items()
            )
            if self.questions
            else "(None recorded yet.)"
        )
        bugs_lines = (
            "\n".join(
                f"- [{b.get('severity', 'major').upper()}] ({k}) {b.get('summary', '')}"
                for k, b in self.bugs_found.items()
            )
            if self.bugs_found
            else "(None recorded yet.)"
        )
        folded_text = "\n".join(line for _, line in lines_to_compact)

        prompt = (
            f"You are consolidating your own player memory while playing '{self.store_card.game_title}'.\n"
            "Your confirmed How-To-Play controls/rules, Field Notes, Questions, Bugs, and 6-pillar Running Impressions "
            "are ALREADY stored separately (shown below for context—do NOT worry about duplicating every keybind or repeating pillar notes).\n"
            "Focus your compacted journal on:\n"
            "1. Chronological story, character, and exploration progress so far,\n"
            "2. Your spatial mental map of notable places, objects, doors, or NPCs,\n"
            "3. Failed attempts and dead-ends already tested so you don't repeat loops,\n"
            "4. Open mysteries or hypotheses you still want to test,\n"
            "5. Concise overall vibe takeaway so far.\n\n"
            f"PERMANENT RULEBOOK (STORED SEPARATELY):\n{rulebook_lines}\n\n"
            f"PERMANENT QUESTIONS & ANSWERS (STORED SEPARATELY):\n{questions_lines}\n\n"
            f"PERMANENT BUGS SPOTTED (STORED SEPARATELY):\n{bugs_lines}\n\n"
            f"PERMANENT PILLAR IMPRESSIONS (STORED SEPARATELY):\n{impressions_lines}\n\n"
            f"PRIOR COMPACTED JOURNAL:\n{prior_journal_json}\n\n"
            f"NEW COMPLETED TURNS TO FOLD INTO YOUR JOURNAL ( up through Turn #{up_to_turn} ):\n{folded_text}"
        )

        compact_cfg_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": self.compaction_schema,
            "thinking_config": _playtest_non_thinking_config(self.config.model_name),
        }
        if not _uses_gemini_3_defaults(self.config.model_name):
            compact_cfg_kwargs["temperature"] = 0.3
        gen_cfg = types.GenerateContentConfig(**compact_cfg_kwargs)
        resp = self._call_gemini_resilient(
            model=self.config.model_name,
            contents=prompt,
            config=gen_cfg,
            label=f"BlindPlaytester.compact_{self.compaction_count + 1}",
        )
        tok_stats = self.token_tracker.record_response(resp)
        raw_text = getattr(resp, "text", None) or ""
        if not raw_text.strip():
            raise StudioAgentError("BlindPlaytester compaction returned empty response (Rule 5).")

        updated_journal = PlaytestCompactedJournal.model_validate_json(raw_text)
        self.compacted_journal = updated_journal
        self.compaction_count += 1
        self.covered_up_to_turn = up_to_turn

        # Start new prefix-cached epoch with updated [Block 1A Notebook + Block 1B Journal] + 5-turn verbatim overlap
        self.epoch_header_text = self._render_epoch_header_text()
        self.epoch_turn_lines = carryover_lines
        return updated_journal, up_to_turn, tok_stats

    def _select_stratified_report_screenshots(
        self,
        final_screenshot_path: Optional[Path] = None,
    ) -> List[Tuple[str, Path]]:
        """Selects up to 5 stratified, visually distinct screenshots across the session so that dark modal
        overlays or single end-of-session screens never mask open-world terrain, entities, or UI menus:
        1. Opening screen (`turn_1` / `_opening_screenshot_path`)
        2. Early exploration (~25% session index, preferring non-blackout frames with mean luminance >= 35)
        3. Mid-session open-world gameplay (~55% session index, preferring mean luminance >= 35)
        4. Peak distinct UI / HIGH-resolution inspection frame (highest MSE vs opening frame)
        5. Final screen (`final_screenshot_path` or last recorded turn)
        """
        valid_records: List[Tuple[int, Path, bool]] = [
            (t_idx, p, hi)
            for t_idx, p, hi in self._session_screenshots
            if p is not None and p.is_file()
        ]
        if not valid_records:
            fallback: List[Tuple[str, Path]] = []
            if self._opening_screenshot_path is not None and self._opening_screenshot_path.is_file():
                fallback.append(("Opening screen of your session", self._opening_screenshot_path))
            if self._distinct_mid_screenshot is not None and self._distinct_mid_screenshot[2].is_file():
                mid_t, _, mid_p = self._distinct_mid_screenshot
                fallback.append((f"Mid-session screen (Turn #{mid_t})", mid_p))
            if final_screenshot_path is not None and final_screenshot_path.is_file():
                fallback.append(("Final screen at the end of your session", final_screenshot_path))
            return fallback

        open_turn, open_path, _ = valid_records[0]
        if self._opening_screenshot_path is not None and self._opening_screenshot_path.is_file():
            open_path = self._opening_screenshot_path

        def _pick_near_fraction(frac: float, label_prefix: str) -> Optional[Tuple[str, Path]]:
            if len(valid_records) < 3:
                return None
            target_idx = max(1, min(len(valid_records) - 2, int(round(frac * (len(valid_records) - 1)))))
            window_lo = max(1, target_idx - 2)
            window_hi = min(len(valid_records) - 1, target_idx + 3)
            window = valid_records[window_lo:window_hi] or [valid_records[target_idx]]
            bright = [rec for rec in window if _frame_mean_luminance(rec[1]) >= 35.0]
            pool = bright if bright else window
            best_rec = min(pool, key=lambda rec: abs(valid_records.index(rec) - target_idx))
            return (f"{label_prefix} (Turn #{best_rec[0]})", best_rec[1])

        selected: List[Tuple[str, Path]] = [
            (f"Opening screen of your session (Turn #{open_turn})", open_path)
        ]

        early_pick = _pick_near_fraction(0.25, "Early exploration screen")
        if early_pick is not None:
            selected.append(early_pick)

        mid_pick = _pick_near_fraction(0.55, "Mid-session gameplay screen")
        if mid_pick is not None:
            selected.append(mid_pick)

        # Peak distinct UI / HIGH-resolution inspection frame
        best_hi_rec: Optional[Tuple[int, float, Path]] = None
        best_any_rec: Optional[Tuple[int, float, Path]] = None
        for t_idx, p, is_hi in valid_records[1:]:
            mse_val = compute_frame_mse(open_path, p)
            if mse_val < 0.005:
                continue
            if is_hi and (best_hi_rec is None or mse_val >= best_hi_rec[1]):
                best_hi_rec = (t_idx, mse_val, p)
            if best_any_rec is None or mse_val >= best_any_rec[1]:
                best_any_rec = (t_idx, mse_val, p)

        peak_rec = best_hi_rec or best_any_rec or self._distinct_mid_screenshot
        if peak_rec is not None and peak_rec[2].is_file():
            selected.append((f"Distinct UI / inspection screen (Turn #{peak_rec[0]})", peak_rec[2]))

        if final_screenshot_path is not None and final_screenshot_path.is_file():
            selected.append(("Final screen at the end of your session", final_screenshot_path))
        elif len(valid_records) >= 2:
            last_t, last_p, _ = valid_records[-1]
            selected.append((f"Final screen at the end of your session (Turn #{last_t})", last_p))

        return selected

    def generate_experience_report(
        self,
        *,
        end_status: str,
        end_outcome: Optional[str] = None,
        closing_thoughts: Optional[str] = None,
        final_screenshot_path: Optional[Path] = None,
        final_audio_path: Optional[Path] = None,
        wishes_filed: Optional[List[Dict[str, Any]]] = None,
        representative_screenshots: Optional[List[Tuple[str, Path]]] = None,
    ) -> Tuple[PlaytestExperienceReport, Dict[str, int]]:
        """Synthesizes a complete, honest first-person `PlaytestExperienceReport` at the end of a stopped or
        completed session, reusing the active prefix-cached `[EpochHeader, ...AppendOnlyTurnLines]` prefix
        and inspecting HIGH-resolution representative screenshots with `config.report_model_name`.
        """
        from google.genai import types

        contents: List[Any] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self.epoch_header_text)],
            )
        ]
        for _, tape_line in self.epoch_turn_lines:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=tape_line)],
                )
            )

        rulebook_summary = (
            "\n".join(f"- {k}: {v}" for k, v in self.rulebook.items())
            if self.rulebook
            else "- (No confirmed controls/rules recorded.)"
        )
        field_notes_summary = (
            "\n".join(f"- {k}: {v}" for k, v in self.field_notes.items())
            if self.field_notes
            else "- (No field notes recorded.)"
        )
        answered_q_lines = [
            f"- {q.get('question', '')} -> {q.get('answer', '')}"
            for q in self.questions.values()
            if q.get("status") == "answered" and q.get("answer")
        ]
        open_q_lines = [
            f"- {q.get('question', '')}"
            for q in self.questions.values()
            if q.get("status") != "answered"
        ]
        questions_summary = ""
        if answered_q_lines:
            questions_summary += "Answered During Play:\n" + "\n".join(answered_q_lines) + "\n"
        if open_q_lines:
            questions_summary += "Still Unanswered At End Of Session:\n" + "\n".join(open_q_lines) + "\n"
        if not questions_summary:
            questions_summary = "- (No explicit questions tracked.)\n"

        bugs_summary = (
            "\n".join(
                f"- [{b.get('severity', 'major').upper()} | {b.get('category', 'bug')}] "
                f"(Turn #{b.get('turn_index', '?')}) {b.get('summary', '')}: {b.get('details', '')}"
                for b in self.bugs_found.values()
            )
            if self.bugs_found
            else "- (No explicit bugs logged during play—mention any glitches or broken behaviors you noticed if applicable.)"
        )
        impressions_summary = (
            "\n".join(
                f"- {IMPRESSION_PILLAR_LABELS.get(k, k)} ({k}): {v}"
                for k, v in self.impressions.items()
            )
            if self.impressions
            else "- (No sparse pillar updates were recorded yet—synthesize directly from your turn experience.)"
        )
        wishes_lines: List[str] = []
        for idx, w in enumerate(wishes_filed or [], start=1):
            w_cat = str(w.get("category") or "general")
            w_sum = str(w.get("summary") or "")
            w_rat = str(w.get("rationale") or "")
            wishes_lines.append(f"{idx}. [{w_cat}] {w_sum} — {w_rat}")
        wishes_block = "\n".join(wishes_lines) if wishes_lines else "- (No formal wishes filed during the session.)"

        if self.mood_history:
            mood_arc = " -> ".join(
                f"{m['mood']} (T#{m['turn_index']})" for m in self.mood_history[-12:]
            )
            mood_summary = f"curious (T#0) -> {mood_arc} | Current: {self.current_mood}"
        else:
            mood_summary = f"{self.current_mood} (no shifts recorded)"

        end_context_bits = [f"Session End Status: {end_status}"]
        if end_outcome:
            end_context_bits.append(f"Player End Outcome: {end_outcome}")
        if closing_thoughts:
            end_context_bits.append(f"Immediate Closing Note: {closing_thoughts}")

        report_prompt = (
            "=== END OF PLAYTEST SESSION — WRITE YOUR HONEST PLAYER EXPERIENCE REPORT ===\n"
            + "\n".join(end_context_bits)
            + "\n"
            f"Player Mood Trajectory: {mood_summary}\n\n"
            f"Confirmed Controls & Rules You Discovered:\n{rulebook_summary}\n\n"
            f"Your Field Notes & Screen Observations:\n{field_notes_summary}\n\n"
            f"Your Questions & Hypotheses:\n{questions_summary}\n"
            f"Bugs & Glitches Spotted Along The Way:\n{bugs_summary}\n\n"
            f"Your Sparse Running Pillar Impressions:\n{impressions_summary}\n\n"
            f"Wishes You Filed During Play:\n{wishes_block}\n\n"
            "Now write your complete, candid, first-person `PlaytestExperienceReport` covering your overall session "
            "experience, concrete `visual_and_ui_defects_observed`, graphics & visual clarity, gameplay & feel, sound & "
            "audio feedback, intuitiveness & onboarding, goals/pacing & actual progression accomplished, world & atmosphere, "
            "concrete highlights, concrete pain points, bugs & glitches observed, any unanswered questions/mysteries, and "
            "your honest 1-10 `overall_fun_rating`.\n"
            "BE CANDID, NOT PROMOTIONAL:\n"
            "- VISUAL & UI PIXEL AUDIT FIRST: Closely inspect the attached HIGH-resolution session screenshot(s) BEFORE "
            "writing `graphics_and_visuals`, and override any overly generous early `Sparse Running Pillar Impressions` or "
            "Store Page Description marketing language if high-resolution pixels show defects. Enumerate every visible "
            "visual/UI flaw in `visual_and_ui_defects_observed`, checking specifically for: (1) cut-off sprites, quarter-corner "
            "slices of an object, or stray floating edge slivers bleeding in from adjacent sprite-sheet/atlas frames, "
            "(2) blank/empty UI icon boxes (missing icons inside recipe, inventory, or hotbar slots where other slots show icons) "
            "or stretched, squished, or clipped icons inside UI slots/buttons, (3) text spilling outside panel/box borders, "
            "colliding with siblings, or chopped off mid-word, (4) overlapping HUD panels or unreadable low-contrast labels, "
            "and (5) harsh unblended rectangular texture boxes pasted inside irregular polygons, missing building/world "
            "textures, or crude flat solid-color geometric placeholder triangles/rectangles clashing with detailed raster art. "
            "Never call graphics or UI 'clean', 'polished', or 'highly legible' when such defects appear on screen!\n"
            "- PROGRESSION, AUDIO & AGENCY AUDIT: Report only what visibly and audibly progressed rather than what you "
            "hoped an action did. Explicitly call out if objectives stayed stuck at 0, if repeated actions had no effect, "
            "if audio was silent or cut out, or if mechanics played themselves without meaningful player decisions, and "
            "score `overall_fun_rating` strictly by the rubric (never 7-10 when core progression stalled, audio was silent, "
            "agency was missing, or visual/UI rendering had glaring defects)."
        )

        tail_parts: List[Any] = [types.Part.from_text(text=report_prompt)]
        candidate_shots: List[Tuple[str, Path]] = []
        if representative_screenshots:
            candidate_shots.extend(representative_screenshots)
        else:
            candidate_shots.extend(self._select_stratified_report_screenshots(final_screenshot_path))

        seen_shot_paths: set[str] = set()
        dedup_shots: List[Tuple[str, Path]] = []
        for label, shot_path in candidate_shots:
            if shot_path is None or not shot_path.is_file():
                continue
            resolved_key = str(shot_path.resolve())
            if resolved_key in seen_shot_paths:
                continue
            seen_shot_paths.add(resolved_key)
            dedup_shots.append((label, shot_path))

        for label, shot_path in dedup_shots[:5]:
            img_bytes = shot_path.read_bytes()
            if img_bytes:
                tail_parts.append(types.Part.from_text(text=f"{label} (HIGH resolution for visual/UI craft audit):"))
                tail_parts.append(
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png",
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                    )
                )
        if self.config.enable_audio_perception and final_audio_path is not None and final_audio_path.is_file():
            wav_bytes = final_audio_path.read_bytes()
            if wav_bytes:
                tail_parts.append(types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"))

        contents.append(types.Content(role="user", parts=tail_parts))

        report_model = (self.config.report_model_name or self.config.model_name).strip()
        report_cfg_kwargs: Dict[str, Any] = {
            "system_instruction": self.system_prompt,
            "response_mime_type": "application/json",
            "response_schema": self.report_schema,
            "thinking_config": _playtest_report_thinking_config(report_model),
        }
        if not _uses_gemini_3_defaults(report_model):
            report_cfg_kwargs["temperature"] = 0.35
        gen_cfg = types.GenerateContentConfig(**report_cfg_kwargs)
        resp = self._call_gemini_resilient(
            model=report_model,
            contents=contents,
            config=gen_cfg,
            label="BlindPlaytester.experience_report",
        )
        tok_stats = self.token_tracker.record_response(resp)
        raw_text = getattr(resp, "text", None) or ""
        if not raw_text.strip():
            raise StudioAgentError("BlindPlaytester experience report returned empty response (Rule 5).")

        report = PlaytestExperienceReport.model_validate_json(raw_text)
        return report, tok_stats

    def export_memory_snapshot(self) -> Dict[str, Any]:
        """Serializes the permanent player notebook and Compacted Episodic Journal for SQLite & UI inspection."""
        return {
            "rulebook": dict(self.rulebook),
            "field_notes": dict(self.field_notes),
            "questions": copy.deepcopy(self.questions),
            "bugs_found": copy.deepcopy(self.bugs_found),
            "impressions": dict(self.impressions),
            "mood_history": list(self.mood_history),
            "filed_wish_summaries": list(self.filed_wish_summaries),
            "current_plan": self.current_plan,
            "current_mood": self.current_mood,
            "compaction_count": self.compaction_count,
            "covered_up_to_turn": self.covered_up_to_turn,
            "compacted_journal": (
                self.compacted_journal.model_dump() if self.compacted_journal is not None else None
            ),
            "active_epoch_turn_count": len(self.epoch_turn_lines),
            "epoch_header_preview": self.epoch_header_text,
        }

    def _apply_field_note_updates(self, updates: List[PlaytestFieldNoteUpdate]) -> None:
        for upd in updates:
            topic = " ".join(upd.topic.split())
            obs_txt = " ".join(upd.observation.split())
            if topic and obs_txt:
                self.field_notes[topic] = obs_txt

    def _apply_question_updates(self, turn_index: int, updates: List[PlaytestQuestionUpdate]) -> None:
        for upd in updates:
            q_key = "_".join(upd.question_key.strip().lower().split())
            q_txt = " ".join(upd.question.split())
            ans_txt = " ".join(upd.answer.split()) if upd.answer else None
            if not q_key or not q_txt:
                continue
            existing = self.questions.get(q_key)
            asked_turn = existing.get("asked_turn", turn_index) if existing else turn_index
            if upd.status == "open" and not ans_txt:
                status = "open"
                resolved_answer = None
                resolved_answered_turn = None
            elif upd.status == "answered" or bool(ans_txt):
                status = "answered"
                resolved_answer = ans_txt if ans_txt else (existing.get("answer") if existing else None)
                resolved_answered_turn = turn_index
            else:
                status = "open"
                resolved_answer = None
                resolved_answered_turn = None
            self.questions[q_key] = {
                "question_key": q_key,
                "question": q_txt,
                "status": status,
                "answer": resolved_answer,
                "asked_turn": asked_turn,
                "answered_turn": resolved_answered_turn,
            }

    def _apply_bug_reports(self, turn_index: int, obs: TurnObservation, reports: List[PlaytestBugReport]) -> None:
        for rep in reports:
            b_key = "_".join(rep.bug_key.strip().lower().split())
            summary = " ".join(rep.summary.split())
            details = " ".join(rep.details.split())
            if not b_key or not summary:
                continue
            if b_key in self.bugs_found:
                # Allow updating severity/details if player learned more about the same bug, preserving initial turn & screenshot
                prev = self.bugs_found[b_key]
                prev["summary"] = summary
                prev["details"] = details or prev.get("details", "")
                prev["severity"] = rep.severity
                prev["category"] = rep.category
                continue
            self.bugs_found[b_key] = {
                "bug_key": b_key,
                "category": rep.category,
                "severity": rep.severity,
                "summary": summary,
                "details": details,
                "turn_index": turn_index,
                "sim_time_sec": obs.sim_time_sec,
                "video_timestamp_sec": round(obs.tick / 60.0, 3),
                "screenshot_path": str(obs.screenshot_path) if obs.screenshot_path else None,
            }

    def _apply_rulebook_updates(self, updates: List[PlaytestRulebookUpdate]) -> None:
        for upd in updates:
            key = " ".join(upd.key_or_topic.split())
            val = " ".join(upd.behavior.split())
            if not key:
                continue
            matched_existing_key = next(
                (ek for ek in self.rulebook.keys() if ek.lower() == key.lower()),
                None,
            )
            if getattr(upd, "action", "confirm") == "retract":
                if matched_existing_key is not None:
                    del self.rulebook[matched_existing_key]
                continue
            if val:
                target_key = matched_existing_key if matched_existing_key is not None else key
                self.rulebook[target_key] = val

    def _apply_impression_updates(self, updates: List[PlaytestImpressionUpdate]) -> None:
        for upd in updates:
            pillar = str(upd.pillar).strip()
            val = " ".join(upd.impression.split())
            if pillar and val:
                self.impressions[pillar] = val

    def _format_append_tape_line(
        self,
        turn_index: int,
        obs: TurnObservation,
        decision: PlaytestTurnDecision,
    ) -> str:
        norm_cx = normalize_screen_x(obs.cursor_x)
        norm_cy = normalize_screen_y(obs.cursor_y)
        act = decision.motor_action
        act_bits: List[str] = []
        if act.hold_keys:
            act_bits.append(f"hold={','.join(act.hold_keys)}")
        if act.tap_keys:
            act_bits.append(f"tap={','.join(act.tap_keys)}")
        if act.mouse_button_action == "drag_left" and (act.drag_from_x is not None or act.drag_from_y is not None):
            fx = act.drag_from_x if act.drag_from_x is not None else norm_cx
            fy = act.drag_from_y if act.drag_from_y is not None else norm_cy
            tx = act.mouse_x if act.mouse_x is not None else norm_cx
            ty = act.mouse_y if act.mouse_y is not None else norm_cy
            act_bits.append(f"drag=({fx:.0f},{fy:.0f})->({tx:.0f},{ty:.0f})")
        elif act.mouse_x is not None or act.mouse_y is not None:
            tx = act.mouse_x if act.mouse_x is not None else norm_cx
            ty = act.mouse_y if act.mouse_y is not None else norm_cy
            act_bits.append(f"move_to=({tx:.0f},{ty:.0f})")
        if act.mouse_button_action != "none":
            act_bits.append(f"mouse={act.mouse_button_action}")
        act_bits.append(f"{act.duration_ms}ms")
        act_str = " ".join(act_bits)

        extras: List[str] = []
        if decision.field_note_updates:
            fn_str = "; ".join(
                f"{' '.join(u.topic.split())}={' '.join(u.observation.split())}"
                for u in decision.field_note_updates
            )
            extras.append(f"NOTE+[{fn_str}]")
        if decision.question_updates:
            q_parts: List[str] = []
            for u in decision.question_updates:
                qk = "_".join(u.question_key.strip().lower().split())
                if u.status == "answered" and u.answer:
                    q_parts.append(f"ANSWERED({qk})={' '.join(u.answer.split())}")
                else:
                    q_parts.append(f"OPEN({qk})={' '.join(u.question.split())}")
            extras.append(f"QUESTION+[{'; '.join(q_parts)}]")
        if decision.rulebook_updates:
            rb_str = "; ".join(
                (
                    f"RETRACTED({' '.join(u.key_or_topic.split())})"
                    if getattr(u, "action", "confirm") == "retract"
                    else f"{' '.join(u.key_or_topic.split())}={' '.join(u.behavior.split())}"
                )
                for u in decision.rulebook_updates
            )
            extras.append(f"RULEBOOK+[{rb_str}]")
        if decision.bug_reports:
            b_str = "; ".join(
                f"{u.category}/{u.severity}:{' '.join(u.summary.split())}"
                for u in decision.bug_reports
            )
            extras.append(f"BUG+[{b_str}]")
        if decision.impression_updates:
            imp_str = "; ".join(
                f"{u.pillar}={' '.join(u.impression.split())}"
                for u in decision.impression_updates
            )
            extras.append(f"IMPRESSION+[{imp_str}]")
        if decision.mood_shift is not None:
            extras.append(
                f"MOOD[{' '.join(decision.mood_shift.mood.split())}: {' '.join(decision.mood_shift.note.split())}]"
            )
        if decision.submit_wish is not None:
            extras.append(f"WISH[{decision.submit_wish.category}: {' '.join(decision.submit_wish.summary.split())}]")
        extra_suffix = (" | " + " | ".join(extras)) if extras else ""
        clean_log = " ".join(decision.turn_log_entry.split())

        return (
            f"[Turn {turn_index} | t={obs.sim_time_sec:.1f}s | cursor=({norm_cx:.0f},{norm_cy:.0f}) | {act_str}] "
            f"{clean_log}{extra_suffix}"
        )

    def _call_gemini_resilient(
        self,
        *,
        model: str,
        contents: Any,
        config: Any,
        label: str,
        max_retries: int = 3,
    ) -> Any:
        for attempt in range(max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except PlaytestBudgetExceededError:
                raise
            except Exception as exc:
                if attempt < max_retries and is_transient_gemini_error(exc):
                    delay = 1.2 * (2 ** attempt) + random.uniform(0.0, 0.3)
                    logger.warning("%s transient Gemini error (attempt %d/%d), retrying in %.2fs: %s", label, attempt + 1, max_retries + 1, delay, exc)
                    time.sleep(delay)
                    continue
                raise StudioAgentError(f"{label} failed: {exc}") from exc
