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
    PlaytestBudgetExceededError,
    PlaytestCompactedJournal,
    PlaytestExperienceReport,
    PlaytestImpressionUpdate,
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
        f"You are a human player sitting down to play '{store_card.game_title}' for the first time.\n"
        f"Store Page Description: {desc_line}\n\n"
        "You have no manual, walkthrough, or knowledge of how the game was built—only what you see on the "
        "screen (including your white pointer arrow cursor and any in-game cursor or UI) and hear in the game audio. "
        "Figure out how to play naturally, experiment with the world at a human pace, and form an honest, nuanced "
        "impression of the whole player experience: graphics & visual clarity, gameplay & feel, sound & audio feedback, "
        "intuitiveness & controls, goals & sense of accomplishment, and world/story atmosphere.\n\n"
        "CONTROLLER INTERFACE:\n"
        "- Mouse (Normalized 0..1000 Coordinates): Both X and Y use standard normalized 0..1000 image coordinates "
        "where (0, 0) is top-left, (500, 500) is center, and (1000, 1000) is bottom-right. Set `mouse_x` (0..1000) "
        "and `mouse_y` (0..1000) to the visual spot you want to point at; your hand sweeps the cursor smoothly there. "
        "Leave `mouse_x` and `mouse_y` null when keeping the cursor at its current position. Set `mouse_button_action` "
        "(`none`, `left_click`, `right_click`, `double_click`, `drag_left`, `hold_left`, `release_left`, `wheel_up`, `wheel_down`) "
        "for clicks, drags, or scrolling. For `drag_left`, if your cursor is not already on the object you want to grab, "
        "set `drag_from_x` and `drag_from_y` (0..1000) to where the drag starts and `mouse_x`/`mouse_y` (0..1000) to where you drag it.\n"
        "- Keyboard: `hold_keys` holds keys across `duration_ms` (350..2000ms); `tap_keys` taps keys briefly.\n"
        "- Visual detail: Set `inspect_high_res_next=true` only when small text or UI details are hard to read.\n\n"
        "KEEPING YOUR MENTAL NOTES:\n"
        "- `rulebook_updates`: Whenever you confirm what an input or game element does (including confirming "
        "a tested key does nothing), record it so your permanent Rulebook stays accurate.\n"
        "- `current_plan`: Brief note of what you are actively doing across the next few steps.\n"
        "- `turn_log_entry`: Single compact line noting what happened since your last input and what you are trying next.\n\n"
        "SPARSE IMPRESSIONS, MOOD SHIFTS, WISHES & STOPPING (DO NOT REPEAT YOURSELF):\n"
        "- `impression_updates`: Leave empty (`[]`) on most turns! As you play, sparsely build and refine your impression across "
        "`graphics_and_visuals`, `gameplay_and_feel`, `sound_and_audio`, `intuitiveness_and_controls`, `goals_and_pacing`, "
        "and `world_and_atmosphere`. Emit an update ONLY when you genuinely form a new impression or revise an earlier one—never "
        "repeat or rephrase what is already recorded in your Running Impressions.\n"
        "- `mood_shift`: Leave `null` on most turns. Emit a `mood_shift` ONLY when your mood or understanding "
        "genuinely changes, explaining why in 1-2 short, direct sentences without repeating earlier notes.\n"
        "- `submit_wish`: Leave `null` on most turns. File a concrete player wish whenever you hit real control/UX confusion, "
        "missing visual/audio feedback, gameplay friction, or something you genuinely wish worked better—never file duplicate wishes.\n"
        "- `end_playtest`: Populate when you complete the available objectives/content or conclude you cannot progress any further."
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
        self.turn_schema: Dict[str, Any] = _sanitize_gemini_json_schema(
            copy.deepcopy(PlaytestTurnDecision.model_json_schema())
        )
        self.compaction_schema: Dict[str, Any] = _sanitize_gemini_json_schema(
            copy.deepcopy(PlaytestCompactedJournal.model_json_schema())
        )
        self.report_schema: Dict[str, Any] = _sanitize_gemini_json_schema(
            copy.deepcopy(PlaytestExperienceReport.model_json_schema())
        )

        # Permanent deterministic How-To-Play Rulebook & Pillar Impressions (NEVER summarized away)
        self.rulebook: Dict[str, str] = {}
        self.impressions: Dict[str, str] = {}
        self.filed_wish_summaries: List[str] = []
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
        self.prev_observation: Optional[TurnObservation] = None

    def reset_memory(self) -> None:
        """Clears all player memory, rulebook, impressions, and append-only turn tape for a clean Reset."""
        self.rulebook.clear()
        self.impressions.clear()
        self.filed_wish_summaries.clear()
        self.compacted_journal = None
        self.compaction_count = 0
        self.covered_up_to_turn = 0
        self.epoch_header_text = self._render_epoch_header_text()
        self.epoch_turn_lines.clear()
        self.current_plan = "Observe the opening screen and figure out how to start or interact."
        self.current_mood = "curious"
        self.inspect_high_res_next = False
        self.prev_observation = None

    def _render_epoch_header_text(self) -> str:
        """Renders `[Block 1A — Permanent Rulebook & Impressions] + [Block 1B — Compacted Episodic Journal]`.
        Frozen across all turns of the current epoch so byte-0 prefix caching stays 100% invariant.
        """
        lines: List[str] = ["=== PLAYER PERMANENT HOW-TO-PLAY RULEBOOK (CONFIRMED SO FAR) ==="]
        if not self.rulebook:
            lines.append("- (No controls or rules confirmed yet.)")
        else:
            for k, v in self.rulebook.items():
                lines.append(f"- {k}: {v}")

        lines.append("")
        lines.append("=== PLAYER RUNNING IMPRESSIONS (SPARSE — DO NOT REPEAT UNCHANGED NOTES) ===")
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

        # Determine visual delta against prior turn
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
        if not self.config.enable_audio_perception:
            status_lines.append("Audio Note: Audio perception disabled for this session.")
        elif obs.audio_is_silent or obs.wav_path is None:
            status_lines.append("Audio Note: Game audio was silent during this step.")
        else:
            status_lines.append(f"Audio Note: Game audio clip attached below (RMS={obs.audio_rms:.3f}).")

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
            cur_res = (
                types.MediaResolution.MEDIA_RESOLUTION_HIGH
                if self.inspect_high_res_next
                else types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
            )
            cur_res_label = "HIGH" if self.inspect_high_res_next else "MEDIUM"
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

        gen_cfg = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            response_mime_type="application/json",
            response_schema=self.turn_schema,
            temperature=0.55,
            thinking_config=bounded_thinking_config(256),
        )

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

        # 1. Deterministically merge confirmed rulebook & sparse pillar impression updates (never lost across compactions)
        self._apply_rulebook_updates(decision.rulebook_updates)
        self._apply_impression_updates(decision.impression_updates)
        if decision.submit_wish is not None:
            w_headline = f"[{decision.submit_wish.category}] {' '.join(decision.submit_wish.summary.split())}"
            if w_headline not in self.filed_wish_summaries:
                self.filed_wish_summaries.append(w_headline)

        # 2. Update verbatim continuity fields
        self.current_plan = decision.current_plan.strip()
        self.inspect_high_res_next = bool(decision.motor_action.inspect_high_res_next)
        if decision.mood_shift is not None:
            self.current_mood = decision.mood_shift.mood.strip()

        # 3. Format compact single-line append-only turn tape entry
        tape_line = self._format_append_tape_line(turn_index, obs, decision)
        self.epoch_turn_lines.append((turn_index, tape_line))
        self.prev_observation = obs

        combined_stats = {**tok_stats, **perception_meta}
        return decision, tape_line, combined_stats

    def should_compact(self) -> bool:
        return len(self.epoch_turn_lines) >= self.config.compaction_interval_turns

    def compact_journal(self) -> Tuple[PlaytestCompactedJournal, int, Dict[str, int]]:
        """Compacts older turns in `epoch_turn_lines` into `PlaytestCompactedJournal` while:
        1. Keeping `self.rulebook` 100% deterministic in `[Block 1A]` (never LLM-summarized), and
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
        folded_text = "\n".join(line for _, line in lines_to_compact)

        prompt = (
            f"You are consolidating your own player memory while playing '{self.store_card.game_title}'.\n"
            "Your confirmed How-To-Play controls/rules AND your structured 6-pillar Running Impressions are ALREADY "
            "stored separately (shown below for context—do NOT worry about duplicating every keybind or repeating pillar notes).\n"
            "Focus your compacted journal on:\n"
            "1. Chronological story, character, and exploration progress so far,\n"
            "2. Your spatial mental map of notable places, objects, doors, or NPCs,\n"
            "3. Failed attempts and dead-ends already tested so you don't repeat loops,\n"
            "4. Open mysteries or hypotheses you still want to test,\n"
            "5. Concise overall vibe takeaway so far.\n\n"
            f"PERMANENT RULEBOOK (STORED SEPARATELY):\n{rulebook_lines}\n\n"
            f"PERMANENT PILLAR IMPRESSIONS (STORED SEPARATELY):\n{impressions_lines}\n\n"
            f"PRIOR COMPACTED JOURNAL:\n{prior_journal_json}\n\n"
            f"NEW COMPLETED TURNS TO FOLD INTO YOUR JOURNAL ( up through Turn #{up_to_turn} ):\n{folded_text}"
        )

        gen_cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=self.compaction_schema,
            temperature=0.3,
            thinking_config=bounded_thinking_config(1024),
        )
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

        # Start new prefix-cached epoch with updated [Block 1A Rulebook + Block 1B Journal] + 5-turn verbatim overlap
        self.epoch_header_text = self._render_epoch_header_text()
        self.epoch_turn_lines = carryover_lines
        return updated_journal, up_to_turn, tok_stats

    def generate_experience_report(
        self,
        *,
        end_status: str,
        end_outcome: Optional[str] = None,
        closing_thoughts: Optional[str] = None,
        final_screenshot_path: Optional[Path] = None,
        wishes_filed: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[PlaytestExperienceReport, Dict[str, int]]:
        """Synthesizes a complete, honest first-person `PlaytestExperienceReport` at the end of a stopped or
        completed session, reusing the active prefix-cached `[EpochHeader, ...AppendOnlyTurnLines]` prefix.
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

        end_context_bits = [f"Session End Status: {end_status}"]
        if end_outcome:
            end_context_bits.append(f"Player End Outcome: {end_outcome}")
        if closing_thoughts:
            end_context_bits.append(f"Immediate Closing Note: {closing_thoughts}")

        report_prompt = (
            "=== END OF PLAYTEST SESSION — WRITE YOUR HONEST PLAYER EXPERIENCE REPORT ===\n"
            + "\n".join(end_context_bits)
            + "\n"
            f"Final Player Mood: {self.current_mood}\n\n"
            f"Confirmed Controls & Rules You Discovered:\n{rulebook_summary}\n\n"
            f"Your Sparse Running Pillar Impressions:\n{impressions_summary}\n\n"
            f"Wishes You Filed During Play:\n{wishes_block}\n\n"
            "Now write your complete, candid, first-person `PlaytestExperienceReport` covering your overall session "
            "experience, graphics & visual clarity, gameplay & feel, sound & audio feedback, intuitiveness & onboarding, "
            "goals/pacing & sense of accomplishment, world & atmosphere, concrete highlights, concrete pain points, "
            "and your honest 1-10 `overall_fun_rating` based purely on what you actually experienced on screen and in audio."
        )

        tail_parts: List[Any] = [types.Part.from_text(text=report_prompt)]
        if final_screenshot_path is not None and final_screenshot_path.is_file():
            img_bytes = final_screenshot_path.read_bytes()
            if img_bytes:
                tail_parts.append(types.Part.from_text(text="Final screen at the end of your session:"))
                tail_parts.append(
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png",
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                    )
                )

        contents.append(types.Content(role="user", parts=tail_parts))

        gen_cfg = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            response_mime_type="application/json",
            response_schema=self.report_schema,
            temperature=0.35,
            thinking_config=bounded_thinking_config(1024),
        )
        resp = self._call_gemini_resilient(
            model=self.config.model_name,
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
        """Serializes the permanent Rulebook, sparse Pillar Impressions, and Compacted Episodic Journal for SQLite & UI inspection."""
        return {
            "rulebook": dict(self.rulebook),
            "impressions": dict(self.impressions),
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

    def _apply_rulebook_updates(self, updates: List[PlaytestRulebookUpdate]) -> None:
        for upd in updates:
            key = upd.key_or_topic.strip()
            val = upd.behavior.strip()
            if key and val:
                self.rulebook[key] = val

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
        if decision.rulebook_updates:
            rb_str = "; ".join(
                f"{' '.join(u.key_or_topic.split())}={' '.join(u.behavior.split())}"
                for u in decision.rulebook_updates
            )
            extras.append(f"RULEBOOK+[{rb_str}]")
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
