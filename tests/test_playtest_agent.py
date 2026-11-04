"""tests/test_playtest_agent.py
Unit and epistemic integrity tests for the Blind Playtester Agent (`engine/playtest/agent.py` & `models.py`).

Verifies:
1. Rationale-First schema ordering (`deliberation` at index 0) across all playtest Pydantic models (Rule 4).
2. Strict Opaque-Box Blindfold: `extract_blind_store_card` and `build_blind_player_system_prompt` expose ONLY
   public title + short store description, stripping internal rules, Game Bible, requirements, and scripts (Rule 1).
3. Prefix-Cache Byte-0 Invariance: `contents[:-1]` is strictly append-only text across an epoch with zero
   historical image/audio binary parts.
4. Ephemeral Tail & Multimodal Perception: `LOW` previous + `MEDIUM`/`HIGH` current frames, no-op frame
   deduplication (`LOW_DEDUP`), and silent vs non-silent `.wav` attachment.
5. Zero-Amnesia Compaction: Deterministic Python `rulebook` (`[Block 1A]`) + `PlaytestCompactedJournal` (`[Block 1B]`)
   + 5-turn verbatim tail overlap (`[Block 2]`) + `current_plan` continuity.
6. Complete decoupling from Studio build circuit breakers (`TOKEN_USAGE_TRACKER` & `.studio_paused.json`).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from google.genai import types
from PIL import Image, ImageDraw
import pytest

from engine.playtest.agent import (
    BlindPlaytesterAgent,
    _playtest_non_thinking_config,
    build_blind_player_system_prompt,
    compute_frame_mse,
    create_isolated_playtest_genai_client,
    extract_blind_store_card,
)
from engine.playtest.godot_bridge import TurnObservation, write_pcm16_to_wav
from engine.playtest.models import (
    DEFAULT_PLAYTEST_MODEL,
    BlindStoreCard,
    PlaytestBudgetExceededError,
    PlaytestBugReport,
    PlaytestCompactedJournal,
    PlaytestEndDecision,
    PlaytestExperienceReport,
    PlaytestFieldNoteUpdate,
    PlaytestImpressionUpdate,
    PlaytestMoodNote,
    PlaytestMotorAction,
    PlaytestQuestionUpdate,
    PlaytestRulebookUpdate,
    PlaytestSessionConfig,
    PlaytestSessionControlRequest,
    PlaytestSessionStartRequest,
    PlaytestTokenTracker,
    PlaytestTurnDecision,
    PlaytestWishDispatchRequest,
    PlaytestWishDraft,
)
from engine.studio.models import (
    TOKEN_USAGE_TRACKER,
    BlindPlaytestCouncilTurn,
    BlindPlaytestCouncilVerdict,
)


class _QueuedFakeGeminiClient:
    """Deterministic fake `google.genai.Client` recording exact `contents` and `config` payloads."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self._queue = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        assert self._queue, f"Unexpected extra Gemini call for model={model}"
        spec = self._queue.pop(0)
        self.calls.append({"model": model, "contents": contents, "config": config})
        usage = SimpleNamespace(
            prompt_token_count=spec.get("prompt_tokens", 1200),
            cached_content_token_count=spec.get("cached_tokens", 900),
            candidates_token_count=spec.get("output_tokens", 110),
            thoughts_token_count=spec.get("thought_tokens", 40),
        )
        return SimpleNamespace(text=json.dumps(spec["payload"]), usage_metadata=usage)


def _make_png(path: Path, bg_rgb: tuple[int, int, int], marker_x: int = 100) -> Path:
    img = Image.new("RGB", (320, 180), bg_rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(marker_x, 40), (marker_x + 40, 90)], fill=(240, 220, 80))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def _make_obs(
    tmp_path: Path,
    idx: int,
    *,
    bg_rgb: tuple[int, int, int] = (20, 30, 50),
    marker_x: int = 100,
    cursor_xy: tuple[float, float] = (640.0, 360.0),
    audio_is_silent: bool = False,
) -> TurnObservation:
    png_path = _make_png(tmp_path / f"obs_{idx:03d}.png", bg_rgb=bg_rgb, marker_x=marker_x)
    wav_path = None
    rms = 0.0
    if not audio_is_silent:
        pcm = (b"\x00\x20\x00\xe0" * 800)
        wav_path = write_pcm16_to_wav(pcm, tmp_path / f"obs_{idx:03d}.wav")
        rms = 0.18
    return TurnObservation(
        observation_index=idx,
        tick=idx * 45,
        sim_time_sec=round((idx * 45) / 60.0, 2),
        cursor_x=cursor_xy[0],
        cursor_y=cursor_xy[1],
        screenshot_path=png_path,
        preview_path=png_path,
        wav_path=wav_path,
        audio_rms=rms,
        audio_peak=rms * 1.4,
        audio_is_silent=audio_is_silent,
        interrupted_early=False,
    )


def _turn_payload(
    turn_idx: int,
    *,
    plan: str = "Explore the starting room and test interactions.",
    rulebook_updates: List[Dict[str, str]] | None = None,
    mood_shift: Dict[str, str] | None = None,
    submit_wish: Dict[str, str] | None = None,
    end_playtest: Dict[str, str] | None = None,
    inspect_high_res_next: bool = False,
) -> Dict[str, Any]:
    return {
        "deliberation": f"Turn {turn_idx}: observing screen layout and testing responsive controls.",
        "current_plan": plan,
        "turn_log_entry": f"Tested movement step {turn_idx} and watched entity position.",
        "rulebook_updates": rulebook_updates or [],
        "motor_action": {
            "deliberation": f"Applying motor burst for turn {turn_idx}.",
            "mouse_x": 720.0 if turn_idx % 2 == 0 else None,
            "mouse_y": 395.0 if turn_idx % 2 == 0 else None,
            "mouse_button_action": "left_click" if turn_idx % 2 == 0 else "none",
            "drag_from_x": None,
            "drag_from_y": None,
            "hold_keys": ["d"] if turn_idx % 2 == 1 else [],
            "tap_keys": ["e"] if turn_idx == 2 else [],
            "duration_ms": 600,
            "inspect_high_res_next": inspect_high_res_next,
        },
        "mood_shift": mood_shift,
        "submit_wish": submit_wish,
        "end_playtest": end_playtest,
    }


def test_all_playtest_models_enforce_rationale_first_deliberation():
    model_classes = [
        BlindStoreCard,
        PlaytestSessionConfig,
        PlaytestMotorAction,
        PlaytestRulebookUpdate,
        PlaytestFieldNoteUpdate,
        PlaytestQuestionUpdate,
        PlaytestBugReport,
        PlaytestImpressionUpdate,
        PlaytestMoodNote,
        PlaytestWishDraft,
        PlaytestEndDecision,
        PlaytestTurnDecision,
        PlaytestCompactedJournal,
        PlaytestExperienceReport,
        PlaytestSessionStartRequest,
        PlaytestSessionControlRequest,
        PlaytestWishDispatchRequest,
        BlindPlaytestCouncilTurn,
        BlindPlaytestCouncilVerdict,
    ]
    for cls in model_classes:
        field_names = list(cls.model_fields.keys())
        assert field_names[0] == "deliberation", f"{cls.__name__} must place 'deliberation' at index 0, got {field_names}"

    # Verify terminal decision/score/status/severity fields are strictly LAST (Rule 4)
    assert list(PlaytestQuestionUpdate.model_fields.keys())[-1] == "status"
    assert list(PlaytestBugReport.model_fields.keys())[-1] == "severity"
    assert list(PlaytestExperienceReport.model_fields.keys())[-1] == "overall_fun_rating"
    assert list(BlindPlaytestCouncilTurn.model_fields.keys())[-1] == "recommends_slice_approval"
    assert list(BlindPlaytestCouncilVerdict.model_fields.keys())[-1] == "verdict"


def test_opaque_box_blindfold_strips_internal_specs_and_secrets(tmp_path: Path):
    cp_dir = tmp_path / "cp_alpha"
    cp_dir.mkdir()
    (cp_dir / "project.godot").write_text('[application]\nconfig/name="Project Fallback"\n', encoding="utf-8")
    (cp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "deliberation": "INTERNAL_MANIFEST_SECRET_DELIBERATION_991",
                "title": "Synthetic Horizon",
                "narrative_premise": "A quiet observatory perched above shifting clouds.",
                "core_rules": ["SECRET_RULE_PRESS_Z_TO_OPEN_VAULT"],
                "gameplay_pillars": ["SECRET_PILLAR_HIDDEN_TELEMETRY"],
                "candidate_id": "candidate_001",
                "slice_id": "slice_0",
            }
        ),
        encoding="utf-8",
    )
    (cp_dir / "game_bible.json").write_text(
        json.dumps({"secret_puzzle_solution": "HIDDEN_COMBINATION_4821"}),
        encoding="utf-8",
    )
    (cp_dir / "requirements_spec.json").write_text(
        json.dumps({"requirements": [{"id": "R1", "secret": "INTERNAL_ACCEPTANCE_CRITERIA"}]}),
        encoding="utf-8",
    )
    (cp_dir / "main.gd").write_text("# SECRET_GDSCRIPT_IMPLEMENTATION_TOKEN\n", encoding="utf-8")

    card = extract_blind_store_card(cp_dir, checkpoint_id="cp_alpha")
    assert card.game_title == "Synthetic Horizon"
    assert card.game_description == "A quiet observatory perched above shifting clouds."
    assert card.candidate_id == "candidate_001"
    assert card.slice_id == "slice_0"

    prompt = build_blind_player_system_prompt(card)
    assert "Synthetic Horizon" in prompt
    assert "A quiet observatory perched above shifting clouds." in prompt
    for forbidden in (
        "INTERNAL_MANIFEST_SECRET_DELIBERATION_991",
        "SECRET_RULE_PRESS_Z_TO_OPEN_VAULT",
        "SECRET_PILLAR_HIDDEN_TELEMETRY",
        "HIDDEN_COMBINATION_4821",
        "INTERNAL_ACCEPTANCE_CRITERIA",
        "SECRET_GDSCRIPT_IMPLEMENTATION_TOKEN",
    ):
        assert forbidden not in card.model_dump_json()
        assert forbidden not in prompt


def test_prefix_cache_invariance_and_multimodal_ephemeral_tail(tmp_path: Path):
    card = BlindStoreCard(
        deliberation="Test blind store card.",
        checkpoint_id="cp_test",
        game_title=" Test Title ",
        game_description="A short public blurb.",
    )
    responses = [
        {"payload": _turn_payload(1, inspect_high_res_next=True)},
        {"payload": _turn_payload(2, inspect_high_res_next=False)},
        {"payload": _turn_payload(3, inspect_high_res_next=False)},
        {"payload": _turn_payload(4, inspect_high_res_next=False)},
    ]
    fake_client = _QueuedFakeGeminiClient(responses)
    agent = BlindPlaytesterAgent(store_card=card, client=fake_client)

    obs1 = _make_obs(tmp_path, 1, bg_rgb=(20, 30, 50), marker_x=40, cursor_xy=(640.0, 360.0), audio_is_silent=False)
    obs2 = _make_obs(tmp_path, 2, bg_rgb=(35, 50, 80), marker_x=120, cursor_xy=(710.0, 385.0), audio_is_silent=True)
    # obs3 is visually identical to obs2 with stationary cursor -> triggers LOW_DEDUP
    obs3 = TurnObservation(
        observation_index=3,
        tick=135,
        sim_time_sec=2.25,
        cursor_x=710.2,
        cursor_y=385.1,
        screenshot_path=obs2.screenshot_path,
        preview_path=obs2.preview_path,
        wav_path=None,
        audio_rms=0.0,
        audio_peak=0.0,
        audio_is_silent=True,
        interrupted_early=False,
    )
    obs4 = _make_obs(tmp_path, 4, bg_rgb=(60, 30, 70), marker_x=200, cursor_xy=(500.0, 300.0), audio_is_silent=False)

    _, _, stats1 = agent.decide_turn(1, obs1)
    _, _, stats2 = agent.decide_turn(2, obs2)
    _, _, stats3 = agent.decide_turn(3, obs3)
    _, _, stats4 = agent.decide_turn(4, obs4)

    # Verify multimodal tail resolutions, normalized 0..1000 cursor coords, & audio behavior
    assert stats1["image_resolution"] == "MEDIUM"
    assert stats1["audio_attached"] is True
    t1_status_text = fake_client.calls[0]["contents"][-1].parts[0].text
    assert "Your Cursor Position: (500, 500)" in t1_status_text

    # Turn 1 requested inspect_high_res_next=True -> Turn 2 uses HIGH and skips silent audio
    assert stats2["image_resolution"] == "HIGH"
    assert stats2["audio_attached"] is False

    # Turn 3 had unchanged screen + stationary cursor -> deduplicated single LOW_DEDUP image
    assert stats3["screen_unchanged"] is True
    assert stats3["image_resolution"] == "LOW_DEDUP"
    tail_parts_t3 = fake_client.calls[2]["contents"][-1].parts
    img_parts_t3 = [p for p in tail_parts_t3 if p.inline_data and p.inline_data.mime_type == "image/png"]
    assert len(img_parts_t3) == 1
    assert img_parts_t3[0].media_resolution.level.value == "MEDIA_RESOLUTION_LOW"

    # Turn 4 moved again -> attaches LOW prev + MEDIUM current + audio/wav
    assert stats4["image_resolution"] == "MEDIUM"
    assert stats4["audio_attached"] is True
    tail_parts_t4 = fake_client.calls[3]["contents"][-1].parts
    img_parts_t4 = [p for p in tail_parts_t4 if p.inline_data and p.inline_data.mime_type == "image/png"]
    wav_parts_t4 = [p for p in tail_parts_t4 if p.inline_data and p.inline_data.mime_type == "audio/wav"]
    assert len(img_parts_t4) == 2
    assert img_parts_t4[0].media_resolution.level.value == "MEDIA_RESOLUTION_LOW"
    assert img_parts_t4[1].media_resolution.level.value == "MEDIA_RESOLUTION_MEDIUM"
    assert len(wav_parts_t4) == 1

    # Verify strict byte-0 prefix invariance across all 4 calls in the epoch:
    # 1. System prompt is byte-for-byte identical
    sys_prompts = [c["config"].system_instruction for c in fake_client.calls]
    assert len(set(sys_prompts)) == 1

    # 2. Every prefix `contents[:-1]` contains ONLY text parts (zero historical binary parts)
    #    and strictly extends the prior turn's `contents[:-1]`!
    serialized_prefixes: List[List[str]] = []
    for call in fake_client.calls:
        prefix_contents = call["contents"][:-1]
        texts: List[str] = []
        for content_item in prefix_contents:
            assert len(content_item.parts) == 1
            part = content_item.parts[0]
            assert part.inline_data is None, "Historical prefix must never retain binary image/audio bytes"
            assert isinstance(part.text, str) and part.text
            texts.append(part.text)
        serialized_prefixes.append(texts)

    for k in range(1, len(serialized_prefixes)):
        prev_prefix = serialized_prefixes[k - 1]
        cur_prefix = serialized_prefixes[k]
        assert len(cur_prefix) == len(prev_prefix) + 1
        assert cur_prefix[: len(prev_prefix)] == prev_prefix


def test_zero_amnesia_compaction_preserves_rulebook_plan_and_5_turn_overlap(tmp_path: Path):
    card = BlindStoreCard(
        deliberation="Compaction continuity test.",
        checkpoint_id="cp_compact",
        game_title="Continuity Test Game",
        game_description="Test description.",
    )
    cfg = PlaytestSessionConfig(
        compaction_interval_turns=8,
        compaction_overlap_turns=3,
    )

    queued: List[Dict[str, Any]] = []
    for t in range(1, 9):
        rb_upd = []
        if t == 1:
            rb_upd = [{"deliberation": "Confirmed D key.", "key_or_topic": "Key D", "behavior": "Moves character right"}]
        elif t == 4:
            rb_upd = [{"deliberation": "Confirmed Space.", "key_or_topic": "SPACE", "behavior": "Does nothing on ground"}]
        queued.append(
            {
                "payload": _turn_payload(
                    t,
                    plan=f"Carry keycard from East Corridor to Northern Gate (step {t})",
                    rulebook_updates=rb_upd,
                )
            }
        )

    # Compaction response deliberately omits keybind details to prove Rulebook is preserved deterministically in code!
    queued.append(
        {
            "payload": {
                "deliberation": "Consolidating turns 1..5 while turns 6..8 stay verbatim in overlap.",
                "episodic_story_and_world_chronicle": ["Entered East Corridor and picked up keycard."],
                "world_landmarks_map": ["Northern Gate is up the stairs from East Corridor."],
                "failed_attempts_and_dead_ends": ["Locked cellar hatch could not be pried open."],
                "open_questions_and_hypotheses": ["See whether keycard opens Northern Gate."],
                "fun_and_aesthetic_impressions": "Atmospheric lighting and crisp movement response so far.",
            }
        }
    )
    # Turn 9 after compaction
    queued.append({"payload": _turn_payload(9, plan="Use keycard on Northern Gate terminal")})

    fake_client = _QueuedFakeGeminiClient(queued)
    agent = BlindPlaytesterAgent(store_card=card, config=cfg, client=fake_client)

    for t in range(1, 9):
        obs = _make_obs(tmp_path, t, marker_x=20 * t, audio_is_silent=True)
        agent.decide_turn(t, obs)

    assert agent.should_compact() is True
    journal, covered_up_to, _ = agent.compact_journal()
    assert covered_up_to == 5
    assert agent.compaction_count == 1

    # 1. Permanent Rulebook is 100% intact in [Block 1A]
    assert agent.rulebook == {
        "Key D": "Moves character right",
        "SPACE": "Does nothing on ground",
    }
    assert "Key D: Moves character right" in agent.epoch_header_text
    assert "SPACE: Does nothing on ground" in agent.epoch_header_text
    # 2. Compacted Journal is in [Block 1B]
    assert "Northern Gate is up the stairs" in agent.epoch_header_text
    # 3. Overlap turns 6, 7, 8 were carried verbatim into the new epoch's [Block 2]
    assert [t_idx for t_idx, _ in agent.epoch_turn_lines] == [6, 7, 8]

    # 4. Turn 9 sees both the compacted header, verbatim turns 6..8, AND verbatim Turn 8 current_plan!
    obs9 = _make_obs(tmp_path, 9, marker_x=210, audio_is_silent=True)
    agent.decide_turn(9, obs9)
    turn9_call = fake_client.calls[-1]["contents"]
    tail_text_t9 = turn9_call[-1].parts[0].text
    assert "Carry keycard from East Corridor to Northern Gate (step 8)" in tail_text_t9


def test_decoupled_from_studio_pause_sentinel_and_global_token_breaker(tmp_path: Path):
    card = BlindStoreCard(
        deliberation="Isolation check.",
        checkpoint_id="cp_iso",
        game_title="Isolated Slice",
        game_description="Playable regardless of Studio build state.",
    )
    fake_client = _QueuedFakeGeminiClient(
        [
            {
                "payload": _turn_payload(
                    1,
                    mood_shift={
                        "deliberation": "Figured out opening interaction.",
                        "mood": "satisfied",
                        "note": "Clicking the lantern lit up the path right away.",
                    },
                    end_playtest={
                        "deliberation": "Reached slice credits screen.",
                        "outcome": "completed",
                        "closing_thoughts": "Short, cohesive slice with clear visual feedback.",
                    },
                ),
                "prompt_tokens": 6000,
                "output_tokens": 5000,
                "thought_tokens": 1000,
            }
        ]
    )

    # Simulate tripped global Studio SWE circuit breaker + pause file in tmp_path
    pause_sentinel = tmp_path / ".studio_paused.json"
    pause_sentinel.write_text('{"paused": true}', encoding="utf-8")
    prev_streak = TOKEN_USAGE_TRACKER.consecutive_swe_pev_calls
    prev_calls = TOKEN_USAGE_TRACKER.total_calls
    try:
        TOKEN_USAGE_TRACKER.consecutive_swe_pev_calls = 9999

        tracker = PlaytestTokenTracker(max_session_tokens=15_000)
        agent = BlindPlaytesterAgent(store_card=card, client=fake_client, token_tracker=tracker)
        obs = _make_obs(tmp_path, 1, audio_is_silent=True)
        decision, _, _ = agent.decide_turn(1, obs)
        assert decision.end_playtest is not None
        assert decision.end_playtest.outcome == "completed"
        assert agent.current_mood == "satisfied"
        # Studio global build tracker was untouched by decoupled Playtester turns
        assert TOKEN_USAGE_TRACKER.total_calls == prev_calls

        # Verify isolated PlaytestTokenTracker enforces its own per-session ceiling on subsequent turns
        tracker.max_session_tokens = 10_000
        with pytest.raises(PlaytestBudgetExceededError):
            agent.decide_turn(2, obs)
    finally:
        TOKEN_USAGE_TRACKER.consecutive_swe_pev_calls = prev_streak


def test_compute_frame_mse_detects_identical_and_changed_frames(tmp_path: Path):
    p1 = _make_png(tmp_path / "f1.png", (25, 35, 45), marker_x=50)
    p2 = _make_png(tmp_path / "f2.png", (25, 35, 45), marker_x=50)
    p3 = _make_png(tmp_path / "f3.png", (180, 90, 40), marker_x=220)
    assert compute_frame_mse(p1, p2) == 0.0
    assert compute_frame_mse(p1, p3) > 0.05


def test_create_isolated_playtest_genai_client_loads_dotenv_when_env_unset(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text('export GEMINI_API_KEY="synthetic_dotenv_key_12345"\n', encoding="utf-8")

    client = create_isolated_playtest_genai_client(dotenv_path=dotenv_file)
    assert client is not None


def test_turn_decision_accepts_verbose_closing_thoughts_wishes_and_clamps_duration(tmp_path: Path):
    verbose_closing = " ".join(["Detailed player reflection on gameplay feel, world atmosphere, and controls."] * 12)
    assert len(verbose_closing) > 750

    payload = {
        "deliberation": "Wrapping up session with thorough player reflection.",
        "current_plan": " ".join(["Concluding exploration and recording final impressions.\nNext line."] * 6),
        "turn_log_entry": " ".join(["Observed ending sequence\nand summarized takeaways."] * 6),
        "rulebook_updates": [
            {
                "deliberation": "Confirmed control.",
                "key_or_topic": "Primary Interaction Binding With Extended Description " * 2,
                "behavior": "Triggers the nearby interactive mechanism when standing within reach. " * 4,
            }
        ],
        "motor_action": {
            "deliberation": "Brief idle step.",
            "mouse_x": 1200,
            "mouse_y": -20,
            "mouse_button_action": "none",
            "duration_ms": 100,
        },
        "mood_shift": {
            "deliberation": "Mood shifted upon finishing.",
            "mood": "thoughtful and satisfied after exploring",
            "note": " ".join(["Enjoyed the visual tone, though one mid-game affordance took experimentation."] * 6),
        },
        "submit_wish": {
            "deliberation": "Filing UX clarity wish.",
            "category": "controls_ux",
            "summary": " ".join(["Clearer hover feedback when an interactive target requires an unselected state"] * 3),
            "rationale": " ".join(["When clicking with a tool active, it was not obvious how to deselect it."] * 8),
        },
        "end_playtest": {
            "deliberation": "Ending session cleanly.",
            "outcome": "stuck_cannot_progress",
            "closing_thoughts": verbose_closing,
        },
    }

    card = BlindStoreCard(
        deliberation="Verbose schema validation test.",
        checkpoint_id="cp_verbose",
        game_title="Schema Resilience Test",
    )
    fake_client = _QueuedFakeGeminiClient([{"payload": payload}])
    agent = BlindPlaytesterAgent(store_card=card, client=fake_client)
    obs = _make_obs(tmp_path, 1, audio_is_silent=True)

    decision, tape_line, _ = agent.decide_turn(1, obs)
    assert decision.end_playtest is not None
    assert decision.end_playtest.closing_thoughts == verbose_closing
    assert decision.motor_action.mouse_x == 1000.0
    assert decision.motor_action.mouse_y == 0.0
    assert decision.motor_action.duration_ms == 350
    assert "\n" not in tape_line


def test_sparse_impressions_non_repetition_compaction_and_experience_report(tmp_path: Path):
    card = BlindStoreCard(
        deliberation="Sparse pillar impressions and end-of-session report test.",
        checkpoint_id="cp_impressions",
        game_title="Observatory Echoes",
        game_description="Explore an acoustic courtyard.",
    )
    cfg = PlaytestSessionConfig(compaction_interval_turns=6, compaction_overlap_turns=2)

    t1 = _turn_payload(1, plan="Inspect courtyard layout")
    t1["impression_updates"] = [
        {
            "deliberation": "First look at art direction.",
            "pillar": "graphics_and_visuals",
            "impression": "Warm high-contrast lighting makes foreground structures easy to read.",
        },
        {
            "deliberation": "Initial onboarding impression.",
            "pillar": "intuitiveness_and_controls",
            "impression": "Not immediately clear which courtyard object can be clicked first.",
        },
    ]
    t1["submit_wish"] = {
        "deliberation": "Wish for hover outline.",
        "category": "controls_ux",
        "summary": "Show subtle hover outline on interactive props",
        "rationale": "First-time click targets blend slightly into decorative props.",
    }

    # Turn 2 has NO impression updates (`[]`) — verifying sparse updates
    t2 = _turn_payload(2, plan="Click central chime")
    t2["impression_updates"] = []

    # Turn 3 revises `intuitiveness_and_controls` once figured out
    t3 = _turn_payload(3, plan="Follow lit conduit")
    t3["impression_updates"] = [
        {
            "deliberation": "Revised onboarding impression after discovering chime pulse.",
            "pillar": "intuitiveness_and_controls",
            "impression": "Once clicked, glowing conduit lines clearly teach cause and effect.",
        }
    ]

    report_payload = {
        "deliberation": "Synthesizing candid end-of-session player report across all 6 pillars.",
        "overall_experience_summary": "Had a rewarding short session once I discovered the central chime.",
        "graphics_and_visuals": "Clean silhouette readability and warm palette.",
        "gameplay_and_feel": "Satisfying chain reaction when activating chimes.",
        "sound_and_audio": "Resonant chime tones reinforced each activation.",
        "intuitiveness_and_onboarding": "Slight initial hesitation on what to click first, then smooth.",
        "goals_pacing_and_accomplishment": "Clear sense of progress lighting the courtyard.",
        "world_and_atmosphere": "Cohesive, calm acoustic mystery tone.",
        "key_highlights": ["Audio-visual chime feedback"],
        "key_pain_points": ["Initial interactive prop hover cue was subtle"],
        "overall_fun_rating": 8,
    }

    fake_client = _QueuedFakeGeminiClient(
        [
            {"payload": t1},
            {"payload": t2},
            {"payload": t3},
            {"payload": report_payload},
        ]
    )
    tracker = PlaytestTokenTracker(max_session_tokens=50_000)
    agent = BlindPlaytesterAgent(store_card=card, config=cfg, client=fake_client, token_tracker=tracker)

    obs1 = _make_obs(tmp_path, 1, marker_x=40, audio_is_silent=True)
    obs2 = _make_obs(tmp_path, 2, marker_x=90, audio_is_silent=True)
    obs3 = _make_obs(tmp_path, 3, marker_x=140, audio_is_silent=True)

    _, tape1, _ = agent.decide_turn(1, obs1)
    assert "IMPRESSION+[graphics_and_visuals=" in tape1
    assert len(agent.impressions) == 2

    _, tape2, _ = agent.decide_turn(2, obs2)
    assert "IMPRESSION+" not in tape2

    # Verify Turn 2 ephemeral tail showed recorded impressions & filed wishes (preventing repetition)
    # while keeping `contents[:-1]` byte-0 prefix strictly append-only!
    t2_tail_text = fake_client.calls[1]["contents"][-1].parts[0].text
    assert "Your Recorded Impressions So Far" in t2_tail_text
    assert "Wishes You Already Filed" in t2_tail_text
    assert fake_client.calls[1]["contents"][0].parts[0].text == fake_client.calls[0]["contents"][0].parts[0].text

    agent.decide_turn(3, obs3)
    assert agent.impressions["intuitiveness_and_controls"].startswith("Once clicked, glowing conduit lines")

    # Simulate budget reached right as session stops -> generate_experience_report MUST still succeed!
    tracker.max_session_tokens = 100
    report, _ = agent.generate_experience_report(
        end_status="stopped",
        end_outcome="operator_stopped",
        final_screenshot_path=obs3.screenshot_path,
        wishes_filed=[{"category": "controls_ux", "summary": "Show subtle hover outline on interactive props", "rationale": "Blends in."}],
    )
    assert report.overall_fun_rating == 8
    assert "central chime" in report.overall_experience_summary
    # Report call reuses the exact same byte-0 epoch header + 3 append-only turn lines!
    rep_contents = fake_client.calls[3]["contents"]
    assert len(rep_contents[:-1]) == 1 + 3
    assert rep_contents[0].parts[0].text == fake_client.calls[0]["contents"][0].parts[0].text


def test_fast_non_thinking_model_config_and_live_notebook_questions_bugs_mood_and_audio_transition(tmp_path: Path):
    assert DEFAULT_PLAYTEST_MODEL == "gemini-3.8-flash"
    lite_tc = _playtest_non_thinking_config("gemini-3.5-flash-lite")
    assert lite_tc.thinking_level == types.ThinkingLevel.MINIMAL
    flash_tc = _playtest_non_thinking_config("gemini-3.8-flash")
    assert flash_tc.thinking_level == types.ThinkingLevel.MEDIUM

    card = BlindStoreCard(
        deliberation="Test player notes, questions, bug spotting, live mood, and perceptual change note.",
        checkpoint_id="cp_notebook",
        game_title="Clockwork Terrace",
        game_description="An atmospheric mechanical terrace.",
    )
    cfg = PlaytestSessionConfig(compaction_interval_turns=6, compaction_overlap_turns=1)

    # Turn 1: Logs a field note + open question + sets current_mood="intrigued" (without explicit mood_shift)
    t1 = _turn_payload(1, plan="Test clicking the brass dial")
    t1["field_note_updates"] = [
        {
            "deliberation": "Noting brass dial at top-left.",
            "topic": "Brass Dial",
            "observation": "Shows three unlit notches above a small lever.",
        }
    ]
    t1["question_updates"] = [
        {
            "deliberation": "Wondering what the brass dial controls.",
            "question_key": "brass_dial_purpose",
            "question": "What does the brass dial in the corner control?",
            "answer": None,
            "status": "open",
        },
        {
            "deliberation": "Wondering why the side arch is sealed.",
            "question_key": "sealed_arch_reason",
            "question": "Why is the eastern stone arch barred?",
            "answer": None,
            "status": "open",
        },
    ]
    t1["impression_updates"] = [
        {
            "deliberation": "Initial audio impression while background music plays.",
            "pillar": "sound_and_audio",
            "impression": "Pleasant ambient chime loop plays on boot.",
        }
    ]
    t1["current_mood"] = "intrigued"

    # Turn 2: Audio stops -> answers first question + logs a bug + revises audio impression + shifts mood to puzzled
    t2 = _turn_payload(2, plan="Inspect eastern arch")
    t2["question_updates"] = [
        {
            "deliberation": "Clicking the lever lit one notch on the brass dial.",
            "question_key": "brass_dial_purpose",
            "question": "What does the brass dial in the corner control?",
            "answer": "Pulling the lever charges one notch on the brass dial.",
            "status": "answered",
        }
    ]
    t2["bug_reports"] = [
        {
            "deliberation": "Ambient audio abruptly cut out to silence.",
            "bug_key": "ambient_audio_dropout",
            "category": "audio_glitch",
            "summary": "Ambient audio cut out completely after opening seconds",
            "details": "Background sound played briefly on Turn 1 and then dropped to total silence.",
            "severity": "major",
        }
    ]
    t2["impression_updates"] = [
        {
            "deliberation": "Revising earlier audio impression now that sound dropped out.",
            "pillar": "sound_and_audio",
            "impression": "Opening chime sounded nice briefly, then audio cut out completely.",
        }
    ]
    t2["current_mood"] = "puzzled"

    # Turn 3: Still silent -> verify Perceptual Change Note is NOT spammed every turn!
    t3 = _turn_payload(3, plan="Try right-click on eastern arch")
    t3["current_mood"] = "puzzled"

    report_payload = {
        "deliberation": "Summarizing session with answered/unanswered questions and spotted audio glitch.",
        "overall_experience_summary": "Figured out the dial lever, though audio cut out early and the eastern arch stayed barred.",
        "graphics_and_visuals": "Readable brass and stone contrast.",
        "gameplay_and_feel": "Lever click felt responsive.",
        "sound_and_audio": "Opening chime cut out to complete silence after the first turn.",
        "intuitiveness_and_onboarding": "Dial notches made lever progress clear once tested.",
        "goals_pacing_and_accomplishment": "Charged one dial notch; arch remained locked.",
        "world_and_atmosphere": "Quiet mechanical terrace vibe.",
        "key_highlights": ["Clear dial notch response"],
        "key_pain_points": ["Ambient audio cut out after first turn"],
        "bugs_and_glitches_observed": ["Ambient audio dropped out completely after opening seconds"],
        "unanswered_questions_and_mysteries": ["Why is the eastern stone arch barred?"],
        "overall_fun_rating": 5,
    }

    fake_client = _QueuedFakeGeminiClient(
        [
            {"payload": t1},
            {"payload": t2},
            {"payload": t3},
            {"payload": report_payload},
        ]
    )
    agent = BlindPlaytesterAgent(store_card=card, config=cfg, client=fake_client)

    obs1 = _make_obs(tmp_path, 1, marker_x=30, audio_is_silent=False)
    obs2 = _make_obs(tmp_path, 2, marker_x=80, audio_is_silent=True)
    obs3 = _make_obs(tmp_path, 3, marker_x=130, audio_is_silent=True)

    dec1, tape1, _ = agent.decide_turn(1, obs1)
    assert agent.current_mood == "intrigued"
    assert dec1.mood_shift is not None and dec1.mood_shift.mood == "intrigued"
    assert "NOTE+[Brass Dial=" in tape1
    assert "QUESTION+[OPEN(brass_dial_purpose)=" in tape1
    # Verify NO per-turn Audio Note spam on Turn 1
    t1_tail = fake_client.calls[0]["contents"][-1].parts[0].text
    assert "Audio Note:" not in t1_tail
    assert "Perceptual Change Note:" not in t1_tail

    dec2, tape2, _ = agent.decide_turn(2, obs2)
    # Turn 2 transitioned from audible -> silent: Perceptual Change Note appears ONCE on transition!
    t2_tail = fake_client.calls[1]["contents"][-1].parts[0].text
    assert "Perceptual Change Note: Game audio stopped and is now silent." in t2_tail
    assert "Open Questions You Are Trying To Answer" in t2_tail
    assert agent.current_mood == "puzzled"
    assert agent.questions["brass_dial_purpose"]["status"] == "answered"
    assert "Pulling the lever" in agent.questions["brass_dial_purpose"]["answer"]
    assert "ambient_audio_dropout" in agent.bugs_found
    assert "QUESTION+[ANSWERED(brass_dial_purpose)=" in tape2
    assert "BUG+[audio_glitch/major:" in tape2

    agent.decide_turn(3, obs3)
    # Turn 3 remained silent -> transition note is NOT repeated!
    t3_tail = fake_client.calls[2]["contents"][-1].parts[0].text
    assert "Perceptual Change Note:" not in t3_tail
    assert "Questions Already Answered" in t3_tail
    assert "Bugs & Glitches Already Logged" in t3_tail

    rep, _ = agent.generate_experience_report(
        end_status="completed",
        end_outcome="completed",
        final_screenshot_path=obs3.screenshot_path,
    )
    rep_prompt_text = fake_client.calls[3]["contents"][-1].parts[0].text
    assert "Player Mood Trajectory: curious (T#0) -> intrigued (T#1) -> puzzled (T#2)" in rep_prompt_text
    assert "Brass Dial: Shows three unlit notches" in rep_prompt_text
    assert "Answered During Play:" in rep_prompt_text
    assert "Still Unanswered At End Of Session:" in rep_prompt_text
    assert "Ambient audio cut out completely" in rep_prompt_text
    assert rep.bugs_and_glitches_observed == ["Ambient audio dropped out completely after opening seconds"]
    assert rep.unanswered_questions_and_mysteries == ["Why is the eastern stone arch barred?"]


def test_mood_dedup_anti_ping_pong_and_same_spot_click_continuity_note(tmp_path: Path):
    card = BlindStoreCard(
        deliberation="Verify mood deduplication, bounce-back throttling, post-turn rule confirmation, and click-streak note.",
        checkpoint_id="cp_mood_and_clicks",
        game_title="Lantern Relay",
        game_description="Connect lantern relays across a misty terrace.",
    )
    sys_prompt = build_blind_player_system_prompt(card)
    assert "motor_action` executes AFTER your turn decision" in sys_prompt
    rating_desc = PlaytestExperienceReport.model_fields["overall_fun_rating"].description
    assert "played itself" in rating_desc

    # T1: explicit mood_shift="curious" when already "curious" -> MUST be deduplicated/suppressed!
    t1 = _turn_payload(
        1,
        plan="Click corner crest #1",
        mood_shift={"deliberation": "Still curious.", "mood": "curious", "note": "Looking around."},
    )
    t1["motor_action"]["mouse_x"] = 850.0
    t1["motor_action"]["mouse_y"] = 220.0
    t1["motor_action"]["mouse_button_action"] = "left_click"

    # T2: genuine shift to "satisfied" + clicks same spot #2
    t2 = _turn_payload(
        2,
        plan="Click corner crest #2",
        mood_shift={"deliberation": "First lantern lit.", "mood": "satisfied", "note": "Lantern lit nicely."},
    )
    t2["motor_action"]["mouse_x"] = 852.0
    t2["motor_action"]["mouse_y"] = 219.0
    t2["motor_action"]["mouse_button_action"] = "left_click"

    # T3: duplicate "satisfied" -> MUST be suppressed! + clicks same spot #3
    t3 = _turn_payload(
        3,
        plan="Click corner crest #3",
        mood_shift={"deliberation": "Still satisfied.", "mood": "satisfied", "note": "Still clicking crest."},
    )
    t3["motor_action"]["mouse_x"] = 849.0
    t3["motor_action"]["mouse_y"] = 221.0
    t3["motor_action"]["mouse_button_action"] = "left_click"

    # T4: bounce-back to "curious" within 8 turns -> MUST be suppressed (anti-ping-pong)!
    # Also on T4 input, `Action Continuity Note` MUST appear in ephemeral tail because streak == 3!
    # Moves to a new spot (300, 500) so streak resets.
    t4 = _turn_payload(4, plan="Try central switch instead")
    t4["current_mood"] = "curious"
    t4["motor_action"]["mouse_x"] = 300.0
    t4["motor_action"]["mouse_y"] = 500.0
    t4["motor_action"]["mouse_button_action"] = "left_click"

    # T5: genuine new mood "frustrated" accompanied by a bug report -> MUST record cleanly!
    t5 = _turn_payload(5, plan="Log unresponsive decorative crest")
    t5["current_mood"] = "frustrated"
    t5["bug_reports"] = [
        {
            "deliberation": "Corner crest looked like a button but never changed any counter.",
            "bug_key": "unresponsive_crest",
            "category": "input_ui_bug",
            "summary": "Corner crest badge looks interactive but does nothing",
            "details": "Clicked corner badge 3 times and counters stayed unchanged.",
            "severity": "minor",
        }
    ]

    fake_client = _QueuedFakeGeminiClient(
        [
            {"payload": t1},
            {"payload": t2},
            {"payload": t3},
            {"payload": t4},
            {"payload": t5},
        ]
    )
    agent = BlindPlaytesterAgent(store_card=card, client=fake_client)

    obs = [_make_obs(tmp_path, i, marker_x=20 * i) for i in range(1, 6)]

    dec1, tape1, _ = agent.decide_turn(1, obs[0])
    assert dec1.mood_shift is None
    assert "MOOD[" not in tape1
    assert len(agent.mood_history) == 0

    dec2, tape2, _ = agent.decide_turn(2, obs[1])
    assert dec2.mood_shift is not None and dec2.mood_shift.mood == "satisfied"
    assert "MOOD[satisfied:" in tape2
    assert len(agent.mood_history) == 1

    dec3, tape3, _ = agent.decide_turn(3, obs[2])
    assert dec3.mood_shift is None
    assert "MOOD[" not in tape3
    assert len(agent.mood_history) == 1
    assert agent._same_spot_click_streak == 3

    dec4, tape4, _ = agent.decide_turn(4, obs[3])
    # Verify Turn 4 ephemeral tail warned about 3 consecutive clicks on (850, 220) while keeping byte-0 prefix intact
    t4_tail = fake_client.calls[3]["contents"][-1].parts[0].text
    assert "Action Continuity Note: You have clicked nearly the same spot" in t4_tail
    assert "3 turns in a row" in t4_tail
    assert fake_client.calls[3]["contents"][0].parts[0].text == fake_client.calls[0]["contents"][0].parts[0].text
    # Rapid bounce-back to "curious" was suppressed
    assert dec4.mood_shift is None
    assert agent.current_mood == "satisfied"
    assert "MOOD[" not in tape4

    dec5, tape5, _ = agent.decide_turn(5, obs[4])
    # Streak reset after clicking (300, 500), so Turn 5 tail has NO Action Continuity Note
    t5_tail = fake_client.calls[4]["contents"][-1].parts[0].text
    assert "Action Continuity Note:" not in t5_tail
    # Genuine shift to "frustrated" on bug discovery recorded!
    assert dec5.mood_shift is not None and dec5.mood_shift.mood == "frustrated"
    assert agent.current_mood == "frustrated"
    assert [m["mood"] for m in agent.mood_history] == ["satisfied", "frustrated"]
    assert "MOOD[frustrated:" in tape5


def test_hold_left_and_drag_streak_triggers_continuity_note_and_high_res_upgrade(tmp_path: Path):
    """Regression guard: repeated `hold_left` (and `drag_left`) actions on the same coordinate
    must increment `_same_spot_click_streak`, surface `Action Continuity Note`, automatically
    upgrade the attached screenshot to `HIGH` resolution so small HUD counters are read crisply,
    and escalate at streak >= 5 so the playtester never loops 30+ turns on a stalled counter.
    """
    card = BlindStoreCard(
        deliberation="Test repeated hold_left streak tracking.",
        checkpoint_id="cp_hold_loop",
        game_title="Synthetic Workshop",
        game_description="Test interactive objects.",
    )

    queued = []
    for idx in range(1, 7):
        payload = _turn_payload(idx, plan="Hold left mouse button on central object and verify counter")
        payload["motor_action"]["mouse_x"] = 500.0
        payload["motor_action"]["mouse_y"] = 480.0
        payload["motor_action"]["mouse_button_action"] = "hold_left"
        payload["motor_action"]["hold_keys"] = []
        payload["motor_action"]["tap_keys"] = []
        payload["motor_action"]["duration_ms"] = 1500
        payload["motor_action"]["inspect_high_res_next"] = False
        queued.append({"payload": payload})

    fake_client = _QueuedFakeGeminiClient(queued)
    agent = BlindPlaytesterAgent(store_card=card, client=fake_client)
    obs = [_make_obs(tmp_path, i, marker_x=15 * i, cursor_xy=(640.0, 345.6)) for i in range(1, 7)]

    _, _, s1 = agent.decide_turn(1, obs[0])
    assert agent._same_spot_click_streak == 1
    assert s1["image_resolution"] == "MEDIUM"

    _, _, s2 = agent.decide_turn(2, obs[1])
    assert agent._same_spot_click_streak == 2
    assert s2["image_resolution"] == "MEDIUM"

    _, _, s3 = agent.decide_turn(3, obs[2])
    assert agent._same_spot_click_streak == 3
    assert s3["image_resolution"] == "MEDIUM"

    # Turn 4 sees streak == 3 from Turns 1..3 -> emits Action Continuity Note AND upgrades to HIGH res!
    _, _, s4 = agent.decide_turn(4, obs[3])
    t4_tail = fake_client.calls[3]["contents"][-1].parts[0].text
    assert "Action Continuity Note: You have repeated 'hold_left' on nearly the same spot (500, 480) 3 turns in a row." in t4_tail
    assert s4["image_resolution"] == "HIGH"
    assert agent._same_spot_click_streak == 4

    _, _, s5 = agent.decide_turn(5, obs[4])
    assert s5["image_resolution"] == "HIGH"
    assert agent._same_spot_click_streak == 5

    # Turn 6 sees streak == 5 -> escalates with explicit STOP REPEATING directive!
    _, _, s6 = agent.decide_turn(6, obs[5])
    t6_tail = fake_client.calls[5]["contents"][-1].parts[0].text
    assert "STOP REPEATING 'hold_left' AT (500, 480)" in t6_tail
    assert s6["image_resolution"] == "HIGH"


def test_evidence_first_turn_schema_order_and_rulebook_question_retraction(tmp_path: Path):
    """Verifies that:
    1. `PlaytestTurnDecision` orders prior-turn observational fields (`turn_log_entry`,
       `field_note_updates`, `question_updates`, `rulebook_updates`, `bug_reports`) BEFORE
       `current_plan` and `motor_action` so unexecuted future plans cannot contaminate confirmed rules.
    2. `Your Confirmed Rulebook So Far` is shown in every turn's ephemeral status block.
    3. Mistaken Rulebook entries can be retracted (`action='retract'`) and prematurely answered
       questions can be reopened (`status='open'`, `answer=None`, clearing the stale answer).
    """
    turn_fields = list(PlaytestTurnDecision.model_fields.keys())
    assert turn_fields[0] == "deliberation"
    assert turn_fields.index("question_updates") < turn_fields.index("current_plan")
    assert turn_fields.index("rulebook_updates") < turn_fields.index("current_plan")
    assert turn_fields.index("current_plan") < turn_fields.index("motor_action")
    assert list(PlaytestRulebookUpdate.model_fields.keys())[-1] == "action"

    card = BlindStoreCard(
        deliberation="Test rulebook retraction and question reopening.",
        checkpoint_id="cp_retract",
        game_title="Synthetic Control Room",
        game_description="Test switches and gauges.",
    )

    t1 = _turn_payload(
        1,
        plan="Hold left button on central node",
        rulebook_updates=[
            {
                "deliberation": "Prematurely recorded rule.",
                "key_or_topic": "Hold Left Button",
                "observed_change_from_prior_turn": "Assumed gauge moved.",
                "behavior": "Increments primary counter.",
                "action": "confirm",
            }
        ],
    )
    t1["question_updates"] = [
        {
            "deliberation": "Marked answered too early.",
            "question_key": "node_hold_effect",
            "question": "Does holding left click increment the primary counter?",
            "observed_change_from_prior_turn": "Thought gauge ticked.",
            "answer": "Yes, holding increments the counter.",
            "status": "answered",
        }
    ]

    t2 = _turn_payload(
        2,
        plan="Try pressing Tab instead since counter stayed 0/10",
        rulebook_updates=[
            {
                "deliberation": "Counter stayed at 0/10 on Turn 2 high-res check; retracting mistaken rule.",
                "key_or_topic": "hold left button",
                "observed_change_from_prior_turn": "Counter remained 0/10 across previous and current screen.",
                "behavior": "Did not increment counter.",
                "action": "retract",
            }
        ],
    )
    t2["question_updates"] = [
        {
            "deliberation": "Reopening question because counter is still 0/10.",
            "question_key": "node_hold_effect",
            "question": "Does holding left click increment the primary counter?",
            "observed_change_from_prior_turn": "Counter remained 0/10.",
            "answer": None,
            "status": "open",
        }
    ]

    fake_client = _QueuedFakeGeminiClient([{"payload": t1}, {"payload": t2}])
    agent = BlindPlaytesterAgent(store_card=card, client=fake_client)
    obs1 = _make_obs(tmp_path, 1, marker_x=25)
    obs2 = _make_obs(tmp_path, 2, marker_x=50)

    agent.decide_turn(1, obs1)
    assert "Hold Left Button" in agent.rulebook
    assert agent.questions["node_hold_effect"]["status"] == "answered"

    _, tape2, _ = agent.decide_turn(2, obs2)
    # Turn 2 ephemeral tail showed the live Rulebook so the player could spot and retract it
    t2_tail = fake_client.calls[1]["contents"][-1].parts[0].text
    assert "Your Confirmed Rulebook So Far" in t2_tail
    assert "Hold Left Button: Increments primary counter." in t2_tail

    # Mistaken rule was retracted and question reopened with stale answer cleared
    assert "Hold Left Button" not in agent.rulebook
    assert "RULEBOOK+[RETRACTED(hold left button)]" in tape2
    assert agent.questions["node_hold_effect"]["status"] == "open"
    assert agent.questions["node_hold_effect"]["answer"] is None
    assert agent.questions["node_hold_effect"]["answered_turn"] is None


def test_visual_craft_audit_high_res_verification_and_report_model(tmp_path: Path):
    from engine.playtest.models import DEFAULT_PLAYTEST_REPORT_MODEL

    # 1. Rationale-First ordering: visual_and_ui_defects_observed precedes graphics_and_visuals & overall_fun_rating
    rep_fields = list(PlaytestExperienceReport.model_fields.keys())
    assert rep_fields[0] == "deliberation"
    assert rep_fields.index("visual_and_ui_defects_observed") < rep_fields.index("graphics_and_visuals")
    assert rep_fields[-1] == "overall_fun_rating"

    card = BlindStoreCard(
        deliberation="Test anti-sycophancy visual audit across turns and end-of-session report.",
        checkpoint_id="cp_visual_audit",
        game_title="Clockwork Relay",
        game_description="A gleaming, ultra-polished hand-crafted world of pristine beauty.",
    )
    sys_prompt = build_blind_player_system_prompt(card)
    assert "promotional box copy" in sys_prompt
    assert "sprite-sheet/atlas frames" in sys_prompt
    assert "spilling outside" in sys_prompt

    # Turn 1: Player records premature graphics_and_visuals praise at MEDIUM resolution (without inspect_high_res_next)
    t1 = _turn_payload(1, inspect_high_res_next=False, plan="Look closely at HUD and sprites")
    t1["impression_updates"] = [
        {
            "deliberation": "Quick glance at opening thumbnail.",
            "pillar": "graphics_and_visuals",
            "impression": "Bright palette at first glance.",
        }
    ]

    # Turn 2: Agent auto-upgrades to HIGH resolution + injects Visual Acuity Note -> player spots defects & revises!
    t2 = _turn_payload(2, inspect_high_res_next=False, plan="Open side panel")
    t2["impression_updates"] = [
        {
            "deliberation": "At HIGH resolution, sprite edges bleed adjacent frames and top-left text overflows box.",
            "pillar": "graphics_and_visuals",
            "impression": "Rough presentation: sprite-sheet neighbor slivers bleed into props and HUD text overflows.",
        }
    ]
    t2["bug_reports"] = [
        {
            "deliberation": "Sprite atlas slicing cuts into neighboring frame.",
            "bug_key": "sprite_atlas_bleed",
            "category": "visual_glitch",
            "summary": "Sprites show stray bleeding edge slivers from adjacent atlas frames",
            "details": "Foreground props display cut-off edges and stray floating slivers from neighboring frames.",
            "severity": "major",
        }
    ]

    t3 = _turn_payload(3, inspect_high_res_next=False, plan="Wrap up check")

    report_payload = {
        "deliberation": "Auditing HIGH-res screenshots before writing graphics_and_visuals.",
        "overall_experience_summary": "Playable controls, but visual presentation has obvious sprite-slicing and UI overflow bugs.",
        "visual_and_ui_defects_observed": [
            "Foreground sprites show cut-off edges and stray slivers bleeding from neighboring atlas frames",
            "Top-left counter label spills outside its panel border",
        ],
        "graphics_and_visuals": "Rough and unfinished: sprite-sheet slicing bleed and overflowing HUD labels undermine readability.",
        "gameplay_and_feel": "Basic click response works.",
        "sound_and_audio": "Silent throughout the session.",
        "intuitiveness_and_onboarding": "Controls are discoverable despite cramped labels.",
        "goals_pacing_and_accomplishment": "Completed initial interaction check.",
        "world_and_atmosphere": "Inconsistent mix of detailed raster sprites and flat untextured polygons.",
        "key_highlights": ["Responsive cursor targeting"],
        "key_pain_points": ["Sprite-sheet bleed slivers and overflowing HUD text"],
        "bugs_and_glitches_observed": ["Sprites show stray bleeding edge slivers from adjacent atlas frames"],
        "unanswered_questions_and_mysteries": [],
        "overall_fun_rating": 4,
    }

    fake_client = _QueuedFakeGeminiClient(
        [
            {"payload": t1},
            {"payload": t2},
            {"payload": t3},
            {"payload": report_payload},
        ]
    )
    agent = BlindPlaytesterAgent(store_card=card, client=fake_client)
    assert "visual_and_ui_defects_observed" in agent.report_schema.get("required", [])
    assert agent.config.report_model_name == DEFAULT_PLAYTEST_REPORT_MODEL == "gemini-3.8-flash"

    obs1 = _make_obs(tmp_path, 1, bg_rgb=(20, 30, 50), marker_x=30, audio_is_silent=True)
    obs2 = _make_obs(tmp_path, 2, bg_rgb=(180, 140, 40), marker_x=220, audio_is_silent=True)
    obs3 = _make_obs(tmp_path, 3, bg_rgb=(50, 120, 90), marker_x=400, audio_is_silent=True)

    _, _, stats1 = agent.decide_turn(1, obs1)
    assert stats1["image_resolution"] == "MEDIUM"

    # Turn 2 MUST auto-upgrade to HIGH resolution because graphics_and_visuals was recorded prior to any HIGH frame
    _, _, stats2 = agent.decide_turn(2, obs2)
    assert stats2["image_resolution"] == "HIGH"
    t2_tail_text = fake_client.calls[1]["contents"][-1].parts[0].text
    assert "Visual Acuity Note:" in t2_tail_text
    assert "Rough presentation" in agent.impressions["graphics_and_visuals"]

    # Turn 3 returns to standard MEDIUM since HIGH verification already occurred
    _, _, stats3 = agent.decide_turn(3, obs3)
    assert stats3["image_resolution"] == "MEDIUM"

    rep, _ = agent.generate_experience_report(
        end_status="completed",
        end_outcome="finished_exploring",
        final_screenshot_path=obs3.screenshot_path,
    )
    assert len(rep.visual_and_ui_defects_observed) == 2
    assert "sprite-sheet slicing bleed" in rep.graphics_and_visuals

    rep_call = fake_client.calls[3]
    assert rep_call["model"] == "gemini-3.8-flash"
    assert rep_call["config"].thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert rep_call["config"].temperature is None
    assert fake_client.calls[0]["config"].temperature is None
    rep_img_parts = [
        p for p in rep_call["contents"][-1].parts if p.inline_data and p.inline_data.mime_type == "image/png"
    ]
    # Opening (obs1) + distinct mid-session (obs2) + final (obs3) attached at HIGH resolution
    assert len(rep_img_parts) == 3
    for img_part in rep_img_parts:
        assert img_part.media_resolution.level.value == "MEDIA_RESOLUTION_HIGH"


def test_stratified_5_frame_report_sampling_skips_dark_modal_blackouts(tmp_path: Path):
    card = BlindStoreCard(
        deliberation="Test 5-frame stratified report screenshot selection across open-world and UI turns.",
        checkpoint_id="cp_stratified",
        game_title="Archipelago Survey",
        game_description="Explore islands and craft tools.",
    )
    turns = [_turn_payload(i, inspect_high_res_next=(i == 6), plan=f"Step {i}") for i in range(1, 9)]
    report_payload = {
        "deliberation": "Auditing all 5 stratified HIGH-res frames.",
        "overall_experience_summary": "Tested open world and UI.",
        "visual_and_ui_defects_observed": ["Blank icon slots in workbench panel"],
        "graphics_and_visuals": "Workbench slots missing icons.",
        "gameplay_and_feel": "Responsive movement.",
        "sound_and_audio": "Silent.",
        "intuitiveness_and_onboarding": "Clear prompts.",
        "goals_pacing_and_accomplishment": "Surveyed first zone.",
        "world_and_atmosphere": "Bright coastline.",
        "key_highlights": ["Responsive movement"],
        "key_pain_points": ["Missing recipe icons"],
        "bugs_and_glitches_observed": ["Empty icon boxes in crafting panel"],
        "unanswered_questions_and_mysteries": [],
        "overall_fun_rating": 5,
    }
    fake_client = _QueuedFakeGeminiClient([{"payload": t} for t in turns] + [{"payload": report_payload}])
    agent = BlindPlaytesterAgent(store_card=card, client=fake_client)

    # Turn 5 is a dark modal blackout (luminance < 35); neighbors Turn 4 & Turn 6 are bright open-world/UI frames!
    bgs = [
        (40, 90, 140),
        (60, 130, 170),
        (180, 170, 120),
        (150, 190, 110),
        (5, 5, 8),       # Turn 5 dark blackout
        (200, 185, 140), # Turn 6 bright
        (220, 160, 90),  # Turn 7 HIGH-res (requested by Turn 6 inspect_high_res_next=True)
        (80, 110, 150),
    ]
    obs_list = [
        _make_obs(tmp_path, idx, bg_rgb=bgs[idx - 1], marker_x=40 * idx, audio_is_silent=True)
        for idx in range(1, 9)
    ]
    for idx, obs in enumerate(obs_list, start=1):
        agent.decide_turn(idx, obs)

    _, _ = agent.generate_experience_report(
        end_status="completed",
        end_outcome="finished_exploring",
        final_screenshot_path=obs_list[-1].screenshot_path,
    )
    rep_parts = fake_client.calls[-1]["contents"][-1].parts
    rep_imgs = [p for p in rep_parts if p.inline_data and p.inline_data.mime_type == "image/png"]
    rep_labels = [p.text for p in rep_parts if p.text and "HIGH resolution for visual/UI craft audit" in p.text]
    assert len(rep_imgs) == 5
    # Turn 5 dark blackout was avoided in favor of bright neighboring open-world frame
    assert not any("Turn #5)" in lbl for lbl in rep_labels)


def test_turn_timing_telemetry_reports_inter_frame_dt_and_ballistic_click_window(tmp_path: Path):
    card = BlindStoreCard(
        deliberation="Verify domain-agnostic frame delta and ballistic click timing telemetry.",
        checkpoint_id="cp_timing",
        game_title="Timing Telemetry Fixture",
        game_description="A minimal interactive viewport.",
    )
    cfg = PlaytestSessionConfig()
    assert cfg.reaction_ticks == 6

    fake_client = _QueuedFakeGeminiClient(
        [
            {"payload": _turn_payload(1, inspect_high_res_next=False)},
            {"payload": _turn_payload(2, inspect_high_res_next=False)},
        ]
    )
    agent = BlindPlaytesterAgent(store_card=card, config=cfg, client=fake_client)
    assert "velocity = (pos_now - pos_prev) / Δt" in agent.system_prompt
    assert "remainder of `duration_ms` simulates the world AFTER your click" in agent.system_prompt

    obs1 = _make_obs(tmp_path, 1, marker_x=60, audio_is_silent=True)   # t = 0.75s
    obs2 = _make_obs(tmp_path, 2, marker_x=180, audio_is_silent=True)  # t = 1.50s (Δt = 0.75s)

    _, tape1, _ = agent.decide_turn(1, obs1)
    _, tape2, _ = agent.decide_turn(2, obs2)

    t1_status = fake_client.calls[0]["contents"][-1].parts[0].text
    assert "Click Timing: A click/tap issued now lands ~0.15-0.23s after Current screen (100ms reaction + fast cursor flick)" in t1_status
    assert "[Turn 1 | t=0.75s |" in tape1

    t2_parts = fake_client.calls[1]["contents"][-1].parts
    t2_status = t2_parts[0].text
    assert "+0.75s since Previous turn screen" in t2_status
    assert "Frame & Click Timing: Δt = 0.75s elapsed between Previous turn screen (0.75s) and Current screen (1.50s)." in t2_status
    assert any(p.text and "Δt=0.75s before Current screen" in p.text for p in t2_parts)
    assert "[Turn 2 | t=1.50s (+0.75s) |" in tape2
