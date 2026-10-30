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
    build_blind_player_system_prompt,
    compute_frame_mse,
    create_isolated_playtest_genai_client,
    extract_blind_store_card,
)
from engine.playtest.godot_bridge import TurnObservation, write_pcm16_to_wav
from engine.playtest.models import (
    BlindStoreCard,
    PlaytestBudgetExceededError,
    PlaytestCompactedJournal,
    PlaytestEndDecision,
    PlaytestExperienceReport,
    PlaytestImpressionUpdate,
    PlaytestMoodNote,
    PlaytestMotorAction,
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

    # Verify terminal decision/score fields are strictly LAST (Rule 4)
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



