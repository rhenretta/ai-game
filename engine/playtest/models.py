"""engine/playtest/models.py
Rationale-First Epistemic Data Models for the Blind Playtester Agent (Rule 4).
Every model inherits from BaseEpistemicModel with `deliberation` strictly at index 0.
Zero domain-specific heuristics or hardcoded game assumptions (Rules 1 & 2).
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, field_validator

from engine.studio.models import BaseEpistemicModel, DEFAULT_STUDIO_MODEL, StudioAgentError


DEFAULT_PLAYTEST_MODEL = "gemini-3.5-flash-lite"
DEFAULT_PLAYTEST_REPORT_MODEL = DEFAULT_STUDIO_MODEL
PLAYTEST_VIEWPORT_W = 1280
PLAYTEST_VIEWPORT_H = 720
NORM_COORD_SCALE = 1000.0


def denormalize_screen_x(norm_x: float, viewport_w: int = PLAYTEST_VIEWPORT_W) -> float:
    """Converts Gemini's native [0..1000] normalized X coordinate to viewport pixels."""
    return max(0.0, min(float(viewport_w), (float(norm_x) / NORM_COORD_SCALE) * float(viewport_w)))


def denormalize_screen_y(norm_y: float, viewport_h: int = PLAYTEST_VIEWPORT_H) -> float:
    """Converts Gemini's native [0..1000] normalized Y coordinate to viewport pixels."""
    return max(0.0, min(float(viewport_h), (float(norm_y) / NORM_COORD_SCALE) * float(viewport_h)))


def normalize_screen_x(px_x: float, viewport_w: int = PLAYTEST_VIEWPORT_W) -> float:
    """Converts viewport pixel X into Gemini's native [0..1000] normalized X coordinate."""
    return max(0.0, min(NORM_COORD_SCALE, (float(px_x) / float(max(1, viewport_w))) * NORM_COORD_SCALE))


def normalize_screen_y(px_y: float, viewport_h: int = PLAYTEST_VIEWPORT_H) -> float:
    """Converts viewport pixel Y into Gemini's native [0..1000] normalized Y coordinate."""
    return max(0.0, min(NORM_COORD_SCALE, (float(px_y) / float(max(1, viewport_h))) * NORM_COORD_SCALE))


class PlaytestRunnerError(StudioAgentError):
    """Raised when the Godot playtest runner or media compositor encounters an authentic failure (Rule 5)."""
    deliberation: str = "Authentic playtest runtime or media error surfaced directly to observability."


class PlaytestBudgetExceededError(StudioAgentError):
    """Raised when an isolated playtest session reaches its configured token ceiling."""
    deliberation: str = "Playtest session reached its configured token budget ceiling."


class BlindStoreCard(BaseEpistemicModel):
    """Opaque-box player briefing containing strictly what a new player sees on a store page."""
    checkpoint_id: str = Field(..., description="Target slice checkpoint identifier.")
    candidate_id: Optional[str] = Field(default=None, description="Parent candidate cartridge identifier.")
    slice_id: Optional[str] = Field(default=None, description="Completed vertical slice identifier if applicable.")
    game_title: str = Field(..., min_length=1, description="Public title of the game.")
    game_description: str = Field(
        default="",
        description="Short public store premise/blurb with zero internal rules, code, or walkthrough hints.",
    )


class PlaytestSessionConfig(BaseEpistemicModel):
    """Per-session execution and pacing parameters for the Blind Playtester."""
    deliberation: str = Field(
        default="Configuring decoupled blind playtest session with human reaction window, ballistic cursor pacing, and prefix-cached compaction.",
        description="MANDATORY FIRST PROPERTY: Epistemic justification for session parameters.",
    )
    model_name: str = Field(default=DEFAULT_PLAYTEST_MODEL, description="Multimodal Gemini model for player turns.")
    report_model_name: str = Field(
        default=DEFAULT_PLAYTEST_REPORT_MODEL,
        description="Higher-acuity multimodal Gemini model used once at session end to audit HIGH-resolution screenshots and synthesize PlaytestExperienceReport.",
    )
    reaction_ticks: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Simulated 60 FPS frames (~250ms at 15 ticks) advanced after frame capture before holding for LLM reply.",
    )
    compaction_interval_turns: int = Field(
        default=30,
        ge=6,
        le=120,
        description="Number of new append-only turns before compacting older history into the episodic journal.",
    )
    compaction_overlap_turns: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Most recent raw turns carried verbatim across a compaction seam to prevent mid-task amnesia.",
    )
    max_turns: int = Field(
        default=150,
        ge=1,
        le=1000,
        description="Safety turn ceiling backstopping autonomous end_playtest termination.",
    )
    max_session_tokens: int = Field(
        default=2_000_000,
        ge=10_000,
        description="Isolated per-session token safety ceiling.",
    )
    enable_audio_perception: bool = Field(
        default=True,
        description="Attach non-silent 16kHz WAV game audio slices (~25 tokens/sec) to player turns.",
    )


class PlaytestMotorAction(BaseEpistemicModel):
    """Human-scale motor command executed across deterministic 60 FPS ticks.
    Uses Gemini's native [0..1000] x [0..1000] normalized image coordinate system.
    """
    mouse_x: Optional[float] = Field(
        default=None,
        description="Target normalized horizontal coordinate from 0 to 1000 (0=left edge, 500=center, 1000=right edge). Leave null when keeping the cursor at its current position.",
    )
    mouse_y: Optional[float] = Field(
        default=None,
        description="Target normalized vertical coordinate from 0 to 1000 (0=top edge, 500=center, 1000=bottom edge). Leave null when keeping the cursor at its current position.",
    )
    mouse_button_action: Literal[
        "none",
        "left_click",
        "right_click",
        "double_click",
        "drag_left",
        "hold_left",
        "release_left",
        "wheel_up",
        "wheel_down",
    ] = Field(
        default="none",
        description="Mouse button or scroll interaction performed at the target position (or dragged to target when drag_left).",
    )
    drag_from_x: Optional[float] = Field(
        default=None,
        description="Optional drag start X (0..1000 normalized) when mouse_button_action is 'drag_left' and your cursor is not already on the object you want to grab; leave null otherwise.",
    )
    drag_from_y: Optional[float] = Field(
        default=None,
        description="Optional drag start Y (0..1000 normalized) when mouse_button_action is 'drag_left'; leave null otherwise.",
    )
    hold_keys: List[str] = Field(
        default_factory=list,
        description="Keyboard keys held down across duration_ms (e.g. ['W'], ['D', 'SHIFT'], ['ARROW_RIGHT']).",
    )
    tap_keys: List[str] = Field(
        default_factory=list,
        description="Keyboard keys tapped briefly at the start of the action (e.g. ['E'], ['1'], ['SPACE'], ['ENTER'], ['ESC']).",
    )
    duration_ms: int = Field(
        default=550,
        description="Duration of this action burst in milliseconds (350ms to 2000ms for natural human pacing).",
    )
    inspect_high_res_next: bool = Field(
        default=False,
        description="Set True early (e.g. Turn 1 -> Turn 2) before praising visual/UI cleanliness and whenever closely inspecting UI panels, icons, sprite edges, or small on-screen text.",
    )

    @field_validator("mouse_x", "mouse_y", "drag_from_x", "drag_from_y", mode="before")
    @classmethod
    def _clamp_norm_coord(cls, v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        return max(0.0, min(1000.0, float(v)))

    @field_validator("duration_ms", mode="before")
    @classmethod
    def _clamp_duration_ms(cls, v: Any) -> int:
        if v is None or v == "":
            return 550
        return max(350, min(2000, int(round(float(v)))))

    @field_validator("hold_keys", "tap_keys", mode="before")
    @classmethod
    def _clean_key_list(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            return [str(k).strip().upper() for k in v if str(k).strip()][:6]
        return []


PlaytestImpressionPillar = Literal[
    "graphics_and_visuals",
    "gameplay_and_feel",
    "sound_and_audio",
    "intuitiveness_and_controls",
    "goals_and_pacing",
    "world_and_atmosphere",
]

IMPRESSION_PILLAR_LABELS: Dict[str, str] = {
    "graphics_and_visuals": "Graphics & Visual Clarity",
    "gameplay_and_feel": "Gameplay, Agency & Feel",
    "sound_and_audio": "Sound & Audio Feedback",
    "intuitiveness_and_controls": "Intuitiveness & Controls",
    "goals_and_pacing": "Goals, Pacing & Accomplishment",
    "world_and_atmosphere": "World, Story & Atmosphere",
}


class PlaytestRulebookUpdate(BaseEpistemicModel):
    """Structured entry merged deterministically into the permanent, uncompacted How-To-Play Rulebook."""
    key_or_topic: str = Field(
        ...,
        min_length=1,
        description="Concise input key or mechanic name (e.g. 'WASD', 'Key E', 'Left Click', 'Energy Meter').",
    )
    observed_change_from_prior_turn: str = Field(
        default="",
        description="Concrete visual or audio delta ALREADY observed between the previous turn's screen/audio and the current screen/audio (never describe a predicted outcome of the motor_action you are about to execute).",
    )
    behavior: str = Field(
        ...,
        min_length=1,
        description="Concise 1-sentence behavior confirmed ONLY after observing the result on a subsequent turn (never assume an input worked on the same turn you trigger it; record 'does nothing / non-interactive' if repeated tries produce no state change).",
    )
    action: Literal["confirm", "retract"] = Field(
        default="confirm",
        description="Set 'confirm' to record or update a verified rule; set 'retract' to remove an earlier Rulebook entry if subsequent turns show counters/objects did not actually change. Strictly last.",
    )


class PlaytestFieldNoteUpdate(BaseEpistemicModel):
    """Structured screen/world observation merged deterministically into the player's Field Notes."""
    topic: str = Field(
        ...,
        min_length=1,
        description="Short label for what you are looking at or noting (e.g. 'Top-Left Counter', 'Center Structure', 'Bottom Toolbar', 'Cursor Icon').",
    )
    observation: str = Field(
        ...,
        min_length=1,
        description="Concise 1-2 sentence note describing what you see on screen, what state it shows, or how elements appear connected.",
    )


class PlaytestQuestionUpdate(BaseEpistemicModel):
    """Live player question ('what am I looking at?', 'what does X do?', 'why is Y like that?') tracked until answered."""
    question_key: str = Field(
        ...,
        min_length=1,
        description="Short stable snake_case key identifying this question so you can later update the exact same key to 'answered' once you figure it out.",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Natural first-person player question about what you are looking at, what an element/control does, or why something behaved that way.",
    )
    observed_change_from_prior_turn: Optional[str] = Field(
        default=None,
        description="Leave null while status='open'. Before setting status='answered', cite the concrete visual or audio delta ALREADY visible between the previous turn screen and current screen—never answer a question on the same turn you are still testing it.",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Leave null while still investigating (or when reopening a mistaken answer); populate with your concise discovered answer ONLY once a subsequent turn visibly or audibly confirms it.",
    )
    status: Literal["open", "answered"] = Field(
        default="open",
        description="Set 'open' when posing, refining, or reopening an unconfirmed question; set 'answered' ONLY after comparing previous vs current screen/audio proves the answer already happened. Strictly last.",
    )


class PlaytestBugReport(BaseEpistemicModel):
    """Concrete bug or glitch spotted by the blind player during live gameplay."""
    bug_key: str = Field(
        ...,
        min_length=1,
        description="Short snake_case identifier for deduplicating this specific bug or glitch across the session.",
    )
    category: Literal[
        "visual_glitch",
        "audio_glitch",
        "input_ui_bug",
        "gameplay_bug",
        "progression_blocker",
        "performance_issue",
    ] = Field(..., description="Category of bug or glitch observed.")
    summary: str = Field(
        ...,
        min_length=1,
        description="Crisp 1-line headline summarizing the bug or glitch.",
    )
    details: str = Field(
        ...,
        min_length=1,
        description=(
            "1-3 concise sentences describing what happened or looked broken versus what a player expects "
            "(for visual_glitch / input_ui_bug, cite concrete screen flaws such as cut-off sprites or stray bleeding "
            "edge slivers from neighboring atlas frames, squished/distorted UI icons, text overflowing panel borders "
            "or chopped off mid-word, overlapping HUD elements, harsh unblended rectangular texture boxes inside shapes, "
            "or crude untextured placeholder polygons clashing with detailed sprites)."
        ),
    )
    severity: Literal["minor", "major", "blocker"] = Field(
        default="major",
        description="Impact on the player experience ('minor'=small cosmetic/polish issue, 'major'=noticeable bug or broken behavior, 'blocker'=prevents further progress). Strictly last.",
    )


class PlaytestImpressionUpdate(BaseEpistemicModel):
    """Sparse, non-repetitive update to the player's evolving evaluation of a core experience dimension."""
    pillar: PlaytestImpressionPillar = Field(
        ...,
        description="Experience pillar being formed or revised.",
    )
    impression: str = Field(
        ...,
        min_length=1,
        description=(
            "Concise 1-2 sentence candid player impression grounded strictly in what is visibly/audibly rendered—"
            "never echo Store Page Description marketing copy, and never call graphics_and_visuals or UI 'clean', "
            "'polished', or 'legible' if sprites have cut-off/bleeding edges, icons are squished, text overflows or "
            "clips panel borders, textures show harsh rectangular seams, or raw flat polygons clash with detailed art. "
            "Revise earlier impressions whenever HIGH-resolution inspection or new screens reveal flaws."
        ),
    )


class PlaytestMoodNote(BaseEpistemicModel):
    """Sparse player mood-shift entry emitted ONLY on genuine emotional turning points."""
    mood: str = Field(
        ...,
        min_length=1,
        description="Natural 1-3 word mood label (e.g. intrigued, frustrated, amused, bored, satisfied, confused).",
    )
    note: str = Field(
        ...,
        min_length=1,
        description="1-2 concise sentences explaining what genuinely changed how you feel. Do not emit for routine clicks or unchanged moods.",
    )


class PlaytestWishDraft(BaseEpistemicModel):
    """Explicit command payload when the player decides to submit a wish/feedback item for the studio."""
    category: Literal[
        "controls_ux",
        "gameplay_feel",
        "world_exploration",
        "visual_clarity",
        "story_characters",
        "audio",
        "progression_blocker",
    ] = Field(..., description="Area of the player experience this wish addresses.")
    summary: str = Field(
        ...,
        min_length=1,
        description="Short 1-line player-facing headline summarizing the wish.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="1-3 concise sentences from your perspective as the player explaining what happened and why.",
    )


class PlaytestEndDecision(BaseEpistemicModel):
    """Explicit command to stop playing once the game is finished or further progress is impossible."""
    outcome: Literal["completed", "stuck_cannot_progress", "finished_exploring"] = Field(
        ...,
        description="Why you are ending the playtest session now.",
    )
    closing_thoughts: str = Field(
        ...,
        min_length=1,
        description="Concise 2-4 sentence closing takeaway on whether you had fun and how you felt about the world, visuals, story, and characters.",
    )


class PlaytestTurnDecision(BaseEpistemicModel):
    """Complete structured output for a single Blind Playtester turn.
    Evaluates concrete prior-turn observations (`turn_log_entry`, `field_note_updates`, `question_updates`,
    `rulebook_updates`, `bug_reports`) BEFORE formulating `current_plan` and `motor_action`, preventing
    autoregressive bleed from unexecuted future plans into confirmed facts (Rule 4).
    """
    turn_log_entry: str = Field(
        ...,
        min_length=1,
        description="Compact 1-line factual note of what visibly or audibly changed between the previous turn screen and current screen (say 'no counter/state change observed' when values stayed identical).",
    )
    field_note_updates: List[PlaytestFieldNoteUpdate] = Field(
        default_factory=list,
        description="Jot or revise concise notes on what you are currently looking at (screen layout, exact HUD numbers, objects, UI states). Leave [] when nothing new is noticed.",
    )
    question_updates: List[PlaytestQuestionUpdate] = Field(
        default_factory=list,
        description="Ask genuine player questions with status='open', update an existing question_key to status='answered' ONLY when the current screen/audio proves the outcome of a PRIOR turn's input (never for the motor_action you are about to run now), or set status='open' with answer=null to reopen an earlier answer that proved mistaken.",
    )
    rulebook_updates: List[PlaytestRulebookUpdate] = Field(
        default_factory=list,
        description="Update your permanent Rulebook ONLY after comparing the previous screen vs current screen confirms what a PRIOR turn's input actually did (or didn't do). Use action='retract' to remove any earlier Rulebook entry that subsequent turns disproved.",
    )
    bug_reports: List[PlaytestBugReport] = Field(
        default_factory=list,
        description="Log any bugs, glitches, unresponsive prompted controls, or things that feel broken as you spot them along the way. Leave [] on most turns and do not duplicate already-logged bugs.",
    )
    impression_updates: List[PlaytestImpressionUpdate] = Field(
        default_factory=list,
        description="Leave empty ([]) on most turns. Update when you form a new impression or revise an earlier one because things changed as you kept playing—never repeat unchanged notes.",
    )
    current_mood: str = Field(
        default="",
        description="Your current 1-3 word emotional state as a player (keep the exact same word as Your Current Mood unless how you genuinely feel has meaningfully changed—do NOT flip back and forth on routine clicks or steps).",
    )
    mood_shift: Optional[PlaytestMoodNote] = Field(
        default=None,
        description="Leave null on almost all turns. Populate ONLY when your emotional state genuinely shifts at a real turning point (e.g. delight at a breakthrough, surprise, mounting confusion, boredom, or frustration when stuck)—never for routine clicks, minor confirmations, or unchanged moods.",
    )
    current_plan: str = Field(
        ...,
        min_length=1,
        description="1-2 concise sentences on what you are actively doing or testing across the next few steps (carried verbatim across compactions).",
    )
    motor_action: PlaytestMotorAction = Field(
        ...,
        description="Mouse and/or keyboard input burst to execute now (runs AFTER this turn decision).",
    )
    submit_wish: Optional[PlaytestWishDraft] = Field(
        default=None,
        description="Leave null unless you want to file an explicit player wish or UX/design improvement for the studio.",
    )
    end_playtest: Optional[PlaytestEndDecision] = Field(
        default=None,
        description="Leave null while still actively progressing or testing fresh ideas; populate once you complete the available objectives or confirm you cannot progress any further.",
    )


class PlaytestCompactedJournal(BaseEpistemicModel):
    """Compacted episodic memory (how-to-play Rulebook, Field Notes, Questions, and Bugs are stored separately and never summarized away)."""
    episodic_story_and_world_chronicle: List[str] = Field(
        default_factory=list,
        description="Concise chronological bullets of areas explored, story/character moments, and milestones achieved.",
    )
    world_landmarks_map: List[str] = Field(
        default_factory=list,
        description="Spatial mental map of notable places, characters, doors, or objects and their relative directions.",
    )
    failed_attempts_and_dead_ends: List[str] = Field(
        default_factory=list,
        description="Approaches already tested that failed or led nowhere, preventing repetitive loops.",
    )
    open_questions_and_hypotheses: List[str] = Field(
        default_factory=list,
        description="Mysteries, unexplored paths, or mechanics you still want to test.",
    )
    fun_and_aesthetic_impressions: str = Field(
        default="",
        description="Honest running player impression of fun, world atmosphere, graphics, audio, story, and characters.",
    )


class PlaytestExperienceReport(BaseEpistemicModel):
    """Holistic player experience report synthesized at the end of a stopped or completed playtest session."""
    overall_experience_summary: str = Field(
        ...,
        min_length=1,
        description="First-person narrative overview of your playtest session: what you tried, how far you actually progressed, why the session ended, and whether you genuinely had fun.",
    )
    visual_and_ui_defects_observed: List[str] = Field(
        default_factory=list,
        description=(
            "Concrete visual, sprite, texture, or UI layout defects visible in the attached HIGH-resolution session screenshots "
            "(e.g. cut-off sprites or stray floating edge slivers bleeding from adjacent sprite-sheet frames, squished/stretched "
            "or microscopic icons in UI slots, text spilling outside panel borders or truncated mid-word, overlapping HUD panels, "
            "harsh unblended rectangular texture boxes pasted inside irregular polygons, or crude flat untextured placeholder "
            "shapes clashing with detailed raster art). Audit the screenshots carefully BEFORE writing graphics_and_visuals; "
            "leave [] ONLY when rendering, sprites, textures, and UI layout are genuinely free of visible defects."
        ),
    )
    graphics_and_visuals: str = Field(
        ...,
        min_length=1,
        description=(
            "Candid, unvarnished player impression of visual craft, sprite/texture cleanliness, art cohesion, spatial readability, "
            "and UI/HUD layout legibility grounded directly in visual_and_ui_defects_observed—never parrot store-blurb marketing "
            "prose or describe graphics/UI as 'clean', 'polished', or 'highly legible' when visual_and_ui_defects_observed is non-empty."
        ),
    )
    gameplay_and_feel: str = Field(
        ...,
        min_length=1,
        description="Honest player impression of core loop engagement, control responsiveness, player agency, interesting choices, and moment-to-moment feel.",
    )
    sound_and_audio: str = Field(
        ...,
        min_length=1,
        description="Honest player impression of sound effects, ambience/music, and audio feedback across the session.",
    )
    intuitiveness_and_onboarding: str = Field(
        ...,
        min_length=1,
        description="Honest player impression of how intuitive it was to figure out how to play, discover controls, and read affordances without a manual.",
    )
    goals_pacing_and_accomplishment: str = Field(
        ...,
        min_length=1,
        description="Honest player impression of short- and long-term objective clarity, pacing, actual progression achieved, and sense of accomplishment.",
    )
    world_and_atmosphere: str = Field(
        ...,
        min_length=1,
        description="Honest player impression of world cohesion, tone, setting, and any story or character presence.",
    )
    key_highlights: List[str] = Field(
        default_factory=list,
        description="Concrete moments or design elements that worked well and felt rewarding or fun.",
    )
    key_pain_points: List[str] = Field(
        default_factory=list,
        description="Concrete confusions, UX/control snags, missing feedback, balance issues, or progression blockers encountered.",
    )
    bugs_and_glitches_observed: List[str] = Field(
        default_factory=list,
        description="Concrete bugs, glitches, or broken behaviors noticed during play.",
    )
    unanswered_questions_and_mysteries: List[str] = Field(
        default_factory=list,
        description="Player questions ('what am I looking at?', 'what does X do?', 'why is Y like that?') that remained unanswered when the session ended.",
    )
    overall_fun_rating: int = Field(
        ...,
        description=(
            "Honest, non-inflated 1-to-10 player fun & experience rating grounded in verified screen/audio evidence, "
            "actual objective progression, and real player agency "
            "(1-2=crashes, blank, or completely unresponsive; "
            "2-3=boots and renders visuals/UI, but core gameplay loop or primary objectives stay stuck/unresponsive or trap you in repetitive no-op actions; "
            "4=partially playable, but undermined by stalled objectives, broken/overlapping/overflowing UI, noticeable sprite-slicing or texture-seam glitches, or auto-playing shortcuts lacking real player agency; "
            "5-6=core gameplay loop works end-to-end with real player-driven progression, though held back by silent/missing audio, rough UI/visual craft defects, or shallow pacing; "
            "7-8=cohesive, intuitive, responsive, player-driven loop with clean glitch-free visuals/UI, working audio/visual feedback, and genuine fun; "
            "9-10=standout, polished, bug-free, deeply engaging experience—never award 7-10 if objectives stalled, actions looped without effect, audio was silent, visual/UI rendering had glaring defects, or the game played itself). Strictly last."
        ),
    )

    @field_validator("overall_fun_rating", mode="before")
    @classmethod
    def _clamp_fun_rating(cls, v: Any) -> int:
        if v is None or v == "":
            return 5
        return max(1, min(10, int(round(float(v)))))


class PlaytestSessionStartRequest(BaseEpistemicModel):
    """API request payload to launch a Blind Playtester session on a completed checkpoint or slice."""
    deliberation: str = Field(
        default="Operator requested a new Blind Playtester session from Mission Control.",
        description="MANDATORY FIRST PROPERTY: Epistemic justification.",
    )
    checkpoint_id: Optional[str] = Field(default=None, description="Explicit checkpoint directory name.")
    candidate_id: Optional[str] = Field(default=None, description="Candidate cartridge ID when resolving by slice_id.")
    slice_id: Optional[str] = Field(default=None, description="Completed slice_id to resolve to its checkpoint.")
    initiated_by: Literal["operator", "council"] = Field(
        default="operator",
        description="Whether launched manually by the human operator ('operator' — wishes stay pending) or automatically by the Council at slice end ('council' — report and wishes auto-delivered to Council).",
    )
    store_title_override: Optional[str] = Field(
        default=None,
        description="Optional public game title when launching directly against a candidate slice directory.",
    )
    store_description_override: Optional[str] = Field(
        default=None,
        description="Optional public 1-sentence store elevator pitch when launching against a candidate slice directory.",
    )
    max_turns: int = Field(default=150, ge=1, le=1000)
    reaction_ticks: int = Field(default=15, ge=1, le=60)
    compaction_interval_turns: int = Field(default=30, ge=6, le=120)
    enable_audio_perception: bool = Field(default=True)


class PlaytestSessionControlRequest(BaseEpistemicModel):
    """API request payload to pause, resume, stop, or reset a playtest session."""
    deliberation: str = Field(
        default="Operator issued lifecycle control command for Blind Playtester session.",
        description="MANDATORY FIRST PROPERTY: Epistemic justification.",
    )
    action: Literal["pause", "resume", "stop", "reset"]


class PlaytestWishDispatchRequest(BaseEpistemicModel):
    """API request payload to dispatch 1 or more approved Wish items immediately to the Council."""
    deliberation: str = Field(
        default="Executive Producer approved Playtester Wish item(s) for immediate Council ingestion.",
        description="MANDATORY FIRST PROPERTY: Epistemic justification.",
    )
    wish_ids: List[str] = Field(..., min_length=1, description="One or more wish_ids to send immediately to the Council.")
    producer_note: Optional[str] = Field(
        default=None,
        description="Optional Executive Producer direction accompanying the player wish(es).",
    )


class PlaytestReportDispatchRequest(BaseEpistemicModel):
    """API request payload to dispatch a finished Blind Playtester session's full report to the Council."""
    deliberation: str = Field(
        default="Executive Producer dispatched full Blind Playtester session report for immediate Council ingestion.",
        description="MANDATORY FIRST PROPERTY: Epistemic justification.",
    )
    include_pending_wishes: bool = Field(
        default=True,
        description="Whether to also approve and include any pending player wishes from this session in the Council dispatch.",
    )
    producer_note: Optional[str] = Field(
        default=None,
        description="Optional Executive Producer direction accompanying the full playtest session report.",
    )


class PlaytestTokenTracker:
    """Isolated per-session token & prefix-cache tracker (decoupled from Studio's build circuit breaker)."""

    def __init__(self, max_session_tokens: int = 2_000_000) -> None:
        self._lock = threading.Lock()
        self.max_session_tokens = max(10_000, int(max_session_tokens))
        self.total_calls: int = 0
        self.prompt_tokens: int = 0
        self.cached_tokens: int = 0
        self.output_tokens: int = 0
        self.thought_tokens: int = 0

    def check_budget(self) -> None:
        with self._lock:
            total = self.prompt_tokens + self.output_tokens + self.thought_tokens
            if total >= self.max_session_tokens:
                raise PlaytestBudgetExceededError(
                    f"Playtest session token budget reached ({total:,} / {self.max_session_tokens:,} tokens)."
                )

    def record_response(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        p_tok = int(getattr(usage, "prompt_token_count", 0) or getattr(usage, "input_tokens", 0) or 0) if usage else 0
        c_tok = int(getattr(usage, "cached_content_token_count", 0) or getattr(usage, "cached_tokens", 0) or 0) if usage else 0
        o_tok = int(getattr(usage, "candidates_token_count", 0) or getattr(usage, "output_tokens", 0) or 0) if usage else 0
        t_tok = int(getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
        with self._lock:
            self.total_calls += 1
            self.prompt_tokens += p_tok
            self.cached_tokens += c_tok
            self.output_tokens += o_tok
            self.thought_tokens += t_tok
        return {
            "prompt_tokens": p_tok,
            "cached_tokens": c_tok,
            "output_tokens": o_tok,
            "thought_tokens": t_tok,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            cache_hit_pct = (100.0 * self.cached_tokens / self.prompt_tokens) if self.prompt_tokens > 0 else 0.0
            return {
                "total_calls": self.total_calls,
                "prompt_tokens": self.prompt_tokens,
                "cached_tokens": self.cached_tokens,
                "cache_hit_pct": round(cache_hit_pct, 1),
                "output_tokens": self.output_tokens,
                "thought_tokens": self.thought_tokens,
                "total_tokens": self.prompt_tokens + self.output_tokens + self.thought_tokens,
                "max_session_tokens": self.max_session_tokens,
            }
