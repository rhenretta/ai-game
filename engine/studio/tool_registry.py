"""engine/studio/tool_registry.py

Canonical, 100% domain-agnostic Studio Tool & Persona Capability Registry.

Provides:
1. A compact baseline capability directory (`render_compact_tool_directory`) injected into
   TaskLead, Specialist Persona, and Council prompts so every agent always knows what
   Personas, internal sub-tools, and board/epistemic actions exist.
2. Granular semantic capability chunks (`build_tool_capability_chunks`) indexed into
   `CartridgeSemanticIndex` (`category="tool_capability"`) so any agent can progressively
   expand detailed tool schemas, modes, parameters, and composition strategies on demand
   via `tool_search_query`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

CANONICAL_SPECIALIST_PERSONAS: Tuple[str, ...] = (
    "visual_artist",
    "audio_designer",
    "script_writer",
    "swe_engineer",
    "qa_test_engineer",
)

SPECIALIST_ALIASES: Dict[str, str] = {
    "audio_evaluator": "audio_designer",
}


def normalize_specialist_persona(name: str) -> str:
    """Normalizes legacy or alias specialist identifiers to the 5 canonical Specialist Personas."""
    cleaned = str(name or "").strip()
    return SPECIALIST_ALIASES.get(cleaned, cleaned)


@dataclass(frozen=True)
class StudioCapabilitySpec:
    """Describes a Studio Specialist Persona, internal specialist sub-tool, or orchestration action."""

    capability_id: str
    persona_or_scope: str
    kind: Literal["persona", "subtool", "orchestration"]
    title: str
    compact_summary: str
    detailed_guide: str


STUDIO_CAPABILITIES: Tuple[StudioCapabilitySpec, ...] = (
    # ── 5 Canonical Unified Specialist Personas (Creator + Evaluator) ─────────
    StudioCapabilitySpec(
        capability_id="persona.visual_artist",
        persona_or_scope="visual_artist",
        kind="persona",
        title="Visual Artist Persona (Creator + Perceptual Visual Inspector)",
        compact_summary=(
            "Fabricates sprites/sheets/tilesets/UI/backgrounds/VFX (`AssetFabricator`), renders 1:1 pixel diagnostic "
            "contact sheets (with optional grids/crops), and runs native VLM visual inspection against style guide & 16:9 mockups."
        ),
        detailed_guide=(
            "SPECIALIST PERSONA: `visual_artist`\n"
            "Role: Unified Visual Creator & Perceptual Visual Evaluator.\n"
            "Delegation Inputs (`TaskLeadDecision` / `CouncilDiagnosticProbe`):\n"
            "  - `specialist_agent`: 'visual_artist'\n"
            "  - `brief`: Natural-language creative or diagnostic brief (what to fabricate, overwrite in place, render on a plate, or visually inspect).\n"
            "  - `target_paths`: Relative `.png` paths in the cartridge (e.g. `['assets/sprites/actor.png']`) to create, update in place, or inspect.\n"
            "Internal Operations (`VisualArtistPlan.operation`):\n"
            "  1. `fabricate`: Synthesizes or overwrites `.png` visual assets via `AssetFabricator` across modes (`static_sprite`, `sprite_sheet`, `articulated_rig`, `tileset`, `parallax_layer`, `ui_element`, `vfx_texture`), preserving the exact `target_path` requested without renaming.\n"
            "  2. `render_contact_sheet`: Deterministically renders true-pixel 1:1 diagnostic inspection plates (`proofs/*.png`) showing checkerboard alpha transparency, exact pixel dimensions (`WxH`), optional coordinate grid lines (`grid_px`), and optional zoomed region callouts (`crop_boxes`).\n"
            "  3. `inspect`: Sends target `.png` files + 1:1 diagnostic plate + optional 16:9 slice concept mockup to `VisualEvaluatorAgent` (Gemini native multimodal vision) to answer any perceptual question about silhouette, transparency/alpha cleanliness, orientation, sub-element layout, or mockup fidelity, and persists `perceptual_description` in `assets/asset_library.json`.\n"
            "  4. `fabricate_and_inspect`: Generates the visual asset(s) and immediately runs a 1:1 perceptual visual audit in a single delegation turn.\n"
            "  5. `cleanup`: Deletes obsolete/orphaned `assets/...` files listed in `delete_paths` and prunes them from `assets/asset_library.json`.\n"
            "  6. `reroute`: When sent the wrong brief (or when inspecting proves the `.png` is clean and the bug is in `.gd` region slicing/layout), sets `reroute_to_specialist` (e.g. `'swe_engineer'`), `reroute_brief`, and `reroute_target_paths` to hand off directly.\n"
            "Composition Tip: Before wiring unfamiliar sprite sheets or atlases in GDScript, delegate an `inspect` or `render_contact_sheet` brief to `visual_artist` (or spawn a `discovery` subtask) to verify exact pixel sub-regions and visual orientation."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="persona.audio_designer",
        persona_or_scope="audio_designer",
        kind="persona",
        title="Audio Designer Persona (Synthesizer + Perceptual Audio/Audiovisual Evaluator)",
        compact_summary=(
            "Synthesizes SFX, ambient loops, music stems, and neural voice barks (`.wav`), and listens directly to `.wav` "
            "files or synchronized `.mp4` gameplay clips via multimodal perception (`AudioEvaluatorAgent`)."
        ),
        detailed_guide=(
            "SPECIALIST PERSONA: `audio_designer` (also accepts legacy alias `'audio_evaluator'`)\n"
            "Role: Unified Sound Designer, Voice/Music Synthesizer & Perceptual Acoustic/Audiovisual Evaluator.\n"
            "Delegation Inputs:\n"
            "  - `specialist_agent`: 'audio_designer'\n"
            "  - `brief`: Acoustic synthesis spec OR perceptual listening question (timbre, clarity, loop seamlessness, voice delivery, event-to-sound sync, mix balance).\n"
            "  - `target_paths`: `.wav` file(s) to synthesize/overwrite in place and/or `.wav` + `.mp4` gameplay clips to listen to.\n"
            "Internal Operations (`AudioDesignerPlan.operation`):\n"
            "  1. `synthesize`: Generates `.wav` assets (`audio_sfx`, `audio_loop`, `audio_music`, `audio_bark` neural speech with unlimited spoken script fidelity), preserving exact `target_path`.\n"
            "  2. `inspect`: Listens directly to `.wav` cues and/or synchronized `.mp4` gameplay video+audio clips via Gemini native multimodal audio+video perception (`AudioEvaluatorAgent`), returning `AudioEvaluationReport` and enriching `perceptual_description` in `assets/asset_library.json`.\n"
            "  3. `synthesize_and_inspect`: Synthesizes `.wav` cues and immediately listens to verify acoustic quality, voice intelligibility, and tonal fit before returning.\n"
            "  4. `cleanup` / `reroute`: Delete obsolete `assets/audio/...` files via `delete_paths` (`cleanup`), or set `reroute_to_specialist` + `reroute_brief` (`reroute`) when the issue belongs to another domain (e.g. `'swe_engineer'` for GDScript `AudioStreamPlayer` trigger code)."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="persona.script_writer",
        persona_or_scope="script_writer",
        kind="persona",
        title="Script Writer Persona (Narrative Author + Continuity/Tone Inspector)",
        compact_summary=(
            "Authors character sheets, cutscenes, branching dialogue trees, and contextual barks (`NarrativeBible` & JSON), "
            "or performs read-only narrative continuity, voice, and hook audits."
        ),
        detailed_guide=(
            "SPECIALIST PERSONA: `script_writer`\n"
            "Role: Dramatic Writer & Narrative Continuity/Hook Auditor.\n"
            "Internal Operations (`ScriptWritingDeliverable.operation_mode`):\n"
            "  1. `author`: Crafts or updates character voice sheets, cutscenes, branching dialogue trees, and event-triggered barks in `game_bible.json` and `assets/narrative/dialogue_data.json`.\n"
            "  2. `inspect_only`: Lock-free read-only audit verifying character voice consistency, branching flag completeness, dialogue-to-bark audio coverage, and GDScript narrative hook alignment (`inspection_report` + `recommended_followups`).\n"
            "  3. `reroute`: Set `reroute_to_specialist` + `reroute_brief` when the brief belongs to another persona (`'swe_engineer'`, `'audio_designer'`, `'visual_artist'`)."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="persona.swe_engineer",
        persona_or_scope="swe_engineer",
        kind="persona",
        title="SWE Engineer Persona (Read-Only Architecture Inspector + Lazy-Locked Code Synthesizer)",
        compact_summary=(
            "Performs lock-free read-only code/scene/shader/asset-wiring audits (`inspect_only`) OR surgical GDScript/scene/shader "
            "synthesis (`edit`) verified by the headless Godot 4 compiler."
        ),
        detailed_guide=(
            "SPECIALIST PERSONA: `swe_engineer`\n"
            "Role: Lead Gameplay & Engine Software Engineer + Technical Architecture Auditor.\n"
            "Delegation Inputs:\n"
            "  - `specialist_agent`: 'swe_engineer'\n"
            "  - `brief`: Behavioral/architectural goal OR read-only technical audit question (never paste raw code into `brief`).\n"
            "  - `target_paths`: `.gd`, `.tscn`, `.tres`, `.gdshader`, or `.json` files to inspect or modify.\n"
            "Internal Operations (`TrackCodeSynthesisPayload.operation_mode`):\n"
            "  1. `inspect_only`: Lock-free read-only inspection! Does NOT acquire `code_lock`, so it can run concurrently while other workers edit code. Returns `inspection_report` citing exact file lines, node hierarchies, signal wiring, shader uniforms, and `asset_library.json` metadata.\n"
            "  2. `edit`: Lazily acquires per-file locks only when writing `patches` (`SurgicalCodePatch`), new `files` (`ModuleFileSpec`), or `delete_paths` (removing obsolete/orphaned cartridge files), plus optional `requested_assets`, followed by mandatory headless Godot 4 compiler syntax & boot verification.\n"
            "  3. `reroute`: Set `reroute_to_specialist` + `reroute_brief` (`operation_mode='reroute'`) when inspecting code proves the root cause belongs to `'visual_artist'`, `'audio_designer'`, `'script_writer'`, or `'qa_test_engineer'`."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="persona.qa_test_engineer",
        persona_or_scope="qa_test_engineer",
        kind="persona",
        title="QA Test Engineer Persona (Exploratory Runtime Probe + Causal Filmstrip & VLM Auditor)",
        compact_summary=(
            "Runs headless Godot 4 runtime probes (`exploratory_diagnostic`) or interactive multi-tick gameplay filmstrips "
            "(`causal_gameplay`), capturing 1:1 `.keyframe_fullres.png` plates, `.mp4` clips, telemetry, and VLM audits."
        ),
        detailed_guide=(
            "SPECIALIST PERSONA: `qa_test_engineer`\n"
            "Role: Runtime Simulation Engineer, Diagnostic Scene Builder & Multimodal Post-Capture Auditor.\n"
            "Internal Operations (`FilmstripPlan.verification_mode`):\n"
            "  1. `exploratory_diagnostic`: Executes an isolated or full-scene Godot runtime probe (`test_scene_gd_content`, `setup_calls`, `get_state_snapshot()` telemetry, visual debug overlays, or asset/layout staging) WITHOUT requiring player input choreography. Ideal for investigating runtime coordinates, node bounds, shader output, or camera framing.\n"
            "  2. `causal_gameplay`: Drives authentic player `input_segments` (`actions`, `mouse_click_pos`, `mouse_world_pos`, reactive `when_snapshot_key` triggers) and `state_assertions` across `capture_ticks`, optionally recording synchronized `.mp4` video+audio (`record_audiovisual_clip=True`).\n"
            "  3. `reroute`: Set `reroute_to_specialist` + `reroute_brief` (`verification_mode='reroute'`) when sent a brief that belongs to `'swe_engineer'`, `'visual_artist'`, `'audio_designer'`, or `'script_writer'`.\n"
            "Multimodal Audit (`QAPostCaptureAudit`): Automatically inspects the captured filmstrip AND crisp 1:1 `.keyframe_fullres.png` sidecar frames alongside the active slice's 16:9 concept art mockup, reporting honest pixel observations and any `production_defect_report`."
        ),
    ),
    # ── Detailed Sub-Tool Chunks for Progressive `tool_search_query` ──────────
    StudioCapabilitySpec(
        capability_id="subtool.visual_contact_sheet_and_vlm",
        persona_or_scope="visual_artist",
        kind="subtool",
        title="1:1 Pixel Contact Sheets, Grid Rulers, Crop Callouts & VLM Asset Inspection",
        compact_summary="Render pixel-accurate sprite/atlas plates (`grid_px`, `crop_boxes`) and enrich `perceptual_description` via VLM.",
        detailed_guide=(
            "SUB-TOOL: `VisualEvaluatorAgent` (invoked inside `visual_artist`)\n"
            "Parameters on `VisualArtistPlan`:\n"
            "  - `target_asset_paths`: List of `.png` files in cartridge (e.g. `['assets/sprites/sheet.png', 'assets/ui/icon.png']`).\n"
            "  - `contact_sheet_output_path`: Optional destination such as `'proofs/atlas_inspection.png'`.\n"
            "  - `grid_px`: Integer pixel spacing (e.g. `16`, `32`, `64`) to overlay subtle coordinate grid lines and pixel tick labels so frame/region boundaries can be read directly.\n"
            "  - `crop_boxes`: Optional list of `(x, y, w, h)` pixel rectangles on the primary asset rendered as 1:1 sub-region callout cards to verify individual frames or sub-sprites before wiring `region_rect` / `AtlasTexture`.\n"
            "  - `include_concept_mockup`: Attaches the slice's 16:9 concept mockup so VLM checks palette, proportion, and style cohesion.\n"
            "Side Effect: Updates `entries[asset_id].perceptual_description` in `assets/asset_library.json` so `swe_engineer` and `ContextCuratorAgent` know what each PNG actually depicts."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="subtool.visual_fabrication_modes",
        persona_or_scope="visual_artist",
        kind="subtool",
        title="AssetFabricator Visual Modes & In-Place Overwrite Semantics",
        compact_summary="Modes: `static_sprite`, `sprite_sheet`, `articulated_rig`, `tileset`, `parallax_layer`, `ui_element`, `vfx_texture`.",
        detailed_guide=(
            "SUB-TOOL: `AssetFabricator` Visual Synthesis (`plan_asset_requirement` + `visual.py`)\n"
            "In-Place Path Fidelity: When `AssetRequirement.target_path` specifies an explicit path (e.g. `'assets/ui/portrait_ protagonist.png'`), the fabricator writes directly to that exact path and updates any matching `asset_library.json` entry in place—never renaming or prefixing filenames.\n"
            "Modes (`fabrication_mode`):\n"
            "  - `static_sprite`: Single isolated character, prop, or item with clean alpha background (`target_display_size`).\n"
            "  - `sprite_sheet`: Multi-frame animation sheet (`hframes`, `vframes`, `animation_states`) with uniform cell sizing.\n"
            "  - `articulated_rig`: Multi-part character/object rig (`articulated_parts` with `pivot_normalized` and `attach_offset_px`).\n"
            "  - `tileset` / `parallax_layer` / `ui_element` / `vfx_texture`: Seamless tiles, multi-depth backgrounds, crisp HUD/UI chrome, or additive/alpha particle textures."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="subtool.audio_synthesis_and_multimodal_listening",
        persona_or_scope="audio_designer",
        kind="subtool",
        title="Audio Synthesis Modes & Native Multimodal `.wav` / `.mp4` Listening",
        compact_summary="Modes: `audio_sfx`, `audio_loop`, `audio_music`, `audio_bark` + `AudioEvaluatorAgent` listening.",
        detailed_guide=(
            "SUB-TOOL: `AssetFabricator` Audio Synthesis + `AudioEvaluatorAgent` Listening (invoked inside `audio_designer`)\n"
            "Synthesis Modes (`AssetRequirement.fabrication_mode`):\n"
            "  - `audio_sfx`: One-shot gameplay sound effects (impacts, UI clicks, footsteps, pickups).\n"
            "  - `audio_loop`: Seamless environmental/ambient bed loops.\n"
            "  - `audio_music`: Musical cues, themes, or adaptive stems.\n"
            "  - `audio_bark`: Neural speech / character voice lines (`spoken_text`, `voice_profile`—full script length honored without truncation).\n"
            "Perceptual Listening (`AudioDesignerPlan.operation = 'inspect' | 'synthesize_and_inspect'`):\n"
            "  - Accepts `.wav` files and/or `.mp4` gameplay clips in `media_paths_to_inspect`.\n"
            "  - Evaluates acoustic clarity, silence/clipping artifacts, voice intelligibility, emotional delivery, loop seam smoothness, and visual-to-audio timing sync in `.mp4` captures.\n"
            "  - Writes `perceptual_description` to `assets/asset_library.json`."
        ),
    ),
    # ── Orchestration, Discovery Subtasks, Task Splitting & Epistemic Search ──
    StudioCapabilitySpec(
        capability_id="orchestration.discovery_subtasks",
        persona_or_scope="all",
        kind="orchestration",
        title="Discovery Subtasks (`kind='discovery'`) & Live Council Diagnostic Probes",
        compact_summary=(
            "Spawn lock-free `kind='discovery'` subtasks (`spawn_tasks`) to investigate assets, code, or runtime state "
            "in parallel; worker-spawned discovery tasks auto-complete and inject findings + plates into the parent."
        ),
        detailed_guide=(
            "ORCHESTRATION CAPABILITY: Discovery Subtasks (`kind='discovery'`) & `CouncilDiagnosticProbe`\n"
            "When to Use:\n"
            "  - You need to inspect existing sprite sheets, verify runtime scene coordinates, test a hypothesis in Godot, or audit audio/code across modules before committing to an implementation strategy.\n"
            "How `TaskLead` Uses It (`TaskLeadDecision`):\n"
            "  - Set `action = 'spawn_tasks'`, populate `spawned_tasks` with one or more `WorkItemProposal(kind='discovery', target_modules=[], ...)` describing the exact question and expected diagnostic evidence (`proof_spec`), and set `wait_for_spawned_tasks = True` (to pause, release locks, and resume automatically once answered) or `False` (non-blocking).\n"
            "Execution & Fast Unblocking:\n"
            "  - `discovery` tasks run lock-free in parallel alongside active build workers and have access to all 5 Specialist Personas (`visual_artist`, `audio_designer`, `script_writer`, `swe_engineer`, `qa_test_engineer`).\n"
            "  - When a worker-spawned `discovery` task submits `discovery_findings` (+ optional diagnostic plates/contact sheets), it transitions directly to `approved` without waiting on a 4-person Council review queue, immediately waking the parent `TaskLead` with a `COMPLETED PREREQUISITE & DISCOVERY SUBTASK REPORTS` block and visual attachments.\n"
            "How Council Uses It:\n"
            "  - Council leads can emit `CouncilDiagnosticProbe` targeting ANY of the 5 Specialist Personas during review, escalation, or playtest, or propose `kind='discovery'` items on the Production Board."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="orchestration.split_task",
        persona_or_scope="task_lead",
        kind="orchestration",
        title="Self-Directed Task Decomposition (`action='split_task'`)",
        compact_summary=(
            "Break an oversized or multi-concern active task into `2..N` cohesive child tasks (`split_tasks`) with "
            "intra-batch `depends_on` ordering; downstream board dependents automatically rewire to the new children."
        ),
        detailed_guide=(
            "ORCHESTRATION CAPABILITY: Active Task Self-Splitting (`TaskLeadDecision.action = 'split_task'`)\n"
            "When to Use:\n"
            "  - A task bundles too many independent subsystems, requires multiple sequential milestones to verify cleanly, or reveals distinct asset + mechanic + integration phases that are safer and clearer as separate board items.\n"
            "Parameters on `TaskLeadDecision`:\n"
            "  - `split_tasks`: List of `>= 2` `WorkItemProposal`s. Children may reference each other in `depends_on` (e.g. `child_assets` -> `child_mechanic` -> `child_integration`) and automatically inherit the parent's upstream `depends_on`, `slice_id`, and `shaping_notes`.\n"
            "  - `preserve_verified_changes_on_split`: `False` (default) cleanly reverts uncommitted staging edits from the current partial turn before releasing locks; `True` keeps already-verified disk work completed in earlier turns.\n"
            "Board Semantics:\n"
            "  - `ProductionBoard.split_active_item` marks the parent `cancelled` (split), releases its file locks, adds all children, and rewires every downstream board item that depended on `parent_item_id` so it now waits on the leaf split children."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="orchestration.epistemic_queries",
        persona_or_scope="all",
        kind="orchestration",
        title="Epistemic Queries (`self_question`, `memory_query`, `tool_search_query`)",
        compact_summary=(
            "Query codebase/Bible/catalog chunks (`self_question`), institutional memory (`memory_query`), or detailed "
            "Studio tool schemas/strategies (`tool_search_query`) before acting."
        ),
        detailed_guide=(
            "ORCHESTRATION CAPABILITY: Progressive Epistemic & Tool Discovery\n"
            "Available across `TaskLead`, all 5 Specialist Personas, and Council:\n"
            "  1. `self_question`: Hybrid semantic + lexical retrieval over cartridge `.gd`/`.tscn`/`.gdshader` symbols, `assets/asset_library.json` (including `perceptual_description`), `game_bible.json`, and board history.\n"
            "  2. `memory_query` (+ `memory_query_bank`): Semantic lookup in `GameMemoryStore` for cross-slice architectural invariants and hard-won lessons.\n"
            "  3. `tool_search_query`: Progressive lookup over `STUDIO_CAPABILITIES` (`category='tool_capability'`) returning full parameter schemas, sub-modes, and composition recipes for any Persona, sub-tool, or board orchestration action without bloating baseline prompts."
        ),
    ),
    StudioCapabilitySpec(
        capability_id="orchestration.peer_rerouting_and_cleanup",
        persona_or_scope="all",
        kind="orchestration",
        title="Autonomous Peer Re-Routing (`reroute_to_specialist` / `route_to_council_role`) & Orphan Cleanup (`delete_paths`)",
        compact_summary=(
            "Any Specialist Persona or Council member sent the wrong issue can dynamically re-route to the right specialist "
            "(`reroute_to_specialist` / `route_to_council_role`), and specialists can delete obsolete files via `delete_paths`."
        ),
        detailed_guide=(
            "ORCHESTRATION CAPABILITY: Autonomous Peer Re-Routing & Sandboxed File Cleanup\n"
            "1. Specialist Persona Peer Handoff (`reroute_to_specialist`, `reroute_brief`, `reroute_target_paths`):\n"
            "   - Available on ALL 5 Specialist Personas (`VisualArtistPlan`, `VisualEvaluationReport`, `AudioDesignerPlan`, `ScriptWritingDeliverable`, `TrackCodeSynthesisPayload`, `FilmstripPlan`).\n"
            "   - If a specialist receives a brief belonging to another domain—or inspects an artifact and proves the defect lies elsewhere (e.g. `visual_artist` inspects an atlas PNG, verifies all cells are intact, and determines `.gd` region slicing is wrong)—it sets `reroute_to_specialist` (e.g. `'swe_engineer'`), `reroute_brief`, and `reroute_target_paths`.\n"
            "   - The dispatcher automatically forwards the handoff to the target specialist (up to 2 hops, cycle-safe) and unlocks cross-domain specialists on the task.\n"
            "2. Council Peer Handoff (`WritersRoomTurnProposal.route_to_council_role`, `routing_handoff_reason`):\n"
            "   - Any Council speaker can route the floor directly to another Council specialist when a design or steering issue belongs to or intersects their domain.\n"
            "3. Sandboxed Orphan Asset Cleanup (`delete_paths`):\n"
            "   - `visual_artist`, `audio_designer`, and `swe_engineer` can list obsolete/unreferenced cartridge files in `delete_paths` (`operation='cleanup'` or alongside edits) to delete dead files on disk and prune them from `assets/asset_library.json`."
        ),
    ),
)


def render_compact_tool_directory(
    allowed_specialists: Optional[Sequence[str]] = None,
    *,
    layer: Literal["task_lead", "specialist", "council"] = "task_lead",
    active_persona: str = "",
) -> str:
    """Renders a compact, token-efficient baseline directory of available Studio capabilities."""
    normalized_allowed: Optional[List[str]] = None
    if allowed_specialists is not None:
        normalized_allowed = []
        for s in allowed_specialists:
            norm = normalize_specialist_persona(s)
            if norm and norm not in normalized_allowed:
                normalized_allowed.append(norm)

    lines: List[str] = ["STUDIO TOOL & PERSONA DIRECTORY (progressive expansion available via `tool_search_query`):"]

    if layer in ("task_lead", "council"):
        lines.append("  Specialist Personas (each persona is BOTH creator and perceptual evaluator/inspector for its domain):")
        for cap in STUDIO_CAPABILITIES:
            if cap.kind != "persona":
                continue
            if normalized_allowed is not None and cap.persona_or_scope not in normalized_allowed:
                continue
            lines.append(f"    - `{cap.persona_or_scope}`: {cap.compact_summary}")
    elif layer == "specialist":
        norm_active = normalize_specialist_persona(active_persona)
        for cap in STUDIO_CAPABILITIES:
            if cap.persona_or_scope == norm_active and cap.kind in ("persona", "subtool"):
                lines.append(f"  - [{cap.title}]: {cap.compact_summary}")

    if layer == "task_lead":
        lines.append("  Board & Epistemic Actions:")
        lines.append(
            "    - `action='delegate'`: Invoke an allowed Specialist Persona to create, modify, OR perceptually inspect/probe artifacts."
        )
        lines.append(
            "    - `action='spawn_tasks'`: Create lock-free `kind='discovery'` investigation subtasks or prerequisite/follow-up build tasks (`wait_for_spawned_tasks=True|False`)."
        )
        lines.append(
            "    - `action='split_task'`: Decompose this task into `2..N` cohesive `split_tasks` (with optional intra-batch `depends_on`); downstream dependents rewire automatically."
        )
        lines.append(
            "    - `self_question` / `memory_query` / `tool_search_query`: Retrieve codebase/Bible chunks, institutional memories, or detailed tool schemas before acting."
        )
    elif layer == "council":
        lines.append("  Council Discovery & Orchestration Actions:")
        lines.append(
            "    - `CouncilDiagnosticProbe`: Dispatch a live read-only inspection/probe to ANY of the 5 Specialist Personas (`visual_artist`, `audio_designer`, `script_writer`, `swe_engineer`, `qa_test_engineer`)."
        )
        lines.append(
            "    - `kind='discovery'` Board Proposals: Propose parallel investigation tasks when architectural, visual, or runtime unknowns should be resolved before or alongside build items."
        )
        lines.append(
            "    - `route_to_council_role`: Dynamically route the next Council turn to another Council specialist when an issue belongs in or intersects their domain."
        )
    else:
        lines.append(
            "  Peer Re-Routing & Cleanup: If this brief belongs to another Specialist Persona (`visual_artist`, `audio_designer`, `script_writer`, `swe_engineer`, `qa_test_engineer`), set `reroute_to_specialist` + `reroute_brief` (and `operation`/`operation_mode`/`verification_mode`='reroute'); use `delete_paths` to remove obsolete/orphaned files."
        )
        lines.append(
            "  Epistemic Queries: Populate `self_question` (codebase/Bible/catalog), `memory_query` (long-term memory), or `tool_search_query` (detailed tool/mode schemas) whenever you need deeper grounding before executing."
        )

    return "\n".join(lines)


def build_tool_capability_chunks(
    allowed_specialists: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Builds granular semantic chunks for `CartridgeSemanticIndex` (`category='tool_capability'`)."""
    normalized_allowed: Optional[set[str]] = None
    if allowed_specialists is not None:
        normalized_allowed = {normalize_specialist_persona(s) for s in allowed_specialists if str(s or "").strip()}

    chunks: List[Dict[str, Any]] = []
    for idx, cap in enumerate(STUDIO_CAPABILITIES):
        if (
            normalized_allowed is not None
            and cap.persona_or_scope not in ("all", "task_lead", "council")
            and cap.persona_or_scope not in normalized_allowed
        ):
            continue
        chunks.append(
            {
                "chunk_id": f"tool_capability::{cap.capability_id}",
                "category": "tool_capability",
                "source_path": "engine/studio/tool_registry.py",
                "symbol_name": cap.capability_id,
                "persona_or_scope": cap.persona_or_scope,
                "start_line": idx + 1,
                "end_line": idx + 1,
                "content": f"### {cap.title} (`{cap.capability_id}`)\nSummary: {cap.compact_summary}\n\n{cap.detailed_guide}",
            }
        )
    return chunks
