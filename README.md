# Autonomous Agentic Game Studio Harness & Observability Platform

An autonomous, requirement-driven multi-agent game studio harness that scans a player's local Steam library, extracts their behavioral **Steam Player DNA**, convenes a **9-Specialist Creative Writers' Room** (including a **Gameplay Systems Engineer** and **Level & Progression Designer**) to author a multimodal **Living Game Bible** and **Continuous Production Board**, and dynamically orchestrates autonomous **Task Lead Strategists** and domain specialists (`visual_artist`, `audio_designer`, `swe_engineer`, `qa_test_engineer`) to specify, illustrate, voice-act, code, instrument, playtest, and sign off on an original **Godot 4.3** game from scratch—then continuously evolves the game across episodes based on live SQLite WAL telemetry hypotheses.

---

## Core Architecture

### 1. Steam Library Discovery & DNA Profiler (`engine/steam/`)
- Parses local macOS Steam VDF configurations (`libraryfolders.vdf`, `localconfig.vdf`, `appmanifest_*.acf`) and enriches top-played titles via the Steam Store API.
- Computes genre affinities, continuous pacing preference (`tactical_methodical` to `kinetic_high_mobility`), challenge tolerance, and aesthetic palette mood.

### 2. Hierarchical Creative Writers' Room, Iterative Slice Roadmap & Living Game Bible (`engine/studio/writers_room.py`)
- **Emergent Concept Genesis & Hierarchical Specialist Roster (`CreativeWritersRoom`)**:
  - Convenes 5 **Core Council Leads** (`CreativeDirector`, `GameplayDesigner`, `VisualArtDirector`, `StoryWorldArchitect`, `TechAndQAArchitect`) and 6 **Specialized Sub-Agents** (`LevelAndProgressionDesigner`, `GameplaySystemsEngineer`, `CharacterDesigner`, `ScriptwriterAgent`, `UXExperienceDesigner`, `AudioAtmosphereDesigner`).
  - Generates orthogonal creative provocations (`CreativeProvocationSet`), collects independent domain fragments (`ConceptFragment`), collides them into candidate premises (`ConceptCollisionProposal`), verifies novelty against prior studio games (`NoveltyJudgement`), and ensures all core gameplay/level/systems/art/UX/audio specialists contribute (`_missing_required_board_specialists`) before locking a full Production Board build.
- **Upfront 16:9 Concept Art Mockup & Iterative Vertical Slices (`GameBibleSpec`, `GameSliceSpec`)**:
  - Synthesizes an upfront 16:9 Concept Art Mockup establishing the game's visual target, camera/spatial paradigm, and color palette (`visual_mockups`), while deferring character portraits and voice barks to just-in-time slice tasks.
  - Plans an iterative `slice_roadmap` (`Slice 0` vertical slice onward) and supports post-drain iterative expansion (`run_slice_expansion`, `POST /api/studio/next-slice`).
  - Authors a cohesive whole-game design blueprint:
    - `LevelGenerationStrategy`: Bespoke `generation_approach`, `world_structure`, `runtime_construction_plan`, `level_archetypes`, and `generation_rules`, validated against `LevelLayoutSpec` (`validate_level_layout`) so multi-archetype worlds span multiple connected regions and viewports.
    - `CoreLoopSpec`, `EngagementPlan`, `OnboardingPlan`, and `GoalHierarchy` across multiple player horizons.
    - `MechanicEngineeringSpec` (`technical_architecture_blueprint`): Godot 4 node/collider topology, deterministic state machines, required sprite/tile/animation assets, and falsifiable test invariants.
- **Continuous Production Board, Dynamic Task Spawning, Council Escalation & Whole-Game Slice Assembly Check (`engine/studio/production_board.py`, `engine/studio/writers_room.py`)**:
  - Specialists and Task Leads raise and dynamically spawn (`action="spawn_tasks"`, `spawn_tasks_from_item`) dependency-linked `WorkItemProposal` tasks (`visual_asset`, `animation`, `audio_asset`, `level_map`, `mechanic_code`, `ui`, `integration`) onto `ProductionBoard` (`backlog.json`).
  - Every completed item returns visual/waveform/script proof artifacts inspected by a cross-functional Specialist Council panel (`_reviewers_for`) in a sequential **Conversational Proof Review Roundtable** (`_run_review_round`, `ReviewBatch.spoken_dialogue`, `ReviewDecision.validation_criteria_for_my_role`), where each specialist determines from first principles what must be validated for their discipline (including human-scale responsiveness over time and subsystem harmony), converses with fellow panelists on the proof filmstrip, and requires unanimous approval (`ReviewBatch`, `ReviewDecision`, `CapDecision`) while streaming the review dialogue and inline filmstrip cards to the Writers' Room transcript.
  - Items that fail worker execution twice in a live build automatically escalate to Council `needs_decision` (`escalate_failed_to_decision`) with the exact worker failure trace so the Council can `rewrite_spec` or `cancel` and re-point dependents.
  - **Automated Slice Playtest Script & Council Playtest Roundtable (`_run_slice_assembly_check`, `SlicePlaytestRoundtableTurn`, `SliceAssemblyDecision`)**: Whenever the active Production Board drains, `QATestEngineer` authors and executes a dedicated **Slice Playtest Script** (`res://tests/test_slice_playtest_<slice_id>_pass<N>.tscn` + `.gd`) that instantiates the live game entrypoint (`res://main.tscn`), stages the game state at the active slice (`slice_index` / `slice_id`), attaches test-only diagnostic overlays and telemetry (`player_pos`, velocities, goal state), and scripts real player `input_segments` to play through at least the first goal of the current slice (`filmstrip_slice_assembly_check_pass*.png`). The Council (`GameplayDesigner`, `UXExperienceDesigner`, `VisualArtDirector`, and Chair `GameplaySystemsEngineer`) then holds a multi-perspective **Slice Playtest Roundtable** in the Writers' Room evaluating entrypoint reachability (`compute_entrypoint_reachability`), human-scale control responsiveness, HUD clarity (with zero production debug overlays), and fun—spawning fix/integration tasks (`is_slice_assembled=False`) if any gap is found, and rendering the playtest filmstrip and conversation inline in the Studio UI.
- **Live Executive Producer Steering, Auto-Wake & Greenlight Gate**:
  - Supports live mid-session, mid-production, and post-drain steering (`POST /api/studio/greenlight`, `POST /api/writers-room/steer`) persisted via `.pending_steering.json` and `game_bible.json`. Injecting steering while the studio is idle automatically wakes `StudioDaemon(resume_current=True)` in the background and runs code-grounded `CreativeDirector` and `GameplaySystemsEngineer` pivot turns (`_execute_steering_pivot_turns` with `ContextCuratorAgent` entrypoint/script chunks) that raise concrete `backlog_additions` fix tasks on the Production Board.

### 3. Semantic Context Curator, Iterative Task Lead Strategist & Asset/Audio Engines (`engine/studio/`)
- **Semantic Context Curator, Entrypoint Topology, Self-Inquiry Memory & Prefix-Cache-Aligned Prompts (`engine/studio/context_curator.py`)**:
  - `CartridgeSemanticIndex` extracts granular `ContextChunk` entries across concept art mockups, catalog sprites (`hframes`, `vframes`, `tex_size`), Council utterances, rendering treatments, Production Board items, build execution traces (`board:trace:<item_id>` from `lead_trace_rev*.json` with turn deliberations, `code_diffs`, and `.telemetry.json` per-frame coordinate trajectories), GDScript modules (`const` declarations, `KEY_*` input bindings, `collision_layer`/`CanvasModulate` semantics, and a 3-hop bidirectional import/reference graph), and whole-game entrypoint reachability (`code:entrypoint_topology` via `compute_entrypoint_reachability`, walking transitive `preload`/`load`/`ExtResource` references from `run/main_scene` to surface unmounted production modules), caching Gemini embeddings by SHA-256 in `artifacts/context_index.json`.
  - `ContextCuratorAgent` inspects multimodal proof images alongside candidate chunks to curate strictly neutral, factual, role-tailored context packs (`artifacts/work_items/<item_id>/context_pack_<role>.json` with `context_overview` and `UNVERIFIED_WORKER_CLAIMS` labeling) strictly from LLM-selected `selected_chunk_ids`, automatically refreshes packs mid-revision when traces or modules change (`intra_rev_sig`), and supports iterative agent self-inquiry (`self_question` -> `research_question_context` -> `self_question_answer` persisted in `self_inquiry_<role>.json`).
  - All high-frequency prompts (`CreativeWritersRoom`, `plan_task_lead_turn`, `plan_filmstrip`, `LeadGameplaySWEAgent`) place static shared context at byte 0 and dynamic per-turn/retry state at the tail to maximize Gemini implicit prefix caching, tracked live via `TOKEN_USAGE_TRACKER` (`/api/status` and `/api/studio/status`).
- **Iterative Task Lead Strategist (`engine/studio/workers.py`)**:
  - Every `ProductionBoard` work item (`visual_asset`, `animation`, `audio_asset`, `level_map`, `mechanic_code`, `ui`, `integration`, etc.) runs through `run_task_lead` (`TaskLeadDecision`) from Revision 0 onward in an iterative `Observe -> Strategize -> Delegate -> Inspect -> Submit` loop across up to 6 turns per revision, formulating both a creation strategy (`current_strategy`) and an explicit proof-generation strategy (`proof_strategy`).
  - The Task Lead briefs five autonomous domain specialists via open-ended natural-language directives, visually inspects captured proof plates/filmstrips (`_build_task_lead_multimodal_parts`) before submitting, and can dynamically spawn prerequisite board tasks (`action="spawn_tasks"`):
    - `visual_artist`: Plans (`AssetRequirement`) and fabricates single sprites, multi-cell grids/state sheets/atlases (`grid_cols`, `grid_rows`, `cell_prompts`), biomechanical animation sheets, articulated rigs, seamless `tile_surface` textures, portraits, backdrops, and diegetic UI elements, returning inspection plates (`plate_*.png`, `anim_sheet_*.png`).
    - `audio_designer`: Synthesizes character voice barks via Gemini TTS or multi-layer parametric DSP sound effects/ambient loops (`engine/studio/sfx_synth.py`), returning rendered waveform proofs (`waveform_<item_id>_rev<N>.png`).
    - `script_writer`: Crafts scene scripts (`cutscenes`), branching `dialogue_trees`, contextual `dialogue_barks`, and character sheets into `narrative_bible.json` / `narrative.json`, returning rendered script table proof sheets (`script_sheet_<item_id>_rev<N>.png`).
    - `swe_engineer`: Authors/updates modular GDScript files (`main.gd`, `scenes.gd`, `mechanics.gd`, `entities.gd`, `ui_hud.gd`, and decomposed submodules under `scripts/*.gd` via `CodeImplementationPlan.module_path`) using full target-module source and full sibling/curated `.gd` implementation blocks (`_summarize_non_target_gd_module`) to prevent cross-module force/input conflicts, or generates machine-readable `LevelLayoutSpec` data (`level_map_<item_id>_rev<N>.png`), gated by headless Godot syntax & runtime smoke checks and recorded with unified diffs (`code_diffs`).
    - `qa_test_engineer`: Designs a scripted `FilmstripPlan` (`capture_ticks`, `input_segments`, `setup_calls` via `--setup-script`, `state_assertions`, and isolated test scenes via `test_scene_gd_content` written to `res://tests/test_<item_id>.tscn` + `res://tests/test_<item_id>.gd` with rich test-only diagnostic overlays while keeping production `main.gd` and `ui_hud.gd` free of debug overlays and `func test_*()` helpers) and runs `capture_filmstrip` (`engine/studio/proofs.py`) at a deterministic `--fixed-fps 60 --headless` timestep (auto-detecting player coordinates via `godot_harness.gd` when `get_state_snapshot()` omits `player_pos`) to record a labelled multi-tick Godot gameplay proof contact sheet (`filmstrip_<item_id>_rev<N>.png`) and per-frame telemetry sidecar (`filmstrip_<item_id>_rev<N>.telemetry.json`).
  - **Granular Per-File Locking (`FileLockTable`) & Cooperative Non-Blocking Yielding (`WorkerYielded`)**: Concurrent `swe_engineer` tasks acquire per-file locks (`FileLockTable.acquire_many` in sorted lexical order) on only the target `.gd` modules they touch (`waiting_on_locks`), enabling parallel GDScript engineering across disjoint files. When a required file lock or `code_lock` is held by another worker, the Task Lead persists its `pending_decision`, `turns`, and `collected_proofs` to `lead_trace_rev{N}.json` and raises `WorkerYielded` (`yield_item`) so the worker thread immediately moves on to other ready tasks and resumes at the exact paused turn once the lock frees.
- **Hybrid Procedural + Generated Visual Craft & Modern 2D Lighting (`engine/studio/rendering.py`)**:
  - Formulates a bespoke `RenderingStrategySpec` per game and provides `StudioRigHelper` (`create_tiled_polygon` for tiling `tile_surface` PNGs across procedural `Polygon2D` contours, `create_rig_from_manifest`, `load_sprite_sheet`) plus Godot 4 `WorldEnvironment` glow, `DirectionalLight2D`, `PointLight2D`, and `GPUParticles2D` helpers.
- **Parametric Multi-Layer DSP SFX Synthesizer (`engine/studio/sfx_synth.py`)**:
  - Synthesizes procedural 24kHz PCM `.wav` sound effects and seamless ambient loops (`SfxSynthesisSpec`, `SfxLayerSpec`) combining oscillator sweeps, filtered noise, FM synthesis, Karplus-Strong plucked strings, and modal resonator banks.
- **`VisualArtSpriteDesignerAgent` & `CartridgeAssetLibrary` (`engine/studio/visual.py`, `engine/studio/asset_library.py`)**:
  - Single-subject base pose prompting (separating single neutral base sprite generation from multi-row `animation_states` keyframe articulation), multi-tone perimeter flood-fill alpha matting (stripping both chroma backdrops and 2-tone alternating transparency checkerboards) + `rembg` (`u2net`) + morphological single-subject isolation (`extract_clean_alpha_sprite`), continuous UV-displacement keyframe animation (`cv2.remap`), non-mirrored toroidal tiles (`ensure_full_bleed_tile`), recursive indexing across `assets/sprites/`, `assets/tiles/`, `assets/backgrounds/`, and `assets/ui/`, and Tier 1 / Tier 2 vision QA gates.
- **`LeadGameplaySWEAgent` (`engine/studio/swe.py`)**:
  - Upfront production strategy (`formulate_production_strategy`), machine-readable level layout formulation (`formulate_level_layout`), per-scene multimodal critique (`analyze_scene_and_restrategize`), atomic Godot TDD (`GodotAtomicTestRunner`), and 2-hop scoped Plan-Execute-Verify surgical patching (`debug_and_patch_pev_cycle`).
- **`TelemetryArchitectAgent` (`engine/studio/telemetry_architect.py`)**:
  - Designs custom behavioral telemetry probes (`telemetry_manifest.json`) and instruments the cartridge to stream live runtime state (`[TELEMETRY_EVENT]`) into SQLite WAL.
- **Resumable Builds (`rehydrate_resumable_candidate_from_db`)**:
  - Automatically rehydrates interrupted or failed candidate builds from SQLite and existing disk artifacts (`backlog.json`, `game_bible.json`, `requirements_spec.json`), resuming pending/revised Production Board items or running final quality gates (`POST /api/genesis/trigger` with `resume_current: true` or `python run_studio.py --resume-current`).

### 4. 6-Gate Headless QA Playtester & Multi-Agent Final Sign-Off Council (`engine/supervisor/`, `engine/sim/`)
- **Gate A (Syntax, AST & Entrypoint)**: Validates Godot 4.3 headless compiler diagnostics (`verify_godot_cartridge_syntax`), AST security invariants, and Rule 4 Rationale-First contracts.
- **Gate B (Multimodal Asset Integrity)**: Verifies RIFF `.wav` audio headers, `.png` sprite/portrait integrity, and acyclic dialogue graphs.
- **Gate C (Headless Execution, Real GPU Viewport Capture & Scene Audit)**: Executes the cartridge via `godot_harness.gd` / `GodotHeadlessSimulator`, supporting scripted input ranges (`--input-script`) and QA scenario setup calls (`--setup-script`), capturing `t=0s`, `t=end`, multi-tick filmstrips, and collision-overlay screenshots, auditing the live scene tree, and flagging passive auto-win or unbounded sky-or-void states.
- **Gate D (60 FPS Frame Budget)**: Verifies tick execution stays within the 16.6ms frame budget.
- **Gate E (Custom Telemetry Emission)**: Verifies that custom probes declared in `telemetry_manifest.json` actively emit into SQLite WAL during simulation.
- **Gate F (Challenger Gate & Multi-Agent Final Sign-Off Council)**:
  - Independent Gemini structured evaluation (`ChallengerGate`) grounded in AST inspection, runtime telemetry, and scene audits verifying 100% of acceptance criteria.
  - **Multi-Agent Specialist Sign-Off Council (`run_final_signoff_council`)**: Each creative specialist inspects the rendered candidate build (`playtest_t0.png`, `playtest_tend.png`, `playtest_collision_overlay.png`) to issue a domain sign-off (`SpecialistSignOffVerdict`) before atomic promotion to `cartridges/active/`.

### 5. Dual-Workspace Mission Control Web UI & SQLite WAL Persistence (`engine/ui/`, `engine/telemetry/`)
- All studio events (`studio_events`), Living Game Bibles (`studio_game_bibles`), agent task handoff records (`agent_task_records`), requirements (`studio_requirements`), world specs (`studio_world_specs`), and telemetry probes are persisted in **SQLite WAL** (`data/telemetry.db`).
- Real-time dual-workspace dashboard (`http://127.0.0.1:8000`):
  - **📖 Creative Writers' Room, Live Production Board & Built Renders Showcase**:
    - **🏗️ Live Production Board & Proofs**: Real-time kanban/filter view of all `ProductionBoard` items (`in_progress`, `awaiting_review`, `changes_requested`, `approved`, `ready`, `blocked`), displaying clickable proof filmstrips, asset inspection plates, audio players, worker notes, and Council review verdicts.
    - **🖼️ Built Assets, Renders & Code Showcase**: Unified gallery of generated sprites, animation contact sheets, audio tracks, and inspectable `.gd` source modules for any selected game cartridge.
    - **💬 Live Writers' Room Transcript & Executive Steering**: Incremental transcript streaming, **💬 Steer Room** modal, Concept Evolution & Experience Storyboard, 16:9 Visual Mockups gallery, Character Sheets & Voice Auditions, Gameplay Pillars & Technical Architecture Blueprint, and Final Sign-Off Council verdicts.
  - **📡 Swarm Telemetry & Playable Preview**: Live 9-node execution DAG, Steam Player DNA & Game Blueprint, Dynamic Agent-Designed Telemetry & Hypotheses heatmap, Deliberation Stream, PEV Execution Cycle Cards, and unified Code Diff Explorer.
  - **Studio Live Pulse Ribbon & Header Controls**: Real-time weighted progress bar, per-specialist turn headlines, multi-game cartridge switcher, authoritative Pause/Resume Build toggle, **🚀 New Game from Scratch** modal, and slide-out **Tasks, Requirements & Checkpoints** drawer.

---

## Quickstart

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 2. Launch the Autonomous Studio ("Mission Control")

Start the Studio observability server and background daemon:

```bash
python run_studio.py --port 8000
```

Or resume an interrupted game build directly from its persisted Living Game Bible and requirement tracks:

```bash
python run_studio.py --port 8000 --resume-current
```

Open **`http://127.0.0.1:8000`** in your browser to watch the Creative Writers' Room brainstorm the Living Game Bible, steer or greenlight the concept, and observe the 9-agent engineering & art swarm implement and verify each requirement track (`R1..Rn`).

### 3. Play the Generated Game (or Any Verified Checkpoint)

Launch the promoted active cartridge natively in Godot 4.3 at 60 FPS:

```bash
python run_game.py
```

Or launch and compare incremental requirement checkpoints (`R1`, `R2`, ..., `R5`):

```bash
python run_game.py --list-checkpoints
python run_game.py --latest-checkpoint
python run_game.py --checkpoint candidate_genesis_db8d94_R1
```

### 4. Run Typecheck & Hermetic Test Suite

All unit and integration tests run hermetically (`tests/conftest.py` automatically redirects `TELEMETRY_DB_PATH` to an isolated temporary SQLite database and blocks unmocked external LLM or HTTP network calls):

```bash
pytest
```
