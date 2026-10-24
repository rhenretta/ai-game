# Autonomous Agentic Game Studio Harness & Observability Platform

An autonomous, requirement-driven multi-agent game studio harness that scans a player's local Steam library, extracts their behavioral **Steam Player DNA**, convenes a **9-Specialist Creative Writers' Room** (including a **Gameplay Systems Engineer** and **Level & Progression Designer**) to author a multimodal **Living Game Bible** and **Continuous Production Board**, and dynamically orchestrates autonomous **Task Lead Strategists** and domain specialists (`visual_artist`, `audio_designer`, `swe_engineer`, `qa_test_engineer`) to specify, illustrate, voice-act, code, instrument, playtest, and sign off on an original **Godot 4.3** game from scratch—then continuously evolves the game across episodes based on live SQLite WAL telemetry hypotheses.

---

## Core Architecture

### 1. Steam Library Discovery & DNA Profiler (`engine/steam/`)
- Parses local macOS Steam VDF configurations (`libraryfolders.vdf`, `localconfig.vdf`, `appmanifest_*.acf`) and enriches top-played titles via the Steam Store API.
- Computes genre affinities, continuous pacing preference (`tactical_methodical` to `kinetic_high_mobility`), challenge tolerance, and aesthetic palette mood.

### 2. 9-Specialist Creative Writers' Room & Living Game Bible (`engine/studio/writers_room.py`)
- **Emergent Concept Genesis & Collaborative Brainstorming (`CreativeWritersRoom`)**:
  - Convenes `CreativeDirector` plus 8 specialists (`GameplayDesigner`, `LevelAndProgressionDesigner`, `GameplaySystemsEngineer`, `StoryWorldArchitect`, `CharacterDesigner`, `UXExperienceDesigner`, `VisualArtDirector`, and `AudioAtmosphereDesigner`).
  - Generates orthogonal creative provocations (`CreativeProvocationSet`), collects independent domain fragments (`ConceptFragment`), collides them into candidate premises (`ConceptCollisionProposal`), verifies novelty against prior studio games (`NoveltyJudgement`), and evolves the selected concept across multi-turn room debates.
- **Whole-Game Design Blueprint (`GameBibleSpec`)**:
  - Authors a cohesive whole-game design rather than a single static room:
    - `LevelGenerationStrategy`: Bespoke `generation_approach`, `world_structure`, `runtime_construction_plan`, `level_archetypes`, and `generation_rules`.
    - `CoreLoopSpec`, `EngagementPlan`, `OnboardingPlan`, and `GoalHierarchy` across multiple player horizons.
    - `MechanicEngineeringSpec` (`technical_architecture_blueprint`): Godot 4 node/collider topology (1:1 visual-to-physical alignment), deterministic state machines that start unsolved at `t=0`, required sprite/tile/animation assets, and falsifiable test invariants.
- **Continuous Production Board & Cross-Functional Council Review (`engine/studio/production_board.py`)**:
  - While designing, specialists raise dependency-linked `WorkItemProposal` tasks (`visual_asset`, `animation`, `audio_asset`, `level_map`, `mechanic_code`, `ui`, `integration`) onto `ProductionBoard` (`backlog.json`).
  - `BoardScheduler` dispatches ready items to parallel worker threads while the Writers' Room continues.
  - Every completed item returns visual/waveform proof artifacts inspected by a cross-functional Specialist Council panel (`_reviewers_for`), requiring unanimous approval (`ReviewBatch`, `ReviewDecision`, `CapDecision`).
- **Live Executive Producer Steering & Greenlight Gate**:
  - Supports live mid-session and mid-production steering (`POST /api/studio/greenlight`, `POST /api/writers-room/steer`) persisted via `.pending_steering.json` so concurrent worker turns never overwrite Executive Producer directives, immediately triggering `CreativeDirector` and `GameplaySystemsEngineer` pivot turns to adapt the Bible and Production Board.

### 3. Iterative Task Lead Strategist, Autonomous Specialists & Asset/Audio Engines (`engine/studio/`)
- **Iterative Task Lead Strategist (`engine/studio/workers.py`)**:
  - Every `ProductionBoard` work item is managed by `run_task_lead` (`TaskLeadDecision`), which runs an iterative `Observe -> Strategize -> Delegate -> Inspect -> Submit` loop across up to 6 turns per revision.
  - Rather than choosing from pre-baked option menus, the Task Lead invents a bespoke strategy and briefs four autonomous domain specialists via open-ended natural-language directives:
    - `visual_artist`: Plans (`AssetRequirement`) and fabricates sprites, animation sheets, tiles, portraits, backdrops, and diegetic UI elements, returning Tier 1/2 inspection plates (`plate_*.png`, `anim_sheet_*.png`).
    - `audio_designer`: Synthesizes character voice barks via Gemini TTS or multi-layer parametric DSP sound effects/ambient loops (`engine/studio/sfx_synth.py`), returning rendered waveform proofs (`waveform_<item_id>_rev<N>.png`).
    - `swe_engineer`: Formulates an upfront `CodeImplementationPlan` before authoring/updating GDScript modules (`main.gd`, `scenes.gd`, `mechanics.gd`, `entities.gd`, `ui_hud.gd`), or generates machine-readable `LevelLayoutSpec` data (`level_map_<item_id>_rev<N>.png`), gated by headless Godot syntax & runtime smoke checks.
    - `qa_test_engineer`: Designs a scripted `FilmstripPlan` (`capture_ticks`, `input_segments`, `setup_calls` via `--setup-script`, and `state_assertions`) and runs `capture_filmstrip` (`engine/studio/proofs.py`) to record a labelled multi-tick Godot gameplay proof contact sheet (`filmstrip_<item_id>_rev<N>.png`, with external `[QA]` slate margins and 1px viewport borders) and counterfactual zero-input verification (`verify_counterfactual=True`).
  - **Cooperative Non-Blocking `code_lock` Yielding & Parallel Asset Prioritization**: When `code_lock` is held by another worker, `ProductionBoard.claim_ready()` prioritizes parallel asset tasks (`visual_asset`, `animation`, `audio_asset`), and any Task Lead turn requiring `code_lock` (`swe_engineer`, `qa_test_engineer`) persists its `pending_decision`, `turns`, and `collected_proofs` to `lead_trace_rev{N}.json` and raises `WorkerYielded` (`yield_item`) so the worker thread immediately moves on to other ready tasks and resumes at the exact paused turn once `code_lock` frees.
  - **Multimodal Task Lead Proof Inspection & Council Contact-Sheet Guidance**: Before choosing `submit_to_council`, the Task Lead Strategist multimodally inspects the 16:9 Concept Art Mockup and captured proof images (`_build_task_lead_multimodal_parts`), and Council reviewers receive explicit QA contact-sheet framing guidance distinguishing external `[QA]` telemetry margin strips from in-game UI.
- **Per-Game Rendering Strategy & Modern 2D Lighting (`engine/studio/rendering.py`)**:
  - Formulates a bespoke `RenderingStrategySpec` per game (`lighting_plan`, `material_treatment`, `custom_shaders`, `particle_systems`, `post_processing`, `camera_feel`) and provides Godot 4 `WorldEnvironment` glow, `DirectionalLight2D`, `PointLight2D`, and `GPUParticles2D` helpers.
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
