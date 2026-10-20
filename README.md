# Autonomous Agentic Game Studio Harness & Observability Platform

An autonomous, requirement-driven multi-agent game studio harness that scans a player's local Steam library, extracts their behavioral **Steam Player DNA**, convenes an **8-Specialist Creative Writers' Room** (including a **Gameplay Systems Engineer**) to author a multimodal **Living Game Bible**, and dynamically orchestrates a 9-agent engineering and art swarm to specify, illustrate, voice-act, code, instrument, playtest, and sign off on an original **Godot 4.3** game from scratch—then continuously evolves the game across episodes based on live SQLite WAL telemetry hypotheses.

---

## Core Architecture

### 1. Steam Library Discovery & DNA Profiler (`engine/steam/`)
- Parses local macOS Steam VDF configurations (`libraryfolders.vdf`, `localconfig.vdf`, `appmanifest_*.acf`) and enriches top-played titles via the Steam Store API.
- Computes genre affinities, continuous pacing preference (`tactical_methodical` to `kinetic_high_mobility`), challenge tolerance, and aesthetic palette mood.

### 2. 8-Specialist Creative Writers' Room & Living Game Bible (`engine/studio/writers_room.py`)
- **Collaborative Multi-Turn Brainstorming (`CreativeWritersRoom`)**:
  - Convenes `CreativeDirector` plus 7 specialists (`GameplayDesigner`, `GameplaySystemsEngineer`, `StoryWorldArchitect`, `CharacterDesigner`, `UXExperienceDesigner`, `VisualArtDirector`, and `AudioAtmosphereDesigner`) at elevated creative temperature to debate, critique, and evolve original game concepts grounded in the player's Steam DNA or an optional Executive Producer seed prompt.
- **Gameplay Systems Engineer & Technical Architecture Blueprint (`MechanicEngineeringSpec`)**:
  - The engineer consults on every pitched mechanic and records a `technical_architecture_blueprint` in the Game Bible: Godot 4 node/collider topology (1:1 visual-to-physical alignment), deterministic state machines that start unsolved at t=0, required sprite/tile/animation assets (so code synthesis never falls back to programmer-art shapes), and falsifiable `unsolved_baseline` / `spatial_boundary` / `positive_causal` test invariants.
  - The blueprint feeds downstream: `CartridgeAssetLibrary.decompose_concept_into_elements` (asset commissioning), `GameConceptDirectorAgent.generate_requirements_spec` (track authoring), and `LeadGameplaySWEAgent` (code synthesis and atomic-unit decomposition).
- **Living Game Bible (`GameBibleSpec`)**:
  - Persisted in SQLite WAL (`studio_game_bibles`) and streamed live over WebSockets (`/ws/swarm`, `/ws/stream`):
    - **Writers' Room Transcript & Concept Evolution Log**: Round-by-round dialogue utterances and major design pivots (`concept_evolution_log`).
    - **First 60 Seconds Hook & Diegetic UI Philosophy**: Explicit playable hook within 60 seconds and a strict ban on giant onboarding text manuals in favor of diegetic in-world cues (`diegetic_ui_philosophy`).
    - **Experience & Pacing Storyboard**: Ordered sequence of playable beats, emotional targets, and visual/audio cues.
    - **16:9 Visual Concept & UI Mockups**: Multimodal 16:9 concept art and HUD layout mockups generated during brainstorming (`assets/mockups/`).
    - **Distinct Character Sheets & Live Voice Auditions**: Distinct cast profiles (`character_sheets`) with individualized vocal timbres and playable Gemini TTS `.wav` voice auditions (`assets/audio/audition_*.wav`).
    - **Cutscene Scripts, Gameplay Pillars & Technical Architecture Blueprint**: Structured narrative cutscenes, core mechanical loops (`gameplay_pillars`), and engineering specs (`technical_architecture_blueprint`).
- **Executive Producer Steering & Greenlight Gate**:
  - Supports interactive mid-session steering (`POST /api/writers-room/steer`), iterative refinement (`POST /api/writers-room/iterate`), and optional **Pause for Executive Greenlight Before Coding** (`POST /api/writers-room/greenlight`).

### 3. Requirement-Driven Engineering & Art Swarm (`engine/studio/`)
- **`StudioOrchestrator`**: Performs live Gemini meta-planning over the player's `SteamPlayerDNA` (or `EvolutionHypothesisReport`) and locked `GameBibleSpec` to assemble a topological execution DAG, iterate requirement-by-requirement (`R1..Rn`), and persist playable verified checkpoints (`cartridges/checkpoints/`).
- **`CreativeDirectorWorldbuilderAgent`**: Authors the `GenesisWorldSpec` blueprint and the structured **`GameRequirementsSpec`** (`requirements_spec.json`) aligned directly with the locked Game Bible and its technical architecture blueprint, decomposing the game into ordered feature tracks (`R1..Rn`), verifiable acceptance criteria, and a multimodal asset manifest.
- **`NarrativeCharacterWriterAgent`**: Translates the Game Bible's character sheets and cutscenes into the cartridge's `NarrativeBible`, branching acyclic dialogue trees, and situational voice barks.
- **`VoiceActingAudioProducerAgent`**: Synthesizes spoken `.wav` dialogue clips and sound cues via Gemini TTS (`assets/audio/`).
- **`VisualArtSpriteDesignerAgent` & `CartridgeAssetLibrary` (`engine/studio/visual.py`, `engine/studio/asset_library.py`)**:
  - **Concept decomposition**: a 16:9 concept mockup (plus the engineer's required-asset list) is decomposed via a flat structured-output schema into modular tiles, props, and animated elements. Characters are conditioned on their portraits and prompted as a single full-body sprite, never a multi-angle turnaround.
  - **Alpha matting**: perimeter-connected flood fill plus `rembg` (`u2net`) salient-object matting, so interior highlights (e.g. eye whites) are never punched out.
  - **Keyframe animation**: character animation sheets are produced by a continuous, topology-preserving 2D UV displacement field (`cv2.remap`) driven by per-motion-profile keyframe tracks, not rigid quadrant slicing.
  - **Tiles**: seamless non-mirrored toroidal tiles (`ensure_full_bleed_tile`) with a Rorschach/mirror-symmetry gate.
  - **Vision QA tiers**: Tier 1 per-asset alpha/tile inspection plates and Tier 2 animation contact sheets with optical-flow checks; failures raise `StudioAgentError` with no silent fallback.
  - Maintains a structured metadata catalog (`assets/asset_library.json`) with strict portrait-vs-world-sprite separation.
- **`LeadGameplaySWEAgent` (`engine/studio/swe.py`)**:
  - **Upfront Production Strategy (`formulate_production_strategy` / `GameProductionStrategySpec`)**: Formulates a blueprint covering environment & level topology, camera & terrain perspective coherence, psychoacoustic audio comfort, required visual/audio assets, modular GDScript architecture, and diegetic HUD layout before level synthesis.
  - **Per-Scene Multimodal Critique & Iterative Refinement (`analyze_scene_and_restrategize` / `execute_scene_strategy_and_refine_loop`)**: Captures live 60-tick Godot 4 viewport screenshots (`playtest_t0.png`, `playtest_tend.png`), compares them against the Game Bible and 16:9 concept mockups (`SceneCritiqueAndStrategySpec`), and dispatches targeted remediation passes.
  - **Atomic TDD (`decompose_track_into_atomic_units`, `synthesize_atomic_godot_test`)**: Decomposes tracks into `AtomicWorkUnit`s with `unsolved_baseline` / `spatial_boundary` / `positive_causal` invariants, runs headless Godot `extends SceneTree` tests via `GodotAtomicTestRunner` (`engine/sim/atomic_runner.py`), and renders a collision-overlay capture (`playtest_collision_overlay.png`) for visual-to-collider alignment checks.
  - **Two-Hop Scoped Plan-Execute-Verify (PEV) Debugging Cycle (`debug_and_patch_pev_cycle`)**: The SWE agent inspects a compact **Cartridge Symbol Index**, requests only relevant code slices, synthesizes minimal surgical patches, and verifies each against the headless Godot 4.3 compiler and runtime simulator.
- **`TelemetryArchitectAgent`**: Inspects the generated game code, designs custom behavioral telemetry probes (`telemetry_manifest.json`), and instruments the cartridge to stream live runtime state (`[TELEMETRY_EVENT]`) into SQLite WAL.
- **Resumable Builds (`rehydrate_resumable_candidate_from_db`)**:
  - Automatically rehydrates interrupted or failed candidate builds from SQLite and existing disk artifacts, skipping already-verified tracks (`R1..Rk`) and resuming at the next pending track or, when all tracks are done, at the quality gates (`POST /api/genesis/trigger` with `resume_current: true`, the header **Run Quality Gates** / **Resume Build** button, or `python run_studio.py --resume-current`).
  - **Note:** the studio server imports its modules once per process. Restart `run_studio.py` after changing engine code, or in-flight sessions keep running the old code.

### 4. 6-Gate Headless QA Playtester & Multi-Agent Final Sign-Off Council (`engine/supervisor/`, `engine/sim/`)
- **Gate A (Syntax, AST & Entrypoint)**: Validates Godot 4.3 headless compiler diagnostics (`verify_godot_cartridge_syntax`), AST security invariants, and Rule 4 Rationale-First contracts.
- **Gate B (Multimodal Asset Integrity)**: Verifies RIFF `.wav` audio headers, `.png` sprite/portrait integrity, and acyclic dialogue graphs.
- **Gate C (Headless Execution, Real GPU Viewport Capture & Scene Audit)**: Executes the cartridge via `godot_harness.gd` / `GodotHeadlessSimulator`, driving synthetic inputs across ticks, capturing `t=0s`, `t=end` and collision-overlay screenshots, auditing the live scene tree, and flagging passive auto-win / unbounded sky-or-void states and failing atomic tests (`tests/test_*.gd`).
- **Gate D (60 FPS Frame Budget)**: Verifies tick execution stays within the 16.6ms frame budget.
- **Gate E (Custom Telemetry Emission)**: Verifies that custom probes declared in `telemetry_manifest.json` actively emit into SQLite WAL during simulation.
- **Gate F (Challenger Gate & 6-Specialist Final Sign-Off Council)**:
  - Independent Gemini structured evaluation (`ChallengerGate`) grounded in AST inspection, runtime telemetry, and scene audits verifying 100% of `requirements_spec.json` acceptance criteria.
  - **Multi-Agent Specialist Sign-Off Council (`run_final_signoff_council`)**: Each creative specialist inspects the rendered candidate build (`playtest_t0.png`, `playtest_tend.png`, `playtest_collision_overlay.png`), camera/terrain perspective alignment, collider alignment, psychoacoustic comfort, and scene-tree audit to issue a domain sign-off (`SpecialistSignOffVerdict`) before atomic promotion to `cartridges/active/`.

### 5. Dual-Workspace Mission Control Web UI & SQLite WAL Persistence (`engine/ui/`, `engine/telemetry/`)
- All studio events (`studio_events`), Living Game Bibles (`studio_game_bibles`), agent task handoff records (`agent_task_records`), requirements (`studio_requirements`), world specs (`studio_world_specs`), and telemetry probes are persisted in **SQLite WAL** (`data/telemetry.db`).
- Real-time dual-workspace dashboard (`http://127.0.0.1:8000`):
  - **📡 Swarm Telemetry & Playable Preview**: Live 9-node execution DAG, Steam Player DNA & Game Blueprint, Dynamic Agent-Designed Telemetry & Hypotheses heatmap, Deliberation Stream, PEV Execution Cycle Cards, Multimodal Voice & Sprite Previews, and unified Code Diff Explorer.
  - **📖 Creative Writers' Room & Living Game Bible**: Live Writers' Room chat transcript, Executive Producer Steering & Greenlight banner, Concept Evolution & Experience Storyboard, 16:9 Visual Mockups gallery with lightbox, Character Sheets & Voice Auditions, Gameplay Pillars, and Specialist Sign-Off Council verdicts.
  - **Header Controls**: Live build status / resume indicator (`⚙️ Building R2 (1/5)...` / `▶️ Resume Build at R2` / `▶️ Tracks Complete (4/4) — Run Quality Gates`), **🚀 New Game from Scratch** modal, and slide-out **Tasks, Requirements & Checkpoints** drawer. Tracks being complete does not mean the quality gates passed.

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
