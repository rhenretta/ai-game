# Autonomous Agentic Game Studio Harness & Observability Platform

An autonomous, requirement-driven multi-agent game studio harness that scans a player's local Steam library, extracts their behavioral **Steam Player DNA**, convenes a **7-Specialist Creative Writers' Room** to author a multimodal **Living Game Bible**, and dynamically orchestrates a 9-agent engineering and art swarm to specify, illustrate, voice-act, code, instrument, playtest, and sign off on an original **Godot 4.3** game from scratch—then continuously evolves the game across episodes based on live SQLite WAL telemetry hypotheses.

---

## Core Architecture

### 1. Steam Library Discovery & DNA Profiler (`engine/steam/`)
- Parses local macOS Steam VDF configurations (`libraryfolders.vdf`, `localconfig.vdf`, `appmanifest_*.acf`) and enriches top-played titles via the Steam Store API.
- Computes genre affinities, continuous pacing preference (`tactical_methodical` to `kinetic_high_mobility`), challenge tolerance, and aesthetic palette mood.

### 2. 7-Specialist Creative Writers' Room & Living Game Bible (`engine/studio/writers_room.py`)
- **Collaborative Multi-Turn Brainstorming (`CreativeWritersRoomOrchestrator`)**:
  - Convenes 7 specialized creative personas (`CreativeDirector`, `GameplayDesigner`, `StoryWorldArchitect`, `CharacterDesigner`, `UXExperienceDesigner`, `VisualArtDirector`, and `AudioAtmosphereDesigner`) at elevated creative temperature (`1.0`) to debate, critique, and evolve original game concepts grounded in the player's Steam DNA or an optional Executive Producer seed prompt.
- **Living Game Bible (`LivingGameBible`)**:
  - Persisted in SQLite WAL (`studio_game_bibles`) and streamed live over WebSockets (`/ws/swarm`, `/ws/stream`):
    - **Writers' Room Transcript & Concept Evolution Log**: Round-by-round dialogue utterances and major design pivots (`concept_evolution_log`).
    - **First 60 Seconds Hook & Diegetic UI Philosophy**: Explicit playable hook within 60 seconds and a strict ban on giant onboarding text manuals in favor of diegetic in-world cues (`diegetic_ui_philosophy`).
    - **Experience & Pacing Storyboard**: Ordered sequence of playable beats, emotional targets, and visual/audio cues.
    - **16:9 Visual Concept & UI Mockups**: Multimodal 16:9 concept art and HUD layout mockups generated during brainstorming (`assets/mockups/`).
    - **Distinct Character Sheets & Live Voice Auditions**: Distinct cast profiles (`character_sheets`) with individualized vocal timbres and playable Gemini TTS `.wav` voice auditions (`assets/audio/audition_*.wav`).
    - **Cutscene Scripts & Gameplay Pillars**: Structured narrative cutscenes and core mechanical loops (`gameplay_pillars`).
- **Executive Producer Steering & Greenlight Gate**:
  - Supports interactive mid-session steering (`POST /api/writers-room/steer`), iterative refinement (`POST /api/writers-room/iterate`), and optional **Pause for Executive Greenlight Before Coding** (`POST /api/writers-room/greenlight`).

### 3. Requirement-Driven Engineering & Art Swarm (`engine/studio/`)
- **`StudioOrchestrator`**: Performs live Gemini meta-planning over the player's `SteamPlayerDNA` (or `EvolutionHypothesisReport`) and locked `LivingGameBible` to assemble a topological execution DAG, iterate requirement-by-requirement (`R1..Rn`), and persist playable verified checkpoints (`cartridges/checkpoints/`).
- **`CreativeDirectorWorldbuilderAgent`**: Authors the `GenesisWorldSpec` blueprint and the structured **`GameRequirementsSpec`** (`requirements_spec.json`) aligned directly with the locked `LivingGameBible`, decomposing the game into ordered feature tracks (`R1..Rn`), verifiable acceptance criteria, and a multimodal asset manifest.
- **`NarrativeCharacterWriterAgent`**: Translates the Game Bible's character sheets and cutscenes into the cartridge's `NarrativeBible`, branching acyclic dialogue trees, and situational voice barks.
- **`VoiceActingAudioProducerAgent`**: Synthesizes spoken `.wav` dialogue clips and sound cues via Gemini TTS (`assets/audio/`).
- **`VisualArtSpriteDesignerAgent` & `CartridgeAssetLibrary` (`engine/studio/visual.py`, `engine/studio/asset_library.py`)**:
  - Synthesizes custom pixel-art sprites, 2x2 multi-state animation sheets, articulated multi-part rigs, character portraits, and 90-degree orthographic top-down seamless terrain/wall tiles (`ensure_full_bleed_tile` with toroidal border cross-fading and corner-defect validation) via Gemini multimodal image generation (`assets/sprites/`, `assets/portraits/`).
  - Maintains a structured, concept-grounded metadata catalog (`asset_catalog.json`) with strict portrait-vs-world-sprite separation and zero hardcoded GDScript string templates.
- **`LeadGameplaySWEAgent` (`engine/studio/swe.py`)**:
  - **Upfront Production Strategy (`formulate_production_strategy` / `GameProductionStrategySpec`)**: Formulates a comprehensive blueprint covering environment & level topology, **camera & terrain perspective coherence** (`camera_and_perspective_coherence_strategy`), **psychoacoustic audio comfort** (`psychoacoustic_audio_strategy`), required visual/audio assets, modular GDScript architecture, and diegetic HUD layout before level synthesis.
  - **Per-Scene Multimodal Critique & Iterative Refinement (`analyze_scene_and_restrategize` / `execute_scene_strategy_and_refine_loop`)**: Captures live 60-tick Godot 4 GPU viewport screenshots (`playtest_t0.png`, `playtest_tend.png`), compares the rendered scene side-by-side against the Living Game Bible and 16:9 Concept Mockups (`SceneCritiqueAndStrategySpec`), audits 3/4 top-down plateau perspective alignment, multi-pass fluid canal continuity, and non-looping low/mid-register acoustic comfort, and automatically dispatches targeted remediation passes.
  - **Two-Hop Scoped Plan-Execute-Verify (PEV) Debugging Cycle (`debug_and_patch_pev_cycle`)**: Eliminates naive full-file code dumps during debugging and refinement. Instead, the SWE agent inspects a compact **Cartridge Symbol Index** (`_build_cartridge_symbol_index`), formulates a `DebugResearchPlan` requesting only relevant class/method or line-window slices (`_extract_code_slice`), synthesizes a minimal `SurgicalPatchPlan`, and verifies every patch against the headless Godot 4.3 compiler and runtime simulator.
- **`TelemetryArchitectAgent`**: Inspects the generated game code, designs custom behavioral telemetry probes (`telemetry_manifest.json`), and instruments the cartridge to stream live runtime state (`[TELEMETRY_EVENT]`) into SQLite WAL.
- **Resumable Builds (`rehydrate_resumable_candidate_from_db`)**:
  - Automatically rehydrates interrupted or failed candidate builds from SQLite (`studio_game_bibles`, `studio_world_specs`, `studio_requirements`, and existing disk artifacts), skipping already-verified tracks (`R1..Rk`) and resuming directly at the next pending or rejected requirement track (`POST /api/genesis/trigger` with `resume_current: true` or `python run_studio.py --resume-current`).

### 4. 6-Gate Headless QA Playtester & Multi-Agent Final Sign-Off Council (`engine/supervisor/`, `engine/sim/`)
- **Gate A (Syntax, AST & Entrypoint)**: Validates Godot 4.3 headless compiler diagnostics (`verify_godot_cartridge_syntax`), AST security invariants, and Rule 4 Rationale-First contracts.
- **Gate B (Multimodal Asset Integrity)**: Verifies RIFF `.wav` audio headers, `.png` sprite/portrait integrity, and acyclic dialogue graphs.
- **Gate C (Headless Execution, Real GPU Viewport Capture & Scene Audit)**: Executes the cartridge via `godot_harness.gd` / `GodotHeadlessSimulator` using real GPU rendering (`--rendering-driver opengl3` with minimized window capture) and strict 2D canvas validation (`_validate_rendered_viewport_image`), driving synthetic inputs across ticks, capturing `t=0s` (`playtest_t0.png`) and `t=end` (`playtest_tend.png`) screenshots, and auditing the live scene tree (`audio_players_found`, `audio_streams_loaded`, `center_occluding_panels`, camera & visual canvas nodes).
- **Gate D (60 FPS Frame Budget)**: Verifies tick execution stays within the 16.6ms frame budget.
- **Gate E (Custom Telemetry Emission)**: Verifies that custom probes declared in `telemetry_manifest.json` actively emit into SQLite WAL during simulation.
- **Gate F (Challenger Gate & 6-Specialist Final Sign-Off Council)**:
  - Independent Gemini structured evaluation (`ChallengerGate`) grounded in AST inspection, runtime telemetry, and scene audits verifying 100% of `requirements_spec.json` acceptance criteria.
  - **Multi-Agent Specialist Sign-Off Council (`run_final_signoff_council`)**: Each creative specialist who authored the Living Game Bible inspects the rendered candidate build (`playtest_t0.png`, `playtest_tend.png`), camera/terrain perspective alignment, psychoacoustic comfort (zero continuous high-pitched sine/square wave drones), and scene tree audit to issue a domain sign-off (`SpecialistSignOffVerdict`) before atomic promotion to `cartridges/active/`.

### 5. Dual-Workspace Mission Control Web UI & SQLite WAL Persistence (`engine/ui/`, `engine/telemetry/`)
- All studio events (`studio_events`), Living Game Bibles (`studio_game_bibles`), agent task handoff records (`agent_task_records`), requirements (`studio_requirements`), world specs (`studio_world_specs`), and telemetry probes are persisted in **SQLite WAL** (`data/telemetry.db`).
- Real-time dual-workspace dashboard (`http://127.0.0.1:8000`):
  - **📡 Swarm Telemetry & Playable Preview**: Live 9-node execution DAG, Steam Player DNA & Game Blueprint, Dynamic Agent-Designed Telemetry & Hypotheses heatmap, Deliberation Stream, PEV Execution Cycle Cards, Multimodal Voice & Sprite Previews, and unified Code Diff Explorer.
  - **📖 Creative Writers' Room & Living Game Bible**: Live Writers' Room chat transcript, Executive Producer Steering & Greenlight banner, Concept Evolution & Experience Storyboard, 16:9 Visual Mockups gallery with lightbox, Character Sheets & Voice Auditions, Gameplay Pillars, and Specialist Sign-Off Council verdicts.
  - **Header Controls**: Live build status / resume indicator (`⚙️ Building R2 (1/5)...` / `▶️ Resume Build at R2`), **🚀 New Game from Scratch** modal, and slide-out **Tasks, Requirements & Checkpoints** drawer.

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
