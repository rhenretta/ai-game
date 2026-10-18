# Autonomous Agentic Game Studio Harness & Observability Platform

An autonomous, requirement-driven multi-agent game studio harness that scans a player's local Steam library, extracts their behavioral **Steam Player DNA**, and dynamically orchestrates a swarm of specialist AI agents to conceive, specify, illustrate, voice-act, code, instrument, and playtest an original **Godot 4.3** (or Python/Pygame) game from scratch—then continuously evolves the game across episodes based on live SQLite WAL telemetry hypotheses.

---

## Core Architecture

1. **Steam Library Discovery & DNA Profiler (`engine/steam/`)**
   - Parses local macOS Steam VDF configurations (`libraryfolders.vdf`, `localconfig.vdf`, `appmanifest_*.acf`) and enriches top-played titles via the Steam Store API.
   - Computes genre affinities, continuous pacing preference (`tactical_methodical` to `kinetic_high_mobility`), challenge tolerance, and aesthetic palette mood.

2. **Requirement-Driven Multi-Agent Studio Swarm (`engine/studio/`)**
   - **`StudioOrchestrator`**: Performs live Gemini meta-planning over the player's `SteamPlayerDNA` (or `EvolutionHypothesisReport`) to assemble a topological execution DAG, iterate requirement-by-requirement (`R1..Rn`), and persist playable verified checkpoints (`cartridges/checkpoints/`).
   - **`CreativeDirectorWorldbuilderAgent`**: Authors the `GenesisWorldSpec` blueprint and the structured **`GameRequirementsSpec`** (`requirements_spec.json`) decomposing the game into ordered feature tracks (`R1..Rn`), verifiable acceptance criteria, and a multimodal asset manifest.
   - **`NarrativeCharacterWriterAgent`**: Authors the `NarrativeBible`, characters, branching acyclic dialogue trees, and situational voice barks.
   - **`VoiceActingAudioProducerAgent`**: Synthesizes spoken `.wav` dialogue clips and sound cues via Gemini TTS (`assets/audio/`).
   - **`VisualArtSpriteDesignerAgent`**: Synthesizes custom pixel-art sprites and character portraits via Gemini multimodal generation with background-to-alpha flood-fill transparency (`assets/sprites/`).
   - **`LeadGameplaySWEAgent`**: Iteratively synthesizes and refines modular **Godot 4.3 GDScript** cartridges (`project.godot`, `main.tscn`, `main.gd`, `scenes.gd`, `mechanics.gd`, `entities.gd`, `ui_hud.gd`) with compiler-verified self-healing.
   - **`TelemetryArchitectAgent`**: Inspects the generated game code, designs custom behavioral telemetry probes (`telemetry_manifest.json`), and instruments the cartridge to stream live runtime state (`[TELEMETRY_EVENT]`) into SQLite WAL.

3. **6-Gate Headless QA Playtester, Game Designer Signoff & Closed-Loop Self-Healing (`engine/supervisor/`, `engine/sim/`)**
   - **Gate A (Syntax, AST & Entrypoint)**: Validates Godot 4.3 headless compiler diagnostics (`verify_godot_cartridge_syntax`), AST security invariants, and Rule 4 Rationale-First contracts.
   - **Gate B (Multimodal Asset Integrity)**: Verifies RIFF `.wav` audio headers, `.png` sprite/portrait integrity, and acyclic dialogue graphs.
   - **Gate C (Headless Execution & Runtime Scene Audit)**: Executes the cartridge headlessly via `godot_harness.gd` / `GodotHeadlessSimulator`, driving synthetic inputs across ticks and auditing the live scene tree (`audio_players_found`, `audio_streams_loaded`, `center_occluding_panels`, camera & visual canvas nodes).
   - **Gate D (60 FPS Frame Budget)**: Verifies tick execution stays within the 16.6ms frame budget.
   - **Gate E (Custom Telemetry Emission)**: Verifies that custom probes declared in `telemetry_manifest.json` actively emit into SQLite WAL during simulation.
   - **Gate F (Challenger & Game Designer Signoff Gate)**: Independent Gemini structured evaluation (`ChallengerGate`) grounded in AST inspection, runtime telemetry, and scene audits. Verifies 100% of `requirements_spec.json` acceptance criteria plus human UX/playability signoff:
     - Raw `.wav` runtime loading via `FileAccess` $\rightarrow$ `AudioStreamWAV` (`audio_streams_loaded > 0`) and procedural action SFX.
     - HiDPI/Retina-accurate mouse coordinate picking via `get_global_mouse_position()`.
     - Placement cursor Z-ordering (`z_index > 0`) and zero center-screen UI panel occlusion (`center_occluding_panels == []`).
     - Visible on-screen mission onboarding objectives, control legend, and landmark labels.
   - Routes any rejection (`RejectionDiagnosticTrace`) back to `LeadGameplaySWEAgent` for holistic multi-module self-healing before atomic promotion to `cartridges/active/`.

4. **SQLite WAL Persistence & Mission Control Web UI (`engine/ui/`, `engine/telemetry/`)**
   - All studio events (`studio_events`), agent task handoff records (`agent_task_records`), requirements (`studio_requirements`), world specs, and telemetry probes are persisted in **SQLite WAL** (`data/telemetry.db`).
   - Real-time WebSocket dashboard (`http://127.0.0.1:8000`) tails SQLite WAL across processes (`/ws/swarm`, `/ws/stream`) to stream live Rationale-First deliberations, PEV tool cards, task handoff envelopes (`ArtifactMetadataEnvelope`), requirement progress, and playable checkpoints without page reloads.

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

Open **`http://127.0.0.1:8000`** in your browser to watch the agent swarm analyze your Steam DNA, author `requirements_spec.json`, synthesize sprites/audio/GDScript modules track by track, run the 6-Gate QA & Designer Signoff verifier, and promote the compiled game to `cartridges/active/`.

### 3. Play the Generated Game (or Any Verified Checkpoint)

Launch the promoted active cartridge natively in Godot 4.3 at 60 FPS:

```bash
python run_game.py
```

Or launch and compare incremental requirement checkpoints (`R1`, `R2`, ..., `R5`):

```bash
python run_game.py --list-checkpoints
python run_game.py --latest-checkpoint
python run_game.py --checkpoint candidate_genesis_2ab772_R5
```

### 4. Run the Verification & Test Suite

```bash
pytest
```
