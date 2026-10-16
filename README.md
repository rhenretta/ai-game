# Autonomous Agentic Game Studio Harness & Observability Platform

An autonomous, self-assembling multi-agent game studio harness that scans a player's local Steam library, extracts their behavioral **Steam Player DNA**, and dynamically orchestrates a swarm of specialist AI agents to invent, write, voice-act, illustrate, code, instrument, and playtest an original 2D Pygame game from scratch—then continuously evolves the game across episodes based on live telemetry hypotheses.

---

## Core Architecture

1. **Steam Library Discovery & DNA Profiler (`engine/steam/`)**
   - Parses local macOS Steam VDF configurations (`libraryfolders.vdf`, `localconfig.vdf`, `appmanifest_*.acf`) and enriches top-played titles via the Steam Store API.
   - Computes genre affinities, continuous pacing preference (`tactical_methodical` to `kinetic_high_mobility`), challenge tolerance, and aesthetic palette mood.

2. **Self-Assembling Multi-Agent Studio Swarm (`engine/studio/`)**
   - **`StudioOrchestrator`**: Performs live Gemini meta-planning over the player's `SteamPlayerDNA` (or `PlayerPreferenceHypothesisReport`) to assemble a topological execution DAG and task manifest.
   - **`CreativeDirectorWorldbuilderAgent`**: Designs the game blueprint (`GenesisWorldSpec`), title, genre, lore, core rules, and aesthetic direction.
   - **`NarrativeCharacterWriterAgent`**: Authors the `NarrativeBible`, characters, branching acyclic dialogue trees, and situational voice barks.
   - **`VoiceActingAudioProducerAgent`**: Synthesizes spoken `.wav` dialogue clips via Gemini TTS (`assets/audio/`).
   - **`VisualArtSpriteDesignerAgent`**: Generates character portraits and 16-color indexed pixel-art sprites (`assets/sprites/`).
   - **`LeadGameplaySWEAgent`**: Writes self-contained, AST-sandboxed Python/Pygame cartridge modules (`mechanics.py`) from scratch without static templates.
   - **`TelemetryArchitectAgent`**: Inspects the generated game code, designs custom behavioral telemetry probes (`telemetry_manifest.json`), and instruments the cartridge AST.

3. **5-Gate Headless QA Playtester & Closed-Loop Self-Healing (`engine/supervisor/`, `engine/sim/`)**
   - **Gate A (AST & Entrypoint)**: Validates syntax, AST security sandbox invariants, Rule 4 Rationale-First class attributes, and `BaseCartridge` / `BaseScene` contracts.
   - **Gate B (Multimodal Assets)**: Verifies RIFF `.wav` audio headers, `.png` sprite integrity, and acyclic dialogue graphs.
   - **Gate C (Headless Execution & Progression)**: Steps the cartridge headlessly (`SDL_VIDEODRIVER=dummy`) with synthetic inputs, verifying zero unhandled exceptions and observable state progression.
   - **Gate D (60 FPS Frame Budget)**: Verifies tick execution stays within the 16.6ms frame budget.
   - **Gate E (Custom Telemetry Emission)**: Verifies that custom probes declared in `telemetry_manifest.json` actively emit during simulation.
   - Automatically routes any gate rejection (`RejectionDiagnosticTrace`) back to `LeadGameplaySWEAgent` for closed-loop self-healing before atomic promotion to `cartridges/active/`.

4. **Mission Control Web Observability UI (`engine/ui/`)**
   - Game-agnostic real-time WebSocket dashboard (`http://127.0.0.1:8000`) streaming the live 9-agent DAG, Rationale-First deliberations, PEV tool invocation cards, Steam Player DNA & Game Blueprint, custom telemetry probes & hypotheses, voice clip previewer, and code diff inspector.

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

Before the Studio runs a Genesis build, no pre-baked game exists on disk. Start the Studio server and background daemon:

```bash
.venv/bin/python run_studio.py
```

Open **`http://127.0.0.1:8000`** in your browser to watch the agent swarm analyze your Steam DNA, design the game blueprint, synthesize code/audio/sprites, run the 5-Gate QA verifier, and promote the compiled game to `cartridges/active/`.

### 3. Play the Generated Game

Once a cartridge is promoted to `cartridges/active/`:

```bash
.venv/bin/python run_game.py
```

### 4. Run the Verification & Test Suite

```bash
.venv/bin/pytest
```
