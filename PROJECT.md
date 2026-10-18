# Project: Autonomous Agentic Game Studio & Observability Platform

## Architecture
The system is an autonomous, requirement-driven multi-agent game studio swarm that scans a player's Steam library, extracts their behavioral `SteamPlayerDNA`, authors a comprehensive `GameRequirementsSpec`, and iteratively conceives, illustrates, voice-acts, codes, instruments, and verifies complete, original **Godot 4.3** (and Pygame) games—persisting all telemetry, studio events, task records, and requirement progress in **SQLite WAL** (`data/telemetry.db`).

### Core Modules & Responsibilities
- `engine/studio/`: Orchestration and specialist agents.
  - `orchestrator.py`: Meta-planning DAG and iterative requirement-by-requirement (`R1..Rn`) execution loop with verified checkpointing (`cartridges/checkpoints/`).
  - `models.py`: Epistemic data models (Pydantic models inheriting from `BaseEpistemicModel` with `deliberation` strictly at field index 0).
  - `director.py`: Concept (`GenesisWorldSpec`) & Game Requirements (`GameRequirementsSpec`) authoring from `SteamPlayerDNA` / `EvolutionHypothesisReport`.
  - `swe.py`: Iterative multi-module Godot 4.3 GDScript (`project.godot`, `main.tscn`, `main.gd`, `scenes.gd`, `mechanics.gd`, `entities.gd`, `ui_hud.gd`) and Python code synthesis & holistic self-healing (`LeadGameplaySWEAgent`).
  - `visual.py`: Live Gemini pixel-art sprite and portrait synthesis with corner flood-fill alpha transparency extraction (`VisualArtSpriteDesignerAgent`).
  - `audio.py`: Dynamic Gemini TTS `.wav` voice and sound synthesis (`VoiceActingAudioProducerAgent`).
  - `narrative.py`: Narrative bible, character sheets, dialogue trees, and situational barks (`NarrativeCharacterWriterAgent`).
  - `telemetry_architect.py`: Custom behavioral telemetry probe design (`telemetry_manifest.json`) and GDScript/Python AST instrumentation (`TelemetryArchitectAgent`).
- `engine/supervisor/`: Verification, QA playtesting, and Game Designer signoff.
  - `supervisor.py`: `BuildSupervisor` executing Quality Gates A–F and closed-loop self-healing before atomic promotion to `cartridges/active/`.
  - `challenger.py`: Gate F (`ChallengerGate` / Adversarial Design Critic & Game Designer Signoff) auditing 100% of acceptance criteria in `requirements_spec.json` plus runtime audio stream binding (`AudioStreamWAV`), HiDPI-exact mouse picking (`get_global_mouse_position()`), placement cursor Z-ordering (`z_index`), non-occluding HUD panels (`center_occluding_panels`), and player onboarding clarity.
  - `models.py`: Verification models and `RejectionDiagnosticTrace` schema.
- `engine/sim/`: Headless simulation and runtime scene auditing.
  - `godot_verifier.py` & `godot_harness.gd`: Headless Godot 4.3 compiler verification (`verify_godot_cartridge_syntax`), multi-tick exploratory input driver (`GodotHeadlessSimulator`), live scene-tree UX/audio audit (`_audit_scene_tree`), and `[TELEMETRY_EVENT]` bridge into SQLite WAL.
  - `simulator.py` & `headless.py`: Headless Pygame simulation runner (`SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`).
- `engine/telemetry/`: Continuous SQLite WAL persistence.
  - `db.py`, `schema.sql`, `schema.py`, `recorder.py`, `writer.py`: Zero-lock SQLite WAL storage (`data/telemetry.db`) for `studio_events`, `agent_task_records`, `studio_requirements`, `studio_state_kv`, `telemetry_probes`, and `custom_telemetry_events`.
- `engine/ui/`: Mission Control Web Observability UI.
  - `server.py`: Starlette WebSocket server and `SwarmEventBus` with cross-process SQLite WAL live tailing (`/ws/swarm`, `/ws/stream`).
  - `static/index.html`: Real-time cybernetic observability dashboard with deduplicated live event streaming, interactive Agent Task & Handoff Drawer (`ArtifactMetadataEnvelope`), Requirement Backlog tracker, and Checkpoint Progression timeline.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Agent-Authored Requirements Spec Model | Pydantic models for `GameRequirementsSpec`, `RequirementTrack`, `AcceptanceCriterion`, `AssetRequirement` with `deliberation` first | M1 | R1 |
| 2 | Pure Generalizability & Domain-Agnostic Concept Prompts | Remove hardcoded arena/character assumptions from `GenesisWorldSpec`, `NarrativeBible`, and director prompts | M1 | R1, Rule 1 |
| 3 | Requirements Backlog Generation Step | Upstream step in Studio swarm that authors `requirements_spec.json` from `SteamPlayerDNA` / `EvolutionHypothesisReport` | M1 | R1 |
| 4 | Requirements Spec Emitted to Event Bus | Emit `REQUIREMENTS_SPEC_GENERATED` event with `deliberation` first to `SwarmEventBus` and SQLite WAL | M1 | R1, Rule 4 |
| 5 | Live Gemini Pixel-Art Sprite Synthesis | Dynamically synthesize sprites/tiles via Gemini image generation with alpha flood-fill transparency in `visual.py` | M2, M8 | R2, Rule 5 |
| 6 | Dynamic Audio Voice Synthesis & Runtime WAV Loader | Gemini TTS `.wav` generation in `audio.py` + Godot 4 runtime `FileAccess` -> `AudioStreamWAV` PCM loader & procedural SFX | M2, M8 | R2, Rule 5 |
| 7 | Upstream Asset Generation in Swarm DAG | Order DAG so asset generation occurs before SWE code implementation, sealing assets into candidate staging directory | M2 | R2 |
| 8 | Multi-Module Godot 4.3 & Python SWE Synthesis | Modular GDScript 4.3 synthesis (`project.godot`, `main.tscn`, `main.gd`, `scenes.gd`, `mechanics.gd`, `entities.gd`, `ui_hud.gd`) | M3, M7 | R2 |
| 9 | Iterative Requirement-by-Requirement Execution Loop | Orchestrator iterates over ordered `RequirementTrack` items, executing incremental code refinement and per-track headless smoke checks | M3 | R2 |
| 10 | Real-Time SQLite WAL WebSocket Streaming | Cross-process SQLite WAL tailing (`studio_events WHERE rowid > last_rowid`) streaming live updates over `/ws/swarm` and `/ws/stream` | M3, M8 | R2, Rule 4 |
| 11 | Independent Challenger & Game Designer Signoff Gate (Gate F) | Gemini structured evaluation gate in `engine/supervisor/challenger.py` auditing 100% of acceptance criteria + UX/audio/mouse/onboarding signoff | M4, M8 | R3 |
| 12 | Triangulated Challenger Grounding & Scene Audit | Ground Challenger decisions in AST inspection, runtime telemetry events, and `godot_harness.gd` scene audits (`audio_streams_loaded`, `center_occluding_panels`) | M4, M8 | R3 |
| 13 | Rejection Diagnostic Trace & Holistic Repair | Structured `RejectionDiagnosticTrace` routing multi-module Godot compiler & UX repairs (`repair_godot_cartridge_holistic`) | M4, M8 | R3 |
| 14 | Challenger Verdict Event Streaming | Emit `CHALLENGER_GATE_VERDICT` and `QA_GATE_RESULT` events with full rejection diagnostics and Challenger rationale | M4, M8 | R3, Rule 4 |
| 15 | Test Isolation & Baseline Test Fixes | Clean pre-Genesis test isolation, signal handling in `run_game.py`, and SQLite WAL concurrency | M5 | Acceptance Criteria, Rule 3 |
| 16 | E2E Requirement-Driven Test Suite | Opaque-box test suite across Tiers 1–5 verifying R1–R6 requirements and adversarial stress cases | E2E Track | Acceptance Criteria |
| 17 | Live Godot 4.3 Genesis Build & Promotion | End-to-end Genesis build promoted to `cartridges/active/` and verified via `run_game.py` | M6, M7, M8 | Acceptance Criteria |
| 18 | Agent Task History & Handoff Observability UI | Task history persistence in SQLite WAL (`agent_task_records`) and interactive drawer with `ArtifactMetadataEnvelope` cards in `index.html` | M_UI | R5 |
| 19 | Checkpoint Resilience & Per-Track Retry | Transient error handling, `.gd`/`.py` snapshotting/restoration, and completed-track resumption in `orchestrator.py` | M6, M7 | Follow-up |
| 20 | Incremental Playable Verified Checkpoints | Immutable checkpoint persistence in `cartridges/checkpoints/`, launcher `--checkpoint` / `--latest-checkpoint` support, and UI timeline | M6, M7 | R6 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Requirement-driven test suite across Tiers 1–5 | none | DONE |
| M1 | Requirements Spec & Generalizable Models | `engine/studio/models.py`, `director.py`, `narrative.py`, `engine/ui/models.py` | none | DONE |
| M2 | Dynamic Multimodal Asset Synthesis | `engine/studio/visual.py`, `engine/studio/audio.py` | M1 | DONE |
| M3 | Iterative Requirement Execution Loop | `engine/studio/orchestrator.py`, `engine/studio/swe.py`, `engine/ui/server.py` | M1, M2 | DONE |
| M_UI | Agent Task History & Handoff Observability UI | `UI_DESIGN_SPEC.md`, SQLite WAL task history API, interactive inspector in `engine/ui/static/index.html` | M1, M3 | DONE |
| M4 | Challenger Verification Gate | `engine/supervisor/challenger.py`, `supervisor.py`, `models.py` | M1, M3 | DONE |
| M5 | Test Isolation & Baseline Hygiene | `run_game.py`, `engine/telemetry/writer.py`, test fixtures | none | DONE |
| M6 | Final Milestone & Live Genesis Verification | Phase 1 (100% E2E tests pass), Phase 2 (Tier 5 hardening), live Genesis build, pytest | E2E, M1–M5, M_UI | DONE |
| M7 | Godot 4.3 Transition & SQLite WAL Persistence | Code-First Godot 4.3 Engine, GDScript 4 Synthesis, Headless Verification, SQLite WAL & Checkpoints | M6 | DONE |
| M8 | Live WAL Streaming, Gemini Sprites & Designer Signoff Gate | Real-time `/ws/swarm` SQLite WAL tailing, live Gemini sprite synthesis, HiDPI mouse & `AudioStreamWAV` repair, and Designer UX Signoff | M7 | DONE |

## Interface Contracts

### 1. `GameRequirementsSpec` Contract
- Location: `engine/studio/models.py`
- Inherits: `BaseEpistemicModel`
- Fields:
  - `deliberation: str = Field(..., description="Epistemic rationale analyzing player DNA, genre feasibility, and required systems")`
  - `game_title: str`
  - `genre_paradigm: str`
  - `target_audience_summary: str`
  - `requirements: list[RequirementTrack]`
  - `asset_manifest: list[AssetRequirement]`

### 2. `ChallengerGateEvaluation` & Game Designer Signoff Contract
- Location: `engine/supervisor/models.py` & `engine/supervisor/challenger.py`
- Inherits: `BaseEpistemicModel`
- Fields:
  - `deliberation: str = Field(..., description="Multi-perspective engineering & Game Designer UX critique evaluating candidate against every acceptance criterion and human playability mandate")`
  - `candidate_id: str`
  - `overall_verdict: bool` (`True` = PASS, `False` = REJECT)
  - `evaluated_requirements: list[RequirementEvaluationRecord]`
  - `rejection_trace: Optional[RejectionDiagnosticTrace] = None`

### 3. Swarm Event Bus & SQLite WAL Contract
- Location: `engine/ui/models.py`, `engine/ui/server.py`, `engine/telemetry/schema.sql`
- SQLite WAL Tables (`data/telemetry.db`):
  - `studio_events`: Stores all swarm events (`REQUIREMENTS_SPEC_GENERATED`, `REQUIREMENT_STATUS_UPDATE`, `QA_GATE_RESULT`, `CHALLENGER_GATE_VERDICT`, `CARTRIDGE_PROMOTED`, `CHECKPOINT_SAVED`, `AGENT_STREAM`) tailed live by `/ws/swarm` and `/ws/stream`.
  - `agent_task_records`: Stores per-task execution records, deliberations, verification verdicts, and `ArtifactMetadataEnvelope` handoffs.
  - `studio_requirements`: Stores per-track requirement status and acceptance criteria progress.
  - `studio_state_kv`: Stores serialized `world_spec` and `requirements_spec` state.

## Code Layout
- `engine/godot_runtime.py`: Portable Godot 4.3 runtime resolution and headless/windowed execution wrapper.
- `engine/sim/godot_harness.gd`: Headless Godot 4.3 scene-tree auditor (`_audit_scene_tree`), exploratory input driver, and telemetry bridge.
- `engine/sim/godot_verifier.py`: Godot 4.3 stderr compiler parser (`parse_godot_stderr`, `verify_godot_cartridge_syntax`) and `GodotHeadlessSimulator`.
- `engine/studio/director.py`: Concept (`GenesisWorldSpec`) & Requirements (`GameRequirementsSpec`) authoring.
- `engine/studio/orchestrator.py`: Swarm DAG, iterative requirement execution loop, and checkpoint manager.
- `engine/studio/swe.py`: Multi-module GDScript 4.3 / Python synthesis and holistic self-healing (`repair_godot_cartridge_holistic`).
- `engine/studio/visual.py`: Live Gemini pixel-art sprite & portrait generator with background-to-alpha transparency.
- `engine/studio/audio.py`: Live Gemini TTS voice & sound generator.
- `engine/supervisor/challenger.py`: Gate F Adversarial Design Critic & Game Designer Signoff (`CandidateASTInspector`, `ChallengerGate`).
- `engine/supervisor/supervisor.py`: 6-Gate `BuildSupervisor` and self-healing loop.
- `engine/ui/server.py`: Starlette WebSocket server & SQLite WAL live tailing (`SwarmEventBus`).
- `engine/ui/static/index.html`: Mission Control Web UI.
- `run_studio.py`: Studio Mission Control server & daemon entrypoint.
- `run_game.py`: Native 60 FPS game & checkpoint launcher.
