# Project: Autonomous Game Engine & Showrunner Harness for macOS

## Architecture
The deliverable is an **Autonomous Game Engine that builds and evolves the game**—a continuous, rationale-first studio that ingests the user's Steam library to build the initial game (Episode 0 / Genesis Cartridge), and continuously analyzes live player telemetry to autonomously design, compile, validate, and hot-deliver subsequent episodic gameplay updates on macOS.

```
+-----------------------------------------------------------------------------------+
|                        Steam Library & Gameplay Ingestion                         |
|  - Local macOS Steam Scanner: ~/Library/Application Support/Steam/ (vdf, acf)    |
|  - Steam Web API Bridge: IPlayerService, ISteamUserStats, store appdetails        |
|  - Fallback / Test Fixtures: Deterministic Steam library profiles for CI/testing |
+-----------------------------------------+-----------------------------------------+
                                          |
                        (Historical Gameplay Data)
                                          v
+-----------------------------------------------------------------------------------+
|                   Initial Game Genesis Pipeline (Episode 0)                       |
|  - The Analyst: Extracts Initial Player DNA Profile (playtime, genres, mechanics) |
|  - The Director: Emits Rationale-First `genesis_spec.md` (`reasoning` first)      |
|  - Autonomous SWE Compiler: Compiles initial game cartridge (`cartridges/seed/`)  |
|  - Synthetic Playtester: Validates traversability & critical path                 |
+-----------------------------------------+-----------------------------------------+
                                          |
                     (Initial Playable Cartridge Booted)
                                          v
+-----------------------------------------------------------------------------------+
|                        Autonomous Game Engine Kernel & Substrate                  |
|                                                                                   |
|  1. Core Runtime (60 FPS): Pygame-CE Cocoa Loop (Thread 0), Liquid Retina Scaling  |
|  2. Physics & Spatial Kernel: Vector2, AABB, Circle, SpatialHashGrid, NavMesh     |
|  3. Dynamic Module/Cartridge Loader (`engine/loader.py`):                         |
|     - Hot-reloads executable Python game modules (entities, mechanics, hazards)   |
|     - Loads cartridges from `cartridges/` or compiled episodic packages           |
|  4. Engine SDK / Base API (`engine/api/`):                                        |
|     - Canonical base classes: BaseEntity, BasePlayer, BaseEnemy, BaseHazard,      |
|       BaseWeapon, BaseLevelSpec, BaseScene (pinned and exposed via MCP)           |
|  5. Non-blocking SQLite WAL Telemetry Host:                                       |
|     - Sub-microsecond queue (put_nowait < 2us) -> Background Writer Thread        |
|  6. Base OS Priority Scheduler:                                                   |
|     - P0 (Gameplay) > P1 (Telemetry) > P2 (Background Studio compilation)         |
+-----------------------------------------+-----------------------------------------+
                                          |
                      (Continuous In-Game Telemetry Stream)
                                          v
+-----------------------------------------------------------------------------------+
|                        SQLite WAL Telemetry Store (telemetry.db)                  |
|  - PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;                        |
|  - Batched transactions without frame drops                                       |
+-----------------------------------------+-----------------------------------------+
                                          |
               (Read committed events via separate reader connection)
                                          v
+-----------------------------------------------------------------------------------+
|                   Background Showrunner Studio (Engine as Author)                 |
|                                                                                   |
|  1. The Analyst (R3): Queries WAL events -> Generates BehavioralProfile           |
|  2. The Director (R3): Strict Rationale-First update_spec.md generation          |
|     (deliberation/reasoning first, bounded +/-15% adaptation clamp)              |
|  3. Local MCP Server Data Bridge (R4): Exposes telemetry profiles, live NavMesh,  |
|     version-pinned Engine SDK API docs (`engine/api/`), and `get_steam_player_dna`|
|  4. Autonomous SWE Worker / Engine Compiler (R4):                                 |
|     - Programmatic SWE execution: WRITES NEW EXECUTABLE PYTHON GAME MODULES       |
|       (`mechanics.py`, `entities.py`, `level_spec.json`) based on update_spec.md  |
|     - Validates syntax & AST imports (no prohibited system calls)                 |
|  5. Asset Liaison & Style Bible (R4): Generates 16-color procedural textures       |
|  6. Build Supervisor & QA Guardrail (R5): Spawns HeadlessSimulator & Synthetic   |
|     Playtesting Bots (A* pathfinder) to test dynamically compiled game code       |
|  7. Self-Healing Rejection Loop: Rejects defective game code, provides diagnostic |
|     trace, drives SWE Worker to repair and re-evaluate                            |
+-----------------------------------------+-----------------------------------------+
                                          |
                  (Promoted Game Cartridge / Episode Bundle)
                                          v
+-----------------------------------------------------------------------------------+
|                        Hot-Reloading Game Cartridge Runtime                       |
|  - Dynamic Module Loader (`engine/loader.py`) loads newly compiled game package:  |
|    `cartridges/active/` or `episodes/episode_N/` into live game runtime!          |
|  - Between-run transition loads newly authored mechanics, enemy AI, and layout    |
+-----------------------------------------------------------------------------------+
```

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Python Runtime & Project Scaffolding | Isolated Python 3.11 virtual environment, project configuration, and dependencies (`pygame-ce`, `pydantic`, `mcp`, `pytest`, `pillow`). | M1 | Survey 1 |
| 2 | Game Engine Kernel & Dynamic Cartridge Substrate | 60 FPS macOS runtime, spatial/collision kernel, InputManager, SceneManager, and dynamic module loader (`engine/loader.py`) that loads executable game packages. | M1 | User Directive, ORIGINAL_REQUEST R1 |
| 3 | Engine SDK / Base API Surface | Canonical, version-pinned base classes (`BaseEntity`, `BasePlayer`, `BaseEnemy`, `BaseHazard`, `BaseWeapon`, `BaseLevelSpec`) for dynamically authored game code. | M1 | User Directive, ORIGINAL_REQUEST R4 |
| 4 | Continuous SQLite WAL Telemetry Host | High-throughput, non-blocking producer-consumer queue streaming events to SQLite in WAL mode without frame drops. | M1 | Survey 1, ORIGINAL_REQUEST R1 |
| 5 | Steam Library Ingestion Engine | Scans local macOS Steam paths (`~/Library/Application Support/Steam`) and Steam Web API; extracts playtimes, genres, mechanical tags, and achievement depth. | M2 | User Directive (Steam) |
| 6 | Steam Player DNA Profiler | Parses Steam data into a structured `SteamPlayerDNA` profile stored in SQLite and exposed via MCP (`get_steam_player_dna`). | M2 | User Directive (Steam) |
| 7 | Asynchronous Resource-Aware OS Layer | Three-tier priority scheduler (P0 gameplay, P1 telemetry HUD, P2 showrunner) with cooperative yielding on macOS to preserve 60 FPS. | M2 | Survey 2, ORIGINAL_REQUEST R2 |
| 8 | Artifact Hand-Off Protocol | Typed `ArtifactMetadataEnvelope` with SHA256 hashes routing structured data between agents to prevent context window collapse. | M2 | Survey 2, ORIGINAL_REQUEST R2 |
| 9 | Telemetry Analyst Engine | Parses raw SQLite telemetry into `BehavioralProfile` (kinetic velocity vectors, dash frequency, spatial occupancy heatmaps, retry metrics). | M3 | Survey 2, ORIGINAL_REQUEST R3 |
| 10 | Cognitive Director Swarm & Genesis Spec | Strict Rationale-First `genesis_spec.md` and `update_spec.md` generator placing deliberation/reasoning first before any decision tokens, with bounded adaptation clamp. | M3 | User Directive, ORIGINAL_REQUEST R3 |
| 11 | Local MCP Server Data Bridge | Exposes standard MCP tools: `get_steam_player_dna`, `get_telemetry_profile`, `query_spatial_map` (NavMesh), and `get_engine_api_docs` (pinned Engine SDK API). | M4 | Survey 2, ORIGINAL_REQUEST R4 |
| 12 | Autonomous SWE Engine Compiler | Programmatic SWE worker that **writes, compiles, and validates new executable Python game code** (`mechanics.py`, `entities.py`) for Episode 0 (Genesis) and subsequent episodes. | M4 | User Directive, ORIGINAL_REQUEST R4 |
| 13 | Asset Liaison & Style Bible Pipeline | Enforces strict 16-color indexed palette, bounds, and deterministic local procedural synthesis via Pillow without external network dependencies. | M4 | Survey 2, ORIGINAL_REQUEST R4 |
| 14 | Decoupled Headless Simulation Host | Discrete fixed-timestep simulation (`HeadlessSimulator`, `dt=1/60s`) running dynamic game cartridges without GUI display overhead. | M5 | Survey 3, ORIGINAL_REQUEST R5 |
| 15 | Synthetic Playtesting Bots | Headless bots verifying collision boundaries, critical-path A* traversability from spawn to exit, and time-to-clear on dynamically compiled game code. | M5 | Survey 3, ORIGINAL_REQUEST R5 |
| 16 | Build Supervisor & Self-Healing Guardrail | Evaluates newly authored game code; rejects non-traversable levels with diagnostic traces; drives SWE code self-healing before promotion. | M5 | Survey 3, ORIGINAL_REQUEST R5 |
| 17 | Dual Steam Profile & Playstyle Divergence | Proves two distinct Steam library profiles (e.g. Action/Roguelike rusher vs Strategic/Tactical methodical) trigger distinct `genesis_spec.md` and distinct initial game code. | M6 | User Directive (Steam) |
| 18 | Autonomous E2E Engine Acceptance Harness (`verify_engine.py`) | Automated 6-stage script proving Steam Library -> Player DNA -> Genesis Spec -> Initial Game Compilation -> Headless Playtest -> Live Telemetry Evolution. | M6 | User Directive, Acceptance Criteria |
| 19 | Single-Command macOS Launcher (`run_game.sh`) | Shell launcher starting the playable game window at 60 FPS alongside the background Showrunner monitor and Steam ingestion status. | M6 | Survey 1, Acceptance Criteria |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Author test infrastructure, 4-tier test hierarchy, and publish `TEST_READY.md`. | none | DONE |
| M1 | Engine Kernel, Dynamic Loader & SDK | Engine runtime, spatial kernel, Engine SDK (`engine/api/`), Dynamic Cartridge Loader (`engine/loader.py`), SQLite WAL Telemetry, and Seed Cartridge (`cartridges/seed/`). | none | DONE |
| M2 | Steam Ingestion, Resource OS Layer & Artifacts | Local macOS Steam library scanner (`~/Library/Application Support/Steam`), Steam Web API bridge, `SteamPlayerDNA` profiler, 3-tier priority scheduler, and `ArtifactMetadataEnvelope`. | M1 | DONE |
| M3 | Live Gemini Studio Swarm & Multimodal Assets | 9-agent live Gemini studio swarm (`StudioOrchestrator`, `Analyst`, `Director`, `Narrative`, `Audio/TTS`, `Visual Art`, `Gameplay SWE`, `Telemetry Architect`, `QA Supervisor`), real multimodal inference via `google-genai`, replacing legacy templates. | M1, M2 | IN_PROGRESS |
| M4 | SWE Engine Compiler, MCP Server & Assets | Local MCP server exposing Steam DNA and Engine SDK docs, Autonomous SWE Compiler dynamically writing Python game code (Episode 0 & beyond), Style Bible asset liaison. | M3 | PLANNED |
| M5 | Automated QA, Headless Sim & Self-Healing | Headless simulation host, A* playtesting bots, Build Supervisor rejection gate and closed-loop code self-healing repair. | M4 | PLANNED |
| M6 | Dual Profile E2E Verification & Final Launcher | Dual Steam profile & playstyle simulation proving Engine-authored game code divergence, `verify_engine.py` acceptance harness, and `run_game.sh`. | M5, E2E | PLANNED |

---

## Code Layout
```
/Users/rhenretta/workspace/rhenretta/ai-game/
├── config.py                           # Resolution, 60 FPS, paths, database configuration
├── requirements.txt                    # pygame-ce, pydantic, mcp, pytest, pillow
├── run_game.py                         # Single unified Python entrypoint for game + HUD
├── run_game.sh                         # Executable macOS shell launcher
├── verify_engine.py                    # Autonomous E2E acceptance script (Steam -> Genesis -> Playtest -> Telemetry)
├── engine/                             # The Reusable Engine Substrate
│   ├── __init__.py
│   ├── loader.py                       # Dynamic Module/Cartridge Loader & Hot-Swapper
│   ├── api/                            # Version-Pinned Engine SDK / Base Classes
│   │   ├── __init__.py
│   │   ├── base_entity.py              # BaseEntity, LivingEntity
│   │   ├── base_combat.py              # BaseWeapon, Hitbox, Hurtbox, DamagePacket
│   │   ├── base_enemy.py               # BaseEnemy (AI interface, health bar hook)
│   │   ├── base_hazard.py              # BaseHazard
│   │   └── base_level.py               # BaseLevelSpec
│   ├── steam/                          # Steam Discovery & Ingestion Engine
│   │   ├── __init__.py
│   │   ├── scanner.py                  # Local macOS Steam scanner (~/Library/Application Support/Steam)
│   │   ├── web_api.py                  # Steam Web API client (IPlayerService, ISteamUserStats)
│   │   └── dna.py                      # SteamPlayerDNA profiler & SQLite persistence
│   ├── core/                           # 60 FPS Cocoa loop, scene manager, clock, input
│   ├── spatial/                        # NavMesh, spatial hash grid, collision boundaries
│   ├── scheduler/                      # 3-tier priority OS scheduler & checkpointer
│   ├── telemetry/                      # SQLite WAL logger, queue worker, schema
│   ├── analyst/                        # Telemetry parser & BehavioralProfile extractor
│   ├── director/                       # Rationale-First genesis_spec.md & update_spec.md generator
│   ├── mcp/                            # Local MCP server data bridge & tools (incl. get_steam_player_dna)
│   ├── compiler/                       # SWE Engine Compiler (dynamically writes game code)
│   ├── assets/                         # Style Bible, 16-color palette, procedural synthesizer
│   ├── supervisor/                     # Build Supervisor & self-healing feedback loop
│   └── sim/                            # Decoupled headless simulator (fixed dt=1/60s)
├── cartridges/                         # Dynamically Authored Game Packages
│   ├── seed/                           # Initial starter cartridge
│   │   ├── cartridge.json              # Manifest
│   │   ├── entities.py                 # Seed Player, Enemy, Hazard classes
│   │   ├── mechanics.py                # Combat rules, weapon behavior
│   │   └── level.json                  # Spatial room & portal layout
│   └── active/                         # Current hot-loaded game cartridge
├── qa/
│   ├── __init__.py
│   ├── playtester.py                   # A* pathfinding bots & traversability checker
│   └── profiles.py                     # Rusher vs Cautious synthetic profiles
├── tests/
│   ├── __init__.py
│   ├── test_tier1_units.py             # Schema, AST rationale check, WAL integrity, Steam parsing
│   ├── test_tier2_playtest.py          # Headless simulation, A* navigation, clipping
│   └── test_tier3_showrunner.py        # Steam DNA -> Genesis -> Analyst -> Director -> MCP -> Supervisor loop
└── data/                               # Local SQLite database & generated artifacts
    ├── telemetry.db
    └── specs/
```
