# Studio Observability UI Design Specification (Requirement R5)
**Document Version:** 2.1.0  
**Status:** IMPLEMENTED  
**Target Milestone:** M_UI → M19  
**Module Scope:** `studio-web/`, `engine/ui/server.py`, `engine/ui/models.py`  
**Reference Requirements:** R1, R2, R3, R5  

---

## 1. Architectural & Epistemic Principles

### 1.1 Mandate & Objective
This specification defines the user interface, interaction architecture, data models, and persistence layer for the **Studio Observability UI** (`studio-web/` and `engine/ui/`). The system provides complete, real-time epistemic observability over the autonomous multi-agent studio swarm as it conceives, specifies, builds, playtests, and verifies complete games from player Steam DNA.

The UI guarantees full transparency into:
1. Every task and requirement track executed by any agent or subagent.
2. The full pre-decision epistemic reasoning (`deliberation`) motivating each step.
3. The cryptographically sealed handoff documentation (`ArtifactMetadataEnvelope`) including inputs, outputs, SHA-256 digests, and verification verdicts.
4. The live, iterative progress of the agent-authored `GameRequirementsSpec` and `ProductionBoard` backlogs (including tracked `qa_gate_pass*` 5-Gate QA self-repair items).
5. Persistent session state surviving full browser refreshes and immediate state reconciliation across Uvicorn server reloads.

### 1.2 Core Architectural Invariants
1. **Rationale-First Epistemic Ordering (Rule 4)**:
   - In all UI cards, modal inspectors, and drawer panels, the epistemic reasoning (`deliberation`) block is placed **strictly first** at visual index 0 before any status pills, numerical metrics, hashes, or synthesized assets.
2. **Zero Silent Fallbacks & Error Concealment (Rule 5)**:
   - When a quality gate, Challenger gate, or code compilation fails, the UI surfaces authentic, unmasked diagnostic traces (`RejectionDiagnosticTrace`), complete tracebacks, and defect coordinates. Fallbacks to mock data or placeholder assets are strictly prohibited.
3. **Production Container Invariance & SPA Routing Parity (Rule 6)**:
   - Originally prototyped in vanilla ES6 (`engine/ui/static/index.html` in Milestone `M_UI`), Mission Control is now consolidated exclusively into the modular React 19 + TypeScript + TanStack Query single-page application in `studio-web/` (validated via `npm run typecheck` and `npm test` / `vitest run`). `engine/ui/server.py` serves compiled static bundles directly from `studio-web/dist/` in headless Linux containers and redirects HTML browser visits to `STUDIO_WEB_ORIGIN` (`http://127.0.0.1:5173/`) during local Vite development.

### 1.3 Design System & CSS Token Registry
The UI adheres to the dark cybernetic aesthetic grounded in the 16-color Style Bible palette:

```css
:root {
  /* Style Bible 16-Color Palette */
  --bg: #000000;
  --navy: #1D2B53;
  --panel: #0d1630;
  --panel-elevated: #111e42;
  --panel-active: #16285a;
  --border: #29ADFF;
  --border-subtle: rgba(41, 173, 255, 0.25);
  --crimson: #7E2553;
  --slate: #5F574F;
  --light-gray: #C2C3C7;
  --white: #FFF1E8;
  --neon-red: #FF004D;
  --amber: #FFA300;
  --yellow: #FFEC27;
  --green: #00E436;
  --cyan: #29ADFF;
  --purple: #83769C;
  --pink: #FF77A8;
  --peach: #FFCCAA;

  /* Lifecycle State Tokens */
  --state-idle: #5F574F;
  --state-analyzing: #FFA300;
  --state-synthesizing: #29ADFF;
  --state-verifying: #FF77A8;
  --state-promoted: #00E436;
  --state-rejected: #FF004D;

  /* Typography */
  --font-mono: "SF Mono", Monaco, "Courier New", monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;

  /* Dimensions & Z-Index */
  --drawer-width: 680px;
  --z-base: 1;
  --z-elevated: 100;
  --z-overlay: 1000;
  --z-drawer: 1200;
  --z-toast: 1500;
}
```

---

## 2. Global Layout & Structural Wireframe

### 2.1 DOM Hierarchy
The page layout consists of a fixed top header, an offline alert banner, a two-column responsive grid, and the sliding Task & Handoff History Drawer:

```
+-------------------------------------------------------------------------------------------------------------+
| HEADER: Title, Studio Status Badge ("ONLINE" / "RECONNECTING"), Genesis Build Button, Drawer Toggle Button  |
+-------------------------------------------------------------------------------------------------------------+
| OFFLINE BANNER: Displays during WebSocket disconnection / reconnection backoff                              |
+------------------------------------------------------+------------------------------------------------------+
| LEFT COLUMN (800px fixed width)                      | RIGHT COLUMN (Flex 1fr)                              |
| 1. Live Multi-Agent Swarm DAG (SVG 800x400)          | 1. Rationale Deliberation Stream (Live Terminal Log) |
|    - 9 Agent Persona Nodes (Clickable to Inspect)    |    - Token Search, Persona Filter, Type Filter       |
|    - Topological Flow Edges & Feedback Loops         | 2. Active Tool Invocations List (PEV Cycle Cards)    |
|    - Dynamic Selection Halo & Edge Highlighting      | 3. Dynamic Multimodal Asset Explorer & Audio Player  |
| 2. Steam Player DNA & Conceived Blueprint Panel      | 4. Style Bible 16-Color Palette Grid                 |
| 3. Dynamic Agent Telemetry & Hypotheses Panel        | 5. Agent Code Diff Viewer (Unified Python Diff)      |
|    - 640x360 2D Spatial Heatmap Canvas               |                                                      |
|    - Registered Probes Catalog & Metrics Table       |                                                      |
+------------------------------------------------------+------------------------------------------------------+
| FIXED SLIDE-OUT DRAWER (680px width, right-docked):                                                         |
| ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ |
| │ DRAWER HEADER: "Agent Task & Handoff History" | [Tasks Tab] [Requirements Backlog Tab] | [✕ Close]       │ |
| │ CONTROL BAR: [Search Tokens...] [Agent Filter ▾] [Req ID ▾] [Status ▾] [Reset Filters]                  │ |
| │ TAB CONTENT AREA:                                                                                       │ |
| │   - TAB 1: Chronological Task & Handoff Timeline (Expandable Deliberation & Envelope Cards)             │ |
| │   - TAB 2: Live Requirements Backlog View (Tracks R1..Rn, Acceptance Criteria, Live Status Badges)      │ |
| └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ |
+-------------------------------------------------------------------------------------------------------------+
```

---

## 3. Interactive Agent Node Inspector Specification

### 3.1 Node Interaction & Visual States
The 9 personas defined in `engine/studio/orchestrator.py` are mapped to SVG elements in `#agent-dag-canvas`:
- `StudioOrchestrator`
- `SteamAndTelemetryAnalystAgent`
- `CreativeDirectorWorldbuilderAgent`
- `NarrativeCharacterWriterAgent`
- `VoiceActingAudioProducerAgent`
- `VisualArtSpriteDesignerAgent`
- `LeadGameplaySWEAgent`
- `TelemetryArchitectAgent`
- `QAPlaytesterBuildSupervisor`

#### Visual States
1. **Hover State**:
   - Cursor changes to `pointer`.
   - Node brightness increases to `1.25` with cyan glow (`filter: drop-shadow(0 0 8px rgba(41, 173, 255, 0.6))`).
2. **Selected State (`.dag-node.selected`)**:
   - Node border changes to high-intensity gold: `stroke: var(--yellow); stroke-width: 3.5px`.
   - Selected drop-shadow halo: `filter: drop-shadow(0 0 12px rgba(255, 236, 39, 0.85))`.
   - Incident edges (upstream inputs and downstream outputs) highlight with active flow animation.
3. **De-selection**:
   - Clicking the canvas background or clicking the selected node again toggles selection off.

### 3.2 Inspector Actions & Dispatch
Clicking an agent node performs the following coordinated actions:
1. Highlights the SVG node with `.selected`.
2. Updates `localStorage` key `selectedAgent`.
3. Opens the Task & Handoff History Drawer if closed.
4. Switches the Drawer to the **Tasks Tab**.
5. Sets the Drawer's **Agent Filter** dropdown to the clicked agent.
6. Filters the visible task timeline to display only tasks executed by the selected agent.
7. Scrolls the newest task for that agent into view and auto-expands its deliberation block.

---

## 4. Dedicated Task & Handoff History Drawer Specification

### 4.1 Drawer Layout & Animations
- **CSS Specifications**:
  ```css
  #task-history-drawer {
    position: fixed;
    top: 0;
    right: 0;
    width: var(--drawer-width);
    height: 100vh;
    background: #080f24;
    border-left: 2px solid var(--border);
    box-shadow: -12px 0 32px rgba(0, 0, 0, 0.85);
    z-index: var(--z-drawer);
    display: flex;
    flex-direction: column;
    transform: translateX(100%);
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    box-sizing: border-box;
  }
  #task-history-drawer.open {
    transform: translateX(0);
  }
  ```
- **Drawer Backdrop**:
  - A subtle backdrop overlay (`#drawer-backdrop`, `background: rgba(0, 0, 0, 0.55)`, `z-index: var(--z-overlay)`) dims the main dashboard when open, closing the drawer upon click.

### 4.2 Drawer Header & Tab Navigation
- **Header Elements**:
  - Title: `Agent Task & Handoff History` with an animated activity beacon.
  - Quick Metric: `Total Tasks: <N> | Promoted: <M>`.
  - Tab Switcher:
    - Tab A: `Task Timeline (<count>)`
    - Tab B: `Requirements Backlog (<count>)`
  - Close Button: `[✕]` (Escape key closes drawer).

### 4.3 Control & Filtering Bar
The control bar provides client-side query execution:
1. **Search Input (`#drawer-search-input`)**: Real-time debounce (150ms). Matches against task deliberation, title, requirement ID, tool names, and artifact filenames.
2. **Agent Filter Dropdown (`#drawer-agent-filter`)**: Options: `All Agents` + 9 specialist personas.
3. **Requirement Filter Dropdown (`#drawer-req-filter`)**: Options: `All Requirements` + dynamic list of tracks (`R1`, `R2`, ...).
4. **Status Filter Dropdown (`#drawer-status-filter`)**: Options: `All Statuses`, `Promoted / Completed`, `Synthesizing / In Progress`, `Verifying`, `Rejected / Failed`.
5. **Reset Button (`#drawer-btn-reset-filters`)**: Reverts all filters to defaults and restores full view.

---

## 5. Epistemic Task Cards & Cryptographic Handoff Documentation

### 5.1 Card Structure & Field Ordering (Rule 4 Mandatory Compliance)
Every task card rendered in the timeline represents a discrete execution unit. The card strictly adheres to the Rationale-First architecture:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [▼] [SWE] R1: Core Game Loop & Scene Architecture             [COMPLETED] [14,250ms] [10:42:15] │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. EPISTEMIC REASONING (Strictly Index 0):                                                      │
│    > Rationale Deliberation:                                                                   │
│    "Deliberated on SteamPlayerDNA preferences for kinematic action and responsive input.        │
│     Decomposed the architecture into modular state machines: SceneManager, PlayerController,   │
│     and ArenaBounds. Implemented 60 FPS deterministic timestep in mechanics.py."                │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. CRYPTOGRAPHIC HANDOFF ENVELOPE (ArtifactMetadataEnvelope):                                   │
│    Artifact ID: env_e7381f9a-4c21-4890-8432-8df21b06c8a3        Type: GAME_MODULE              │
│    Source: LeadGameplaySWEAgent                               Target: QAPlaytesterSupervisor    │
│    Input SHA256:  a8f9c10243e8... (from CreativeDirector world_spec.json)                      │
│    Output SHA256: d5e28a9b4071... [✓ Cryptographically Verified]                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. MODIFIED ASSETS & PARAMETERS:                                                                │
│    - mechanics.py (14,820 bytes, +412 lines)                                                    │
│    - scenes.py (6,140 bytes, +185 lines)                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. VERIFICATION VERDICT & QUALITY GATES:                                                        │
│    [✓] Gate A: AST Security & Scene Lifecycle (0.8s)                                            │
│    [✓] Gate C: Zero Runtime Exceptions in Headless Simulator (2.1s)                             │
│    [✓] Gate F: Challenger Gate Acceptance Criteria R1 Met (100%)                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Failure & Rejection Rendering (Rule 5 Zero Silent Fallbacks)
When a task or requirement verification fails:
- The card status pill renders `[REJECTED]` in flashing `--neon-red`.
- Card border highlights with a 2px solid `--neon-red` accent.
- Epistemic deliberation highlights the failure rationale.
- The **Diagnostic Trace Box** renders:
  - `failure_gate`: e.g. `Gate F (Adversarial Design Critic)` or `Gate A (Syntax AST)`
  - `failure_type`: e.g. `PROGRESSION_FAILURE`, `ASSET_INTEGRITY_VIOLATION`
  - `failure_reason`: Verbatim failure message with file path and line number
  - `runtime_traceback`: Complete unmasked Python traceback in a monospace scrollbox
  - `remedy_type`: Recommended repair action (e.g. `repair_code_exception`, `fix_syntax`)

---

## 6. Live Requirements Backlog View Specification

### 6.1 Backlog Data Model
The backlog renders the agent-authored `GameRequirementsSpec` persisted at runtime:
- `game_title`: Conceived game title.
- `genre_paradigm`: Conceived genre blend (e.g. `Tactical Orbit Strategy / Physics Sim`).
- `target_audience_summary`: Alignment analysis with `SteamPlayerDNA`.
- `requirements`: Ordered list of `RequirementTrack` items (`R1`, `R2`, ...).

### 6.2 Backlog Board Layout
Inside Tab 2 of the Drawer, the backlog renders:
1. **Overview Header**:
   - Title, Conceived Genre Chip, DNA Match Indicator.
   - Progress Summary: Horizontal progress bar showing percentage of completed criteria.
2. **Requirement Track Cards**:
   - Header: Track ID (e.g. `R1`), Track Title, Responsible Agent badge.
   - Status Badge:
     - `PENDING`: Gray badge, dimmed border.
     - `IN_PROGRESS`: Amber badge with continuous CSS pulse animation.
     - `COMPLETED`: Green badge with checkmark icon.
     - `REJECTED`: Neon-red badge with warning icon.
   - Architectural Deliberation: Epistemic justification why this track is required for game completeness.
   - **Acceptance Criteria Checklist**:
     - Each criterion (`AC1.1`, `AC1.2`, ...) renders with:
       - Criterion ID and description.
       - Verification method badge: `ast_inspection` | `runtime_telemetry` | `headless_playtest` | `multi_modal`.
       - Checkbox indicator: `[✓] Satisfied` (green), `[○] Pending` (slate), `[✗] Rejected` (red).
3. **Bidirectional Navigation**:
   - Clicking `[Inspect Handoff Tasks for R1]` switches to Tab 1 with `Requirement ID: R1` pre-selected, immediately revealing the exact code synthesis, asset generation, and Challenger gate evaluation tasks for R1.

---

## 7. Reload & Session Persistence Specification

### 7.1 Schema: `studio_observability_state_v1`
All ephemeral UI state is synchronized with `window.localStorage` under the key `studio_observability_state_v1`:

```typescript
interface StudioObservabilityState {
  version: 1;
  drawerOpen: boolean;
  activeTab: "tasks" | "requirements";
  selectedAgent: string | null;
  filters: {
    agent: string;          // "" for all, or agent name
    requirementId: string;  // "" for all, or "R1", "R2", etc.
    status: string;         // "" for all, or "completed", etc.
    searchQuery: string;    // free-text search token
  };
  expandedCardIds: string[]; // List of open task card IDs
  cachedRequirementsSpec: Record<string, any> | null;
  cachedTasks: Array<Record<string, any>>;
  lastUpdatedUtc: string;
}
```

### 7.2 Persistence Lifecycle
1. **Save Triggers**:
   - Debounced (250ms) on user filter changes or search keystrokes.
   - Immediate on drawer toggle, tab switch, card expand/collapse, or SVG node click.
   - Immediate on receiving new task handoffs or requirements updates over WebSocket.
2. **Restore Sequence on Page Load (`DOMContentLoaded`)**:
   - Check `localStorage.getItem('studio_observability_state_v1')`.
   - If present:
     - Restore `filters` to input elements and trigger client-side filter computation.
     - If `drawerOpen` is true, add `.open` class immediately to `#task-history-drawer`.
     - Set active tab (`tasks` or `requirements`).
     - Highlight `selectedAgent` on the SVG DAG.
     - Pre-render `cachedTasks` and `cachedRequirementsSpec` to prevent layout flash.
   - Initiate WebSocket connection and fetch `/api/events/history` and `/api/requirements` to merge fresh updates.

---

## 8. Backend Interface Specifications (`server.py` & `models.py`)

### 8.1 Data Models (`engine/ui/models.py`)
All models enforce Rule 4: `deliberation: str` is declared at field index 0.

#### `AgentTaskRecord`
```python
class AgentTaskRecord(BaseModel):
    """Execution history record for an agent task or requirement track."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale justifying task execution")
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    agent_name: str = Field(..., description="Agent executing the task")
    requirement_id: Optional[str] = Field(default=None, description="Associated requirement track (e.g. R1)")
    title: str = Field(..., description="Task title or step description")
    status: str = Field(default="completed", description="Status: synthesizing, verifying, completed, rejected")
    started_at_utc: str = Field(...)
    completed_at_utc: Optional[str] = Field(default=None)
    duration_ms: float = Field(default=0.0)
    envelope: Optional[ArtifactMetadataEnvelope] = Field(default=None, description="Cryptographically sealed envelope")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    gate_results: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostic_trace: Optional[Dict[str, Any]] = Field(default=None)
```

#### `AgentTaskHistoryResponse`
```python
class AgentTaskHistoryResponse(BaseModel):
    """Response containing chronological agent task executions and envelopes."""
    deliberation: str = Field(..., min_length=1, description="Epistemic summary of retrieved task history")
    total_count: int = Field(default=0)
    tasks: List[AgentTaskRecord] = Field(default_factory=list)
    agent_filter: Optional[str] = Field(default=None)
    requirement_filter: Optional[str] = Field(default=None)
```

#### `RequirementsSpecResponse`
```python
class RequirementsSpecResponse(BaseModel):
    """Response containing the active GameRequirementsSpec and track statuses."""
    deliberation: str = Field(..., min_length=1, description="Epistemic summary of active game requirements")
    exists: bool = Field(default=False)
    spec: Optional[Dict[str, Any]] = Field(default=None)
    total_requirements: int = Field(default=0)
    completed_requirements: int = Field(default=0)
    overall_progress: float = Field(default=0.0)
```

### 8.2 Endpoints in `engine/ui/server.py`
1. `GET /api/agents/history`:
   - Query Parameters: `agent` (optional), `requirement_id` (optional), `status` (optional), `limit` (default 200).
   - Reads historical task records directly from SQLite WAL (`agent_task_records` and `studio_events` in `data/telemetry.db`) and `SwarmEventBus`.
   - Hydrates tasks with sealed `ArtifactMetadataEnvelope` instances from SQLite WAL or `cartridges/active/`.
2. `GET /api/requirements`:
   - Queries `studio_requirements` and `studio_state_kv` in SQLite WAL (`data/telemetry.db`) or `cartridges/active/requirements_spec.json`.
   - Returns serialized `GameRequirementsSpec` with computed progress metrics.

---

## 9. Verification & Automated Test Specification

### 9.1 Unit & Contract Tests (`tests/test_ui_observability_contracts.py`)
1. **Schema AST & Index 0 Verification**:
   - Inspect `AgentTaskRecord`, `AgentTaskHistoryResponse`, and `RequirementsSpecResponse`.
   - Assert `deliberation` is declared at index 0 on all models.
   - Assert empty deliberation raises `ValidationError`.
2. **ArtifactMetadataEnvelope Cryptographic Integrity**:
   - Verify `compute_sha256_bytes` matches authentic hex digests for payloads.
   - Verify tampering with payload causes verification failure in envelope.

### 9.2 API & Server Integration Tests (`tests/test_ui_server_history.py`)
1. **`GET /api/agents/history`**:
   - Verify endpoint returns HTTP 200 with `deliberation` first.
   - Verify query filtering by `agent` and `requirement_id`.
2. **`GET /api/requirements`**:
   - Verify endpoint loads and parses `requirements_spec.json` correctly.

### 9.3 Frontend DOM & Reload E2E Tests (`tests/test_ui_frontend_e2e.py`)
1. **Node Click Interaction**:
   - Verify clicking `.dag-node` applies `.selected` class and opens Drawer with matching agent filter.
2. **Rationale-First Card Hierarchy**:
   - Verify `.task-card .deliberation-block` is the first child element of `.task-card-body`.
3. **LocalStorage Persistence**:
   - Verify state stored in `localStorage` restores filter inputs and drawer open state upon page reload.

---

## 10. Feature 20: Verified Checkpoints & Live Playable Progression Architecture

### 10.1 Overview & Motivation
During long-running autonomous game genesis builds, players and developers need the ability to inspect and immediately play intermediate verified milestones as the swarm progresses through requirement tracks ($R_1, R_2, \dots, R_N$). Feature 20 introduces an immutable verified checkpointing pipeline, live atomic cartridge promotion, and full Studio Observability UI support.

### 10.2 Architectural Guarantees
1. **Immutable Snapshotting**:
   - Upon successful headless smoke verification of any requirement track ($R_k$) in `StudioOrchestrator._execute_iterative_swe_loop`, an immutable snapshot is written to `cartridges/checkpoints/<candidate_id>_<track_id>/`.
   - The checkpoint includes all `.py` modules, assets, specifications (`manifest.json`, `world_spec.json`, `requirements_spec.json`, `narrative_bible.json`, `narrative.json`, `level.json`), `task_history.json`, and `checkpoint_metadata.json` containing SHA-256 digests.
2. **Atomic Live Mirroring**:
   - The verified checkpoint directory is atomically mirrored to `cartridges/active/` via temporary staging and atomic directory replacement (`rename`).
   - Running `./run_game.sh` or `python3 run_game.py` at any point during an ongoing Genesis build boots the latest verified checkpoint without interrupting the active swarm.
3. **Rationale-First Epistemic Data Models (Rule 4)**:
   - Models declared in `engine/ui/models.py` place `deliberation: str` strictly at field index 0 with non-empty validation:
     - `VerifiedCheckpointRecord`: Represents metadata, track title, verified track IDs, filesystem path, active status, file inventory, and SHA-256 manifest.
     - `CheckpointsResponse`: Envelopes the list of available verified checkpoints with total counts and latest checkpoint ID.
4. **Zero Silent Fallback REST Endpoints (`engine/ui/server.py`)**:
   - `GET /api/checkpoints`: Discovers and returns all verified checkpoints sorted chronologically.
   - `POST /api/checkpoints/{checkpoint_id}/activate`: Atomically copies the selected verified checkpoint to `cartridges/active/`, returning 404/500 with unmasked error details if resolution fails.
5. **Real-time Observability & UI Components (`engine/ui/static/index.html`)**:
   - **Header Checkpoint Badge**: Displays the last verified checkpoint ID, track title, and quick-play instructions.
   - **Drawer Tab 3 ("Checkpoints")**: Dedicated drawer panel showing the Checkpoint Progression Timeline with requirement badges, verification timestamps, file inventories, SHA-256 manifests, and one-click "Activate / Play" buttons.
   - **WebSocket Sync**: Responds to `CHECKPOINT_VERIFIED` events broadcast over WebSocket, triggering automatic UI refreshes without requiring page reloads.

---

## 11. Decoupled Blind Playtester Studio, Session History Archive & Replay Theatre

### 11.1 Overview, Zero-Leak Player Isolation & Skeptical Council Audit
Mission Control includes a dedicated **🧪 Blind Playtester** workspace powered by `engine/playtest/` (`BlindPlaytesterAgent`, `GodotPlaytestBridge`, `PlaytestSessionManager`). Unlike internal QA smoke/filmstrip harnesses, the Blind Playtester experiences verified checkpoints strictly as a cold external human player—receiving **only** a Steam-style `BlindStoreCard` (`game_title` + short pitch), live 1280×720 viewport screenshots (with `0..1000` normalized screen-coordinate smooth cursor pointing and desktop pointer arrow parity), and captured in-game audio clips (plus optional on-demand `2..20s` rolling gameplay `.mp4` soundscape evaluations via `PlaytestAudioEvaluationRequest` and `AudioEvaluatorAgent` with `curate_studio_context=False`), with zero access to `.gd` source code, `GameBibleSpec`, `backlog.json`, or internal state dictionaries.

Sessions distinguish provenance via `initiated_by` (`"operator"` vs `"council"`), rendered in the session header and archive cards as **`[👤 Manual]`** or **`[🏛️ Council]`**:
- **Manual Operator Runs (`initiated_by="operator"`)**: Persist sparse 6-pillar impressions, player field notes, open/answered questions, live bug reports, the end-of-session `PlaytestExperienceReport`, and the `1600×720` side-by-side broadcast MP4 (`replay.mp4`) for Mission Control inspection while keeping generated `PlaytestWishDraft` items in `status="pending"` until the operator explicitly forwards selected wishes (`POST /api/playtest/wishes/send-to-council`) or dispatches the **Full Playtest Experience Report** (`POST /api/playtest/sessions/{session_id}/send-report-to-council`).
- **Automated End-of-Slice Council Runs (`initiated_by="council"`)**: Invoked automatically by `CreativeWritersRoom._run_end_of_slice_blind_playtest` after `_run_slice_assembly_check` passes; marks all generated wishes `status="sent_to_council"` and delivers the `PlaytestExperienceReport`, 6-pillar impressions, field notes, questions, bug reports, rulebook, chronological player action tape, and up to 5 evenly-spaced session screenshots (`first -> quarter -> mid -> three-quarter -> final`) to the Council for a skeptical anti-auto-play / anti-noop-loop audit (`BlindPlaytestCouncilVerdict`: `iterate_current_slice` vs `approve_slice`). Every pass writes an immutable `artifacts/blind_playtest_report_<slice_id>_pass<k>.json` snapshot and stamps `council_review` onto the archived playtest session (`record_council_verdict`).

### 11.2 Playtest Lab UX, DVR Replay-to-Live Stage & Structured Player Notebook (`studio-web/src/features/playtest/PlaytestLabView.tsx`, `studio-web/src/domain/playtestNotebook.ts`)
1. **Per-Game & Per-Slice Session Archive Strip**:
   - Every playtest session is indexed per cartridge and slice (`list_enriched_sessions` with `game_run_number`, `slice_run_number`, `report_sent_to_council_at_utc`, and `games_catalog`) and snapshotted into `<cartridge>/artifacts/playtest_sessions/<session_id>.json` + `session_archive.json` (with automatic SQLite recovery if the database is reset).
   - Filter pills filter runs by **Selected Game vs All Games**, **Slice (`All Slices`, `Slice #N`, `Active / General`)**, and **Source (`All Runs`, `Council Gates`, `Manual`)**, rendering horizontal session cards with provenance badges, turn counts, fun ratings (`★X/10`), and `Sent to Council` indicators.
   - **Slice Subbar Shortcuts (`SliceSubbar.tsx`)**: Each slice pill surfaces recorded playtest counts/ratings and deep-links into the Playtest Lab filtered to that slice.
2. **Current Playtester Focus Banner (`aria-label="Current playtester focus"`)**:
   - Front-and-center status strip showing what the playtester is actively testing right now (`current_plan` hypothesis headline, formatted motor action badges, clean 1-sentence turn summary, and live vs scrubbed `Turn #k` context), plus a 1-click **`Review & Send Report`** jump button when a manual run has an experience report ready for Council dispatch.
3. **DVR "Replay Up to Live" Viewport & Transport Bar**:
   - **Live Screen**: Real-time `1280×720` viewport stream (`followingLive = true`).
   - **Turn-by-Turn Replay (`1280×720`)**: Frame-by-frame player rendering each turn's raw screenshot (`screenshot_url`) and precision `1280×720` SVG reticle overlay (`action_px`) visualizing clicks (`CLICK` / `DBL-CLICK`), cursor trajectories, drag vectors, and key actions.
   - **DVR Scrubber & Auto-Playback**: Interactive `<input type="range" aria-label="Turn replay scrubber">`, `⏮`, `◀ Prev`, `Next ▶`, `Replay 1→Live` / `Play → Live` / `⏸ Pause` auto-player (`0.5×`, `1×`, `2×`), and **`Jump to Live`** button. Scrubbing backward detaches `followingLive` so incoming live turns never yank the operator out of historical inspection; once auto-playback catches up to the latest live turn on an active session, it transitions seamlessly back to `Live Screen`.
   - **Broadcast Video MP4 (`1600×720`)**: Synchronized side-by-side HUD + audio MP4 player (`replay.mp4`).
4. **4-Tab Right-Hand Observability Panel**:
   - **Activity Log (`Latest First` / `Oldest First`)**: Concise chronological/reverse-chronological action cards showing turn number, tick, mood shift badge, motor action summary, and 1-sentence intent/observation, while tucking full epistemic `deliberation`, raw prompt tape, rulebook deltas, and bug/question updates inside native `<details>` disclosures (`Full turn output & deliberation`). Clicking any card seeks the replay stage to that turn.
   - **Experience Report & Full Council Dispatch**: Renders the Council Gate Verdict banner (`council_review`), 6-pillar player impression cards, `PlaytestExperienceReport` scorecard (`overall_fun_rating` `1..10`, highlights, friction points, player recommendations), and—for manual runs—a semantic `<form>` (**`Dispatch Full Playtest Report to Council`**) with optional Executive Producer direction, a checkbox to bundle pending session wishes (`include_pending_wishes`), `Sent ✓` timestamp state, and one-click re-send support.
   - **Rulebook & Notes (`playtestNotebook.ts`)**: Organizes player memory into five distinct sections: **Bugs & Glitches Spotted** (severity badges + repro context), **Inferred Player Rulebook** (humanized control/mechanic labels with duplicate control Q&A suppressed), **Open Questions & Mysteries**, **Answered Questions** (collapsed `<details>` cards pairing the clean player question with `✓` empirical answer while hiding internal `what_does_*` `snake_case` keys), and **Screen Field Notes**.
   - **Wishes → Council**: Individual `PlaytestWishDraft` triage cards with priority/category pills, dismiss controls, and **Send Selected to Council** batch forwarding.
5. **Lifecycle & Protected Historical Reset Controls**:
   - **Start / Pause / Resume / Stop**: Starting a new run preserves all prior runs in the session archive; stopping finalizes `PlaytestExperienceReport`, updates `session_archive.json`, and muxes (or recovers via `_recover_unmuxed_session_video`) `replay.mp4`.
   - **Protected Reset (`action="reset"`)**: Resetting an actively `running` or `paused` session discards in-flight artifacts and restarts from Turn 0; resetting a recorded historical (`completed` / `stopped` / `failed`) session requires explicit confirmation (`confirm_delete=True`) before deleting that run from SQLite and `session_archive.json`.

### 11.3 SQLite WAL Persistence & REST Endpoints (`engine/telemetry/schema.sql`, `engine/ui/server.py`)
1. **SQLite WAL Tables (`data/telemetry.db`)**:
   - `playtest_sessions`: Session lifecycle (`idle`, `starting`, `running`, `paused`, `stopping`, `stopped`, `completed`, `failed`), `initiated_by` (`operator` | `council`), checkpoint/slice metadata, emergent `rulebook_json`, sparse 6-pillar `impressions_json`, persistent `field_notes_json`, `questions_json`, `bugs_found_json`, end-of-session `session_report_json` (`PlaytestExperienceReport`), `compacted_journal_json`, token usage, and composited MP4 `video_path`.
   - `playtest_turns`: Chronological rationale-first (`deliberation` first) turn records storing cursor trajectories, `action_json` (enriched at API read time with `action_px` in `1280×720` coordinates), `rulebook_updates_json`, `impression_updates_json`, `field_note_updates_json`, `question_updates_json`, `bug_reports_json`, sparse `mood_state` / `mood_note`, `screenshot_path`, and `audio_clip_path`.
   - `playtest_compactions`: Periodic long-horizon memory compactions (`PlaytestCompactedJournal`) preserving recent raw turns while carrying `rulebook_json`, `impressions_json`, `field_notes_json`, `questions_json`, and `bugs_found_json` forward without breaking byte-0 prefix caching.
   - `playtest_wishes`: Structured player-perspective feedback items (`PlaytestWishDraft`) with `status` (`pending`, `sent_to_council`, `dismissed`).
2. **REST API Endpoints**:
   - `GET /api/playtest/sessions` & `POST /api/playtest/sessions/start`: Return enriched session history (`sessions` with `game_run_number`, `slice_run_number`, `has_video`, `has_turn_frames`, `council_review`, `report_sent_to_council_at_utc`, and `games_catalog`, optional `?candidate_id=...&slice_id=...` filters) or launch a new blind playtest against any verified checkpoint.
   - `GET /api/playtest/sessions/{session_id}` & `POST /api/playtest/sessions/{session_id}/control`: Inspect full enriched turn timeline (`action_px`, `screenshot_url`, `audio_url`), notes/questions/bugs, rulebook, impressions, experience report, and `council_review`, or `pause`, `resume`, `stop`, or `reset` (`confirm_delete` guarded for historical runs).
   - `GET /api/playtest/sessions/{session_id}/video` & `GET /api/playtest/sessions/{session_id}/media/{filename}`: Stream the composited `1600×720` MP4 session broadcast (recovering unmuxed frames on demand if interrupted) and per-turn viewport/audio captures.
   - `GET /api/playtest/wishes`, `POST /api/playtest/wishes/send-to-council`, & `POST /api/playtest/wishes/{wish_id}/dismiss`: Triage individual player wishes into Writers' Room steering.
   - `POST /api/playtest/sessions/{session_id}/send-report-to-council`: Dispatches a manual session's full `PlaytestExperienceReport`, 6-pillar impressions, bugs, open/answered questions, field notes, rulebook, optional pending wishes, optional Producer note, and up to 5 evenly-spaced player proof screenshots (`artifacts/proofs/manual_playtest_*.png`) into `.pending_steering.json` and the Writers' Room transcript, stamping `report_sent_to_council_at_utc` and waking `StudioDaemon(resume_current=True)` if idle.

---

## 12. Per-Game Multi-Bank Long-Term Memory Workspace (`studio-web/src/features/memory/GameMemoryView.tsx`, `engine/ui/server.py`)

### 12.1 Overview & Multi-Bank Ownership
The Mission Control **Memory** workspace provides live inspection and Executive Producer lifecycle control over each game's persistent cross-slice memory ledger (`<cartridge_dir>/game_memory.json`, mirrored to `studio_game_memories` in SQLite WAL).

Memories are organized into 17 canonical banks (`CANONICAL_MEMORY_BANKS`):
- **Shared Game Bank**: `council` (game-wide deferred strategic goals, cross-cutting architectural invariants, and studio-wide lessons).
- **Council Specialist Banks (10)**: `CreativeDirector`, `GameplayDesigner`, `LevelAndProgressionDesigner`, `GameplaySystemsEngineer`, `StoryWorldArchitect`, `ScriptwriterAgent`, `CharacterDesigner`, `VisualArtDirector`, `UXExperienceDesigner`, `AudioAtmosphereDesigner`.
- **Production Specialist Banks (6)**: `task_lead`, `swe_engineer`, `qa_test_engineer`, `visual_artist`, `audio_designer`, `script_writer`.

### 12.2 UI Architecture & Accessibility (`GameMemoryView.tsx`)
1. **Multi-Bank Navigation Sidebar (`.memory-bank-sidebar`)**: Displays live active/total badge counts across `All Banks`, `Council Shared`, Council Specialists, and Production Specialists using `aria-pressed` toggle buttons.
2. **Lifecycle & Category Toolbar (`.memory-filter-bar`)**: Filters entries by status (`active`, `fulfilled`, `superseded`, `archived`, `all`) and category (`strategic_goal`, `architectural_decision`, `lesson_learned`, `domain_note`).
3. **Container-Responsive Semantic Memory Cards (`article.memory-card`)**:
   - Uses `<article className="memory-card">` (`container-type: inline-size`) with direct `<header className="memory-card-header">`, `<div className="memory-card-content">`, and `<footer className="memory-card-footer">` child regions, expandable `DeliberationDisclosure` rationale, provenance metadata (`slice_id`, `item_id`, `author_role`), and `Mark Fulfilled` / `Archive` / `Restore Active` lifecycle controls.
4. **Deferred Post-Interaction Composer (`.memory-composer-form`)**:
   - Allows the Executive Producer to seed durable memories into any bank with optional proactive curation `applicable_roles` checkboxes, hints positioned above controls (`aria-describedby`), and post-interaction validation (`:is(:user-invalid, .user-invalid-fallback)` + `aria-invalid` synchronization) that defers error states until blur or submit.

### 12.3 REST & WebSocket Synchronization (`engine/ui/server.py`, `useSwarmSocket.ts`)
1. `GET /api/studio/memories`: Supports `?cartridge=...&bank=...&kind=...&status=...` plus optional semantic search `?q=...` powered by `gemini-embedding-001` (failing fast with HTTP `503` when `GEMINI_API_KEY` is unconfigured rather than falling back to substring matching).
2. `POST /api/studio/memories`: Applies Rationale-First (`deliberation` first) `MemoryMutationProposal` operations (`record`, `update`, `fulfill`, `supersede`, `archive`), persisting to `game_memory.json` and `studio_game_memories` and broadcasting `game_memory_updated` over `/ws/swarm` to invalidate `['game-memories']` queries in real time.

---

## 13. Modular React Mission Control Shell, Reconnect Reconciliation & Post-Board Phase Observability (`studio-web/`)

### 13.1 Workspace Shell & Navigation (`App.tsx`, `TopCommandBar.tsx`, `SliceSubbar.tsx`)
Mission Control (`studio-web/src/`) unifies studio operations across 7 workspace tabs (**Production Board**, **Writers' Room**, **Game Bible**, **Game Memory**, **Assets & Code**, **Playtest Lab**, and **Player DNA**) beneath a global command header (`TopCommandBar.tsx`), per-slice progress subbar (`SliceSubbar.tsx`), and live studio phase strip (`LiveActivityBanner.tsx`).

### 13.2 Immediate WebSocket Reconnect Reconciliation (`useSwarmSocket.ts`)
- Whenever `/ws/swarm` disconnects (e.g., during a Uvicorn `--reload` restart or daemon stop) and subsequently reconnects, `useSwarmSocket.ts` fires `onReconnect` to immediately invalidate and refetch `['status']`, `['studio-status']`, `['production-board']`, and `['writers-room']` without waiting for background polling intervals—ensuring the header immediately transitions from `Building` to `Stopped` and surfaces the **`Resume`** action button without requiring a manual browser refresh.

### 13.3 Live Post-Board Phase & AI Call Cost Observability (`LiveActivityBanner.tsx`, `liveActivity.ts`)
- `/api/studio/status` exposes `board_activity`, `writers_room_activity`, `agent_activity`, and `token_usage` (`TokenUsageSummary`) so `LiveActivityBanner.tsx` continuously displays the exact phase, specialist, and live `llm_gateway_call` telemetry (total tokens, thinking tokens, cached prefix tokens, synthesized image count, per-call estimated USD cost, and session cache-hit percentage) across board execution, **Slice Assembly Check**, **End-of-Slice Blind Playtest**, and **Slice Reflection & Strategy**.
- Continuous Production Board builds use autonomous agent verification (`is_agent_verified_board_build` in `StudioOrchestrator.execute_swarm_plan`), checking headless Godot boot syntax and escalating any boot syntax error directly to the Writers' Room Council rather than running hardcoded post-slice or post-build 5-Gate restrictions.

### 13.4 Production Board Working Lifecycle, Split Tasks & Live Proof Showcase (`ProductionBoardView.tsx`, `WorkItemCard.tsx`, `TaskChronologyDrawer.tsx`, `boardCards.ts`, `liveActivity.ts`)
- **Strict `Working` vs `Queued` Lifecycle & Decomposed `Split` Badges**: `isWorkingBoardItem` (`boardCards.ts`) groups only tasks that have been picked up and are not yet approved (`in_progress`, `waiting_for_lock`, `awaiting_review`, `changes_requested`, `failed`, `needs_decision`, and Council-reviewed `blocked` tasks waiting on newly spawned prerequisites), while `getStateBadge` (`WorkItemCard.tsx`, `TaskChronologyDrawer.tsx`) distinguishes Stage 1 Council-decomposed parent tasks (`decision_reason` indicating split) from cancelled items. Unstarted `ready` items—even if they hold interrupted prior-attempt counters before Council submission—remain strictly under **`Queued`** (`ready`), and `ProductionBoard.requeue_failed` preserves `changes_requested` whenever Council revision feedback already exists.
- **Scoped Live Step Counts & Cache-Busted Mid-Flight Proof Showcases**: `deriveLiveStudioActivity` (`liveActivity.ts`) filters live step entries strictly to active working board item IDs, while `WorkItemCard.tsx` fetches live chronology across `in_progress`, `waiting_for_lock`, and `changes_requested` and resolves card previews via `resolveCardProofShowcase`—skipping deleted/unlinked QA retry filmstrips and appending `&v=` cache-busting tokens so newly captured revision proofs render immediately.






