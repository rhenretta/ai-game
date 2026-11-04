import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DeliberationDisclosure } from '../src/components/DeliberationDisclosure';
import { ErrorBanner } from '../src/components/ErrorBanner';
import { MediaLightbox } from '../src/components/MediaLightbox';
import { ModalDialog } from '../src/components/ModalDialog';
import { ProductionBoardView } from '../src/features/board/ProductionBoardView';
import { PlayerDnaView } from '../src/features/dna/PlayerDnaView';
import { PlaytestLabView } from '../src/features/playtest/PlaytestLabView';
import { TopCommandBar } from '../src/features/shell/TopCommandBar';
import type {
  BoardItemChronologyResponse,
  GameBibleResponse,
  ProductionBoardItem,
} from '../src/api/types';

function renderWithQueryClient(ui: React.ReactElement) {
  const qc = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

describe('UI components', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders DeliberationDisclosure only when non-empty rationale exists', () => {
    const { rerender } = render(<DeliberationDisclosure deliberation="   " />);
    expect(screen.queryByText('Epistemic Rationale')).not.toBeInTheDocument();

    rerender(<DeliberationDisclosure deliberation="Evaluated frame pacing across 120 ticks." defaultOpen />);
    expect(screen.getByText('Epistemic Rationale')).toBeInTheDocument();
    expect(screen.getByText('Evaluated frame pacing across 120 ticks.')).toBeInTheDocument();
  });

  it('opens ModalDialog with closedby="any" and triggers light-dismiss fallback on outside click', () => {
    const onClose = vi.fn();
    render(
      <ModalDialog open title="Synthetic Modal" subtitle="Subheading" onClose={onClose}>
        <p>Modal body content</p>
      </ModalDialog>,
    );

    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('closedby', 'any');
    expect(screen.getByText('Modal body content')).toBeInTheDocument();

    vi.spyOn(dialog, 'getBoundingClientRect').mockReturnValue({
      top: 100,
      left: 100,
      width: 400,
      height: 300,
      bottom: 400,
      right: 500,
      x: 100,
      y: 100,
      toJSON: () => ({}),
    });

    // Click inside dialog bounds -> should NOT close
    fireEvent.click(dialog, { clientX: 200, clientY: 200 });
    expect(onClose).not.toHaveBeenCalled();

    // Click outside dialog bounds on backdrop -> closes via fallback
    fireEvent.click(dialog, { clientX: 20, clientY: 20 });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('filters ProductionBoardView cards and opens TaskChronologyDrawer on selection', async () => {
    const mockChronology: BoardItemChronologyResponse = {
      deliberation: 'Fetched synthetic chronology.',
      cartridge_id: 'candidate_synthetic_01',
      item_id: 'task_alpha',
      title: 'Implement deterministic grid solver',
      kind: 'feature_code',
      state: 'approved',
      spec: 'Core solver logic.',
      proof_spec: 'Zero runtime errors',
      target_modules: ['scripts/solver.gd'],
      depends_on: [],
      attempts: [
        {
          attempt_key: 'rev_0_att_1',
          revision_index: 0,
          attempt_number: 1,
          is_failed_retry: false,
          status: 'approved',
          current_strategy: 'Deterministic row-major sweep.',
          turns: [
            {
              turn_index: 1,
              specialist_agent: 'Gameplay Engineer',
              deliberation: 'Implemented pure step function.',
              summary: 'Wrote solver module and unit tests.',
              verification_passed: true,
              touched_files: ['scripts/solver.gd'],
            },
          ],
          council_review: {
            deliberation: 'Checked all boundary coordinates.',
            approved: true,
            summary: 'Approved cleanly.',
          },
        },
      ],
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockChronology), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    const items: ProductionBoardItem[] = [
      {
        deliberation: 'Solver item deliberation.',
        item_id: 'task_alpha',
        title: 'Implement deterministic grid solver',
        kind: 'feature_code',
        spec: 'Core solver logic.',
        proof_spec: 'Zero runtime errors',
        target_modules: ['scripts/solver.gd'],
        depends_on: [],
        state: 'approved',
        attempt_count: 1,
      },
      {
        deliberation: 'Audio item deliberation.',
        item_id: 'task_beta',
        title: 'Synthesize ambient wind loop',
        kind: 'audio_sfx',
        spec: 'Loopable ambient bed.',
        proof_spec: 'Seamless loop',
        target_modules: ['assets/audio/ambient_loop.wav'],
        depends_on: [],
        state: 'in_progress',
        raised_by_role: 'Audio Director',
        attempt_count: 1,
      },
      {
        deliberation: 'Queued item deliberation.',
        item_id: 'task_gamma',
        title: 'Build queued telemetry aggregator',
        kind: 'mechanic_code',
        spec: 'Aggregate tick counters.',
        proof_spec: 'Deterministic summary',
        target_modules: ['scripts/aggregator.gd'],
        depends_on: [],
        state: 'ready',
        attempt_count: 1,
      },
    ];

    renderWithQueryClient(
      <ProductionBoardView
        items={items}
        cartridgeId="candidate_synthetic_01"
        liveStepsByItem={{}}
        onOpenLightbox={vi.fn()}
      />,
    );

    expect(screen.getByText('Implement deterministic grid solver')).toBeInTheDocument();
    expect(screen.getByText('Synthesize ambient wind loop')).toBeInTheDocument();
    expect(screen.getByText('Build queued telemetry aggregator')).toBeInTheDocument();

    // Filter to Working only: excludes `approved` AND excludes `ready` (even when attempt_count > 0)
    const workingBtn = screen.getByRole('button', { name: /Working/i });
    expect(workingBtn).toHaveTextContent('1');
    fireEvent.click(workingBtn);
    expect(screen.queryByText('Implement deterministic grid solver')).not.toBeInTheDocument();
    expect(screen.queryByText('Build queued telemetry aggregator')).not.toBeInTheDocument();
    expect(screen.getByText('Synthesize ambient wind loop')).toBeInTheDocument();

    // Filter to Queued (`ready`): shows only the ready item
    const queuedBtn = screen.getByRole('button', { name: /Queued/i });
    expect(queuedBtn).toHaveTextContent('1');
    fireEvent.click(queuedBtn);
    expect(screen.getByText('Build queued telemetry aggregator')).toBeInTheDocument();
    expect(screen.queryByText('Synthesize ambient wind loop')).not.toBeInTheDocument();

    // Switch back to All and open TaskChronologyDrawer for task_alpha
    fireEvent.click(screen.getByRole('button', { name: /^All/i }));
    fireEvent.click(screen.getByText('Implement deterministic grid solver'));

    await waitFor(() => {
      expect(screen.getByRole('complementary', { name: /Task Chronology Inspector/i })).toBeInTheDocument();
      expect(screen.getByText('Checked all boundary coordinates.')).toBeInTheDocument();
    });
  });

  it('renders a labeled Dismiss button on ErrorBanner and invokes onDismiss', () => {
    const onDismiss = vi.fn();
    render(
      <ErrorBanner
        error={new Error('Synthetic circuit breaker failure')}
        title="Live Studio Circuit Breaker / Build Error"
        onDismiss={onDismiss}
      />,
    );

    expect(screen.getByText('Live Studio Circuit Breaker / Build Error')).toBeInTheDocument();
    expect(screen.getByText('Synthetic circuit breaker failure')).toBeInTheDocument();
    const dismissBtn = screen.getByRole('button', { name: /Dismiss error/i });
    expect(dismissBtn).toBeInTheDocument();
    fireEvent.click(dismissBtn);
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it('treats blocked DAG tasks as Waiting on Deps rather than Needs Attention', () => {
    const baseItems: ProductionBoardItem[] = [
      {
        deliberation: 'Upstream foundation module.',
        item_id: 'task_upstream',
        title: 'Build upstream state store',
        kind: 'feature_code',
        spec: 'Implement core store.',
        proof_spec: 'Unit test passes',
        target_modules: ['scripts/store.gd'],
        depends_on: [],
        state: 'in_progress',
      },
      {
        deliberation: 'Downstream consumer waiting on store.',
        item_id: 'task_downstream',
        title: 'Bind HUD readout to state store',
        kind: 'ui',
        spec: 'Subscribe to store signals.',
        proof_spec: 'HUD updates on tick',
        target_modules: ['scripts/hud.gd'],
        depends_on: ['task_upstream'],
        state: 'blocked',
      },
    ];

    const { rerender } = renderWithQueryClient(
      <ProductionBoardView
        items={baseItems}
        cartridgeId="candidate_synthetic_02"
        liveStepsByItem={{}}
        onOpenLightbox={vi.fn()}
      />,
    );

    // Needs Attention pill should NOT appear when items are merely blocked on DAG dependencies
    expect(screen.queryByRole('button', { name: /Needs Attention/i })).not.toBeInTheDocument();

    // Waiting on Deps pill appears and filters to the blocked downstream task
    const waitingPill = screen.getByRole('button', { name: /Waiting on Deps/i });
    expect(waitingPill).toBeInTheDocument();
    fireEvent.click(waitingPill);

    expect(screen.queryByText('Build upstream state store')).not.toBeInTheDocument();
    expect(screen.getByText('Bind HUD readout to state store')).toBeInTheDocument();
    expect(screen.getByText(/Depends on:/i)).toBeInTheDocument();
    expect(screen.getByText('task_upstream')).toBeInTheDocument();

    // Verify the card badge does not carry error styling
    const cardBadges = screen.getAllByText('Waiting on Deps');
    const badgeSpan = cardBadges.find((el) => el.classList.contains('badge'));
    expect(badgeSpan).toBeDefined();
    expect(badgeSpan).not.toHaveClass('badge-err');

    // When a task actually fails, Needs Attention appears and isolates only failed/needs_decision tasks
    rerender(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ProductionBoardView
          items={[
            ...baseItems,
            {
              deliberation: 'Shader compile check.',
              item_id: 'task_failed',
              title: 'Compile post-process shader',
              kind: 'shader',
              spec: 'Author fragment shader.',
              proof_spec: 'Zero shader errors',
              target_modules: ['shaders/post.gdshader'],
              depends_on: [],
              state: 'failed',
              failure: 'Uniform type mismatch on line 14',
            },
          ]}
          cartridgeId="candidate_synthetic_02"
          liveStepsByItem={{}}
          onOpenLightbox={vi.fn()}
        />
      </QueryClientProvider>,
    );

    const attentionPill = screen.getByRole('button', { name: /Needs Attention/i });
    fireEvent.click(attentionPill);
    expect(screen.getByText('Compile post-process shader')).toBeInTheDocument();
    expect(screen.queryByText('Bind HUD readout to state store')).not.toBeInTheDocument();
  });

  it('drills down into every turn, step, self-inquiry, curated context, full prompt, response, and file source', async () => {
    const deepChronology: BoardItemChronologyResponse = {
      deliberation: 'HTTP wrapper message that must not overwrite task rationale.',
      cartridge_id: 'candidate_synthetic_03',
      item_id: 'task_deep_trace',
      title: 'Implement deterministic physics integrator',
      kind: 'mechanic_code',
      state: 'approved',
      spec: 'Integrate velocity and position over fixed 60Hz ticks.',
      proof_spec: 'Deterministic state hash after 120 ticks',
      target_modules: ['scripts/integrator.gd'],
      depends_on: [],
      context_packs: [
        {
          deliberation: 'Selected core loop and integrator contracts.',
          target_agent_role: 'swe_engineer',
          revision: 0,
          total_indexed_chunks: 18,
          selected_chunk_ids: ['bible:core_loop', 'code:scripts/integrator.gd'],
          role_tailored_synthesis: 'Keep fixed_Delta strictly invariant across frames.',
          search_queries_used: ['fixed_process integrator'],
        },
      ],
      self_inquiries: [
        {
          deliberation: 'Checked signal bus signature before emitting step.',
          agent_role: 'swe_engineer',
          question: 'What arguments does tick_completed emit?',
          answer: 'It emits (tick_index: int, state_hash: int).',
          researched_chunk_ids: ['code:scripts/bus.gd'],
        },
      ],
      file_contents: {
        'scripts/integrator.gd': {
          path: 'scripts/integrator.gd',
          is_binary: false,
          content: 'extends Node\nfunc step_tick(dt: float) -> void:\n\tpass\n',
          line_count: 3,
          size_bytes: 54,
        },
      },
      attempts: [
        {
          attempt_key: 'rev0',
          revision_index: 0,
          attempt_number: 1,
          is_failed_retry: false,
          status: 'reviewed',
          current_strategy: 'Implement fixed-timestep Verlet step then verify via headless QA.',
          submission_summary: 'Verified 120-tick state hash.',
          turns: [
            {
              turn_index: 0,
              specialist_agent: 'swe_engineer',
              lead_deliberation: 'Delegating core integrator module first.',
              brief: 'Implement step_tick(dt) in scripts/integrator.gd.',
              lead_prompt_text: 'You are the Task Lead Strategist.\nAVAILABLE SPECIALISTS:\nswe_engineer, qa_test_engineer',
              lead_curated_context: '=== CURATED STUDIO CONTEXT ===\n[bible:core_loop] (design) Core Loop\n60Hz fixed tick.\n=== END CURATED STUDIO CONTEXT ===',
              lead_self_inquiries: [
                {
                  question: 'Is scripts/integrator.gd already registered?',
                  answer: 'Yes, autoloaded in project.godot.',
                  researched_chunk_ids: ['code:project.godot'],
                },
              ],
              deliberation: 'Wrote pure Verlet step function.',
              summary: 'Implemented scripts/integrator.gd with 0 compiler warnings.',
              verification_passed: true,
              touched_files: ['scripts/integrator.gd'],
              code_diffs: {
                'scripts/integrator.gd': '--- a/scripts/integrator.gd\n+++ b/scripts/integrator.gd\n@@ -1 +1,2 @@\n extends Node\n+func step_tick(dt: float) -> void: pass',
              },
              step_log: [
                {
                  timestamp_utc: '2026-10-31T14:10:00.000Z',
                  step_type: 'planning_turn',
                  specialist_agent: 'task_lead',
                  message: 'Task Lead planning Turn #1',
                },
                {
                  timestamp_utc: '2026-10-31T14:10:12.500Z',
                  step_type: 'editing_file',
                  specialist_agent: 'swe_engineer',
                  message: 'Patched scripts/integrator.gd',
                  details: { file: 'scripts/integrator.gd', patches: 1 },
                },
              ],
              curated_context: '=== CURATED STUDIO CONTEXT ===\n[bible:core_loop] (design) Core Loop\n60Hz\n[code:scripts/bus.gd] (code) Signal Bus\nsignal tick_completed\n=== END CURATED STUDIO CONTEXT ===',
              prompt_text: 'You are the SWE Engineer.\nARCHITECTURAL SPECIFICATION:\n1. Fixed timestep.\n2. Pure state.\n3. Signal emission.\nEXISTING CARTRIDGE CODE:\n--- Existing scripts/integrator.gd ---\nextends Node',
              self_inquiries: [
                {
                  question: 'What arguments does tick_completed emit?',
                  answer: 'It emits (tick_index: int, state_hash: int).',
                  researched_chunk_ids: ['code:scripts/bus.gd'],
                },
              ],
              proof_urls: [
                {
                  path: 'artifacts/work_items/task_deep_trace/proof_rev0.png',
                  filename: 'proof_rev0.png',
                  url: '/api/artifacts/content?path=proof_rev0.png&raw=true',
                  is_audio: false,
                },
              ],
            },
          ],
          council_review: {
            revision_index: 0,
            verdict: 'approve',
            approved: true,
            deliberation: 'Verified fixed-timestep determinism.',
            reviewer_breakdown: [
              {
                reviewer_role: 'GameplaySystemsEngineer',
                raw_text: '1. Clean signal contract.',
                points: ['1. Clean signal contract.'],
              },
            ],
          },
        },
      ],
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(deepChronology), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    const boardItems: ProductionBoardItem[] = [
      {
        deliberation: 'Authentic task rationale from board item.',
        item_id: 'task_deep_trace',
        title: 'Implement deterministic physics integrator',
        kind: 'mechanic_code',
        spec: 'Integrate velocity and position over fixed 60Hz ticks.',
        proof_spec: 'Deterministic state hash after 120 ticks',
        target_modules: ['scripts/integrator.gd'],
        depends_on: [],
        state: 'approved',
        attempt_count: 1,
      },
    ];

    renderWithQueryClient(
      <ProductionBoardView
        items={boardItems}
        cartridgeId="candidate_synthetic_03"
        liveStepsByItem={{}}
        onOpenLightbox={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByText('Implement deterministic physics integrator'));

    await waitFor(() => {
      expect(screen.getByText('Turn #1')).toBeInTheDocument();
    });

    // Step-by-step execution log renders steps + elapsed delta + detail chips
    expect(screen.getByText('Patched scripts/integrator.gd')).toBeInTheDocument();
    expect(screen.getByText('+12.5s')).toBeInTheDocument();
    expect(screen.getByText('patches: 1')).toBeInTheDocument();

    // Specialist Self-Inquiry Q&A and Context Curator Pack are present on Turn #1
    expect(screen.getByText(/SWE Engineer Self-Inquiry Q&A/i)).toBeInTheDocument();
    expect(screen.getByText(/It emits \(tick_index: int, state_hash: int\)\./i)).toBeInTheDocument();

    // Expand Full Model Prompt inspector and verify structured sections mount lazily
    fireEvent.click(screen.getByText('Full Model Prompt (SWE Engineer)'));
    expect(screen.getByText('ARCHITECTURAL SPECIFICATION')).toBeInTheDocument();
    expect(screen.getByText('EXISTING CARTRIDGE CODE')).toBeInTheDocument();

    // Switch to Spec & Shaping subtab and verify authentic item.deliberation is preserved inside inspector
    const inspector = screen.getByRole('complementary', { name: /Task Chronology Inspector/i });
    fireEvent.click(within(inspector).getByRole('tab', { name: /Spec & Shaping/i }));
    expect(within(inspector).getByText('Authentic task rationale from board item.')).toBeInTheDocument();
    expect(within(inspector).queryByText(/HTTP wrapper message/i)).not.toBeInTheDocument();

    // Switch to Context & Self-Inquiry subtab
    fireEvent.click(within(inspector).getByRole('tab', { name: /Context & Self-Inquiry/i }));
    expect(within(inspector).getByText(/Keep fixed_Delta strictly invariant across frames\./i)).toBeInTheDocument();

    // Switch to Files & Proofs subtab
    fireEvent.click(within(inspector).getByRole('tab', { name: /Files & Proofs/i }));
    expect(within(inspector).getByText('proof_rev0.png')).toBeInTheDocument();
    expect(within(inspector).getByText('scripts/integrator.gd')).toBeInTheDocument();

    // Toggle Wide View mode
    const wideBtn = screen.getByRole('button', { name: /Wide view/i });
    fireEvent.click(wideBtn);
    expect(screen.getByRole('button', { name: /Split view/i })).toBeInTheDocument();
  });

  it('renders latest proofs, live turn step logs, and inline collapsible Spec & Council Direction directly on WorkItemCard', async () => {
    const liveChronology: BoardItemChronologyResponse = {
      deliberation: 'Live chronology trace.',
      cartridge_id: 'candidate_synthetic_04',
      item_id: 'task_visual_working',
      title: 'Render procedural grid tilemap and telemetry overlay',
      kind: 'level_map',
      state: 'in_progress',
      spec: '1) Generate deterministic tilemap; 2) Render telemetry overlay.',
      target_modules: ['scenes/map/tilemap_gen.gd'],
      depends_on: [],
      attempts: [
        {
          attempt_key: 'rev0',
          revision_index: 0,
          attempt_number: 1,
          is_failed_retry: false,
          status: 'running',
          turns: [
            {
              turn_index: 0,
              specialist_agent: 'qa_test_engineer',
              summary: 'Rendered initial tilemap preview.',
              proof_urls: [
                {
                  path: 'artifacts/work_items/task_visual_working/tilemap_preview.png',
                  filename: 'tilemap_preview.png',
                  url: '/api/artifacts/content?path=tilemap_preview.png&raw=true',
                  is_audio: false,
                },
              ],
            },
          ],
        },
      ],
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(liveChronology), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    const onOpenLightbox = vi.fn();
    const items: ProductionBoardItem[] = [
      {
        deliberation: 'Tilemap subsystem rationale.',
        item_id: 'task_visual_working',
        title: 'Render procedural grid tilemap and telemetry overlay',
        kind: 'level_map',
        spec: '1) Generate deterministic tilemap; 2) Render telemetry overlay.\n\n[SystemsArchitect Direction]: Keep chunk coordinates zero-allocated.\n\n[VisualDirector Direction]: Maintain high-contrast grid lines.',
        proof_spec: '4-frame headless render showing tile boundaries',
        target_modules: ['scenes/map/tilemap_gen.gd', 'scenes/ui/overlay.gd'],
        depends_on: [],
        state: 'in_progress',
        raised_by_role: 'LevelArchitect',
        attempt_count: 1,
      },
    ];

    renderWithQueryClient(
      <ProductionBoardView
        items={items}
        cartridgeId="candidate_synthetic_04"
        isBuildRunning
        liveStepsByItem={{
          task_visual_working: {
            itemId: 'task_visual_working',
            agentName: 'swe_engineer',
            specialistAgent: 'swe_engineer',
            turnIndex: 2,
            stepSummary: 'Turn #2 (swe_engineer): Patching scenes/ui/overlay.gd',
            eventType: 'backlog_item_substep',
            timestampUtc: '2026-10-31T15:10:05Z',
            startedAtUtc: '2026-10-31T15:10:00Z',
            stepLog: [
              {
                step_type: 'planning_turn',
                specialist_agent: 'task_lead',
                message: 'Task Lead planning Turn #2 strategy',
              },
              {
                step_type: 'editing_file',
                specialist_agent: 'swe_engineer',
                message: 'Patched scenes/ui/overlay.gd',
              },
            ],
          },
        }}
        onOpenLightbox={onOpenLightbox}
      />,
    );

    // Live turn badge + inline activity log steps render directly on the card
    expect(screen.getByText(/Executing Turn #2 · swe_engineer/i)).toBeInTheDocument();
    expect(screen.getByText(/Live Turn Activity Log \(2 steps\)/i)).toBeInTheDocument();
    expect(screen.getByText('Patched scenes/ui/overlay.gd')).toBeInTheDocument();

    // Collapsible Spec & Council Direction render cleanly on the card
    expect(screen.getByText(/\(2 requirements\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Council Direction \(2 inputs\)/i)).toBeInTheDocument();

    // Mid-flight rendered proof from Turn #1 appears directly on the card and opens lightbox on click
    await waitFor(() => {
      expect(screen.getByText('tilemap_preview.png')).toBeInTheDocument();
    });
    expect(screen.getByText(/Turn #1 · qa_test_engineer/i)).toBeInTheDocument();

    fireEvent.click(screen.getByAltText(/Rendered proof tilemap_preview\.png/i));
    expect(onOpenLightbox).toHaveBeenCalledTimes(1);
  });

  it('displays Council Requested Changes directly inside an Attempt #2 Working card', () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            deliberation: 'Trace loaded.',
            cartridge_id: 'candidate_synthetic_05',
            item_id: 'task_panel_rev2',
            title: 'Build Modular Control Panel',
            kind: 'ui',
            state: 'in_progress',
            spec: 'Implement panel controls.',
            target_modules: ['scenes/ui/control_panel.gd'],
            depends_on: [],
            attempts: [],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );

    const items: ProductionBoardItem[] = [
      {
        deliberation: 'Control panel task rationale.',
        item_id: 'task_panel_rev2',
        title: 'Build Modular Control Panel',
        kind: 'ui',
        spec: 'Implement panel controls.',
        target_modules: ['scenes/ui/control_panel.gd'],
        depends_on: [],
        state: 'in_progress',
        attempt_count: 2,
        revisions: [
          {
            revision_index: 0,
            verdict: 'request_changes',
            feedback:
              '[SystemsDesigner]: 1) Refresh slot affordability when signal fires. | [UXArchitect]: 1) Support keyboard dismissal.',
          },
        ],
      },
    ];

    renderWithQueryClient(
      <ProductionBoardView
        items={items}
        cartridgeId="candidate_synthetic_05"
        isBuildRunning
        liveStepsByItem={{}}
        onOpenLightbox={vi.fn()}
      />,
    );

    expect(screen.getByText('Attempt #2')).toBeInTheDocument();
    expect(screen.getByText('Changes Requested (Rev #1)')).toBeInTheDocument();
    expect(screen.getByText(/Council Requested Changes · Rev #1 \(2 reviewers\)/i)).toBeInTheDocument();
    expect(screen.getByText('SystemsDesigner')).toBeInTheDocument();
    expect(screen.getByText('UXArchitect')).toBeInTheDocument();
    expect(screen.getAllByText(/Refresh slot affordability when signal fires\./i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Support keyboard dismissal\./i)).toBeInTheDocument();
  });

  it('hides broken 404 proof images on WorkItemCard via onError instead of rendering broken alt boxes', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            deliberation: 'Trace with transient missing proof.',
            cartridge_id: 'candidate_synthetic_06',
            item_id: 'task_missing_img',
            title: 'Synthesize Procedural Lattice',
            kind: 'mechanic_code',
            state: 'in_progress',
            spec: 'Build deterministic lattice.',
            target_modules: ['scenes/systems/lattice.gd'],
            depends_on: [],
            attempts: [
              {
                attempt_key: 'rev0',
                revision_index: 0,
                attempt_number: 1,
                is_failed_retry: false,
                status: 'in_progress',
                turns: [
                  {
                    turn_index: 1,
                    specialist_agent: 'qa_test_engineer',
                    verification_passed: true,
                    timestamp_utc: '2026-10-31T16:40:00Z',
                    proof_urls: [
                      {
                        path: 'artifacts/work_items/task_missing_img/filmstrip_task_missing_img_rev0.png',
                        filename: 'filmstrip_task_missing_img_rev0.png',
                        url: '/api/artifacts/content?path=filmstrip_task_missing_img_rev0.png&raw=true',
                        is_audio: false,
                      },
                    ],
                  },
                ],
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );

    const items: ProductionBoardItem[] = [
      {
        deliberation: 'Lattice rationale.',
        item_id: 'task_missing_img',
        title: 'Synthesize Procedural Lattice',
        kind: 'mechanic_code',
        spec: 'Build deterministic lattice.',
        target_modules: ['scenes/systems/lattice.gd'],
        depends_on: [],
        state: 'in_progress',
        attempt_count: 1,
      },
    ];

    renderWithQueryClient(
      <ProductionBoardView
        items={items}
        cartridgeId="candidate_synthetic_06"
        isBuildRunning
        liveStepsByItem={{}}
        onOpenLightbox={vi.fn()}
      />,
    );

    const img = await screen.findByAltText(
      /Rendered proof filmstrip_task_missing_img_rev0\.png/i,
    );
    expect(img).toBeInTheDocument();

    // Simulate HTTP 404 on image load -> Latest Proof showcase cleanly hides instead of leaving a broken box
    fireEvent.error(img);
    await waitFor(() => {
      expect(
        screen.queryByAltText(/Rendered proof filmstrip_task_missing_img_rev0\.png/i),
      ).not.toBeInTheDocument();
      expect(screen.queryByText(/Latest Proof/i)).not.toBeInTheDocument();
    });
  });

  it('updates games count, mastery tiers, genre/category bars, and library table across 30d, 90d, 1yr, and lifetime in PlayerDnaView', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            deliberation: 'Synthesized multi-horizon player profile.',
            player_id: 'synth_player_09',
            profile_tag: 'default',
            total_playtime_hours: 2000,
            total_games_analyzed: 80,
            top_genres: ['Strategy'],
            genre_weights: { Strategy: 1.0 },
            top_categories: ['Single-player'],
            category_weights: { 'Single-player': 1.0 },
            playtime_distribution: {
              mastered_over_100h: 0.75,
              dedicated_50_to_100h: 0.15,
              engaged_10_to_50h: 0.05,
              casual_2_to_10h: 0.03,
              dabbled_under_2h: 0.02,
            },
            top_games: [
              {
                appid: 9001,
                name: 'Lifetime Anchor Title',
                playtime_hours: 600,
                playtime_share: 0.3,
                last_played_date: '2024-06-01',
                days_since_played: 850,
                genres: ['Strategy'],
                categories: ['Single-player'],
              },
            ],
            activity_windows: {
              last_30_days: {
                window_key: 'last_30_days',
                label: 'Last 30 Days',
                days: 30,
                games_count: 2,
                total_playtime_hours: 30,
                recent_2weeks_hours: 9.5,
                top_genres: ['Puzzle'],
                games: [
                  {
                    appid: 9002,
                    name: 'RecentThirtyDayTitle',
                    playtime_hours: 24,
                    playtime_2weeks_hours: 9.5,
                    last_played_date: '2026-10-29',
                    days_since_played: 2,
                    genres: ['Puzzle'],
                    categories: ['Co-op'],
                  },
                ],
              },
              last_90_days: {
                window_key: 'last_90_days',
                label: 'Last 90 Days',
                days: 90,
                games_count: 5,
                total_playtime_hours: 150,
                recent_2weeks_hours: 9.5,
                top_genres: ['Simulation'],
                games: [
                  {
                    appid: 9003,
                    name: 'RecentNinetyDayTitle',
                    playtime_hours: 120,
                    last_played_date: '2026-09-12',
                    days_since_played: 49,
                    genres: ['Simulation'],
                    categories: ['Steam Workshop'],
                  },
                ],
              },
              last_1_year: {
                window_key: 'last_1_year',
                label: 'Last 1 Year',
                days: 365,
                games_count: 18,
                total_playtime_hours: 510,
                recent_2weeks_hours: 9.5,
                top_genres: ['RPG'],
                games: [
                  {
                    appid: 9004,
                    name: 'AnnualHorizonTitle',
                    playtime_hours: 210,
                    last_played_date: '2026-02-15',
                    days_since_played: 258,
                    genres: ['RPG'],
                    categories: ['Steam Cloud'],
                  },
                ],
              },
              lifetime: {
                window_key: 'lifetime',
                label: 'Lifetime',
                days: null,
                games_count: 80,
                total_playtime_hours: 2000,
                recent_2weeks_hours: 9.5,
                top_genres: ['Strategy'],
                games: [
                  {
                    appid: 9001,
                    name: 'Lifetime Anchor Title',
                    playtime_hours: 600,
                    playtime_share: 0.3,
                    last_played_date: '2024-06-01',
                    days_since_played: 850,
                    genres: ['Strategy'],
                    categories: ['Single-player'],
                  },
                ],
              },
            },
            pacing_preference: 'balanced',
            pacing_score: 0.1,
            challenge_tolerance: 0.65,
            thematic_biases: ['industrial'],
            thematic_weights: { industrial: 1.0 },
            mechanical_biases: { logistics: 0.9 },
            preferred_palette_mood: 'muted_slate',
            palette_weights: { muted_slate: 1.0 },
            created_at_utc: '2026-10-31T10:00:00Z',
            ingestion_provenance: {},
            live_writers_room_dna_directive: 'Maintain crisp feedback.',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );

    renderWithQueryClient(<PlayerDnaView />);

    // Lifetime view initially active
    await waitFor(() => {
      expect(screen.getByText('Lifetime Anchor Title')).toBeInTheDocument();
    });

    // Switch to Last 30 Days via pill control
    const topHorizonGroup = screen.getByRole('group', { name: 'Activity Time Horizon' });
    fireEvent.click(within(topHorizonGroup).getByRole('button', { name: 'Last 30 Days' }));

    await waitFor(() => {
      expect(screen.getByText('RecentThirtyDayTitle')).toBeInTheDocument();
    });
    expect(screen.queryByText('Lifetime Anchor Title')).not.toBeInTheDocument();
    expect(screen.getByText(/Active Games \(Last 30 Days\)/i)).toBeInTheDocument();
    expect(screen.getAllByText('Co-op').length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText('Puzzle').length).toBeGreaterThanOrEqual(2);

    // Switch to Last 90 Days
    fireEvent.click(within(topHorizonGroup).getByRole('button', { name: 'Last 90 Days' }));
    await waitFor(() => {
      expect(screen.getByText('RecentNinetyDayTitle')).toBeInTheDocument();
    });
    expect(screen.getByText(/Active Games \(Last 90 Days\)/i)).toBeInTheDocument();
    expect(screen.getAllByText('Steam Workshop').length).toBeGreaterThanOrEqual(2);

    // Switch to Last 1 Year
    fireEvent.click(within(topHorizonGroup).getByRole('button', { name: 'Last 1 Year' }));
    await waitFor(() => {
      expect(screen.getByText('AnnualHorizonTitle')).toBeInTheDocument();
    });
    expect(screen.getByText(/Active Games \(Last 1 Year\)/i)).toBeInTheDocument();
    expect(screen.getAllByText('Steam Cloud').length).toBeGreaterThanOrEqual(2);
  });

  it('shows Build Error / Awaiting Playtest + Resume instead of Slice Verified when active slice playtest is incomplete', () => {
    const onResumeBuild = vi.fn();
    const onOpenNextSliceModal = vi.fn();

    const drainedItems: ProductionBoardItem[] = [
      {
        deliberation: 'Approved item.',
        item_id: 'task_done_1',
        title: 'Deterministic simulation loop',
        kind: 'mechanic_code',
        spec: 'Implement tick.',
        proof_spec: 'Headless pass',
        target_modules: ['scripts/sim.gd'],
        depends_on: [],
        state: 'approved',
        slice_id: 'slice_1_core',
      },
    ];

    const baseBible: GameBibleResponse = {
      deliberation: 'Loaded cartridge state.',
      has_bible: true,
      cartridge_id: 'candidate_synthetic_resume',
      status: 'in_production',
      game_title: 'Synthetic Engine Test',
      core_genre: 'Simulation',
      active_slice_index: 1,
      slice_roadmap: [
        { slice_id: 'slice_0_boot', slice_index: 0, title: 'Slice 0', status: 'completed' },
        { slice_id: 'slice_1_core', slice_index: 1, title: 'Slice 1', status: 'active' },
      ],
      production_board: {
        deliberation: 'All tasks approved.',
        cartridge_id: 'candidate_synthetic_resume',
        design_revision: 1,
        total_items: 1,
        counts_by_state: { approved: 1 },
        is_drained: true,
        items: drainedItems,
      },
    };

    const sharedProps = {
      games: [],
      selectedCartridgeId: 'candidate_synthetic_resume',
      activeCartridgeId: 'candidate_synthetic_resume',
      onSelectCartridge: vi.fn(),
      activeWorkspace: 'playtest' as const,
      onSelectWorkspace: vi.fn(),
      boardCounts: { approved: 1, total: 1 },
      transcriptCount: 10,
      assetsCount: 12,
      playtestRunsCount: 2,
      pendingWishesCount: 0,
      wsConnected: true,
      recentEventCount: 5,
      onActivateGame: vi.fn(),
      activatingGame: false,
      onLaunchGame: vi.fn(),
      launchingGame: false,
      launchStatusText: null,
      onPauseOrResume: vi.fn(),
      togglingPause: false,
      onApproveGreenlight: vi.fn(),
      onResumeBuild,
      resumingBuild: false,
      onOpenNextSliceModal,
      onStartNewGame: vi.fn(),
      onOpenCheckpointsModal: vi.fn(),
      onOpenLivePulseModal: vi.fn(),
    };

    const { rerender } = render(
      <TopCommandBar {...sharedProps} bible={baseBible} hasBuildError={true} />,
    );

    expect(screen.getByText('Build Error')).toBeInTheDocument();
    expect(screen.queryByText('Slice Verified')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Next Slice/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /^Resume$/i }));
    expect(onResumeBuild).toHaveBeenCalledTimes(1);
    expect(onOpenNextSliceModal).not.toHaveBeenCalled();

    // Even after dismissing the error banner (`hasBuildError={false}`), unfinished slice shows Awaiting Playtest + Resume
    rerender(<TopCommandBar {...sharedProps} bible={baseBible} hasBuildError={false} />);
    expect(screen.getByText('Awaiting Playtest')).toBeInTheDocument();
    expect(screen.queryByText('Slice Verified')).not.toBeInTheDocument();

    // Once the slice reaches `completed` and bible is `locked`, it transitions to Slice Verified + Next Slice
    const completedBible: GameBibleResponse = {
      ...baseBible,
      status: 'locked',
      slice_roadmap: [
        { slice_id: 'slice_0_boot', slice_index: 0, title: 'Slice 0', status: 'completed' },
        { slice_id: 'slice_1_core', slice_index: 1, title: 'Slice 1', status: 'completed' },
      ],
    };
    rerender(<TopCommandBar {...sharedProps} bible={completedBible} hasBuildError={false} />);
    expect(screen.getByText('Slice Verified')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Next Slice/i }));
    expect(onOpenNextSliceModal).toHaveBeenCalledTimes(1);
  });

  it('surfaces interrupted Council playtest recovery callout and resume button in PlaytestLabView', async () => {
    const onResumeCouncilPlaytest = vi.fn();
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            deliberation: 'Loaded failed council session.',
            session: {
              deliberation: 'Runner boot timed out.',
              session_id: 'pt_council_err_01',
              checkpoint_id: 'active',
              candidate_id: 'candidate_synthetic_resume',
              slice_id: 'slice_1_core',
              slice_title: 'Slice 1: Core Loop',
              game_title: 'Synthetic Engine Test',
              status: 'error',
              initiated_by: 'council',
              error_message: 'Runner IPC socket handshake timed out.',
              turn_count: 0,
            },
            turns: [],
            wishes: [],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );

    renderWithQueryClient(
      <PlaytestLabView
        selectedCartridgeId="candidate_synthetic_resume"
        sessionsListData={{
          deliberation: 'Sessions list.',
          active_cartridge_id: 'candidate_synthetic_resume',
          pending_wishes_count: 0,
          games_catalog: [],
          playable_targets: [],
          sessions: [
            {
              deliberation: 'Failed council session.',
              session_id: 'pt_council_err_01',
              checkpoint_id: 'active',
              candidate_id: 'candidate_synthetic_resume',
              slice_id: 'slice_1_core',
              slice_title: 'Slice 1: Core Loop',
              game_title: 'Synthetic Engine Test',
              status: 'error',
              initiated_by: 'council',
              error_message: 'Runner IPC socket handshake timed out.',
              turn_count: 0,
              matches_active_cartridge: true,
            },
          ],
        }}
        sessionsError={null}
        onRefetchSessions={vi.fn()}
        canResumeCouncilPlaytest={true}
        onResumeCouncilPlaytest={onResumeCouncilPlaytest}
      />,
    );

    expect(screen.getByText('Council Blind Playtest Interrupted')).toBeInTheDocument();
    expect(screen.getByText('Runner IPC socket handshake timed out.')).toBeInTheDocument();
    const resumeBtn = screen.getByRole('button', {
      name: /Resume Slice & Run Council Playtest/i,
    });
    fireEvent.click(resumeBtn);
    expect(onResumeCouncilPlaytest).toHaveBeenCalledTimes(1);
  });

  it('provides accessible UI text size controls (A-, reset %, A+) in TopCommandBar', () => {
    const onChangeUiScalePct = vi.fn();
    const baseProps = {
      games: [],
      selectedCartridgeId: 'active',
      activeCartridgeId: 'active',
      onSelectCartridge: vi.fn(),
      activeWorkspace: 'board' as const,
      onSelectWorkspace: vi.fn(),
      bible: undefined,
      boardCounts: { approved: 0, total: 0 },
      transcriptCount: 0,
      assetsCount: 0,
      playtestRunsCount: 0,
      pendingWishesCount: 0,
      wsConnected: true,
      recentEventCount: 0,
      onActivateGame: vi.fn(),
      activatingGame: false,
      onLaunchGame: vi.fn(),
      launchingGame: false,
      launchStatusText: null,
      onPauseOrResume: vi.fn(),
      togglingPause: false,
      onApproveGreenlight: vi.fn(),
      onOpenNextSliceModal: vi.fn(),
      onStartNewGame: vi.fn(),
      onOpenCheckpointsModal: vi.fn(),
      onOpenLivePulseModal: vi.fn(),
      onChangeUiScalePct,
    };

    const { rerender } = render(<TopCommandBar {...baseProps} uiScalePct={100} />);

    const group = screen.getByRole('group', { name: 'UI text size' });
    const decBtn = within(group).getByRole('button', { name: /^A- Decrease text size$/i });
    const resetBtn = within(group).getByRole('button', { name: /^100% Reset text size$/i });
    const incBtn = within(group).getByRole('button', { name: /^A\+ Increase text size$/i });

    fireEvent.click(incBtn);
    expect(onChangeUiScalePct).toHaveBeenLastCalledWith(110);

    fireEvent.click(decBtn);
    expect(onChangeUiScalePct).toHaveBeenLastCalledWith(90);

    rerender(<TopCommandBar {...baseProps} uiScalePct={160} />);
    expect(within(group).getByRole('button', { name: /^A\+ Increase text size$/i })).toBeDisabled();
    fireEvent.click(within(group).getByRole('button', { name: /^160% Reset text size$/i }));
    expect(onChangeUiScalePct).toHaveBeenLastCalledWith(100);

    rerender(<TopCommandBar {...baseProps} uiScalePct={90} />);
    expect(within(group).getByRole('button', { name: /^A- Decrease text size$/i })).toBeDisabled();
    expect(resetBtn).toBeInTheDocument();
  });

  it('exposes a keyboard-focusable 2D pan viewport in MediaLightbox when 1:1 Pixel Zoom is active', () => {
    HTMLDialogElement.prototype.showModal = vi.fn(function mockShowModal(this: HTMLDialogElement) {
      this.setAttribute('open', '');
    });
    HTMLDialogElement.prototype.close = vi.fn(function mockClose(this: HTMLDialogElement) {
      this.removeAttribute('open');
    });

    render(
      <MediaLightbox
        media={{
          url: '/api/artifacts/content?path=proof_sheet.png&raw=true',
          title: 'Sprite Atlas Sheet',
        }}
        onClose={vi.fn()}
      />,
    );

    expect(screen.queryByRole('region', { name: /Zoomed preview/i })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /1:1 Pixel Zoom/i }));

    const panRegion = screen.getByRole('region', { name: 'Zoomed preview: Sprite Atlas Sheet' });
    expect(panRegion).toHaveClass('lightbox-pan-viewport');
    expect(panRegion).toHaveAttribute('tabindex', '0');
  });

  it('enforces zoom-friendly 2D panning and relative rem typography invariants in global.css', () => {
    const cssPath = resolve(process.cwd(), 'src/styles/global.css');
    const css = readFileSync(cssPath, 'utf8');

    // Zero overscroll-behavior: contain/none traps that swallow visualViewport pinch-zoom panning
    expect(css).not.toMatch(/overscroll-behavior\s*:\s*(contain|none)/);
    // Zero px font-size declarations (honors root rem scaling and browser font preferences)
    expect(css).not.toMatch(/font-size\s*:\s*[0-9.]+px/);
    // Root relative scaling + unlocked 2D axis panning
    expect(css).toContain('font-size: calc(87.5% * var(--ui-scale, 1));');
    expect(css).toContain('scroll-axis-lock: none;');
    expect(css).toContain('place-items: safe center;');
    // .workspace-main must not clip zoomed multi-column overflow
    const workspaceMainBlock = css.match(/\.workspace-main\s*\{[^}]*\}/)?.[0] ?? '';
    expect(workspaceMainBlock).not.toContain('overflow: clip');
  });

  it('includes Revising, In Review, and all started-unapproved tasks under the Working umbrella filter on ProductionBoardView', () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            deliberation: 'Synthetic chronology.',
            cartridge_id: 'candidate_synthetic_working',
            item_id: 'task_in_review',
            title: 'Verify Shader Pipeline',
            kind: 'mechanic_code',
            state: 'awaiting_review',
            spec: 'Validate uniform bindings.',
            target_modules: ['scenes/systems/shader_pipe.gd'],
            depends_on: [],
            attempts: [],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );

    const items: ProductionBoardItem[] = [
      {
        deliberation: 'Worker active rationale.',
        item_id: 'task_active_worker',
        title: 'Build Kinematic Solver',
        kind: 'mechanic_code',
        spec: 'Implement solver.',
        target_modules: ['scenes/systems/kinematic.gd'],
        depends_on: [],
        state: 'in_progress',
        attempt_count: 1,
      },
      {
        deliberation: 'Council review rationale.',
        item_id: 'task_in_review',
        title: 'Verify Shader Pipeline',
        kind: 'mechanic_code',
        spec: 'Validate uniform bindings.',
        target_modules: ['scenes/systems/shader_pipe.gd'],
        depends_on: [],
        state: 'awaiting_review',
        attempt_count: 1,
      },
      {
        deliberation: 'Revising after Council feedback rationale.',
        item_id: 'task_revising',
        title: 'Refine Collision Broadphase',
        kind: 'mechanic_code',
        spec: 'Tighten AABB bounds.',
        target_modules: ['scenes/systems/broadphase.gd'],
        depends_on: [],
        state: 'changes_requested',
        attempt_count: 1,
      },
      {
        deliberation: 'Started task awaiting spawned prerequisite.',
        item_id: 'task_started_blocked',
        title: 'Integrate Telemetry HUD',
        kind: 'ui',
        spec: 'Bind HUD counters.',
        target_modules: ['scenes/ui/hud.gd'],
        depends_on: ['task_prereq'],
        state: 'blocked',
        attempt_count: 1,
        revisions: [{ revision_index: 0, verdict: 'request_changes', feedback: 'Await prereq.' }],
      },
      {
        deliberation: 'Queued task rationale.',
        item_id: 'task_queued',
        title: 'Compose Ambient Cue',
        kind: 'audio_sfx',
        spec: 'Synthesize loop.',
        target_modules: ['assets/audio/cue.wav'],
        depends_on: [],
        state: 'ready',
      },
      {
        deliberation: 'Completed task rationale.',
        item_id: 'task_done',
        title: 'Bootstrap Entrypoint Scene',
        kind: 'integration',
        spec: 'Initialize root scene.',
        target_modules: ['scenes/main.tscn'],
        depends_on: [],
        state: 'approved',
        attempt_count: 1,
      },
    ];

    renderWithQueryClient(
      <ProductionBoardView
        items={items}
        cartridgeId="candidate_synthetic_working"
        isBuildRunning
        liveStepsByItem={{}}
        onOpenLightbox={vi.fn()}
      />,
    );

    const filterBar = screen.getByRole('group', { name: 'Filter Production Board by Status' });
    // Working counts all 4 started-and-not-yet-approved tasks (in_progress + awaiting_review + changes_requested + started-blocked)
    const workingPill = within(filterBar).getByRole('button', { name: /Working\s*4/i });
    const revisingPill = within(filterBar).getByRole('button', { name: /Revising\s*1/i });
    fireEvent.click(workingPill);

    expect(screen.getByText('Build Kinematic Solver')).toBeInTheDocument();
    expect(screen.getByText('Verify Shader Pipeline')).toBeInTheDocument();
    expect(screen.getByText('Refine Collision Broadphase')).toBeInTheDocument();
    expect(screen.getByText('Integrate Telemetry HUD')).toBeInTheDocument();
    expect(screen.getByText('Specialist Council Reviewing Submission')).toBeInTheDocument();
    expect(screen.queryByText('Compose Ambient Cue')).not.toBeInTheDocument();
    expect(screen.queryByText('Bootstrap Entrypoint Scene')).not.toBeInTheDocument();

    // Clicking the focused Revising sub-filter narrows specifically to the revising item
    fireEvent.click(revisingPill);
    expect(screen.getByText('Refine Collision Broadphase')).toBeInTheDocument();
    expect(screen.queryByText('Build Kinematic Solver')).not.toBeInTheDocument();
  });

  it('updates Live Screen image src cache-buster in real time as playtest_frame_update events arrive', async () => {
    const sessionDetailPayload = {
      deliberation: 'Live playtest snapshot.',
      session: {
        deliberation: 'Running playtest.',
        session_id: 'pt_live_stream_01',
        checkpoint_id: 'candidate_synthetic_live',
        candidate_id: 'candidate_synthetic_live',
        game_title: 'Synthetic Kinetic Sandbox',
        status: 'running',
        initiated_by: 'council',
        turn_count: 3,
        sim_ticks_total: 180,
        latest_frame_name: 'live_preview.jpg',
        has_video: false,
        matches_active_cartridge: true,
        updated_at_utc: '2026-10-31T19:00:01Z',
      },
      turns: [],
      wishes: [],
      compactions: [],
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(sessionDetailPayload), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    const sessionsListData = {
      deliberation: 'Session list.',
      active_cartridge_id: 'candidate_synthetic_live',
      games_catalog: [],
      sessions: [sessionDetailPayload.session],
      playable_targets: [],
      pending_wishes: [],
      pending_wishes_count: 0,
    };

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { rerender } = render(
      <QueryClientProvider client={qc}>
        <PlaytestLabView
          selectedCartridgeId="candidate_synthetic_live"
          sessionsListData={sessionsListData}
          sessionsError={null}
          onRefetchSessions={vi.fn()}
          livePlaytestFrames={{
            pt_live_stream_01: {
              sessionId: 'pt_live_stream_01',
              tick: 192,
              frameName: 'live_preview.jpg',
              timestampUtc: '2026-10-31T19:00:02Z',
              seq: 1,
            },
          }}
        />
      </QueryClientProvider>,
    );

    const liveImg = await screen.findByAltText('Latest player frame');
    const firstSrc = liveImg.getAttribute('src') ?? '';
    expect(firstSrc).toContain('/api/playtest/sessions/pt_live_stream_01/media/live_preview.jpg?v=192-1');
    expect(screen.getByText(/Tick 192/i)).toBeInTheDocument();

    rerender(
      <QueryClientProvider client={qc}>
        <PlaytestLabView
          selectedCartridgeId="candidate_synthetic_live"
          sessionsListData={sessionsListData}
          sessionsError={null}
          onRefetchSessions={vi.fn()}
          livePlaytestFrames={{
            pt_live_stream_01: {
              sessionId: 'pt_live_stream_01',
              tick: 240,
              frameName: 'live_preview.jpg',
              timestampUtc: '2026-10-31T19:00:03Z',
              seq: 2,
            },
          }}
        />
      </QueryClientProvider>,
    );

    const updatedSrc = screen.getByAltText('Latest player frame').getAttribute('src') ?? '';
    expect(updatedSrc).toContain('/api/playtest/sessions/pt_live_stream_01/media/live_preview.jpg?v=240-2');
    expect(updatedSrc).not.toBe(firstSrc);
    expect(screen.getByText(/Tick 240/i)).toBeInTheDocument();
  });

  it('renders visual_and_ui_defects_observed and bugs_and_glitches_observed in PlaytestLabView Experience Report tab', async () => {
    const completedDetailPayload = {
      deliberation: 'Completed session with candid visual defects.',
      session: {
        deliberation: 'Completed session summary.',
        session_id: 'pt_visual_audit_01',
        checkpoint_id: 'cp_visual_01',
        candidate_id: 'candidate_synthetic_visual',
        game_title: 'Synthetic Relay',
        status: 'completed',
        initiated_by: 'council',
        current_mood: 'puzzled',
        turn_count: 12,
        sim_ticks_total: 720,
        has_video: false,
        matches_active_cartridge: true,
        overall_fun_rating: 4,
        session_report: {
          deliberation: 'Audited high-res screenshots before writing graphics_and_visuals.',
          overall_experience_summary: 'Responsive controls, but sprite slicing and HUD layout need fixes.',
          visual_and_ui_defects_observed: [
            'Foreground sprites show stray edge slivers from neighboring atlas frames',
            'Top-left counter text spills outside its panel border',
          ],
          graphics_and_visuals: 'Unfinished visual craft due to sprite-sheet bleed and overflowing HUD labels.',
          bugs_and_glitches_observed: ['Sprite atlas frame bleed on interactive props'],
          key_pain_points: ['Overflowing counter text overlaps background'],
          overall_fun_rating: 4,
        },
      },
      turns: [],
      wishes: [],
      compactions: [],
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(completedDetailPayload), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    renderWithQueryClient(
      <PlaytestLabView
        selectedCartridgeId="candidate_synthetic_visual"
        sessionsListData={{
          deliberation: 'Session list.',
          active_cartridge_id: 'candidate_synthetic_visual',
          games_catalog: [],
          sessions: [completedDetailPayload.session],
          playable_targets: [],
          pending_wishes_count: 0,
        }}
        sessionsError={null}
        onRefetchSessions={vi.fn()}
      />,
    );

    const reportTab = await screen.findByRole('button', { name: /Experience Report/i });
    fireEvent.click(reportTab);

    expect(await screen.findByText(/Visual & UI Defects Observed \(2\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Foreground sprites show stray edge slivers from neighboring atlas frames/i)).toBeInTheDocument();
    expect(screen.getByText(/Bugs & Glitches Observed \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Key Pain Points \(1\)/i)).toBeInTheDocument();
  });
});
