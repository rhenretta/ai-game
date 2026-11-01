import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/App';
import { GameMemoryView } from '../src/features/memory/GameMemoryView';

describe('GameMemoryView & Mission Control Memory Workspace', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders multi-bank counts, semantic <article> cards, and filters by bank, status, and kind', async () => {
    const requestedUrls: string[] = [];

    const entriesFixture = [
      {
        deliberation: 'Cross-slice simulation determinism must hold before adding secondary subsystems.',
        memory_id: 'mem_council_01',
        cartridge_id: 'candidate_synthetic_01',
        memory_bank: 'council',
        author_role: 'CreativeDirector',
        kind: 'strategic_goal' as const,
        status: 'active' as const,
        title: 'Expand Multi-Zone State Transitions',
        content: 'Introduce multi-zone state handoffs once core loop telemetry stabilizes.',
        applicable_roles: ['swe_engineer', 'task_lead'],
        tags: [],
        slice_id: 'slice_0_foundation',
        item_id: null,
        superseded_by_id: null,
        resolution_note: null,
        created_at_utc: '2026-10-31T10:00:00Z',
        updated_at_utc: '2026-10-31T10:00:00Z',
      },
      {
        deliberation: 'Deterministic harness assertions avoid timing drift across headless runs.',
        memory_id: 'mem_qa_02',
        cartridge_id: 'candidate_synthetic_01',
        memory_bank: 'qa_test_engineer',
        author_role: 'qa_test_engineer',
        kind: 'lesson_learned' as const,
        status: 'active' as const,
        title: 'Keep Harness Verification Bounds Compact',
        content: 'Bound deterministic tick intervals tightly so automated harness runs stay fast and reproducible.',
        applicable_roles: ['qa_test_engineer', 'task_lead'],
        tags: [],
        slice_id: 'slice_0_foundation',
        item_id: 'item_01',
        superseded_by_id: null,
        resolution_note: null,
        created_at_utc: '2026-10-31T11:00:00Z',
        updated_at_utc: '2026-10-31T11:00:00Z',
      },
    ];

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      requestedUrls.push(url);
      const parsed = new URL(url, 'http://localhost');
      const bank = parsed.searchParams.get('bank') ?? 'all';
      const kind = parsed.searchParams.get('kind') ?? 'all';
      const status = parsed.searchParams.get('status') ?? 'active';

      const filtered = entriesFixture.filter((e) => {
        if (bank !== 'all' && e.memory_bank !== bank) return false;
        if (kind !== 'all' && e.kind !== kind) return false;
        if (status !== 'all' && e.status !== status) return false;
        return true;
      });

      return new Response(
        JSON.stringify({
          deliberation: 'Loaded cartridge memory banks.',
          cartridge_id: 'candidate_synthetic_01',
          game_title: 'Synthetic Relay Studio',
          bank_counts: { council: 1, qa_test_engineer: 1 },
          total_bank_counts: { council: 1, qa_test_engineer: 1 },
          total_count: 2,
          active_count: 2,
          entries: filtered,
        }),
        { status: 200 },
      );
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={qc}>
        <GameMemoryView
          cartridgeId="candidate_synthetic_01"
          gameTitle="Synthetic Relay Studio"
          activeSliceId="slice_1_routing"
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText('Expand Multi-Zone State Transitions')).toBeInTheDocument();
      expect(screen.getByText('Keep Harness Verification Bounds Compact')).toBeInTheDocument();
    });

    // Cards use semantic <article class="memory-card"> with direct header/content/footer regions
    const articles = container.querySelectorAll('article.memory-card');
    expect(articles.length).toBe(2);
    expect(articles[0].querySelector(':scope > header.memory-card-header')).not.toBeNull();
    expect(articles[0].querySelector(':scope > div.memory-card-content')).not.toBeNull();
    expect(articles[0].querySelector(':scope > footer.memory-card-footer')).not.toBeNull();

    // Filter to QA Test Engineer specialist bank
    fireEvent.click(screen.getByRole('button', { name: /QA Test Engineer/i }));
    await waitFor(() => {
      expect(screen.getByText('Keep Harness Verification Bounds Compact')).toBeInTheDocument();
      expect(screen.queryByText('Expand Multi-Zone State Transitions')).not.toBeInTheDocument();
    });
    expect(requestedUrls.some((u) => u.includes('bank=qa_test_engineer'))).toBe(true);

    // Filter back to All Banks + Strategic Goals kind
    fireEvent.click(screen.getByRole('button', { name: /All Banks/i }));
    fireEvent.click(screen.getByRole('button', { name: /Strategic Goals/i }));
    await waitFor(() => {
      expect(screen.getByText('Expand Multi-Zone State Transitions')).toBeInTheDocument();
      expect(screen.queryByText('Keep Harness Verification Bounds Compact')).not.toBeInTheDocument();
    });
    expect(requestedUrls.some((u) => u.includes('kind=strategic_goal'))).toBe(true);
  });

  it('defers validation until blur/submit and serializes deliberation at JSON key index 0 on record and fulfill', async () => {
    const postedBodies: string[] = [];
    let fulfilled = false;

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (init?.method === 'POST' && url.includes('/api/studio/memories')) {
        const rawBody = String(init.body ?? '');
        postedBodies.push(rawBody);
        const parsedBody = JSON.parse(rawBody) as { operation?: string };
        if (parsedBody.operation === 'fulfill') {
          fulfilled = true;
        }
        return new Response(
          JSON.stringify({
            deliberation: 'Applied memory mutation.',
            ok: true,
            applied_count: 1,
            cartridge_id: 'candidate_synthetic_01',
            game_title: 'Synthetic Relay Studio',
            bank_counts: { council: fulfilled ? 0 : 1 },
            total_bank_counts: { council: 1 },
            total_count: 1,
            active_count: fulfilled ? 0 : 1,
            entries: [],
          }),
          { status: 200 },
        );
      }

      return new Response(
        JSON.stringify({
          deliberation: 'Loaded cartridge memory banks.',
          cartridge_id: 'candidate_synthetic_01',
          game_title: 'Synthetic Relay Studio',
          bank_counts: { council: fulfilled ? 0 : 1 },
          total_bank_counts: { council: 1 },
          total_count: 1,
          active_count: fulfilled ? 0 : 1,
          entries: fulfilled
            ? []
            : [
                {
                  deliberation: 'Persist modular state schema contract across slices.',
                  memory_id: 'mem_goal_01',
                  cartridge_id: 'candidate_synthetic_01',
                  memory_bank: 'council',
                  author_role: 'GameplaySystemsEngineer',
                  kind: 'strategic_goal',
                  status: 'active',
                  title: 'Unify State Snapshot Serialization',
                  content: 'Ensure every simulation module registers deterministic snapshot hooks.',
                  applicable_roles: ['swe_engineer'],
                  tags: [],
                  slice_id: 'slice_0_foundation',
                  item_id: null,
                  superseded_by_id: null,
                  resolution_note: null,
                  created_at_utc: '2026-10-31T10:00:00Z',
                  updated_at_utc: '2026-10-31T10:00:00Z',
                },
              ],
        }),
        { status: 200 },
      );
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <GameMemoryView
          cartridgeId="candidate_synthetic_01"
          gameTitle="Synthetic Relay Studio"
          activeSliceId="slice_1_routing"
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText('Unify State Snapshot Serialization')).toBeInTheDocument();
    });

    // Open composer
    fireEvent.click(screen.getByRole('button', { name: /Record Memory/i }));

    const rationaleInput = screen.getByLabelText(/Epistemic Rationale/i) as HTMLTextAreaElement;
    const headlineInput = screen.getByLabelText(/Memory Headline/i) as HTMLInputElement;
    const contentInput = screen.getByLabelText(/Actionable Guidance/i) as HTMLTextAreaElement;

    // 1. Blurring an untouched field does NOT prematurely flag aria-invalid
    fireEvent.blur(headlineInput);
    expect(headlineInput.getAttribute('aria-invalid')).toBeNull();
    expect(headlineInput.classList.contains('user-invalid-fallback')).toBe(false);

    // 2. Typing partial text does NOT flag aria-invalid before blur
    fireEvent.change(rationaleInput, { target: { value: 'short' } });
    expect(rationaleInput.getAttribute('aria-invalid')).toBeNull();

    // 3. Blurring after typing too-short text activates deferred invalid state
    fireEvent.blur(rationaleInput);
    expect(rationaleInput.getAttribute('aria-invalid')).toBe('true');
    expect(rationaleInput.classList.contains('user-invalid-fallback')).toBe(true);

    // 4. Correcting the input immediately clears the invalid state
    fireEvent.change(rationaleInput, {
      target: { value: 'Persist deterministic audio bus routing rules across upcoming slices.' },
    });
    expect(rationaleInput.getAttribute('aria-invalid')).toBe('false');
    expect(rationaleInput.classList.contains('user-invalid-fallback')).toBe(false);
    expect(rationaleInput.classList.contains('user-valid-fallback')).toBe(true);

    fireEvent.change(headlineInput, { target: { value: 'Route SFX Through Master Bus' } });
    fireEvent.change(contentInput, {
      target: { value: 'Bind every synthesized AudioStreamPlayer to the Master bus with explicit volume floor.' },
    });

    fireEvent.click(screen.getByRole('button', { name: /Save Memory Entry/i }));

    await waitFor(() => {
      expect(postedBodies.length).toBe(1);
    });

    const recordJson = JSON.parse(postedBodies[0]) as Record<string, unknown>;
    // Rule 4 Rationale-First: `deliberation` must be serialized at key index 0
    expect(Object.keys(recordJson)[0]).toBe('deliberation');
    expect(recordJson.operation).toBe('record');
    expect(recordJson.title).toBe('Route SFX Through Master Bus');

    // Fulfill existing active strategic goal
    fireEvent.click(screen.getByRole('button', { name: /Mark Fulfilled/i }));
    await waitFor(() => {
      expect(postedBodies.length).toBe(2);
    });
    const fulfillJson = JSON.parse(postedBodies[1]) as Record<string, unknown>;
    expect(Object.keys(fulfillJson)[0]).toBe('deliberation');
    expect(fulfillJson.operation).toBe('fulfill');
    expect(fulfillJson.target_memory_id).toBe('mem_goal_01');
  });

  it('integrates cleanly into TopCommandBar and App workspace switching', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;

      if (url.includes('/api/cartridges')) {
        return new Response(
          JSON.stringify({
            cartridges: ['candidate_synthetic_01'],
            games: [
              {
                cartridge_id: 'candidate_synthetic_01',
                game_title: 'Synthetic Isometric Builder',
                core_genre: 'Simulation Strategy',
                status: 'slice_complete',
                has_backlog: true,
                approved_items: 1,
                total_items: 1,
                updated_at_utc: '2026-10-31T12:00:00Z',
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/studio/game-bible')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded synthetic game bible.',
            has_bible: true,
            cartridge_id: 'candidate_synthetic_01',
            active_candidate_id: 'candidate_synthetic_01',
            game_title: 'Synthetic Isometric Builder',
            production_board: {
              deliberation: 'Board state.',
              cartridge_id: 'candidate_synthetic_01',
              design_revision: 1,
              total_items: 0,
              counts_by_state: {},
              is_drained: true,
              items: [],
            },
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/studio/memories')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded memories for active cartridge.',
            cartridge_id: 'candidate_synthetic_01',
            game_title: 'Synthetic Isometric Builder',
            bank_counts: { swe_engineer: 1 },
            total_bank_counts: { swe_engineer: 1 },
            total_count: 1,
            active_count: 1,
            entries: [
              {
                deliberation: 'Retain clean signal decoupling across scene reloads.',
                memory_id: 'mem_swe_99',
                cartridge_id: 'candidate_synthetic_01',
                memory_bank: 'swe_engineer',
                author_role: 'swe_engineer',
                kind: 'architectural_decision',
                status: 'active',
                title: 'Decouple Simulation Clock Signals',
                content: 'Emit tick updates through the central simulation coordinator node.',
                applicable_roles: ['swe_engineer'],
                tags: [],
                slice_id: 'slice_0_foundation',
                item_id: null,
                superseded_by_id: null,
                resolution_note: null,
                created_at_utc: '2026-10-31T12:00:00Z',
                updated_at_utc: '2026-10-31T12:00:00Z',
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/artifacts')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Inventory.',
            exists: true,
            manifest: {},
            sprites: [],
            audio_clips: [],
            code_modules: [],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/playtest/sessions')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Sessions.',
            active_cartridge_id: 'candidate_synthetic_01',
            pending_wishes_count: 0,
            games_catalog: [],
            playable_targets: [],
            sessions: [],
          }),
          { status: 200 },
        );
      }

      return new Response(JSON.stringify({ deliberation: 'ok' }), { status: 200 });
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <App />
      </QueryClientProvider>,
    );

    const memoryTab = await screen.findByRole('tab', { name: /^Memory$/i });
    fireEvent.click(memoryTab);

    await waitFor(() => {
      expect(screen.getByText('Decouple Simulation Clock Signals')).toBeInTheDocument();
    });
  });

  it('keeps memory card headers single-column and shrink-safe for long snake_case metadata tokens', async () => {
    const longSyntheticSliceId =
      'slice_4_autonomous_multi_region_telemetry_and_deterministic_state_routing';
    const longSyntheticItemId = 'item_reconcile_cross_module_snapshot_serialization_09';

    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
      return new Response(
        JSON.stringify({
          deliberation: 'Loaded synthetic memory card with long metadata identifiers.',
          cartridge_id: 'candidate_synthetic_01',
          game_title: 'Synthetic Relay Studio',
          bank_counts: { council: 1 },
          total_bank_counts: { council: 1 },
          total_count: 1,
          active_count: 1,
          entries: [
            {
              deliberation: 'Preserve deterministic rendering bounds across nested viewport overlays.',
              memory_id: 'mem_arch_long_01',
              cartridge_id: 'candidate_synthetic_01',
              memory_bank: 'council',
              author_role: 'CreativeDirector',
              kind: 'architectural_decision',
              status: 'active',
              title: 'Contextual Viewport Overlay Coordinate Invariant',
              content: 'Keep overlay coordinate transforms decoupled from camera zoom interpolation.',
              applicable_roles: ['swe_engineer', 'qa_test_engineer'],
              tags: [],
              slice_id: longSyntheticSliceId,
              item_id: longSyntheticItemId,
              superseded_by_id: null,
              resolution_note: null,
              created_at_utc: '2026-10-31T14:00:00Z',
              updated_at_utc: '2026-10-31T14:00:00Z',
            },
          ],
        }),
        { status: 200 },
      );
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={qc}>
        <GameMemoryView
          cartridgeId="candidate_synthetic_01"
          gameTitle="Synthetic Relay Studio"
          activeSliceId={longSyntheticSliceId}
        />
      </QueryClientProvider>,
    );

    const heading = await screen.findByRole('heading', {
      level: 3,
      name: 'Contextual Viewport Overlay Coordinate Invariant',
    });
    const header = heading.closest('header.memory-card-header');
    expect(header).not.toBeNull();

    // Header regions stack in clean source order: badges -> h3 -> meta
    const headerChildren = Array.from(header!.children);
    expect(headerChildren.map((el) => el.tagName)).toEqual(['DIV', 'H3', 'DIV']);
    expect(headerChildren[0]).toHaveClass('memory-card-badges');
    expect(headerChildren[2]).toHaveClass('memory-card-meta');

    const metaSpans = container.querySelectorAll('.memory-card-meta > span');
    expect(metaSpans.length).toBe(4);
    expect(screen.getByText(`Slice: ${longSyntheticSliceId}`)).toBeInTheDocument();
    expect(screen.getByText(`Task: ${longSyntheticItemId}`)).toBeInTheDocument();

    // Guard CSS layout invariants so @container never splits .memory-card-header into `minmax(0, 1fr) auto`
    // (which collapses h3 to a 0px 1-character vertical column beside long metadata tokens)
    const cssSource = readFileSync(resolve(__dirname, '../src/styles/global.css'), 'utf-8');
    const memoryCssStart = cssSource.indexOf('/* Semantic Direct-Region Memory Cards');
    expect(memoryCssStart).toBeGreaterThan(-1);
    const memoryCssEnd = cssSource.indexOf('.live-activity-banner', memoryCssStart);
    const memoryCardCss = cssSource.slice(memoryCssStart, memoryCssEnd);

    expect(memoryCardCss).toContain('container: memory-card / inline-size;');
    expect(memoryCardCss).toMatch(
      /\.memory-card\s*>\s*\.memory-card-header\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\);/,
    );
    expect(memoryCardCss).not.toMatch(
      /@container[^{]*\{[\s\S]*?\.memory-card-header[\s\S]*?minmax\(0,\s*1fr\)\s+auto/,
    );
    expect(memoryCardCss).toMatch(
      /\.memory-card\s*>\s*\.memory-card-header\s*>\s*h3\s*\{[^}]*overflow-wrap:\s*break-word;/,
    );
    expect(memoryCardCss).toMatch(
      /\.memory-card-meta\s*>\s*span\s*\{[^}]*min-inline-size:\s*0;[\s\S]*?overflow-wrap:\s*anywhere;/,
    );
  });
});
