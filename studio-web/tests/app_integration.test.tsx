import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/App';
import { MockWebSocket } from './setup';

describe('Full App workspace navigation & regression coverage', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('navigates cleanly across all 6 workspaces and modals without calling /api/events/history', async () => {
    const requestedUrls: string[] = [];

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      requestedUrls.push(url);

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
                approved_items: 2,
                total_items: 2,
                updated_at_utc: '2026-10-30T18:00:00Z',
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/studio/game-bible')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded synthetic living game bible.',
            has_bible: true,
            cartridge_id: 'candidate_synthetic_01',
            active_candidate_id: 'candidate_synthetic_01',
            status: 'slice_complete',
            game_title: 'Synthetic Isometric Builder',
            core_genre: 'Simulation Strategy',
            elevator_pitch: 'Deterministic resource routing across modular hex grids.',
            first_60_seconds_hook: 'Place two nodes and watch packets flow immediately.',
            color_palette_hex: ['#1B263B', '#415A77', '#E0E1DD'],
            active_slice_index: 1,
            slice_roadmap: [
              { slice_id: 'slice_0_foundation', slice_index: 0, title: 'Foundation Grid', status: 'completed' },
              { slice_id: 'slice_1_routing', slice_index: 1, title: 'Packet Routing', status: 'completed' },
            ],
            core_loop: { primary_action: 'Connect nodes', feedback_signal: 'Throughput pulse' },
            onboarding_plan: { step_1: 'Highlight starter socket' },
            goal_hierarchy: { short_term: 'Stabilize local hub' },
            visual_style_guide: 'Crisp high-contrast vector geometry with muted slate surfaces.',
            audio_direction_guide: 'Tactile relay clicks and low-pass harmonic hum.',
            technical_architecture_blueprint: [
              { subsystem: 'SimulationClock', responsibility: 'Deterministic 60 Hz step loop' },
            ],
            visual_mockups: [
              {
                mockup_id: 'mockup_1',
                title: 'Primary Grid Viewport',
                scene_description: 'Clean orthographic grid with subtle elevation.',
                image_path: 'assets/mockups/grid_view.png',
              },
            ],
            narrative_bible: {
              world_title: 'Modular Relay Network',
              setting_summary: 'An automated relay lattice.',
              characters: [{ character_id: 'char_1', name: 'Operator Unit', role: 'Dispatcher' }],
              dialogue_barks: [
                {
                  bark_id: 'b_1',
                  speaker_name: 'Operator Unit',
                  text: 'Throughput nominal.',
                  voice_clip: 'assets/audio/nominal.wav',
                },
              ],
            },
            writers_room_transcript: [
              {
                utterance_id: 'u_1',
                turn_index: 1,
                speaker_role: 'Systems Architect',
                spoken_dialogue: 'We should keep tick resolution deterministic at 60 Hz.',
                artifact_emitted: {
                  type: 'filmstrip',
                  path: '/srv/studio/cartridges/candidate_synthetic_01/artifacts/renders/anim_sheet_hub.png',
                  item_ids: ['item_grid_01'],
                },
                slice_id: 'slice_1_routing',
              },
            ],
            process_status: {
              process_state: 'stopped',
              is_process_alive: true,
              is_paused: false,
              pause_for_greenlight: false,
              build_actively_running: false,
              lock_held: false,
              active_threads: [],
              scheduler_running_workers: 0,
              orchestrator_state: 'IDLE',
            },
            production_board: {
              deliberation: 'Board state loaded.',
              cartridge_id: 'candidate_synthetic_01',
              design_revision: 1,
              total_items: 2,
              counts_by_state: { approved: 2 },
              is_drained: true,
              items: [
                {
                  deliberation: 'Grid module deliberation.',
                  item_id: 'item_grid_01',
                  kind: 'feature_code',
                  title: 'Build Orthographic Grid Controller',
                  spec: 'Implement grid coordinate projection.',
                  depends_on: [],
                  target_modules: ['scenes/grid/grid_controller.gd'],
                  slice_id: 'slice_1_routing',
                  state: 'approved',
                },
                {
                  deliberation: 'HUD module deliberation.',
                  item_id: 'item_hud_02',
                  kind: 'ui_polish',
                  title: 'Build Throughput Counter Bar',
                  spec: 'Display live packet throughput.',
                  depends_on: ['item_grid_01'],
                  target_modules: ['scripts/hud_bar.gd'],
                  slice_id: 'slice_1_routing',
                  state: 'approved',
                },
              ],
            },
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/artifacts/content')) {
        return new Response(
          JSON.stringify({
            filename: 'scenes/grid/grid_controller.gd',
            size_bytes: 128,
            content: 'extends Node2D\nfunc step_tick() -> void:\n\tpass\n',
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/artifacts')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Inventory loaded.',
            exists: true,
            manifest: {},
            sprites: [{ filename: 'node_hub.png', url: '/api/artifacts/content?filename=node_hub.png&raw=true', size_bytes: 2048 }],
            audio_clips: [{ filename: 'pulse_ok.wav', url: '/api/artifacts/content?filename=pulse_ok.wav&raw=true', size_bytes: 4096 }],
            code_modules: ['main.gd'],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/playtest/sessions/pt_session_01')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Session detail loaded.',
            session: {
              deliberation: 'Completed blind playtest.',
              session_id: 'pt_session_01',
              checkpoint_id: 'active',
              candidate_id: 'candidate_synthetic_01',
              slice_id: null,
              game_title: 'Synthetic Isometric Builder',
              status: 'completed',
              turn_count: 2,
              overall_fun_rating: 8,
              rulebook: { 'Left Click': 'Places or selects a relay node.' },
              field_notes: { 'HUD Layout': 'Top-left shows active throughput.' },
              questions: { q_1: { question: 'How do splitters prioritize lanes?' } },
              bugs_found: {},
              session_report: {
                deliberation: 'Evaluated player flow.',
                overall_experience_summary: 'Responsive controls and clear visual hierarchy.',
                overall_fun_rating: 8,
              },
            },
            turns: [
              {
                turn_id: 't_1',
                session_id: 'pt_session_01',
                turn_index: 1,
                deliberation: 'Observed starter hub on screen.',
                append_log_line: 'Clicked starter socket.',
                mood_state: 'curious',
              },
            ],
            compactions: [],
            wishes: [],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/playtest/sessions')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Sessions list loaded.',
            active_cartridge_id: 'candidate_synthetic_01',
            pending_wishes_count: 0,
            games_catalog: [],
            playable_targets: [
              {
                target_type: 'active',
                checkpoint_id: 'active',
                candidate_id: 'candidate_synthetic_01',
                label: 'Active Build',
                game_title: 'Synthetic Isometric Builder',
              },
            ],
            // Specifically verify sessions with `slice_id: null` are shown (legacy 0-runs bug fix!)
            sessions: [
              {
                deliberation: 'Completed blind playtest.',
                session_id: 'pt_session_01',
                checkpoint_id: 'active',
                candidate_id: 'candidate_synthetic_01',
                slice_id: null,
                game_title: 'Synthetic Isometric Builder',
                status: 'completed',
                turn_count: 2,
                matches_active_cartridge: true,
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/steam/dna')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded Steam DNA profile.',
            player_id: 'synthetic_player',
            profile_tag: 'default',
            total_playtime_hours: 1240,
            total_games_analyzed: 42,
            top_genres: ['Strategy', 'Simulation'],
            genre_weights: { Strategy: 0.65, Simulation: 0.35 },
            top_categories: ['Single-player'],
            category_weights: { 'Single-player': 1.0 },
            playtime_distribution: {},
            top_games: [{ name: 'Synthetic Factory Sim', playtime_hours: 320 }],
            activity_windows: {},
            pacing_preference: 'methodical',
            pacing_score: 0.72,
            challenge_tolerance: 0.8,
            thematic_biases: ['Systems'],
            thematic_weights: { Systems: 0.9 },
            mechanical_biases: { Automation: 0.85 },
            preferred_palette_mood: 'crisp industrial',
            palette_weights: {},
            created_at_utc: '2026-10-30T00:00:00Z',
            ingestion_provenance: {},
            live_writers_room_dna_directive: 'Prioritize clear systemic feedback loops.',
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/checkpoints')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Checkpoints loaded.',
            total_checkpoints: 1,
            latest_checkpoint_id: 'cp_01',
            checkpoints: [
              {
                deliberation: 'Verified slice checkpoint.',
                checkpoint_id: 'cp_01',
                candidate_id: 'candidate_synthetic_01',
                requirement_id: 'slice_1_routing',
                track_title: 'Packet Routing',
                created_at_utc: '2026-10-30T18:00:00Z',
                verified_tracks: ['slice_0_foundation', 'slice_1_routing'],
                cartridge_path: 'cartridges/cp_01',
                is_active: true,
                files: ['main.gd'],
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/studio/status')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Status healthy.',
            status: 'ok',
            active_cartridge: 'candidate_synthetic_01',
            cartridges: ['candidate_synthetic_01'],
            database_connected: true,
            db_event_count: 500,
            supervisor_state: 'IDLE',
            current_episode: 1,
            uptime_seconds: 3600,
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

    // 1. Production Board loads default active slice items
    await waitFor(() => {
      expect(screen.getByText('Build Orthographic Grid Controller')).toBeInTheDocument();
      expect(screen.getByText('Build Throughput Counter Bar')).toBeInTheDocument();
    });

    // 2. Writers' Room workspace
    fireEvent.click(screen.getByRole('tab', { name: /Writers' Room/i }));
    await waitFor(() => {
      expect(screen.getByText(/keep tick resolution deterministic at 60 Hz/i)).toBeInTheDocument();
    });

    // 3. Game Bible workspace
    fireEvent.click(screen.getByRole('tab', { name: /Game Bible/i }));
    await waitFor(() => {
      expect(screen.getByText('Primary Grid Viewport')).toBeInTheDocument();
      expect(screen.getByText('Modular Relay Network')).toBeInTheDocument();
    });

    // 4. Assets & Code workspace (verifies nested module union `scenes/grid/grid_controller.gd`)
    fireEvent.click(screen.getByRole('tab', { name: /Assets & Code/i }));
    await waitFor(() => {
      expect(screen.getByText('node_hub.png')).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole('tab', { name: /GDScript & Scenes/i }));
    await waitFor(() => {
      expect(screen.getByText('scenes/grid/grid_controller.gd')).toBeInTheDocument();
    });

    // 5. Playtester workspace (verifies session with `slice_id: null` is visible immediately!)
    fireEvent.click(screen.getByRole('tab', { name: /Playtester/i }));
    await waitFor(() => {
      expect(screen.getByText(/Clicked starter socket/i)).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole('button', { name: /Experience Report/i }));
    await waitFor(() => {
      expect(screen.getByText(/Responsive controls and clear visual hierarchy/i)).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole('button', { name: /Rulebook/i }));
    await waitFor(() => {
      expect(screen.getByText(/Places or selects a relay node/i)).toBeInTheDocument();
    });

    // 6. Player DNA workspace
    fireEvent.click(screen.getByRole('tab', { name: /Player DNA/i }));
    await waitFor(() => {
      expect(screen.getByText('Synthetic Factory Sim')).toBeInTheDocument();
    });

    // 7. Top Command Bar modals (Checkpoints & Live Pulse)
    fireEvent.click(screen.getByRole('button', { name: /Checkpoints/i }));
    await waitFor(() => {
      expect(screen.getByText('Verified Cartridge Checkpoints')).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole('button', { name: /Close dialog/i }));

    fireEvent.click(screen.getByRole('button', { name: /^Live$/i }));
    await waitFor(() => {
      expect(screen.getByText('Live Swarm WebSocket Stream')).toBeInTheDocument();
    });

    // 8. Verify `/api/events/history` (143.7 MB legacy endpoint) was NEVER requested
    expect(requestedUrls.some((u) => u.includes('/api/events/history'))).toBe(false);
  });

  it('automatically transitions from Running + Pause to Stopped + Resume when the server reloads and WebSocket reconnects', async () => {
    MockWebSocket.instances.length = 0;
    let serverPhase: 'running' | 'reloading' | 'reloaded_stopped' = 'running';

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;

      if (serverPhase === 'reloading') {
        return new Response('Bad Gateway', { status: 502 });
      }

      const buildRunningOnServer = serverPhase === 'running';

      if (url.includes('/api/cartridges')) {
        return new Response(
          JSON.stringify({
            cartridges: ['candidate_synthetic_01'],
            games: [
              {
                cartridge_id: 'candidate_synthetic_01',
                game_title: 'Synthetic Isometric Builder',
                core_genre: 'Simulation Strategy',
                status: buildRunningOnServer ? 'synthesizing' : 'stopped',
                has_backlog: true,
                approved_items: 1,
                total_items: 2,
                updated_at_utc: '2026-10-30T18:00:00Z',
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/studio/game-bible')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded synthetic living game bible.',
            has_bible: true,
            cartridge_id: 'candidate_synthetic_01',
            active_candidate_id: 'candidate_synthetic_01',
            status: buildRunningOnServer ? 'synthesizing' : 'stopped',
            game_title: 'Synthetic Isometric Builder',
            core_genre: 'Simulation Strategy',
            active_slice_index: 0,
            slice_roadmap: [
              { slice_id: 'slice_0_foundation', slice_index: 0, title: 'Foundation Grid', status: 'in_progress' },
            ],
            process_status: buildRunningOnServer
              ? {
                  process_state: 'running',
                  is_process_alive: true,
                  is_paused: false,
                  pause_for_greenlight: false,
                  build_actively_running: true,
                  lock_held: true,
                  active_threads: ['GenesisTrigger_job01'],
                  scheduler_running_workers: 1,
                  orchestrator_state: 'SYNTHESIZING',
                }
              : {
                  process_state: 'stopped',
                  is_process_alive: false,
                  is_paused: false,
                  pause_for_greenlight: false,
                  build_actively_running: false,
                  lock_held: false,
                  active_threads: [],
                  scheduler_running_workers: 0,
                  orchestrator_state: 'IDLE',
                },
            production_board: {
              deliberation: 'Board state loaded.',
              cartridge_id: 'candidate_synthetic_01',
              design_revision: 1,
              total_items: 2,
              counts_by_state: { approved: 1, in_progress: 1 },
              is_drained: false,
              items: [
                {
                  deliberation: 'Grid module deliberation.',
                  item_id: 'item_grid_01',
                  kind: 'feature_code',
                  title: 'Build Orthographic Grid Controller',
                  spec: 'Implement grid coordinate projection.',
                  depends_on: [],
                  target_modules: ['scenes/grid/grid_controller.gd'],
                  slice_id: 'slice_0_foundation',
                  state: 'approved',
                },
                {
                  deliberation: 'HUD module deliberation.',
                  item_id: 'item_hud_02',
                  kind: 'ui_polish',
                  title: 'Build Throughput Counter Bar',
                  spec: 'Display live packet throughput.',
                  depends_on: ['item_grid_01'],
                  target_modules: ['scripts/hud_bar.gd'],
                  slice_id: 'slice_0_foundation',
                  state: 'in_progress',
                },
              ],
            },
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/artifacts')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Inventory loaded.',
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
            deliberation: 'Sessions list loaded.',
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

    // Match main.tsx production defaults (staleTime: 10_000, refetchOnWindowFocus: false)
    const qc = new QueryClient({
      defaultOptions: {
        queries: {
          staleTime: 10_000,
          refetchOnWindowFocus: false,
          retry: false,
        },
      },
    });

    render(
      <QueryClientProvider client={qc}>
        <App />
      </QueryClientProvider>,
    );

    // 1. Initial render shows Running · 1w + Pause button
    await waitFor(() => {
      expect(screen.getByText('Running · 1w')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Pause$/i })).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: /^Resume$/i })).not.toBeInTheDocument();

    const ws1 = MockWebSocket.instances[0];
    expect(ws1).toBeDefined();

    // 2. Emit an active specialist step over WS and verify live step indicator appears
    act(() => {
      ws1.onmessage?.({
        data: JSON.stringify({
          deliberation: 'Specialist executing HUD implementation step.',
          event_type: 'backlog_item_specialist_started',
          agent_name: 'LeadGameplaySWEAgent',
          timestamp_utc: '2026-10-30T18:01:00Z',
          payload: {
            item_id: 'item_hud_02',
            turn_index: 0,
            specialist_agent: 'LeadGameplaySWEAgent',
            brief: 'Implementing live throughput bar',
          },
        }),
      });
    });

    await waitFor(() => {
      expect(screen.getAllByText(/Implementing live throughput bar/i).length).toBeGreaterThan(0);
    });

    // 3. Simulate Uvicorn --reload: backend goes down briefly (502), drops WS, then restarts in stopped state
    serverPhase = 'reloading';
    act(() => {
      ws1.close();
    });

    // Live worker step badge clears immediately upon socket disconnect
    await waitFor(() => {
      expect(screen.queryAllByText(/Implementing live throughput bar/i)).toHaveLength(0);
    });

    // Backend finishes reloading just before WS reconnect (~1s)
    serverPhase = 'reloaded_stopped';

    // Upon WS reconnect, queries are immediately invalidated and UI transitions to Stopped + Resume without refresh
    await waitFor(
      () => {
        expect(MockWebSocket.instances.length).toBeGreaterThanOrEqual(2);
        expect(screen.getByText('Stopped · 1/2')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /^Resume$/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Steer & Resume/i })).toBeInTheDocument();
      },
      { timeout: 3000 },
    );
    expect(screen.queryByText('Running · 1w')).not.toBeInTheDocument();
  });

  it('surfaces post-board QA Self-Healing Repair and live AI call heartbeat when all board items are approved', async () => {
    let historyCalls = 0;

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;

      if (url.includes('/api/events/history')) {
        historyCalls += 1;
        return new Response(
          JSON.stringify({
            deliberation: 'Seeded recent events for mid-flight active build.',
            count: 3,
            events: [
              {
                event_id: 'hist_1',
                deliberation: 'Frame budget gate exceeded threshold.',
                event_type: 'QA_GATE_RESULT',
                agent_name: 'QAPlaytesterBuildSupervisor',
                timestamp_utc: '2026-10-31T17:59:10Z',
                payload: {
                  attempt: 2,
                  max_retries: 3,
                  promoted: false,
                  failure_gate: 'Gate D: Frame Budget',
                  failure_reason: 'Peak frame spike 64.2ms > 16.6ms budget.',
                  remedy_directive: 'Cache per-tick spatial lookups outside _process.',
                },
              },
              {
                event_id: 'hist_2',
                deliberation: 'Starting automated self-healing repair pass.',
                event_type: 'AGENT_STREAM',
                agent_name: 'LeadGameplaySWEAgent',
                timestamp_utc: '2026-10-31T17:59:12Z',
                payload: {
                  phase: 'REPAIR',
                  failure_gate: 'Gate D: Frame Budget',
                  remedy_directive: 'Cache per-tick spatial lookups outside _process.',
                },
              },
              {
                event_id: 'hist_3',
                deliberation: 'Completed Gemini gateway turn.',
                event_type: 'studio_live_activity',
                agent_name: 'genai',
                timestamp_utc: '2026-10-31T17:59:45Z',
                payload: {
                  phase: 'llm_gateway_call',
                  label: 'LeadGameplaySWEAgent.pev_phase2[candidate_synthetic_01:iter2:hop1]',
                  total_calls: 14,
                  token_stats: {
                    prompt_tokens: 79000,
                    output_tokens: 2100,
                    thought_tokens: 3100,
                    total_tokens: 84200,
                  },
                },
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/cartridges')) {
        return new Response(
          JSON.stringify({
            cartridges: ['candidate_synthetic_01'],
            games: [
              {
                cartridge_id: 'candidate_synthetic_01',
                game_title: 'Synthetic Isometric Builder',
                core_genre: 'Simulation Strategy',
                status: 'building',
                has_backlog: true,
                approved_items: 2,
                total_items: 2,
                updated_at_utc: '2026-10-31T17:59:00Z',
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/studio/game-bible')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded active post-board build.',
            has_bible: true,
            cartridge_id: 'candidate_synthetic_01',
            active_candidate_id: 'candidate_synthetic_01',
            status: 'building',
            game_title: 'Synthetic Isometric Builder',
            core_genre: 'Simulation Strategy',
            active_slice_index: 0,
            slice_roadmap: [
              { slice_id: 'slice_0_foundation', slice_index: 0, title: 'Foundation Grid', status: 'in_progress' },
            ],
            writers_room_transcript: [
              {
                utterance_id: 'u_done_1',
                turn_index: 1,
                speaker_role: 'Systems Architect',
                spoken_dialogue: 'All slice board modules have passed review; handing off to QA verification.',
                slice_id: 'slice_0_foundation',
              },
            ],
            process_status: {
              process_state: 'running',
              is_process_alive: true,
              is_paused: false,
              pause_for_greenlight: false,
              build_actively_running: true,
              lock_held: true,
              active_threads: ['GenesisTrigger_unpause_test'],
              scheduler_running_workers: 0,
              orchestrator_state: 'BUILDING',
              last_event_timestamp_utc: '2026-10-31T17:59:45Z',
            },
            production_board: {
              deliberation: 'All items approved.',
              cartridge_id: 'candidate_synthetic_01',
              design_revision: 1,
              total_items: 2,
              counts_by_state: { approved: 2 },
              is_drained: true,
              items: [
                {
                  deliberation: 'Grid module.',
                  item_id: 'item_grid_01',
                  kind: 'feature_code',
                  title: 'Build Orthographic Grid Controller',
                  spec: 'Implement grid coordinate projection.',
                  depends_on: [],
                  target_modules: ['scenes/grid/grid_controller.gd'],
                  slice_id: 'slice_0_foundation',
                  state: 'approved',
                },
                {
                  deliberation: 'HUD module.',
                  item_id: 'item_hud_02',
                  kind: 'ui_polish',
                  title: 'Build Throughput Counter Bar',
                  spec: 'Display live packet throughput.',
                  depends_on: ['item_grid_01'],
                  target_modules: ['scripts/hud_bar.gd'],
                  slice_id: 'slice_0_foundation',
                  state: 'approved',
                },
              ],
            },
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/artifacts')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Inventory loaded.',
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
            deliberation: 'Sessions list loaded.',
            active_cartridge_id: 'candidate_synthetic_01',
            pending_wishes_count: 0,
            games_catalog: [],
            playable_targets: [],
            sessions: [],
          }),
          { status: 200 },
        );
      }

      return new Response(JSON.stringify({ deliberation: 'ok', db_event_count: 120 }), { status: 200 });
    });

    const qc = new QueryClient({
      defaultOptions: {
        queries: { staleTime: 10_000, refetchOnWindowFocus: false, retry: false },
      },
    });

    render(
      <QueryClientProvider client={qc}>
        <App />
      </QueryClientProvider>,
    );

    // TopCommandBar + LiveActivityBanner + ProductionBoardView explain post-board QA repair + AI heartbeat
    await waitFor(() => {
      expect(screen.getByText('Running · QA Repair · 2/3')).toBeInTheDocument();
      expect(screen.getAllByText('QA Self-Healing Repair (Attempt 2/3)').length).toBeGreaterThanOrEqual(2);
      expect(screen.getByText('Board Complete (2/2 Approved)')).toBeInTheDocument();
      expect(screen.getByText(/AI #14 · pev_phase2 \(iter2:hop1\) · 84\.2k tok/)).toBeInTheDocument();
    });

    // History seed must run strictly once on initial attach (no polling!)
    expect(historyCalls).toBe(1);

    // Switch to Writers' Room and confirm transcript footer explains why deliberation is quiet
    fireEvent.click(screen.getByRole('tab', { name: /Writers' Room/i }));
    await waitFor(() => {
      expect(
        screen.getByText(/Council deliberation paused while Studio executes QA Self-Healing Repair/i),
      ).toBeInTheDocument();
    });

    // Open Live Stream modal and verify filter pills + formatted AI & QA gate rows
    fireEvent.click(screen.getByRole('button', { name: /Inspect Stream/i }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Stages & QA Gates/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /AI Gateway Calls/i })).toBeInTheDocument();
      expect(screen.getByText(/AI Call #14: pev_phase2 \(iter2:hop1\)/)).toBeInTheDocument();
      expect(screen.getByText(/Gate D: Frame Budget rejected \(Attempt 2\/3\)/)).toBeInTheDocument();
    });
  });
});
