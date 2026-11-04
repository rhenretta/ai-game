import { describe, expect, it, vi, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { App } from '../src/App';
import { resolveMediaUrl, studioApi } from '../src/api/client';
import { CoProducerView } from '../src/features/coproducer/CoProducerView';
import type { CoProducerMessage, CoProducerThread } from '../src/api/types';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('Co-Producer API client & workspace UI', () => {
  it('resolves draft sketch media URLs without prepending selected cartridgeId and serializes deliberation at index 0', async () => {
    const draftUrl = resolveMediaUrl(
      '_coproducer_drafts/draft_synth01/assets/sketches/mockup_01.png',
      'candidate_genesis_01',
    );
    expect(draftUrl).toContain(
      encodeURIComponent('_coproducer_drafts/draft_synth01/assets/sketches/mockup_01.png'),
    );
    expect(draftUrl).not.toContain('candidate_genesis_01');

    const capturedBodies: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
      if (init?.body) {
        capturedBodies.push(String(init.body));
      }
      return new Response(
        JSON.stringify({
          deliberation: 'ok',
          thread: {
            deliberation: 'ok',
            thread_id: 'draft_synth01',
            scope_kind: 'draft',
            cartridge_id: null,
            title: 'Synthetic Draft',
            concept_brief: null,
            message_count: 0,
            created_at_utc: '2026-11-02T12:00:00Z',
            updated_at_utc: '2026-11-02T12:00:00Z',
          },
          messages: [],
        }),
        { status: 200 },
      );
    });

    await studioApi.createCoProducerDraft('Synthetic Draft');
    await studioApi.sendCoProducerMessage('draft_synth01', {
      message: 'Refine core mechanics',
      selectedSketchPaths: ['_coproducer_drafts/draft_synth01/assets/sketches/mockup_01.png'],
    });
    await studioApi.executeCoProducerAction({
      threadId: 'draft_synth01',
      messageId: 'msg_01',
      proposalId: 'prop_01',
    });

    expect(capturedBodies).toHaveLength(3);
    for (const raw of capturedBodies) {
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      expect(Object.keys(parsed)[0]).toBe('deliberation');
    }
  });

  it('renders rationale-first assistant turns, lazy-loaded historical media, consultation cards, sketch pinning, and 1-click game commissioning', async () => {
    const sampleThread: CoProducerThread = {
      deliberation: 'Persisted draft thread state.',
      thread_id: 'draft_synth_alpha',
      scope_kind: 'draft',
      cartridge_id: null,
      title: 'Synthetic Concept Alpha',
      concept_brief: {
        deliberation: 'Distilled concept pillars from user dialogue.',
        working_title: 'Synthetic Concept Alpha',
        core_genre: 'Systems Sandbox',
        elevator_pitch: 'Compact single-screen spatial feedback loop with immediate tactile response.',
        core_fantasy: 'Conduct harmonic flow across a responsive spatial board.',
        core_player_verbs: ['place', 'rotate', 'harmonize'],
        first_60_seconds_feel: 'Crisp low-latency snapping and harmonic audio cues.',
        visual_style_direction: 'High-contrast geometric silhouettes.',
        audio_mood_direction: 'Warm analog synth pulses.',
        color_palette_hex: ['#0f172a', '#38bdf8'],
        explicit_non_goals: ['No multi-menu inventory crafting'],
        selected_mockup_paths: ['_coproducer_drafts/draft_synth_alpha/assets/sketches/sketch_01.png'],
        selected_sketch_asset_paths: [],
      },
      message_count: 2,
      created_at_utc: '2026-11-02T12:00:00Z',
      updated_at_utc: '2026-11-02T12:05:00Z',
    };

    const historicalAssistantMsg: CoProducerMessage = {
      deliberation: 'Earlier turn curating playtest broadcast replay.',
      message_id: 'msg_hist_1',
      thread_id: 'draft_synth_alpha',
      turn_index: 1,
      sender_role: 'coproducer',
      content_markdown: 'Here is the prior playtest replay and audio cue for comparison.',
      media_cards: [
        {
          deliberation: 'Selected playtest video and synthesized cue.',
          card_id: 'mc_hist_1',
          card_kind: 'playtest_video',
          title: 'Prior Playtest Broadcast',
          subtitle: '1600x720 playtest replay with telemetry sidebar.',
          deep_link_workspace: 'playtest',
          deep_link_target_id: 'pt_session_01',
          items: [
            {
              deliberation: 'Broadcast MP4.',
              item_id: 'vid_hist_1',
              label: 'Session Replay',
              media_kind: 'video',
              artifact_path: '/api/playtest/sessions/pt_session_01/video',
              width: 1600,
              height: 720,
            },
            {
              deliberation: 'Audio loop.',
              item_id: 'aud_hist_1',
              label: 'Pulse Audition',
              media_kind: 'audio',
              artifact_path: '_coproducer_drafts/draft_synth_alpha/assets/sketches/pulse.wav',
            },
          ],
        },
      ],
      consultation_cards: [],
      action_proposals: [],
      tool_trace: [],
      prompt_trace: [],
      created_at_utc: '2026-11-02T12:01:00Z',
    };

    const latestAssistantMsg: CoProducerMessage = {
      deliberation: 'Synthesized Council pitch options, concept sketch, and commission handoff.',
      message_id: 'msg_latest_2',
      thread_id: 'draft_synth_alpha',
      turn_index: 2,
      sender_role: 'coproducer',
      content_markdown:
        'I huddled with the Council, rendered a 16:9 concept sketch, and prepared the refined handoff.',
      media_cards: [
        {
          deliberation: 'Generated 16:9 concept plate and latest clip.',
          card_id: 'mc_latest_2',
          card_kind: 'sketch_showcase',
          title: 'Primary Concept Sketch',
          subtitle: 'Orthographic single-screen layout sketch.',
          items: [
            {
              deliberation: '16:9 mockup image.',
              item_id: 'img_latest_1',
              label: 'Mockup Plate A',
              media_kind: 'image',
              artifact_path: '_coproducer_drafts/draft_synth_alpha/assets/sketches/sketch_01.png',
              width: 1280,
              height: 720,
            },
            {
              deliberation: 'Latest active video preview.',
              item_id: 'vid_latest_1',
              label: 'Latest Motion Proof',
              media_kind: 'video',
              artifact_path: '/api/playtest/sessions/pt_session_02/video',
              width: 1600,
              height: 720,
            },
          ],
        },
      ],
      consultation_cards: [
        {
          deliberation: 'Convened divergent pitch jam across Council.',
          card_id: 'cons_01',
          consultation_kind: 'divergent_pitch_round',
          question_posed: 'Propose contrasting mechanical hooks.',
          participating_roles: ['CreativeDirector', 'GameplayDesigner'],
          synthesis_headline: 'Council Divergent Pitch Jam',
          divergent_pitches: [
            {
              deliberation: 'Option A rationale.',
              pitch_id: 'pitch_a',
              working_title: 'Harmonic Relay',
              core_genre: 'Spatial Puzzle x Rhythm Synth',
              hook_summary: 'Snap relay nodes onto phase lines to harmonize throughput.',
              core_loop_sketch: 'Place relays, align phase pulses, stabilize resonance.',
              visual_and_audio_vibe: 'Crisp vector lines with warm analog chords.',
            },
          ],
        },
      ],
      action_proposals: [
        {
          deliberation: 'Concept pillars are concrete; ready for refined_concept handoff.',
          proposal_id: 'prop_commission_01',
          action_kind: 'commission_game',
          handoff_mode: 'refined_concept',
          title: 'Commission Refined Concept',
          summary: 'Graduates draft into candidate_genesis_new01 and skips blind genesis.',
          status: 'pending',
          concept_brief: sampleThread.concept_brief,
        },
      ],
      tool_trace: [
        {
          deliberation: 'Needed contrasting pitches.',
          tool_name: 'consult_council',
          query_or_question: 'Propose contrasting mechanical hooks.',
          observation_excerpt: 'Generated 3 divergent pitches.',
        },
      ],
      prompt_trace: [
        {
          deliberation: 'High-creativity synthesis step.',
          step_label: 'CoProducerAgent.final_turn[creative_ideation]',
          model: 'gemini-3-flash-preview',
          temperature: 0.9,
          prompt_excerpt: '## SYSTEM ROLE\nCreative Co-Producer synthesis prompt.',
        },
      ],
      token_usage: { total_tokens: 4820 },
      created_at_utc: '2026-11-02T12:05:00Z',
    };

    const onSelectCartridge = vi.fn();
    const onSelectWorkspace = vi.fn();
    const onOpenLightbox = vi.fn();

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const rawUrl = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const url = decodeURIComponent(rawUrl);
      if (url.includes('/api/coproducer/threads/draft_synth_alpha')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded draft thread detail.',
            thread: sampleThread,
            messages: [historicalAssistantMsg, latestAssistantMsg],
          }),
          { status: 200 },
        );
      }
      if (url.includes('/api/coproducer/threads')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Listed Co-Producer threads.',
            active_cartridge_thread: null,
            draft_threads: [sampleThread],
            cartridge_threads: [],
            threads: [sampleThread],
          }),
          { status: 200 },
        );
      }
      if (url.includes('/api/coproducer/actions/execute')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Executed refined_concept handoff.',
            status: 'executed',
            action_kind: 'commission_game',
            result: {
              target_cartridge_id: 'candidate_genesis_new01',
              target_thread_id: 'thread_candidate_genesis_new01',
              handoff_mode: 'refined_concept',
            },
          }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ deliberation: 'ok' }), { status: 200 });
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={qc}>
        <CoProducerView
          cartridgeId="active"
          hasAnyCartridge={false}
          boardCounts={{ approved: 0, total: 0 }}
          liveStepsByThread={{}}
          onSelectCartridge={onSelectCartridge}
          onSelectWorkspace={onSelectWorkspace}
          onOpenLightbox={onOpenLightbox}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(
        screen.getByText(/I huddled with the Council, rendered a 16:9 concept sketch/i),
      ).toBeInTheDocument();
    });

    // Verify Rule 4 Epistemic Rationale disclosure is present
    expect(screen.getAllByText('Co-Producer Epistemic Rationale').length).toBeGreaterThanOrEqual(2);

    // Verify progressive media lazy loading (modern-web-guidance: lazy-load-video-and-audio):
    // Historical video/audio have loading="lazy", while latest message's video omits loading="lazy"
    const videos = container.querySelectorAll('video');
    expect(videos).toHaveLength(2);
    expect(videos[0].getAttribute('loading')).toBe('lazy');
    expect(videos[0].getAttribute('preload')).toBe('metadata');
    expect(videos[0].getAttribute('width')).toBe('1600');
    expect(videos[0].getAttribute('height')).toBe('720');
    expect(videos[1].getAttribute('loading')).toBeNull();

    const audios = container.querySelectorAll('audio');
    expect(audios).toHaveLength(1);
    expect(audios[0].getAttribute('loading')).toBe('lazy');
    expect(audios[0].getAttribute('preload')).toBe('metadata');
    expect(audios[0].hasAttribute('controls')).toBe(true);

    // Verify sketch pinning toggle adds/removes pinned ref badge in composer
    fireEvent.click(screen.getByRole('button', { name: /Pin Ref/i }));
    expect(screen.getByText(/Pinned Sketch Refs:/i)).toBeInTheDocument();

    // Verify Consultation Card disclosure renders divergent pitch option
    expect(screen.getByText('Council Divergent Pitch Jam')).toBeInTheDocument();
    expect(screen.getByText('Harmonic Relay')).toBeInTheDocument();

    // Verify 1-click Action Proposal execution switches Mission Control to newly commissioned cartridge
    fireEvent.click(screen.getByRole('button', { name: /Commission Refined Concept/i }));
    await waitFor(() => {
      expect(onSelectCartridge).toHaveBeenCalledWith('candidate_genesis_new01');
    });
  });

  it('surfaces authentic Rule 5 HTTP 503 API errors via ErrorBanner without silent fallback', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
      return new Response(
        JSON.stringify({
          deliberation: 'Interactive Co-Producer requires an authenticated Gemini client.',
          error: 'GEMINI_API_KEY is not configured in environment.',
        }),
        { status: 503 },
      );
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <CoProducerView
          cartridgeId="candidate_01"
          hasAnyCartridge={true}
          gameTitle="Synthetic Build"
          boardCounts={{ approved: 1, total: 2 }}
          liveStepsByThread={{}}
          onSelectCartridge={vi.fn()}
          onSelectWorkspace={vi.fn()}
          onOpenLightbox={vi.fn()}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/HTTP 503/i);
      expect(screen.getByRole('alert')).toHaveTextContent(/GEMINI_API_KEY is not configured/i);
    });
  });

  it('consolidates top-bar New Game directly into Co-Producer Concept Lab without a modal, supports 1-click Run Autonomous Build, and guards IME Enter composition', async () => {
    const draftCalls: Array<Record<string, unknown>> = [];
    const genesisCalls: Array<Record<string, unknown>> = [];
    const messageCalls: Array<Record<string, unknown>> = [];

    const freshDraft: CoProducerThread = {
      deliberation: 'Created fresh Concept Lab draft.',
      thread_id: 'draft_consolidated_01',
      scope_kind: 'draft',
      cartridge_id: null,
      title: 'New Game Concept',
      concept_brief: null,
      message_count: 0,
      created_at_utc: '2026-11-02T18:00:00Z',
      updated_at_utc: '2026-11-02T18:00:00Z',
    };

    const existingCartridgeThread: CoProducerThread = {
      deliberation: 'Existing cartridge showrunner thread.',
      thread_id: 'thread_candidate_01',
      scope_kind: 'cartridge',
      cartridge_id: 'candidate_01',
      title: 'Synthetic Active Cartridge',
      concept_brief: null,
      message_count: 1,
      created_at_utc: '2026-11-02T17:00:00Z',
      updated_at_utc: '2026-11-02T17:05:00Z',
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const rawUrl =
        typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const url = decodeURIComponent(rawUrl);
      const method = (init?.method || 'GET').toUpperCase();

      if (url.includes('/api/cartridges')) {
        return new Response(
          JSON.stringify({
            cartridges: ['candidate_01'],
            games: [
              {
                cartridge_id: 'candidate_01',
                game_title: 'Synthetic Active Cartridge',
                core_genre: 'Simulation',
                status: 'slice_complete',
                has_backlog: true,
                approved_items: 2,
                total_items: 2,
                updated_at_utc: '2026-11-02T17:00:00Z',
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/studio/game-bible')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded game bible.',
            has_bible: true,
            cartridge_id: 'candidate_01',
            active_candidate_id: 'candidate_01',
            game_title: 'Synthetic Active Cartridge',
            pause_for_greenlight: false,
            writers_room_transcript: [],
            production_board: {
              deliberation: 'Loaded board.',
              cartridge_id: 'candidate_01',
              design_revision: 1,
              total_items: 2,
              counts_by_state: { approved: 2 },
              is_drained: true,
              items: [],
            },
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
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/playtest/sessions')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded sessions.',
            active_cartridge_id: 'candidate_01',
            pending_wishes_count: 0,
            games_catalog: [],
            playable_targets: [],
            sessions: [],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/artifacts')) {
        return new Response(
          JSON.stringify({ sprites: [], audio_clips: [], code_modules: [] }),
          { status: 200 },
        );
      }

      if (url.includes('/api/coproducer/threads/draft') && method === 'POST') {
        draftCalls.push(JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>);
        return new Response(
          JSON.stringify({
            deliberation: 'Created fresh Concept Lab draft.',
            thread: freshDraft,
            messages: [],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/messages') && method === 'POST') {
        messageCalls.push(JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>);
        return new Response(
          JSON.stringify({
            deliberation: 'Processed message.',
            thread: freshDraft,
            messages: [],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/coproducer/threads/draft_consolidated_01')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded fresh Concept Lab draft.',
            thread: freshDraft,
            messages: [],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/coproducer/threads/thread_candidate_01')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded cartridge thread.',
            thread: existingCartridgeThread,
            messages: [
              {
                deliberation: 'Existing cartridge welcome.',
                message_id: 'msg_existing_01',
                thread_id: 'thread_candidate_01',
                turn_index: 1,
                sender_role: 'coproducer',
                content_markdown: 'Ready to inspect Synthetic Active Cartridge.',
                media_cards: [],
                consultation_cards: [],
                action_proposals: [],
                tool_trace: [],
                prompt_trace: [],
                created_at_utc: '2026-11-02T17:05:00Z',
              },
            ],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/coproducer/threads')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Listed threads.',
            active_cartridge_thread: existingCartridgeThread,
            draft_threads: draftCalls.length > 0 ? [freshDraft] : [],
            cartridge_threads: [existingCartridgeThread],
            threads:
              draftCalls.length > 0
                ? [freshDraft, existingCartridgeThread]
                : [existingCartridgeThread],
          }),
          { status: 200 },
        );
      }

      if (url.includes('/api/genesis/trigger') && method === 'POST') {
        genesisCalls.push(JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>);
        return new Response(
          JSON.stringify({
            deliberation: 'Triggered autonomous genesis build.',
            status: 'started',
            job_id: 'job_auto_02',
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

    // Eliminate duplicate New Game Idea button in CoProducerView
    expect(screen.queryByRole('button', { name: /New Game Idea/i })).not.toBeInTheDocument();

    // Navigate to Board, then click top-bar New Game
    fireEvent.click(screen.getByRole('tab', { name: /^Board/i }));
    expect(screen.getByRole('tab', { name: /^Board/i })).toHaveAttribute('aria-selected', 'true');

    fireEvent.click(screen.getByRole('button', { name: /New Game/i }));

    // No modal opens; routes directly to Tab #1 Co-Producer Concept Lab with fresh draft
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    await waitFor(() => {
      expect(draftCalls).toHaveLength(1);
      expect(Object.keys(draftCalls[0])[0]).toBe('deliberation');
      expect(screen.getByRole('tab', { name: /Co-Producer/i })).toHaveAttribute(
        'aria-selected',
        'true',
      );
      expect(
        screen.getByText('Shape a New Game Idea with Your Creative Co-Producer'),
      ).toBeInTheDocument();
    });

    // Composer has an associated <label> and guards IME composition on Enter (isComposing / keyCode 229)
    const composer = screen.getByLabelText('Message Creative Co-Producer');
    fireEvent.change(composer, {
      target: { value: 'Tactile clockwork signal routing on a compact brass board' },
    });

    fireEvent.keyDown(composer, { key: 'Enter', isComposing: true });
    fireEvent.keyDown(composer, { key: 'Enter', keyCode: 229 });
    expect(messageCalls).toHaveLength(0);

    // 1-click Run Autonomous Build uses composer seed + Greenlight Gate toggle without opening a modal
    const greenlightCheckbox = screen.getByLabelText(/Pause at Greenlight Gate/i);
    fireEvent.click(greenlightCheckbox);
    fireEvent.click(screen.getByRole('button', { name: /Run Autonomous Build/i }));

    await waitFor(() => {
      expect(genesisCalls).toHaveLength(1);
      expect(Object.keys(genesisCalls[0])[0]).toBe('deliberation');
      expect(genesisCalls[0].creative_prompt).toBe(
        'Tactile clockwork signal routing on a compact brass board',
      );
      expect(genesisCalls[0].pause_for_greenlight).toBe(true);
      expect(genesisCalls[0].reset_active).toBe(true);
      expect(screen.getByRole('tab', { name: /Writers' Room/i })).toHaveAttribute(
        'aria-selected',
        'true',
      );
    });
  });

  it('exposes a persistent Start Game Build button in the Concept Lab header during an active conversation even before an inline proposal card is emitted', async () => {
    const executeBodies: Array<Record<string, unknown>> = [];
    const onSelectCartridge = vi.fn();

    const activeDraft: CoProducerThread = {
      deliberation: 'Active Concept Lab draft with shaped concept brief.',
      thread_id: 'draft_in_progress_01',
      scope_kind: 'draft',
      cartridge_id: null,
      title: 'Clockwork Signal',
      concept_brief: {
        deliberation: 'Shaped pillars from ongoing conversation.',
        working_title: 'Clockwork Signal',
        core_genre: 'Spatial Puzzle',
        elevator_pitch: 'Route brass signal pulses across rotating gear junctions.',
        core_fantasy: 'Master a tactile brass relay board.',
        core_player_verbs: ['rotate_junction', 'emit_pulse'],
        first_60_seconds_feel: 'Crisp mechanical clicks and glowing pulse arcs.',
        visual_style_direction: 'Warm brass silhouettes on dark slate.',
        audio_mood_direction: 'Tactile clockwork chimes.',
        color_palette_hex: ['#0f172a', '#f59e0b'],
        explicit_non_goals: ['No inventory menus'],
        selected_mockup_paths: [],
        selected_sketch_asset_paths: [],
      },
      message_count: 2,
      created_at_utc: '2026-11-02T19:00:00Z',
      updated_at_utc: '2026-11-02T19:05:00Z',
    };

    const userMsg: CoProducerMessage = {
      deliberation: 'User message.',
      message_id: 'msg_u1',
      thread_id: 'draft_in_progress_01',
      turn_index: 1,
      sender_role: 'user',
      content_markdown: 'Let us explore a brass clockwork signal routing puzzle.',
      media_cards: [],
      consultation_cards: [],
      action_proposals: [],
      tool_trace: [],
      prompt_trace: [],
      created_at_utc: '2026-11-02T19:01:00Z',
    };

    const assistantMsgNoProposalYet: CoProducerMessage = {
      deliberation: 'Riffing on core verbs without an inline proposal card yet.',
      message_id: 'msg_a2',
      thread_id: 'draft_in_progress_01',
      turn_index: 2,
      sender_role: 'coproducer',
      content_markdown: 'Love that tactile brass feel—rotating junctions give immediate visual payoff.',
      media_cards: [],
      consultation_cards: [],
      action_proposals: [],
      tool_trace: [],
      prompt_trace: [],
      created_at_utc: '2026-11-02T19:02:00Z',
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const rawUrl =
        typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const url = decodeURIComponent(rawUrl);
      if (url.includes('/api/coproducer/threads/draft_in_progress_01')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Loaded active draft.',
            thread: activeDraft,
            messages: [userMsg, assistantMsgNoProposalYet],
          }),
          { status: 200 },
        );
      }
      if (url.includes('/api/coproducer/threads')) {
        return new Response(
          JSON.stringify({
            deliberation: 'Listed drafts.',
            active_cartridge_thread: null,
            draft_threads: [activeDraft],
            cartridge_threads: [],
            threads: [activeDraft],
          }),
          { status: 200 },
        );
      }
      if (url.includes('/api/coproducer/actions/execute')) {
        executeBodies.push(JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>);
        return new Response(
          JSON.stringify({
            deliberation: 'Graduated draft directly.',
            status: 'executed',
            action_kind: 'commission_game',
            result: {
              target_cartridge_id: 'candidate_genesis_direct01',
              target_thread_id: 'draft_in_progress_01',
              handoff_mode: 'refined_concept',
            },
          }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ deliberation: 'ok' }), { status: 200 });
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <CoProducerView
          cartridgeId="active"
          hasAnyCartridge={false}
          boardCounts={{ approved: 0, total: 0 }}
          liveStepsByThread={{}}
          onSelectCartridge={onSelectCartridge}
          onSelectWorkspace={vi.fn()}
          onOpenLightbox={vi.fn()}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Start Game Build/i })).toBeInTheDocument();
    });

    // Defaults to Refined Concept once working_title is shaped
    const modeSelect = screen.getByLabelText('Council Handoff Mode') as HTMLSelectElement;
    expect(modeSelect.value).toBe('refined_concept');

    fireEvent.click(screen.getByRole('button', { name: /Start Game Build/i }));

    await waitFor(() => {
      expect(executeBodies).toHaveLength(1);
      expect(Object.keys(executeBodies[0])[0]).toBe('deliberation');
      expect(executeBodies[0].direct_action_kind).toBe('commission_game');
      expect(executeBodies[0].override_handoff_mode).toBe('refined_concept');
      expect(onSelectCartridge).toHaveBeenCalledWith('candidate_genesis_direct01');
    });
  });

  it('treats games as Co-Producer sessions, clears composer synchronously with optimistic history, pins scroll to current turn, and tucks background detail behind Behind the scenes', async () => {
    const pendingSendResolvers: Array<(res: Response) => void> = [];

    const threadAlpha: CoProducerThread = {
      deliberation: 'Session Alpha thread.',
      thread_id: 'thread_candidate_alpha',
      scope_kind: 'cartridge',
      cartridge_id: 'candidate_alpha',
      title: 'Session Alpha',
      concept_brief: null,
      message_count: 1,
      created_at_utc: '2026-11-03T10:00:00Z',
      updated_at_utc: '2026-11-03T10:05:00Z',
    };

    const threadBeta: CoProducerThread = {
      deliberation: 'Session Beta thread.',
      thread_id: 'thread_candidate_beta',
      scope_kind: 'cartridge',
      cartridge_id: 'candidate_beta',
      title: 'Session Beta',
      concept_brief: null,
      message_count: 1,
      created_at_utc: '2026-11-03T09:00:00Z',
      updated_at_utc: '2026-11-03T09:30:00Z',
    };

    const freshDraft: CoProducerThread = {
      deliberation: 'Fresh session draft.',
      thread_id: 'draft_fresh_session',
      scope_kind: 'draft',
      cartridge_id: null,
      title: 'New Game Concept',
      concept_brief: null,
      message_count: 0,
      created_at_utc: '2026-11-03T11:00:00Z',
      updated_at_utc: '2026-11-03T11:00:00Z',
    };

    const betaAssistantMsg: CoProducerMessage = {
      deliberation: 'Checked background telemetry silently before giving concise reply.',
      message_id: 'msg_beta_1',
      thread_id: 'thread_candidate_beta',
      turn_index: 1,
      sender_role: 'coproducer',
      content_markdown: 'Session Beta core loop is responsive; here is the requested visual plate.',
      media_cards: [
        {
          deliberation: 'Background progress check.',
          card_id: 'mc_bg_progress',
          card_kind: 'progress_summary',
          title: 'Background Progress Metrics',
          show_by_default: false,
          items: [
            {
              deliberation: 'Metric tile.',
              item_id: 'm_1',
              media_kind: 'metric',
              label: 'Approved',
              metric_value: '4/4',
            },
          ],
        },
        {
          deliberation: 'Explicitly requested visual sketch.',
          card_id: 'mc_fg_sketch',
          card_kind: 'sketch_showcase',
          title: 'Presented Sketch Plate',
          show_by_default: true,
          items: [
            {
              deliberation: 'Sketch image.',
              item_id: 'img_fg_1',
              media_kind: 'image',
              label: 'Chamber Plate',
              artifact_path: 'assets/mockups/plate.png',
              width: 1280,
              height: 720,
            },
          ],
        },
      ],
      consultation_cards: [],
      action_proposals: [],
      tool_trace: [
        {
          deliberation: 'Checked board silently.',
          tool_name: 'inspect_studio_progress',
          observation_excerpt: '4/4 approved',
        },
      ],
      prompt_trace: [],
      token_usage: { total_tokens: 1920 },
      created_at_utc: '2026-11-03T09:30:00Z',
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
      const rawUrl =
        typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const url = decodeURIComponent(rawUrl);
      const method = (init?.method || 'GET').toUpperCase();

      if (url.includes('/api/cartridges')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              cartridges: ['candidate_alpha', 'candidate_beta'],
              games: [
                {
                  cartridge_id: 'candidate_alpha',
                  game_title: 'Session Alpha',
                  core_genre: 'Puzzle',
                  status: 'slice_complete',
                  has_backlog: true,
                  approved_items: 2,
                  total_items: 2,
                  updated_at_utc: '2026-11-03T10:05:00Z',
                },
                {
                  cartridge_id: 'candidate_beta',
                  game_title: 'Session Beta',
                  core_genre: 'Strategy',
                  status: 'slice_complete',
                  has_backlog: true,
                  approved_items: 4,
                  total_items: 4,
                  updated_at_utc: '2026-11-03T09:30:00Z',
                },
              ],
            }),
            { status: 200 },
          ),
        );
      }

      if (url.includes('/api/studio/game-bible')) {
        const isBeta = url.includes('candidate_beta');
        return Promise.resolve(
          new Response(
            JSON.stringify({
              deliberation: 'Loaded bible.',
              has_bible: true,
              cartridge_id: isBeta ? 'candidate_beta' : 'candidate_alpha',
              active_candidate_id: 'candidate_alpha',
              game_title: isBeta ? 'Session Beta' : 'Session Alpha',
              writers_room_transcript: [],
              production_board: {
                deliberation: 'Board.',
                cartridge_id: isBeta ? 'candidate_beta' : 'candidate_alpha',
                design_revision: 1,
                total_items: isBeta ? 4 : 2,
                counts_by_state: { approved: isBeta ? 4 : 2 },
                is_drained: true,
                items: [],
              },
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
            }),
            { status: 200 },
          ),
        );
      }

      if (url.includes('/api/playtest/sessions')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              deliberation: 'Loaded playtest sessions.',
              active_cartridge_id: 'candidate_alpha',
              pending_wishes_count: 0,
              games_catalog: [],
              playable_targets: [],
              sessions: [],
            }),
            { status: 200 },
          ),
        );
      }

      if (url.includes('/api/artifacts')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ sprites: [], audio_clips: [], code_modules: [] }),
            { status: 200 },
          ),
        );
      }

      if (url.includes('/api/coproducer/threads/draft') && method === 'POST') {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              deliberation: 'Created draft.',
              thread: freshDraft,
              messages: [],
            }),
            { status: 200 },
          ),
        );
      }

      if (url.includes('/api/coproducer/threads/draft_fresh_session')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              deliberation: 'Loaded fresh draft.',
              thread: freshDraft,
              messages: [],
            }),
            { status: 200 },
          ),
        );
      }

      if (url.includes('/messages') && method === 'POST') {
        return new Promise<Response>((resolve) => {
          pendingSendResolvers.push(resolve);
        });
      }

      if (url.includes('/api/coproducer/threads/thread_candidate_alpha')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              deliberation: 'Alpha detail.',
              thread: threadAlpha,
              messages: [
                {
                  deliberation: 'Alpha turn.',
                  message_id: 'msg_alpha_1',
                  thread_id: 'thread_candidate_alpha',
                  turn_index: 1,
                  sender_role: 'coproducer',
                  content_markdown: 'Welcome back to Session Alpha.',
                  media_cards: [],
                  consultation_cards: [],
                  action_proposals: [],
                  tool_trace: [],
                  prompt_trace: [],
                  created_at_utc: '2026-11-03T10:05:00Z',
                },
              ],
            }),
            { status: 200 },
          ),
        );
      }

      if (url.includes('/api/coproducer/threads/thread_candidate_beta')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              deliberation: 'Beta detail.',
              thread: threadBeta,
              messages: [betaAssistantMsg],
            }),
            { status: 200 },
          ),
        );
      }

      if (url.includes('/api/coproducer/threads')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              deliberation: 'Listed sessions.',
              active_cartridge_thread: threadAlpha,
              draft_threads: [],
              cartridge_threads: [threadAlpha, threadBeta],
              threads: [threadAlpha, threadBeta],
            }),
            { status: 200 },
          ),
        );
      }

      return Promise.resolve(new Response(JSON.stringify({ deliberation: 'ok' }), { status: 200 }));
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={qc}>
        <App />
      </QueryClientProvider>,
    );

    // 1. Old top-bar cartridge <select> trap is gone; games are listed as Co-Producer Sessions
    expect(screen.queryByLabelText('Select Game Cartridge')).not.toBeInTheDocument();
    const sessionsSidebar = await screen.findByRole('complementary', { name: 'Game Sessions' });
    expect(sessionsSidebar).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText('Welcome back to Session Alpha.')).toBeInTheDocument();
    });

    // 2. Switch to Session Beta in the Sessions sidebar -> loads Beta's conversation & tucks show_by_default=false card inside Behind the scenes
    const feedEl = container.querySelector('.coproducer-feed') as HTMLDivElement;
    expect(feedEl).toBeTruthy();
    Object.defineProperty(feedEl, 'scrollHeight', { value: 1800, configurable: true });
    Object.defineProperty(feedEl, 'clientHeight', { value: 500, configurable: true });

    const betaSessionBtn = screen.getByRole('button', { name: /Session Beta/i });
    fireEvent.click(betaSessionBtn);

    await waitFor(() => {
      expect(
        screen.getByText(/Session Beta core loop is responsive/i),
      ).toBeInTheDocument();
    });

    // Feed automatically pins straight to the bottom (1800) on session switch without manual scrolling
    expect(feedEl.scrollTop).toBe(1800);

    // Primary card is shown inline outside Behind-the-scenes; background card is tucked inside closed <details.coproducer-behind-scenes>
    const behindScenes = container.querySelector(
      'details.coproducer-behind-scenes',
    ) as HTMLDetailsElement;
    expect(behindScenes).toBeTruthy();
    expect(behindScenes.open).toBe(false);
    expect(behindScenes.textContent).toContain('Background Progress Metrics');
    expect(screen.getByText('Presented Sketch Plate').closest('details.coproducer-behind-scenes')).toBeNull();

    // Scrolling up reveals "Jump to current", clicking it jumps right back to 1800
    feedEl.scrollTop = 200;
    fireEvent.scroll(feedEl);
    const jumpBtn = await screen.findByRole('button', { name: /Jump to current/i });
    fireEvent.click(jumpBtn);
    expect(feedEl.scrollTop).toBe(1800);
    expect(screen.queryByRole('button', { name: /Jump to current/i })).not.toBeInTheDocument();

    // 3. Sending a message clears the composer synchronously at t=0ms and moves the message immediately to history while response is still pending
    const composer = screen.getByLabelText('Message Creative Co-Producer') as HTMLTextAreaElement;
    fireEvent.change(composer, {
      target: { value: 'Tighten tactile feedback on the primary verb.' },
    });
    expect(composer.value).toBe('Tighten tactile feedback on the primary verb.');

    fireEvent.keyDown(composer, { key: 'Enter' });

    // Synchronously cleared BEFORE fetch resolves!
    expect(composer.value).toBe('');
    expect(screen.getByText('Tighten tactile feedback on the primary verb.')).toBeInTheDocument();
    expect(screen.getByText(/Sent · awaiting reply/i)).toBeInTheDocument();

    // Complete the in-flight response and verify zero duplicate user messages
    await waitFor(() => {
      expect(pendingSendResolvers).toHaveLength(1);
    });
    pendingSendResolvers[0](
      new Response(
        JSON.stringify({
          deliberation: 'Completed turn.',
          thread: threadBeta,
          user_message: {
            deliberation: 'Persisted user turn.',
            message_id: 'msg_beta_user_2',
            thread_id: 'thread_candidate_beta',
            turn_index: 2,
            sender_role: 'user',
            content_markdown: 'Tighten tactile feedback on the primary verb.',
            media_cards: [],
            consultation_cards: [],
            action_proposals: [],
            tool_trace: [],
            prompt_trace: [],
            created_at_utc: '2026-11-03T09:31:00Z',
          },
          assistant_message: {
            deliberation: 'Concise follow-up response.',
            message_id: 'msg_beta_asst_3',
            thread_id: 'thread_candidate_beta',
            turn_index: 3,
            sender_role: 'coproducer',
            content_markdown: 'Agreed—queuing a focused tactile feedback pass.',
            media_cards: [],
            consultation_cards: [],
            action_proposals: [],
            tool_trace: [],
            prompt_trace: [],
            created_at_utc: '2026-11-03T09:31:05Z',
          },
        }),
        { status: 200 },
      ),
    );

    await waitFor(() => {
      expect(screen.getByText('Agreed—queuing a focused tactile feedback pass.')).toBeInTheDocument();
    });
    expect(screen.getAllByText('Tighten tactile feedback on the primary verb.')).toHaveLength(1);

    // 4. Clicking New Game in the Sessions sidebar isolates UI cleanly (hiding old game's Launch & Board tab)
    fireEvent.click(screen.getByRole('button', { name: /New Game/i }));
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /^Launch$/i })).not.toBeInTheDocument();
      expect(screen.queryByRole('tab', { name: /^Board/i })).not.toBeInTheDocument();
      expect(
        screen.getByText('Shape a New Game Idea with Your Creative Co-Producer'),
      ).toBeInTheDocument();
    });
  });

  it('registers CoProducer as Council Chair in roles metadata and highlights CoProducer turns in WritersRoomView', async () => {
    const { getRoleMetadata } = await import('../src/domain/roles');
    const { WritersRoomView } = await import('../src/features/writers-room/WritersRoomView');

    const chairMeta = getRoleMetadata('CoProducer');
    expect(chairMeta.label).toBe('Co-Producer (Council Chair)');
    expect(chairMeta.shortLabel).toBe('Co-Producer');
    expect(chairMeta.pod).toBe('core');

    render(
      <WritersRoomView
        cartridgeId="candidate_chair_01"
        isBuildRunning={false}
        awaitingGreenlight={false}
        busy={false}
        onSteer={vi.fn(async () => {})}
        onGreenlight={vi.fn(async () => {})}
        onOpenLightbox={vi.fn()}
        utterances={[
          {
            deliberation: 'Chairing opening alignment from Co-Producer session.',
            utterance_id: 'utt_chair_0',
            turn_index: 0,
            speaker_role: 'CoProducer',
            speaker_display_name: 'Co-Producer (Council Chair)',
            spoken_dialogue: 'Bringing our producer-approved pillars to the Council table.',
            addressed_to: ['GameplayDesigner', 'GameplaySystemsEngineer'],
            bible_updates_summary: ['Locked Living Game Bible for production'],
          },
          {
            deliberation: 'Designing manual cursor placement.',
            utterance_id: 'utt_gd_1',
            turn_index: 1,
            speaker_role: 'GameplayDesigner',
            speaker_display_name: 'Gameplay Designer',
            spoken_dialogue: 'Mapping clicks 1:1 to hovered tile coordinates.',
            addressed_to: ['CoProducer'],
            bible_updates_summary: [],
          },
        ]}
      />,
    );

    expect(screen.getByText('Co-Producer (Council Chair)')).toBeInTheDocument();
    expect(screen.getByText('Council Chair')).toBeInTheDocument();
    expect(screen.getByText('→ Co-Producer')).toBeInTheDocument();
    const chairArticles = document.querySelectorAll('[data-council-chair="true"]');
    expect(chairArticles).toHaveLength(1);
  });
});
