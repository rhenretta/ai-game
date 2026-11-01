import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  extractCleanTurnObservation,
  formatPlaytestMotorAction,
  parseBugEntries,
  parseQuestionEntries,
} from '../src/domain/playtestNotebook';
import { PlaytestLabView } from '../src/features/playtest/PlaytestLabView';
import type { PlaytestTurn } from '../src/api/types';

describe('Playtest Lab UX: Current Focus, Activity Log, DVR Replay-to-Live & Structured Notebook', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('strips internal prompt-tape framing and formats motor actions into clean chips', () => {
    const rawTape =
      "[Turn 14 | t=11.2s | cursor=(500,500) | tap=TAB 500ms] Closed the calibration panel and returned to main view. | NOTE+[HUD Counter=0/10] | QUESTION+[ANSWERED(panel_toggle_check)=Pressing ESC closes panel.] | RULEBOOK+[ESC=Closes panel.]";
    expect(extractCleanTurnObservation(rawTape)).toBe(
      'Closed the calibration panel and returned to main view.',
    );

    const turn: PlaytestTurn = {
      turn_id: 'turn_14',
      session_id: 'pt_synthetic_01',
      turn_index: 14,
      deliberation: 'Testing primary toolbar toggle.',
      action: {
        deliberation: 'Tap TAB and click hub.',
        mouse_button_action: 'left_click',
        tap_keys: ['TAB'],
        duration_ms: 500,
      },
      action_px: {
        target_x: 640,
        target_y: 360,
        mouse_button_action: 'left_click',
        tap_keys: ['TAB'],
        hold_keys: [],
        duration_ms: 500,
        has_explicit_mouse: true,
      },
    };

    const formatted = formatPlaytestMotorAction(turn);
    expect(formatted.chips).toEqual(['Left Click (640, 360)', 'Tap [TAB]']);
    expect(formatted.summary).toBe('Left Click (640, 360) · Tap [TAB]');
  });

  it('separates open vs answered questions with answers intact and parses structured bugs without JSON dumps', () => {
    const questions = parseQuestionEntries({
      internal_key_slot_one: {
        question_key: 'internal_key_slot_one',
        question: 'What happens when I activate the primary relay switch?',
        status: 'answered',
        answer: 'Activating the switch powers adjacent nodes.',
        asked_turn: 2,
        answered_turn: 3,
      },
      unresolved_gate_mystery: {
        question_key: 'unresolved_gate_mystery',
        question: 'How do I unlock the secondary routing gate?',
        status: 'open',
        answer: null,
        asked_turn: 6,
      },
    });

    expect(questions.open).toHaveLength(1);
    expect(questions.open[0].question).toBe('How do I unlock the secondary routing gate?');
    expect(questions.answered).toHaveLength(1);
    expect(questions.answered[0].answer).toBe('Activating the switch powers adjacent nodes.');

    const bugs = parseBugEntries({
      unresponsive_header_btn: {
        bug_key: 'unresponsive_header_btn',
        severity: 'major',
        category: 'input_ui_bug',
        summary: 'Inspect button does not respond to clicks',
        details: 'Clicked three times with zero UI change.',
        turn_index: 5,
      },
    });
    expect(bugs).toHaveLength(1);
    expect(bugs[0].summary).toBe('Inspect button does not respond to clicks');
    expect(bugs[0].severity).toBe('major');
  });

  it('shows current intent front-and-center, tucks full telemetry into <details>, fixes Rulebook & Notes, and supports DVR Replay-to-Live', async () => {
    const livePayload = {
      deliberation: 'Loaded live session detail.',
      session: {
        deliberation: 'Running live session.',
        session_id: 'pt_dvr_live_01',
        checkpoint_id: 'active',
        candidate_id: 'candidate_synthetic_dvr',
        game_title: 'Synthetic Relay Sandbox',
        status: 'running',
        initiated_by: 'operator',
        turn_count: 3,
        sim_ticks_total: 210,
        current_plan: 'Drag connector from source node to receiver socket.',
        current_mood: 'focused',
        latest_frame_name: 'live_preview.jpg',
        has_video: false,
        matches_active_cartridge: true,
        rulebook: {
          'Key 1': 'Selects connector tool.',
        },
        field_notes: {
          'Top Bar': 'Displays 2/5 active links.',
        },
        questions: {
          internal_snake_key_1: {
            question_key: 'internal_snake_key_1',
            question: 'What happens when pressing key 1?',
            status: 'answered',
            answer: 'Pressing key 1 highlights the connector slot.',
            asked_turn: 1,
            answered_turn: 2,
          },
          internal_open_mystery: {
            question_key: 'internal_open_mystery',
            question: 'Why is the receiver blinking amber?',
            status: 'open',
            answer: null,
            asked_turn: 3,
          },
        },
        bugs_found: {
          clipped_tooltip_box: {
            bug_key: 'clipped_tooltip_box',
            severity: 'minor',
            category: 'visual_glitch',
            summary: 'Tooltip label clips right border',
            details: 'Rightmost two characters overflow panel.',
            turn_index: 2,
          },
        },
      },
      turns: [
        {
          turn_id: 't1',
          session_id: 'pt_dvr_live_01',
          turn_index: 1,
          video_timestamp_sec: 1.5,
          current_plan: 'Press key 1 to test first toolbar slot.',
          deliberation: 'Internal rationale for turn 1.',
          append_log_line:
            '[Turn 1 | t=1.5s | tap=1] Opening view loaded cleanly. | QUESTION+[OPEN(internal_snake_key_1)=What happens when pressing key 1?]',
          screenshot_name: 'turn_0001.png',
          action_px: {
            mouse_button_action: 'none',
            tap_keys: ['1'],
            hold_keys: [],
            duration_ms: 450,
          },
        },
        {
          turn_id: 't2',
          session_id: 'pt_dvr_live_01',
          turn_index: 2,
          video_timestamp_sec: 3.0,
          current_plan: 'Click the source node to start wiring.',
          deliberation: 'Internal rationale for turn 2.',
          append_log_line:
            '[Turn 2 | t=3.0s | click=(320,240)] Pressing key 1 highlighted slot 1. | RULEBOOK+[Key 1=Selects connector tool.]',
          screenshot_name: 'turn_0002.png',
          action_px: {
            target_x: 320,
            target_y: 240,
            mouse_button_action: 'left_click',
            tap_keys: [],
            hold_keys: [],
            duration_ms: 500,
            has_explicit_mouse: true,
          },
        },
        {
          turn_id: 't3',
          session_id: 'pt_dvr_live_01',
          turn_index: 3,
          video_timestamp_sec: 4.5,
          current_plan: 'Drag connector from source node to receiver socket.',
          deliberation: 'Internal rationale for turn 3.',
          append_log_line:
            '[Turn 3 | t=4.5s | drag=(320,240)->(680,240)] Source node activated and emitted cable preview. | QUESTION+[OPEN(internal_open_mystery)=Why is the receiver blinking amber?]',
          screenshot_name: 'turn_0003.png',
          action_px: {
            drag_from_x: 320,
            drag_from_y: 240,
            target_x: 680,
            target_y: 240,
            mouse_button_action: 'drag_left',
            tap_keys: [],
            hold_keys: [],
            duration_ms: 700,
            has_explicit_mouse: true,
          },
        },
      ],
      wishes: [],
      compactions: [],
    };

    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(livePayload), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    const sessionsListData = {
      deliberation: 'List loaded.',
      active_cartridge_id: 'candidate_synthetic_dvr',
      games_catalog: [],
      playable_targets: [],
      pending_wishes_count: 0,
      sessions: [livePayload.session],
    };

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { rerender } = render(
      <QueryClientProvider client={qc}>
        <PlaytestLabView
          selectedCartridgeId="candidate_synthetic_dvr"
          sessionsListData={sessionsListData}
          sessionsError={null}
          onRefetchSessions={vi.fn()}
          livePlaytestFrames={{
            pt_dvr_live_01: {
              sessionId: 'pt_dvr_live_01',
              tick: 210,
              frameName: 'live_preview.jpg',
              timestampUtc: '2026-10-31T19:20:00Z',
              seq: 1,
            },
          }}
        />
      </QueryClientProvider>,
    );

    // 1. Front-and-center Focus Banner displays current plan & clean observation without raw RULEBOOK+/QUESTION+ noise
    const focusRegion = await screen.findByRole('region', { name: /Current playtester focus/i });
    expect(
      within(focusRegion).getByText('Drag connector from source node to receiver socket.'),
    ).toBeInTheDocument();
    expect(
      await within(focusRegion).findByText(/Source node activated and emitted cable preview\./i),
    ).toBeInTheDocument();
    expect(within(focusRegion).queryByText(/QUESTION\+\[/)).not.toBeInTheDocument();

    // 2. Activity Log defaults to Latest First (#3 at top, #1 at bottom) with clean observations
    const logHeadings = screen.getAllByText(/^#[123]$/);
    expect(logHeadings.map((el) => el.textContent)).toEqual(['#3', '#2', '#1']);

    // 3. Scrub back to Turn #1 during LIVE run by clicking Turn #1 in the Activity Log
    fireEvent.click(screen.getByText('#1'));
    const turn1Img = await screen.findByAltText('Turn 1');
    expect(turn1Img.getAttribute('src')).toContain('/media/turn_0001.png');
    expect(within(focusRegion).getByText(/Replaying Turn #1 of 3/i)).toBeInTheDocument();

    // Verify incoming live frame updates do NOT yank user out of historical replay scrub
    rerender(
      <QueryClientProvider client={qc}>
        <PlaytestLabView
          selectedCartridgeId="candidate_synthetic_dvr"
          sessionsListData={sessionsListData}
          sessionsError={null}
          onRefetchSessions={vi.fn()}
          livePlaytestFrames={{
            pt_dvr_live_01: {
              sessionId: 'pt_dvr_live_01',
              tick: 225,
              frameName: 'live_preview.jpg',
              timestampUtc: '2026-10-31T19:20:01Z',
              seq: 2,
            },
          }}
        />
      </QueryClientProvider>,
    );
    expect(screen.getByAltText('Turn 1')).toBeInTheDocument();

    // 4. Auto-Play → Live advances Turn #1 -> #2 -> #3 -> hands off seamlessly back to Live Screen
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button', { name: /Play → Live/i }));
    act(() => {
      vi.advanceTimersByTime(900);
    });
    expect(screen.getByAltText('Turn 2')).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(900);
    });
    expect(screen.getByAltText('Turn 3')).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(900);
    });
    expect(screen.getByAltText('Latest player frame')).toBeInTheDocument();
    vi.useRealTimers();

    // 5. Rulebook & Notes tab separates Bugs, Rulebook, Open Mysteries, Answered Questions (with answers!), and Field Notes
    fireEvent.click(screen.getByRole('button', { name: /Rulebook & Notes/i }));
    expect(screen.getByText('Tooltip label clips right border')).toBeInTheDocument();
    expect(screen.getByText('Selects connector tool.')).toBeInTheDocument();
    expect(screen.getByText('Why is the receiver blinking amber?')).toBeInTheDocument();
    expect(screen.getByText(/Pressing key 1 highlights the connector slot\./i)).toBeInTheDocument();
    expect(screen.queryByText(/internal_snake_key_1/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/internal_open_mystery/i)).not.toBeInTheDocument();
  });

  it('dispatches full manual playtest report to Council with rationale-first payload, producer direction, and pending wishes', async () => {
    const completedDetail = {
      deliberation: 'Loaded completed manual session detail.',
      session: {
        deliberation: 'Completed manual session.',
        session_id: 'pt_manual_done_01',
        checkpoint_id: 'active',
        candidate_id: 'candidate_synthetic_02',
        game_title: 'Synthetic Circuit Builder',
        status: 'completed',
        initiated_by: 'operator',
        turn_count: 18,
        sim_ticks_total: 1080,
        current_plan: 'Verify final circuit throughput readout.',
        current_mood: 'satisfied',
        has_video: false,
        overall_fun_rating: 7,
        report_sent_to_council_at_utc: null,
        matches_active_cartridge: true,
        rulebook: { 'Left Click': 'Places node.' },
        field_notes: { 'Throughput Meter': 'Reads 12 units/sec.' },
        questions: {},
        bugs_found: {
          overlap_caption: {
            bug_key: 'overlap_caption',
            severity: 'major',
            category: 'visual_glitch',
            summary: 'Footer hint overlaps slot 1',
            details: 'Intersects top border.',
            turn_index: 4,
          },
        },
        session_report: {
          deliberation: 'Synthesized player experience report.',
          overall_fun_rating: 7,
          overall_experience_summary: 'Engaging loop, though footer hint overlaps toolbar slot 1.',
          visual_and_ui_defects_observed: ['Footer hint overlaps toolbar slot 1'],
          gameplay_and_feel: 'Snappy node placement.',
        },
      },
      turns: [
        {
          turn_id: 't1',
          session_id: 'pt_manual_done_01',
          turn_index: 1,
          video_timestamp_sec: 2.0,
          current_plan: 'Verify final circuit throughput readout.',
          deliberation: 'Checking HUD.',
          append_log_line: 'Placed node and checked meter.',
          screenshot_name: 'turn_0001.png',
        },
      ],
      wishes: [
        {
          deliberation: 'Hover cue needed.',
          wish_id: 'wish_01',
          session_id: 'pt_manual_done_01',
          candidate_id: 'candidate_synthetic_02',
          category: 'ux_onboarding',
          title: 'Add socket hover highlight',
          wish_text: 'Show a subtle ring when hovering valid sockets.',
          status: 'pending',
        },
      ],
      compactions: [],
    };

    const capturedBodies: Array<Record<string, unknown>> = [];
    let dispatched = false;

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url.endsWith('/api/playtest/sessions/pt_manual_done_01/send-report-to-council') && init?.method === 'POST') {
        const parsed = JSON.parse(String(init.body ?? '{}')) as Record<string, unknown>;
        capturedBodies.push(parsed);
        dispatched = true;
        return new Response(
          JSON.stringify({
            deliberation: 'Dispatched full report to Council.',
            status: 'council_resumed',
            candidate_id: 'candidate_synthetic_02',
            auto_resumed_job_id: 'ptrep_abc12345',
            session: {
              ...completedDetail.session,
              report_sent_to_council_at_utc: '2026-10-31T20:00:00Z',
            },
            wishes: [{ ...completedDetail.wishes[0], status: 'sent_to_council' }],
            proof_image_paths: ['artifacts/proofs/manual_playtest_pt_manual_done_01_shot1.png'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }

      const payload = dispatched
        ? {
            ...completedDetail,
            session: {
              ...completedDetail.session,
              report_sent_to_council_at_utc: '2026-10-31T20:00:00Z',
            },
            wishes: [{ ...completedDetail.wishes[0], status: 'sent_to_council' }],
          }
        : completedDetail;

      return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <PlaytestLabView
          selectedCartridgeId="candidate_synthetic_02"
          sessionsListData={{
            deliberation: 'Loaded sessions.',
            active_cartridge_id: 'candidate_synthetic_02',
            games_catalog: [],
            playable_targets: [],
            pending_wishes_count: 1,
            sessions: [completedDetail.session],
          }}
          sessionsError={null}
          onRefetchSessions={vi.fn()}
        />
      </QueryClientProvider>,
    );

    // Quick-jump button appears right in the Focus Banner for finished manual playtests
    const quickJumpBtn = await screen.findByRole('button', { name: /Review & Send Report/i });
    fireEvent.click(quickJumpBtn);

    // Experience Report tab shows semantic dispatch form with Producer note + pending wish checkbox
    const noteInput = screen.getByLabelText(/Producer Direction for Council/i);
    fireEvent.change(noteInput, {
      target: { value: 'Fix footer overlap and add socket hover ring.' },
    });

    const wishCheckbox = screen.getByLabelText(/Also approve & include 1 pending player wish/i) as HTMLInputElement;
    expect(wishCheckbox.checked).toBe(true);

    const submitBtn = screen.getByRole('button', { name: /Send Full Report to Council/i });
    fireEvent.click(submitBtn);

    const statusAlert = await screen.findByRole('status');
    expect(statusAlert).toHaveTextContent(/Specialist Council resumed in Writers’ Room/i);

    expect(capturedBodies).toHaveLength(1);
    expect(Object.keys(capturedBodies[0])[0]).toBe('deliberation');
    expect(capturedBodies[0].include_pending_wishes).toBe(true);
    expect(capturedBodies[0].producer_note).toBe('Fix footer overlap and add socket hover ring.');
  });
});
