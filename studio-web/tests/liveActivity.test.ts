import { describe, expect, it } from 'vitest';
import type { GameBibleResponse, SwarmStreamEvent } from '../src/api/types';
import {
  deriveLiveStudioActivity,
  formatRelativeSeconds,
  formatTokenCount,
  summarizeSwarmEvent,
} from '../src/domain/liveActivity';

function buildSyntheticBible(overrides?: Partial<GameBibleResponse>): GameBibleResponse {
  return {
    deliberation: 'Synthetic test bible.',
    has_bible: true,
    cartridge_id: 'candidate_synthetic_99',
    active_candidate_id: 'candidate_synthetic_99',
    status: 'building',
    game_title: 'Synthetic Relay Sandbox',
    core_genre: 'Systems Puzzle',
    process_status: {
      process_state: 'running',
      is_process_alive: true,
      is_paused: false,
      pause_for_greenlight: false,
      build_actively_running: true,
      lock_held: true,
      active_threads: ['GenesisTrigger_test'],
      scheduler_running_workers: 0,
      orchestrator_state: 'BUILDING',
      last_event_timestamp_utc: '2026-10-31T18:00:10Z',
    },
    production_board: {
      deliberation: 'All tasks approved.',
      cartridge_id: 'candidate_synthetic_99',
      design_revision: 1,
      total_items: 3,
      counts_by_state: { approved: 3 },
      is_drained: true,
      items: [
        {
          deliberation: 'Task 1',
          item_id: 'task_01',
          kind: 'feature_code',
          title: 'Core Simulation Loop',
          spec: 'Deterministic tick loop.',
          depends_on: [],
          target_modules: ['scripts/sim.gd'],
          state: 'approved',
        },
        {
          deliberation: 'Task 2',
          item_id: 'task_02',
          kind: 'visual_polish',
          title: 'Grid Shader Pass',
          spec: 'Clean grid lines.',
          depends_on: ['task_01'],
          target_modules: ['shaders/grid.gdshader'],
          state: 'approved',
        },
        {
          deliberation: 'Task 3',
          item_id: 'task_03',
          kind: 'ui_polish',
          title: 'Status HUD Overlay',
          spec: 'Render telemetry counters.',
          depends_on: ['task_01'],
          target_modules: ['scenes/hud.tscn'],
          state: 'approved',
        },
      ],
    },
    ...overrides,
  };
}

describe('deriveLiveStudioActivity & summarizeSwarmEvent', () => {
  it('resolves post-board QA Self-Healing Repair and attaches latest LLM gateway call stats', () => {
    const bible = buildSyntheticBible();
    const recentEvents: SwarmStreamEvent[] = [
      {
        event_id: 'ev_3',
        deliberation: 'Completed Gemini gateway call.',
        event_type: 'studio_live_activity',
        agent_name: 'genai',
        timestamp_utc: '2026-10-31T18:00:12Z',
        payload: {
          phase: 'llm_gateway_call',
          label: 'LeadGameplaySWEAgent.pev_phase2[candidate_synthetic_99:iter2:hop1]',
          total_calls: 19,
          token_stats: {
            prompt_tokens: 78000,
            output_tokens: 2400,
            thought_tokens: 4100,
            total_tokens: 84500,
          },
        },
      },
      {
        event_id: 'ev_2',
        deliberation: 'Dispatching SWE self-healing repair pass.',
        event_type: 'AGENT_STREAM',
        agent_name: 'LeadGameplaySWEAgent',
        timestamp_utc: '2026-10-31T17:59:40Z',
        payload: {
          phase: 'REPAIR',
          failure_gate: 'Gate D: Frame Budget',
          remedy_directive: 'Eliminate per-frame node allocations inside _process.',
        },
      },
      {
        event_id: 'ev_1',
        deliberation: 'Frame budget gate exceeded threshold.',
        event_type: 'QA_GATE_RESULT',
        agent_name: 'QAPlaytesterBuildSupervisor',
        timestamp_utc: '2026-10-31T17:59:38Z',
        payload: {
          attempt: 2,
          max_retries: 3,
          promoted: false,
          failure_gate: 'Gate D: Frame Budget',
          failure_reason: 'Peak frame spike 68.4ms > 16.6ms budget.',
          remedy_directive: 'Eliminate per-frame node allocations inside _process.',
        },
      },
    ];

    const activity = deriveLiveStudioActivity({
      bible,
      recentEvents,
      liveStepsByItem: {
        // Stale entry for an already-approved item must be ignored
        task_03: {
          itemId: 'task_03',
          agentName: 'LeadGameplaySWEAgent',
          stepSummary: 'Old completed step',
          eventType: 'backlog_item_specialist_started',
          timestampUtc: '2026-10-31T17:40:00Z',
        },
      },
    });

    expect(activity.isActive).toBe(true);
    expect(activity.phase).toBe('qa_repair');
    expect(activity.phaseBadge).toBe('QA Repair · 2/3');
    expect(activity.stageTitle).toBe('QA Self-Healing Repair (Attempt 2/3)');
    expect(activity.activeAgentLabel).toBe('Lead Gameplay SWE');
    expect(activity.headline).toContain('Gate D: Frame Budget');
    expect(activity.headline).toContain('Peak frame spike 68.4ms');
    expect(activity.whyPostBoardActive).toContain('All 3/3 board tasks are approved');
    expect(activity.progressValue).toBe(2);
    expect(activity.progressMax).toBe(3);

    expect(activity.latestAiCall).toEqual({
      callNumber: 19,
      agentLabel: 'Lead Gameplay SWE',
      stepLabel: 'pev_phase2 (iter2:hop1)',
      totalTokens: 84500,
      thoughtTokens: 4100,
      timestampUtc: '2026-10-31T18:00:12Z',
    });

    const aiSummary = summarizeSwarmEvent(recentEvents[0]);
    expect(aiSummary.category).toBe('ai');
    expect(aiSummary.title).toBe('AI Call #19: pev_phase2 (iter2:hop1)');
    expect(aiSummary.badgeText).toBe('84.5k tok');

    const gateSummary = summarizeSwarmEvent(recentEvents[2]);
    expect(gateSummary.category).toBe('stage');
    expect(gateSummary.badgeTone).toBe('warn');
    expect(gateSummary.title).toContain('Gate D: Frame Budget rejected (Attempt 2/3)');
  });

  it('distinguishes slice_assembly_check and slice_blind_playtest activity tags', () => {
    const bible = buildSyntheticBible();

    const assemblyActivity = deriveLiveStudioActivity({
      bible,
      recentEvents: [
        {
          deliberation: 'Live agent activity update (slice_assembly_check): Checking boot scene.',
          event_type: 'writers_room_activity',
          agent_name: 'SystemsArchitectAgent',
          timestamp_utc: '2026-10-31T18:01:00Z',
          payload: {
            agent_role: 'SystemsArchitectAgent',
            status_text: 'Verifying main scene boot reachability before playtest...',
          },
        },
      ],
      liveStepsByItem: {},
    });
    expect(assemblyActivity.phase).toBe('slice_assembly_check');
    expect(assemblyActivity.phaseBadge).toBe('Assembly Check');

    const playtestEvalActivity = deriveLiveStudioActivity({
      bible,
      recentEvents: [
        {
          deliberation: 'Live agent activity update (slice_blind_playtest): Reviewing report.',
          event_type: 'writers_room_activity',
          agent_name: 'GameplayDesigner',
          timestamp_utc: '2026-10-31T18:02:00Z',
          payload: {
            agent_role: 'GameplayDesigner',
            status_text: 'Evaluating Blind Playtester Experience Report...',
          },
        },
      ],
      liveStepsByItem: {},
    });
    expect(playtestEvalActivity.phase).toBe('blind_playtest');
    expect(playtestEvalActivity.suggestedWorkspace?.id).toBe('playtest');

    const sliceQaActivity = deriveLiveStudioActivity({
      bible,
      recentEvents: [
        {
          deliberation: 'Live agent activity update (slice_qa_gate_check): Running 5-Gate QA.',
          event_type: 'writers_room_activity',
          agent_name: 'QATestEngineer',
          timestamp_utc: '2026-10-31T18:02:30Z',
          payload: {
            agent_role: 'QATestEngineer',
            phase: 'slice_qa_gate_check',
            status_text: 'Running 5-Gate QA verification for active slice...',
          },
        },
      ],
      liveStepsByItem: {},
    });
    expect(sliceQaActivity.phase).toBe('qa_verification');
    expect(sliceQaActivity.stageTitle).toBe('Slice 5-Gate QA Verification');

    const spawnedSummary = summarizeSwarmEvent({
      deliberation: 'Gate C spawned board remediation task.',
      event_type: 'slice_qa_gate_check_completed',
      agent_name: 'QATestEngineer',
      timestamp_utc: '2026-10-31T18:02:40Z',
      payload: {
        passed: false,
        failure_gate: 'Gate C: Headless Execution',
        failure_reason: 'Runtime null access in main.gd',
        spawned_item_ids: ['qa_gate_c_remediation_p0'],
      },
    });
    expect(spawnedSummary.badgeText).toBe('TASK SPAWNED');
    expect(spawnedSummary.title).toContain('qa_gate_c_remediation_p0');
  });

  it('formats token counts and relative timestamps cleanly', () => {
    expect(formatTokenCount(850)).toBe('850 tok');
    expect(formatTokenCount(84500)).toBe('84.5k tok');
    expect(formatTokenCount(1_250_000)).toBe('1.25M tok');

    const refNow = Date.parse('2026-10-31T18:00:30Z');
    expect(formatRelativeSeconds('2026-10-31T18:00:28Z', refNow)).toBe('just now');
    expect(formatRelativeSeconds('2026-10-31T18:00:12Z', refNow)).toBe('18s ago');
    expect(formatRelativeSeconds('2026-10-31T17:58:25Z', refNow)).toBe('2m 5s ago');
  });
});
