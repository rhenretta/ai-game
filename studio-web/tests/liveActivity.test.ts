import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { GameBibleResponse, SwarmStreamEvent } from '../src/api/types';
import {
  deriveLiveStudioActivity,
  formatRelativeSeconds,
  formatTokenCount,
  formatUsdCost,
  summarizeSwarmEvent,
} from '../src/domain/liveActivity';
import { deriveLiveItemUpdate } from '../src/realtime/useSwarmSocket';

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

  it('prevents raw context_curator deliberations and foreign slice steps from flooding the live banner', () => {
    const verboseCuratorDeliberation =
      'The target role is council_reviewer:SyntheticReflectionCouncil for post-slice evaluation. ' +
      'Chunks selected include `design:slice:slice_2_planning`, `board:item:integration_synthetic_01`, ' +
      '`board:trace:integration_synthetic_01`, `code:module:main.gd`, and `code:module:main.tscn` ' +
      'plus two visual ground-truth mockups so reviewers can verify layout invariants without exceeding token budgets.';

    // 1. Non-board context_curator_packed events (no claimed board `prev`) must NOT create liveStepsByItem entries
    const nonBoardCuratorEv: SwarmStreamEvent = {
      deliberation: verboseCuratorDeliberation,
      event_type: 'context_curator_packed',
      agent_name: 'CreativeWritersRoom',
      timestamp_utc: '2026-10-31T18:05:00Z',
      payload: {
        item_id: 'slice_reflection_synthetic_01',
        target_agent_role: 'council_reviewer:SyntheticReflectionCouncil',
        selected_chunk_ids: ['c1', 'c2', 'c3', 'c4', 'c5'],
        selected_image_paths: ['mockup.png', 'proof.png'],
      },
    };
    expect(deriveLiveItemUpdate(undefined, nonBoardCuratorEv, '2026-10-31T18:05:00Z')).toBeNull();

    // 2. Active board item context curation produces a concise 1-line summary (zero raw chunk-ID dumps)
    const claimed = deriveLiveItemUpdate(
      undefined,
      {
        deliberation: 'Claimed active board task.',
        event_type: 'backlog_item_claimed',
        agent_name: 'TaskLead',
        timestamp_utc: '2026-10-31T18:05:01Z',
        payload: { item_id: 'task_active_01', owner_role: 'task_lead' },
      },
      '2026-10-31T18:05:01Z',
    );
    expect(claimed).not.toBeNull();
    expect(claimed).not.toBe('clear');

    const curatedBoardStep = deriveLiveItemUpdate(
      claimed && claimed !== 'clear' ? claimed : undefined,
      {
        ...nonBoardCuratorEv,
        payload: {
          ...nonBoardCuratorEv.payload,
          item_id: 'task_active_01',
          target_agent_role: 'swe_engineer',
        },
      },
      '2026-10-31T18:05:02Z',
    );
    expect(curatedBoardStep).not.toBeNull();
    expect(curatedBoardStep).not.toBe('clear');
    if (curatedBoardStep && curatedBoardStep !== 'clear') {
      expect(curatedBoardStep.stepSummary).toBe(
        'Turn #1: Curated 5 context chunks + 2 visual refs for swe_engineer',
      );
      expect(curatedBoardStep.stepSummary).not.toContain('board:trace:');
    }

    // 3. Even if a stale/foreign itemId exists in liveStepsByItem, deriveLiveStudioActivity ignores it
    //    and matches only currently working board items with bounded single-line headlines.
    const activeBible = buildSyntheticBible({
      production_board: {
        deliberation: 'Slice board in progress.',
        cartridge_id: 'candidate_synthetic_99',
        design_revision: 2,
        total_items: 2,
        counts_by_state: { in_progress: 1, ready: 1 },
        is_drained: false,
        items: [
          {
            deliberation: 'Active task',
            item_id: 'task_active_01',
            kind: 'feature_code',
            title: 'Modular Telemetry Emitter',
            spec: 'Emit structured state frames.',
            depends_on: [],
            target_modules: ['scripts/emitter.gd'],
            state: 'in_progress',
          },
          {
            deliberation: 'Queued task',
            item_id: 'task_queued_02',
            kind: 'ui_polish',
            title: 'Inspector Panel Layout',
            spec: 'Align inspector badges.',
            depends_on: ['task_active_01'],
            target_modules: ['scenes/inspector.tscn'],
            state: 'ready',
          },
        ],
      },
    });

    const activity = deriveLiveStudioActivity({
      bible: activeBible,
      recentEvents: [],
      liveStepsByItem: {
        slice_reflection_synthetic_01: {
          itemId: 'slice_reflection_synthetic_01',
          agentName: 'CreativeWritersRoom',
          stepSummary: verboseCuratorDeliberation,
          eventType: 'context_curator_packed',
          timestampUtc: '2026-10-31T18:05:10Z',
        },
      },
    });

    expect(activity.phase).toBe('board_synthesis');
    expect(activity.headline).toBe('Executing 1 active board task: Modular Telemetry Emitter');
    expect(activity.headline).not.toContain('SyntheticReflectionCouncil');
    expect(activity.headline.length).toBeLessThanOrEqual(160);

    // 4. summarizeSwarmEvent formats context_curator_packed cleanly for LivePulseModal
    const modalSummary = summarizeSwarmEvent(nonBoardCuratorEv);
    expect(modalSummary.badgeText).toBe('CONTEXT PACK');
    expect(modalSummary.title).toContain('Curated 5 context chunks + 2 visual refs');
    expect(modalSummary.title).not.toContain('board:trace:');

    // 5. CSS enforces 1-line clamping on .live-activity-headline
    const cssSource = readFileSync(resolve(__dirname, '../src/styles/global.css'), 'utf-8');
    expect(cssSource).toMatch(
      /\.live-activity-headline\s*\{[^}]*-webkit-line-clamp:\s*1;[\s\S]*?line-clamp:\s*1;[\s\S]*?overflow:\s*clip;/,
    );
  });

  it('surfaces cached token share, media synthesis counts, and USD cost telemetry', () => {
    expect(formatUsdCost(0)).toBe('$0.00');
    expect(formatUsdCost(0.0004)).toBe('<$0.001');
    expect(formatUsdCost(0.0142)).toBe('$0.014');
    expect(formatUsdCost(1.428)).toBe('$1.43');

    const bible = buildSyntheticBible();
    const cachedCallEvent: SwarmStreamEvent = {
      event_id: 'ev_cached_1',
      deliberation: 'Completed cached Gemini call.',
      event_type: 'studio_live_activity',
      agent_name: 'TaskLead',
      timestamp_utc: '2026-10-31T18:10:00Z',
      payload: {
        phase: 'llm_gateway_call',
        label: 'TaskLead.plan_task_lead_turn[mech_01:t2]',
        total_calls: 42,
        token_stats: {
          prompt_tokens: 18000,
          cached_tokens: 15000,
          output_tokens: 450,
          thought_tokens: 350,
          total_tokens: 18800,
          estimated_call_cost_usd: 0.0064,
          session_estimated_cost_usd: 0.284,
          session_cache_hit_pct: 76.4,
        },
      },
    };
    const act = deriveLiveStudioActivity({
      bible,
      recentEvents: [cachedCallEvent],
      liveStepsByItem: {},
    });
    expect(act.latestAiCall?.cachedTokens).toBe(15000);
    expect(act.latestAiCall?.estimatedCallCostUsd).toBe(0.0064);
    expect(act.latestAiCall?.sessionEstimatedCostUsd).toBe(0.284);
    expect(act.latestAiCall?.sessionCacheHitPct).toBe(76.4);

    const row = summarizeSwarmEvent(cachedCallEvent);
    expect(row.badgeTone).toBe('ok');
    expect(row.subtitle).toContain('15.0k tok cached');
    expect(row.subtitle).toContain('est $0.006');

    const imageSynthEvent: SwarmStreamEvent = {
      event_id: 'ev_img_1',
      deliberation: 'Completed Imagen call.',
      event_type: 'studio_live_activity',
      agent_name: 'AssetFabricator',
      timestamp_utc: '2026-10-31T18:10:05Z',
      payload: {
        phase: 'llm_gateway_call',
        label: 'AssetFabricator.generate_images[hero_rig]',
        total_calls: 43,
        token_stats: {
          total_tokens: 0,
          thought_tokens: 0,
          image_count: 1,
          estimated_call_cost_usd: 0.04,
          session_estimated_cost_usd: 0.324,
        },
      },
    };
    const imgRow = summarizeSwarmEvent(imageSynthEvent);
    expect(imgRow.badgeText).toBe('1 img');
    expect(imgRow.subtitle).toContain('1 synthesized image');
    expect(imgRow.subtitle).toContain('est $0.040');
  });
});
