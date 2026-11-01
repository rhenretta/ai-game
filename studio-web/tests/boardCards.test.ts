import { describe, expect, it } from 'vitest';
import {
  extractLatestRevisionSummary,
  isWorkingBoardState,
  mergeLiveTurnState,
  resolveCardProofShowcase,
  sortBoardItemsByPriority,
  splitSpecRequirements,
} from '../src/domain/boardCards';
import type { BoardItemChronologyResponse, ProductionBoardItem } from '../src/api/types';

function makeItem(overrides: Partial<ProductionBoardItem>): ProductionBoardItem {
  return {
    deliberation: 'Synthetic rationale for modular subsystem.',
    item_id: 'task_synth_01',
    kind: 'mechanic_code',
    title: 'Synthetic Routing Subsystem',
    spec: 'Implement deterministic routing.',
    target_modules: ['scripts/router.gd'],
    depends_on: [],
    state: 'ready',
    ...overrides,
  };
}

describe('boardCards domain helpers', () => {
  it('splits numbered specification requirements cleanly without counting preamble as an item', () => {
    const res = splitSpecRequirements(
      'Implement scripts/router.gd: 1) Register node topology; 2) Dispatch packets per tick; 3) Emit saturation signal.',
    );
    expect(res.isExpandable).toBe(true);
    expect(res.preamble).toBe('Implement scripts/router.gd:');
    expect(res.points).toHaveLength(3);
    expect(res.points[0]).toBe('Register node topology');
    expect(res.leadSummary).toContain('Implement scripts/router.gd:');
  });

  it('never splits mid-sentence on prose ending in numbers like "from 0 to at least 1." or "reaches 2."', () => {
    const item = makeItem({
      item_id: 'task_list_boundary',
      state: 'in_progress',
      attempt_count: 2,
      revisions: [
        {
          revision_index: 0,
          verdict: 'request_changes',
          feedback:
            '[SystemsReviewer]: 1. Ensure clicking the action trigger calls `_run_action(slot_id)`. Verify counter increases from 0 to at least 1. 2. Ensure UI slot dynamically reflects count (`x1` instead of `x0`) until tier reaches 2. 3. Capture frames showing: (a) initial state, (b) incremented state, and (c) clean close.',
        },
      ],
    });

    const digest = extractLatestRevisionSummary(item);
    expect(digest).not.toBeNull();
    expect(digest?.feedbackByRole).toHaveLength(1);
    const pts = digest!.feedbackByRole[0].points;
    expect(pts).toHaveLength(3);
    expect(pts[0]).toContain('increases from 0 to at least 1.');
    expect(pts[0]).not.toMatch(/^1\./);
    expect(pts[1]).toContain('until tier reaches 2.');
    expect(pts[2]).toContain('(a) initial state');
  });

  it('surfaces mid-flight proof images rendered in live chronology turns on in_progress cards', () => {
    const item = makeItem({
      item_id: 'task_live_map',
      state: 'in_progress',
      revisions: [],
      latest_proof_urls: [],
    });

    const chronology: BoardItemChronologyResponse = {
      deliberation: 'Live trace.',
      cartridge_id: 'candidate_synth_01',
      item_id: 'task_live_map',
      title: item.title,
      kind: item.kind,
      state: 'in_progress',
      spec: item.spec,
      target_modules: item.target_modules,
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
              specialist_agent: 'swe_engineer',
              summary: 'Implemented generator.',
            },
            {
              turn_index: 1,
              specialist_agent: 'qa_test_engineer',
              summary: 'Captured 4-frame verification strip.',
              proof_urls: [
                {
                  path: 'artifacts/work_items/task_live_map/strip_turn2.png',
                  filename: 'strip_turn2.png',
                  url: '/api/artifacts/content?path=strip_turn2.png&raw=true',
                  is_audio: false,
                },
              ],
            },
          ],
        },
      ],
    };

    const showcase = resolveCardProofShowcase(item, chronology, 'candidate_synth_01');
    expect(showcase.images).toHaveLength(1);
    expect(showcase.images[0].filename).toBe('strip_turn2.png');
    expect(showcase.sourceLabel).toBe('Turn #2 · qa_test_engineer');
  });

  it('extracts proof_paths from item.revisions and pairs waveform images with audio clips', () => {
    const item = makeItem({
      item_id: 'task_audio_01',
      kind: 'audio_asset',
      state: 'approved',
      revisions: [
        {
          revision_index: 0,
          proof_paths: [
            'artifacts/renders/waveform_relay_hum.png',
            'assets/audio/sfx_relay_hum.wav',
          ],
          worker_notes:
            'Strategy: Synthesize harmonic drone | [Turn 0 audio_designer] Generated sfx_relay_hum.wav',
          verdict: 'approved',
          feedback: '[AudioDirector]: 1) Clean harmonic balance. 2) Zero clipping.',
        },
      ],
    });

    const showcase = resolveCardProofShowcase(item, undefined, 'candidate_synth_01');
    expect(showcase.images).toHaveLength(1);
    expect(showcase.audios).toHaveLength(1);
    expect(showcase.images[0].isWaveform).toBe(true);
    expect(showcase.images[0].pairedAudioUrl).toContain('sfx_relay_hum.wav');
    expect(showcase.sourceLabel).toBe('Rev #1');

    const digest = extractLatestRevisionSummary(item);
    expect(digest).not.toBeNull();
    expect(digest?.revisionNumber).toBe(1);
    expect(digest?.workerTurns).toHaveLength(1);
    expect(digest?.feedbackByRole[0].role).toBe('AudioDirector');
    expect(digest?.feedbackByRole[0].points).toHaveLength(2);
  });

  it('merges live WebSocket step logs with disk active_turn state', () => {
    const item = makeItem({ item_id: 'task_active', state: 'in_progress' });
    const merged = mergeLiveTurnState(
      item,
      {
        itemId: 'task_active',
        agentName: 'swe_engineer',
        specialistAgent: 'swe_engineer',
        turnIndex: 3,
        stepSummary: 'Turn #3 (swe_engineer): Running unit checks',
        eventType: 'backlog_item_substep',
        timestampUtc: '2026-10-31T15:00:10Z',
        startedAtUtc: '2026-10-31T15:00:00Z',
        stepLog: [
          {
            step_type: 'planning_turn',
            specialist_agent: 'task_lead',
            message: 'Task Lead planning Turn #3',
          },
          {
            step_type: 'running_tests',
            specialist_agent: 'swe_engineer',
            message: 'Running unit checks',
          },
        ],
      },
      undefined,
    );

    expect(merged.turnNumber).toBe(3);
    expect(merged.specialistAgent).toBe('swe_engineer');
    expect(merged.steps).toHaveLength(2);
    expect(merged.steps[1].stepType).toBe('running_tests');
  });

  it('sorts active and in-review items ahead of queued, blocked, and approved items', () => {
    const sorted = sortBoardItemsByPriority(
      [
        makeItem({ item_id: 'done_1', state: 'approved' }),
        makeItem({ item_id: 'blocked_1', state: 'blocked' }),
        makeItem({ item_id: 'working_1', state: 'in_progress' }),
        makeItem({ item_id: 'review_1', state: 'awaiting_review' }),
      ],
      true,
    );

    expect(sorted.map((i) => i.item_id)).toEqual([
      'working_1',
      'review_1',
      'blocked_1',
      'done_1',
    ]);
  });

  it('preserves Council requested changes on Attempt #2 while state is in_progress', () => {
    const item = makeItem({
      item_id: 'task_retry_ui',
      state: 'in_progress',
      attempt_count: 2,
      revisions: [
        {
          revision_index: 0,
          verdict: 'request_changes',
          feedback:
            '[SystemsDesigner]: 1) Synchronize slot counts on state signal. 2) Disable button when input count is zero. | [UXArchitect]: 1) Ensure Escape closes modal cleanly.',
          worker_notes: 'Initial UI panel submission.',
        },
      ],
    });

    const digest = extractLatestRevisionSummary(item);
    expect(digest).not.toBeNull();
    expect(digest?.isChangesRequested).toBe(true);
    expect(digest?.isActiveChangesTarget).toBe(true);
    expect(digest?.reviewRevisionNumber).toBe(1);
    expect(digest?.feedbackByRole).toHaveLength(2);
    expect(digest?.feedbackByRole[0].role).toBe('SystemsDesigner');
    expect(digest?.feedbackByRole[1].role).toBe('UXArchitect');
  });

  it('skips deleted QA filmstrips from aborted is_failed_retry attempts and unlinked QA turns while preserving Council-reviewed prior revisions', () => {
    const item = makeItem({
      item_id: 'task_retry_proofs',
      state: 'in_progress',
      attempt_count: 2,
      revisions: [],
      latest_proof_urls: [],
    });

    // Attempt #1 crashed (is_failed_retry: true): Turn 4 wrote rev0.png, Turn 5 unlinked rev0.png during rendering_scene and failed.
    // Attempt #2 is now starting Turn 1 (swe_engineer) with 0 completed turns.
    const crashedRetryChronology: BoardItemChronologyResponse = {
      deliberation: 'Retry trace.',
      cartridge_id: 'candidate_synth_retry',
      item_id: 'task_retry_proofs',
      title: item.title,
      kind: item.kind,
      state: 'in_progress',
      spec: item.spec,
      target_modules: item.target_modules,
      depends_on: [],
      attempts: [
        {
          attempt_key: 'rev0_fail1',
          revision_index: 0,
          attempt_number: 1,
          is_failed_retry: true,
          status: 'failed',
          turns: [
            {
              turn_index: 3,
              specialist_agent: 'qa_test_engineer',
              verification_passed: false,
              timestamp_utc: '2026-10-31T16:16:35Z',
              proof_urls: [
                {
                  path: 'artifacts/work_items/task_retry_proofs/filmstrip_task_retry_proofs_rev0.png',
                  filename: 'filmstrip_task_retry_proofs_rev0.png',
                  url: '/api/artifacts/content?path=artifacts%2Fwork_items%2Ftask_retry_proofs%2Ffilmstrip_task_retry_proofs_rev0.png&raw=true',
                  is_audio: false,
                },
              ],
            },
            {
              turn_index: 4,
              specialist_agent: 'qa_test_engineer',
              verification_passed: false,
              timestamp_utc: '2026-10-31T16:17:15Z',
              step_log: [
                {
                  step_type: 'rendering_scene',
                  specialist_agent: 'qa_test_engineer',
                  message: 'Rendering scene res://tests/test_task_retry_proofs.tscn',
                },
              ],
              proof_urls: [],
            },
          ],
        },
        {
          attempt_key: 'rev0',
          revision_index: 0,
          attempt_number: 2,
          is_failed_retry: false,
          status: 'in_progress',
          turns: [],
        },
      ],
    };

    const emptyWhileRetrying = resolveCardProofShowcase(
      item,
      crashedRetryChronology,
      'candidate_synth_retry',
    );
    expect(emptyWhileRetrying.images).toHaveLength(0);
    expect(emptyWhileRetrying.sourceLabel).toBe('');

    // Once Attempt #2 regenerates filmstrip_task_retry_proofs_rev0.png, it resolves with a deterministic cache-busting &v= param
    const regeneratedChronology: BoardItemChronologyResponse = {
      ...crashedRetryChronology,
      attempts: [
        crashedRetryChronology.attempts[0],
        {
          attempt_key: 'rev0',
          revision_index: 0,
          attempt_number: 2,
          is_failed_retry: false,
          status: 'in_progress',
          turns: [
            {
              turn_index: 1,
              specialist_agent: 'qa_test_engineer',
              verification_passed: true,
              timestamp_utc: '2026-10-31T16:32:10Z',
              proof_urls: [
                {
                  path: 'artifacts/work_items/task_retry_proofs/filmstrip_task_retry_proofs_rev0.png',
                  filename: 'filmstrip_task_retry_proofs_rev0.png',
                  url: '/api/artifacts/content?path=artifacts%2Fwork_items%2Ftask_retry_proofs%2Ffilmstrip_task_retry_proofs_rev0.png&raw=true',
                  is_audio: false,
                },
              ],
            },
          ],
        },
      ],
    };

    const freshShowcase = resolveCardProofShowcase(
      item,
      regeneratedChronology,
      'candidate_synth_retry',
    );
    expect(freshShowcase.images).toHaveLength(1);
    expect(freshShowcase.sourceLabel).toBe('Turn #2 · qa_test_engineer');
    expect(freshShowcase.images[0].url).toContain('v=2026-10-31T16%3A32%3A10Z');

    // Conversely, when Attempt #1 was a completed Council-reviewed revision (is_failed_retry: false, rev0)
    // and Attempt #2 is working on rev1, Rev #1's preserved proof remains visible with a clear Rev #1 prefix!
    const councilRevChronology: BoardItemChronologyResponse = {
      ...crashedRetryChronology,
      attempts: [
        {
          attempt_key: 'rev0',
          revision_index: 0,
          attempt_number: 1,
          is_failed_retry: false,
          status: 'submitted',
          turns: [
            {
              turn_index: 5,
              specialist_agent: 'qa_test_engineer',
              verification_passed: true,
              timestamp_utc: '2026-10-31T15:50:00Z',
              proof_urls: [
                {
                  path: 'artifacts/work_items/task_retry_proofs/filmstrip_task_retry_proofs_rev0.png',
                  filename: 'filmstrip_task_retry_proofs_rev0.png',
                  url: '/api/artifacts/content?path=artifacts%2Fwork_items%2Ftask_retry_proofs%2Ffilmstrip_task_retry_proofs_rev0.png&raw=true',
                  is_audio: false,
                },
              ],
            },
          ],
        },
        {
          attempt_key: 'rev1',
          revision_index: 1,
          attempt_number: 2,
          is_failed_retry: false,
          status: 'in_progress',
          turns: [],
        },
      ],
    };

    const preservedRevShowcase = resolveCardProofShowcase(
      item,
      councilRevChronology,
      'candidate_synth_retry',
    );
    expect(preservedRevShowcase.images).toHaveLength(1);
    expect(preservedRevShowcase.sourceLabel).toBe('Rev #1 · Turn #6 · qa_test_engineer');
  });

  it('treats in_progress, waiting_for_lock, and awaiting_review as active working states', () => {
    expect(isWorkingBoardState('in_progress')).toBe(true);
    expect(isWorkingBoardState('waiting_for_lock')).toBe(true);
    expect(isWorkingBoardState('awaiting_review')).toBe(true);
    expect(isWorkingBoardState('ready')).toBe(false);
    expect(isWorkingBoardState('blocked')).toBe(false);
    expect(isWorkingBoardState('approved')).toBe(false);
  });
});
