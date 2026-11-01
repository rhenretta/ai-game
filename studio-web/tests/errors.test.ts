import { describe, expect, it } from 'vitest';
import {
  shouldSurfaceBuildError,
  splitErrorHeadlineAndTraceback,
} from '../src/domain/errors';
import type { ProductionBoardItem } from '../src/api/types';

function makeItem(overrides: Partial<ProductionBoardItem>): ProductionBoardItem {
  return {
    deliberation: 'Synthetic item rationale.',
    item_id: 'task_alpha',
    kind: 'feature_code',
    title: 'Synthetic Module',
    spec: 'Implement module.',
    target_modules: ['scripts/module.gd'],
    depends_on: [],
    state: 'in_progress',
    ...overrides,
  };
}

describe('build error visibility & self-correction filtering', () => {
  it('hides item-scoped errors once the task has self-corrected or been re-queued', () => {
    const lastError = {
      event_type: 'backlog_item_failed',
      agent_name: 'ProductionBoard',
      item_id: 'task_alpha',
      message: 'Traceback (most recent call last):\n  ...\nStudioAgentError: QA parse error',
      timestamp_utc: '2026-10-31T14:40:00Z',
    };

    // Re-queued / actively self-correcting (`in_progress`) -> hidden
    expect(
      shouldSurfaceBuildError({
        lastError,
        allBoardItems: [makeItem({ item_id: 'task_alpha', state: 'in_progress' })],
        isBuildRunning: false,
      }),
    ).toBe(false);

    // Self-corrected and `approved` -> hidden
    expect(
      shouldSurfaceBuildError({
        lastError,
        allBoardItems: [makeItem({ item_id: 'task_alpha', state: 'approved' })],
        isBuildRunning: false,
      }),
    ).toBe(false);
  });

  it('hides transient failed item errors while the Studio build is actively running to self-correct', () => {
    const lastError = {
      event_type: 'backlog_item_failed',
      agent_name: 'ProductionBoard',
      item_id: 'task_alpha',
      message: 'StudioAgentError: Transient headless check failed on Attempt #1',
      timestamp_utc: '2026-10-31T14:40:00Z',
    };

    expect(
      shouldSurfaceBuildError({
        lastError,
        allBoardItems: [makeItem({ item_id: 'task_alpha', state: 'failed', failure_count: 1 })],
        isBuildRunning: true,
      }),
    ).toBe(false);
  });

  it('surfaces unresolved failures when the build stops or trips the circuit breaker', () => {
    const unresolvedItemError = {
      event_type: 'backlog_item_failed',
      agent_name: 'ProductionBoard',
      item_id: 'task_alpha',
      message: 'StudioAgentError: Exhausted retries after spec resets',
      timestamp_utc: '2026-10-31T14:50:00Z',
    };

    expect(
      shouldSurfaceBuildError({
        lastError: unresolvedItemError,
        allBoardItems: [makeItem({ item_id: 'task_alpha', state: 'failed', failure_count: 4 })],
        isBuildRunning: false,
      }),
    ).toBe(true);

    expect(
      shouldSurfaceBuildError({
        lastError: {
          event_type: 'CIRCUIT_BREAKER_PAUSED',
          agent_name: 'StudioCircuitBreaker',
          message: 'Circuit breaker tripped due to upstream quota exhaustion.',
          timestamp_utc: '2026-10-31T14:51:00Z',
        },
        allBoardItems: [],
        isBuildRunning: false,
      }),
    ).toBe(true);
  });

  it('splits multi-line Python tracebacks into a concise headline plus full stack trace', () => {
    const raw = [
      '[ProductionBoard · backlog_item_failed] Traceback (most recent call last):',
      '  File "worker.py", line 42, in run',
      '    verify()',
      'StudioAgentError: Missing type declaration in test harness',
    ].join('\n');

    const { headline, stackTrace } = splitErrorHeadlineAndTraceback(raw);
    expect(headline).toBe(
      '[ProductionBoard · backlog_item_failed] StudioAgentError: Missing type declaration in test harness',
    );
    expect(stackTrace).toBe(raw);
  });
});
