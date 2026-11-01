import type { LastBuildError, ProductionBoardItem } from '../api/types';

export interface BuildErrorVisibilityInput {
  lastError?: LastBuildError | null;
  allBoardItems: ProductionBoardItem[];
  isBuildRunning: boolean;
  latestActivityUtc?: string | null;
}

/**
 * Determines whether `process_status.last_error` represents an unresolved terminal build/task failure
 * requiring operator attention, versus a transient worker/QA exception that the Task Lead or
 * Specialist Council has already self-corrected or is actively retrying while the build runs.
 */
export function shouldSurfaceBuildError({
  lastError,
  allBoardItems,
  isBuildRunning,
  latestActivityUtc,
}: BuildErrorVisibilityInput): boolean {
  if (!lastError || !lastError.message || !lastError.message.trim()) {
    return false;
  }

  // Authentic circuit breaker pauses always require operator attention
  if (lastError.event_type === 'CIRCUIT_BREAKER_PAUSED') {
    return true;
  }

  // If every non-cancelled board item is approved, any prior item error was resolved
  const activeItems = allBoardItems.filter((it) => it.state !== 'cancelled');
  if (
    lastError.item_id &&
    activeItems.length > 0 &&
    activeItems.every((it) => it.state === 'approved')
  ) {
    return false;
  }

  // Item-scoped errors (`backlog_item_failed`, `backlog_item_blocked`, or worker error with `item_id`)
  if (lastError.item_id) {
    const matched = allBoardItems.find((it) => it.item_id === lastError.item_id);
    if (matched) {
      // If the item already recovered or re-entered active execution/review/queue, hide the resolved error
      if (matched.state !== 'failed' && matched.state !== 'needs_decision') {
        return false;
      }
    }
    // While the Studio build is actively running, `_handle_failed_items` & Council Decision Rounds
    // automatically re-queue and self-correct failed items; only surface if the build stops unresolved.
    if (isBuildRunning) {
      return false;
    }
    return true;
  }

  // Non-item transient backlog events never halt the build on their own while running
  if (
    isBuildRunning &&
    (lastError.event_type === 'backlog_item_failed' ||
      lastError.event_type === 'backlog_item_blocked')
  ) {
    return false;
  }

  // If the build is actively running and newer studio activity has occurred after an earlier daemon error,
  // that error belonged to a prior run before resume.
  if (isBuildRunning) {
    if (lastError.timestamp_utc && latestActivityUtc) {
      const errMs = Date.parse(lastError.timestamp_utc);
      const actMs = Date.parse(latestActivityUtc);
      if (!Number.isNaN(errMs) && !Number.isNaN(actMs) && actMs > errMs + 1500) {
        return false;
      }
    }
    return false;
  }

  return true;
}

export function splitErrorHeadlineAndTraceback(rawMessage: string): {
  headline: string;
  stackTrace: string | null;
} {
  const text = (rawMessage ?? '').trim();
  if (!text) return { headline: '', stackTrace: null };

  const lines = text
    .split('\n')
    .map((l) => l.trimEnd())
    .filter((l) => l.trim().length > 0);

  if (lines.length <= 2 && !text.includes('Traceback (most recent call last):')) {
    return { headline: text, stackTrace: null };
  }

  if (text.includes('Traceback (most recent call last):')) {
    // Extract optional `[Agent · EventType]` prefix before `Traceback`
    const prefixMatch = text.match(/^(\[[^\]]+\])\s*Traceback/i);
    const prefix = prefixMatch ? `${prefixMatch[1]} ` : '';
    const lastLine = lines[lines.length - 1]?.trim() || lines[0]?.trim() || text;
    return {
      headline: `${prefix}${lastLine}`,
      stackTrace: text,
    };
  }

  return {
    headline: lines[0].trim(),
    stackTrace: text,
  };
}
