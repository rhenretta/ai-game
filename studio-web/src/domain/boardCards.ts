import type {
  BoardItemChronologyResponse,
  ChronologyProofArtifact,
  ChronologyStepLogEntry,
  ChronologyTurn,
  ProductionBoardItem,
  WorkItemRevision,
} from '../api/types';
import type { LiveItemStep } from '../realtime/useSwarmSocket';
import {
  normalizeProofArtifacts,
  normalizeStepLog,
  type NormalizedProofItem,
  type NormalizedStepEntry,
} from './promptParser';
import {
  splitSequentialNumberedEnumeration,
  stripLeadingSingleListMarker,
} from './utterances';

export interface StructuredSpecSummary {
  leadSummary: string;
  preamble: string;
  points: string[];
  isExpandable: boolean;
}

export interface CardProofImage extends NormalizedProofItem {
  pairedAudioUrl?: string;
}

export interface CardProofShowcase {
  images: CardProofImage[];
  audios: NormalizedProofItem[];
  sourceLabel: string;
  totalCount: number;
}

export interface WorkerTurnDigest {
  turnNumber: number;
  agent: string;
  summary: string;
}

export interface CouncilRoleFeedback {
  role: string;
  rawText: string;
  points: string[];
}

export interface LatestRevisionDigest {
  revisionNumber: number;
  reviewRevisionNumber: number;
  verdict: string;
  isChangesRequested: boolean;
  isActiveChangesTarget: boolean;
  workerHeadline: string;
  workerStrategy: string;
  workerTurns: WorkerTurnDigest[];
  feedbackHeadline: string;
  feedbackByRole: CouncilRoleFeedback[];
  requiredChanges: string[];
}

export interface LiveTurnCardState {
  turnNumber: number;
  specialistAgent: string;
  headline: string;
  startedAtMs: number | null;
  steps: NormalizedStepEntry[];
}

export function splitNumberedPoints(rawText: string): string[] {
  const txt = String(rawText || '').trim();
  if (!txt) return [];

  const seq = splitSequentialNumberedEnumeration(txt);
  if (seq && seq.items.length >= 2) {
    return seq.items;
  }

  const lines = txt
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length >= 2 && lines.every((l) => /^[-*•]\s+/.test(l))) {
    return lines.map((l) => l.replace(/^[-*•]\s+/, '').trim()).filter(Boolean);
  }

  return [stripLeadingSingleListMarker(txt)];
}

export function splitSpecRequirements(
  cleanSpec: string,
  hasExtraDetails = false,
): StructuredSpecSummary {
  const text = String(cleanSpec || '').trim();
  if (!text) {
    return { leadSummary: '', preamble: '', points: [], isExpandable: hasExtraDetails };
  }
  const seq = splitSequentialNumberedEnumeration(text);
  const preamble = seq?.preamble ?? '';
  const points = seq ? seq.items : splitNumberedPoints(text);
  const firstSegment = preamble || points[0] || text;
  const leadSummary =
    firstSegment.length > 175 ? `${firstSegment.slice(0, 175).trim()}…` : firstSegment;
  const isExpandable =
    hasExtraDetails || points.length > 1 || text.length > leadSummary.length;
  return { leadSummary, preamble, points, isExpandable };
}

function rawPathsToProofUrls(paths: unknown[]): string[] {
  const out: string[] = [];
  for (const raw of paths) {
    if (typeof raw !== 'string' || !raw.trim()) continue;
    const trimmed = raw.trim();
    if (
      trimmed.startsWith('/api/') ||
      trimmed.startsWith('http://') ||
      trimmed.startsWith('https://') ||
      trimmed.startsWith('data:')
    ) {
      out.push(trimmed);
    } else {
      out.push(`/api/artifacts/content?path=${encodeURIComponent(trimmed)}&raw=true`);
    }
  }
  return out;
}

function deduplicateProofs(items: NormalizedProofItem[]): NormalizedProofItem[] {
  const seen = new Set<string>();
  const out: NormalizedProofItem[] = [];
  for (const p of items) {
    const key = p.url || p.path || p.filename;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(p);
  }
  return out;
}

export function appendProofVersionParam(url: string, versionToken: string): string {
  const token = versionToken.trim();
  if (
    !token ||
    !url ||
    url.startsWith('data:') ||
    url.startsWith('blob:') ||
    !url.includes('/api/artifacts/content?') ||
    /[?&]v=/.test(url)
  ) {
    return url;
  }
  return `${url}&v=${encodeURIComponent(token)}`;
}

/**
 * In `engine/studio/proofs.py` (`capture_filmstrip`), every `qa_test_engineer`
 * scene render for a given revision unlinks the shared `*_rev<N>.png` file
 * before launching the engine. If that QA turn subsequently aborts with zero
 * emitted proof images, any earlier `qa_test_engineer` filmstrip recorded in
 * the same revision no longer exists on disk.
 */
function didQaTurnUnlinkRevisionStrip(turn: ChronologyTurn): boolean {
  if (turn.specialist_agent !== 'qa_test_engineer') return false;
  const hasProofs =
    (Array.isArray(turn.proof_urls) && turn.proof_urls.length > 0) ||
    (Array.isArray(turn.proof_images) && turn.proof_images.length > 0);
  if (hasProofs) return false;

  if (!Array.isArray(turn.step_log) || turn.step_log.length === 0) {
    return turn.verification_passed === false;
  }

  return turn.step_log.some((entry) => {
    if (!entry || typeof entry !== 'object') return false;
    const stepType = String((entry as Record<string, unknown>).step_type ?? '');
    return stepType === 'rendering_scene';
  });
}

/**
 * Resolves the most up-to-date rendered visual & audio proofs for a board card,
 * prioritizing active/latest chronology turns (so mid-flight `in_progress` cards
 * immediately show newly rendered scenes, maps, sprites, or QA filmstrips),
 * skipping aborted `is_failed_retry` runs and QA filmstrips unlinked by a
 * subsequent failed QA turn on the same revision, and falling back to
 * `item.latest_proof_urls` and `item.revisions`.
 */
export function resolveCardProofShowcase(
  item: ProductionBoardItem,
  chronology: BoardItemChronologyResponse | undefined,
  cartridgeId: string,
): CardProofShowcase {
  let rawCandidates: Array<ChronologyProofArtifact | string> = [];
  let sourceLabel = '';
  let versionToken = '';

  // 1. Check live/latest chronology turns first when matching this item
  if (
    chronology &&
    chronology.item_id === item.item_id &&
    Array.isArray(chronology.attempts) &&
    chronology.attempts.length > 0
  ) {
    const hasNonFailedRetryAttempt = chronology.attempts.some((a) => !a.is_failed_retry);
    const allowFailedRetryFallback =
      !hasNonFailedRetryAttempt &&
      (item.state === 'failed' || item.state === 'needs_decision');
    const latestAttempt = chronology.attempts[chronology.attempts.length - 1];
    const latestRevisionIndex = latestAttempt?.revision_index ?? 0;
    const unlinkedQaRevisions = new Set<number>();

    for (let aIdx = chronology.attempts.length - 1; aIdx >= 0; aIdx -= 1) {
      const attempt = chronology.attempts[aIdx];
      const revIdx = attempt.revision_index ?? 0;
      const turns = Array.isArray(attempt.turns) ? attempt.turns : [];

      if (attempt.is_failed_retry && !allowFailedRetryFallback) {
        // Record whether this aborted attempt unlinked its revision strip, then skip it
        for (let tIdx = turns.length - 1; tIdx >= 0; tIdx -= 1) {
          if (didQaTurnUnlinkRevisionStrip(turns[tIdx])) {
            unlinkedQaRevisions.add(revIdx);
            break;
          }
        }
        continue;
      }

      for (let tIdx = turns.length - 1; tIdx >= 0; tIdx -= 1) {
        const turn = turns[tIdx];
        const turnProofs =
          Array.isArray(turn.proof_urls) && turn.proof_urls.length > 0
            ? turn.proof_urls
            : Array.isArray(turn.proof_images) && turn.proof_images.length > 0
              ? turn.proof_images
              : [];

        if (turnProofs.length === 0) {
          if (didQaTurnUnlinkRevisionStrip(turn)) {
            unlinkedQaRevisions.add(revIdx);
          }
          continue;
        }

        if (turn.specialist_agent === 'qa_test_engineer' && unlinkedQaRevisions.has(revIdx)) {
          continue;
        }

        rawCandidates = turnProofs;
        const displayTurn = (turn.turn_index ?? tIdx) + 1;
        const spec = turn.specialist_agent ? ` · ${turn.specialist_agent}` : '';
        const revPrefix = revIdx < latestRevisionIndex ? `Rev #${revIdx + 1} · ` : '';
        sourceLabel = `${revPrefix}Turn #${displayTurn}${spec}`;
        versionToken =
          turn.timestamp_utc ||
          `${attempt.attempt_key || `a${aIdx}`}_t${displayTurn}`;
        break;
      }

      if (rawCandidates.length === 0 && attempt.council_review?.proof_artifacts?.length) {
        rawCandidates = attempt.council_review.proof_artifacts;
        sourceLabel = `Rev #${revIdx + 1}`;
        versionToken = `${attempt.attempt_key || `rev${revIdx}`}_council`;
      }

      if (rawCandidates.length > 0) break;
    }
  }

  // 2. Fallback to item.latest_proof_urls or item.revisions[last]
  if (rawCandidates.length === 0) {
    const revCount = Array.isArray(item.revisions) ? item.revisions.length : 0;
    if (Array.isArray(item.latest_proof_urls) && item.latest_proof_urls.length > 0) {
      rawCandidates = item.latest_proof_urls;
      sourceLabel = revCount > 0 ? `Rev #${revCount}` : 'Latest Proof';
      versionToken = `rev${revCount}_att${item.attempt_count ?? 1}`;
    } else if (Array.isArray(item.revisions) && item.revisions.length > 0) {
      for (let rIdx = item.revisions.length - 1; rIdx >= 0; rIdx -= 1) {
        const rev = item.revisions[rIdx] as WorkItemRevision;
        if (!rev || typeof rev !== 'object') continue;
        const urls =
          Array.isArray(rev.proof_urls) && rev.proof_urls.length > 0
            ? rev.proof_urls
            : Array.isArray(rev.proof_paths) && rev.proof_paths.length > 0
              ? rawPathsToProofUrls(rev.proof_paths)
              : [];
        if (urls.length > 0) {
          const dispRev = (rev.revision_index ?? rIdx) + 1;
          rawCandidates = urls;
          sourceLabel = `Rev #${dispRev}`;
          versionToken = `rev${dispRev}`;
          break;
        }
      }
    }
  }

  if (rawCandidates.length === 0) {
    return { images: [], audios: [], sourceLabel: '', totalCount: 0 };
  }

  const normalized = deduplicateProofs(
    normalizeProofArtifacts(rawCandidates, undefined, cartridgeId),
  ).map((p) => ({
    ...p,
    url: appendProofVersionParam(p.url, versionToken),
  }));
  const audios = normalized.filter((p) => p.isAudio);
  const rawImages = normalized.filter((p) => !p.isAudio);

  const images: CardProofImage[] = rawImages.map((img) => {
    if (!img.isWaveform || audios.length === 0) {
      return img;
    }
    const stem = img.filename
      .replace(/^waveform_/i, '')
      .replace(/\.(png|jpg|jpeg|webp)$/i, '')
      .toLowerCase();
    const matchedAudio =
      audios.find((a) => a.filename.toLowerCase().includes(stem)) ?? audios[0];
    return {
      ...img,
      pairedAudioUrl: matchedAudio?.url,
    };
  });

  return {
    images,
    audios,
    sourceLabel: sourceLabel || 'Latest Proof',
    totalCount: normalized.length,
  };
}

function parseFeedbackIntoRoles(rawFeedback: string): CouncilRoleFeedback[] {
  const trimmed = rawFeedback.trim();
  if (!trimmed) return [];

  // Support both `' | '` joined segments and multi-line `\n[RoleName]:` blocks
  const segments = trimmed
    .split(/\s+\|\s+|\n+(?=\[[^\]]+\]:?\s)/)
    .map((s) => s.trim())
    .filter(Boolean);

  const out: CouncilRoleFeedback[] = [];
  for (const seg of segments) {
    const m = seg.match(/^\[([^\]]+)\]:?\s*([\s\S]*)$/);
    const role = m ? m[1].trim() : 'Specialist Council';
    const body = m ? m[2].trim() : seg;
    if (body) {
      out.push({
        role,
        rawText: body,
        points: splitNumberedPoints(body),
      });
    }
  }
  return out;
}

export function extractLatestRevisionSummary(
  item: ProductionBoardItem,
  chronology?: BoardItemChronologyResponse,
): LatestRevisionDigest | null {
  const revisions = Array.isArray(item.revisions) ? item.revisions : [];
  const lastIdx = revisions.length - 1;
  const latestWorkerRev =
    lastIdx >= 0 && revisions[lastIdx] && typeof revisions[lastIdx] === 'object'
      ? (revisions[lastIdx] as WorkItemRevision)
      : null;

  const revisionNumber = latestWorkerRev
    ? typeof latestWorkerRev.revision_index === 'number'
      ? latestWorkerRev.revision_index + 1
      : lastIdx + 1
    : 0;

  // Walk backward to find the most recent revision that received a Council verdict / feedback
  // (e.g., Rev #1's `request_changes` while Attempt #2 is `in_progress` or `awaiting_review`).
  let reviewedRev: WorkItemRevision | null = null;
  let reviewedIdx = -1;
  for (let i = revisions.length - 1; i >= 0; i -= 1) {
    const cand = revisions[i] as WorkItemRevision;
    if (!cand || typeof cand !== 'object') continue;
    const hasFeedback = typeof cand.feedback === 'string' && cand.feedback.trim().length > 0;
    const hasVerdict = typeof cand.verdict === 'string' && cand.verdict.trim().length > 0;
    if (hasFeedback || hasVerdict) {
      reviewedRev = cand;
      reviewedIdx = i;
      break;
    }
  }

  let verdict = typeof reviewedRev?.verdict === 'string' ? reviewedRev.verdict.trim() : '';
  let reviewRevisionNumber =
    reviewedRev !== null
      ? typeof reviewedRev.revision_index === 'number'
        ? reviewedRev.revision_index + 1
        : reviewedIdx + 1
      : revisionNumber;
  let rawFeedback = typeof reviewedRev?.feedback === 'string' ? reviewedRev.feedback.trim() : '';
  let feedbackByRole: CouncilRoleFeedback[] = rawFeedback ? parseFeedbackIntoRoles(rawFeedback) : [];

  // Fallback to chronology council_review if backlog item revisions were not populated
  if (
    feedbackByRole.length === 0 &&
    chronology &&
    chronology.item_id === item.item_id &&
    Array.isArray(chronology.attempts)
  ) {
    for (let aIdx = chronology.attempts.length - 1; aIdx >= 0; aIdx -= 1) {
      const cr = chronology.attempts[aIdx].council_review;
      if (!cr) continue;
      if (Array.isArray(cr.reviewer_breakdown) && cr.reviewer_breakdown.length > 0) {
        feedbackByRole = cr.reviewer_breakdown.map((rb) => {
          const rawBody = (
            rb.raw_text ||
            (Array.isArray(rb.points) ? rb.points.join(' ') : '')
          ).trim();
          return {
            role: rb.reviewer_role || 'Specialist Council',
            rawText: rawBody,
            points: splitNumberedPoints(rawBody),
          };
        });
        verdict = verdict || cr.verdict || (cr.approved ? 'approve' : 'request_changes');
        reviewRevisionNumber = (cr.revision_index ?? chronology.attempts[aIdx].revision_index ?? aIdx) + 1;
        rawFeedback = cr.feedback || cr.summary || '';
        break;
      }
      const crText = (cr.feedback || cr.summary || '').trim();
      if (crText) {
        feedbackByRole = parseFeedbackIntoRoles(crText);
        verdict = verdict || cr.verdict || (cr.approved ? 'approve' : 'request_changes');
        reviewRevisionNumber = (cr.revision_index ?? chronology.attempts[aIdx].revision_index ?? aIdx) + 1;
        rawFeedback = crText;
        break;
      }
    }
  }

  const rawNotes =
    typeof latestWorkerRev?.worker_notes === 'string' ? latestWorkerRev.worker_notes.trim() : '';
  let workerStrategy = '';
  let workerSummary = '';
  const workerTurns: WorkerTurnDigest[] = [];

  if (rawNotes) {
    const segments = rawNotes
      .split(' | ')
      .map((s) => s.trim())
      .filter(Boolean);
    for (const seg of segments) {
      if (seg.startsWith('Strategy:')) {
        workerStrategy = seg.replace(/^Strategy:\s*/, '').trim();
      } else if (seg.startsWith('Summary:')) {
        workerSummary = seg.replace(/^Summary:\s*/, '').trim();
      } else {
        const cleaned = seg.replace(/^Specialist Turns:\s*/, '').trim();
        const m = cleaned.match(/^\[Turn\s+(\d+)\s+([^\]]+)\]\s*([\s\S]*)$/i);
        if (m) {
          workerTurns.push({
            turnNumber: Number(m[1]) + 1,
            agent: m[2].trim(),
            summary: m[3].trim(),
          });
        } else if (cleaned) {
          if (!workerStrategy) workerStrategy = cleaned;
          else workerSummary = workerSummary ? `${workerSummary} · ${cleaned}` : cleaned;
        }
      }
    }
  }

  const rawHeadline =
    workerSummary ||
    workerStrategy ||
    (workerTurns.length > 0 ? workerTurns[workerTurns.length - 1].summary : rawNotes);
  const workerHeadline =
    rawHeadline.length > 175 ? `${rawHeadline.slice(0, 175).trim()}…` : rawHeadline;

  const firstFeedbackPoint =
    feedbackByRole.length > 0 && feedbackByRole[0].points.length > 0
      ? `[${feedbackByRole[0].role}] ${feedbackByRole[0].points[0]}`
      : rawFeedback;
  const feedbackHeadline =
    firstFeedbackPoint.length > 175
      ? `${firstFeedbackPoint.slice(0, 175).trim()}…`
      : firstFeedbackPoint;

  const requiredChanges = Array.isArray(reviewedRev?.required_changes)
    ? reviewedRev.required_changes.filter(
        (c): c is string => typeof c === 'string' && Boolean(c.trim()),
      )
    : [];

  if (!workerHeadline && !feedbackHeadline && requiredChanges.length === 0) {
    return null;
  }

  const verdictLower = verdict.toLowerCase();
  const isApprovedVerdict = verdictLower === 'approve' || verdictLower === 'approved';
  const isChangesRequested =
    item.state !== 'approved' &&
    !isApprovedVerdict &&
    (verdictLower === 'request_changes' ||
      verdictLower === 'changes_requested' ||
      Boolean(feedbackHeadline) ||
      requiredChanges.length > 0);

  // Actively addressing changes when not yet re-submitted (`awaiting_review`) or `approved`
  const isActiveChangesTarget =
    isChangesRequested && item.state !== 'awaiting_review' && item.state !== 'approved';

  return {
    revisionNumber: revisionNumber || reviewRevisionNumber || 1,
    reviewRevisionNumber: reviewRevisionNumber || 1,
    verdict,
    isChangesRequested,
    isActiveChangesTarget,
    workerHeadline,
    workerStrategy,
    workerTurns,
    feedbackHeadline,
    feedbackByRole,
    requiredChanges,
  };
}

export function mergeLiveTurnState(
  item: ProductionBoardItem,
  liveStep: LiveItemStep | undefined,
  chronology: BoardItemChronologyResponse | undefined,
): LiveTurnCardState {
  const matchingChronology =
    chronology && chronology.item_id === item.item_id ? chronology : undefined;
  const latestAttempt =
    matchingChronology && matchingChronology.attempts.length > 0
      ? matchingChronology.attempts[matchingChronology.attempts.length - 1]
      : undefined;
  const activeTurn = latestAttempt?.active_turn ?? null;

  const diskTurnNumber =
    activeTurn && typeof activeTurn.turn_index === 'number'
      ? activeTurn.turn_index + 1
      : latestAttempt
        ? latestAttempt.turns.length + 1
        : undefined;

  const turnNumber = liveStep?.turnIndex ?? diskTurnNumber ?? 1;

  const specialistAgent =
    liveStep?.specialistAgent ||
    (typeof activeTurn?.specialist_agent === 'string' && activeTurn.specialist_agent) ||
    liveStep?.agentName ||
    item.raised_by_role ||
    'task_lead';

  // Prefer the richer step log between live WS events and disk active_turn trace for the same turn
  const diskStepsRaw: Array<ChronologyStepLogEntry | Record<string, unknown> | string> =
    activeTurn &&
    (diskTurnNumber === undefined || diskTurnNumber === turnNumber) &&
    Array.isArray(activeTurn.step_log)
      ? activeTurn.step_log
      : [];
  const wsStepsRaw: ChronologyStepLogEntry[] =
    liveStep?.stepLog && liveStep.stepLog.length > 0 ? liveStep.stepLog : [];

  const chosenRawSteps =
    diskStepsRaw.length >= wsStepsRaw.length && diskStepsRaw.length > 0
      ? diskStepsRaw
      : wsStepsRaw;
  const steps = normalizeStepLog(chosenRawSteps);

  const startedIso =
    (typeof activeTurn?.started_at_utc === 'string' && activeTurn.started_at_utc) ||
    liveStep?.startedAtUtc ||
    liveStep?.timestampUtc ||
    '';
  const parsedStart = startedIso ? Date.parse(startedIso) : Number.NaN;
  const startedAtMs = Number.isNaN(parsedStart) ? null : parsedStart;

  const headline =
    liveStep?.stepSummary ||
    (steps.length > 0 ? steps[steps.length - 1].message : '') ||
    (typeof activeTurn?.brief === 'string' && activeTurn.brief) ||
    `Executing Turn #${turnNumber} on ${item.item_id}...`;

  return {
    turnNumber,
    specialistAgent,
    headline,
    startedAtMs,
    steps,
  };
}

export function formatElapsedDuration(elapsedSeconds: number): string {
  const safe = Math.max(0, Math.floor(elapsedSeconds));
  if (safe < 60) return `${safe}s`;
  const mins = Math.floor(safe / 60);
  const secs = safe % 60;
  return `${mins}m ${String(secs).padStart(2, '0')}s`;
}

export function isWorkingBoardState(state: string): boolean {
  return (
    state === 'in_progress' ||
    state === 'waiting_for_lock' ||
    state === 'awaiting_review'
  );
}

export function sortBoardItemsByPriority(
  items: ProductionBoardItem[],
  isBuildRunning: boolean,
): ProductionBoardItem[] {
  const getRank = (state: string): number => {
    switch (state) {
      case 'in_progress':
        return 1;
      case 'waiting_for_lock':
        return 2;
      case 'awaiting_review':
        return 3;
      case 'changes_requested':
        return 4;
      case 'failed':
      case 'needs_decision':
        return isBuildRunning ? 4 : 5;
      case 'ready':
        return 6;
      case 'blocked':
        return 7;
      case 'approved':
        return 8;
      default:
        return 9;
    }
  };

  return [...items].sort((a, b) => {
    const rankDiff = getRank(a.state) - getRank(b.state);
    if (rankDiff !== 0) return rankDiff;
    const prioA = typeof a.priority === 'number' ? a.priority : 100;
    const prioB = typeof b.priority === 'number' ? b.priority : 100;
    if (prioA !== prioB) return prioA - prioB;
    return 0;
  });
}
