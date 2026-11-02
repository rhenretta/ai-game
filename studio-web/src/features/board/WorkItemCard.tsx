import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AlertCircle,
  Camera,
  CheckCircle2,
  Clock,
  Eye,
  FileCode2,
  GitBranch,
  History,
  Lock,
  RotateCcw,
  Volume2,
  Zap,
} from 'lucide-react';
import { studioApi } from '../../api/client';
import type { ProductionBoardItem } from '../../api/types';
import type { LiveItemStep } from '../../realtime/useSwarmSocket';
import { normalizeSpecAndDirections } from '../../domain/slices';
import { splitErrorHeadlineAndTraceback } from '../../domain/errors';
import {
  extractLatestRevisionSummary,
  formatElapsedDuration,
  mergeLiveTurnState,
  resolveCardProofShowcase,
  splitSpecRequirements,
} from '../../domain/boardCards';
import type { LightboxMedia } from '../../components/MediaLightbox';
import {
  FormattedInlineText,
  FormattedRichBlocks,
} from '../../components/RichTextBlocks';

export interface WorkItemCardProps {
  item: ProductionBoardItem;
  cartridgeId: string;
  isBuildRunning?: boolean;
  selected: boolean;
  liveStep?: LiveItemStep;
  onSelect: (itemId: string) => void;
  onOpenLightbox: (media: LightboxMedia) => void;
}

export function getStateBadge(
  state: string,
  isBuildRunning = false,
  decisionReason?: string | null,
): { label: string; cls: string; Icon: typeof CheckCircle2 } {
  switch (state) {
    case 'approved':
      return { label: 'Approved', cls: 'badge-ok', Icon: CheckCircle2 };
    case 'in_progress':
      return { label: 'Working', cls: 'badge-accent', Icon: Zap };
    case 'waiting_for_lock':
      return { label: 'File Lock', cls: 'badge-warn', Icon: Lock };
    case 'awaiting_review':
      return { label: 'In Review', cls: 'badge-purple', Icon: Eye };
    case 'changes_requested':
      return { label: 'Revising', cls: 'badge-warn', Icon: RotateCcw };
    case 'blocked':
      return { label: 'Waiting on Deps', cls: '', Icon: GitBranch };
    case 'cancelled':
      if (decisionReason && /^split\b/i.test(decisionReason.trim())) {
        return { label: 'Split', cls: 'badge-purple', Icon: GitBranch };
      }
      return { label: 'Cancelled', cls: '', Icon: Clock };
    case 'failed':
    case 'needs_decision':
      if (isBuildRunning) {
        return { label: 'Self-Correcting', cls: 'badge-warn', Icon: RotateCcw };
      }
      return { label: state.replace(/_/g, ' '), cls: 'badge-err', Icon: AlertCircle };
    default:
      return { label: state.replace(/_/g, ' ') || 'Ready', cls: '', Icon: Clock };
  }
}

export function WorkItemCard({
  item,
  cartridgeId,
  isBuildRunning = false,
  selected,
  liveStep,
  onSelect,
  onOpenLightbox,
}: WorkItemCardProps) {
  const { cleanSpec, directions } = useMemo(
    () => normalizeSpecAndDirections(item.spec, item.shaping_notes),
    [item.spec, item.shaping_notes],
  );
  const badge = getStateBadge(item.state, isBuildRunning, item.decision_reason);
  const BadgeIcon = badge.Icon;

  const isActivelyExecuting =
    item.state === 'in_progress' ||
    item.state === 'waiting_for_lock' ||
    item.state === 'awaiting_review' ||
    item.state === 'changes_requested';

  // Hydrate live disk trace & mid-flight turn proofs for active items (at most 2-3 workers at once)
  // and reuse any cached chronology already fetched for completed items.
  const chronologyQuery = useQuery({
    queryKey: ['board-chronology', cartridgeId, item.item_id],
    queryFn: () => studioApi.getBoardItemChronology(item.item_id, cartridgeId),
    enabled: Boolean(cartridgeId && isActivelyExecuting),
    refetchInterval: item.state === 'in_progress' ? 3000 : false,
    staleTime: 1500,
  });

  const proofShowcase = useMemo(
    () => resolveCardProofShowcase(item, chronologyQuery.data, cartridgeId),
    [item, chronologyQuery.data, cartridgeId],
  );

  const liveTurn = useMemo(
    () =>
      item.state === 'in_progress'
        ? mergeLiveTurnState(item, liveStep, chronologyQuery.data)
        : null,
    [item, liveStep, chronologyQuery.data],
  );

  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  useEffect(() => {
    if (item.state !== 'in_progress' || !liveTurn?.startedAtMs) return undefined;
    setNowMs(Date.now());
    const timer = window.setInterval(() => {
      setNowMs(Date.now());
    }, 1000);
    return () => window.clearInterval(timer);
  }, [item.state, liveTurn?.startedAtMs]);

  const elapsedStr =
    liveTurn?.startedAtMs !== null && liveTurn?.startedAtMs !== undefined
      ? formatElapsedDuration((nowMs - liveTurn.startedAtMs) / 1000)
      : null;

  const specSummary = useMemo(
    () =>
      splitSpecRequirements(
        cleanSpec,
        Boolean((item.proof_spec && item.proof_spec.trim()) || (item.deliberation && item.deliberation.trim())),
      ),
    [cleanSpec, item.proof_spec, item.deliberation],
  );

  const revisionDigest = useMemo(
    () => extractLatestRevisionSummary(item, chronologyQuery.data),
    [item, chronologyQuery.data],
  );

  const [failedProofUrls, setFailedProofUrls] = useState<Record<string, true>>({});

  const loadableProofImages = useMemo(
    () => proofShowcase.images.filter((img) => !failedProofUrls[img.url]),
    [proofShowcase.images, failedProofUrls],
  );
  const visibleProofImages = loadableProofImages.slice(-3);
  const hiddenProofCount = Math.max(0, loadableProofImages.length - visibleProofImages.length);

  const lockHoldersList: string[] = Array.isArray(item.lock_holders)
    ? item.lock_holders
    : item.lock_holders && typeof item.lock_holders === 'object'
      ? Object.values(item.lock_holders)
      : [];

  const showAttemptBadge =
    typeof item.attempt_count === 'number' &&
    item.attempt_count > 0 &&
    (item.attempt_count > 1 ||
      item.state === 'in_progress' ||
      item.state === 'awaiting_review' ||
      item.state === 'changes_requested');

  const parsedFailure =
    item.failure &&
    (item.state === 'failed' || item.state === 'needs_decision') &&
    !isBuildRunning
      ? splitErrorHeadlineAndTraceback(item.failure)
      : null;

  return (
    <div className="work-card-wrapper">
      <article
        className={`work-card${selected ? ' is-selected' : ''}`}
        data-state={item.state}
      >
        <header style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '6px', minInlineSize: 0 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '6px',
              flexWrap: 'wrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px', flexWrap: 'wrap', minInlineSize: 0 }}>
              <span className={`badge ${badge.cls}`}>
                <BadgeIcon size={11} />
                {badge.label}
              </span>
              <span
                className={`badge${item.kind === 'discovery' ? ' badge-purple' : ''}`}
                style={{ textTransform: 'lowercase' }}
              >
                {item.kind === 'discovery' ? <Eye size={10} /> : null}
                {item.kind}
              </span>
              {item.raised_by_role ? (
                <span className="badge" style={{ textTransform: 'none', color: 'var(--text-muted)' }}>
                  {item.raised_by_role}
                </span>
              ) : null}
              {showAttemptBadge ? (
                <span
                  className="badge badge-warn"
                  style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}
                  title="Execution attempt number"
                >
                  Attempt #{item.attempt_count}
                </span>
              ) : null}
              {revisionDigest?.isChangesRequested && item.state !== 'changes_requested' ? (
                <span
                  className="badge badge-warn"
                  style={{ textTransform: 'none' }}
                  title={`Specialist Council requested changes on Rev #${revisionDigest.reviewRevisionNumber}`}
                >
                  <RotateCcw size={10} />
                  Changes Requested (Rev #{revisionDigest.reviewRevisionNumber})
                </span>
              ) : null}
            </div>

            <button
              type="button"
              className="btn"
              style={{ paddingBlock: '2px', paddingInline: '8px', fontSize: '0.8rem' }}
              onClick={() => onSelect(item.item_id)}
            >
              <History size={11} />
              Timeline &amp; Details
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '2px' }}>
            <h3 style={{ margin: 0, fontSize: '1.02rem', fontWeight: 700, lineHeight: 1.32 }}>
              <button
                type="button"
                onClick={() => onSelect(item.item_id)}
                style={{
                  all: 'unset',
                  cursor: 'pointer',
                  color: 'var(--text-primary)',
                  display: 'block',
                  inlineSize: '100%',
                  overflowWrap: 'anywhere',
                }}
              >
                {item.title}
              </button>
            </h3>
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.76rem',
                color: 'var(--text-muted)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={item.item_id}
            >
              {item.item_id}
            </span>
          </div>
        </header>

        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '8px', minInlineSize: 0 }}>
          {/* 1. Latest Proof Rendered Directly Inside Card */}
          {visibleProofImages.length > 0 || proofShowcase.audios.length > 0 ? (
            <div className="card-proof-showcase">
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '8px',
                  flexWrap: 'wrap',
                  fontSize: '0.8rem',
                }}
              >
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '5px',
                    fontWeight: 700,
                    color: 'var(--accent)',
                  }}
                >
                  {visibleProofImages.length > 0 ? <Camera size={12} /> : <Volume2 size={12} />}
                  Latest Proof
                  {proofShowcase.sourceLabel ? (
                    <span
                      style={{
                        fontWeight: 500,
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.76rem',
                        color: 'var(--text-secondary)',
                      }}
                    >
                      · {proofShowcase.sourceLabel}
                    </span>
                  ) : null}
                </span>

                {hiddenProofCount > 0 ? (
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ padding: '1px 6px', fontSize: '0.76rem', color: 'var(--accent)' }}
                    onClick={() => onSelect(item.item_id)}
                  >
                    +{hiddenProofCount} more in timeline
                  </button>
                ) : (
                  <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                    Click proof to enlarge
                  </span>
                )}
              </div>

              {visibleProofImages.length > 0 ? (
                <div className={`card-proof-strip${visibleProofImages.length > 1 ? ' is-multi' : ''}`}>
                  {visibleProofImages.map((img) => (
                    <button
                      key={img.url}
                      type="button"
                      className="card-proof-frame"
                      onClick={() =>
                        onOpenLightbox({
                          url: img.url,
                          title: item.title,
                          subtitle: `${item.item_id} · ${img.filename} (${proofShowcase.sourceLabel})`,
                          deliberation: item.deliberation,
                          audioUrl: img.pairedAudioUrl,
                        })
                      }
                      title={
                        img.pairedAudioUrl
                          ? `Click to hear audio and enlarge waveform: ${img.filename}`
                          : `Click to inspect full-size proof: ${img.filename}`
                      }
                    >
                      <img
                        src={img.url}
                        alt={`Rendered proof ${img.filename} for ${item.title}`}
                        width={640}
                        height={360}
                        loading="lazy"
                        onError={() =>
                          setFailedProofUrls((prev) =>
                            prev[img.url] ? prev : { ...prev, [img.url]: true },
                          )
                        }
                      />
                      <span className="card-proof-caption">
                        <span
                          style={{
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            minInlineSize: 0,
                          }}
                        >
                          {img.filename}
                        </span>
                        {img.pairedAudioUrl ? (
                          <span style={{ color: 'var(--warn)', flexShrink: 0 }}>Audio</span>
                        ) : null}
                      </span>
                    </button>
                  ))}
                </div>
              ) : null}

              {proofShowcase.audios.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '5px' }}>
                  {proofShowcase.audios.slice(0, 2).map((aud) => (
                    <div
                      key={aud.url}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '4px 8px',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--bg-canvas)',
                        border: '1px solid var(--border-subtle)',
                        minInlineSize: 0,
                      }}
                    >
                      <Volume2 size={12} style={{ color: 'var(--warn)', flexShrink: 0 }} />
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.76rem',
                          color: 'var(--text-secondary)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          maxInlineSize: '180px',
                        }}
                        title={aud.filename}
                      >
                        {aud.filename}
                      </span>
                      <audio
                        controls
                        preload="metadata"
                        src={aud.url}
                        style={{ blockSize: '26px', flex: 1, minInlineSize: '110px' }}
                      />
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}

          {/* 2. Council Requested Changes (Visible prominently during Attempt #2+ and Revising!) */}
          {revisionDigest?.isChangesRequested &&
          (revisionDigest.feedbackByRole.length > 0 ||
            revisionDigest.feedbackHeadline ||
            revisionDigest.requiredChanges.length > 0) ? (
            <details
              open={revisionDigest.isActiveChangesTarget ? true : undefined}
              className="card-disclosure is-feedback-warn"
            >
              <summary>
                <strong style={{ color: 'var(--warn)' }}>
                  Council Requested Changes · Rev #{revisionDigest.reviewRevisionNumber}
                  {revisionDigest.feedbackByRole.length > 0
                    ? ` (${revisionDigest.feedbackByRole.length} reviewer${
                        revisionDigest.feedbackByRole.length === 1 ? '' : 's'
                      })`
                    : ''}
                </strong>
                {revisionDigest.feedbackHeadline ? (
                  <span className="disclosure-preview">: {revisionDigest.feedbackHeadline}</span>
                ) : null}
              </summary>
              <div
                style={{
                  marginBlockStart: '7px',
                  display: 'grid',
                  gridTemplateColumns: 'minmax(0, 1fr)',
                  gap: '6px',
                  fontSize: '0.8rem',
                  maxBlockSize: '185px',
                  overflowY: 'auto',
                }}
              >
                {revisionDigest.feedbackByRole.map((rb, idx) => (
                  <div
                    key={`${rb.role}_${idx}`}
                    style={{
                      padding: '6px 8px',
                      borderRadius: '4px',
                      background: 'var(--bg-surface)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    <div style={{ fontWeight: 700, color: 'var(--warn)', marginBlockEnd: '4px' }}>
                      {rb.role}
                    </div>
                    <FormattedRichBlocks text={rb.rawText || rb.points.join('\n')} />
                  </div>
                ))}
                {revisionDigest.requiredChanges.length > 0 ? (
                  <ul className="studio-rich-list" style={{ color: 'var(--text-primary)' }}>
                    {revisionDigest.requiredChanges.map((rc, rIdx) => (
                      <li key={rIdx}>
                        <FormattedInlineText text={rc.replace(/^(?:\d+[.)]|[-*•])\s+/, '')} />
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            </details>
          ) : null}

          {/* 3. Live Executing Turn & Inline Step Log */}
          {item.state === 'in_progress' && liveTurn ? (
            <div className="card-live-turn">
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '8px',
                  flexWrap: 'wrap',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '7px', flexWrap: 'wrap', minInlineSize: 0 }}>
                  <span className="live-pulse-dot" aria-hidden="true" />
                  <span
                    className="badge badge-accent"
                    style={{ textTransform: 'none', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}
                  >
                    Executing Turn #{liveTurn.turnNumber} · {liveTurn.specialistAgent}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {elapsedStr ? (
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        color: 'var(--warn)',
                      }}
                    >
                      {elapsedStr}
                    </span>
                  ) : null}
                  <button
                    type="button"
                    className="btn btn-primary"
                    style={{ paddingBlock: '2px', paddingInline: '8px', fontSize: '0.8rem' }}
                    onClick={() => onSelect(item.item_id)}
                  >
                    Drill Down Turn
                  </button>
                </div>
              </div>

              <div
                style={{
                  fontSize: '0.84rem',
                  color: 'var(--text-primary)',
                  lineHeight: 1.42,
                  overflowWrap: 'anywhere',
                }}
              >
                {liveTurn.headline}
              </div>

              {liveTurn.steps.length > 0 ? (
                <details
                  open
                  style={{
                    background: 'color-mix(in srgb, var(--bg-canvas) 82%, transparent)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '5px 8px',
                    minInlineSize: 0,
                  }}
                >
                  <summary
                    style={{
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      fontWeight: 700,
                      color: 'var(--ok)',
                      userSelect: 'none',
                    }}
                  >
                    Live Turn Activity Log ({liveTurn.steps.length} step
                    {liveTurn.steps.length === 1 ? '' : 's'})
                  </summary>
                  <div
                    style={{
                      marginBlockStart: '6px',
                      display: 'grid',
                      gridTemplateColumns: 'minmax(0, 1fr)',
                      gap: '4px',
                      maxBlockSize: '155px',
                      overflowY: 'auto',
                    }}
                  >
                    {liveTurn.steps.map((step) => (
                      <div
                        key={`${step.index}_${step.stepType}`}
                        style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '6px',
                          padding: '3px 6px',
                          borderRadius: '4px',
                          background: 'var(--bg-surface)',
                          borderInlineStart: `2px solid ${
                            step.badgeClass === 'badge-err' ? 'var(--err)' : 'var(--accent)'
                          }`,
                          fontSize: '0.8rem',
                          minInlineSize: 0,
                        }}
                      >
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--warn)',
                            fontSize: '0.73rem',
                            flexShrink: 0,
                          }}
                        >
                          #{step.index}
                        </span>
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--accent)',
                            fontWeight: 700,
                            fontSize: '0.73rem',
                            flexShrink: 0,
                          }}
                        >
                          [{step.stepType}]
                        </span>
                        <span
                          style={{
                            color: 'var(--text-primary)',
                            flex: 1,
                            minInlineSize: 0,
                            overflowWrap: 'anywhere',
                          }}
                        >
                          {step.message}
                        </span>
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}
            </div>
          ) : null}

          {/* 3. Paused File Lock State */}
          {item.state === 'waiting_for_lock' ? (
            <div
              style={{
                padding: '8px 10px',
                borderRadius: 'var(--radius-sm)',
                background: 'color-mix(in srgb, var(--warn) 10%, var(--bg-canvas))',
                border: '1px solid color-mix(in srgb, var(--warn) 45%, var(--border-subtle))',
                fontSize: '0.84rem',
                display: 'grid',
                gridTemplateColumns: 'minmax(0, 1fr)',
                gap: '4px',
                overflowWrap: 'anywhere',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap' }}>
                <span className="badge badge-warn" style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}>
                  Paused Before Turn #{item.yielded_turn_info?.next_turn_index ?? 1}
                  {item.yielded_turn_info?.specialist_agent
                    ? ` · ${item.yielded_turn_info.specialist_agent}`
                    : ''}
                </span>
                {item.yielded_turn_info?.completed_turns ? (
                  <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                    {item.yielded_turn_info.completed_turns} turn
                    {item.yielded_turn_info.completed_turns === 1 ? '' : 's'} completed
                  </span>
                ) : null}
              </div>
              <div style={{ color: 'var(--text-secondary)' }}>
                Waiting on file lock:{' '}
                <code style={{ color: 'var(--warn)' }}>
                  {(item.waiting_on_locks && item.waiting_on_locks.length > 0
                    ? item.waiting_on_locks
                    : item.target_modules
                  ).join(', ') || 'shared module'}
                </code>
                {lockHoldersList.length > 0 ? (
                  <span>
                    {' '}
                    (held by <code style={{ color: 'var(--accent)' }}>{lockHoldersList.join(', ')}</code>)
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}

          {/* 3b. Active Specialist Council Review State */}
          {item.state === 'awaiting_review' ? (
            <div
              style={{
                padding: '8px 10px',
                borderRadius: 'var(--radius-sm)',
                background: 'color-mix(in srgb, var(--purple) 11%, var(--bg-canvas))',
                border: '1px solid color-mix(in srgb, var(--purple) 48%, var(--border-subtle))',
                fontSize: '0.84rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '8px',
                flexWrap: 'wrap',
                overflowWrap: 'anywhere',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '7px', minInlineSize: 0 }}>
                {isBuildRunning ? (
                  <span
                    className="live-pulse-dot"
                    style={{ background: 'var(--purple)', boxShadow: '0 0 8px var(--purple)' }}
                    aria-hidden="true"
                  />
                ) : null}
                <span style={{ fontWeight: 700, color: 'var(--purple)' }}>
                  Specialist Council Reviewing Submission
                </span>
              </div>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Evaluating rendered proofs &amp; spec gates
              </span>
            </div>
          ) : null}

          {/* 4. Collapsible Spec & Proof Gate */}
          {cleanSpec ? (
            specSummary.isExpandable ? (
              <details className="card-disclosure">
                <summary>
                  <strong style={{ color: 'var(--text-primary)' }}>Spec:</strong>{' '}
                  <span className="disclosure-preview">
                    <FormattedInlineText text={specSummary.leadSummary} />{' '}
                  </span>
                  <span
                    style={{
                      color: 'var(--accent)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.76rem',
                      marginInlineStart: '4px',
                    }}
                  >
                    (
                    {specSummary.points.length > 1
                      ? `${specSummary.points.length} requirements`
                      : 'details'}
                    )
                  </span>
                </summary>

                <div
                  style={{
                    marginBlockStart: '8px',
                    display: 'grid',
                    gridTemplateColumns: 'minmax(0, 1fr)',
                    gap: '7px',
                    fontSize: '0.84rem',
                    color: 'var(--text-secondary)',
                  }}
                >
                  {item.deliberation ? (
                    <div
                      style={{
                        borderInlineStart: '2px solid var(--purple)',
                        paddingInlineStart: '8px',
                        color: 'var(--purple)',
                      }}
                    >
                      <strong>Rationale:</strong>{' '}
                      <FormattedInlineText text={item.deliberation} />
                    </div>
                  ) : null}

                  <FormattedRichBlocks text={cleanSpec} />

                  {item.proof_spec ? (
                    <div
                      style={{
                        padding: '5px 8px',
                        borderRadius: '4px',
                        background: 'color-mix(in srgb, var(--ok) 9%, var(--bg-canvas))',
                        borderInlineStart: '2px solid var(--ok)',
                        color: 'var(--text-primary)',
                        fontSize: '0.8rem',
                      }}
                    >
                      <strong style={{ color: 'var(--ok)' }}>Proof Gate:</strong>{' '}
                      <FormattedInlineText text={item.proof_spec} />
                    </div>
                  ) : null}
                </div>
              </details>
            ) : (
              <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                <strong style={{ color: 'var(--text-primary)' }}>Spec:</strong>{' '}
                <FormattedInlineText text={cleanSpec} />
              </p>
            )
          ) : null}

          {/* 5. Collapsible Council Task Direction & Decomposition */}
          {directions.length > 0 || item.spawned_by_item_id ? (
            <details className="card-disclosure is-directions">
              <summary>
                <strong style={{ color: 'var(--warn)' }}>
                  Council Direction ({directions.length} input{directions.length === 1 ? '' : 's'}
                  {item.spawned_by_item_id ? ` · from ${item.spawned_by_item_id}` : ''})
                </strong>
                <span className="disclosure-preview">
                  :{' '}
                  {directions.length > 0
                    ? `[${directions[0].role}] ${directions[0].text.slice(0, 125)}${
                        directions[0].text.length > 125 ? '…' : ''
                      }`
                    : `Split from parent task ${item.spawned_by_item_id}`}
                </span>
              </summary>
              <div
                style={{
                  marginBlockStart: '8px',
                  display: 'grid',
                  gridTemplateColumns: 'minmax(0, 1fr)',
                  gap: '6px',
                }}
              >
                {directions.map((dir, idx) => (
                  <div
                    key={`${dir.role}_${idx}`}
                    style={{
                      padding: '6px 8px',
                      borderRadius: '4px',
                      background: 'var(--bg-surface)',
                      border: '1px solid var(--border-subtle)',
                      fontSize: '0.8rem',
                    }}
                  >
                    <div style={{ fontWeight: 700, color: 'var(--warn)', marginBlockEnd: '4px' }}>
                      {dir.role}
                    </div>
                    <FormattedRichBlocks text={dir.text} />
                  </div>
                ))}
              </div>
            </details>
          ) : null}

          {/* 6. Approved Council Review Note or Worker Summary */}
          {!revisionDigest?.isChangesRequested && revisionDigest?.feedbackHeadline ? (
            <details className="card-disclosure is-feedback-ok">
              <summary>
                <strong style={{ color: 'var(--ok)' }}>
                  Council Approved (Rev #{revisionDigest.reviewRevisionNumber}):
                </strong>{' '}
                <span>{revisionDigest.feedbackHeadline}</span>
              </summary>
              <div
                style={{
                  marginBlockStart: '7px',
                  display: 'grid',
                  gridTemplateColumns: 'minmax(0, 1fr)',
                  gap: '6px',
                  fontSize: '0.8rem',
                }}
              >
                {revisionDigest.feedbackByRole.map((rb, idx) => (
                  <div
                    key={`${rb.role}_${idx}`}
                    style={{
                      padding: '5px 8px',
                      borderRadius: '4px',
                      background: 'var(--bg-surface)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    <div style={{ fontWeight: 700, color: 'var(--ok)', marginBlockEnd: '4px' }}>
                      {rb.role}
                    </div>
                    <FormattedRichBlocks text={rb.rawText || rb.points.join('\n')} />
                  </div>
                ))}
              </div>
            </details>
          ) : null}

          {item.state !== 'in_progress' && revisionDigest?.workerHeadline ? (
            <details className="card-disclosure">
              <summary>
                <strong style={{ color: 'var(--accent)' }}>
                  Worker Summary (Rev #{revisionDigest.revisionNumber}):
                </strong>{' '}
                <span>{revisionDigest.workerHeadline}</span>
              </summary>
              {revisionDigest.workerTurns.length > 0 ? (
                <div
                  style={{
                    marginBlockStart: '7px',
                    display: 'grid',
                    gridTemplateColumns: 'minmax(0, 1fr)',
                    gap: '4px',
                    fontSize: '0.8rem',
                  }}
                >
                  {revisionDigest.workerTurns.map((wt) => (
                    <div
                      key={`${wt.turnNumber}_${wt.agent}`}
                      style={{
                        padding: '4px 7px',
                        borderRadius: '4px',
                        background: 'var(--bg-surface)',
                        border: '1px solid var(--border-subtle)',
                      }}
                    >
                      <strong style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>
                        Turn #{wt.turnNumber} ({wt.agent}):
                      </strong>{' '}
                      <span style={{ color: 'var(--text-secondary)' }}>{wt.summary}</span>
                    </div>
                  ))}
                </div>
              ) : null}
            </details>
          ) : null}

          {/* 7. Unresolved Halted Error (Hidden automatically while Studio is actively self-correcting) */}
          {parsedFailure ? (
            <details
              className="card-disclosure"
              style={{
                borderColor: 'color-mix(in srgb, var(--err) 55%, var(--border-subtle))',
                background: 'color-mix(in srgb, var(--err) 9%, var(--bg-canvas))',
              }}
            >
              <summary style={{ color: 'var(--err)', fontWeight: 700 }}>
                Halted Error: <span style={{ fontWeight: 400 }}>{parsedFailure.headline}</span>
              </summary>
              {parsedFailure.stackTrace ? (
                <pre className="prompt-pre" style={{ marginBlockStart: '6px', maxBlockSize: '180px', color: 'var(--err)' }}>
                  {parsedFailure.stackTrace}
                </pre>
              ) : null}
            </details>
          ) : null}
        </div>

        <footer
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '6px 12px',
            paddingBlockStart: '6px',
            borderBlockStart: '1px solid var(--border-subtle)',
            fontSize: '0.8rem',
            color: 'var(--text-muted)',
            minInlineSize: 0,
          }}
        >
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'flex-start',
              gap: '5px',
              minInlineSize: 0,
              flexWrap: 'nowrap',
            }}
            title={item.target_modules.join(', ')}
          >
            <FileCode2 size={12} style={{ flexShrink: 0, marginBlockStart: '2px' }} />
            <code
              style={{
                fontSize: '0.76rem',
                color: 'var(--text-secondary)',
                minInlineSize: 0,
                overflowWrap: 'anywhere',
              }}
            >
              {item.target_modules.length > 0
                ? item.target_modules.slice(0, 3).join(', ')
                : item.item_id}
            </code>
            {item.target_modules.length > 3 ? (
              <span className="badge" style={{ fontSize: '0.73rem', textTransform: 'none', flexShrink: 0 }}>
                +{item.target_modules.length - 3} more
              </span>
            ) : null}
          </span>

          {item.depends_on.length > 0 ? (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'flex-start',
                gap: '4px',
                minInlineSize: 0,
                flexWrap: 'nowrap',
              }}
              title={`Depends on: ${item.depends_on.join(', ')}`}
            >
              <GitBranch size={11} style={{ flexShrink: 0, marginBlockStart: '2px' }} />
              <span style={{ flexShrink: 0 }}>Depends on:</span>
              <code
                style={{
                  fontSize: '0.76rem',
                  color: 'var(--text-secondary)',
                  minInlineSize: 0,
                  overflowWrap: 'anywhere',
                }}
              >
                {item.depends_on.slice(0, 2).join(', ')}
                {item.depends_on.length > 2 ? ` +${item.depends_on.length - 2}` : ''}
              </code>
            </span>
          ) : null}
        </footer>
      </article>
    </div>
  );
}
