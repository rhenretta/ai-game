import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AlertTriangle,
  CheckCircle2,
  FileCode,
  HelpCircle,
  Image as ImageIcon,
  Layers,
  Maximize2,
  Minimize2,
  RefreshCw,
  ShieldAlert,
  X,
} from 'lucide-react';
import { studioApi } from '../../api/client';
import type {
  ChronologyContextPack,
  ChronologySelfInquiry,
  ProductionBoardItem,
} from '../../api/types';
import type { LiveItemStep } from '../../realtime/useSwarmSocket';
import { normalizeSpecAndDirections } from '../../domain/slices';
import { normalizeProofArtifacts, type NormalizedProofItem } from '../../domain/promptParser';
import { getRoleMetadata } from '../../domain/roles';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';
import { ErrorBanner } from '../../components/ErrorBanner';
import type { LightboxMedia } from '../../components/MediaLightbox';
import {
  FormattedInlineText,
  FormattedRichBlocks,
} from '../../components/RichTextBlocks';
import { getStateBadge } from './WorkItemCard';
import { TurnDetailCard, formatSpecialistLabel } from './TurnDetailCard';

export type InspectorSubtab = 'timeline' | 'spec' | 'context' | 'files';

export interface TaskChronologyDrawerProps {
  item: ProductionBoardItem;
  cartridgeId: string;
  isBuildRunning?: boolean;
  liveStep?: LiveItemStep;
  wideMode?: boolean;
  onToggleWideMode?: () => void;
  onClose: () => void;
  onOpenLightbox: (media: LightboxMedia) => void;
}

export function TaskChronologyDrawer({
  item,
  cartridgeId,
  isBuildRunning = false,
  liveStep,
  wideMode = false,
  onToggleWideMode,
  onClose,
  onOpenLightbox,
}: TaskChronologyDrawerProps) {
  const [activeSubtab, setActiveSubtab] = useState<InspectorSubtab>('timeline');
  const [attemptFilter, setAttemptFilter] = useState<string>('all');

  useEffect(() => {
    setAttemptFilter('all');
    setActiveSubtab('timeline');
  }, [item.item_id]);

  const isLiveState =
    item.state === 'in_progress' ||
    item.state === 'awaiting_review' ||
    item.state === 'waiting_for_lock';

  const chronQuery = useQuery({
    queryKey: ['board-chronology', cartridgeId, item.item_id],
    queryFn: () => studioApi.getBoardItemChronology(item.item_id, cartridgeId),
    refetchInterval: isLiveState ? 2500 : false,
  });

  // Immediately pull fresh disk trace when WebSocket streams a new substep for this task
  useEffect(() => {
    if (liveStep?.timestampUtc) {
      void chronQuery.refetch();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveStep?.timestampUtc]);

  const chron = chronQuery.data;
  const effectiveState = chron?.state || item.state;
  const stateBadge = getStateBadge(effectiveState, isBuildRunning, item.decision_reason);
  const StateBadgeIcon = stateBadge.Icon;

  const { cleanSpec, directions } = useMemo(
    () =>
      normalizeSpecAndDirections(
        chron?.spec ?? item.spec,
        chron?.shaping_notes ?? item.shaping_notes,
      ),
    [chron?.spec, chron?.shaping_notes, item.spec, item.shaping_notes],
  );

  const attempts = useMemo(() => chron?.attempts ?? [], [chron?.attempts]);
  const contextPacks: ChronologyContextPack[] = useMemo(
    () => chron?.context_packs ?? [],
    [chron?.context_packs],
  );
  const fileContents = useMemo(() => chron?.file_contents ?? {}, [chron?.file_contents]);

  // Aggregate global + per-turn self-inquiries (deduplicated by question+answer)
  const allInquiries: ChronologySelfInquiry[] = useMemo(() => {
    const seen = new Set<string>();
    const out: ChronologySelfInquiry[] = [];
    const pushInquiry = (qa: ChronologySelfInquiry, fallbackRole?: string) => {
      if (!qa || (!qa.question && !qa.answer)) return;
      const key = `${qa.agent_role || fallbackRole || ''}::${qa.question}::${qa.answer}`;
      if (seen.has(key)) return;
      seen.add(key);
      out.push({
        ...qa,
        agent_role: qa.agent_role || fallbackRole,
      });
    };

    for (const qa of chron?.self_inquiries ?? []) {
      pushInquiry(qa);
    }
    for (const att of attempts) {
      for (const t of att.turns) {
        for (const lq of t.lead_self_inquiries ?? []) pushInquiry(lq, 'task_lead');
        for (const sq of t.self_inquiries ?? []) pushInquiry(sq, t.specialist_agent);
      }
      if (att.active_turn) {
        for (const lq of att.active_turn.lead_self_inquiries ?? []) pushInquiry(lq, 'task_lead');
        for (const sq of att.active_turn.self_inquiries ?? []) {
          pushInquiry(sq, att.active_turn.specialist_agent);
        }
      }
    }
    return out;
  }, [chron?.self_inquiries, attempts]);

  // Aggregate deduplicated proof artifacts across card + turns + council reviews
  const allProofs: NormalizedProofItem[] = useMemo(() => {
    const seen = new Set<string>();
    const out: NormalizedProofItem[] = [];
    const addList = (list: NormalizedProofItem[]) => {
      for (const p of list) {
        if (!p.url || seen.has(p.url)) continue;
        seen.add(p.url);
        out.push(p);
      }
    };
    addList(normalizeProofArtifacts(item.latest_proof_urls, undefined, cartridgeId));
    for (const att of attempts) {
      for (const t of att.turns) {
        addList(normalizeProofArtifacts(t.proof_urls, t.proof_images, cartridgeId));
      }
      if (att.council_review?.proof_artifacts) {
        addList(normalizeProofArtifacts(att.council_review.proof_artifacts, undefined, cartridgeId));
      }
    }
    return out;
  }, [item.latest_proof_urls, attempts, cartridgeId]);

  const totalTurnsCount = useMemo(
    () => attempts.reduce((acc, a) => acc + a.turns.length + (a.active_turn ? 1 : 0), 0),
    [attempts],
  );

  const visibleAttempts = useMemo(() => {
    if (attemptFilter === 'all') return attempts;
    return attempts.filter((a) => String(a.attempt_key || a.revision_index) === attemptFilter);
  }, [attempts, attemptFilter]);

  const failureCount = chron?.failure_count ?? item.failure_count ?? 0;
  const specResets = chron?.spec_resets ?? item.spec_resets ?? 0;
  const effectiveFailure = (chron?.failure || item.failure || '').trim();
  const isTerminalFailure =
    !isBuildRunning &&
    (effectiveState === 'failed' || effectiveState === 'needs_decision');
  const showFailureTrace =
    Boolean(effectiveFailure) &&
    effectiveState !== 'approved' &&
    effectiveState !== 'cancelled';

  const fileEntries = useMemo(() => Object.entries(fileContents), [fileContents]);

  const renderSpecAndShapingContent = () => (
    <div style={{ display: 'grid', gap: '12px' }}>
      {/* Rule 4: True Task Epistemic Rationale (never overwritten by HTTP wrapper message) */}
      <DeliberationDisclosure
        deliberation={item.deliberation}
        label="Task Epistemic Rationale"
        defaultOpen
      />

      <section style={{ display: 'grid', gap: '6px' }}>
        <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
          Specification &amp; Acceptance Criteria
        </div>
        <div
          style={{
            padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-canvas)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.87rem',
            lineHeight: 1.5,
            overflowWrap: 'anywhere',
          }}
        >
          <FormattedRichBlocks text={cleanSpec} />
        </div>
        {(chron?.proof_spec || item.proof_spec) ? (
          <div
            style={{
              padding: '8px 10px',
              borderRadius: 'var(--radius-sm)',
              background: 'color-mix(in srgb, var(--ok) 8%, var(--bg-canvas))',
              border: '1px solid color-mix(in srgb, var(--ok) 35%, var(--border-subtle))',
              fontSize: '0.84rem',
            }}
          >
            <strong style={{ color: 'var(--ok)' }}>Proof Requirement: </strong>
            <FormattedInlineText text={chron?.proof_spec || item.proof_spec || ''} />
          </div>
        ) : null}
      </section>

      {(item.target_modules.length > 0 || item.depends_on.length > 0 || chron?.spawned_by_item_id || item.spawned_by_item_id) ? (
        <section style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {item.target_modules.map((m) => (
            <span
              key={m}
              className="badge badge-accent"
              style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}
            >
              <FileCode size={11} /> {m}
            </span>
          ))}
          {item.depends_on.map((dep) => (
            <span
              key={dep}
              className="badge"
              style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}
            >
              dep: {dep}
            </span>
          ))}
          {(chron?.spawned_by_item_id || item.spawned_by_item_id) ? (
            <span
              className="badge badge-purple"
              style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}
            >
              spawned by: {chron?.spawned_by_item_id || item.spawned_by_item_id}
            </span>
          ) : null}
        </section>
      ) : null}

      {directions.length > 0 ? (
        <details
          open
          style={{
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)',
            background: 'var(--bg-elevated)',
            padding: '8px 10px',
          }}
        >
          <summary style={{ cursor: 'pointer', fontWeight: 700, fontSize: '0.87rem' }}>
            Specialist Council Task Directions ({directions.length})
          </summary>
          <div style={{ display: 'grid', gap: '8px', marginBlockStart: '8px' }}>
            {directions.map((d, idx) => {
              const meta = getRoleMetadata(d.role);
              return (
                <div
                  key={idx}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-canvas)',
                    border: '1px solid var(--border-subtle)',
                    display: 'grid',
                    gap: '4px',
                  }}
                >
                  <span style={{ fontWeight: 700, fontSize: '0.84rem', color: meta.accentVar }}>
                    {meta.label}
                  </span>
                  <DeliberationDisclosure deliberation={d.deliberation} />
                  <div
                    style={{
                      fontSize: '0.87rem',
                      color: 'var(--text-secondary)',
                      overflowWrap: 'anywhere',
                    }}
                  >
                    <FormattedRichBlocks text={d.text} />
                  </div>
                </div>
              );
            })}
          </div>
        </details>
      ) : null}
    </div>
  );

  return (
    <aside className="inspector-pane" aria-label="Task Chronology Inspector">
      {/* Top Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: '10px',
          padding: '10px 14px',
          borderBlockEnd: '1px solid var(--border-subtle)',
          background: 'var(--bg-elevated)',
        }}
      >
        <div style={{ minInlineSize: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <span className={`badge ${stateBadge.cls}`}>
              <StateBadgeIcon size={11} />
              {stateBadge.label}
            </span>
            <span className="badge" style={{ textTransform: 'lowercase' }}>
              {item.kind}
            </span>
            {failureCount > 0 ? (
              <span className="badge badge-warn">Retry #{failureCount}</span>
            ) : null}
            {specResets > 0 ? (
              <span className="badge badge-purple">
                {specResets} Spec Reset{specResets === 1 ? '' : 's'}
              </span>
            ) : null}
            <code style={{ fontSize: '0.8rem', color: 'var(--text-muted)', overflowWrap: 'anywhere' }}>
              {item.item_id}
            </code>
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: '1.02rem', fontWeight: 700, lineHeight: 1.3 }}>
            {item.title}
          </h2>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexShrink: 0 }}>
          <button
            type="button"
            className="btn"
            style={{ padding: '4px 8px', fontSize: '0.8rem' }}
            onClick={() => void chronQuery.refetch()}
            title="Refresh task chronology from disk"
            aria-label="Refresh chronology"
          >
            <RefreshCw size={13} />
            {wideMode ? 'Refresh' : null}
          </button>
          {onToggleWideMode ? (
            <button
              type="button"
              className={`btn${wideMode ? ' btn-primary' : ''}`}
              style={{ padding: '4px 8px', fontSize: '0.8rem' }}
              onClick={onToggleWideMode}
              title={wideMode ? 'Return to split board + inspector view' : 'Expand inspector across full width'}
              aria-label={wideMode ? 'Split view' : 'Wide view'}
            >
              {wideMode ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              {wideMode ? 'Split View' : 'Wide'}
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn-ghost"
            style={{ padding: '4px 6px' }}
            onClick={onClose}
            aria-label="Close inspector"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Sub-Navigation Tabs */}
      <div className="inspector-subtabs" role="tablist" aria-label="Task Inspector Sections">
        <button
          type="button"
          role="tab"
          aria-selected={activeSubtab === 'timeline'}
          className="inspector-subtab"
          onClick={() => setActiveSubtab('timeline')}
        >
          <span>Turns &amp; Steps</span>
          <span className="tab-count">
            {attempts.length > 0 ? `${totalTurnsCount}` : isLiveState ? 'live' : '0'}
          </span>
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeSubtab === 'spec'}
          className="inspector-subtab"
          onClick={() => setActiveSubtab('spec')}
        >
          <span>Spec &amp; Shaping</span>
          {directions.length > 0 ? <span className="tab-count">{directions.length}</span> : null}
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeSubtab === 'context'}
          className="inspector-subtab"
          onClick={() => setActiveSubtab('context')}
        >
          <span>Context &amp; Self-Inquiry</span>
          <span className="tab-count">{contextPacks.length + allInquiries.length}</span>
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeSubtab === 'files'}
          className="inspector-subtab"
          onClick={() => setActiveSubtab('files')}
        >
          <span>Files &amp; Proofs</span>
          <span className="tab-count">{fileEntries.length + allProofs.length}</span>
        </button>
      </div>

      {/* Body Scroll Region */}
      <div
        className="scroll-pane"
        style={{
          padding: '12px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
          flex: '1 1 auto',
        }}
      >
        <ErrorBanner error={chronQuery.error} onRetry={() => void chronQuery.refetch()} />

        {/* TAB 1: TIMELINE & TURNS */}
        {activeSubtab === 'timeline' ? (
          <>
            {/* Compact Quick-Peek Spec Disclosure when turns exist so turns appear immediately above the fold */}
            {attempts.length > 0 || isLiveState ? (
              <details
                style={{
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-elevated)',
                  padding: '7px 10px',
                }}
              >
                <summary
                  style={{
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '8px',
                    fontSize: '0.84rem',
                    fontWeight: 600,
                    color: 'var(--text-secondary)',
                  }}
                >
                  <span className="clamp-2" style={{ flex: 1, minInlineSize: 0 }}>
                    <strong style={{ color: 'var(--text-primary)' }}>Task Brief: </strong>
                    {cleanSpec.replace(/\s+/g, ' ').slice(0, 140)}
                    {cleanSpec.length > 140 ? '…' : ''}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.73rem', color: 'var(--accent)', flexShrink: 0 }}>
                    Expand Spec
                  </span>
                </summary>
                <div style={{ marginBlockStart: '10px' }}>{renderSpecAndShapingContent()}</div>
              </details>
            ) : null}

            {/* Failure / Prior Retry Diagnostic Banner */}
            {showFailureTrace ? (
              <details
                open={isTerminalFailure}
                style={{
                  padding: '8px 11px',
                  borderRadius: 'var(--radius-sm)',
                  border: `1px solid ${isTerminalFailure ? 'var(--err)' : 'var(--warn)'}`,
                  background: `color-mix(in srgb, ${isTerminalFailure ? 'var(--err)' : 'var(--warn)'} 10%, var(--bg-canvas))`,
                }}
              >
                <summary
                  style={{
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontWeight: 700,
                    fontSize: '0.84rem',
                    color: isTerminalFailure ? 'var(--err)' : 'var(--warn)',
                  }}
                >
                  {isTerminalFailure ? <ShieldAlert size={14} /> : <AlertTriangle size={14} />}
                  {isTerminalFailure
                    ? 'Latest Runtime / Verification Exception Trace'
                    : 'Prior Worker Attempt Diagnostic (Auto-Retrying — Passed to Task Lead)'}
                </summary>
                <pre
                  tabIndex={0}
                  className="prompt-pre"
                  style={{
                    marginBlockStart: '7px',
                    color: isTerminalFailure ? 'var(--err)' : 'var(--text-primary)',
                  }}
                >
                  <code>{effectiveFailure}</code>
                </pre>
              </details>
            ) : null}

            {/* Attempt Filter Pills when > 1 attempt */}
            {attempts.length > 1 ? (
              <div className="pill-list" role="group" aria-label="Filter by Execution Attempt">
                <button
                  type="button"
                  className={`pill${attemptFilter === 'all' ? ' is-active' : ''}`}
                  aria-pressed={attemptFilter === 'all'}
                  onClick={() => setAttemptFilter('all')}
                >
                  All Attempts ({attempts.length})
                </button>
                {attempts.map((att) => {
                  const key = String(att.attempt_key || att.revision_index);
                  const labelSuffix = att.is_failed_retry
                    ? 'Failed Retry'
                    : att.council_review
                      ? att.council_review.approved
                        ? 'Approved'
                        : 'Changes Req'
                      : att.status;
                  return (
                    <button
                      key={key}
                      type="button"
                      className={`pill${attemptFilter === key ? ' is-active' : ''}`}
                      aria-pressed={attemptFilter === key}
                      onClick={() => setAttemptFilter(key)}
                    >
                      Attempt #{att.attempt_number} · {labelSuffix} ({att.turns.length}T)
                    </button>
                  );
                })}
              </div>
            ) : null}

            {chronQuery.isLoading ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.87rem', padding: '12px 0' }}>
                Loading attempt chronology, prompts, curated context &amp; Council reviews...
              </div>
            ) : visibleAttempts.length > 0 ? (
              <div style={{ display: 'grid', gap: '12px' }}>
                {visibleAttempts.map((att, attIdx) => {
                  const isLatestAttempt = attIdx === visibleAttempts.length - 1;
                  const review = att.council_review;
                  const reviewApproved =
                    review?.approved === true ||
                    review?.verdict === 'approve' ||
                    review?.verdict === 'APPROVED';
                  const reviewVerdictLabel = review
                    ? reviewApproved
                      ? 'APPROVED'
                      : 'CHANGES REQUESTED'
                    : '';
                  const reviewProofs = review?.proof_artifacts
                    ? normalizeProofArtifacts(review.proof_artifacts, undefined, cartridgeId)
                    : [];

                  // Synthesize live active turn fallback if worker just started before first disk turn completes
                  const activeTurnData =
                    att.active_turn ??
                    (isLatestAttempt && att.status === 'in_progress' && liveStep
                      ? {
                          turn_index: att.turns.length,
                          specialist_agent: liveStep.agentName || 'task_lead',
                          step_log: [
                            {
                              step_type: 'active',
                              specialist_agent: liveStep.agentName,
                              message: liveStep.stepSummary,
                              timestamp_utc: liveStep.timestampUtc,
                            },
                          ],
                        }
                      : null);

                  return (
                    <details
                      key={att.attempt_key || attIdx}
                      open={isLatestAttempt || visibleAttempts.length === 1}
                      style={{
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--border-strong)',
                        background: 'var(--bg-elevated)',
                        padding: '10px 12px',
                      }}
                    >
                      <summary
                        style={{
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: '8px',
                          flexWrap: 'wrap',
                          fontWeight: 700,
                        }}
                      >
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                          <span>
                            Attempt #{att.attempt_number} (Rev {att.revision_index})
                          </span>
                          {att.is_failed_retry ? (
                            <span className="badge badge-warn">Auto-Retried Failure</span>
                          ) : null}
                          <span style={{ fontWeight: 500, fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                            · {att.turns.length} completed turn{att.turns.length === 1 ? '' : 's'}
                            {activeTurnData ? ' + 1 live' : ''}
                          </span>
                        </span>

                        <span
                          className={`badge ${
                            att.status === 'approved' || reviewApproved
                              ? 'badge-ok'
                              : att.is_failed_retry || att.status === 'failed'
                                ? 'badge-warn'
                                : 'badge-accent'
                          }`}
                        >
                          {att.status}
                        </span>
                      </summary>

                      <div style={{ display: 'grid', gap: '10px', marginBlockStart: '10px' }}>
                        {(att.current_strategy || att.proof_strategy || att.submission_summary) ? (
                          <div
                            style={{
                              padding: '8px 10px',
                              borderRadius: 'var(--radius-sm)',
                              background: 'var(--bg-canvas)',
                              border: '1px solid var(--border-subtle)',
                              display: 'grid',
                              gap: '4px',
                              fontSize: '0.87rem',
                            }}
                          >
                            {att.current_strategy ? (
                              <div>
                                <strong style={{ color: 'var(--accent)' }}>Task Lead Strategy: </strong>
                                <span style={{ color: 'var(--text-primary)' }}>{att.current_strategy}</span>
                              </div>
                            ) : null}
                            {att.proof_strategy ? (
                              <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                                <strong>Proof Strategy: </strong>
                                {att.proof_strategy}
                              </div>
                            ) : null}
                            {att.submission_summary ? (
                              <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                                <strong style={{ color: 'var(--ok)' }}>Submission Summary: </strong>
                                {att.submission_summary}
                              </div>
                            ) : null}
                          </div>
                        ) : null}

                        {/* Completed Turns */}
                        {att.turns.map((turn, tIdx) => (
                          <TurnDetailCard
                            key={`${att.attempt_key}_t_${turn.turn_index ?? tIdx}`}
                            turn={turn}
                            fallbackIndex={tIdx}
                            cartridgeId={cartridgeId}
                            itemTitle={item.title}
                            isLatestTurn={
                              isLatestAttempt &&
                              !activeTurnData &&
                              tIdx === att.turns.length - 1
                            }
                            deferOffscreen={tIdx >= 2}
                            contextPacks={contextPacks}
                            fileContents={fileContents}
                            onOpenLightbox={onOpenLightbox}
                          />
                        ))}

                        {/* Live In-Flight Turn */}
                        {activeTurnData ? (
                          <TurnDetailCard
                            turn={activeTurnData}
                            fallbackIndex={att.turns.length}
                            cartridgeId={cartridgeId}
                            itemTitle={item.title}
                            isActiveTurn
                            isLatestTurn
                            contextPacks={contextPacks}
                            fileContents={fileContents}
                            onOpenLightbox={onOpenLightbox}
                          />
                        ) : att.turns.length === 0 ? (
                          <div style={{ fontSize: '0.87rem', color: 'var(--text-muted)', padding: '6px 2px' }}>
                            Task Lead is initializing Turn #1...
                          </div>
                        ) : null}

                        {/* Specialist Council Review Gate */}
                        {review ? (
                          <section
                            style={{
                              padding: '10px 12px',
                              borderRadius: 'var(--radius-sm)',
                              background: reviewApproved
                                ? 'color-mix(in srgb, var(--ok) 8%, var(--bg-canvas))'
                                : 'color-mix(in srgb, var(--warn) 8%, var(--bg-canvas))',
                              border: `1px solid color-mix(in srgb, ${reviewApproved ? 'var(--ok)' : 'var(--warn)'} 40%, var(--border-subtle))`,
                              display: 'grid',
                              gap: '8px',
                            }}
                          >
                            <div
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                gap: '8px',
                                flexWrap: 'wrap',
                              }}
                            >
                              <span
                                style={{
                                  fontWeight: 700,
                                  fontSize: '0.87rem',
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  gap: '5px',
                                }}
                              >
                                <CheckCircle2
                                  size={13}
                                  style={{ color: reviewApproved ? 'var(--ok)' : 'var(--warn)' }}
                                />
                                Specialist Council Gate Review (Attempt #{att.attempt_number})
                              </span>
                              <span className={`badge ${reviewApproved ? 'badge-ok' : 'badge-warn'}`}>
                                {reviewVerdictLabel}
                              </span>
                            </div>

                            <DeliberationDisclosure
                              deliberation={review.deliberation}
                              label="Council Review Deliberation"
                              defaultOpen
                            />

                            {review.reviewer_breakdown && review.reviewer_breakdown.length > 0 ? (
                              <div style={{ display: 'grid', gap: '6px' }}>
                                {review.reviewer_breakdown.map((rb, rbIdx) => {
                                  const rMeta = getRoleMetadata(rb.reviewer_role);
                                  return (
                                    <div
                                      key={rbIdx}
                                      style={{
                                        padding: '7px 10px',
                                        borderRadius: 'var(--radius-sm)',
                                        background: 'var(--bg-surface)',
                                        border: '1px solid var(--border-subtle)',
                                        display: 'grid',
                                        gap: '4px',
                                      }}
                                    >
                                      <div style={{ fontWeight: 700, fontSize: '0.84rem', color: rMeta.accentVar }}>
                                        {rMeta.label}
                                      </div>
                                      <div style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>
                                        <FormattedRichBlocks
                                          text={rb.raw_text || rb.points.join(' ')}
                                        />
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            ) : (review.feedback || review.summary) ? (
                              <div
                                style={{
                                  fontSize: '0.87rem',
                                  color: 'var(--text-secondary)',
                                  whiteSpace: 'pre-wrap',
                                  overflowWrap: 'anywhere',
                                }}
                              >
                                {review.feedback || review.summary}
                              </div>
                            ) : null}

                            {reviewProofs.filter((p) => !p.isAudio).length > 0 ? (
                              <div
                                style={{
                                  display: 'grid',
                                  gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
                                  gap: '6px',
                                }}
                              >
                                {reviewProofs
                                  .filter((p) => !p.isAudio)
                                  .map((pr, pIdx) => (
                                    <button
                                      key={pIdx}
                                      type="button"
                                      onClick={() =>
                                        onOpenLightbox({
                                          url: pr.url,
                                          title: `${item.title} — Council Review Proof`,
                                          subtitle: pr.filename,
                                          deliberation: review.feedback || review.summary,
                                        })
                                      }
                                      style={{
                                        padding: 0,
                                        border: '1px solid var(--border-subtle)',
                                        borderRadius: 'var(--radius-sm)',
                                        background: '#05080f',
                                        cursor: 'zoom-in',
                                        overflow: 'clip',
                                      }}
                                    >
                                      <img
                                        src={pr.url}
                                        alt={pr.filename}
                                        loading="lazy"
                                        style={{ aspectRatio: '16 / 9', inlineSize: '100%', objectFit: 'cover' }}
                                      />
                                    </button>
                                  ))}
                              </div>
                            ) : null}
                          </section>
                        ) : null}
                      </div>
                    </details>
                  );
                })}
              </div>
            ) : isLiveState ? (
              <TurnDetailCard
                turn={{
                  turn_index: 0,
                  specialist_agent: liveStep?.agentName || 'task_lead',
                  step_log: liveStep
                    ? [
                        {
                          step_type: 'active',
                          specialist_agent: liveStep.agentName,
                          message: liveStep.stepSummary,
                          timestamp_utc: liveStep.timestampUtc,
                        },
                      ]
                    : [],
                }}
                fallbackIndex={0}
                cartridgeId={cartridgeId}
                itemTitle={item.title}
                isActiveTurn
                isLatestTurn
                contextPacks={contextPacks}
                fileContents={fileContents}
                onOpenLightbox={onOpenLightbox}
              />
            ) : (
              /* Task has not started executing yet (queued / waiting on deps): show full Spec inline */
              <div style={{ display: 'grid', gap: '12px' }}>
                <div
                  style={{
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px dashed var(--border-subtle)',
                    color: 'var(--text-muted)',
                    fontSize: '0.87rem',
                  }}
                >
                  No execution turns recorded yet ({stateBadge.label}). Showing task specification &amp; Council shaping below:
                </div>
                {renderSpecAndShapingContent()}
              </div>
            )}
          </>
        ) : null}

        {/* TAB 2: SPEC & SHAPING */}
        {activeSubtab === 'spec' ? renderSpecAndShapingContent() : null}

        {/* TAB 3: CURATED CONTEXT PACKS & SELF-INQUIRY MEMORY */}
        {activeSubtab === 'context' ? (
          <div style={{ display: 'grid', gap: '14px' }}>
            <section style={{ display: 'grid', gap: '8px' }}>
              <div
                style={{
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  color: 'var(--warn)',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '5px',
                }}
              >
                <HelpCircle size={13} /> Agent Self-Inquiry Q&amp;A Memory ({allInquiries.length})
              </div>
              {allInquiries.length === 0 ? (
                <div
                  style={{
                    padding: '12px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px dashed var(--border-subtle)',
                    color: 'var(--text-muted)',
                    fontSize: '0.87rem',
                  }}
                >
                  No self-inquiry questions were raised on this task.
                </div>
              ) : (
                allInquiries.map((qa, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '9px 11px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-canvas)',
                      border: '1px solid var(--border-subtle)',
                      borderInlineStart: '3px solid var(--warn)',
                      display: 'grid',
                      gap: '5px',
                      fontSize: '0.87rem',
                    }}
                  >
                    <DeliberationDisclosure deliberation={qa.deliberation} label="Self-Inquiry Rationale" />
                    <div style={{ fontWeight: 700, color: 'var(--warn)' }}>
                      Q ({formatSpecialistLabel(qa.agent_role)}): {qa.question}
                    </div>
                    <div style={{ whiteSpace: 'pre-wrap' }}>
                      <strong style={{ color: 'var(--ok)' }}>Self-Answer: </strong>
                      {qa.answer}
                    </div>
                    {qa.researched_chunk_ids && qa.researched_chunk_ids.length > 0 ? (
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        Researched chunks:{' '}
                        <code style={{ color: 'var(--accent)' }}>{qa.researched_chunk_ids.join(', ')}</code>
                      </div>
                    ) : null}
                  </div>
                ))
              )}
            </section>

            <section style={{ display: 'grid', gap: '8px' }}>
              <div
                style={{
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  color: 'var(--purple)',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '5px',
                }}
              >
                <Layers size={13} /> Curated Studio Context Packs ({contextPacks.length})
              </div>
              {contextPacks.length === 0 ? (
                <div
                  style={{
                    padding: '12px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px dashed var(--border-subtle)',
                    color: 'var(--text-muted)',
                    fontSize: '0.87rem',
                  }}
                >
                  No standalone context pack snapshots persisted for this task yet (per-turn curated context is also available inside each Turn card).
                </div>
              ) : (
                contextPacks.map((cp, idx) => (
                  <details
                    key={cp.filename || idx}
                    open={idx === 0}
                    style={{
                      padding: '9px 11px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-canvas)',
                      border: '1px solid var(--border-subtle)',
                      borderInlineStart: '3px solid var(--purple)',
                    }}
                  >
                    <summary
                      style={{
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '8px',
                        fontWeight: 700,
                        fontSize: '0.87rem',
                      }}
                    >
                      <span style={{ color: 'var(--purple)' }}>
                        Role: {cp.target_agent_role} (Rev {cp.revision ?? 0})
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {(cp.selected_chunk_ids ?? []).length} / {cp.total_indexed_chunks ?? 0} chunks
                      </span>
                    </summary>
                    <div style={{ display: 'grid', gap: '7px', marginBlockStart: '8px', fontSize: '0.87rem' }}>
                      <DeliberationDisclosure
                        deliberation={cp.deliberation}
                        label="Context Curator Deliberation"
                        defaultOpen
                      />
                      {cp.role_tailored_synthesis ? (
                        <div
                          style={{
                            padding: '7px 9px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-surface)',
                            border: '1px solid var(--border-subtle)',
                            whiteSpace: 'pre-wrap',
                          }}
                        >
                          <strong style={{ color: 'var(--accent)' }}>Role-Tailored Synthesis: </strong>
                          {cp.role_tailored_synthesis}
                        </div>
                      ) : null}
                      {cp.search_queries_used && cp.search_queries_used.length > 0 ? (
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                          Search queries: <code>{cp.search_queries_used.join(' · ')}</code>
                        </div>
                      ) : null}
                      {cp.selected_chunk_ids && cp.selected_chunk_ids.length > 0 ? (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                          {cp.selected_chunk_ids.map((cid) => (
                            <span
                              key={cid}
                              className="badge"
                              style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}
                            >
                              {cid}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </details>
                ))
              )}
            </section>
          </div>
        ) : null}

        {/* TAB 4: TOUCHED FILES & PROOF ARTIFACTS */}
        {activeSubtab === 'files' ? (
          <div style={{ display: 'grid', gap: '14px' }}>
            {allProofs.length > 0 ? (
              <section style={{ display: 'grid', gap: '7px' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                  Verified Proof Artifacts ({allProofs.length})
                </div>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
                    gap: '8px',
                  }}
                >
                  {allProofs.map((pr, idx) =>
                    pr.isAudio ? (
                      <div
                        key={idx}
                        style={{
                          gridColumn: '1 / -1',
                          padding: '6px 8px',
                          borderRadius: 'var(--radius-sm)',
                          background: 'var(--bg-elevated)',
                          border: '1px solid var(--border-subtle)',
                        }}
                      >
                        <div style={{ fontSize: '0.8rem', marginBlockEnd: '4px', fontFamily: 'var(--font-mono)' }}>
                          {pr.filename}
                        </div>
                        <audio controls preload="metadata" src={pr.url} style={{ inlineSize: '100%' }} />
                      </div>
                    ) : (
                      <button
                        key={idx}
                        type="button"
                        onClick={() =>
                          onOpenLightbox({
                            url: pr.url,
                            title: `${item.title} — ${pr.filename}`,
                            subtitle: pr.path,
                            deliberation: item.deliberation,
                          })
                        }
                        style={{
                          padding: 0,
                          border: '1px solid var(--border-subtle)',
                          borderRadius: 'var(--radius-sm)',
                          background: '#05080f',
                          cursor: 'zoom-in',
                          overflow: 'clip',
                          textAlign: 'start',
                        }}
                      >
                        <img
                          src={pr.url}
                          alt={pr.filename}
                          loading="lazy"
                          style={{ aspectRatio: '16 / 9', inlineSize: '100%', objectFit: 'cover' }}
                        />
                        <div
                          style={{
                            padding: '4px 6px',
                            fontSize: '0.73rem',
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--text-secondary)',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                          }}
                        >
                          <ImageIcon size={10} /> {pr.filename}
                        </div>
                      </button>
                    ),
                  )}
                </div>
              </section>
            ) : null}

            <section style={{ display: 'grid', gap: '7px' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                Touched &amp; Target Module Files on Disk ({fileEntries.length})
              </div>
              {fileEntries.length === 0 ? (
                <div
                  style={{
                    padding: '12px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px dashed var(--border-subtle)',
                    color: 'var(--text-muted)',
                    fontSize: '0.87rem',
                  }}
                >
                  No disk files captured for this task yet.
                </div>
              ) : (
                fileEntries.map(([pathKey, rawEntry]) => {
                  const isStr = typeof rawEntry === 'string';
                  const isBinary = !isStr && rawEntry.is_binary;
                  const contentStr = isStr ? rawEntry : rawEntry.content || '';
                  const lineCount = !isStr && rawEntry.line_count ? rawEntry.line_count : contentStr.split('\n').length;
                  const sizeBytes = !isStr ? rawEntry.size_bytes : contentStr.length;

                  return (
                    <details
                      key={pathKey}
                      style={{
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-subtle)',
                        background: 'var(--bg-canvas)',
                        padding: '7px 10px',
                      }}
                    >
                      <summary
                        style={{
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: '8px',
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.84rem',
                          color: 'var(--accent)',
                          fontWeight: 600,
                        }}
                      >
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', minInlineSize: 0, overflowWrap: 'anywhere' }}>
                          <FileCode size={12} style={{ flexShrink: 0 }} /> {pathKey}
                        </span>
                        <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)', flexShrink: 0 }}>
                          {isBinary
                            ? `binary${sizeBytes ? ` · ${(sizeBytes / 1024).toFixed(1)} KB` : ''}`
                            : `${lineCount} lines${sizeBytes ? ` · ${(sizeBytes / 1024).toFixed(1)} KB` : ''}`}
                        </span>
                      </summary>
                      {isBinary && !isStr && rawEntry.url ? (
                        <div style={{ marginBlockStart: '8px' }}>
                          {/\.(wav|mp3|ogg)(\?|$)/i.test(pathKey) ? (
                            <audio controls preload="metadata" src={rawEntry.url} style={{ inlineSize: '100%' }} />
                          ) : (
                            <img
                              src={rawEntry.url}
                              alt={pathKey}
                              loading="lazy"
                              style={{ maxBlockSize: '260px', borderRadius: 'var(--radius-sm)' }}
                            />
                          )}
                        </div>
                      ) : (
                        <pre tabIndex={0} className="prompt-pre" style={{ marginBlockStart: '8px' }}>
                          <code>{contentStr}</code>
                        </pre>
                      )}
                    </details>
                  );
                })
              )}
            </section>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
