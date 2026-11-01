import { useMemo, useState } from 'react';
import { Cpu, Radio } from 'lucide-react';
import type { StudioStatusResponse, SwarmStreamEvent } from '../../api/types';
import {
  formatTokenCount,
  summarizeSwarmEvent,
  type LiveStudioActivity,
} from '../../domain/liveActivity';
import { ModalDialog } from '../../components/ModalDialog';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';

export interface LivePulseModalProps {
  open: boolean;
  onClose: () => void;
  wsConnected: boolean;
  lastEventAt: string | null;
  recentEvents: SwarmStreamEvent[];
  studioStatus?: StudioStatusResponse;
  liveActivity?: LiveStudioActivity;
}

type StreamCategoryFilter = 'all' | 'stage' | 'board' | 'ai';

function toneClass(tone: 'accent' | 'ok' | 'warn' | 'danger' | 'neutral'): string {
  switch (tone) {
    case 'ok':
      return 'badge badge-ok';
    case 'warn':
      return 'badge badge-warn';
    case 'danger':
      return 'badge badge-danger';
    case 'accent':
      return 'badge badge-accent';
    default:
      return 'badge';
  }
}

export function LivePulseModal({
  open,
  onClose,
  wsConnected,
  lastEventAt,
  recentEvents,
  studioStatus,
  liveActivity,
}: LivePulseModalProps) {
  const [filter, setFilter] = useState<StreamCategoryFilter>('all');

  const annotated = useMemo(
    () => recentEvents.map((ev) => ({ ev, summary: summarizeSwarmEvent(ev) })),
    [recentEvents],
  );

  const counts = useMemo(() => {
    let stage = 0;
    let board = 0;
    let ai = 0;
    for (const row of annotated) {
      if (row.summary.category === 'stage') stage += 1;
      else if (row.summary.category === 'board') board += 1;
      else if (row.summary.category === 'ai') ai += 1;
    }
    return { all: annotated.length, stage, board, ai };
  }, [annotated]);

  const filtered = useMemo(
    () => (filter === 'all' ? annotated : annotated.filter((r) => r.summary.category === filter)),
    [annotated, filter],
  );

  return (
    <ModalDialog
      open={open}
      onClose={onClose}
      wide
      title="Live Swarm WebSocket Stream"
      subtitle={
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <Radio
            size={12}
            style={{ color: wsConnected ? 'var(--ok)' : 'var(--warn)' }}
            aria-hidden="true"
          />
          <span>{wsConnected ? 'Connected to /ws/swarm (Delta Streaming)' : 'Reconnecting...'}</span>
          {lastEventAt ? <span>· Last event {new Date(lastEventAt).toLocaleTimeString()}</span> : null}
          {studioStatus ? <span>· DB Events: {studioStatus.db_event_count.toLocaleString()}</span> : null}
        </span>
      }
    >
      <div style={{ display: 'grid', gap: '10px' }}>
        {liveActivity?.isActive ? (
          <div
            className="post-board-callout"
            data-tone={liveActivity.phase === 'qa_repair' ? 'warn' : undefined}
          >
            <div style={{ display: 'grid', gap: '4px', minInlineSize: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                <span className="live-pulse-dot" aria-hidden="true" />
                <span
                  className={
                    liveActivity.phase === 'qa_repair' ? 'badge badge-warn' : 'badge badge-accent'
                  }
                >
                  {liveActivity.stageTitle}
                </span>
                <span className="badge" style={{ textTransform: 'none', fontWeight: 600 }}>
                  {liveActivity.activeAgentLabel}
                </span>
              </div>
              <div style={{ fontSize: '0.86rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                {liveActivity.headline}
              </div>
              {liveActivity.detail ? (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {liveActivity.detail}
                </div>
              ) : null}
            </div>
            {liveActivity.latestAiCall ? (
              <span className="live-ai-heartbeat-pill">
                <Cpu size={12} style={{ color: 'var(--accent)' }} aria-hidden="true" />
                <span>
                  {liveActivity.latestAiCall.callNumber
                    ? `AI #${liveActivity.latestAiCall.callNumber}`
                    : 'AI Call'}{' '}
                  · {liveActivity.latestAiCall.stepLabel} ·{' '}
                  {formatTokenCount(liveActivity.latestAiCall.totalTokens)}
                </span>
              </span>
            ) : null}
          </div>
        ) : null}

        {annotated.length > 0 ? (
          <div className="pill-list" role="group" aria-label="Filter Live Swarm Events">
            {(
              [
                { id: 'all', label: 'All Events', count: counts.all },
                { id: 'stage', label: 'Stages & QA Gates', count: counts.stage },
                { id: 'board', label: 'Board & Council', count: counts.board },
                { id: 'ai', label: 'AI Gateway Calls', count: counts.ai },
              ] as const
            ).map((pill) => (
              <button
                key={pill.id}
                type="button"
                className={`pill${filter === pill.id ? ' is-active' : ''}`}
                aria-pressed={filter === pill.id}
                onClick={() => setFilter(pill.id)}
              >
                <span>{pill.label}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', opacity: 0.85 }}>
                  {pill.count}
                </span>
              </button>
            ))}
          </div>
        ) : null}

        {filtered.length === 0 ? (
          <div style={{ padding: '20px', color: 'var(--text-muted)' }}>
            {recentEvents.length === 0
              ? 'Connected and listening for live build, Council, or Playtester delta events. (No background 143 MB history polling.)'
              : 'No events match the selected stream filter.'}
          </div>
        ) : (
          <div style={{ display: 'grid', gap: '8px' }}>
            {filtered.map(({ ev, summary }, idx) => (
              <article
                key={ev.event_id || `${ev.timestamp_utc}-${idx}`}
                className={idx >= 12 ? 'deferred-compact-row' : undefined}
                style={{
                  padding: '9px 11px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-elevated)',
                  display: 'grid',
                  gap: '5px',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: '8px',
                    flexWrap: 'wrap',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                    {summary.badgeText ? (
                      <span className={toneClass(summary.badgeTone)}>{summary.badgeText}</span>
                    ) : null}
                    <span className="badge" style={{ textTransform: 'none', fontWeight: 600 }}>
                      {summary.agentLabel}
                    </span>
                    <code style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                      {ev.event_type}
                    </code>
                  </div>
                  <span
                    style={{
                      fontSize: '0.78rem',
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    {ev.timestamp_utc ? new Date(ev.timestamp_utc).toLocaleTimeString() : ''}
                  </span>
                </div>

                <div style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {summary.title}
                </div>

                {summary.subtitle && summary.subtitle !== summary.title ? (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    {summary.subtitle}
                  </div>
                ) : null}

                <DeliberationDisclosure
                  deliberation={ev.deliberation}
                  label="Event Rationale"
                  defaultOpen={idx < 3 && summary.category !== 'ai'}
                />
              </article>
            ))}
          </div>
        )}
      </div>
    </ModalDialog>
  );
}
