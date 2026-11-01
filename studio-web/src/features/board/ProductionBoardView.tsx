import { useMemo, useState } from 'react';
import { Radio, Search } from 'lucide-react';
import type { ProductionBoardItem } from '../../api/types';
import type { LiveItemStep } from '../../realtime/useSwarmSocket';
import type { LiveStudioActivity } from '../../domain/liveActivity';
import { WorkItemCard } from './WorkItemCard';
import { TaskChronologyDrawer } from './TaskChronologyDrawer';
import {
  isWorkingBoardItem,
  sortBoardItemsByPriority,
} from '../../domain/boardCards';
import type { LightboxMedia } from '../../components/MediaLightbox';

export type BoardStatusFilter =
  | 'all'
  | 'active_now'
  | 'awaiting_review'
  | 'changes_requested'
  | 'ready'
  | 'blocked'
  | 'approved'
  | 'attention';

export interface ProductionBoardViewProps {
  items: ProductionBoardItem[];
  cartridgeId: string;
  isBuildRunning?: boolean;
  liveStepsByItem: Record<string, LiveItemStep>;
  liveActivity?: LiveStudioActivity;
  onOpenLivePulse?: () => void;
  onOpenLightbox: (media: LightboxMedia) => void;
}

export function ProductionBoardView({
  items,
  cartridgeId,
  isBuildRunning = false,
  liveStepsByItem,
  liveActivity,
  onOpenLivePulse,
  onOpenLightbox,
}: ProductionBoardViewProps) {
  const [statusFilter, setStatusFilter] = useState<BoardStatusFilter>('all');
  const [kindFilter, setKindFilter] = useState<string>('all');
  const [query, setQuery] = useState('');
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);
  const [wideInspector, setWideInspector] = useState(false);

  const counts = useMemo(() => {
    let activeNow = 0;
    let review = 0;
    let revising = 0;
    let ready = 0;
    let blocked = 0;
    let approved = 0;
    let attention = 0;
    const kinds = new Set<string>();

    for (const it of items) {
      if (it.kind) kinds.add(it.kind);
      // "Working" is the overall umbrella status for any started, not-yet-approved task
      // (including in_progress, waiting_for_lock, awaiting_review, changes_requested/Revising, etc.)
      if (isWorkingBoardItem(it)) {
        activeNow += 1;
      }
      if (it.state === 'awaiting_review') {
        review += 1;
      } else if (it.state === 'changes_requested') {
        revising += 1;
      } else if (it.state === 'ready') {
        ready += 1;
      } else if (it.state === 'blocked') {
        blocked += 1;
      } else if (it.state === 'approved') {
        approved += 1;
      } else if (it.state === 'failed' || it.state === 'needs_decision') {
        // While the Studio build is actively running, failed/needs_decision items are being
        // autonomously re-queued or re-shaped by the Council; only flag Needs Attention if halted.
        if (isBuildRunning) revising += 1;
        else attention += 1;
      }
    }

    return {
      total: items.length,
      activeNow,
      review,
      revising,
      ready,
      blocked,
      approved,
      attention,
      kinds: Array.from(kinds).sort(),
    };
  }, [items, isBuildRunning]);

  const filteredItems = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matched = items.filter((it) => {
      if (statusFilter === 'active_now' && !isWorkingBoardItem(it)) return false;
      if (statusFilter === 'awaiting_review' && it.state !== 'awaiting_review') return false;
      if (statusFilter === 'changes_requested') {
        const isSelfCorrecting =
          isBuildRunning && (it.state === 'failed' || it.state === 'needs_decision');
        if (it.state !== 'changes_requested' && !isSelfCorrecting) return false;
      }
      if (statusFilter === 'ready' && it.state !== 'ready') return false;
      if (statusFilter === 'blocked' && it.state !== 'blocked') return false;
      if (statusFilter === 'approved' && it.state !== 'approved') return false;
      if (statusFilter === 'attention') {
        if (isBuildRunning) return false;
        if (it.state !== 'failed' && it.state !== 'needs_decision') return false;
      }
      if (kindFilter !== 'all' && it.kind !== kindFilter) return false;
      if (q) {
        const hay = `${it.title} ${it.item_id} ${it.spec} ${it.target_modules.join(' ')}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
    return sortBoardItemsByPriority(matched, isBuildRunning);
  }, [items, statusFilter, kindFilter, query, isBuildRunning]);

  const selectedItem = useMemo(
    () => (selectedItemId ? items.find((i) => i.item_id === selectedItemId) ?? null : null),
    [items, selectedItemId],
  );

  const filterPills: Array<{ id: BoardStatusFilter; label: string; count: number }> = [
    { id: 'all', label: 'All', count: counts.total },
    { id: 'active_now', label: 'Working', count: counts.activeNow },
    { id: 'awaiting_review', label: 'In Review', count: counts.review },
    { id: 'changes_requested', label: 'Revising', count: counts.revising },
    { id: 'ready', label: 'Queued', count: counts.ready },
    { id: 'blocked', label: 'Waiting on Deps', count: counts.blocked },
    { id: 'attention', label: 'Needs Attention', count: counts.attention },
    { id: 'approved', label: 'Approved', count: counts.approved },
  ];

  return (
    <div
      className={`board-split-layout${selectedItem ? ' has-inspector' : ''}${selectedItem && wideInspector ? ' is-wide-inspector' : ''}`}
    >
      <div
        className="scroll-pane board-cards-pane"
        style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '12px' }}
      >
        {/* Single clean Board Filter Toolbar (no duplicate bars!) */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '8px',
          }}
        >
          <div className="pill-list" role="group" aria-label="Filter Production Board by Status">
            {filterPills
              .filter((p) => p.id === 'all' || p.id === 'approved' || p.count > 0 || statusFilter === p.id)
              .map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className={`pill${statusFilter === p.id ? ' is-active' : ''}`}
                  aria-pressed={statusFilter === p.id}
                  onClick={() => setStatusFilter(p.id)}
                >
                  <span>{p.label}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem', opacity: 0.85 }}>
                    {p.count}
                  </span>
                </button>
              ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            {counts.kinds.length > 1 ? (
              <select
                aria-label="Filter by Task Kind"
                className="select-control"
                value={kindFilter}
                onChange={(e) => setKindFilter(e.target.value)}
              >
                <option value="all">All Kinds ({counts.kinds.length})</option>
                {counts.kinds.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            ) : null}

            <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
              <Search
                size={13}
                style={{ position: 'absolute', insetInlineStart: '8px', color: 'var(--text-muted)', pointerEvents: 'none' }}
              />
              <input
                type="search"
                className="input-control"
                style={{ paddingInlineStart: '26px', inlineSize: '210px' }}
                placeholder="Search title, module, spec..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
          </div>
        </div>

        {isBuildRunning && counts.total > 0 && counts.approved === counts.total && liveActivity?.isActive ? (
          <div
            className="post-board-callout"
            data-tone={liveActivity.phase === 'qa_repair' ? 'warn' : undefined}
          >
            <div style={{ display: 'grid', gap: '4px', minInlineSize: 0, flex: '1 1 360px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                <span className="badge badge-ok">
                  Board Complete ({counts.approved}/{counts.total} Approved)
                </span>
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
              <div style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                {liveActivity.headline}
              </div>
              {liveActivity.detail ? (
                <div style={{ fontSize: '0.79rem', color: 'var(--text-secondary)' }}>
                  {liveActivity.detail}
                </div>
              ) : null}
            </div>
            {onOpenLivePulse ? (
              <button type="button" className="btn" onClick={onOpenLivePulse}>
                <Radio size={12} style={{ color: 'var(--accent)' }} aria-hidden="true" />
                <span>Inspect Live Stream</span>
              </button>
            ) : null}
          </div>
        ) : null}

        {filteredItems.length === 0 ? (
          <div
            style={{
              padding: '32px',
              textAlign: 'center',
              borderRadius: 'var(--radius-md)',
              border: '1px dashed var(--border-subtle)',
              color: 'var(--text-muted)',
            }}
          >
            {items.length === 0
              ? 'No work items posted to this slice yet. The Specialist Council is brainstorming or awaiting greenlight.'
              : 'No work items match the current board filter.'}
          </div>
        ) : (
          <div className="card-grid">
            {filteredItems.map((it) => (
              <WorkItemCard
                key={it.item_id}
                item={it}
                cartridgeId={cartridgeId}
                isBuildRunning={isBuildRunning}
                selected={selectedItem?.item_id === it.item_id}
                liveStep={liveStepsByItem[it.item_id]}
                onSelect={(id) => setSelectedItemId((prev) => (prev === id ? null : id))}
                onOpenLightbox={onOpenLightbox}
              />
            ))}
          </div>
        )}
      </div>

      {selectedItem ? (
        <TaskChronologyDrawer
          item={selectedItem}
          cartridgeId={cartridgeId}
          isBuildRunning={isBuildRunning}
          liveStep={liveStepsByItem[selectedItem.item_id]}
          wideMode={wideInspector}
          onToggleWideMode={() => setWideInspector((v) => !v)}
          onClose={() => {
            setSelectedItemId(null);
            setWideInspector(false);
          }}
          onOpenLightbox={onOpenLightbox}
        />
      ) : null}
    </div>
  );
}
