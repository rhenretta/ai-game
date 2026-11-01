import { type FormEvent, useState } from 'react';
import { ChevronDown, ChevronRight, Layers, Send } from 'lucide-react';
import type { ProductionBoardItem, VerticalSliceSpec } from '../../api/types';
import { filterBoardItemsBySlice } from '../../domain/slices';

export interface SliceSubbarProps {
  slices: VerticalSliceSpec[];
  selectedSlice: VerticalSliceSpec | null;
  selectedSliceId: string | null;
  onSelectSliceId: (sliceId: string) => void;
  allBoardItems: ProductionBoardItem[];
  isBuildRunning: boolean;
  steeringNotice?: string | null;
  onSteerCouncil: (feedback: string) => Promise<void>;
  steeringBusy: boolean;
}

export function SliceSubbar({
  slices,
  selectedSlice,
  selectedSliceId,
  onSelectSliceId,
  allBoardItems,
  isBuildRunning,
  steeringNotice,
  onSteerCouncil,
  steeringBusy,
}: SliceSubbarProps) {
  const [showBrief, setShowBrief] = useState(false);
  const [steeringNote, setSteeringNote] = useState('');

  if (slices.length === 0) return null;

  const isAllMode = selectedSliceId === 'all' || selectedSlice === null;
  const activeItems = allBoardItems.filter((i) => i.state !== 'cancelled');
  const totalApproved = activeItems.filter((i) => i.state === 'approved').length;

  const handleSubmitSteer = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = steeringNote.trim();
    if (!trimmed) return;
    await onSteerCouncil(trimmed);
    setSteeringNote('');
  };

  return (
    <div className="slice-subbar">
      <div className="slice-subbar-row">
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            fontSize: '0.8rem',
            fontWeight: 700,
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
          }}
        >
          <Layers size={13} /> Slices
        </span>

        <div className="pill-list" role="group" aria-label="Vertical Slice Filter">
          {slices.map((s) => {
            const itemsForSlice = filterBoardItemsBySlice(allBoardItems, s, slices.length);
            const approvedCnt = itemsForSlice.filter((i) => i.state === 'approved').length;
            const isActive = !isAllMode && selectedSlice?.slice_id === s.slice_id;
            return (
              <button
                key={s.slice_id}
                type="button"
                className={`pill${isActive ? ' is-active' : ''}`}
                aria-pressed={isActive}
                onClick={() => onSelectSliceId(s.slice_id)}
                title={s.title}
              >
                <span>Slice #{s.slice_index}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem', opacity: 0.85 }}>
                  {approvedCnt}/{itemsForSlice.length}
                </span>
              </button>
            );
          })}
          {slices.length > 1 ? (
            <button
              type="button"
              className={`pill${isAllMode ? ' is-active' : ''}`}
              aria-pressed={isAllMode}
              onClick={() => onSelectSliceId('all')}
            >
              <span>All Slices</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem', opacity: 0.85 }}>
                {totalApproved}/{activeItems.length}
              </span>
            </button>
          ) : null}
        </div>

        {selectedSlice ? (
          <button
            type="button"
            className="btn btn-ghost"
            style={{ fontSize: '0.84rem', paddingBlock: '3px', paddingInline: '7px' }}
            onClick={() => setShowBrief((v) => !v)}
          >
            {showBrief ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{selectedSlice.title}</span>
            <span style={{ color: 'var(--text-muted)' }}>· Brief</span>
          </button>
        ) : null}

        <form
          onSubmit={(e) => void handleSubmitSteer(e)}
          style={{
            marginInlineStart: 'auto',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            flex: '1 1 280px',
            maxInlineSize: '460px',
          }}
        >
          {steeringNotice ? (
            <span className="badge badge-ok" style={{ whiteSpace: 'nowrap', textTransform: 'none' }}>
              {steeringNotice}
            </span>
          ) : null}
          <input
            type="text"
            className="input-control"
            style={{ flex: '1 1 auto' }}
            placeholder={
              isBuildRunning
                ? 'Inject live steering into active Council...'
                : 'Send direction & resume Studio build...'
            }
            value={steeringNote}
            onChange={(e) => setSteeringNote(e.target.value)}
            disabled={steeringBusy}
          />
          <button
            type="submit"
            className="btn btn-primary"
            disabled={steeringBusy || !steeringNote.trim()}
          >
            <Send size={12} />
            {steeringBusy ? 'Sending...' : isBuildRunning ? 'Steer' : 'Steer & Resume'}
          </button>
        </form>
      </div>

      {showBrief && selectedSlice ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '10px',
            padding: '8px 10px',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.87rem',
          }}
        >
          {selectedSlice.slice_objective ? (
            <div>
              <strong style={{ color: 'var(--accent)', fontSize: '0.8rem', textTransform: 'uppercase' }}>
                Slice Objective
              </strong>
              <div style={{ color: 'var(--text-secondary)', marginBlockStart: '2px' }}>
                {selectedSlice.slice_objective}
              </div>
            </div>
          ) : null}
          {selectedSlice.player_acclimation_and_mechanics_focus ? (
            <div>
              <strong style={{ color: 'var(--ok)', fontSize: '0.8rem', textTransform: 'uppercase' }}>
                Onboarding &amp; Mechanics Focus
              </strong>
              <div style={{ color: 'var(--text-secondary)', marginBlockStart: '2px' }}>
                {selectedSlice.player_acclimation_and_mechanics_focus}
              </div>
            </div>
          ) : null}
          {selectedSlice.scope_boundaries ? (
            <div>
              <strong style={{ color: 'var(--warn)', fontSize: '0.8rem', textTransform: 'uppercase' }}>
                Scope Boundaries
              </strong>
              <div style={{ color: 'var(--text-secondary)', marginBlockStart: '2px' }}>
                {selectedSlice.scope_boundaries}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
