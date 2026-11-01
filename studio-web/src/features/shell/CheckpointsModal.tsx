import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, Gamepad2, Star } from 'lucide-react';
import { studioApi } from '../../api/client';
import { ModalDialog } from '../../components/ModalDialog';
import { ErrorBanner } from '../../components/ErrorBanner';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';

export interface CheckpointsModalProps {
  open: boolean;
  onClose: () => void;
  selectedCartridgeId: string;
  onLaunchCheckpoint: (checkpointId: string) => void;
}

export function CheckpointsModal({
  open,
  onClose,
  selectedCartridgeId,
  onLaunchCheckpoint,
}: CheckpointsModalProps) {
  const queryClient = useQueryClient();
  const [scopeToSelected, setScopeToSelected] = useState(true);

  const checkpointsQuery = useQuery({
    queryKey: ['checkpoints'],
    queryFn: () => studioApi.getCheckpoints(),
    enabled: open,
  });

  const activateMut = useMutation({
    mutationFn: (checkpointId: string) => studioApi.activateCheckpoint(checkpointId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['checkpoints'] });
      void queryClient.invalidateQueries({ queryKey: ['cartridges'] });
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
    },
  });

  const allCheckpoints = checkpointsQuery.data?.checkpoints ?? [];
  const filtered =
    scopeToSelected && selectedCartridgeId && selectedCartridgeId !== 'active'
      ? allCheckpoints.filter((cp) => cp.candidate_id === selectedCartridgeId)
      : allCheckpoints;

  return (
    <ModalDialog
      open={open}
      onClose={onClose}
      wide
      title="Verified Cartridge Checkpoints"
      subtitle="Immutable slice & milestone snapshots available for instant rollback or native Godot launch."
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
          <div className="pill-list">
            <button
              type="button"
              className={`pill${scopeToSelected ? ' is-active' : ''}`}
              onClick={() => setScopeToSelected(true)}
            >
              Selected Game ({
                selectedCartridgeId && selectedCartridgeId !== 'active'
                  ? allCheckpoints.filter((c) => c.candidate_id === selectedCartridgeId).length
                  : allCheckpoints.length
              })
            </button>
            <button
              type="button"
              className={`pill${!scopeToSelected ? ' is-active' : ''}`}
              onClick={() => setScopeToSelected(false)}
            >
              All Games ({allCheckpoints.length})
            </button>
          </div>
          <span style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
            Showing {filtered.length} checkpoint(s)
          </span>
        </div>

        <ErrorBanner error={checkpointsQuery.error || activateMut.error} onRetry={() => void checkpointsQuery.refetch()} />

        {checkpointsQuery.isLoading ? (
          <div style={{ padding: '24px', color: 'var(--text-muted)' }}>Loading verified checkpoints...</div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: '24px', color: 'var(--text-muted)' }}>
            No verified checkpoints recorded for this filter yet.
          </div>
        ) : (
          <div style={{ display: 'grid', gap: '8px' }}>
            {filtered.map((cp, idx) => (
              <article
                key={cp.checkpoint_id}
                className={idx >= 12 ? 'deferred-compact-row' : undefined}
                style={{
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-md)',
                  border: `1px solid ${cp.is_active ? 'var(--ok)' : 'var(--border-subtle)'}`,
                  background: 'var(--bg-elevated)',
                  display: 'grid',
                  gap: '6px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.91rem' }}>
                      {cp.track_title || cp.requirement_id}
                    </span>
                    <code style={{ fontSize: '0.8rem', color: 'var(--accent)' }}>{cp.checkpoint_id}</code>
                    {cp.is_active ? (
                      <span className="badge badge-ok">
                        <CheckCircle2 size={11} /> Active Playable
                      </span>
                    ) : null}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      {cp.created_at_utc ? new Date(cp.created_at_utc).toLocaleString() : ''}
                    </span>
                    {!cp.is_active ? (
                      <button
                        type="button"
                        className="btn"
                        disabled={activateMut.isPending}
                        onClick={() => activateMut.mutate(cp.checkpoint_id)}
                      >
                        <Star size={12} /> Set Active
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="btn btn-success"
                      onClick={() => onLaunchCheckpoint(cp.checkpoint_id)}
                    >
                      <Gamepad2 size={12} /> Launch
                    </button>
                  </div>
                </div>

                <DeliberationDisclosure deliberation={cp.deliberation} label="Checkpoint Provenance" />

                {cp.verified_tracks.length > 0 ? (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    <strong>{cp.verified_tracks.length} verified track(s):</strong>{' '}
                    {cp.verified_tracks.slice(0, 8).join(', ')}
                    {cp.verified_tracks.length > 8 ? ` +${cp.verified_tracks.length - 8} more` : ''}
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        )}
      </div>
    </ModalDialog>
  );
}
