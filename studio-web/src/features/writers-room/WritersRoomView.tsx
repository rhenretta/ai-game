import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, Check, ChevronsDownUp, ChevronsUpDown, ChevronRight, Radio, Search, Send } from 'lucide-react';
import { resolveMediaUrl } from '../../api/client';
import type { WritersRoomUtterance } from '../../api/types';
import type { LiveStudioActivity } from '../../domain/liveActivity';
import { getRoleMetadata, POD_OPTIONS, type SpecialistPod } from '../../domain/roles';
import {
  formatWritersRoomUtterance,
  type BadgeTone,
  type StructuredTextBlock,
} from '../../domain/utterances';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';
import type { LightboxMedia } from '../../components/MediaLightbox';
import { StructuredBlocksView } from '../../components/RichTextBlocks';

export interface WritersRoomViewProps {
  utterances: WritersRoomUtterance[];
  cartridgeId: string;
  isBuildRunning: boolean;
  awaitingGreenlight: boolean;
  liveActivity?: LiveStudioActivity;
  onOpenLivePulse?: () => void;
  onSteer: (feedback: string) => Promise<void>;
  onGreenlight: (feedback: string) => Promise<void>;
  busy: boolean;
  onOpenLightbox: (media: LightboxMedia) => void;
}

function badgeToneClass(tone: BadgeTone): string {
  switch (tone) {
    case 'ok':
      return 'badge badge-ok';
    case 'accent':
      return 'badge badge-accent';
    case 'warn':
      return 'badge badge-warn';
    default:
      return 'badge';
  }
}

function StructuredProse({ blocks }: { blocks: StructuredTextBlock[] }) {
  return (
    <StructuredBlocksView
      blocks={blocks}
      className="utterance-prose"
      listClassName="utterance-list"
    />
  );
}

export function WritersRoomView({
  utterances,
  cartridgeId,
  isBuildRunning,
  awaitingGreenlight,
  liveActivity,
  onOpenLivePulse,
  onSteer,
  onGreenlight,
  busy,
  onOpenLightbox,
}: WritersRoomViewProps) {
  const [pod, setPod] = useState<SpecialistPod>('all');
  const [query, setQuery] = useState('');
  const [composerText, setComposerText] = useState('');
  const [expandAll, setExpandAll] = useState(false);
  const [stickToBottom, setStickToBottom] = useState(true);
  const feedRef = useRef<HTMLDivElement | null>(null);
  const prevCountRef = useRef<number>(utterances.length);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return utterances.filter((u) => {
      const meta = getRoleMetadata(u.speaker_role, u.speaker_display_name);
      if (pod !== 'all' && meta.pod !== pod) return false;
      if (q) {
        const hay = `${meta.label} ${u.spoken_dialogue ?? ''} ${u.message ?? ''} ${u.deliberation ?? ''}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [utterances, pod, query]);

  const scrollToLatest = () => {
    const el = feedRef.current;
    if (el && el.scrollHeight > el.clientHeight + 8) {
      el.scrollTop = el.scrollHeight;
    } else if (typeof window !== 'undefined' && typeof window.scrollTo === 'function') {
      window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
    }
    setStickToBottom(true);
  };

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const updateStickState = () => {
      const doc = document.documentElement;
      const dist = doc.scrollHeight - window.scrollY - window.innerHeight;
      setStickToBottom(dist <= 180);
    };
    updateStickState();
    window.addEventListener('scroll', updateStickState, { passive: true });
    return () => window.removeEventListener('scroll', updateStickState);
  }, []);

  useEffect(() => {
    const grew = filtered.length > prevCountRef.current;
    prevCountRef.current = filtered.length;
    if (!grew || !stickToBottom) return;
    const el = feedRef.current;
    if (el && el.scrollHeight > el.clientHeight + 8) {
      el.scrollTop = el.scrollHeight;
    } else if (typeof window !== 'undefined' && typeof window.scrollTo === 'function') {
      window.scrollTo({ top: document.documentElement.scrollHeight });
    }
  }, [filtered.length, cartridgeId, stickToBottom]);

  const handleFeedScroll = () => {
    const el = feedRef.current;
    if (!el || el.scrollHeight <= el.clientHeight + 8) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setStickToBottom(distanceFromBottom <= 140);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = composerText.trim();
    if (!trimmed) return;
    await onSteer(trimmed);
    setComposerText('');
    scrollToLatest();
  };

  return (
    <div className="writers-room-layout">
      {/* Pod Filter, Summary/Expand Toggle & Search Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
          padding: '10px 14px',
          borderBlockEnd: '1px solid var(--border-subtle)',
          background: 'var(--bg-surface)',
        }}
      >
        <div className="pill-list" role="group" aria-label="Filter Writers' Room by Specialist Pod">
          {POD_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              className={`pill${pod === opt.id ? ' is-active' : ''}`}
              aria-pressed={pod === opt.id}
              onClick={() => setPod(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {!stickToBottom && filtered.length > 0 ? (
            <button type="button" className="btn btn-ghost" onClick={scrollToLatest} style={{ fontSize: '0.74rem' }}>
              <ArrowDown size={13} /> Latest Turn
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn-ghost"
            aria-pressed={expandAll}
            onClick={() => setExpandAll((v) => !v)}
            title={expandAll ? 'Collapse long turns to lead summaries' : 'Expand all specialist turns in full'}
            style={{ fontSize: '0.74rem' }}
          >
            {expandAll ? <ChevronsDownUp size={13} /> : <ChevronsUpDown size={13} />}
            {expandAll ? 'Compact Summaries' : 'Expand All'}
          </button>
          <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
            <Search
              size={13}
              style={{ position: 'absolute', insetInlineStart: '8px', color: 'var(--text-muted)', pointerEvents: 'none' }}
            />
            <input
              type="search"
              className="input-control"
              style={{ paddingInlineStart: '26px', inlineSize: '230px' }}
              placeholder="Search dialogue or rationale..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Transcript Feed */}
      <div
        ref={feedRef}
        onScroll={handleFeedScroll}
        className="scroll-pane"
        style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}
      >
        {filtered.length === 0 ? (
          <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
            {utterances.length === 0
              ? "No Specialist Council turns recorded for this slice yet."
              : 'No utterances match the selected pod or search query.'}
          </div>
        ) : (
          filtered.map((u, idx) => {
            const meta = getRoleMetadata(u.speaker_role, u.speaker_display_name);
            const formatted = formatWritersRoomUtterance(u);
            const emittedPath =
              typeof u.artifact_emitted === 'string'
                ? u.artifact_emitted
                : u.artifact_emitted && typeof u.artifact_emitted === 'object'
                  ? u.artifact_emitted.path || u.artifact_emitted.url || u.artifact_emitted.image_path || ''
                  : '';
            const proofPaths = Array.isArray(u.proof_image_paths) ? u.proof_image_paths : [];
            const images = [
              ...proofPaths,
              ...(emittedPath && !emittedPath.toLowerCase().endsWith('.json') ? [emittedPath] : []),
            ].filter((p): p is string => typeof p === 'string' && p.trim().length > 0);
            const addressedTo = Array.isArray(u.addressed_to) ? u.addressed_to : [];
            const primaryBadges = formatted.badges.slice(0, 5);
            const extraBadges = formatted.badges.slice(5);
            const isQueryActive = Boolean(query.trim());
            const isCouncilChair = u.speaker_role === 'CoProducer';

            return (
              <article
                key={u.utterance_id || `${u.turn_index ?? idx}-${idx}`}
                className={idx >= 12 ? 'deferred-turn-row' : undefined}
                data-council-chair={isCouncilChair ? 'true' : undefined}
                style={{
                  padding: '12px 15px',
                  borderRadius: 'var(--radius-md)',
                  background:
                    meta.pod === 'producer'
                      ? 'color-mix(in srgb, var(--warn) 8%, var(--bg-surface))'
                      : isCouncilChair
                        ? 'color-mix(in srgb, var(--accent) 6%, var(--bg-surface))'
                        : 'var(--bg-surface)',
                  border: `1px solid ${
                    meta.pod === 'producer'
                      ? 'color-mix(in srgb, var(--warn) 45%, var(--border-subtle))'
                      : isCouncilChair
                        ? 'color-mix(in srgb, var(--accent) 42%, var(--border-subtle))'
                        : 'var(--border-subtle)'
                  }`,
                  display: 'grid',
                  gridTemplateColumns: 'minmax(0, 1fr)',
                  gap: '8px',
                  maxInlineSize: '880px',
                  overflow: 'clip',
                }}
              >
                <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '7px', flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.8rem', color: meta.accentVar }}>
                      {meta.label}
                    </span>
                    {isCouncilChair ? (
                      <span className="badge badge-accent">Council Chair</span>
                    ) : null}
                    <span className="badge">{meta.pod}</span>
                    {formatted.turnKindLabel ? (
                      <span
                        className={
                          meta.pod === 'producer'
                            ? 'badge badge-warn'
                            : formatted.turnKindLabel === 'Slice Strategy'
                              ? 'badge badge-accent'
                              : 'badge badge-purple'
                        }
                      >
                        {formatted.turnKindLabel}
                      </span>
                    ) : null}
                    {addressedTo.length > 0 ? (
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        → {addressedTo.map((r) => getRoleMetadata(String(r)).shortLabel).join(', ')}
                      </span>
                    ) : null}
                  </div>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {u.turn_index !== undefined ? `#${u.turn_index}` : ''}
                    {u.timestamp_utc ? ` · ${new Date(u.timestamp_utc).toLocaleTimeString()}` : ''}
                  </span>
                </header>

                {/* Rule 4: Pre-dialogue internal specialist deliberation (omitted when identical to task-shaping lead) */}
                <DeliberationDisclosure
                  deliberation={formatted.distinctDeliberation}
                  label="Internal Specialist Deliberation"
                />

                {/* Lead Summary */}
                <StructuredProse blocks={formatted.leadBlocks} />

                {/* Native <details> Expandable Full Detail — Find-in-Page searchable by default */}
                {formatted.hasExpandableDetail ? (
                  <details
                    key={expandAll || isQueryActive ? 'expanded' : 'compact'}
                    open={expandAll || isQueryActive ? true : undefined}
                    className="utterance-detail-disclosure"
                  >
                    <summary className="utterance-detail-summary">
                      <ChevronRight size={13} />
                      <span>Show full turn (+{formatted.detailWordCount} words)</span>
                    </summary>
                    <StructuredProse blocks={formatted.detailBlocks} />
                  </details>
                ) : null}

                {images.length > 0 ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '8px' }}>
                    {images.map((imgPath, imgIdx) => {
                      const resolved = resolveMediaUrl(imgPath, cartridgeId);
                      return (
                        <button
                          key={imgIdx}
                          type="button"
                          onClick={() =>
                            onOpenLightbox({
                              url: resolved,
                              title: `${meta.label} — Turn #${u.turn_index ?? idx}`,
                              subtitle: imgPath,
                              deliberation: u.deliberation,
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
                            src={resolved}
                            alt={`Council attachment ${imgIdx + 1}`}
                            loading="lazy"
                            style={{ aspectRatio: '16 / 9', inlineSize: '100%', objectFit: 'cover' }}
                          />
                        </button>
                      );
                    })}
                  </div>
                ) : null}

                {primaryBadges.length > 0 ? (
                  <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '5px' }}>
                    {primaryBadges.map((b, bi) => (
                      <span
                        key={bi}
                        className={badgeToneClass(b.tone)}
                        title={b.title}
                        style={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        {b.label}
                      </span>
                    ))}
                    {extraBadges.length > 0 ? (
                      <details style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                        <summary
                          className="badge"
                          style={{ cursor: 'pointer', textTransform: 'none', listStyle: 'none' }}
                        >
                          +{extraBadges.length} more
                        </summary>
                        <span style={{ display: 'inline-flex', flexWrap: 'wrap', gap: '5px', marginInlineStart: '5px' }}>
                          {extraBadges.map((b, bi) => (
                            <span
                              key={bi}
                              className={badgeToneClass(b.tone)}
                              title={b.title}
                              style={{ textTransform: 'none', fontWeight: 600 }}
                            >
                              {b.label}
                            </span>
                          ))}
                        </span>
                      </details>
                    ) : null}
                  </div>
                ) : null}
              </article>
            );
          })
        )}

        {isBuildRunning && liveActivity?.isActive && !query.trim() ? (
          <div
            className="post-board-callout"
            data-tone={liveActivity.phase === 'qa_repair' ? 'warn' : undefined}
          >
            <div style={{ display: 'grid', gap: '4px', minInlineSize: 0, flex: '1 1 360px' }}>
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
              <div style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                {liveActivity.phase === 'writers_room'
                  ? liveActivity.headline
                  : `Council deliberation paused while Studio executes ${liveActivity.stageTitle} — ${liveActivity.headline}`}
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
      </div>

      {/* Docked Executive Producer Composer */}
      <form
        className="writers-room-composer"
        onSubmit={(e) => void handleSubmit(e)}
      >
        <input
          type="text"
          className="input-control"
          style={{ flex: '1 1 auto', paddingBlock: '7px' }}
          placeholder={
            isBuildRunning || awaitingGreenlight
              ? "Speak directly into the live Writers' Room as Executive Producer..."
              : 'Send direction & resume Studio Council for this game...'
          }
          value={composerText}
          onChange={(e) => setComposerText(e.target.value)}
          disabled={busy}
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={busy || !composerText.trim()}
        >
          <Send size={13} />{' '}
          {busy ? 'Sending...' : isBuildRunning || awaitingGreenlight ? 'Send Direction' : 'Steer & Resume'}
        </button>
        {awaitingGreenlight ? (
          <button
            type="button"
            className="btn btn-success"
            disabled={busy}
            onClick={() => void onGreenlight(composerText.trim())}
          >
            <Check size={14} /> Greenlight &amp; Lock Bible
          </button>
        ) : null}
      </form>
    </div>
  );
}
