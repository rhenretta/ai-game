import { useId, useMemo, useState, type CSSProperties } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Anchor, Clock, Flame, RefreshCw, Search, Sparkles, Trophy } from 'lucide-react';
import { studioApi } from '../../api/client';
import type { GameBibleResponse } from '../../api/types';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';
import { ErrorBanner } from '../../components/ErrorBanner';
import {
  STEAM_HORIZONS,
  getRecent2WeeksHours,
  resolveSteamDnaHorizon,
  type HorizonKey,
} from '../../domain/steamDna';

export interface PlayerDnaViewProps {
  bible?: GameBibleResponse;
}

function WeightBars({
  weights,
  accentColor = 'var(--accent)',
  highlightedLabels = [],
  emptyMessage = 'No weights recorded for this window.',
}: {
  weights: Record<string, number> | undefined;
  accentColor?: string;
  highlightedLabels?: string[];
  emptyMessage?: string;
}) {
  const highlightSet = new Set(highlightedLabels.map((s) => s.toLowerCase()));
  const entries = Object.entries(weights ?? {}).sort(
    (a, b) => Number(b[1] || 0) - Number(a[1] || 0),
  );
  if (entries.length === 0) {
    return <div style={{ fontSize: '0.87rem', color: 'var(--text-muted)' }}>{emptyMessage}</div>;
  }
  return (
    <div style={{ display: 'grid', gap: '6px' }}>
      {entries.slice(0, 10).map(([name, val]) => {
        const pct = Math.max(0, Math.min(100, Number(val || 0) * 100));
        const isTop = highlightSet.has(name.toLowerCase());
        return (
          <div key={name} style={{ display: 'grid', gap: '3px' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '8px',
                fontSize: '0.84rem',
              }}
            >
              <span style={{ fontWeight: isTop ? 700 : 400, color: isTop ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                {name}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', color: accentColor, fontWeight: 600 }}>
                {pct.toFixed(1)}%
              </span>
            </div>
            <div
              style={{
                blockSize: '6px',
                background: 'var(--bg-subtle)',
                borderRadius: '3px',
                overflow: 'clip',
              }}
            >
              <div
                style={{
                  inlineSize: `${Math.max(val > 0 ? 2 : 0, pct)}%`,
                  blockSize: '100%',
                  background: accentColor,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function PlayerDnaView({ bible }: PlayerDnaViewProps) {
  const queryClient = useQueryClient();
  const [horizon, setHorizon] = useState<HorizonKey>('lifetime');
  const [gameFilter, setGameFilter] = useState('');
  const tableCaptionId = useId();
  const searchInputId = useId();

  const dnaQuery = useQuery({
    queryKey: ['steam-dna'],
    queryFn: () => studioApi.getSteamDna(false),
  });

  const rescanMut = useMutation({
    mutationFn: () => studioApi.getSteamDna(true),
    onSuccess: (fresh) => {
      queryClient.setQueryData(['steam-dna'], fresh);
    },
  });

  const dna = dnaQuery.data;

  const horizonView = useMemo(
    () =>
      dna
        ? resolveSteamDnaHorizon(dna, horizon, bible?.target_player_dna_summary ?? '')
        : null,
    [dna, horizon, bible?.target_player_dna_summary],
  );

  const filteredGames = useMemo(() => {
    if (!horizonView) return [];
    const q = gameFilter.trim().toLowerCase();
    if (!q) return horizonView.games;
    return horizonView.games.filter((g) => {
      const hay = [
        g.name,
        String(g.appid ?? ''),
        ...(g.genres ?? []),
        ...(g.categories ?? []),
        g.short_description ?? '',
      ]
        .join(' ')
        .toLowerCase();
      return hay.includes(q);
    });
  }, [horizonView, gameFilter]);

  const prov = (dna?.ingestion_provenance ?? {}) as Record<string, unknown>;
  const methodology = ((prov.methodology as Record<string, string> | undefined) ?? {});
  const pacingScore = Number(dna?.pacing_score || 0);
  const pacingPct = Math.max(0, Math.min(100, ((pacingScore + 1) / 2) * 100));
  const challengeVal = Number(dna?.challenge_tolerance ?? 0.5);
  const challengePct = Math.max(0, Math.min(100, challengeVal * 100));

  const liveDirective = (dna?.live_writers_room_dna_directive || bible?.target_player_dna_summary || '').trim();
  const snapshotDirective = (bible?.target_player_dna_summary || '').trim();
  const conceptCandidates = Array.isArray(bible?.concept_candidates) ? bible.concept_candidates : [];
  const conceptFragments = Array.isArray(bible?.concept_fragments) ? bible.concept_fragments : [];

  return (
    <div className="scroll-pane" style={{ padding: '18px 22px' }}>
      <div style={{ maxInlineSize: '1240px', display: 'grid', gap: '16px' }}>
        {/* Header & Horizon Switcher */}
        <header
          style={{
            display: 'flex',
            alignItems: 'safe center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '10px',
          }}
        >
          <div>
            <h1 style={{ margin: 0, fontSize: '1.31rem', fontWeight: 800 }}>
              Steam Player DNA &amp; Epistemic Synthesis
            </h1>
            {dna ? (
              <div style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginBlockStart: '3px' }}>
                SteamID: <code>{dna.player_id}</code> · Profile: <code>{dna.profile_tag}</code> ·{' '}
                <span>
                  {Number(dna.total_games_analyzed || 0)} games ({Number(dna.total_playtime_hours || 0).toFixed(1)}h lifetime)
                </span>
                {dna.created_at_utc
                  ? ` · Synthesized ${new Date(dna.created_at_utc).toLocaleString()}`
                  : ''}
              </div>
            ) : null}
          </div>

          <div style={{ display: 'flex', alignItems: 'safe center', gap: '8px', flexWrap: 'wrap' }}>
            <div className="pill-list" role="group" aria-label="Activity Time Horizon">
              {STEAM_HORIZONS.map((h) => (
                <button
                  key={h.key}
                  type="button"
                  aria-pressed={horizon === h.key}
                  className={`pill${horizon === h.key ? ' is-active' : ''}`}
                  onClick={() => setHorizon(h.key)}
                >
                  {h.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="btn btn-primary"
              disabled={rescanMut.isPending}
              onClick={() => rescanMut.mutate()}
            >
              <RefreshCw size={13} />
              {rescanMut.isPending ? 'Re-Synthesizing...' : 'Re-Synthesize via LLM'}
            </button>
          </div>
        </header>

        <ErrorBanner
          error={dnaQuery.error || rescanMut.error}
          onRetry={() => void dnaQuery.refetch()}
        />

        {dnaQuery.isLoading ? (
          <div style={{ padding: '24px', color: 'var(--text-muted)' }}>
            Loading Steam Player DNA profile...
          </div>
        ) : dna && horizonView ? (
          <>
            {/* Rule 4: LLM Epistemic Deliberation strictly first */}
            <DeliberationDisclosure
              deliberation={dna.deliberation}
              label="LLM Epistemic Library Synthesis"
              defaultOpen
            />

            {/* 1. Interactive 4-Horizon Comparison Cards (30d · 90d · 1yr · Lifetime) */}
            <section style={{ display: 'grid', gap: '8px' }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '8px',
                }}
              >
                <h2 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700 }}>
                  Activity Horizons Overview (30d · 90d · 1yr · Lifetime)
                </h2>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Select any horizon to update stats, mastery tiers, genre/category distributions, and library table
                </span>
              </div>

              <div className="dna-horizon-grid" role="group" aria-label="Activity Horizon Summary Cards">
                {horizonView.horizonSummaries.map((card) => {
                  const selected = card.meta.key === horizon;
                  return (
                    <button
                      key={card.meta.key}
                      type="button"
                      aria-pressed={selected}
                      className="dna-horizon-card"
                      style={{ '--horizon-accent': card.meta.accentColor } as CSSProperties}
                      onClick={() => setHorizon(card.meta.key)}
                    >
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          gap: '8px',
                        }}
                      >
                        <strong style={{ fontSize: '0.87rem', color: card.meta.accentColor }}>
                          {card.meta.label}
                        </strong>
                        <span
                          className="badge"
                          style={{
                            fontFamily: 'var(--font-mono)',
                            textTransform: 'none',
                            fontWeight: 700,
                          }}
                        >
                          {card.gamesCount} {card.gamesCount === 1 ? 'game' : 'games'}
                        </span>
                      </div>

                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'baseline',
                          justifyContent: 'space-between',
                          gap: '8px',
                          fontFamily: 'var(--font-mono)',
                        }}
                      >
                        <span style={{ fontSize: '1.24rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                          {card.totalPlaytimeHours.toLocaleString(undefined, {
                            minimumFractionDigits: 1,
                            maximumFractionDigits: 1,
                          })}
                          h
                        </span>
                        <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                          {card.shareOfLifetimePct.toFixed(1)}% of lifetime
                        </span>
                      </div>

                      {card.recent2WeeksHours > 0 ? (
                        <div
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '0.76rem',
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--ok)',
                          }}
                        >
                          <Flame size={11} />
                          {card.recent2WeeksHours.toFixed(1)}h in last 2 weeks
                        </div>
                      ) : null}

                      {card.topGenres.length > 0 ? (
                        <div className="dna-chip-wrap" style={{ marginBlockStart: '2px' }}>
                          {card.topGenres.slice(0, 3).map((gn) => (
                            <span
                              key={gn}
                              className="badge"
                              style={{ fontSize: '0.69rem', textTransform: 'none', padding: '1px 5px' }}
                            >
                              {gn}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                          No active games in window
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </section>

            {/* 2. Selected Horizon KPIs & Continuum Behavioral Gauges */}
            <section
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                gap: '10px',
              }}
            >
              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  display: 'grid',
                  gap: '4px',
                }}
              >
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  {horizonView.meta.label} Playtime
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
                  <span
                    style={{
                      fontSize: '1.6rem',
                      fontWeight: 800,
                      color: 'var(--accent)',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    {horizonView.totalPlaytimeHours.toLocaleString(undefined, {
                      minimumFractionDigits: 1,
                      maximumFractionDigits: 1,
                    })}
                    h
                  </span>
                  <span style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                    ({horizonView.shareOfLifetimePct.toFixed(1)}% of{' '}
                    {horizonView.lifetimePlaytimeHours.toFixed(0)}h)
                  </span>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--ok)', fontFamily: 'var(--font-mono)' }}>
                  {horizonView.recent2WeeksHours > 0
                    ? `${horizonView.recent2WeeksHours.toFixed(1)}h logged in last 14 days`
                    : `${horizonView.masteredSharePct.toFixed(1)}% in Mastered (>100h) titles`}
                </div>
              </div>

              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  display: 'grid',
                  gap: '4px',
                }}
              >
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  {horizonView.isLifetime
                    ? 'Total Games Analyzed'
                    : `Active Games (${horizonView.meta.label})`}
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
                  <span
                    style={{
                      fontSize: '1.6rem',
                      fontWeight: 800,
                      color: 'var(--ok)',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    {horizonView.gamesCount}
                  </span>
                  <span style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                    avg {horizonView.avgHoursPerGame.toFixed(1)}h / game
                  </span>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {horizonView.latestPlayedDate
                    ? `Latest session: ${horizonView.latestPlayedDate}${
                        horizonView.latestDaysAgo !== null ? ` (${horizonView.latestDaysAgo}d ago)` : ''
                      }`
                    : `${horizonView.masteredSharePct.toFixed(1)}% playtime in >100h titles`}
                </div>
              </div>

              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  display: 'grid',
                  gap: '6px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                    Cognitive Pacing Index
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--warn)', fontWeight: 700 }}>
                    {pacingScore >= 0 ? '+' : ''}
                    {pacingScore.toFixed(2)} ({dna.pacing_preference || 'balanced'})
                  </span>
                </div>
                <div
                  style={{
                    position: 'relative',
                    blockSize: '9px',
                    background: 'linear-gradient(90deg, #38bdf8 0%, #a78bfa 50%, #f87171 100%)',
                    borderRadius: '999px',
                    marginBlock: '2px',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      insetInlineStart: `calc(${pacingPct}% - 5px)`,
                      insetBlockStart: '-3px',
                      inlineSize: '10px',
                      blockSize: '15px',
                      background: 'var(--text-primary)',
                      border: '2px solid var(--bg-canvas)',
                      borderRadius: '3px',
                    }}
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                  <span>-1.0 Tactical</span>
                  <span>0.0 Balanced</span>
                  <span>+1.0 Kinetic</span>
                </div>
              </div>

              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  display: 'grid',
                  gap: '6px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                    Challenge &amp; Complexity Appetite
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.84rem', color: 'var(--warn)', fontWeight: 700 }}>
                    {(challengeVal * 100).toFixed(0)}% ({challengeVal.toFixed(2)})
                  </span>
                </div>
                <div
                  style={{
                    blockSize: '8px',
                    background: 'var(--bg-subtle)',
                    borderRadius: '999px',
                    overflow: 'clip',
                  }}
                >
                  <div
                    style={{
                      inlineSize: `${challengePct}%`,
                      blockSize: '100%',
                      background: 'linear-gradient(90deg, var(--ok), var(--warn), var(--err))',
                    }}
                  />
                </div>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }} className="clamp-2">
                  {methodology.challenge_evaluation ||
                    'Holistic LLM assessment across systemic/engineering complexity and mechanical skill ceiling.'}
                </div>
              </div>
            </section>

            {/* 3. 5-Tier Playtime & Mastery Depth Distribution (Dynamic per Selected Horizon!) */}
            <section
              style={{
                padding: '14px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                display: 'grid',
                gap: '10px',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '8px',
                }}
              >
                <h2 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                  <Trophy size={14} style={{ color: 'var(--ok)' }} />
                  5-Tier Playtime &amp; Mastery Depth ({horizonView.meta.label})
                </h2>
                <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  {horizonView.masteredSharePct.toFixed(1)}% of {horizonView.meta.shortLabel} playtime in Mastered (&gt;100h) titles
                </span>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(185px, 1fr))',
                  gap: '10px',
                }}
              >
                {horizonView.playtimeTiers.map((tier) => {
                  const pct = Math.max(0, Math.min(100, tier.share * 100));
                  return (
                    <div
                      key={tier.key}
                      style={{
                        padding: '9px 11px',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--bg-canvas)',
                        border: '1px solid var(--border-subtle)',
                        display: 'grid',
                        gap: '4px',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '6px' }}>
                        <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>
                          {tier.label}{' '}
                          <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.76rem' }}>
                            ({tier.rangeLabel})
                          </span>
                        </strong>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.84rem', fontWeight: 700, color: tier.color }}>
                          {pct.toFixed(1)}%
                        </span>
                      </div>
                      <div
                        style={{
                          blockSize: '6px',
                          background: 'var(--bg-subtle)',
                          borderRadius: '3px',
                          overflow: 'clip',
                        }}
                      >
                        <div
                          style={{
                            inlineSize: `${Math.max(tier.share > 0 ? 2 : 0, pct)}%`,
                            blockSize: '100%',
                            background: tier.color,
                          }}
                        />
                      </div>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          fontSize: '0.76rem',
                          fontFamily: 'var(--font-mono)',
                          color: 'var(--text-muted)',
                        }}
                      >
                        <span>~{tier.hours.toFixed(1)}h</span>
                        {tier.gamesCount > 0 || !horizonView.isLifetime ? (
                          <span>
                            {tier.gamesCount} {tier.gamesCount === 1 ? 'title' : 'titles'}
                          </span>
                        ) : null}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* 4. Multi-Dimensional Affinity Vectors (Window-Dynamic Genres & Categories + LLM Synthesis Vectors) */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(275px, 1fr))',
                gap: '12px',
              }}
            >
              <section
                style={{
                  padding: '14px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  display: 'grid',
                  gap: '10px',
                  alignContent: 'start',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
                  <h2 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700 }}>
                    Genre Affinities
                  </h2>
                  <span className="badge badge-accent" style={{ textTransform: 'none', fontSize: '0.73rem' }}>
                    {horizonView.meta.shortLabel}
                  </span>
                </div>
                <WeightBars
                  weights={horizonView.genreWeights}
                  accentColor="var(--accent)"
                  highlightedLabels={horizonView.topGenres}
                  emptyMessage={`No genre activity recorded in ${horizonView.meta.label}.`}
                />
              </section>

              <section
                style={{
                  padding: '14px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  display: 'grid',
                  gap: '10px',
                  alignContent: 'start',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
                  <h2 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700 }}>
                    Play Mode &amp; Categories
                  </h2>
                  <span className="badge badge-warn" style={{ textTransform: 'none', fontSize: '0.73rem' }}>
                    {horizonView.meta.shortLabel}
                  </span>
                </div>
                <WeightBars
                  weights={horizonView.categoryWeights}
                  accentColor="var(--warn)"
                  highlightedLabels={horizonView.topCategories}
                  emptyMessage={`No category activity recorded in ${horizonView.meta.label}.`}
                />
              </section>

              <section
                style={{
                  padding: '14px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  display: 'grid',
                  gap: '12px',
                  alignContent: 'start',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
                  <h2 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700 }}>
                    Mechanical Verbs &amp; Themes
                  </h2>
                  <span className="badge badge-purple" style={{ textTransform: 'none', fontSize: '0.73rem' }}>
                    LLM Synthesis
                  </span>
                </div>

                <div style={{ display: 'grid', gap: '6px' }}>
                  <h3 style={{ margin: 0, fontSize: '0.76rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                    Core Mechanical Verbs
                  </h3>
                  <WeightBars weights={dna.mechanical_biases} accentColor="var(--ok)" />
                </div>

                <div style={{ display: 'grid', gap: '6px' }}>
                  <h3 style={{ margin: 0, fontSize: '0.76rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                    Thematic Motifs &amp; Palette ({dna.preferred_palette_mood || 'default'})
                  </h3>
                  <WeightBars
                    weights={
                      Object.keys(dna.thematic_weights ?? {}).length > 0
                        ? dna.thematic_weights
                        : dna.palette_weights
                    }
                    accentColor="var(--purple)"
                    highlightedLabels={dna.thematic_biases}
                  />
                </div>
              </section>
            </div>

            {/* 5. Analyzed Steam Library Titles Table Filtered by Active Horizon (30d / 90d / 1yr / Lifetime) */}
            <section
              style={{
                padding: '14px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                display: 'grid',
                gap: '10px',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'safe center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '10px',
                }}
              >
                <h2 style={{ margin: 0, fontSize: '1.02rem', fontWeight: 700 }}>
                  {horizonView.isLifetime
                    ? `Top Analyzed Steam Library Titles (${filteredGames.length}${
                        horizonView.lifetimeGamesCount > filteredGames.length && !gameFilter.trim()
                          ? ` of ${horizonView.lifetimeGamesCount}`
                          : ''
                      })`
                    : `Active Steam Library Titles in ${horizonView.meta.label} (${filteredGames.length})`}
                </h2>

                <search>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <label
                      htmlFor={searchInputId}
                      style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
                    >
                      <Search size={12} />
                      Filter titles:
                    </label>
                    <input
                      id={searchInputId}
                      type="search"
                      className="input-text"
                      style={{ inlineSize: '220px', paddingBlock: '4px', fontSize: '0.84rem' }}
                      placeholder="Title, genre, category..."
                      value={gameFilter}
                      onChange={(e) => setGameFilter(e.target.value)}
                    />
                  </div>
                </search>
              </div>

              <div
                tabIndex={0}
                role="region"
                aria-labelledby={tableCaptionId}
                className="dna-table-region"
              >
                <table className="dna-library-table">
                  <caption id={tableCaptionId}>
                    {horizonView.meta.label} Steam games ordered by logged engagement ({filteredGames.length}{' '}
                    {filteredGames.length === 1 ? 'title' : 'titles'} shown)
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">#</th>
                      <th scope="col">Game Title &amp; Store Description</th>
                      <th scope="col">AppID</th>
                      <th scope="col">Last Played &amp; 2wk Activity</th>
                      <th scope="col">Logged Playtime</th>
                      <th scope="col">
                        {horizonView.isLifetime ? 'Library Playtime Share' : `${horizonView.meta.shortLabel} / Library Share`}
                      </th>
                      <th scope="col">Steam Store Genres &amp; Categories</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredGames.length === 0 ? (
                      <tr>
                        <td
                          colSpan={7}
                          style={{
                            padding: '18px 12px',
                            textAlign: 'center',
                            color: 'var(--text-muted)',
                          }}
                        >
                          {gameFilter.trim()
                            ? `No titles in ${horizonView.meta.label} matched "${gameFilter.trim()}".`
                            : `No Steam games recorded in the ${horizonView.meta.label} window.`}
                        </td>
                      </tr>
                    ) : (
                      filteredGames.map((g, idx) => {
                        const gName = g.name || `App ${g.appid ?? idx + 1}`;
                        const nameLower = gName.toLowerCase();
                        const hrs = Math.max(0, Number(g.playtime_hours || 0));
                        const recent2wk = getRecent2WeeksHours(g);
                        const lifetimeSharePct =
                          (typeof g.playtime_share === 'number'
                            ? g.playtime_share
                            : horizonView.lifetimePlaytimeHours > 0
                              ? hrs / horizonView.lifetimePlaytimeHours
                              : 0) * 100;
                        const windowSharePct =
                          horizonView.totalPlaytimeHours > 0
                            ? Math.min(100, (hrs / horizonView.totalPlaytimeHours) * 100)
                            : 0;
                        const primarySharePct = horizonView.isLifetime
                          ? lifetimeSharePct
                          : windowSharePct;

                        const isAnchor =
                          (horizonView.isLifetime && !gameFilter.trim() && idx < 8) ||
                          horizonView.anchorTitleSet.has(nameLower);
                        const isSpotlight = horizonView.spotlightTitleSet.has(nameLower);
                        const genres = Array.isArray(g.genres) ? g.genres : [];
                        const categories = Array.isArray(g.categories) ? g.categories : [];
                        const lpDate = (g.last_played_date ?? '').trim();
                        const daysSince =
                          typeof g.days_since_played === 'number' ? g.days_since_played : null;

                        return (
                          <tr key={`${g.appid ?? gName}-${idx}`}>
                            <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                              {idx + 1}
                            </td>
                            <th scope="row">
                              <div
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '6px',
                                  flexWrap: 'wrap',
                                }}
                              >
                                <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
                                  {gName}
                                </span>
                                {isAnchor ? (
                                  <span
                                    className="badge badge-accent"
                                    style={{ fontSize: '0.69rem', textTransform: 'none' }}
                                  >
                                    <Anchor size={9} /> Core Anchor
                                  </span>
                                ) : null}
                                {isSpotlight ? (
                                  <span
                                    className="badge badge-warn"
                                    style={{ fontSize: '0.69rem', textTransform: 'none' }}
                                  >
                                    <Sparkles size={9} /> Session Spotlight
                                  </span>
                                ) : null}
                              </div>
                              {g.short_description ? (
                                <div
                                  className="clamp-2"
                                  style={{
                                    fontSize: '0.76rem',
                                    fontWeight: 400,
                                    color: 'var(--text-muted)',
                                    marginBlockStart: '3px',
                                    maxInlineSize: '460px',
                                  }}
                                >
                                  {g.short_description}
                                </div>
                              ) : null}
                            </th>
                            <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                              {g.appid ?? '—'}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                              {lpDate ? (
                                <div>
                                  <span style={{ color: 'var(--text-primary)' }}>{lpDate}</span>
                                  {daysSince !== null ? (
                                    <span style={{ color: 'var(--text-muted)' }}> ({daysSince}d ago)</span>
                                  ) : null}
                                </div>
                              ) : (
                                <span style={{ color: 'var(--text-muted)' }}>—</span>
                              )}
                              {recent2wk > 0 ? (
                                <div
                                  style={{
                                    color: 'var(--ok)',
                                    fontSize: '0.73rem',
                                    fontWeight: 700,
                                    marginBlockStart: '2px',
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '3px',
                                  }}
                                >
                                  <Clock size={10} />
                                  {recent2wk.toFixed(1)}h in last 2 wks
                                </div>
                              ) : null}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
                              <strong style={{ color: 'var(--warn)' }}>{hrs.toFixed(1)}h</strong>
                            </td>
                            <td style={{ minInlineSize: '165px' }}>
                              <div
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'space-between',
                                  gap: '6px',
                                  fontFamily: 'var(--font-mono)',
                                  fontSize: '0.76rem',
                                }}
                              >
                                <span style={{ color: 'var(--ok)', fontWeight: 700 }}>
                                  {primarySharePct.toFixed(1)}%{' '}
                                  <span style={{ fontWeight: 400, color: 'var(--text-muted)' }}>
                                    {horizonView.isLifetime
                                      ? 'lifetime'
                                      : `of ${horizonView.meta.shortLabel}`}
                                  </span>
                                </span>
                                {!horizonView.isLifetime ? (
                                  <span style={{ color: 'var(--text-muted)', fontSize: '0.73rem' }}>
                                    ({lifetimeSharePct.toFixed(1)}% all)
                                  </span>
                                ) : null}
                              </div>
                              <div
                                style={{
                                  blockSize: '5px',
                                  background: 'var(--bg-subtle)',
                                  borderRadius: '3px',
                                  overflow: 'clip',
                                  marginBlockStart: '3px',
                                }}
                              >
                                <div
                                  style={{
                                    inlineSize: `${Math.min(
                                      100,
                                      horizonView.isLifetime ? primarySharePct * 4 : primarySharePct,
                                    )}%`,
                                    blockSize: '100%',
                                    background: 'var(--ok)',
                                  }}
                                />
                              </div>
                            </td>
                            <td>
                              <div className="dna-chip-wrap">
                                {genres.map((gn) => (
                                  <span
                                    key={`g-${gn}`}
                                    className="badge badge-accent"
                                    style={{ fontSize: '0.69rem', textTransform: 'none', padding: '1px 5px' }}
                                  >
                                    {gn}
                                  </span>
                                ))}
                                {categories.slice(0, 3).map((ct) => (
                                  <span
                                    key={`c-${ct}`}
                                    className="badge"
                                    style={{
                                      fontSize: '0.69rem',
                                      textTransform: 'none',
                                      padding: '1px 5px',
                                      color: 'var(--warn)',
                                    }}
                                  >
                                    {ct}
                                  </span>
                                ))}
                              </div>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            {/* 6. 4-Stage Creation Provenance & Writers' Room Concept Translation */}
            <section
              style={{
                padding: '14px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                display: 'grid',
                gap: '12px',
              }}
            >
              <h2 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700 }}>
                4-Stage Steam DNA Ingestion &amp; Writers&apos; Room Genesis Provenance
              </h2>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                  gap: '8px',
                }}
              >
                <div style={{ padding: '9px 11px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)', fontSize: '0.8rem' }}>
                  <div style={{ fontWeight: 700, color: 'var(--accent)', marginBlockEnd: '3px' }}>
                    1. Local Steam VDF Scanner
                  </div>
                  <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Scans <code>loginusers.vdf</code>, <code>libraryfolders.vdf</code>,{' '}
                    <code>localconfig.vdf</code> &amp; <code>appmanifest_*.acf</code> across{' '}
                    {horizonView.lifetimeGamesCount} titles ({horizonView.lifetimePlaytimeHours.toFixed(1)}h).
                  </div>
                </div>

                <div style={{ padding: '9px 11px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)', fontSize: '0.8rem' }}>
                  <div style={{ fontWeight: 700, color: 'var(--warn)', marginBlockEnd: '3px' }}>
                    2. Store API Metadata Cache
                  </div>
                  <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Enriches AppIDs via <code>{String(prov.store_cache_path || 'data/steam_app_cache.json')}</code>{' '}
                    ({Number(prov.store_cached_apps_count || 0)} cached store entries) with genres, categories &amp; descriptions.
                  </div>
                </div>

                <div style={{ padding: '9px 11px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)', fontSize: '0.8rem' }}>
                  <div style={{ fontWeight: 700, color: 'var(--ok)', marginBlockEnd: '3px' }}>
                    3. Gemini LLM Epistemic Synthesis
                  </div>
                  <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    {methodology.epistemic_synthesis ||
                      'Inspects raw playtimes, recent windows, mastery depth, and store descriptions directly to synthesize player affinities.'}
                  </div>
                </div>

                <div style={{ padding: '9px 11px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)', fontSize: '0.8rem' }}>
                  <div style={{ fontWeight: 700, color: 'var(--purple)', marginBlockEnd: '3px' }}>
                    4. Writers&apos; Room DNA Injection
                  </div>
                  <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Injects 4-horizon activity + Core Anchor Games + Spotlight Cross-Pollination titles
                    {conceptCandidates.length > 0 || conceptFragments.length > 0
                      ? ` (${conceptFragments.length} specialist fragments → ${conceptCandidates.length} candidates)`
                      : ' into all Specialist Council prompts'}
                    .
                  </div>
                </div>
              </div>

              {liveDirective ? (
                <details
                  open
                  style={{
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-canvas)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <summary style={{ cursor: 'pointer', fontWeight: 700, fontSize: '0.87rem', color: 'var(--accent)' }}>
                    Live 4-Horizon Specialist Council DNA Prompt Directive (30d · 90d · 1yr · Lifetime)
                  </summary>
                  <pre tabIndex={0} className="prompt-pre" style={{ marginBlockStart: '8px' }}>
                    <code>{liveDirective}</code>
                  </pre>
                </details>
              ) : null}

              {snapshotDirective && snapshotDirective !== liveDirective ? (
                <details
                  style={{
                    padding: '8px 12px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-canvas)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <summary style={{ cursor: 'pointer', fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                    Historical DNA Snapshot Captured When &ldquo;{bible?.game_title || 'Active Build'}&rdquo; Was Started
                  </summary>
                  <pre tabIndex={0} className="prompt-pre" style={{ marginBlockStart: '8px' }}>
                    <code>{snapshotDirective}</code>
                  </pre>
                </details>
              ) : null}

              {conceptCandidates.length > 0 ? (
                <div style={{ display: 'grid', gap: '8px' }}>
                  <h3 style={{ margin: 0, fontSize: '0.87rem', fontWeight: 700, color: 'var(--ok)' }}>
                    Phase 0.5 — Concept Candidates Collided from Steam DNA ({conceptCandidates.length})
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(270px, 1fr))', gap: '8px' }}>
                    {conceptCandidates.map((cc, idx) => {
                      const rec = cc as unknown as Record<string, unknown>;
                      const title = String(rec.working_title ?? rec.candidate_id ?? `Candidate #${idx + 1}`);
                      const isChosen =
                        Boolean(bible?.game_title) &&
                        title.toLowerCase() === String(bible?.game_title ?? '').toLowerCase();
                      const frags = Array.isArray(rec.source_fragment_ids)
                        ? rec.source_fragment_ids.map(String)
                        : [];
                      return (
                        <div
                          key={`${title}-${idx}`}
                          style={{
                            padding: '10px 12px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-canvas)',
                            border: isChosen
                              ? '1.5px solid var(--ok)'
                              : '1px solid var(--border-subtle)',
                            display: 'grid',
                            gap: '5px',
                            fontSize: '0.84rem',
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
                            <strong style={{ color: isChosen ? 'var(--ok)' : 'var(--text-primary)' }}>
                              #{idx + 1}: {title}
                            </strong>
                            {isChosen ? (
                              <span className="badge badge-ok" style={{ fontSize: '0.69rem' }}>
                                Selected
                              </span>
                            ) : null}
                          </div>
                          {rec.premise ? (
                            <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                              {String(rec.premise)}
                            </div>
                          ) : null}
                          {rec.core_tension ? (
                            <div style={{ fontSize: '0.8rem', color: 'var(--warn)' }}>
                              <strong>Core Tension:</strong> {String(rec.core_tension)}
                            </div>
                          ) : null}
                          {frags.length > 0 ? (
                            <div style={{ fontSize: '0.73rem', fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
                              Fragments: {frags.join(' + ')}
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}

              {conceptFragments.length > 0 ? (
                <details
                  style={{
                    padding: '8px 12px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-canvas)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <summary style={{ cursor: 'pointer', fontSize: '0.84rem', fontWeight: 600, color: 'var(--accent)' }}>
                    Inspect All {conceptFragments.length} Phase 0 Specialist Concept Fragments
                  </summary>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                      gap: '8px',
                      marginBlockStart: '8px',
                    }}
                  >
                    {conceptFragments.map((cf, idx) => {
                      const rec = cf as Record<string, unknown>;
                      return (
                        <div
                          key={`${String(rec.fragment_id ?? idx)}`}
                          style={{
                            padding: '8px 10px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-surface)',
                            border: '1px solid var(--border-subtle)',
                            fontSize: '0.8rem',
                            display: 'grid',
                            gap: '4px',
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
                            <strong style={{ color: 'var(--accent)' }}>
                              {String(rec.fragment_id ?? rec.contributor_role ?? `Fragment #${idx + 1}`)}
                            </strong>
                            {rec.fragment_kind ? (
                              <span className="badge" style={{ fontSize: '0.69rem', textTransform: 'none' }}>
                                {String(rec.fragment_kind)}
                              </span>
                            ) : null}
                          </div>
                          <div style={{ color: 'var(--text-primary)', lineHeight: 1.42 }}>
                            {String(rec.fragment ?? '')}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </details>
              ) : null}
            </section>

            {/* 7. Raw /api/steam/dna JSON Inspector */}
            <details
              style={{
                padding: '10px 14px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <summary style={{ cursor: 'pointer', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                Inspect Raw <code>/api/steam/dna</code> JSON Payload
              </summary>
              <pre tabIndex={0} className="prompt-pre" style={{ marginBlockStart: '8px', maxBlockSize: '320px' }}>
                <code>{JSON.stringify(dna, null, 2)}</code>
              </pre>
            </details>
          </>
        ) : null}
      </div>
    </div>
  );
}
