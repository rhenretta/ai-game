import type { SteamActivityWindow, SteamDnaResponse, SteamTopGame } from '../api/types';

export type HorizonKey = 'last_30_days' | 'last_90_days' | 'last_1_year' | 'lifetime';

export interface HorizonMeta {
  key: HorizonKey;
  label: string;
  shortLabel: string;
  days: number | null;
  accentColor: string;
}

export const STEAM_HORIZONS: readonly HorizonMeta[] = [
  {
    key: 'last_30_days',
    label: 'Last 30 Days',
    shortLabel: '30d',
    days: 30,
    accentColor: 'var(--ok)',
  },
  {
    key: 'last_90_days',
    label: 'Last 90 Days',
    shortLabel: '90d',
    days: 90,
    accentColor: 'var(--accent)',
  },
  {
    key: 'last_1_year',
    label: 'Last 1 Year',
    shortLabel: '1yr',
    days: 365,
    accentColor: 'var(--warn)',
  },
  {
    key: 'lifetime',
    label: 'Lifetime',
    shortLabel: 'Lifetime',
    days: null,
    accentColor: 'var(--purple)',
  },
] as const;

export interface PlaytimeTierStat {
  key: string;
  label: string;
  rangeLabel: string;
  color: string;
  share: number;
  hours: number;
  gamesCount: number;
}

export interface HorizonSummaryCardData {
  meta: HorizonMeta;
  gamesCount: number;
  totalPlaytimeHours: number;
  recent2WeeksHours: number;
  shareOfLifetimePct: number;
  topGenres: string[];
}

export interface ResolvedHorizonView {
  meta: HorizonMeta;
  isLifetime: boolean;
  totalPlaytimeHours: number;
  lifetimePlaytimeHours: number;
  shareOfLifetimePct: number;
  recent2WeeksHours: number;
  gamesCount: number;
  lifetimeGamesCount: number;
  avgHoursPerGame: number;
  masteredSharePct: number;
  latestPlayedDate: string | null;
  latestDaysAgo: number | null;
  genreWeights: Record<string, number>;
  topGenres: string[];
  categoryWeights: Record<string, number>;
  topCategories: string[];
  playtimeTiers: PlaytimeTierStat[];
  games: SteamTopGame[];
  horizonSummaries: HorizonSummaryCardData[];
  anchorTitleSet: Set<string>;
  spotlightTitleSet: Set<string>;
}

const TIER_DEFS = [
  {
    key: 'mastered_over_100h',
    label: 'Mastered',
    rangeLabel: '> 100h',
    color: 'var(--ok)',
    minHours: 100,
    maxHours: Number.POSITIVE_INFINITY,
  },
  {
    key: 'dedicated_50_to_100h',
    label: 'Dedicated',
    rangeLabel: '50h – 100h',
    color: 'var(--accent)',
    minHours: 50,
    maxHours: 100,
  },
  {
    key: 'engaged_10_to_50h',
    label: 'Engaged',
    rangeLabel: '10h – 50h',
    color: 'var(--warn)',
    minHours: 10,
    maxHours: 50,
  },
  {
    key: 'casual_2_to_10h',
    label: 'Casual',
    rangeLabel: '2h – 10h',
    color: 'var(--purple)',
    minHours: 2,
    maxHours: 10,
  },
  {
    key: 'dabbled_under_2h',
    label: 'Dabbled',
    rangeLabel: '< 2h',
    color: 'var(--text-muted)',
    minHours: 0,
    maxHours: 2,
  },
] as const;

export function getRecent2WeeksHours(game: SteamTopGame): number {
  if (typeof game.playtime_2weeks_hours === 'number' && Number.isFinite(game.playtime_2weeks_hours)) {
    return Math.max(0, game.playtime_2weeks_hours);
  }
  if (
    typeof game.playtime_2weeks_minutes === 'number' &&
    Number.isFinite(game.playtime_2weeks_minutes)
  ) {
    return Math.max(0, game.playtime_2weeks_minutes / 60);
  }
  return 0;
}

function getWindowGames(
  dna: SteamDnaResponse,
  meta: HorizonMeta,
  win: SteamActivityWindow | undefined,
): SteamTopGame[] {
  if (win) {
    if (Array.isArray(win.games)) return win.games;
    if (Array.isArray(win.top_games)) return win.top_games;
  }
  if (meta.key === 'lifetime') {
    return Array.isArray(dna.top_games) ? dna.top_games : [];
  }
  // Fallback when activity_windows was omitted on synthetic/older payloads but per-game recency exists
  if (meta.days !== null && Array.isArray(dna.top_games)) {
    const hasRecency = dna.top_games.some(
      (g) => typeof g.days_since_played === 'number' || getRecent2WeeksHours(g) > 0,
    );
    if (hasRecency) {
      return dna.top_games.filter((g) => {
        if (typeof g.days_since_played === 'number') {
          return g.days_since_played <= meta.days!;
        }
        return meta.days! >= 14 && getRecent2WeeksHours(g) > 0;
      });
    }
  }
  return [];
}

function getWindowGamesCount(
  dna: SteamDnaResponse,
  meta: HorizonMeta,
  win: SteamActivityWindow | undefined,
  games: SteamTopGame[],
): number {
  if (win) {
    if (typeof win.games_count === 'number') return win.games_count;
    if (typeof win.active_games_count === 'number') return win.active_games_count;
  }
  if (meta.key === 'lifetime') {
    return Number(dna.total_games_analyzed || games.length || 0);
  }
  return games.length;
}

function getWindowPlaytimeHours(
  dna: SteamDnaResponse,
  meta: HorizonMeta,
  win: SteamActivityWindow | undefined,
  games: SteamTopGame[],
): number {
  if (win && typeof win.total_playtime_hours === 'number') {
    return win.total_playtime_hours;
  }
  if (meta.key === 'lifetime') {
    return Number(dna.total_playtime_hours || 0);
  }
  return games.reduce((sum, g) => sum + Math.max(0, Number(g.playtime_hours || 0)), 0);
}

function getWindowRecent2WeeksHours(
  win: SteamActivityWindow | undefined,
  games: SteamTopGame[],
): number {
  if (win && typeof win.recent_2weeks_hours === 'number') {
    return win.recent_2weeks_hours;
  }
  return games.reduce((sum, g) => sum + getRecent2WeeksHours(g), 0);
}

/**
 * Computes normalized L1 weights across a window's active games using the same
 * engagement weight (`max(0.25, playtime_hours) + 2.0 * recent_2weeks_hours`)
 * used by `compute_activity_windows` in `engine/steam/dna.py`.
 */
function computeWindowLabelWeights(
  games: SteamTopGame[],
  extractor: (game: SteamTopGame) => string[] | undefined,
): { weights: Record<string, number>; topLabels: string[] } {
  const rawScores: Record<string, number> = {};
  for (const g of games) {
    const labels = (extractor(g) ?? []).map((s) => String(s).trim()).filter(Boolean);
    if (labels.length === 0) continue;
    const engagementWeight =
      Math.max(0.25, Number(g.playtime_hours || 0)) + 2.0 * getRecent2WeeksHours(g);
    const perLabel = engagementWeight / labels.length;
    for (const lbl of labels) {
      rawScores[lbl] = (rawScores[lbl] ?? 0) + perLabel;
    }
  }

  const total = Object.values(rawScores).reduce((acc, v) => acc + v, 0);
  if (total <= 0) {
    return { weights: {}, topLabels: [] };
  }

  const sorted = Object.entries(rawScores).sort((a, b) => b[1] - a[1]);
  const weights: Record<string, number> = {};
  for (const [label, score] of sorted) {
    weights[label] = Number((score / total).toFixed(4));
  }
  return {
    weights,
    topLabels: sorted.slice(0, 5).map(([label]) => label),
  };
}

function computePlaytimeTiers(
  dna: SteamDnaResponse,
  isLifetime: boolean,
  games: SteamTopGame[],
  windowTotalHours: number,
): PlaytimeTierStat[] {
  const tierHours: Record<string, number> = {};
  const tierCounts: Record<string, number> = {};
  for (const def of TIER_DEFS) {
    tierHours[def.key] = 0;
    tierCounts[def.key] = 0;
  }

  for (const g of games) {
    const hrs = Math.max(0, Number(g.playtime_hours || 0));
    if (hrs <= 0) continue;
    for (const def of TIER_DEFS) {
      if (hrs >= def.minHours && hrs < def.maxHours) {
        tierHours[def.key] += hrs;
        tierCounts[def.key] += 1;
        break;
      }
    }
  }

  const dist = dna.playtime_distribution ?? {};
  const hasLifetimeDist =
    isLifetime && Object.values(dist).some((v) => typeof v === 'number' && v > 0);
  const gamesSumHours = Object.values(tierHours).reduce((acc, v) => acc + v, 0);
  const denomHours = windowTotalHours > 0 ? windowTotalHours : gamesSumHours;

  return TIER_DEFS.map((def) => {
    if (hasLifetimeDist) {
      const share = Math.max(0, Math.min(1, Number(dist[def.key] ?? 0)));
      return {
        key: def.key,
        label: def.label,
        rangeLabel: def.rangeLabel,
        color: def.color,
        share,
        hours: share * Math.max(0, Number(dna.total_playtime_hours || 0)),
        gamesCount: tierCounts[def.key],
      };
    }

    const hrs = tierHours[def.key];
    const share = denomHours > 0 ? Math.max(0, Math.min(1, hrs / denomHours)) : 0;
    return {
      key: def.key,
      label: def.label,
      rangeLabel: def.rangeLabel,
      color: def.color,
      share,
      hours: hrs,
      gamesCount: tierCounts[def.key],
    };
  });
}

function parseDirectiveTitles(directiveText: string, headingPrefix: string): Set<string> {
  const out = new Set<string>();
  if (!directiveText) return out;
  const escaped = headingPrefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = directiveText.match(new RegExp(`${escaped}[^:]*:\\s*([^\\n]+)`, 'i'));
  if (!match || !match[1]) return out;

  // Format in writers_room._format_dna_summary: "Title A (120.5h), Title B (45.0h)"
  for (const part of match[1].split(/,\s*(?=[A-Za-z0-9])/)) {
    const cleaned = part.replace(/\s*\([^)]*\)\s*$/, '').trim().toLowerCase();
    if (cleaned) out.add(cleaned);
  }
  return out;
}

export function resolveSteamDnaHorizon(
  dna: SteamDnaResponse,
  horizonKey: HorizonKey,
  fallbackDirectiveText = '',
): ResolvedHorizonView {
  const lifetimeHours = Math.max(0, Number(dna.total_playtime_hours || 0));
  const lifetimeGamesCount = Math.max(
    0,
    Number(dna.total_games_analyzed || dna.top_games?.length || 0),
  );

  const horizonSummaries: HorizonSummaryCardData[] = STEAM_HORIZONS.map((meta) => {
    const w = dna.activity_windows?.[meta.key];
    const wGames = getWindowGames(dna, meta, w);
    const gamesCount = getWindowGamesCount(dna, meta, w, wGames);
    const totalPlaytimeHours = getWindowPlaytimeHours(dna, meta, w, wGames);
    const recent2WeeksHours = getWindowRecent2WeeksHours(w, wGames);
    const shareOfLifetimePct =
      meta.key === 'lifetime'
        ? lifetimeHours > 0
          ? 100
          : 0
        : lifetimeHours > 0
          ? Math.min(100, (totalPlaytimeHours / lifetimeHours) * 100)
          : 0;
    const derivedGenres =
      Array.isArray(w?.top_genres) && w.top_genres.length > 0
        ? w.top_genres
        : meta.key === 'lifetime' && Array.isArray(dna.top_genres) && dna.top_genres.length > 0
          ? dna.top_genres
          : computeWindowLabelWeights(wGames, (g) => g.genres).topLabels;

    return {
      meta,
      gamesCount,
      totalPlaytimeHours,
      recent2WeeksHours,
      shareOfLifetimePct,
      topGenres: derivedGenres.slice(0, 4),
    };
  });

  const activeMeta =
    STEAM_HORIZONS.find((h) => h.key === horizonKey) ?? STEAM_HORIZONS[STEAM_HORIZONS.length - 1];
  const isLifetime = activeMeta.key === 'lifetime';
  const activeWin = dna.activity_windows?.[activeMeta.key];
  const games = getWindowGames(dna, activeMeta, activeWin);
  const gamesCount = getWindowGamesCount(dna, activeMeta, activeWin, games);
  const totalPlaytimeHours = getWindowPlaytimeHours(dna, activeMeta, activeWin, games);
  const recent2WeeksHours = getWindowRecent2WeeksHours(activeWin, games);

  const shareOfLifetimePct = isLifetime
    ? lifetimeHours > 0
      ? 100
      : 0
    : lifetimeHours > 0
      ? Math.min(100, (totalPlaytimeHours / lifetimeHours) * 100)
      : 0;

  const avgHoursPerGame = gamesCount > 0 ? totalPlaytimeHours / gamesCount : 0;

  let latestPlayedDate: string | null = null;
  let latestDaysAgo: number | null = null;
  for (const g of games) {
    const lp = (g.last_played_date ?? '').trim();
    const dAgo = typeof g.days_since_played === 'number' ? g.days_since_played : null;
    if (dAgo !== null && (latestDaysAgo === null || dAgo < latestDaysAgo)) {
      latestDaysAgo = dAgo;
      if (lp) latestPlayedDate = lp;
    } else if (lp && (!latestPlayedDate || lp > latestPlayedDate) && latestDaysAgo === null) {
      latestPlayedDate = lp;
    }
  }

  let genreWeights: Record<string, number> = {};
  let topGenres: string[] = [];
  if (isLifetime) {
    genreWeights = dna.genre_weights ?? {};
    topGenres = dna.top_genres ?? [];
  } else if (activeWin?.genre_weights && Object.keys(activeWin.genre_weights).length > 0) {
    genreWeights = activeWin.genre_weights;
    topGenres =
      activeWin.top_genres && activeWin.top_genres.length > 0
        ? activeWin.top_genres
        : Object.keys(activeWin.genre_weights).slice(0, 5);
  } else {
    const computed = computeWindowLabelWeights(games, (g) => g.genres);
    genreWeights = computed.weights;
    topGenres =
      activeWin?.top_genres && activeWin.top_genres.length > 0
        ? activeWin.top_genres
        : computed.topLabels;
  }

  let categoryWeights: Record<string, number> = {};
  let topCategories: string[] = [];
  if (isLifetime) {
    categoryWeights = dna.category_weights ?? {};
    topCategories = dna.top_categories ?? [];
  } else if (activeWin?.category_weights && Object.keys(activeWin.category_weights).length > 0) {
    categoryWeights = activeWin.category_weights;
    topCategories = Object.keys(activeWin.category_weights).slice(0, 4);
  } else {
    const computed = computeWindowLabelWeights(games, (g) => g.categories);
    categoryWeights = computed.weights;
    topCategories = computed.topLabels.slice(0, 4);
  }

  const playtimeTiers = computePlaytimeTiers(dna, isLifetime, games, totalPlaytimeHours);
  const masteredTier = playtimeTiers.find((t) => t.key === 'mastered_over_100h');
  const masteredSharePct = (masteredTier?.share ?? 0) * 100;

  const directiveSource = dna.live_writers_room_dna_directive || fallbackDirectiveText || '';
  const anchorTitleSet = parseDirectiveTitles(directiveSource, 'Most-Played Core Anchor Games');
  const spotlightTitleSet = parseDirectiveTitles(
    directiveSource,
    'Session Spotlight Cross-Pollination Subset',
  );

  return {
    meta: activeMeta,
    isLifetime,
    totalPlaytimeHours,
    lifetimePlaytimeHours: lifetimeHours,
    shareOfLifetimePct,
    recent2WeeksHours,
    gamesCount,
    lifetimeGamesCount,
    avgHoursPerGame,
    masteredSharePct,
    latestPlayedDate,
    latestDaysAgo,
    genreWeights,
    topGenres,
    categoryWeights,
    topCategories,
    playtimeTiers,
    games,
    horizonSummaries,
    anchorTitleSet,
    spotlightTitleSet,
  };
}
