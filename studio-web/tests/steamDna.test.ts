import { describe, expect, it } from 'vitest';
import type { SteamDnaResponse } from '../src/api/types';
import { resolveSteamDnaHorizon } from '../src/domain/steamDna';

function makeSyntheticDna(): SteamDnaResponse {
  return {
    deliberation: 'Holistic library synthesis across multi-year play history.',
    player_id: 'synth_user_01',
    profile_tag: 'default',
    total_playtime_hours: 1000,
    total_games_analyzed: 50,
    top_genres: ['Strategy', 'Simulation', 'RPG'],
    genre_weights: { Strategy: 0.55, Simulation: 0.3, RPG: 0.15 },
    top_categories: ['Single-player', 'Steam Cloud'],
    category_weights: { 'Single-player': 0.7, 'Steam Cloud': 0.3 },
    playtime_distribution: {
      mastered_over_100h: 0.6,
      dedicated_50_to_100h: 0.2,
      engaged_10_to_50h: 0.12,
      casual_2_to_10h: 0.06,
      dabbled_under_2h: 0.02,
    },
    top_games: [
      {
        appid: 1001,
        name: 'Synthetic Grand Strategy',
        playtime_hours: 400,
        playtime_share: 0.4,
        last_played_date: '2025-05-01',
        days_since_played: 500,
        genres: ['Strategy'],
        categories: ['Single-player'],
      },
      {
        appid: 1002,
        name: 'Synthetic Co-op Tinkerer',
        playtime_hours: 40,
        playtime_share: 0.04,
        playtime_2weeks_hours: 12,
        last_played_date: '2026-10-28',
        days_since_played: 3,
        genres: ['Puzzle', 'Action'],
        categories: ['Co-op', 'Online Co-op'],
      },
    ],
    activity_windows: {
      last_30_days: {
        window_key: 'last_30_days',
        label: 'Last 30 Days',
        days: 30,
        games_count: 2,
        total_playtime_hours: 45,
        recent_2weeks_hours: 14.5,
        top_genres: ['Puzzle', 'Action'],
        games: [
          {
            appid: 1002,
            name: 'Synthetic Co-op Tinkerer',
            playtime_hours: 40,
            playtime_share: 0.04,
            playtime_2weeks_hours: 12,
            last_played_date: '2026-10-28',
            days_since_played: 3,
            genres: ['Puzzle', 'Action'],
            categories: ['Co-op', 'Online Co-op'],
          },
          {
            appid: 1003,
            name: 'Synthetic Arcade Sprint',
            playtime_hours: 5,
            playtime_share: 0.005,
            playtime_2weeks_hours: 2.5,
            last_played_date: '2026-10-20',
            days_since_played: 11,
            genres: ['Action'],
            categories: ['Full controller support'],
          },
        ],
      },
      last_90_days: {
        window_key: 'last_90_days',
        label: 'Last 90 Days',
        days: 90,
        games_count: 3,
        total_playtime_hours: 165,
        recent_2weeks_hours: 14.5,
        top_genres: ['Simulation', 'Action'],
        games: [
          {
            appid: 1004,
            name: 'Synthetic Colony Grid',
            playtime_hours: 120,
            playtime_share: 0.12,
            last_played_date: '2026-09-01',
            days_since_played: 60,
            genres: ['Simulation'],
            categories: ['Single-player', 'Steam Workshop'],
          },
          {
            appid: 1002,
            name: 'Synthetic Co-op Tinkerer',
            playtime_hours: 40,
            playtime_share: 0.04,
            playtime_2weeks_hours: 12,
            last_played_date: '2026-10-28',
            days_since_played: 3,
            genres: ['Puzzle', 'Action'],
            categories: ['Co-op', 'Online Co-op'],
          },
          {
            appid: 1003,
            name: 'Synthetic Arcade Sprint',
            playtime_hours: 5,
            playtime_share: 0.005,
            playtime_2weeks_hours: 2.5,
            last_played_date: '2026-10-20',
            days_since_played: 11,
            genres: ['Action'],
            categories: ['Full controller support'],
          },
        ],
      },
      last_1_year: {
        window_key: 'last_1_year',
        label: 'Last 1 Year',
        days: 365,
        games_count: 12,
        total_playtime_hours: 420,
        recent_2weeks_hours: 14.5,
        top_genres: ['Simulation', 'RPG'],
        games: [
          {
            appid: 1005,
            name: 'Synthetic Chronicle RPG',
            playtime_hours: 255,
            playtime_share: 0.255,
            last_played_date: '2026-03-10',
            days_since_played: 235,
            genres: ['RPG'],
            categories: ['Single-player', 'Steam Cloud'],
          },
        ],
      },
      lifetime: {
        window_key: 'lifetime',
        label: 'Lifetime',
        days: null,
        games_count: 50,
        total_playtime_hours: 1000,
        recent_2weeks_hours: 14.5,
        top_genres: ['Strategy', 'Simulation', 'RPG'],
        games: [
          {
            appid: 1001,
            name: 'Synthetic Grand Strategy',
            playtime_hours: 400,
            playtime_share: 0.4,
            last_played_date: '2025-05-01',
            days_since_played: 500,
            genres: ['Strategy'],
            categories: ['Single-player'],
          },
        ],
      },
    },
    pacing_preference: 'balanced',
    pacing_score: -0.15,
    challenge_tolerance: 0.78,
    thematic_biases: ['sci-fi'],
    thematic_weights: { 'sci-fi': 1.0 },
    mechanical_biases: { 'network routing': 0.9 },
    preferred_palette_mood: 'crisp_industrial',
    palette_weights: { crisp_industrial: 1.0 },
    created_at_utc: '2026-10-31T12:00:00Z',
    ingestion_provenance: {},
    live_writers_room_dna_directive:
      '- Most-Played Core Anchor Games: Synthetic Grand Strategy (400.0h)\n- Session Spotlight Cross-Pollination Subset: Synthetic Co-op Tinkerer (40.0h)',
  };
}

describe('resolveSteamDnaHorizon', () => {
  it('computes distinct games_count, games list, mastery tiers, genre weights, and category weights across 30d, 90d, 1yr, and lifetime', () => {
    const dna = makeSyntheticDna();

    const v30 = resolveSteamDnaHorizon(dna, 'last_30_days');
    const v90 = resolveSteamDnaHorizon(dna, 'last_90_days');
    const v1y = resolveSteamDnaHorizon(dna, 'last_1_year');
    const vLife = resolveSteamDnaHorizon(dna, 'lifetime');

    // Games count & playtime update per horizon via backend `games_count` & `games`
    expect(v30.gamesCount).toBe(2);
    expect(v90.gamesCount).toBe(3);
    expect(v1y.gamesCount).toBe(12);
    expect(vLife.gamesCount).toBe(50);

    expect(v30.totalPlaytimeHours).toBe(45);
    expect(v90.totalPlaytimeHours).toBe(165);
    expect(v1y.totalPlaytimeHours).toBe(420);
    expect(vLife.totalPlaytimeHours).toBe(1000);

    expect(v30.shareOfLifetimePct).toBeCloseTo(4.5, 2);
    expect(v30.latestPlayedDate).toBe('2026-10-28');
    expect(v30.latestDaysAgo).toBe(3);

    // Genre & category weights dynamically reflect the active window's games rather than staying frozen on lifetime
    expect(v30.genreWeights.Action).toBeGreaterThan(0.4);
    expect(v30.genreWeights.Strategy).toBeUndefined();
    expect(vLife.genreWeights.Strategy).toBe(0.55);

    expect(v30.categoryWeights['Co-op']).toBeGreaterThan(0.3);
    expect(vLife.categoryWeights['Single-player']).toBe(0.7);

    // 5-tier mastery depth updates per horizon: 30d has 0% >100h, 90d has 120h/165h (~72.7%) >100h, lifetime has 60%
    expect(v30.masteredSharePct).toBe(0);
    expect(v90.masteredSharePct).toBeCloseTo((120 / 165) * 100, 1);
    expect(vLife.masteredSharePct).toBeCloseTo(60, 1);

    // Parses anchor and spotlight sets from directive
    expect(v30.anchorTitleSet.has('synthetic grand strategy')).toBe(true);
    expect(v30.spotlightTitleSet.has('synthetic co-op tinkerer')).toBe(true);
  });
});
