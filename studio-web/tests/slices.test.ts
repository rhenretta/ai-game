import { describe, expect, it } from 'vitest';
import {
  filterBoardItemsBySlice,
  normalizeGameCatalog,
  normalizeSpecAndDirections,
  partitionTranscriptBySlice,
  resolveSliceContext,
  resolveSliceVerificationState,
} from '../src/domain/slices';
import type { GameBibleResponse, ProductionBoardItem, VerticalSliceSpec, WritersRoomUtterance } from '../src/api/types';

function makeBoardItem(overrides: Partial<ProductionBoardItem>): ProductionBoardItem {
  return {
    deliberation: 'Synthetic item deliberation.',
    item_id: 'item_default',
    title: 'Synthetic Work Item',
    kind: 'feature_code',
    spec: 'Implement deterministic state step.',
    proof_spec: 'Passes headless verification',
    target_modules: ['scripts/core_state.gd'],
    depends_on: [],
    state: 'approved',
    attempt_count: 1,
    slice_id: 'slice_0_foundation',
    ...overrides,
  };
}

describe('slice domain utilities', () => {
  it('resolves active and selected slices from roadmap and board items', () => {
    const items = [
      makeBoardItem({ item_id: 'item_0', slice_id: 'slice_0_foundation', state: 'approved' }),
      makeBoardItem({ item_id: 'item_1', slice_id: 'slice_1_expansion', state: 'in_progress' }),
      makeBoardItem({ item_id: 'item_2', slice_id: 'slice_2_polish', state: 'ready' }),
    ];

    const bible: GameBibleResponse = {
      deliberation: 'Loaded synthetic game bible.',
      has_bible: true,
      cartridge_id: 'candidate_synthetic_01',
      status: 'active',
      game_title: 'Synthetic Cartridge',
      core_genre: 'Systems Simulation',
      active_slice_index: 1,
      slice_roadmap: [
        { slice_id: 'slice_0_foundation', slice_index: 0, title: 'Slice 0', status: 'completed' },
        { slice_id: 'slice_1_expansion', slice_index: 1, title: 'Slice 1', status: 'active' },
      ],
      production_board: {
        deliberation: 'Synthetic board summary.',
        cartridge_id: 'candidate_synthetic_01',
        design_revision: 1,
        total_items: items.length,
        counts_by_state: { approved: 1, in_progress: 1, ready: 1 },
        is_drained: false,
        items,
      },
    };

    const resolvedDefault = resolveSliceContext(bible, null);
    expect(resolvedDefault.slices.map((s) => s.slice_id)).toEqual([
      'slice_0_foundation',
      'slice_1_expansion',
      'slice_2_polish',
    ]);
    // Slice 2 has an unapproved `ready` item so it becomes the highest active slice
    expect(resolvedDefault.currentSlice?.slice_id).toBe('slice_2_polish');
    expect(resolvedDefault.selectedSlice?.slice_id).toBe('slice_2_polish');

    const resolvedAll = resolveSliceContext(bible, 'all');
    expect(resolvedAll.selectedSlice).toBeNull();

    const resolvedExplicit = resolveSliceContext(bible, 'slice_0_foundation');
    expect(resolvedExplicit.selectedSlice?.slice_id).toBe('slice_0_foundation');
  });

  it('filters cancelled items and scopes board items to the selected slice', () => {
    const slice0: VerticalSliceSpec = {
      slice_id: 'slice_0_foundation',
      slice_index: 0,
      title: 'Slice 0',
      status: 'completed',
    };
    const slice1: VerticalSliceSpec = {
      slice_id: 'slice_1_expansion',
      slice_index: 1,
      title: 'Slice 1',
      status: 'active',
    };

    const items: ProductionBoardItem[] = [
      makeBoardItem({ item_id: 'i_unscoped', slice_id: '', state: 'approved' }),
      makeBoardItem({ item_id: 'i_s0', slice_id: 'slice_0_foundation', state: 'approved' }),
      makeBoardItem({ item_id: 'i_s1', slice_id: 'slice_1_expansion', state: 'in_progress' }),
      makeBoardItem({ item_id: 'i_cancelled', slice_id: 'slice_1_expansion', state: 'cancelled' }),
    ];

    expect(filterBoardItemsBySlice(items, null, 2).map((i) => i.item_id)).toEqual([
      'i_unscoped',
      'i_s0',
      'i_s1',
    ]);
    expect(filterBoardItemsBySlice(items, slice0, 2).map((i) => i.item_id)).toEqual([
      'i_unscoped',
      'i_s0',
    ]);
    expect(filterBoardItemsBySlice(items, slice1, 2).map((i) => i.item_id)).toEqual(['i_s1']);
  });

  it('partitions Writers Room transcript across slice boundaries', () => {
    const slices: VerticalSliceSpec[] = [
      { slice_id: 'slice_0_foundation', slice_index: 0, title: 'Slice 0', status: 'completed' },
      { slice_id: 'slice_1_expansion', slice_index: 1, title: 'Slice 1', status: 'active' },
    ];
    const transcript: WritersRoomUtterance[] = [
      {
        utterance_id: 'utt_0_pitch',
        turn_index: 1,
        speaker_role: 'Creative Director',
        message: 'Initial foundation scope.',
      },
      {
        utterance_id: 'utt_slice_1_reflection',
        turn_index: 4,
        speaker_role: 'Systems Architect',
        message: 'Reflecting before Slice 1.',
        bible_updates_summary: ['slice_reflection:slice_1'],
      },
      {
        utterance_id: 'utt_1_followup',
        turn_index: 5,
        speaker_role: 'Art Director',
        message: 'Palette adjustments for Slice 1.',
      },
    ];

    const grouped = partitionTranscriptBySlice(transcript, slices);
    expect(grouped.get('slice_0_foundation')?.map((u) => u.utterance_id)).toEqual(['utt_0_pitch']);
    expect(grouped.get('slice_1_expansion')?.map((u) => u.utterance_id)).toEqual([
      'utt_slice_1_reflection',
      'utt_1_followup',
    ]);
  });

  it('places ExecutiveProducer slice-kickoff directives at the start of the slice they initiate', () => {
    const slices: VerticalSliceSpec[] = [
      { slice_id: 'slice_0_alpha', slice_index: 0, title: 'Slice 0', status: 'completed' },
      { slice_id: 'slice_1_beta', slice_index: 1, title: 'Slice 1', status: 'completed' },
      { slice_id: 'slice_2_gamma', slice_index: 2, title: 'Slice 2', status: 'active' },
    ];
    const transcript: WritersRoomUtterance[] = [
      {
        utterance_id: 'utt_s0_open',
        turn_index: 0,
        speaker_role: 'CreativeDirector',
        spoken_dialogue: 'Opening concept pitch.',
      },
      {
        utterance_id: 'utt_s0_mid_steer',
        turn_index: 1,
        speaker_role: 'ExecutiveProducer',
        spoken_dialogue: 'Mid-slice 0 direction.',
        bible_updates_summary: ['Executive Producer creative direction injected'],
      },
      {
        utterance_id: 'utt_s0_close',
        turn_index: 2,
        speaker_role: 'GameplayDesigner',
        spoken_dialogue: 'Incorporated mid-slice 0 direction.',
      },
      {
        utterance_id: 'utt_s1_kickoff_wish',
        turn_index: 3,
        speaker_role: 'ExecutiveProducer',
        spoken_dialogue: 'Approved playtest wish before Slice 1.',
        bible_updates_summary: ['Dispatched 1 Blind Playtester wish(es)'],
      },
      {
        utterance_id: 'utt_s1_qa_capture',
        turn_index: 4,
        speaker_role: 'QATestEngineer',
        spoken_dialogue: 'Empirical capture of Slice 0.',
        bible_updates_summary: ['slice_reflection_capture:slice_0_alpha'],
      },
      {
        utterance_id: 'utt_s1_strategy',
        turn_index: 5,
        speaker_role: 'CreativeDirector',
        spoken_dialogue: 'Locked Slice 1 strategy.',
        bible_updates_summary: ['next_slice_strategy:slice_1_beta'],
      },
      {
        utterance_id: 'utt_s2_kickoff_feedback',
        turn_index: 6,
        speaker_role: 'ExecutiveProducer',
        spoken_dialogue: 'Kickoff feedback for Slice 2 before QA finishes.',
        bible_updates_summary: ['Executive Producer slice feedback injected'],
      },
    ];

    const grouped = partitionTranscriptBySlice(transcript, slices);
    expect(grouped.get('slice_0_alpha')?.map((u) => u.utterance_id)).toEqual([
      'utt_s0_open',
      'utt_s0_mid_steer',
      'utt_s0_close',
    ]);
    expect(grouped.get('slice_1_beta')?.map((u) => u.utterance_id)).toEqual([
      'utt_s1_kickoff_wish',
      'utt_s1_qa_capture',
      'utt_s1_strategy',
    ]);
    expect(grouped.get('slice_2_gamma')?.map((u) => u.utterance_id)).toEqual([
      'utt_s2_kickoff_feedback',
    ]);
  });

  it('deduplicates bracketed specialist directions from raw item specs', () => {
    const directionText = 'Ensure contrast ratio exceeds 4.5:1 on primary HUD counters and labels.';
    const rawSpec = `Build primary HUD overlay.\n\n[Art Director Direction] ${directionText}`;
    const result = normalizeSpecAndDirections(rawSpec, [
      {
        deliberation: 'HUD legibility requires strong contrast.',
        specialist_role: 'Art Director',
        direction: directionText,
      },
    ]);

    expect(result.cleanSpec).toBe('Build primary HUD overlay.');
    expect(result.directions).toEqual([
      {
        role: 'Art Director',
        text: directionText,
        deliberation: 'HUD legibility requires strong contrast.',
      },
    ]);

    // Also handles string-shaped shaping_notes prefixed with `[RoleName Direction]: ...`
    const stringNote =
      '[GameplaySystemsEngineer Direction]: Define clean node hierarchy and signal contracts without frame spikes.';
    const rawWithStringDir = `Implement chunk loader.\n\n${stringNote}`;
    const strResult = normalizeSpecAndDirections(rawWithStringDir, [stringNote]);
    expect(strResult.cleanSpec).toBe('Implement chunk loader.');
    expect(strResult.directions).toEqual([
      {
        role: 'GameplaySystemsEngineer',
        text: 'Define clean node hierarchy and signal contracts without frame spikes.',
        deliberation: undefined,
      },
    ]);
  });

  it('collapses the `active` staging mirror and flags distinct candidate folders sharing a title', () => {
    const normalized = normalizeGameCatalog([
      {
        cartridge_id: 'active',
        game_title: 'Synthetic Factory Alpha',
        core_genre: 'Simulation',
        status: 'locked',
        has_backlog: true,
        approved_items: 20,
        total_items: 20,
        updated_at_utc: '2026-10-30T18:00:00Z',
      },
      {
        cartridge_id: 'candidate_genesis_a1b2c3',
        game_title: 'Synthetic Factory Alpha',
        core_genre: 'Simulation',
        status: 'locked',
        has_backlog: true,
        approved_items: 20,
        total_items: 20,
        updated_at_utc: '2026-10-30T17:55:00Z',
      },
      {
        cartridge_id: 'candidate_genesis_clone1',
        game_title: 'Synthetic Prototype Beta',
        core_genre: 'Strategy',
        status: 'brainstorming',
        has_backlog: false,
        approved_items: 0,
        total_items: 0,
        updated_at_utc: '2026-10-20T00:00:00Z',
      },
      {
        cartridge_id: 'candidate_genesis_clone2',
        game_title: 'Synthetic Prototype Beta',
        core_genre: 'Strategy',
        status: 'brainstorming',
        has_backlog: false,
        approved_items: 0,
        total_items: 0,
        updated_at_utc: '2026-10-19T00:00:00Z',
      },
    ]);

    // `active` mirror is collapsed into canonical `candidate_genesis_a1b2c3`
    expect(normalized.map((g) => g.cartridge_id)).toEqual([
      'candidate_genesis_a1b2c3',
      'candidate_genesis_clone1',
      'candidate_genesis_clone2',
    ]);
    expect(normalized[0].isPromotedActive).toBe(true);
    expect(normalized[0].hasDuplicateTitle).toBe(false);

    // Distinct folders with the same title are flagged for shortId disambiguation
    expect(normalized[1].hasDuplicateTitle).toBe(true);
    expect(normalized[1].shortId).toBe('#clone1');
    expect(normalized[2].hasDuplicateTitle).toBe(true);
    expect(normalized[2].shortId).toBe('#clone2');
  });

  it('requires active slice completion and zero build errors before reporting isSliceVerified', () => {
    const drainedItems = [
      makeBoardItem({ item_id: 'i_0', slice_id: 'slice_0_foundation', state: 'approved' }),
      makeBoardItem({ item_id: 'i_1', slice_id: 'slice_1_expansion', state: 'approved' }),
    ];

    const unfinishedPlaytestBible: GameBibleResponse = {
      deliberation: 'Board items approved, awaiting Council Blind Playtest.',
      has_bible: true,
      cartridge_id: 'candidate_synthetic_playtest',
      status: 'in_production',
      game_title: 'Synthetic Systems Prototype',
      core_genre: 'Simulation',
      active_slice_index: 1,
      slice_roadmap: [
        { slice_id: 'slice_0_foundation', slice_index: 0, title: 'Slice 0', status: 'completed' },
        { slice_id: 'slice_1_expansion', slice_index: 1, title: 'Slice 1', status: 'active' },
      ],
      production_board: {
        deliberation: 'Drained board.',
        cartridge_id: 'candidate_synthetic_playtest',
        design_revision: 2,
        total_items: 2,
        counts_by_state: { approved: 2 },
        is_drained: true,
        items: drainedItems,
      },
    };

    // Unfinished explicit slice (`status: 'active'` while `bible.status === 'in_production'`) must NOT be verified
    const awaitingState = resolveSliceVerificationState(unfinishedPlaytestBible, false);
    expect(awaitingState.isBoardDrained).toBe(true);
    expect(awaitingState.hasUnfinishedExplicitSlice).toBe(true);
    expect(awaitingState.isAwaitingCouncilPlaytest).toBe(true);
    expect(awaitingState.isSliceVerified).toBe(false);

    // Completed slice with a build error must also NOT report verified
    const completedSlicesBible: GameBibleResponse = {
      ...unfinishedPlaytestBible,
      status: 'locked',
      slice_roadmap: [
        { slice_id: 'slice_0_foundation', slice_index: 0, title: 'Slice 0', status: 'completed' },
        { slice_id: 'slice_1_expansion', slice_index: 1, title: 'Slice 1', status: 'completed' },
      ],
    };
    expect(resolveSliceVerificationState(completedSlicesBible, true).isSliceVerified).toBe(false);

    // Once slice is completed (`locked`) and has zero build errors, `isSliceVerified` becomes true
    const verifiedState = resolveSliceVerificationState(completedSlicesBible, false);
    expect(verifiedState.hasUnfinishedExplicitSlice).toBe(false);
    expect(verifiedState.isAwaitingCouncilPlaytest).toBe(false);
    expect(verifiedState.isSliceVerified).toBe(true);
  });
});
