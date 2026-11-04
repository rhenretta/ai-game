import type {
  GameBibleResponse,
  GameCatalogEntry,
  ProductionBoardItem,
  ShapingNote,
  VerticalSliceSpec,
  WritersRoomUtterance,
} from '../api/types';

export interface NormalizedGameCatalogEntry extends GameCatalogEntry {
  isPromotedActive: boolean;
  shortId: string;
  hasDuplicateTitle: boolean;
}

function formatShortCartridgeId(cartridgeId: string): string {
  if (cartridgeId === 'active') return 'active';
  if (cartridgeId.startsWith('candidate_genesis_')) {
    return `#${cartridgeId.slice('candidate_genesis_'.length)}`;
  }
  if (cartridgeId.startsWith('candidate_')) {
    return `#${cartridgeId.slice('candidate_'.length)}`;
  }
  return cartridgeId;
}

function parseCatalogIsoMillis(raw?: string | null): number {
  if (!raw) return 0;
  const ts = Date.parse(raw);
  return Number.isFinite(ts) ? ts : 0;
}

/**
 * Deduplicates the `cartridges/active` staging mirror when its canonical `candidate_*`
 * directory is already present in `/api/cartridges`, marks any remaining distinct
 * folders that share the same `game_title` so the UI can disambiguate them by short ID,
 * and orders games strictly by most recently worked on (`updated_at_utc` DESC).
 */
export function normalizeGameCatalog(
  rawGames: GameCatalogEntry[],
  activeCandidateId?: string | null,
): NormalizedGameCatalogEntry[] {
  if (rawGames.length === 0) return [];

  const activeMirror = rawGames.find((g) => g.cartridge_id === 'active');
  const candidates = rawGames.filter((g) => g.cartridge_id !== 'active');

  let promotedCanonicalId: string | null = null;
  if (activeCandidateId && activeCandidateId !== 'active' && candidates.some((c) => c.cartridge_id === activeCandidateId)) {
    promotedCanonicalId = activeCandidateId;
  } else if (activeMirror) {
    const mirrorMatch = candidates.find(
      (c) =>
        c.game_title === activeMirror.game_title &&
        c.approved_items === activeMirror.approved_items &&
        c.total_items === activeMirror.total_items,
    );
    if (mirrorMatch) {
      promotedCanonicalId = mirrorMatch.cartridge_id;
    }
  }

  // Keep `active` only when no canonical `candidate_*` folder represents it on disk
  const baseList = promotedCanonicalId ? candidates : rawGames;

  const titleCounts = new Map<string, number>();
  for (const g of baseList) {
    const key = g.game_title.trim();
    titleCounts.set(key, (titleCounts.get(key) ?? 0) + 1);
  }

  const normalized = baseList.map((g) => {
    const isPromoted = g.cartridge_id === promotedCanonicalId || g.cartridge_id === 'active';
    let effectiveUpdatedAt = g.updated_at_utc;
    if (
      g.cartridge_id === promotedCanonicalId &&
      activeMirror &&
      activeMirror.game_title === g.game_title &&
      parseCatalogIsoMillis(activeMirror.updated_at_utc) > parseCatalogIsoMillis(g.updated_at_utc)
    ) {
      effectiveUpdatedAt = activeMirror.updated_at_utc;
    }
    return {
      ...g,
      updated_at_utc: effectiveUpdatedAt,
      isPromotedActive: isPromoted,
      shortId: formatShortCartridgeId(g.cartridge_id),
      hasDuplicateTitle: (titleCounts.get(g.game_title.trim()) ?? 0) > 1,
    };
  });

  // Order catalog strictly by most recently worked on (`updated_at_utc` DESC)
  normalized.sort(
    (a, b) => parseCatalogIsoMillis(b.updated_at_utc) - parseCatalogIsoMillis(a.updated_at_utc),
  );

  return normalized;
}

export interface ResolvedSliceContext {
  slices: VerticalSliceSpec[];
  currentSlice: VerticalSliceSpec | null;
  selectedSlice: VerticalSliceSpec | null; // null when `selectedSliceId === 'all'`
}

function parseSliceIndexFromId(sliceId: string, fallbackIndex: number): number {
  if (!sliceId.startsWith('slice_')) return fallbackIndex;
  const parts = sliceId.split('_');
  const parsed = Number(parts[1]);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallbackIndex;
}

export function resolveSliceContext(
  bible: GameBibleResponse | null | undefined,
  selectedSliceId: string | null,
): ResolvedSliceContext {
  const byId = new Map<string, VerticalSliceSpec>();
  const roadmap = bible?.slice_roadmap ?? [];

  for (const s of roadmap) {
    if (!s) continue;
    const sid = String(s.slice_id || `slice_${s.slice_index ?? byId.size}`).trim();
    const sidx = typeof s.slice_index === 'number' ? s.slice_index : byId.size;
    byId.set(sid, {
      ...s,
      slice_id: sid,
      slice_index: sidx,
      title: s.title || `Slice #${sidx}`,
      status: String(s.status || 'active').toLowerCase(),
    });
  }

  const allItems = bible?.production_board?.items ?? [];
  for (const it of allItems) {
    const sid = (it?.slice_id ?? '').trim();
    if (sid && !byId.has(sid)) {
      const idx = parseSliceIndexFromId(sid, byId.size);
      byId.set(sid, {
        slice_id: sid,
        slice_index: idx,
        title: `Slice #${idx}`,
        status: 'active',
      });
    }
  }

  if (byId.size === 0 && bible?.has_bible) {
    byId.set('slice_0', {
      slice_id: 'slice_0',
      slice_index: 0,
      title: 'Slice #0: Core Foundation',
      slice_objective: bible.first_60_seconds_hook || '',
      status: 'active',
    });
  }

  const slices = Array.from(byId.values()).sort((a, b) => a.slice_index - b.slice_index);
  if (slices.length === 0) {
    return { slices: [], currentSlice: null, selectedSlice: null };
  }

  let currentSlice = slices[slices.length - 1];
  if (typeof bible?.active_slice_index === 'number') {
    const match = slices.find((s) => s.slice_index === bible.active_slice_index);
    if (match) currentSlice = match;
  }

  const highestActive = [...slices].reverse().find(
    (s) =>
      s.status === 'active' ||
      s.status === 'in_progress' ||
      allItems.some((it) => it.slice_id === s.slice_id && it.state !== 'approved' && it.state !== 'cancelled'),
  );
  if (highestActive && highestActive.slice_index >= currentSlice.slice_index) {
    currentSlice = highestActive;
  }

  if (selectedSliceId === 'all') {
    return { slices, currentSlice, selectedSlice: null };
  }

  const explicit = selectedSliceId ? slices.find((s) => s.slice_id === selectedSliceId) : undefined;
  return {
    slices,
    currentSlice,
    selectedSlice: explicit ?? currentSlice,
  };
}

export function filterBoardItemsBySlice(
  items: ProductionBoardItem[],
  slice: VerticalSliceSpec | null,
  totalSlicesCount: number,
): ProductionBoardItem[] {
  const activeItems = items.filter((it) => it.state !== 'cancelled');
  if (!slice) return activeItems;

  return activeItems.filter((it) => {
    const sid = (it.slice_id ?? '').trim();
    if (sid === slice.slice_id) return true;
    if (!sid && slice.slice_index === 0) return true;
    if (totalSlicesCount <= 1 && slice.slice_index === 0) return true;
    const itemIdx = parseSliceIndexFromId(sid, -1);
    return itemIdx === slice.slice_index;
  });
}

export function partitionTranscriptBySlice(
  transcript: WritersRoomUtterance[],
  slices: VerticalSliceSpec[],
): Map<string, WritersRoomUtterance[]> {
  const bySlice = new Map<string, WritersRoomUtterance[]>();
  for (const s of slices) {
    bySlice.set(s.slice_id, []);
  }
  if (slices.length === 0 || transcript.length === 0) {
    return bySlice;
  }

  const isProducerTurn = (utt: WritersRoomUtterance): boolean =>
    utt.speaker_role === 'ExecutiveProducer' || utt.speaker_role === 'UserSteering';

  let activeSlice = slices[0];
  for (let i = 0; i < transcript.length; i += 1) {
    const u = transcript[i];
    if (u.slice_id && bySlice.has(u.slice_id)) {
      const target = slices.find((s) => s.slice_id === u.slice_id);
      if (target) activeSlice = target;
      bySlice.get(u.slice_id)?.push(u);
      continue;
    }

    const uid = u.utterance_id ?? '';
    const summaryJoined = (u.bible_updates_summary ?? []).join(' ');

    // If the Executive Producer explicitly injected next-slice resume feedback after prior specialist turns,
    // advance immediately so the kickoff directive opens the new slice even before QA filmstrip capture completes.
    if (
      isProducerTurn(u) &&
      summaryJoined.includes('Executive Producer slice feedback injected')
    ) {
      const currentBucket = bySlice.get(activeSlice.slice_id) ?? [];
      const hasSpecialistHistory = currentBucket.some((prev) => !isProducerTurn(prev));
      const immediateNext = slices.find((s) => s.slice_index > activeSlice.slice_index);
      if (hasSpecialistHistory && immediateNext) {
        activeSlice = immediateNext;
        bySlice.get(activeSlice.slice_id)?.push(u);
        continue;
      }
    }

    let nextBoundarySlice: VerticalSliceSpec | null = null;
    for (const s of slices) {
      if (s.slice_index <= activeSlice.slice_index) continue;
      const k = s.slice_index;
      const prevSlice = slices.find((cand) => cand.slice_index === k - 1);
      const isNextBoundary =
        uid.includes(`_slice_${k}_reflection`) ||
        uid.includes(`_slice_${k}_strategy`) ||
        summaryJoined.includes(`slice_reflection:slice_${k}`) ||
        summaryJoined.includes(`next_slice_strategy:${s.slice_id}`) ||
        summaryJoined.includes(s.slice_id) ||
        (summaryJoined.includes('slice_reflection_capture:') &&
          Boolean(prevSlice && summaryJoined.includes(prevSlice.slice_id)));

      if (isNextBoundary) {
        nextBoundarySlice = s;
        break;
      }
    }

    if (nextBoundarySlice) {
      // Carry any contiguous trailing ExecutiveProducer / UserSteering kickoff directives
      // at the end of the prior slice into the head of the newly opened slice.
      const prevBucket = bySlice.get(activeSlice.slice_id);
      const nextBucket = bySlice.get(nextBoundarySlice.slice_id);
      if (prevBucket && nextBucket && prevBucket.some((prev) => !isProducerTurn(prev))) {
        const trailingProducerTurns: WritersRoomUtterance[] = [];
        while (prevBucket.length > 0 && isProducerTurn(prevBucket[prevBucket.length - 1])) {
          const popped = prevBucket.pop();
          if (popped) trailingProducerTurns.unshift(popped);
        }
        nextBucket.push(...trailingProducerTurns);
      }
      activeSlice = nextBoundarySlice;
    }

    bySlice.get(activeSlice.slice_id)?.push(u);
  }

  return bySlice;
}

export interface NormalizedShapingNote {
  role: string;
  text: string;
  deliberation?: string;
}

/**
 * Normalizes specialist shaping directions and strips duplicated `[Role Direction]` appendices
 * from `spec` when already represented in `shaping_notes` (fixing the legacy card duplication bug).
 */
export function normalizeSpecAndDirections(
  rawSpec: string,
  shapingNotes?: Array<ShapingNote | string>,
): { cleanSpec: string; directions: NormalizedShapingNote[] } {
  const directions: Array<NormalizedShapingNote & { rawLine?: string }> = [];
  for (const entry of shapingNotes ?? []) {
    if (typeof entry === 'string') {
      const trimmed = entry.trim();
      if (trimmed) {
        const prefixMatch = trimmed.match(/^\[([^\]]+?)(?:\s+Direction)?\]:?\s*([\s\S]*)$/i);
        if (prefixMatch) {
          directions.push({
            role: prefixMatch[1].trim() || 'Specialist',
            text: prefixMatch[2].trim(),
            rawLine: trimmed,
          });
        } else {
          directions.push({ role: 'Specialist', text: trimmed, rawLine: trimmed });
        }
      }
    } else if (entry && typeof entry === 'object') {
      const text = (entry.direction || entry.note || '').trim();
      if (text) {
        directions.push({
          role: entry.specialist_role || entry.author_role || 'Specialist',
          text,
          deliberation: entry.deliberation,
        });
      }
    }
  }

  let cleanSpec = (rawSpec || '').trim();
  for (const dir of directions) {
    if (dir.rawLine && dir.rawLine.length > 20 && cleanSpec.includes(dir.rawLine)) {
      cleanSpec = cleanSpec.replace(dir.rawLine, '').trim();
      continue;
    }
    if (dir.text.length > 24 && cleanSpec.includes(dir.text)) {
      const bracketed = `[${dir.role} Direction] ${dir.text}`;
      const colonBracketed = `[${dir.role} Direction]: ${dir.text}`;
      if (cleanSpec.includes(bracketed)) {
        cleanSpec = cleanSpec.replace(bracketed, '').trim();
      } else if (cleanSpec.includes(colonBracketed)) {
        cleanSpec = cleanSpec.replace(colonBracketed, '').trim();
      }
    }
  }
  // Also extract any remaining `[RoleName Direction]: ...` paragraphs embedded directly in `cleanSpec`
  const paragraphs = cleanSpec.split(/\n{2,}/);
  const keptParagraphs: string[] = [];
  for (const para of paragraphs) {
    const trimmedPara = para.trim();
    const inlineMatch = trimmedPara.match(/^\[([^\]]+?)\s+Direction\]:?\s*([\s\S]+)$/i);
    if (inlineMatch) {
      const role = inlineMatch[1].trim() || 'Specialist';
      const text = inlineMatch[2].trim();
      const alreadyPresent = directions.some(
        (d) => d.role.toLowerCase() === role.toLowerCase() && d.text === text,
      );
      if (!alreadyPresent && text) {
        directions.push({ role, text });
      }
    } else if (trimmedPara) {
      keptParagraphs.push(trimmedPara);
    }
  }
  cleanSpec = keptParagraphs.join('\n\n').trim();

  return {
    cleanSpec: cleanSpec || rawSpec,
    directions: directions.map(({ role, text, deliberation }) => ({ role, text, deliberation })),
  };
}

export interface SliceVerificationState {
  hasUnfinishedExplicitSlice: boolean;
  isBoardDrained: boolean;
  isAwaitingCouncilPlaytest: boolean;
  isSliceVerified: boolean;
}

/**
 * Determines whether the active cartridge slice has genuinely completed end-to-end verification
 * (including its post-board Council Blind Playtest & slice lock), versus still awaiting its
 * Council Blind Playtest (`status === 'active' | 'in_progress'`) or stopped due to a build error.
 */
export function resolveSliceVerificationState(
  bible: GameBibleResponse | null | undefined,
  hasBuildError = false,
): SliceVerificationState {
  const proc = bible?.process_status;
  const isRunning = Boolean(proc?.build_actively_running || proc?.process_state === 'running');
  const isPaused = Boolean(proc?.is_paused);
  const isAwaitingGreenlight =
    proc?.process_state === 'awaiting_greenlight' ||
    bible?.orchestrator_state === 'AWAITING_GREENLIGHT';
  const isErrorState = Boolean(hasBuildError || proc?.process_state === 'error');

  const activeItems = (bible?.production_board?.items ?? []).filter(
    (it) => it.state !== 'cancelled',
  );
  const approvedItems = activeItems.filter((it) => it.state === 'approved').length;
  const isBoardDrained = activeItems.length > 0 && approvedItems >= activeItems.length;

  const bibleStatus = String(bible?.status ?? '').toLowerCase();
  const roadmap = bible?.slice_roadmap ?? [];
  const hasUnfinishedExplicitSlice =
    bibleStatus !== 'locked' &&
    roadmap.length > 0 &&
    roadmap.some((s) =>
      ['active', 'in_progress'].includes(String(s?.status ?? '').toLowerCase()),
    );

  const isAwaitingCouncilPlaytest =
    !isRunning &&
    !isPaused &&
    !isAwaitingGreenlight &&
    isBoardDrained &&
    hasUnfinishedExplicitSlice;

  const isSliceVerified =
    !isRunning &&
    !isPaused &&
    !isAwaitingGreenlight &&
    !isErrorState &&
    isBoardDrained &&
    !hasUnfinishedExplicitSlice;

  return {
    hasUnfinishedExplicitSlice,
    isBoardDrained,
    isAwaitingCouncilPlaytest,
    isSliceVerified,
  };
}
