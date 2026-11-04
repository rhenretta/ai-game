import type {
  CoProducerThread,
  CoProducerThreadsListResponse,
} from '../api/types';
import type { NormalizedGameCatalogEntry } from './slices';

export type StudioSessionSelection =
  | { kind: 'cartridge'; cartridgeId: string }
  | { kind: 'draft'; threadId: string };

export interface UnifiedCoProducerSession {
  key: string;
  selection: StudioSessionSelection;
  kind: 'cartridge' | 'draft';
  title: string;
  subtitle: string;
  statusLabel: string;
  statusTone: 'ok' | 'accent' | 'purple' | 'warn' | 'muted';
  isPromotedActive: boolean;
  messageCount: number;
  boardProgress?: string;
  updatedAtUtc: string;
  threadId?: string;
  cartridgeId?: string;
}

export interface FallbackCartridgeSessionHint {
  cartridgeId: string;
  title?: string;
  approved?: number;
  total?: number;
  status?: string;
}

function parseIsoMillis(raw?: string | null): number {
  if (!raw) return 0;
  const ts = Date.parse(raw);
  return Number.isFinite(ts) ? ts : 0;
}

function newerIsoTimestamp(a?: string | null, b?: string | null): string {
  const ma = parseIsoMillis(a);
  const mb = parseIsoMillis(b);
  if (mb > ma) return b || '';
  return a || b || '';
}

function resolveGameStatusMeta(
  rawStatus?: string | null,
  approved = 0,
  total = 0,
): { label: string; tone: UnifiedCoProducerSession['statusTone'] } {
  const normalized = String(rawStatus || '')
    .trim()
    .toLowerCase();
  if (
    normalized === 'running' ||
    normalized === 'building' ||
    normalized === 'synthesizing' ||
    normalized === 'in_progress'
  ) {
    return { label: 'Building', tone: 'ok' };
  }
  if (normalized === 'awaiting_greenlight') {
    return { label: 'Greenlight', tone: 'warn' };
  }
  if (normalized === 'paused') {
    return { label: 'Paused', tone: 'warn' };
  }
  if (normalized === 'error' || normalized === 'failed') {
    return { label: 'Attention', tone: 'warn' };
  }
  if (
    normalized === 'slice_complete' ||
    normalized === 'locked' ||
    normalized === 'completed' ||
    (total > 0 && approved >= total)
  ) {
    return { label: 'Playable', tone: 'ok' };
  }
  return { label: 'In Studio', tone: 'accent' };
}

export function isSameStudioSession(
  a: StudioSessionSelection | null | undefined,
  b: StudioSessionSelection | null | undefined,
): boolean {
  if (!a || !b || a.kind !== b.kind) return false;
  if (a.kind === 'draft' && b.kind === 'draft') {
    return a.threadId === b.threadId;
  }
  if (a.kind === 'cartridge' && b.kind === 'cartridge') {
    return a.cartridgeId === b.cartridgeId;
  }
  return false;
}

/**
 * Merges pre-build Concept Draft threads and built/building Cartridge games into a single
 * unified Co-Producer Session list so every game is treated as a session for the Co-Producer.
 */
export function buildUnifiedCoProducerSessions(params: {
  games: NormalizedGameCatalogEntry[];
  threadsData?: CoProducerThreadsListResponse | null;
  activeDraftThread?: CoProducerThread | null;
  fallbackCartridge?: FallbackCartridgeSessionHint | null;
}): UnifiedCoProducerSession[] {
  const { games, threadsData, activeDraftThread, fallbackCartridge } = params;

  // 1. Collect ungraduated draft sessions (active New Game concepts)
  const draftById = new Map<string, CoProducerThread>();
  if (activeDraftThread && activeDraftThread.scope_kind === 'draft') {
    draftById.set(activeDraftThread.thread_id, activeDraftThread);
  }
  for (const dt of threadsData?.draft_threads ?? []) {
    if (!dt?.thread_id) continue;
    if (!draftById.has(dt.thread_id)) {
      draftById.set(dt.thread_id, dt);
    }
  }

  const draftSessions: UnifiedCoProducerSession[] = [];
  for (const dt of draftById.values()) {
    if (dt.graduated_to_cartridge_id) continue;
    const workingTitle = dt.concept_brief?.working_title?.trim();
    const rawTitle = workingTitle || dt.title?.trim() || 'New Game Concept';
    const genre = dt.concept_brief?.core_genre?.trim();
    draftSessions.push({
      key: `draft:${dt.thread_id}`,
      selection: { kind: 'draft', threadId: dt.thread_id },
      kind: 'draft',
      title: rawTitle,
      subtitle: genre || 'New Game Concept',
      statusLabel: 'Draft',
      statusTone: 'purple',
      isPromotedActive: false,
      messageCount: dt.message_count ?? 0,
      updatedAtUtc: dt.updated_at_utc || dt.created_at_utc || '',
      threadId: dt.thread_id,
    });
  }

  draftSessions.sort(
    (a, b) => parseIsoMillis(b.updatedAtUtc) - parseIsoMillis(a.updatedAtUtc),
  );

  // 2. Index cartridge threads by cartridge_id
  const threadByCartridgeId = new Map<string, CoProducerThread>();
  const registerCartridgeThread = (ct?: CoProducerThread | null) => {
    const cid = ct?.cartridge_id?.trim();
    if (!ct || !cid) return;
    const prev = threadByCartridgeId.get(cid);
    if (!prev || parseIsoMillis(ct.updated_at_utc) >= parseIsoMillis(prev.updated_at_utc)) {
      threadByCartridgeId.set(cid, ct);
    }
  };

  registerCartridgeThread(threadsData?.active_cartridge_thread);
  for (const ct of threadsData?.cartridge_threads ?? []) {
    registerCartridgeThread(ct);
  }
  for (const anyThread of threadsData?.threads ?? []) {
    if (anyThread?.scope_kind === 'cartridge' || anyThread?.graduated_to_cartridge_id) {
      const targetCid = anyThread.cartridge_id || anyThread.graduated_to_cartridge_id;
      if (targetCid) {
        registerCartridgeThread({ ...anyThread, cartridge_id: targetCid });
      }
    }
  }

  // 3. Build game sessions from normalized game catalog + any brand-new graduated cartridge threads
  const seenCartridgeIds = new Set<string>();
  const gameSessions: UnifiedCoProducerSession[] = [];

  for (const g of games) {
    if (!g?.cartridge_id || seenCartridgeIds.has(g.cartridge_id)) continue;
    seenCartridgeIds.add(g.cartridge_id);
    const matchingThread = threadByCartridgeId.get(g.cartridge_id);
    const baseTitle = g.game_title || matchingThread?.title || g.cartridge_id;
    const displayTitle = g.hasDuplicateTitle ? `${baseTitle} (${g.shortId})` : baseTitle;
    const statusMeta = resolveGameStatusMeta(g.status, g.approved_items, g.total_items);

    gameSessions.push({
      key: `cartridge:${g.cartridge_id}`,
      selection: { kind: 'cartridge', cartridgeId: g.cartridge_id },
      kind: 'cartridge',
      title: displayTitle,
      subtitle: g.core_genre || matchingThread?.concept_brief?.core_genre || 'Game Session',
      statusLabel: statusMeta.label,
      statusTone: statusMeta.tone,
      isPromotedActive: Boolean(g.isPromotedActive),
      messageCount: matchingThread?.message_count ?? 0,
      boardProgress:
        g.total_items > 0 ? `${g.approved_items}/${g.total_items}` : undefined,
      updatedAtUtc: newerIsoTimestamp(g.updated_at_utc, matchingThread?.updated_at_utc),
      threadId: matchingThread?.thread_id,
      cartridgeId: g.cartridge_id,
    });
  }

  for (const [cid, ct] of threadByCartridgeId.entries()) {
    if (seenCartridgeIds.has(cid)) continue;
    seenCartridgeIds.add(cid);
    const title = ct.concept_brief?.working_title || ct.title || cid;
    gameSessions.push({
      key: `cartridge:${cid}`,
      selection: { kind: 'cartridge', cartridgeId: cid },
      kind: 'cartridge',
      title,
      subtitle: ct.concept_brief?.core_genre || 'Game Session',
      statusLabel: 'In Studio',
      statusTone: 'accent',
      isPromotedActive: false,
      messageCount: ct.message_count ?? 0,
      updatedAtUtc: ct.updated_at_utc || ct.created_at_utc || '',
      threadId: ct.thread_id,
      cartridgeId: cid,
    });
  }

  if (
    fallbackCartridge?.cartridgeId &&
    !seenCartridgeIds.has(fallbackCartridge.cartridgeId)
  ) {
    const cid = fallbackCartridge.cartridgeId;
    const matchingThread = threadByCartridgeId.get(cid);
    const approved = fallbackCartridge.approved ?? 0;
    const total = fallbackCartridge.total ?? 0;
    const statusMeta = resolveGameStatusMeta(fallbackCartridge.status, approved, total);
    gameSessions.push({
      key: `cartridge:${cid}`,
      selection: { kind: 'cartridge', cartridgeId: cid },
      kind: 'cartridge',
      title: fallbackCartridge.title || matchingThread?.title || cid,
      subtitle: matchingThread?.concept_brief?.core_genre || 'Game Session',
      statusLabel: statusMeta.label,
      statusTone: statusMeta.tone,
      isPromotedActive: true,
      messageCount: matchingThread?.message_count ?? 0,
      boardProgress: total > 0 ? `${approved}/${total}` : undefined,
      updatedAtUtc: matchingThread?.updated_at_utc || '',
      threadId: matchingThread?.thread_id,
      cartridgeId: cid,
    });
  }

  gameSessions.sort((a, b) => {
    if (a.isPromotedActive && !b.isPromotedActive) return -1;
    if (!a.isPromotedActive && b.isPromotedActive) return 1;
    return parseIsoMillis(b.updatedAtUtc) - parseIsoMillis(a.updatedAtUtc);
  });

  return [...draftSessions, ...gameSessions];
}

export function formatConversationTimestamp(rawIso?: string | null): string {
  if (!rawIso) return '';
  const ts = Date.parse(rawIso);
  if (!Number.isFinite(ts)) return rawIso;
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    }).format(new Date(ts));
  } catch {
    return rawIso;
  }
}
