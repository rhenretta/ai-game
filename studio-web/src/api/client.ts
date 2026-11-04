import type {
  ArtifactContentResponse,
  ArtifactInventoryResponse,
  BoardItemChronologyResponse,
  CartridgesResponse,
  CheckpointsResponse,
  CoProducerExecuteActionResponse,
  CoProducerSendMessageResponse,
  CoProducerThreadDetailResponse,
  CoProducerThreadsListResponse,
  EventsHistoryResponse,
  GameBibleResponse,
  GameMemoryListResponse,
  GameMemoryMutationRequest,
  PlaytestReportDispatchResponse,
  PlaytestSessionDetailResponse,
  PlaytestSessionsListResponse,
  PlaytestWish,
  SteamDnaResponse,
  StudioStatusResponse,
} from './types';

/**
 * Rule 5: Authentic, unmasked API error preserving HTTP status, endpoint, and server deliberation/stack context.
 * Zero silent fallbacks or placeholder payloads.
 */
export class StudioApiError extends Error {
  public readonly status: number;
  public readonly endpoint: string;
  public readonly deliberation?: string;
  public readonly rawBody: string;

  constructor(params: {
    message: string;
    status: number;
    endpoint: string;
    deliberation?: string;
    rawBody: string;
  }) {
    super(params.message);
    this.name = 'StudioApiError';
    this.status = params.status;
    this.endpoint = params.endpoint;
    this.deliberation = params.deliberation;
    this.rawBody = params.rawBody;
  }
}

const API_BASE = (import.meta.env.VITE_STUDIO_API_ORIGIN as string | undefined)?.replace(/\/$/, '') ?? '';

export function buildApiUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

export function buildWsUrl(path = '/ws/swarm'): string {
  if (API_BASE) {
    return `${API_BASE.replace(/^http/, 'ws')}${path}`;
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}${path}`;
}

/**
 * Resolves cartridge-scoped media URLs cleanly across mockups, sprites, audio, and QA proof captures.
 */
export function resolveMediaUrl(rawPathOrUrl: string | undefined | null, cartridgeId?: string): string {
  if (!rawPathOrUrl) return '';
  const trimmed = rawPathOrUrl.trim();
  if (!trimmed) return '';

  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
    return trimmed;
  }

  if (trimmed.startsWith('/api/')) {
    if (cartridgeId && cartridgeId !== 'active' && !trimmed.includes('cartridge=')) {
      const sep = trimmed.includes('?') ? '&' : '?';
      return buildApiUrl(`${trimmed}${sep}cartridge=${encodeURIComponent(cartridgeId)}`);
    }
    return buildApiUrl(trimmed);
  }

  // Normalize absolute disk paths containing `/cartridges/` into workspace-relative cartridge paths
  const cartridgesMarkerIdx = trimmed.indexOf('/cartridges/');
  const normalizedRel =
    cartridgesMarkerIdx >= 0
      ? trimmed.slice(cartridgesMarkerIdx + '/cartridges/'.length)
      : trimmed.replace(/^\.?\//, '');

  const fullRel =
    cartridgeId &&
    cartridgeId !== 'active' &&
    !normalizedRel.startsWith('cartridges/') &&
    !normalizedRel.startsWith('candidate_') &&
    !normalizedRel.startsWith('_coproducer_drafts/') &&
    !normalizedRel.startsWith('active/') &&
    !normalizedRel.startsWith(cartridgeId)
      ? `${cartridgeId}/${normalizedRel}`
      : normalizedRel.replace(/^cartridges\//, '');
  return buildApiUrl(`/api/artifacts/content?path=${encodeURIComponent(fullRel)}&raw=true`);
}

async function requestJson<T>(endpoint: string, init?: RequestInit): Promise<T> {
  const url = buildApiUrl(endpoint);
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...(init?.headers ?? {}),
      },
    });
  } catch (networkErr) {
    const msg = networkErr instanceof Error ? networkErr.message : String(networkErr);
    throw new StudioApiError({
      message: `Network request failed for ${endpoint}: ${msg}`,
      status: 0,
      endpoint,
      rawBody: msg,
    });
  }

  const rawText = await res.text();
  let parsed: Record<string, unknown> | null = null;
  if (rawText) {
    try {
      parsed = JSON.parse(rawText) as Record<string, unknown>;
    } catch {
      parsed = null;
    }
  }

  if (!res.ok) {
    const serverMsg =
      (parsed && (typeof parsed.error === 'string' ? parsed.error : undefined)) ||
      (parsed && (typeof parsed.detail === 'string' ? parsed.detail : undefined)) ||
      (parsed && (typeof parsed.deliberation === 'string' ? parsed.deliberation : undefined)) ||
      rawText ||
      `HTTP ${res.status} ${res.statusText}`;
    throw new StudioApiError({
      message: serverMsg,
      status: res.status,
      endpoint,
      deliberation: parsed && typeof parsed.deliberation === 'string' ? parsed.deliberation : undefined,
      rawBody: rawText,
    });
  }

  if (parsed === null) {
    throw new StudioApiError({
      message: `Expected JSON response from ${endpoint}, received non-JSON body.`,
      status: res.status,
      endpoint,
      rawBody: rawText,
    });
  }

  return parsed as T;
}

export const studioApi = {
  getCartridges(): Promise<CartridgesResponse> {
    return requestJson<CartridgesResponse>('/api/cartridges');
  },

  getStudioStatus(): Promise<StudioStatusResponse> {
    return requestJson<StudioStatusResponse>('/api/studio/status');
  },

  getRecentEvents(params?: { limit?: number; cartridgeId?: string }): Promise<EventsHistoryResponse> {
    const search = new URLSearchParams();
    search.set('limit', String(params?.limit ?? 50));
    if (params?.cartridgeId && params.cartridgeId !== 'active') {
      search.set('cartridge', params.cartridgeId);
    }
    return requestJson<EventsHistoryResponse>(`/api/events/history?${search.toString()}`);
  },

  getGameBible(cartridgeId?: string): Promise<GameBibleResponse> {
    const qs =
      cartridgeId && cartridgeId !== 'active'
        ? `?cartridge=${encodeURIComponent(cartridgeId)}`
        : '';
    return requestJson<GameBibleResponse>(`/api/studio/game-bible${qs}`);
  },

  getBoardItemChronology(itemId: string, cartridgeId?: string): Promise<BoardItemChronologyResponse> {
    const qs =
      cartridgeId && cartridgeId !== 'active'
        ? `?cartridge=${encodeURIComponent(cartridgeId)}`
        : '';
    return requestJson<BoardItemChronologyResponse>(
      `/api/studio/board-item/${encodeURIComponent(itemId)}/chronology${qs}`,
    );
  },

  getArtifacts(cartridgeId?: string): Promise<ArtifactInventoryResponse> {
    const qs =
      cartridgeId && cartridgeId !== 'active'
        ? `?cartridge=${encodeURIComponent(cartridgeId)}`
        : '';
    return requestJson<ArtifactInventoryResponse>(`/api/artifacts${qs}`);
  },

  getArtifactContent(params: { path?: string; filename?: string; cartridgeId?: string }): Promise<ArtifactContentResponse> {
    const search = new URLSearchParams();
    if (params.path) {
      const clean = params.path.replace(/^\.?\//, '');
      const scoped =
        params.cartridgeId &&
        params.cartridgeId !== 'active' &&
        !clean.startsWith('cartridges/') &&
        !clean.startsWith(params.cartridgeId)
          ? `${params.cartridgeId}/${clean}`
          : clean;
      search.set('path', scoped);
    } else if (params.filename) {
      search.set('filename', params.filename);
    }
    return requestJson<ArtifactContentResponse>(`/api/artifacts/content?${search.toString()}`);
  },

  getCheckpoints(): Promise<CheckpointsResponse> {
    return requestJson<CheckpointsResponse>('/api/checkpoints');
  },

  activateCheckpoint(checkpointOrCartridgeId: string): Promise<{ deliberation: string; status?: string }> {
    return requestJson(`/api/checkpoints/${encodeURIComponent(checkpointOrCartridgeId)}/activate`, {
      method: 'POST',
    });
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  launchGame(cartridgeId: string): Promise<{ deliberation: string; status: string; game_title?: string; pid?: number }> {
    const payload = {
      deliberation: `Executive Producer requested native Godot 4.3 launch for cartridge '${cartridgeId}'.`,
      cartridge_id: cartridgeId,
    };
    return requestJson('/api/game/launch', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  updateStudioSettings(params: {
    pauseForGreenlight: boolean;
    paused?: boolean;
    reason?: string;
  }): Promise<{ deliberation: string; pause_for_greenlight: boolean; paused: boolean }> {
    const payload = {
      deliberation:
        params.reason ||
        `Executive Producer updated studio orchestration settings (paused=${String(params.paused)}, pause_for_greenlight=${String(params.pauseForGreenlight)}).`,
      pause_for_greenlight: params.pauseForGreenlight,
      ...(params.paused !== undefined ? { paused: params.paused } : {}),
    };
    return requestJson('/api/studio/settings', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  submitGreenlightOrSteering(params: {
    approved: boolean;
    feedback: string;
  }): Promise<{ deliberation: string; status: string; auto_resumed_job_id?: string | null }> {
    const cleanFeedback = params.feedback.trim();
    const payload = {
      deliberation: params.approved
        ? 'Executive Producer greenlit the Living Game Bible to begin slice production.'
        : `Executive Producer injected creative steering into the Specialist Council: ${cleanFeedback}`,
      approved: params.approved,
      feedback: cleanFeedback,
    };
    return requestJson('/api/studio/greenlight', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  triggerGenesisBuild(params: {
    resumeCurrent: boolean;
    creativePrompt?: string;
    pauseForGreenlight?: boolean;
    resetActive?: boolean;
    candidateId?: string;
    profileTag?: string;
  }): Promise<{ deliberation: string; job_id: string; status: string }> {
    const prompt = (params.creativePrompt ?? '').trim();
    const payload = {
      deliberation: params.resumeCurrent
        ? `Executive Producer resumed / iterated next slice${params.candidateId ? ` for '${params.candidateId}'` : ''}${prompt ? ` with direction: ${prompt}` : ''}.`
        : `Executive Producer commissioned a new autonomous Genesis game build${prompt ? ` with creative seed: ${prompt}` : ''}.`,
      profile_tag: params.profileTag || 'default',
      resume_current: params.resumeCurrent,
      reset_active: Boolean(params.resetActive),
      ...(params.pauseForGreenlight !== undefined ? { pause_for_greenlight: params.pauseForGreenlight } : {}),
      ...(params.candidateId ? { candidate_id: params.candidateId } : {}),
      ...(prompt ? { creative_prompt: prompt, feedback: prompt } : {}),
    };
    return requestJson('/api/genesis/trigger', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  getPlaytestSessions(params?: {
    candidateId?: string;
    sliceId?: string;
    limit?: number;
  }): Promise<PlaytestSessionsListResponse> {
    const search = new URLSearchParams();
    if (params?.candidateId && params.candidateId !== 'all') {
      search.set('candidate_id', params.candidateId);
    }
    if (params?.sliceId && params.sliceId !== 'all') {
      search.set('slice_id', params.sliceId);
    }
    if (params?.limit) {
      search.set('limit', String(params.limit));
    }
    const qs = search.toString();
    return requestJson<PlaytestSessionsListResponse>(`/api/playtest/sessions${qs ? `?${qs}` : ''}`);
  },

  getPlaytestSessionDetail(sessionId: string): Promise<PlaytestSessionDetailResponse> {
    return requestJson<PlaytestSessionDetailResponse>(
      `/api/playtest/sessions/${encodeURIComponent(sessionId)}`,
    );
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  startPlaytestSession(params: {
    checkpointId: string;
    candidateId?: string;
    sliceId?: string | null;
    maxTurns: number;
    enableAudio: boolean;
  }): Promise<{ deliberation: string; session: PlaytestSessionDetailResponse['session'] }> {
    const payload = {
      deliberation: `Executive Producer launched Blind Playtester session on target '${params.checkpointId}' (max_turns=${params.maxTurns}, audio=${params.enableAudio}).`,
      checkpoint_id: params.checkpointId,
      ...(params.candidateId ? { candidate_id: params.candidateId } : {}),
      ...(params.sliceId ? { slice_id: params.sliceId } : {}),
      max_turns: params.maxTurns,
      capture_audio: params.enableAudio,
      enable_audio: params.enableAudio,
    };
    return requestJson('/api/playtest/sessions/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  controlPlaytestSession(
    sessionId: string,
    action: 'pause' | 'resume' | 'stop' | 'reset',
  ): Promise<{ deliberation: string; action: string }> {
    const payload = {
      deliberation: `Executive Producer issued '${action}' control command to Blind Playtester session '${sessionId}'.`,
      action,
    };
    return requestJson(`/api/playtest/sessions/${encodeURIComponent(sessionId)}/control`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  getPlaytestWishes(params?: { sessionId?: string; candidateId?: string }): Promise<{ deliberation: string; wishes: PlaytestWish[] }> {
    const search = new URLSearchParams();
    if (params?.sessionId) search.set('session_id', params.sessionId);
    if (params?.candidateId) search.set('candidate_id', params.candidateId);
    const qs = search.toString();
    return requestJson(`/api/playtest/wishes${qs ? `?${qs}` : ''}`);
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  sendWishesToCouncil(params: {
    wishIds: string[];
    producerNote?: string;
  }): Promise<{ deliberation: string; status: string; wishes: PlaytestWish[] }> {
    const payload = {
      deliberation: `Executive Producer approved and dispatched ${params.wishIds.length} Blind Playtester wish(es) to the Specialist Council.`,
      wish_ids: params.wishIds,
      producer_note: (params.producerNote ?? '').trim(),
    };
    return requestJson('/api/playtest/wishes/send-to-council', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  sendPlaytestReportToCouncil(params: {
    sessionId: string;
    includePendingWishes?: boolean;
    producerNote?: string;
  }): Promise<PlaytestReportDispatchResponse> {
    const includePending = params.includePendingWishes ?? true;
    const cleanNote = (params.producerNote ?? '').trim();
    const payload = {
      deliberation: `Executive Producer dispatched full Blind Playtester session report '${params.sessionId}' (include_pending_wishes=${includePending}) to the Specialist Council.`,
      include_pending_wishes: includePending,
      ...(cleanNote ? { producer_note: cleanNote } : {}),
    };
    return requestJson(`/api/playtest/sessions/${encodeURIComponent(params.sessionId)}/send-report-to-council`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  dismissWish(wishId: string): Promise<{ deliberation: string; wish: PlaytestWish }> {
    return requestJson(`/api/playtest/wishes/${encodeURIComponent(wishId)}/dismiss`, {
      method: 'POST',
    });
  },

  getSteamDna(forceRescan = false): Promise<SteamDnaResponse> {
    const qs = forceRescan ? '?rescan=true' : '';
    return requestJson<SteamDnaResponse>(`/api/steam/dna${qs}`);
  },

  getGameMemories(params?: {
    cartridgeId?: string;
    bank?: string;
    kind?: string;
    status?: string;
    semanticQuery?: string;
  }): Promise<GameMemoryListResponse> {
    const search = new URLSearchParams();
    if (params?.cartridgeId && params.cartridgeId !== 'active') {
      search.set('cartridge', params.cartridgeId);
    }
    if (params?.bank && params.bank !== 'all') {
      search.set('bank', params.bank);
    }
    if (params?.kind && params.kind !== 'all') {
      search.set('kind', params.kind);
    }
    if (params?.status && params.status !== 'all') {
      search.set('status', params.status);
    }
    if (params?.semanticQuery?.trim()) {
      search.set('q', params.semanticQuery.trim());
    }
    const qs = search.toString();
    return requestJson<GameMemoryListResponse>(`/api/studio/memories${qs ? `?${qs}` : ''}`);
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  mutateGameMemory(req: GameMemoryMutationRequest): Promise<GameMemoryListResponse> {
    const payload = {
      deliberation: req.deliberation.trim(),
      ...(req.cartridge && req.cartridge !== 'active' ? { cartridge: req.cartridge } : {}),
      ...(req.author_role ? { author_role: req.author_role } : {}),
      operation: req.operation,
      ...(req.target_memory_id ? { target_memory_id: req.target_memory_id } : {}),
      ...(req.target_bank ? { target_bank: req.target_bank } : {}),
      ...(req.kind ? { kind: req.kind } : {}),
      ...(req.title ? { title: req.title.trim() } : {}),
      ...(req.content ? { content: req.content.trim() } : {}),
      ...(req.applicable_roles && req.applicable_roles.length > 0
        ? { applicable_roles: req.applicable_roles }
        : {}),
      ...(req.resolution_note ? { resolution_note: req.resolution_note.trim() } : {}),
      ...(req.slice_id ? { slice_id: req.slice_id } : {}),
      ...(req.item_id ? { item_id: req.item_id } : {}),
    };
    return requestJson<GameMemoryListResponse>('/api/studio/memories', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  getCoProducerThreads(cartridgeId?: string): Promise<CoProducerThreadsListResponse> {
    const qs =
      cartridgeId && cartridgeId !== 'active'
        ? `?cartridge=${encodeURIComponent(cartridgeId)}`
        : '';
    return requestJson<CoProducerThreadsListResponse>(`/api/coproducer/threads${qs}`);
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  createCoProducerDraft(title?: string): Promise<CoProducerThreadDetailResponse> {
    const cleanTitle = (title ?? '').trim() || 'New Game Concept';
    const payload = {
      deliberation: `Executive Producer created a new pre-genesis Co-Producer concept draft ('${cleanTitle}').`,
      title: cleanTitle,
    };
    return requestJson<CoProducerThreadDetailResponse>('/api/coproducer/threads/draft', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  getCoProducerThread(threadId: string): Promise<CoProducerThreadDetailResponse> {
    return requestJson<CoProducerThreadDetailResponse>(
      `/api/coproducer/threads/${encodeURIComponent(threadId)}`,
    );
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  sendCoProducerMessage(
    threadId: string,
    params: {
      message: string;
      selectedSketchPaths?: string[];
    },
  ): Promise<CoProducerSendMessageResponse> {
    const cleanMsg = params.message.trim();
    const payload = {
      deliberation: `Executive Producer sent conversational message to Co-Producer thread '${threadId}'.`,
      message: cleanMsg,
      ...(params.selectedSketchPaths && params.selectedSketchPaths.length > 0
        ? { selected_sketch_paths: params.selectedSketchPaths }
        : {}),
    };
    return requestJson<CoProducerSendMessageResponse>(
      `/api/coproducer/threads/${encodeURIComponent(threadId)}/messages`,
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
    );
  },

  /** Rule 4: `deliberation` serialized strictly at key index 0 */
  executeCoProducerAction(params: {
    threadId: string;
    messageId?: string;
    proposalId?: string;
    directActionKind?: 'commission_game';
    carryOverAssetPaths?: string[];
    dismiss?: boolean;
    overrideBrief?: string;
    overrideHandoffMode?: 'loose_spark' | 'refined_concept';
    overridePauseForGreenlight?: boolean;
  }): Promise<CoProducerExecuteActionResponse> {
    const payload = {
      deliberation: params.dismiss
        ? `Executive Producer dismissed Co-Producer action proposal '${params.proposalId ?? ''}'.`
        : params.directActionKind
          ? `Executive Producer directly commissioned thread '${params.threadId}' (${params.directActionKind}).`
          : `Executive Producer confirmed and executed Co-Producer action proposal '${params.proposalId ?? ''}'.`,
      thread_id: params.threadId,
      message_id: params.messageId ?? '',
      proposal_id: params.proposalId ?? '',
      ...(params.directActionKind ? { direct_action_kind: params.directActionKind } : {}),
      ...(params.carryOverAssetPaths && params.carryOverAssetPaths.length > 0
        ? { carry_over_asset_paths: params.carryOverAssetPaths }
        : {}),
      dismiss: Boolean(params.dismiss),
      ...(params.overrideBrief !== undefined ? { override_brief: params.overrideBrief } : {}),
      ...(params.overrideHandoffMode
        ? { override_handoff_mode: params.overrideHandoffMode }
        : {}),
      ...(params.overridePauseForGreenlight !== undefined
        ? { override_pause_for_greenlight: params.overridePauseForGreenlight }
        : {}),
    };
    return requestJson<CoProducerExecuteActionResponse>('/api/coproducer/actions/execute', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },
};
