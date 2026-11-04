import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { studioApi } from './api/client';
import type { CoProducerThread, WorkspaceId } from './api/types';
import {
  buildUnifiedCoProducerSessions,
  type StudioSessionSelection,
} from './domain/coproducerSessions';
import {
  filterBoardItemsBySlice,
  normalizeGameCatalog,
  partitionTranscriptBySlice,
  resolveSliceContext,
  resolveSliceVerificationState,
} from './domain/slices';
import { shouldSurfaceBuildError } from './domain/errors';
import { deriveLiveStudioActivity } from './domain/liveActivity';
import { useSwarmSocket, type LiveItemStep } from './realtime/useSwarmSocket';
import {
  TopCommandBar,
  UI_SCALE_MAX_PCT,
  UI_SCALE_MIN_PCT,
} from './features/shell/TopCommandBar';
import { SliceSubbar } from './features/shell/SliceSubbar';
import { LiveActivityBanner } from './features/shell/LiveActivityBanner';
import { NextSliceModal } from './features/shell/NextSliceModal';
import { CheckpointsModal } from './features/shell/CheckpointsModal';
import { LivePulseModal } from './features/shell/LivePulseModal';
import { ProductionBoardView } from './features/board/ProductionBoardView';
import { WritersRoomView } from './features/writers-room/WritersRoomView';
import { GameBibleView } from './features/bible/GameBibleView';
import { GameMemoryView } from './features/memory/GameMemoryView';
import { AssetsAndCodeView } from './features/assets/AssetsAndCodeView';
import { PlaytestLabView } from './features/playtest/PlaytestLabView';
import { PlayerDnaView } from './features/dna/PlayerDnaView';
import { CoProducerView } from './features/coproducer/CoProducerView';
import { ErrorBanner } from './components/ErrorBanner';
import { MediaLightbox, type LightboxMedia } from './components/MediaLightbox';
import { WorkspaceErrorBoundary } from './components/WorkspaceErrorBoundary';

const EMPTY_LIVE_STEPS: Record<string, LiveItemStep> = {};

export function App() {
  const queryClient = useQueryClient();
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceId>('coproducer');
  const [selectedCartridgeId, setSelectedCartridgeId] = useState<string>('');
  const [selectedSession, setSelectedSession] = useState<StudioSessionSelection | null>(null);
  const [latestCreatedDraft, setLatestCreatedDraft] = useState<CoProducerThread | null>(null);
  const [selectedSliceId, setSelectedSliceId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<unknown>(null);
  const [dismissedQueryErrorMsg, setDismissedQueryErrorMsg] = useState<string | null>(null);
  const [dismissedBuildErrorKeys, setDismissedBuildErrorKeys] = useState<Set<string>>(() => {
    try {
      const raw = window.sessionStorage.getItem('studio-web.dismissedBuildErrors');
      if (!raw) return new Set();
      const parsed = JSON.parse(raw) as unknown;
      return Array.isArray(parsed) ? new Set(parsed.map(String)) : new Set();
    } catch {
      return new Set();
    }
  });
  const [launchStatusText, setLaunchStatusText] = useState<string | null>(null);
  const [steeringNotice, setSteeringNotice] = useState<string | null>(null);
  const [uiScalePct, setUiScalePct] = useState<number>(() => {
    try {
      const raw = window.localStorage.getItem('studio-web.uiScalePct');
      const parsed = raw ? Number(raw) : NaN;
      if (Number.isFinite(parsed)) {
        return Math.min(UI_SCALE_MAX_PCT, Math.max(UI_SCALE_MIN_PCT, Math.round(parsed)));
      }
    } catch {
      // Ignore storage access failures in restricted environments
    }
    return 100;
  });

  useEffect(() => {
    const clamped = Math.min(UI_SCALE_MAX_PCT, Math.max(UI_SCALE_MIN_PCT, Math.round(uiScalePct)));
    document.documentElement.style.setProperty('--ui-scale', String(clamped / 100));
    try {
      window.localStorage.setItem('studio-web.uiScalePct', String(clamped));
    } catch {
      // Ignore storage write errors
    }
  }, [uiScalePct]);

  const [requestedDraftThreadId, setRequestedDraftThreadId] = useState<string | null>(null);
  const [nextSliceOpen, setNextSliceOpen] = useState(false);
  const [checkpointsOpen, setCheckpointsOpen] = useState(false);
  const [livePulseOpen, setLivePulseOpen] = useState(false);
  const [lightboxMedia, setLightboxMedia] = useState<LightboxMedia | null>(null);

  const realtime = useSwarmSocket();

  const cartridgesQuery = useQuery({
    queryKey: ['cartridges'],
    queryFn: () => studioApi.getCartridges(),
  });

  const studioStatusQuery = useQuery({
    queryKey: ['studio-status'],
    queryFn: () => studioApi.getStudioStatus(),
    enabled: livePulseOpen,
  });

  const playtestListQuery = useQuery({
    queryKey: ['playtest-sessions'],
    queryFn: () => studioApi.getPlaytestSessions({ limit: 200 }),
    refetchInterval: (query) => {
      const sessions = query.state.data?.sessions ?? [];
      const anyLive = sessions.some(
        (s) => s.status === 'running' || s.status === 'starting' || s.status === 'stopping',
      );
      if (anyLive) return realtime.connected ? 5000 : 2500;
      return !realtime.connected && activeWorkspace === 'playtest' ? 6000 : false;
    },
  });

  const games = useMemo(
    () =>
      normalizeGameCatalog(
        cartridgesQuery.data?.games ?? [],
        playtestListQuery.data?.active_cartridge_id,
      ),
    [cartridgesQuery.data?.games, playtestListQuery.data?.active_cartridge_id],
  );

  useEffect(() => {
    if (games.length === 0) return;
    if (!selectedCartridgeId || !games.some((g) => g.cartridge_id === selectedCartridgeId)) {
      setSelectedCartridgeId(games[0].cartridge_id);
    }
  }, [games, selectedCartridgeId]);

  const effectiveCartridgeId = selectedCartridgeId || games[0]?.cartridge_id || 'active';
  const cartridgeThreadParam =
    effectiveCartridgeId && effectiveCartridgeId !== 'active' ? effectiveCartridgeId : undefined;

  const coproducerThreadsQuery = useQuery({
    queryKey: ['coproducer-threads', cartridgeThreadParam ?? 'none'],
    queryFn: () => studioApi.getCoProducerThreads(cartridgeThreadParam),
  });

  const bibleQuery = useQuery({
    queryKey: ['game-bible', effectiveCartridgeId],
    queryFn: () => studioApi.getGameBible(effectiveCartridgeId),
    // Fast 2.5s sync while WS reconnects; lightweight 8s liveness heartbeat ONLY while a build is marked active; never calls /api/events/history!
    refetchInterval: (query) => {
      if (!realtime.connected) return 2500;
      const proc = query.state.data?.process_status;
      const buildActive = Boolean(
        proc?.build_actively_running ||
          proc?.process_state === 'running' ||
          proc?.process_state === 'pausing' ||
          proc?.process_state === 'awaiting_greenlight' ||
          query.state.data?.orchestrator_state === 'AWAITING_GREENLIGHT',
      );
      return buildActive ? 8000 : false;
    },
  });

  const artifactsQuery = useQuery({
    queryKey: ['artifacts', effectiveCartridgeId],
    queryFn: () => studioApi.getArtifacts(effectiveCartridgeId),
  });

  const bible = bibleQuery.data;
  const resolvedCartridgeId =
    (selectedCartridgeId && selectedCartridgeId !== 'active' ? selectedCartridgeId : undefined) ||
    bible?.active_candidate_id ||
    bible?.cartridge_id ||
    'active';

  const hasAnyCartridge =
    games.length > 0 ||
    Boolean(bible?.has_bible) ||
    (resolvedCartridgeId !== '' && resolvedCartridgeId !== 'active');

  const sliceCtx = useMemo(
    () => resolveSliceContext(bible, selectedSliceId),
    [bible, selectedSliceId],
  );

  const allBoardItems = useMemo(() => bible?.production_board?.items ?? [], [bible]);

  const sliceBoardItems = useMemo(
    () => filterBoardItemsBySlice(allBoardItems, sliceCtx.selectedSlice, sliceCtx.slices.length),
    [allBoardItems, sliceCtx.selectedSlice, sliceCtx.slices.length],
  );

  const sliceTranscript = useMemo(() => {
    const fullTx = bible?.writers_room_transcript ?? [];
    if (!sliceCtx.selectedSlice || sliceCtx.slices.length <= 1) return fullTx;
    const bySlice = partitionTranscriptBySlice(fullTx, sliceCtx.slices);
    return bySlice.get(sliceCtx.selectedSlice.slice_id) ?? fullTx;
  }, [bible, sliceCtx.selectedSlice, sliceCtx.slices]);

  const boardCounts = useMemo(() => {
    const approved = sliceBoardItems.filter((i) => i.state === 'approved').length;
    return { approved, total: sliceBoardItems.length };
  }, [sliceBoardItems]);

  const unifiedSessions = useMemo(
    () =>
      buildUnifiedCoProducerSessions({
        games,
        threadsData: coproducerThreadsQuery.data,
        activeDraftThread: latestCreatedDraft,
        fallbackCartridge: hasAnyCartridge
          ? {
              cartridgeId: resolvedCartridgeId,
              title: bible?.game_title,
              approved: boardCounts.approved,
              total: boardCounts.total,
              status: bible?.process_status?.process_state || bible?.status,
            }
          : null,
      }),
    [
      games,
      coproducerThreadsQuery.data,
      latestCreatedDraft,
      hasAnyCartridge,
      resolvedCartridgeId,
      bible?.game_title,
      bible?.process_status?.process_state,
      bible?.status,
      boardCounts.approved,
      boardCounts.total,
    ],
  );

  // Synchronize initial / graduated session selection
  useEffect(() => {
    if (selectedSession === null) {
      const firstGame = unifiedSessions.find((s) => s.kind === 'cartridge');
      if (firstGame) {
        setSelectedSession(firstGame.selection);
        if (firstGame.cartridgeId) {
          setSelectedCartridgeId(firstGame.cartridgeId);
        }
        return;
      }
      const firstDraft = unifiedSessions.find((s) => s.kind === 'draft');
      if (firstDraft) {
        setSelectedSession(firstDraft.selection);
      }
      return;
    }

    if (selectedSession.kind === 'cartridge') {
      if (selectedSession.cartridgeId === 'active' && games.length > 0) {
        setSelectedSession({ kind: 'cartridge', cartridgeId: games[0].cartridge_id });
        setSelectedCartridgeId(games[0].cartridge_id);
      }
      return;
    }

    // If a selected draft graduated to a cartridge on the server, follow it seamlessly
    if (selectedSession.kind === 'draft' && selectedSession.threadId !== '__creating__') {
      const allKnownThreads = [
        ...(coproducerThreadsQuery.data?.threads ?? []),
        ...(coproducerThreadsQuery.data?.draft_threads ?? []),
      ];
      const matched = allKnownThreads.find((t) => t.thread_id === selectedSession.threadId);
      if (matched?.graduated_to_cartridge_id) {
        const nextCid = matched.graduated_to_cartridge_id;
        setSelectedCartridgeId(nextCid);
        setSelectedSliceId(null);
        setSelectedSession({ kind: 'cartridge', cartridgeId: nextCid });
      }
    }
  }, [selectedSession, unifiedSessions, games, coproducerThreadsQuery.data]);

  const initialCatalogLoading = cartridgesQuery.isPending || bibleQuery.isPending;
  const isDraftSession =
    selectedSession?.kind === 'draft' ||
    (!initialCatalogLoading && !hasAnyCartridge && selectedSession === null);

  const activeSessionMeta = useMemo(() => {
    if (!selectedSession) return unifiedSessions[0] ?? null;
    return (
      unifiedSessions.find((s) =>
        selectedSession.kind === 'draft'
          ? s.kind === 'draft' && s.threadId === selectedSession.threadId
          : s.kind === 'cartridge' && s.cartridgeId === selectedSession.cartridgeId,
      ) ?? null
    );
  }, [selectedSession, unifiedSessions]);

  const handleSelectSession = (next: StudioSessionSelection) => {
    setSelectedSession(next);
    setActionError(null);
    if (next.kind === 'cartridge') {
      setSelectedCartridgeId(next.cartridgeId);
      setSelectedSliceId(null);
      setRequestedDraftThreadId(null);
    } else {
      setRequestedDraftThreadId(next.threadId);
      if (activeWorkspace !== 'coproducer' && activeWorkspace !== 'dna') {
        setActiveWorkspace('coproducer');
      }
    }
  };

  const handleSelectCartridge = (cid: string) => {
    setSelectedCartridgeId(cid);
    setSelectedSliceId(null);
    setRequestedDraftThreadId(null);
    setSelectedSession({ kind: 'cartridge', cartridgeId: cid });
  };

  const assetsCount =
    (artifactsQuery.data?.sprites.length ?? 0) +
    (artifactsQuery.data?.audio_clips.length ?? 0) +
    (artifactsQuery.data?.code_modules.length ?? 0);

  const playtestRunsForSelected = useMemo(() => {
    const allRuns = playtestListQuery.data?.sessions ?? [];
    const activeCid = playtestListQuery.data?.active_cartridge_id ?? '';
    return allRuns.filter(
      (s) =>
        s.candidate_id === resolvedCartridgeId ||
        (resolvedCartridgeId === activeCid && s.matches_active_cartridge),
    ).length;
  }, [playtestListQuery.data, resolvedCartridgeId]);

  // Executive Producer Mutations
  const activateMut = useMutation({
    mutationFn: () => studioApi.activateCheckpoint(resolvedCartridgeId),
    onSuccess: () => {
      setActionError(null);
      void queryClient.invalidateQueries({ queryKey: ['cartridges'] });
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
    },
    onError: (err) => setActionError(err),
  });

  const launchMut = useMutation({
    mutationFn: (overrideId?: string) => studioApi.launchGame(overrideId || resolvedCartridgeId),
    onSuccess: () => {
      setActionError(null);
      setLaunchStatusText('Window Opened');
      window.setTimeout(() => setLaunchStatusText(null), 2400);
    },
    onError: (err) => setActionError(err),
  });

  const pauseMut = useMutation({
    mutationFn: (pause: boolean) =>
      studioApi.updateStudioSettings({
        pauseForGreenlight: Boolean(bible?.pause_for_greenlight),
        paused: pause,
      }),
    onSuccess: () => {
      setActionError(null);
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
    },
    onError: (err) => setActionError(err),
  });

  const steerOrGreenlightMut = useMutation({
    mutationFn: (params: { approved: boolean; feedback: string }) =>
      studioApi.submitGreenlightOrSteering(params),
    onSuccess: (_res, vars) => {
      setActionError(null);
      if (!vars.approved && vars.feedback.trim()) {
        setSteeringNotice('Live direction queued');
        window.setTimeout(() => setSteeringNotice(null), 3200);
      }
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
    },
    onError: (err) => setActionError(err),
  });

  const genesisMut = useMutation({
    mutationFn: async (params: {
      resumeCurrent: boolean;
      creativePrompt?: string;
      pauseForGreenlight?: boolean;
    }) => {
      if (params.resumeCurrent) {
        await studioApi.updateStudioSettings({
          pauseForGreenlight: false,
          paused: false,
          reason: `Executive Producer resumed slice build for '${resolvedCartridgeId}'.`,
        });
      }
      return studioApi.triggerGenesisBuild({
        resumeCurrent: params.resumeCurrent,
        creativePrompt: params.creativePrompt,
        pauseForGreenlight: params.pauseForGreenlight,
        resetActive: !params.resumeCurrent,
        candidateId: params.resumeCurrent ? resolvedCartridgeId : undefined,
      });
    },
    onSuccess: (_res, vars) => {
      setActionError(null);
      if (!vars.resumeCurrent) {
        setSelectedCartridgeId('');
        setSelectedSliceId(null);
        setRequestedDraftThreadId(null);
        setSelectedSession({ kind: 'cartridge', cartridgeId: 'active' });
        setActiveWorkspace('writers-room');
      } else {
        setSelectedSliceId(null);
        setSteeringNotice(
          vars.creativePrompt?.trim() ? 'Direction sent · Studio resumed' : 'Studio build resumed',
        );
        window.setTimeout(() => setSteeringNotice(null), 3500);
      }
      void queryClient.invalidateQueries({ queryKey: ['cartridges'] });
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
      void queryClient.invalidateQueries({ queryKey: ['coproducer-threads'] });
    },
    onError: (err) => setActionError(err),
  });

  const startNewGameDraftMut = useMutation({
    mutationFn: () => studioApi.createCoProducerDraft('New Game Concept'),
    onSuccess: (res) => {
      setActionError(null);
      if (res?.thread?.thread_id) {
        setLatestCreatedDraft(res.thread);
        queryClient.setQueryData(['coproducer-thread', res.thread.thread_id], res);
        setRequestedDraftThreadId(res.thread.thread_id);
        setSelectedSession({ kind: 'draft', threadId: res.thread.thread_id });
      }
      void queryClient.invalidateQueries({ queryKey: ['coproducer-threads'] });
    },
    onError: (err) => setActionError(err),
  });

  const handleStartNewGameSession = () => {
    setActiveWorkspace('coproducer');
    setSelectedSession({ kind: 'draft', threadId: '__creating__' });
    startNewGameDraftMut.mutate();
  };

  const isBuildRunning = Boolean(
    bible?.process_status?.build_actively_running || bible?.process_status?.process_state === 'running',
  );
  const isAwaitingGreenlight =
    bible?.process_status?.process_state === 'awaiting_greenlight' ||
    bible?.orchestrator_state === 'AWAITING_GREENLIGHT';
  const isPausingBuild = bible?.process_status?.process_state === 'pausing';
  const shouldSeedLiveHistory =
    !isDraftSession && (isBuildRunning || isAwaitingGreenlight || isPausingBuild);

  useEffect(() => {
    if (!shouldSeedLiveHistory) return;
    realtime.ensureHistorySeeded(resolvedCartridgeId);
  }, [shouldSeedLiveHistory, resolvedCartridgeId, realtime.ensureHistorySeeded]);

  const liveActivity = useMemo(
    () =>
      deriveLiveStudioActivity({
        bible,
        recentEvents: realtime.recentEvents,
        liveStepsByItem: isBuildRunning ? realtime.liveStepsByItem : EMPTY_LIVE_STEPS,
        playtestSessions: playtestListQuery.data?.sessions,
      }),
    [
      bible,
      realtime.recentEvents,
      isBuildRunning,
      realtime.liveStepsByItem,
      playtestListQuery.data?.sessions,
    ],
  );
  const steeringBusy = steerOrGreenlightMut.isPending || genesisMut.isPending;

  const handleSteerCouncil = async (feedback: string) => {
    const trimmed = feedback.trim();
    if (!trimmed) return;
    if (isBuildRunning || isAwaitingGreenlight) {
      await steerOrGreenlightMut.mutateAsync({ approved: false, feedback: trimmed });
    } else {
      await genesisMut.mutateAsync({ resumeCurrent: true, creativePrompt: trimmed });
    }
  };

  const lastBuildError = bible?.process_status?.last_error;
  const lastBuildErrorKey = lastBuildError
    ? `${lastBuildError.timestamp_utc ?? ''}|${lastBuildError.agent_name}|${lastBuildError.event_type}|${lastBuildError.message}`
    : '';
  const rawHasBuildError = Boolean(
    lastBuildError &&
      shouldSurfaceBuildError({
        lastError: lastBuildError,
        allBoardItems,
        isBuildRunning,
        latestActivityUtc:
          bible?.process_status?.last_event_timestamp_utc || bible?.updated_at_utc,
      }),
  );
  const visibleBuildError =
    !isDraftSession &&
    rawHasBuildError &&
    lastBuildError &&
    !dismissedBuildErrorKeys.has(lastBuildErrorKey)
      ? lastBuildError
      : null;
  const sliceVerification = useMemo(
    () => resolveSliceVerificationState(bible, rawHasBuildError),
    [bible, rawHasBuildError],
  );

  const queryErrorMsg =
    bibleQuery.error instanceof Error
      ? bibleQuery.error.message
      : bibleQuery.error
        ? String(bibleQuery.error)
        : '';
  const visiblePrimaryError =
    actionError ||
    (!isDraftSession && bibleQuery.error && queryErrorMsg !== dismissedQueryErrorMsg
      ? bibleQuery.error
      : null);

  const dismissBuildError = (key: string) => {
    setDismissedBuildErrorKeys((prev) => {
      const next = new Set(prev);
      next.add(key);
      try {
        window.sessionStorage.setItem(
          'studio-web.dismissedBuildErrors',
          JSON.stringify([...next].slice(-40)),
        );
      } catch {
        // Ignore storage quota errors
      }
      return next;
    });
  };

  const showSliceBar =
    !isDraftSession && (activeWorkspace === 'board' || activeWorkspace === 'writers-room');

  return (
    <div className="app-shell" data-workspace={activeWorkspace}>
      <TopCommandBar
        games={games}
        selectedCartridgeId={effectiveCartridgeId}
        activeCartridgeId={playtestListQuery.data?.active_cartridge_id ?? ''}
        onSelectCartridge={handleSelectCartridge}
        activeWorkspace={activeWorkspace}
        onSelectWorkspace={setActiveWorkspace}
        bible={bible}
        boardCounts={boardCounts}
        transcriptCount={sliceTranscript.length}
        assetsCount={assetsCount}
        playtestRunsCount={playtestRunsForSelected}
        pendingWishesCount={playtestListQuery.data?.pending_wishes_count ?? 0}
        wsConnected={realtime.connected}
        recentEventCount={realtime.recentEvents.length}
        onActivateGame={() => activateMut.mutate()}
        activatingGame={activateMut.isPending}
        onLaunchGame={() => launchMut.mutate(undefined)}
        launchingGame={launchMut.isPending}
        launchStatusText={launchStatusText}
        onPauseOrResume={(pause) => pauseMut.mutate(pause)}
        togglingPause={pauseMut.isPending}
        onApproveGreenlight={() => steerOrGreenlightMut.mutate({ approved: true, feedback: '' })}
        hasBuildError={rawHasBuildError}
        livePhaseBadge={liveActivity.isActive ? liveActivity.phaseBadge : undefined}
        onResumeBuild={() => genesisMut.mutate({ resumeCurrent: true })}
        resumingBuild={genesisMut.isPending}
        uiScalePct={uiScalePct}
        onChangeUiScalePct={setUiScalePct}
        onOpenNextSliceModal={() => setNextSliceOpen(true)}
        onStartNewGame={handleStartNewGameSession}
        startingNewGame={startNewGameDraftMut.isPending}
        onOpenCheckpointsModal={() => setCheckpointsOpen(true)}
        onOpenLivePulseModal={() => setLivePulseOpen(true)}
        isDraftSession={isDraftSession}
        activeSessionTitle={activeSessionMeta?.title}
      />

      <div>
        {showSliceBar ? (
          <SliceSubbar
            slices={sliceCtx.slices}
            selectedSlice={sliceCtx.selectedSlice}
            selectedSliceId={selectedSliceId}
            onSelectSliceId={setSelectedSliceId}
            allBoardItems={allBoardItems}
            isBuildRunning={isBuildRunning || isAwaitingGreenlight}
            steeringNotice={steeringNotice}
            onSteerCouncil={handleSteerCouncil}
            steeringBusy={steeringBusy}
          />
        ) : null}

        {!isDraftSession ? (
          <LiveActivityBanner
            activity={liveActivity}
            activeWorkspace={activeWorkspace}
            onSelectWorkspace={setActiveWorkspace}
            onOpenLivePulse={() => setLivePulseOpen(true)}
          />
        ) : null}

        {(visiblePrimaryError || visibleBuildError) ? (
          <div style={{ padding: '8px 14px', display: 'grid', gap: '6px' }}>
            {visiblePrimaryError ? (
              <ErrorBanner
                error={visiblePrimaryError}
                onRetry={
                  bibleQuery.error && !actionError
                    ? () => {
                        setDismissedQueryErrorMsg(null);
                        void bibleQuery.refetch();
                      }
                    : undefined
                }
                onDismiss={() => {
                  if (actionError) {
                    setActionError(null);
                  } else if (queryErrorMsg) {
                    setDismissedQueryErrorMsg(queryErrorMsg);
                  }
                }}
              />
            ) : null}
            {visibleBuildError ? (
              <ErrorBanner
                error={
                  new Error(
                    `[${visibleBuildError.agent_name} · ${visibleBuildError.event_type}] ${visibleBuildError.message}`,
                  )
                }
                title="Live Studio Circuit Breaker / Build Error"
                onDismiss={() => dismissBuildError(lastBuildErrorKey)}
              />
            ) : null}
          </div>
        ) : null}
      </div>

      <main className="workspace-main">
        <WorkspaceErrorBoundary
          key={
            activeWorkspace === 'coproducer'
              ? 'coproducer'
              : `${activeWorkspace}:${effectiveCartridgeId}`
          }
          workspaceName={activeWorkspace}
        >
          {activeWorkspace === 'coproducer' ? (
            <CoProducerView
              cartridgeId={resolvedCartridgeId}
              hasAnyCartridge={hasAnyCartridge}
              gameTitle={bible?.game_title}
              orchestratorState={
                bible?.process_status?.process_state || bible?.orchestrator_state
              }
              boardCounts={boardCounts}
              liveStepsByThread={realtime.liveCoProducerSteps}
              onSelectCartridge={handleSelectCartridge}
              onSelectWorkspace={setActiveWorkspace}
              onOpenLightbox={setLightboxMedia}
              requestedDraftThreadId={requestedDraftThreadId}
              onConsumeRequestedDraftThread={() => setRequestedDraftThreadId(null)}
              defaultPauseForGreenlight={Boolean(bible?.pause_for_greenlight)}
              onRunAutonomousBuild={async ({ creativePrompt, pauseForGreenlight }) => {
                await genesisMut.mutateAsync({
                  resumeCurrent: false,
                  creativePrompt,
                  pauseForGreenlight,
                });
              }}
              runningAutonomousBuild={genesisMut.isPending}
              sessions={unifiedSessions}
              selectedSession={selectedSession}
              onSelectSession={handleSelectSession}
              onStartNewGame={handleStartNewGameSession}
              startingNewGame={startNewGameDraftMut.isPending}
            />
          ) : null}

          {activeWorkspace === 'board' ? (
            <ProductionBoardView
              items={sliceBoardItems}
              cartridgeId={resolvedCartridgeId}
              isBuildRunning={isBuildRunning}
              liveStepsByItem={isBuildRunning ? realtime.liveStepsByItem : EMPTY_LIVE_STEPS}
              liveActivity={liveActivity}
              onOpenLivePulse={() => setLivePulseOpen(true)}
              onOpenLightbox={setLightboxMedia}
            />
          ) : null}

          {activeWorkspace === 'writers-room' ? (
            <WritersRoomView
              utterances={sliceTranscript}
              cartridgeId={resolvedCartridgeId}
              isBuildRunning={isBuildRunning}
              awaitingGreenlight={isAwaitingGreenlight}
              liveActivity={liveActivity}
              onOpenLivePulse={() => setLivePulseOpen(true)}
              onSteer={handleSteerCouncil}
              onGreenlight={async (feedback) => {
                await steerOrGreenlightMut.mutateAsync({ approved: true, feedback });
              }}
              busy={steeringBusy}
              onOpenLightbox={setLightboxMedia}
            />
          ) : null}

          {activeWorkspace === 'bible' ? (
            <GameBibleView
              bible={bible}
              cartridgeId={resolvedCartridgeId}
              onOpenLightbox={setLightboxMedia}
            />
          ) : null}

          {activeWorkspace === 'memory' ? (
            <GameMemoryView
              cartridgeId={resolvedCartridgeId}
              gameTitle={bible?.game_title}
              activeSliceId={sliceCtx.currentSlice?.slice_id}
            />
          ) : null}

          {activeWorkspace === 'assets' ? (
            <AssetsAndCodeView
              cartridgeId={resolvedCartridgeId}
              bible={bible}
              onOpenLightbox={setLightboxMedia}
            />
          ) : null}

          {activeWorkspace === 'playtest' ? (
            <PlaytestLabView
              selectedCartridgeId={resolvedCartridgeId}
              sessionsListData={playtestListQuery.data}
              sessionsError={playtestListQuery.error}
              onRefetchSessions={() => void playtestListQuery.refetch()}
              livePlaytestFrames={realtime.livePlaytestFrames}
              canResumeCouncilPlaytest={
                Boolean(bible?.has_bible) &&
                !isBuildRunning &&
                !isAwaitingGreenlight &&
                (sliceVerification.isAwaitingCouncilPlaytest || rawHasBuildError)
              }
              onResumeCouncilPlaytest={() => genesisMut.mutate({ resumeCurrent: true })}
              resumingBuild={genesisMut.isPending}
            />
          ) : null}

          {activeWorkspace === 'dna' ? <PlayerDnaView bible={bible} /> : null}
        </WorkspaceErrorBoundary>
      </main>

      <NextSliceModal
        open={nextSliceOpen}
        onClose={() => setNextSliceOpen(false)}
        gameTitle={bible?.game_title ?? ''}
        currentSliceNumber={sliceCtx.currentSlice?.slice_index ?? 0}
        onSubmit={async (feedback) => {
          await genesisMut.mutateAsync({
            resumeCurrent: true,
            creativePrompt: feedback,
          });
        }}
        busy={genesisMut.isPending}
      />

      <CheckpointsModal
        open={checkpointsOpen}
        onClose={() => setCheckpointsOpen(false)}
        selectedCartridgeId={resolvedCartridgeId}
        onLaunchCheckpoint={(cpId) => launchMut.mutate(cpId)}
      />

      <LivePulseModal
        open={livePulseOpen}
        onClose={() => setLivePulseOpen(false)}
        wsConnected={realtime.connected}
        lastEventAt={realtime.lastEventAt}
        recentEvents={realtime.recentEvents}
        studioStatus={studioStatusQuery.data}
        liveActivity={liveActivity}
      />

      <MediaLightbox media={lightboxMedia} onClose={() => setLightboxMedia(null)} />
    </div>
  );
}
