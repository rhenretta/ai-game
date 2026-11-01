import { type CSSProperties, useEffect, useId, useMemo, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ChevronLeft,
  ChevronRight,
  Film,
  Pause,
  Play,
  Radio,
  RotateCcw,
  Send,
  SkipBack,
  Sparkles,
  Square,
  Trash2,
} from 'lucide-react';
import { buildApiUrl, studioApi } from '../../api/client';
import type { PlaytestSessionsListResponse, PlaytestTurn } from '../../api/types';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';
import { ErrorBanner } from '../../components/ErrorBanner';
import {
  extractCleanTurnObservation,
  formatPlaytestMotorAction,
  parseBugEntries,
  parseFieldNoteEntries,
  parseQuestionEntries,
  parseRulebookEntries,
} from '../../domain/playtestNotebook';
import type { LivePlaytestFrame } from '../../realtime/useSwarmSocket';

export interface PlaytestLabViewProps {
  selectedCartridgeId: string;
  sessionsListData: PlaytestSessionsListResponse | undefined;
  sessionsError: unknown;
  onRefetchSessions: () => void;
  livePlaytestFrames?: Record<string, LivePlaytestFrame>;
  canResumeCouncilPlaytest?: boolean;
  onResumeCouncilPlaytest?: () => void;
  resumingBuild?: boolean;
}

type StageMode = 'video' | 'frames' | 'live';
type InspectorTab = 'turns' | 'report' | 'rulebook' | 'wishes';
type DvrSpeed = 0.5 | 1 | 2;

function formatClock(sec: number | undefined): string {
  const s = Math.max(0, Number(sec || 0));
  const mins = Math.floor(s / 60);
  const rem = (s - mins * 60).toFixed(1).padStart(4, '0');
  return `${String(mins).padStart(2, '0')}:${rem}`;
}

export function PlaytestLabView({
  selectedCartridgeId,
  sessionsListData,
  sessionsError,
  onRefetchSessions,
  livePlaytestFrames,
  canResumeCouncilPlaytest = false,
  onResumeCouncilPlaytest,
  resumingBuild = false,
}: PlaytestLabViewProps) {
  const queryClient = useQueryClient();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const dvrSliderId = useId();
  const reportNoteInputId = useId();
  const includeWishesInputId = useId();

  const [targetId, setTargetId] = useState<string>('active');
  const [maxTurns, setMaxTurns] = useState<number>(120);
  const [enableAudio, setEnableAudio] = useState<boolean>(true);

  const [gameScope, setGameScope] = useState<'selected' | 'all'>('selected');
  const [sourceScope, setSourceScope] = useState<'all' | 'council' | 'operator'>('all');
  const [sliceScope, setSliceScope] = useState<string>('all');

  const [selectedSessionId, setSelectedSessionId] = useState<string>('');
  const [stageMode, setStageMode] = useState<StageMode>('video');
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>('turns');
  const [selectedTurnIdx, setSelectedTurnIdx] = useState<number>(0);
  const [isDvrPlaying, setIsDvrPlaying] = useState<boolean>(false);
  const [dvrSpeed, setDvrSpeed] = useState<DvrSpeed>(1);
  const [logSortOrder, setLogSortOrder] = useState<'newest' | 'oldest'>('newest');

  const [selectedWishIds, setSelectedWishIds] = useState<Set<string>>(new Set());
  const [producerNote, setProducerNote] = useState<string>('');
  const [reportProducerNote, setReportProducerNote] = useState<string>('');
  const [includePendingWishes, setIncludePendingWishes] = useState<boolean>(true);
  const [lastReportDispatchStatus, setLastReportDispatchStatus] = useState<{
    sessionId: string;
    status: string;
  } | null>(null);

  const allSessions = sessionsListData?.sessions ?? [];
  const activeCartId = sessionsListData?.active_cartridge_id ?? '';
  const targets = sessionsListData?.playable_targets ?? [];

  const groupedTargets = useMemo(() => {
    const activeTargets = targets.filter((t) => t.target_type === 'active');
    const selectedSlices = targets.filter(
      (t) =>
        t.target_type === 'completed_slice' &&
        (!selectedCartridgeId || selectedCartridgeId === 'active' || t.candidate_id === selectedCartridgeId),
    );
    const otherSlices = targets.filter(
      (t) =>
        t.target_type === 'completed_slice' &&
        selectedCartridgeId &&
        selectedCartridgeId !== 'active' &&
        t.candidate_id !== selectedCartridgeId,
    );
    const checkpoints = targets
      .filter((t) => t.target_type !== 'active' && t.target_type !== 'completed_slice')
      .filter((t) => !selectedCartridgeId || selectedCartridgeId === 'active' || t.candidate_id === selectedCartridgeId)
      .slice(0, 25);

    return { activeTargets, selectedSlices, otherSlices, checkpoints };
  }, [targets, selectedCartridgeId]);

  const gameMatchedSessions = useMemo(() => {
    if (gameScope === 'all') return allSessions;
    const effCart = selectedCartridgeId || activeCartId || 'active';
    return allSessions.filter((s) => {
      if (!effCart || effCart === 'active') {
        return Boolean(
          s.matches_active_cartridge ||
            (activeCartId && s.candidate_id === activeCartId) ||
            s.checkpoint_id === 'active',
        );
      }
      return Boolean(
        s.candidate_id === effCart || (effCart === activeCartId && s.matches_active_cartridge),
      );
    });
  }, [allSessions, gameScope, selectedCartridgeId, activeCartId]);

  const sliceOptions = useMemo(() => {
    const map = new Map<string, number>();
    let generalCount = 0;
    for (const s of gameMatchedSessions) {
      if (s.slice_id) {
        map.set(s.slice_id, (map.get(s.slice_id) ?? 0) + 1);
      } else {
        generalCount += 1;
      }
    }
    return { slices: Array.from(map.entries()), generalCount };
  }, [gameMatchedSessions]);

  const filteredSessions = useMemo(() => {
    return gameMatchedSessions.filter((s) => {
      if (sourceScope === 'council' && s.initiated_by !== 'council') return false;
      if (sourceScope === 'operator' && s.initiated_by === 'council') return false;
      if (sliceScope === '__general__' && s.slice_id) return false;
      if (sliceScope !== 'all' && sliceScope !== '__general__' && s.slice_id !== sliceScope) return false;
      return true;
    });
  }, [gameMatchedSessions, sourceScope, sliceScope]);

  const activeSessionSummary = useMemo(() => {
    if (selectedSessionId) {
      const match = allSessions.find((s) => s.session_id === selectedSessionId);
      if (match) return match;
    }
    return filteredSessions[0] ?? allSessions[0] ?? null;
  }, [allSessions, filteredSessions, selectedSessionId]);

  const effectiveSessionId = activeSessionSummary?.session_id ?? '';

  const detailQuery = useQuery({
    queryKey: ['playtest-session-detail', effectiveSessionId],
    queryFn: () => studioApi.getPlaytestSessionDetail(effectiveSessionId),
    enabled: Boolean(effectiveSessionId),
    refetchInterval: (query) => {
      const st = query.state.data?.session?.status ?? activeSessionSummary?.status;
      return st === 'running' || st === 'starting' || st === 'stopping' ? 2000 : false;
    },
  });

  const effectiveSession =
    detailQuery.data?.session && detailQuery.data.session.session_id === effectiveSessionId
      ? detailQuery.data.session
      : activeSessionSummary;
  const effectiveStatus = effectiveSession?.status;
  const isSessionLive =
    effectiveStatus === 'running' ||
    effectiveStatus === 'starting' ||
    effectiveStatus === 'stopping';
  const isSessionPaused = effectiveStatus === 'paused';
  const effectiveHasVideo = Boolean(effectiveSession?.has_video);

  const detail = detailQuery.data;
  const turns = detail?.turns ?? [];
  const wishes = detail?.wishes ?? [];
  const report = detail?.session?.session_report ?? activeSessionSummary?.session_report;

  const liveFrame = effectiveSessionId ? livePlaytestFrames?.[effectiveSessionId] : undefined;
  const liveFrameName =
    liveFrame?.frameName ||
    effectiveSession?.latest_frame_name ||
    turns[turns.length - 1]?.screenshot_name ||
    '';
  const liveFrameCacheKey = liveFrame
    ? `${liveFrame.tick}-${liveFrame.seq}`
    : `${effectiveSession?.sim_ticks_total ?? 0}-${effectiveSession?.turn_count ?? turns.length}-${effectiveSession?.updated_at_utc ?? ''}-${detailQuery.dataUpdatedAt}`;

  const hasLiveEdge = Boolean((isSessionLive || isSessionPaused) && liveFrameName);

  // Preserve manual DVR replay scrubbing across live polling refetches, resetting only on session or lifecycle transitions
  const prevLifecycleRef = useRef<{
    sessionId: string;
    wasLiveOrPaused: boolean;
    hadVideo: boolean;
  } | null>(null);

  useEffect(() => {
    if (!effectiveSessionId) {
      prevLifecycleRef.current = null;
      return;
    }
    const liveOrPaused = isSessionLive || isSessionPaused;
    const prev = prevLifecycleRef.current;

    if (!prev || prev.sessionId !== effectiveSessionId) {
      setIsDvrPlaying(false);
      if (liveOrPaused) {
        setStageMode('live');
      } else if (effectiveHasVideo) {
        setStageMode('video');
      } else {
        setStageMode('frames');
      }
      prevLifecycleRef.current = {
        sessionId: effectiveSessionId,
        wasLiveOrPaused: liveOrPaused,
        hadVideo: effectiveHasVideo,
      };
      return;
    }

    if (!prev.wasLiveOrPaused && liveOrPaused) {
      setIsDvrPlaying(false);
      setStageMode('live');
    } else if (prev.wasLiveOrPaused && !liveOrPaused) {
      setIsDvrPlaying(false);
      setStageMode((curr) => (curr === 'live' ? (effectiveHasVideo ? 'video' : 'frames') : curr));
    } else if (!prev.hadVideo && effectiveHasVideo && !liveOrPaused) {
      setStageMode((curr) => (curr === 'live' ? 'video' : curr));
    }

    prevLifecycleRef.current = {
      sessionId: effectiveSessionId,
      wasLiveOrPaused: liveOrPaused,
      hadVideo: effectiveHasVideo,
    };
  }, [effectiveSessionId, isSessionLive, isSessionPaused, effectiveHasVideo]);

  // DVR auto-playback timer: steps through captured turns up to the live edge, then hands off seamlessly to Live
  useEffect(() => {
    if (!isDvrPlaying) return;
    if (turns.length === 0) {
      setIsDvrPlaying(false);
      return;
    }
    const intervalMs = Math.round(850 / dvrSpeed);
    const timer = window.setInterval(() => {
      setSelectedTurnIdx((curr) => {
        if (curr < turns.length - 1) {
          return curr + 1;
        }
        setIsDvrPlaying(false);
        if (hasLiveEdge) {
          setStageMode('live');
        }
        return curr;
      });
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [isDvrPlaying, dvrSpeed, turns.length, hasLiveEdge]);

  const startMut = useMutation({
    mutationFn: () => {
      const matchedTarget = targets.find(
        (t) => `${t.checkpoint_id}::${t.slice_id ?? ''}` === targetId || t.checkpoint_id === targetId,
      );
      return studioApi.startPlaytestSession({
        checkpointId: matchedTarget?.checkpoint_id ?? targetId,
        candidateId:
          matchedTarget?.candidate_id ?? (selectedCartridgeId !== 'active' ? selectedCartridgeId : undefined),
        sliceId: matchedTarget?.slice_id,
        maxTurns,
        enableAudio,
      });
    },
    onSuccess: (res) => {
      setSelectedSessionId(res.session.session_id);
      setSelectedTurnIdx(0);
      setIsDvrPlaying(false);
      void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });
    },
  });

  const controlMut = useMutation({
    mutationFn: (action: 'pause' | 'resume' | 'stop' | 'reset') =>
      studioApi.controlPlaytestSession(effectiveSessionId, action),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });
      void queryClient.invalidateQueries({ queryKey: ['playtest-session-detail', effectiveSessionId] });
    },
  });

  const sendWishesMut = useMutation({
    mutationFn: (wishIds: string[]) => studioApi.sendWishesToCouncil({ wishIds, producerNote }),
    onSuccess: () => {
      setSelectedWishIds(new Set());
      setProducerNote('');
      void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });
      void queryClient.invalidateQueries({ queryKey: ['playtest-session-detail', effectiveSessionId] });
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
    },
  });

  const dismissWishMut = useMutation({
    mutationFn: (wishId: string) => studioApi.dismissWish(wishId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });
      void queryClient.invalidateQueries({ queryKey: ['playtest-session-detail', effectiveSessionId] });
    },
  });

  const sendReportMut = useMutation({
    mutationFn: () =>
      studioApi.sendPlaytestReportToCouncil({
        sessionId: effectiveSessionId,
        includePendingWishes,
        producerNote: reportProducerNote,
      }),
    onSuccess: (res) => {
      setReportProducerNote('');
      setLastReportDispatchStatus({
        sessionId: effectiveSessionId,
        status: res.status,
      });
      void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });
      void queryClient.invalidateQueries({ queryKey: ['playtest-session-detail', effectiveSessionId] });
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
      void queryClient.invalidateQueries({ queryKey: ['studio-status'] });
    },
  });

  const clampedTurnIdx = turns.length > 0 ? Math.min(selectedTurnIdx, turns.length - 1) : 0;
  const activeTurn: PlaytestTurn | undefined = turns[clampedTurnIdx];
  const latestTurn: PlaytestTurn | undefined = turns[turns.length - 1];
  const activeActionInfo = useMemo(() => formatPlaytestMotorAction(activeTurn), [activeTurn]);
  const latestActionInfo = useMemo(() => formatPlaytestMotorAction(latestTurn), [latestTurn]);

  const isViewingLiveFeed = stageMode === 'live' && Boolean(liveFrameName);
  const isReplayingTurn =
    stageMode === 'frames' &&
    Boolean(activeTurn) &&
    (hasLiveEdge || clampedTurnIdx < Math.max(0, turns.length - 1) || isDvrPlaying);

  const seekToTurn = (turn: PlaytestTurn, idx: number) => {
    setIsDvrPlaying(false);
    setSelectedTurnIdx(idx);
    if (stageMode === 'video' && effectiveHasVideo && videoRef.current && turn.video_timestamp_sec !== undefined) {
      videoRef.current.currentTime = turn.video_timestamp_sec;
    } else {
      setStageMode('frames');
    }
  };

  const jumpToLive = () => {
    setIsDvrPlaying(false);
    if (turns.length > 0) {
      setSelectedTurnIdx(turns.length - 1);
    }
    setStageMode('live');
  };

  const toggleDvrReplay = () => {
    if (turns.length === 0) return;
    if (isDvrPlaying) {
      setIsDvrPlaying(false);
      return;
    }
    if (stageMode === 'live' || stageMode === 'video' || clampedTurnIdx >= turns.length - 1) {
      setSelectedTurnIdx(0);
    }
    setStageMode('frames');
    setIsDvrPlaying(true);
  };

  // DVR Slider bounds: 0 .. turns.length - 1, plus an extra slot at `turns.length` representing LIVE when active
  const sliderMax = hasLiveEdge ? turns.length : Math.max(0, turns.length - 1);
  const sliderValue = hasLiveEdge && isViewingLiveFeed ? turns.length : clampedTurnIdx;
  const sliderValueText =
    hasLiveEdge && isViewingLiveFeed
      ? `Live feed (Turn ${effectiveSession?.turn_count ?? turns.length})`
      : activeTurn
        ? `Turn ${activeTurn.turn_index} of ${turns.length} at ${formatClock(activeTurn.video_timestamp_sec)}`
        : 'No turns recorded yet';

  const handleDvrSliderChange = (rawVal: number) => {
    setIsDvrPlaying(false);
    if (hasLiveEdge && rawVal >= turns.length) {
      setStageMode('live');
      return;
    }
    const bounded = Math.max(0, Math.min(turns.length - 1, rawVal));
    setSelectedTurnIdx(bounded);
    const targetTurn = turns[bounded];
    if (stageMode === 'video' && effectiveHasVideo && videoRef.current && targetTurn?.video_timestamp_sec !== undefined) {
      videoRef.current.currentTime = targetTurn.video_timestamp_sec;
    } else {
      setStageMode('frames');
    }
  };

  // Ordered Activity Log entries (defaulting to Latest First so recent actions are immediately visible)
  const orderedTurns = useMemo(() => {
    const indexed = turns.map((turn, idx) => ({ turn, idx }));
    return logSortOrder === 'newest' ? [...indexed].reverse() : indexed;
  }, [turns, logSortOrder]);

  // Structured Notebook data (fixes snake_case question keys, hidden answers, and raw JSON bug objects)
  const parsedBugs = useMemo(() => parseBugEntries(detail?.session?.bugs_found), [detail?.session?.bugs_found]);
  const parsedRulebook = useMemo(() => parseRulebookEntries(detail?.session?.rulebook), [detail?.session?.rulebook]);
  const parsedQuestions = useMemo(
    () => parseQuestionEntries(detail?.session?.questions),
    [detail?.session?.questions],
  );
  const parsedFieldNotes = useMemo(
    () => parseFieldNoteEntries(detail?.session?.field_notes),
    [detail?.session?.field_notes],
  );
  const pendingSessionWishes = useMemo(() => wishes.filter((w) => w.status === 'pending'), [wishes]);
  const reportSentAtUtc =
    effectiveSession?.report_sent_to_council_at_utc || report?.dispatched_to_council_at_utc || null;
  const canDispatchFullReport = Boolean(
    effectiveSessionId &&
      !isSessionLive &&
      !isSessionPaused &&
      (report || turns.length > 0 || parsedBugs.length > 0 || wishes.length > 0),
  );

  // Front-and-center Focus Banner content
  const liveCurrentPlan =
    effectiveSession?.current_plan?.trim() ||
    latestTurn?.current_plan?.trim() ||
    (isSessionLive ? 'Observing opening screen and choosing first interaction...' : '');
  const latestCleanObs = extractCleanTurnObservation(latestTurn?.append_log_line) || latestTurn?.mood_note || '';
  const activeCleanObs = extractCleanTurnObservation(activeTurn?.append_log_line) || activeTurn?.mood_note || '';
  const focusHeadline =
    isReplayingTurn && activeTurn
      ? activeTurn.current_plan?.trim() || activeCleanObs || 'Inspecting turn frame.'
      : liveCurrentPlan;

  return (
    <div className="scroll-pane" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <ErrorBanner
        error={
          sessionsError ||
          detailQuery.error ||
          startMut.error ||
          controlMut.error ||
          sendWishesMut.error ||
          sendReportMut.error
        }
        onRetry={onRefetchSessions}
      />

      {/* Session Launcher & Control Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '8px',
          padding: '10px 12px',
          borderRadius: 'var(--radius-md)',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        <select
          aria-label="Select Playable Slice or Checkpoint Target"
          className="select-control"
          style={{ maxInlineSize: '360px', flex: '1 1 240px' }}
          value={targetId}
          onChange={(e) => setTargetId(e.target.value)}
        >
          {groupedTargets.activeTargets.length > 0 ? (
            <optgroup label="Active Cartridge">
              {groupedTargets.activeTargets.map((t) => (
                <option key={`act-${t.checkpoint_id}`} value={`${t.checkpoint_id}::${t.slice_id ?? ''}`}>
                  {t.label}
                </option>
              ))}
            </optgroup>
          ) : (
            <option value="active">Active Cartridge</option>
          )}
          {groupedTargets.selectedSlices.length > 0 ? (
            <optgroup label="Completed Vertical Slices (Selected Game)">
              {groupedTargets.selectedSlices.map((t) => (
                <option key={`sl-${t.checkpoint_id}-${t.slice_id}`} value={`${t.checkpoint_id}::${t.slice_id ?? ''}`}>
                  {t.label}
                </option>
              ))}
            </optgroup>
          ) : null}
          {groupedTargets.checkpoints.length > 0 ? (
            <optgroup label="Verified Checkpoints (Selected Game)">
              {groupedTargets.checkpoints.map((t) => (
                <option key={`cp-${t.checkpoint_id}`} value={`${t.checkpoint_id}::${t.slice_id ?? ''}`}>
                  {t.label}
                </option>
              ))}
            </optgroup>
          ) : null}
          {groupedTargets.otherSlices.length > 0 ? (
            <optgroup label="Other Games' Slices">
              {groupedTargets.otherSlices.map((t) => (
                <option key={`oth-${t.checkpoint_id}-${t.slice_id}`} value={`${t.checkpoint_id}::${t.slice_id ?? ''}`}>
                  {t.label}
                </option>
              ))}
            </optgroup>
          ) : null}
        </select>

        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', fontSize: '0.87rem' }}>
          <span>Max Turns:</span>
          <input
            type="number"
            min={5}
            max={500}
            className="input-control"
            style={{ inlineSize: '68px' }}
            value={maxTurns}
            onChange={(e) => setMaxTurns(Number(e.target.value) || 120)}
          />
        </label>

        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', fontSize: '0.87rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={enableAudio}
            onChange={(e) => setEnableAudio(e.target.checked)}
          />
          <span>Audio</span>
        </label>

        <button
          type="button"
          className="btn btn-success"
          disabled={startMut.isPending}
          onClick={() => startMut.mutate()}
        >
          <Play size={13} aria-hidden="true" />
          {startMut.isPending ? 'Starting...' : 'Run New Session'}
        </button>

        {effectiveSessionId ? (
          <div style={{ marginInlineStart: 'auto', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              className={`badge ${
                isSessionLive
                  ? 'badge-accent'
                  : effectiveStatus === 'error'
                    ? 'badge-danger'
                    : effectiveStatus === 'stopped' || effectiveStatus === 'paused'
                      ? 'badge-warn'
                      : 'badge-ok'
              }`}
            >
              {effectiveStatus}
            </span>
            {isSessionPaused ? (
              <button
                type="button"
                className="btn"
                disabled={controlMut.isPending}
                onClick={() => controlMut.mutate('resume')}
              >
                <Play size={12} aria-hidden="true" /> Resume
              </button>
            ) : (
              <button
                type="button"
                className="btn"
                disabled={!isSessionLive || controlMut.isPending}
                onClick={() => controlMut.mutate('pause')}
              >
                <Pause size={12} aria-hidden="true" /> Pause
              </button>
            )}
            <button
              type="button"
              className="btn"
              disabled={controlMut.isPending}
              onClick={() => controlMut.mutate('stop')}
            >
              <Square size={12} aria-hidden="true" /> Stop
            </button>
            <button
              type="button"
              className="btn"
              disabled={controlMut.isPending}
              onClick={() => controlMut.mutate('reset')}
            >
              <RotateCcw size={12} aria-hidden="true" /> Reset
            </button>
          </div>
        ) : null}
      </div>

      {/* Recorded Sessions Strip */}
      <div
        style={{
          padding: '10px 12px',
          borderRadius: 'var(--radius-md)',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-subtle)',
          display: 'grid',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
          <div className="pill-list">
            <button
              type="button"
              className={`pill${gameScope === 'selected' ? ' is-active' : ''}`}
              onClick={() => setGameScope('selected')}
            >
              Selected Game ({gameMatchedSessions.length})
            </button>
            <button
              type="button"
              className={`pill${gameScope === 'all' ? ' is-active' : ''}`}
              onClick={() => setGameScope('all')}
            >
              All Games ({allSessions.length})
            </button>

            {sliceOptions.slices.length > 0 || sliceOptions.generalCount > 0 ? (
              <>
                <span style={{ color: 'var(--border-strong)' }}>|</span>
                <button
                  type="button"
                  className={`pill${sliceScope === 'all' ? ' is-active' : ''}`}
                  onClick={() => setSliceScope('all')}
                >
                  All Slices
                </button>
                {sliceOptions.slices.map(([sid, cnt]) => (
                  <button
                    key={sid}
                    type="button"
                    className={`pill${sliceScope === sid ? ' is-active' : ''}`}
                    onClick={() => setSliceScope(sid)}
                  >
                    {sid.replace(/^slice_(\d+).*/, 'Slice #$1')} ({cnt})
                  </button>
                ))}
                {sliceOptions.generalCount > 0 ? (
                  <button
                    type="button"
                    className={`pill${sliceScope === '__general__' ? ' is-active' : ''}`}
                    onClick={() => setSliceScope('__general__')}
                  >
                    Active / General ({sliceOptions.generalCount})
                  </button>
                ) : null}
              </>
            ) : null}
          </div>

          <div className="pill-list">
            {(['all', 'council', 'operator'] as const).map((src) => (
              <button
                key={src}
                type="button"
                className={`pill${sourceScope === src ? ' is-active' : ''}`}
                onClick={() => setSourceScope(src)}
              >
                {src === 'all' ? 'All Runs' : src === 'council' ? 'Council Gates' : 'Manual'}
              </button>
            ))}
          </div>
        </div>

        {filteredSessions.length === 0 ? (
          <div style={{ fontSize: '0.87rem', color: 'var(--text-muted)', paddingBlock: '4px' }}>
            No recorded playtest runs match this filter.
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBlockEnd: '4px' }}>
            {filteredSessions.map((rawSession) => {
              const selected = rawSession.session_id === effectiveSessionId;
              const s = selected && effectiveSession ? { ...rawSession, ...effectiveSession } : rawSession;
              return (
                <button
                  key={s.session_id}
                  type="button"
                  onClick={() => {
                    setSelectedSessionId(s.session_id);
                    setSelectedTurnIdx(0);
                    setIsDvrPlaying(false);
                  }}
                  style={{
                    flex: '0 0 auto',
                    minInlineSize: '230px',
                    maxInlineSize: '280px',
                    textAlign: 'left',
                    padding: '8px 10px',
                    borderRadius: 'var(--radius-sm)',
                    border: `2px solid ${selected ? 'var(--accent)' : 'var(--border-subtle)'}`,
                    background: selected
                      ? 'color-mix(in srgb, var(--accent) 12%, var(--bg-elevated))'
                      : 'var(--bg-elevated)',
                    cursor: 'pointer',
                    display: 'grid',
                    gap: '3px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '6px' }}>
                    <span className={`badge ${s.initiated_by === 'council' ? 'badge-purple' : 'badge-accent'}`}>
                      {s.initiated_by === 'council' ? 'Council' : 'Manual'}
                    </span>
                    <span style={{ fontSize: '0.76rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      {s.turn_count}t{s.overall_fun_rating ? ` · ★${s.overall_fun_rating}/10` : ''}
                      {s.report_sent_to_council_at_utc && s.initiated_by !== 'council' ? ' · ✓ Council' : ''}
                    </span>
                  </div>
                  <div className="clamp-2" style={{ fontWeight: 700, fontSize: '0.87rem' }}>
                    {s.slice_title || s.game_title}
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                    {s.status} · {s.started_at_utc ? new Date(s.started_at_utc).toLocaleString() : s.session_id}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {(activeSessionSummary?.status === 'error' || (canResumeCouncilPlaytest && onResumeCouncilPlaytest)) ? (
        <div
          role="region"
          aria-label="Playtest session status and recovery"
          style={{
            padding: '10px 12px',
            borderRadius: 'var(--radius-md)',
            background:
              activeSessionSummary?.status === 'error'
                ? 'color-mix(in srgb, var(--danger) 12%, var(--bg-surface))'
                : 'color-mix(in srgb, var(--warn) 12%, var(--bg-surface))',
            border: `1px solid ${
              activeSessionSummary?.status === 'error'
                ? 'color-mix(in srgb, var(--danger) 42%, var(--border-subtle))'
                : 'color-mix(in srgb, var(--warn) 42%, var(--border-subtle))'
            }`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '10px',
          }}
        >
          <div style={{ display: 'grid', gap: '3px', minInlineSize: 0, flex: '1 1 320px' }}>
            <div style={{ fontWeight: 700, fontSize: '0.91rem' }}>
              {activeSessionSummary?.status === 'error'
                ? `${activeSessionSummary.initiated_by === 'council' ? 'Council Blind Playtest' : 'Playtest Session'} Interrupted`
                : 'Current Slice Awaiting Council Blind Playtest'}
            </div>
            <div style={{ fontSize: '0.87rem', color: 'var(--text-secondary)', overflowWrap: 'anywhere' }}>
              {detail?.session?.error_message ||
                activeSessionSummary?.error_message ||
                detail?.session?.end_summary ||
                activeSessionSummary?.end_summary ||
                'All slice tasks on the Production Board are approved. Resume the slice build to run the automated Council Blind Playtest.'}
            </div>
          </div>
          {canResumeCouncilPlaytest && onResumeCouncilPlaytest ? (
            <button
              type="button"
              className="btn btn-success"
              disabled={resumingBuild}
              onClick={onResumeCouncilPlaytest}
            >
              <Play size={13} aria-hidden="true" />
              {resumingBuild ? 'Resuming...' : 'Resume Slice & Run Council Playtest'}
            </button>
          ) : null}
        </div>
      ) : null}

      {/* Front-and-Center Current Player Focus Banner */}
      {activeSessionSummary && Boolean(focusHeadline) ? (
        <section
          aria-label="Current playtester focus"
          className="playtest-focus-banner"
          data-mode={isReplayingTurn ? 'replay' : 'live'}
        >
          <div className="playtest-focus-main">
            <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
              {isReplayingTurn && activeTurn ? (
                <span className="badge badge-warn">
                  Replaying Turn #{activeTurn.turn_index} of {turns.length} · {formatClock(activeTurn.video_timestamp_sec)}
                </span>
              ) : isSessionLive ? (
                <span className="badge badge-ok">
                  Currently Trying · Turn #{effectiveSession?.turn_count ?? turns.length}
                </span>
              ) : isSessionPaused ? (
                <span className="badge badge-warn">
                  Paused Intent · Turn #{effectiveSession?.turn_count ?? turns.length}
                </span>
              ) : (
                <span className="badge badge-accent">
                  Final Intent · {turns.length} Turns
                </span>
              )}

              {(isReplayingTurn ? activeActionInfo.chips : latestActionInfo.chips).map((chip, idx) => (
                <span key={idx} className="playtest-action-chip">
                  {chip}
                </span>
              ))}

              {effectiveSession?.current_mood ? (
                <span className="badge">Mood: {effectiveSession.current_mood}</span>
              ) : null}
            </div>

            <p className="playtest-focus-plan">{focusHeadline}</p>

            {isReplayingTurn && activeTurn ? (
              <>
                {activeCleanObs && activeCleanObs !== focusHeadline ? (
                  <p className="playtest-focus-sub">
                    <strong>Observed on Turn #{activeTurn.turn_index}:</strong> {activeCleanObs}
                  </p>
                ) : null}
                {isSessionLive && liveCurrentPlan && liveCurrentPlan !== focusHeadline ? (
                  <p className="playtest-focus-sub" style={{ color: 'var(--text-muted)' }}>
                    <strong>Live now (Turn #{effectiveSession?.turn_count ?? turns.length}):</strong> {liveCurrentPlan}
                  </p>
                ) : null}
              </>
            ) : latestCleanObs && latestCleanObs !== focusHeadline ? (
              <p className="playtest-focus-sub">
                <strong>Last observed:</strong> {latestCleanObs}
              </p>
            ) : null}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
            {hasLiveEdge && !isViewingLiveFeed ? (
              <button type="button" className="btn btn-success" onClick={jumpToLive}>
                <Radio size={13} aria-hidden="true" /> Jump to Live
              </button>
            ) : turns.length > 1 && isViewingLiveFeed ? (
              <button type="button" className="btn btn-primary" onClick={toggleDvrReplay}>
                <Play size={13} aria-hidden="true" /> Replay 1→Live ({turns.length})
              </button>
            ) : null}

            {canDispatchFullReport && effectiveSession?.initiated_by !== 'council' ? (
              reportSentAtUtc ? (
                <span className="badge badge-ok">✓ Report Sent to Council</span>
              ) : inspectorTab !== 'report' ? (
                <button
                  type="button"
                  className="btn btn-success"
                  onClick={() => setInspectorTab('report')}
                >
                  <Sparkles size={13} aria-hidden="true" /> Review &amp; Send Report
                </button>
              ) : null
            ) : null}
          </div>
        </section>
      ) : null}

      {/* Main Split Stage: Left Live/DVR Video & Scrubber + Right 4-Tab Inspector */}
      {activeSessionSummary ? (
        <div className="playtest-stage-grid">
          {/* Left: Unified Live + DVR Replay Stage */}
          <div
            style={{
              padding: '12px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              display: 'grid',
              gap: '10px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
              <div className="pill-list">
                {isSessionLive || isSessionPaused || Boolean(liveFrameName) ? (
                  <button
                    type="button"
                    className={`pill${stageMode === 'live' ? ' is-active' : ''}`}
                    onClick={jumpToLive}
                  >
                    <Radio size={12} aria-hidden="true" /> Live Screen
                  </button>
                ) : null}
                <button
                  type="button"
                  className={`pill${stageMode === 'frames' ? ' is-active' : ''}`}
                  onClick={() => {
                    if (stageMode !== 'frames' && turns.length > 0 && stageMode === 'live') {
                      setSelectedTurnIdx(Math.max(0, turns.length - 1));
                    }
                    setStageMode('frames');
                  }}
                >
                  {hasLiveEdge ? `Replay Scrubber (${turns.length})` : `Turn-by-Turn Replay (${turns.length})`}
                </button>
                {effectiveHasVideo ? (
                  <button
                    type="button"
                    className={`pill${stageMode === 'video' ? ' is-active' : ''}`}
                    onClick={() => {
                      setIsDvrPlaying(false);
                      setStageMode('video');
                    }}
                  >
                    <Film size={12} aria-hidden="true" /> Broadcast Replay MP4
                  </button>
                ) : null}
              </div>

              {hasLiveEdge && !isViewingLiveFeed ? (
                <button type="button" className="pill is-active" onClick={jumpToLive}>
                  ● Return to Live (Turn #{effectiveSession?.turn_count ?? turns.length})
                </button>
              ) : effectiveSession?.current_mood ? (
                <span className="badge badge-accent">Mood: {effectiveSession.current_mood}</span>
              ) : null}
            </div>

            {stageMode === 'video' && effectiveHasVideo ? (
              <video
                ref={videoRef}
                key={effectiveSessionId}
                controls
                width={1600}
                height={720}
                src={buildApiUrl(
                  `/api/playtest/sessions/${encodeURIComponent(effectiveSessionId)}/video?v=${encodeURIComponent(effectiveSession?.updated_at_utc || '')}`,
                )}
                style={{
                  inlineSize: '100%',
                  aspectRatio: '1600 / 720',
                  background: '#05080f',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                }}
              />
            ) : stageMode === 'frames' && activeTurn ? (
              <div style={{ display: 'grid', gap: '8px' }}>
                <div style={{ position: 'relative', background: '#05080f', borderRadius: 'var(--radius-sm)', overflow: 'clip' }}>
                  {activeTurn.screenshot_name ? (
                    <img
                      src={buildApiUrl(
                        `/api/playtest/sessions/${encodeURIComponent(effectiveSessionId)}/media/${encodeURIComponent(activeTurn.screenshot_name)}`,
                      )}
                      alt={`Turn ${activeTurn.turn_index}`}
                      width={1280}
                      height={720}
                      style={{ inlineSize: '100%', aspectRatio: '16 / 9', objectFit: 'contain' }}
                    />
                  ) : null}
                  {activeActionInfo.targetX !== undefined && activeActionInfo.targetY !== undefined ? (
                    <svg
                      viewBox="0 0 1280 720"
                      aria-hidden="true"
                      style={{ position: 'absolute', inset: 0, inlineSize: '100%', blockSize: '100%', pointerEvents: 'none' }}
                    >
                      {activeActionInfo.isDrag &&
                      activeActionInfo.dragFromX !== undefined &&
                      activeActionInfo.dragFromY !== undefined ? (
                        <line
                          x1={activeActionInfo.dragFromX}
                          y1={activeActionInfo.dragFromY}
                          x2={activeActionInfo.targetX}
                          y2={activeActionInfo.targetY}
                          stroke="#fbbf24"
                          strokeWidth={3}
                          strokeDasharray="7 5"
                        />
                      ) : activeTurn.cursor_start_x !== undefined &&
                        activeTurn.cursor_start_y !== undefined &&
                        Math.hypot(
                          activeActionInfo.targetX - activeTurn.cursor_start_x,
                          activeActionInfo.targetY - activeTurn.cursor_start_y,
                        ) > 16 ? (
                        <line
                          x1={activeTurn.cursor_start_x}
                          y1={activeTurn.cursor_start_y}
                          x2={activeActionInfo.targetX}
                          y2={activeActionInfo.targetY}
                          stroke="rgba(56, 189, 248, 0.55)"
                          strokeWidth={2}
                          strokeDasharray="4 4"
                        />
                      ) : null}
                      <circle
                        cx={activeActionInfo.targetX}
                        cy={activeActionInfo.targetY}
                        r={activeActionInfo.mouseButtonAction !== 'none' ? 20 : 13}
                        fill={
                          activeActionInfo.mouseButtonAction !== 'none'
                            ? 'rgba(52, 211, 153, 0.24)'
                            : 'rgba(56, 189, 248, 0.16)'
                        }
                        stroke={activeActionInfo.mouseButtonAction !== 'none' ? '#34d399' : '#38bdf8'}
                        strokeWidth={3}
                      />
                      <circle cx={activeActionInfo.targetX} cy={activeActionInfo.targetY} r={4} fill="#f8fafc" />
                    </svg>
                  ) : null}

                  <div
                    style={{
                      position: 'absolute',
                      insetBlockStart: '8px',
                      insetInlineStart: '8px',
                      padding: '3px 9px',
                      borderRadius: '999px',
                      background: 'rgba(5, 8, 15, 0.82)',
                      border: '1px solid rgba(251, 191, 36, 0.45)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.73rem',
                      color: 'var(--text-primary)',
                      pointerEvents: 'none',
                    }}
                  >
                    REPLAY · Turn #{activeTurn.turn_index} / {turns.length} · {formatClock(activeTurn.video_timestamp_sec)} ·{' '}
                    {activeActionInfo.summary}
                  </div>
                </div>
              </div>
            ) : liveFrameName ? (
              <div style={{ position: 'relative', background: '#05080f', borderRadius: 'var(--radius-sm)', overflow: 'clip' }}>
                <img
                  src={buildApiUrl(
                    `/api/playtest/sessions/${encodeURIComponent(effectiveSessionId)}/media/${encodeURIComponent(liveFrameName)}?v=${encodeURIComponent(liveFrameCacheKey)}`,
                  )}
                  alt="Latest player frame"
                  width={1280}
                  height={720}
                  style={{
                    inlineSize: '100%',
                    aspectRatio: '16 / 9',
                    objectFit: 'contain',
                    background: '#05080f',
                    borderRadius: 'var(--radius-sm)',
                  }}
                />
                {isSessionLive ? (
                  <div
                    style={{
                      position: 'absolute',
                      insetBlockStart: '8px',
                      insetInlineEnd: '8px',
                      padding: '3px 8px',
                      borderRadius: '999px',
                      background: 'rgba(5, 8, 15, 0.78)',
                      border: '1px solid rgba(56, 189, 248, 0.35)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.73rem',
                      color: 'var(--text-secondary)',
                      pointerEvents: 'none',
                    }}
                  >
                    LIVE · Turn {effectiveSession?.turn_count ?? turns.length} · Tick{' '}
                    {liveFrame?.tick ?? effectiveSession?.sim_ticks_total ?? 0}
                  </div>
                ) : null}
              </div>
            ) : (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                No replay frames captured for this session yet.
              </div>
            )}

            {/* DVR Timeline Scrubber & Auto-Replay Controls (works both during Live runs and Completed runs!) */}
            {turns.length > 0 ? (
              <div className="playtest-dvr-bar" role="group" aria-label="DVR Replay Scrubber">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <label htmlFor={dvrSliderId} className="visually-hidden">
                    Scrub playtest timeline
                  </label>
                  <input
                    id={dvrSliderId}
                    type="range"
                    className="playtest-dvr-slider"
                    min={0}
                    max={sliderMax}
                    step={1}
                    value={sliderValue}
                    aria-valuetext={sliderValueText}
                    data-live-edge={hasLiveEdge && isViewingLiveFeed ? 'true' : 'false'}
                    style={
                      {
                        '--attr-min': 0,
                        '--attr-max': Math.max(1, sliderMax),
                        '--control-value': sliderValue,
                      } as CSSProperties
                    }
                    onChange={(e) => handleDvrSliderChange(Number(e.target.value))}
                  />
                </div>

                <div className="playtest-dvr-controls">
                  <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '5px' }}>
                    <button
                      type="button"
                      className={`btn ${isDvrPlaying ? 'btn-warn' : 'btn-primary'}`}
                      onClick={toggleDvrReplay}
                    >
                      {isDvrPlaying ? (
                        <>
                          <Pause size={13} aria-hidden="true" /> Pause Replay
                        </>
                      ) : (
                        <>
                          <Play size={13} aria-hidden="true" />{' '}
                          {isViewingLiveFeed
                            ? 'Replay 1→Live'
                            : hasLiveEdge && clampedTurnIdx < turns.length - 1
                              ? 'Play → Live'
                              : 'Play Turns'}
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      className="btn"
                      aria-label="Jump to Turn 1"
                      disabled={turns.length === 0 || (!isViewingLiveFeed && clampedTurnIdx <= 0)}
                      onClick={() => handleDvrSliderChange(0)}
                    >
                      <SkipBack size={13} aria-hidden="true" />
                    </button>

                    <button
                      type="button"
                      className="btn"
                      aria-label="Previous Turn"
                      disabled={turns.length === 0 || (!isViewingLiveFeed && clampedTurnIdx <= 0)}
                      onClick={() =>
                        handleDvrSliderChange(isViewingLiveFeed ? Math.max(0, turns.length - 1) : clampedTurnIdx - 1)
                      }
                    >
                      <ChevronLeft size={14} aria-hidden="true" /> Prev
                    </button>

                    <button
                      type="button"
                      className="btn"
                      aria-label="Next Turn"
                      disabled={isViewingLiveFeed || (!hasLiveEdge && clampedTurnIdx >= turns.length - 1)}
                      onClick={() => handleDvrSliderChange(clampedTurnIdx + 1)}
                    >
                      Next <ChevronRight size={14} aria-hidden="true" />
                    </button>

                    <div className="pill-list" role="group" aria-label="Replay speed">
                      {([0.5, 1, 2] as const).map((spd) => (
                        <button
                          key={spd}
                          type="button"
                          className={`pill${dvrSpeed === spd ? ' is-active' : ''}`}
                          onClick={() => setDvrSpeed(spd)}
                        >
                          {spd}×
                        </button>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {hasLiveEdge && isViewingLiveFeed
                        ? `● LIVE (Turn #${effectiveSession?.turn_count ?? turns.length})`
                        : activeTurn
                          ? `Turn #${activeTurn.turn_index} / ${turns.length} (${formatClock(activeTurn.video_timestamp_sec)})`
                          : ''}
                    </span>

                    {hasLiveEdge && !isViewingLiveFeed ? (
                      <button type="button" className="btn btn-success" onClick={jumpToLive}>
                        <Radio size={12} aria-hidden="true" /> Live
                      </button>
                    ) : null}
                  </div>
                </div>
              </div>
            ) : null}

            {/* Mood & Bug Event Seek Markers */}
            {turns.some((t) => Boolean(t.mood_note) || Boolean(t.bug_reports && t.bug_reports.length > 0)) ? (
              <div style={{ display: 'grid', gap: '6px' }}>
                <div style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                  Key Moments &amp; Bugs (Click to Scrub)
                </div>
                <div className="pill-list">
                  {turns
                    .map((t, idx) => ({ t, idx }))
                    .filter(({ t }) => Boolean(t.mood_note || (t.bug_reports && t.bug_reports.length > 0)))
                    .slice(0, 24)
                    .map(({ t, idx }) => {
                      const hasBug = Boolean(t.bug_reports && t.bug_reports.length > 0);
                      return (
                        <button
                          key={t.turn_id || idx}
                          type="button"
                          className={`pill${!isViewingLiveFeed && clampedTurnIdx === idx ? ' is-active' : ''}`}
                          onClick={() => seekToTurn(t, idx)}
                        >
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.73rem' }}>
                            #{t.turn_index} [{formatClock(t.video_timestamp_sec)}]
                          </span>
                          <span>{hasBug ? 'Bug' : t.mood_state || 'Shift'}</span>
                        </button>
                      );
                    })}
                </div>
              </div>
            ) : null}
          </div>

          {/* Right: 4-Tab Activity Log, Experience Report, Structured Notebook & Wishes */}
          <div
            style={{
              padding: '12px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              display: 'grid',
              gap: '10px',
            }}
          >
            <div className="pill-list" role="tablist">
              <button
                type="button"
                className={`pill${inspectorTab === 'turns' ? ' is-active' : ''}`}
                onClick={() => setInspectorTab('turns')}
              >
                Activity Log ({turns.length})
              </button>
              <button
                type="button"
                className={`pill${inspectorTab === 'report' ? ' is-active' : ''}`}
                onClick={() => setInspectorTab('report')}
              >
                Experience Report
              </button>
              <button
                type="button"
                className={`pill${inspectorTab === 'rulebook' ? ' is-active' : ''}`}
                onClick={() => setInspectorTab('rulebook')}
              >
                Rulebook &amp; Notes
              </button>
              <button
                type="button"
                className={`pill${inspectorTab === 'wishes' ? ' is-active' : ''}`}
                onClick={() => setInspectorTab('wishes')}
              >
                Wishes → Council ({wishes.length})
              </button>
            </div>

            {inspectorTab === 'turns' ? (
              <div className="playtest-tab-scroll" style={{ gap: '8px' }}>
                {turns.length > 1 ? (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      flexWrap: 'wrap',
                      gap: '6px',
                    }}
                  >
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Click any step to scrub replay; expand telemetry for full turn output.
                    </span>
                    <div className="pill-list" role="group" aria-label="Activity log sort order">
                      <button
                        type="button"
                        className={`pill${logSortOrder === 'newest' ? ' is-active' : ''}`}
                        onClick={() => setLogSortOrder('newest')}
                      >
                        Latest First
                      </button>
                      <button
                        type="button"
                        className={`pill${logSortOrder === 'oldest' ? ' is-active' : ''}`}
                        onClick={() => setLogSortOrder('oldest')}
                      >
                        Oldest First
                      </button>
                    </div>
                  </div>
                ) : null}

                {orderedTurns.length === 0 ? (
                  <div style={{ color: 'var(--text-muted)', padding: '14px', fontSize: '0.86rem' }}>
                    {isSessionLive
                      ? 'Waiting for Turn #1 observation and action...'
                      : 'No turns recorded for this session.'}
                  </div>
                ) : (
                  orderedTurns.map(({ turn: t, idx }, renderIdx) => {
                    const actInfo = formatPlaytestMotorAction(t);
                    const cleanObs = extractCleanTurnObservation(t.append_log_line) || t.mood_note || '';
                    const turnRules = parseRulebookEntries(t.rulebook_updates);
                    const turnQuestions = parseQuestionEntries(t.question_updates);
                    const turnNotes = parseFieldNoteEntries(t.field_note_updates);
                    const turnBugs = parseBugEntries(t.bug_reports);
                    const isSelected = !isViewingLiveFeed && clampedTurnIdx === idx;

                    return (
                      <article
                        key={t.turn_id || idx}
                        className={`playtest-log-item${renderIdx >= 12 ? ' deferred-compact-row' : ''}`}
                        data-selected={isSelected ? 'true' : 'false'}
                      >
                        <button
                          type="button"
                          className="playtest-log-seek-btn"
                          onClick={() => seekToTurn(t, idx)}
                        >
                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              flexWrap: 'wrap',
                              gap: '6px',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '5px' }}>
                              <strong style={{ color: 'var(--accent)', fontSize: '0.85rem' }}>
                                #{t.turn_index}
                              </strong>
                              {actInfo.chips.map((c, cIdx) => (
                                <span key={cIdx} className="playtest-action-chip">
                                  {c}
                                </span>
                              ))}
                              {turnBugs.length > 0 ? (
                                <span className="badge badge-danger">
                                  {turnBugs.length} {turnBugs.length === 1 ? 'Bug' : 'Bugs'}
                                </span>
                              ) : null}
                              {turnRules.length > 0 ? (
                                <span className="badge badge-ok">+{turnRules.length} Rule</span>
                              ) : null}
                              {turnQuestions.answered.length > 0 ? (
                                <span className="badge badge-accent">
                                  {turnQuestions.answered.length} Answered
                                </span>
                              ) : null}
                              {t.mood_note && t.mood_state ? (
                                <span className="badge badge-warn">{t.mood_state}</span>
                              ) : null}
                            </div>

                            <span
                              style={{
                                fontFamily: 'var(--font-mono)',
                                fontSize: '0.76rem',
                                color: 'var(--text-muted)',
                              }}
                            >
                              {formatClock(t.video_timestamp_sec)}
                            </span>
                          </div>

                          {cleanObs ? (
                            <div style={{ fontSize: '0.86rem', color: 'var(--text-primary)', lineHeight: 1.42 }}>
                              {cleanObs}
                            </div>
                          ) : null}

                          {t.current_plan && t.current_plan !== cleanObs ? (
                            <div style={{ fontSize: '0.81rem', color: 'var(--text-secondary)', lineHeight: 1.38 }}>
                              <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Trying next:</span>{' '}
                              {t.current_plan}
                            </div>
                          ) : null}
                        </button>

                        {/* Full Turn Output — tucked cleanly inside native <details> for observability on demand */}
                        <details className="playtest-turn-inspector">
                          <summary>Full Turn Output &amp; Thought</summary>
                          <div style={{ display: 'grid', gap: '8px', fontSize: '0.82rem' }}>
                            {t.deliberation ? (
                              <div className="deliberation-box">
                                <div style={{ fontWeight: 700, fontSize: '0.76rem', color: 'var(--purple)' }}>
                                  Player Deliberation
                                </div>
                                <div style={{ marginBlockStart: '3px', whiteSpace: 'pre-wrap' }}>
                                  {t.deliberation}
                                </div>
                              </div>
                            ) : null}

                            {actInfo.actionDeliberation ? (
                              <div style={{ color: 'var(--text-secondary)' }}>
                                <strong>Motor Rationale:</strong> {actInfo.actionDeliberation}
                              </div>
                            ) : null}

                            <div
                              style={{
                                display: 'flex',
                                flexWrap: 'wrap',
                                gap: '6px 12px',
                                fontFamily: 'var(--font-mono)',
                                fontSize: '0.75rem',
                                color: 'var(--text-muted)',
                              }}
                            >
                              {t.sim_tick_start !== undefined && t.sim_tick_end !== undefined ? (
                                <span>
                                  Ticks: {t.sim_tick_start}→{t.sim_tick_end}
                                </span>
                              ) : null}
                              {t.cursor_start_x !== undefined && t.cursor_end_x !== undefined ? (
                                <span>
                                  Cursor: ({Math.round(t.cursor_start_x)}, {Math.round(t.cursor_start_y ?? 0)}) → (
                                  {Math.round(t.cursor_end_x)}, {Math.round(t.cursor_end_y ?? 0)})
                                </span>
                              ) : null}
                              {t.prompt_tokens !== undefined ? (
                                <span>
                                  Tokens: {t.prompt_tokens.toLocaleString()} in
                                  {t.cached_tokens ? ` (${t.cached_tokens.toLocaleString()} cached)` : ''} ·{' '}
                                  {(t.output_tokens ?? 0).toLocaleString()} out
                                </span>
                              ) : null}
                            </div>

                            {turnRules.length > 0 ? (
                              <div>
                                <strong style={{ color: 'var(--ok)' }}>Rulebook Updates:</strong>
                                <ul style={{ margin: '3px 0 0', paddingInlineStart: '18px', color: 'var(--text-secondary)' }}>
                                  {turnRules.map((r, rIdx) => (
                                    <li key={rIdx}>
                                      {r.topic ? <strong>{r.topic}: </strong> : null}
                                      {r.behavior}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            ) : null}

                            {turnQuestions.answered.length > 0 || turnQuestions.open.length > 0 ? (
                              <div>
                                <strong style={{ color: 'var(--accent)' }}>Question Updates:</strong>
                                <ul style={{ margin: '3px 0 0', paddingInlineStart: '18px', color: 'var(--text-secondary)' }}>
                                  {turnQuestions.answered.map((q) => (
                                    <li key={`ans-${q.key}`}>
                                      <span className="badge badge-ok" style={{ marginInlineEnd: '5px' }}>
                                        Answered
                                      </span>
                                      {q.question}
                                      {q.answer ? (
                                        <div style={{ color: 'var(--text-primary)', marginBlockStart: '2px' }}>
                                          → {q.answer}
                                        </div>
                                      ) : null}
                                    </li>
                                  ))}
                                  {turnQuestions.open.map((q) => (
                                    <li key={`open-${q.key}`}>
                                      <span className="badge badge-warn" style={{ marginInlineEnd: '5px' }}>
                                        Open
                                      </span>
                                      {q.question}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            ) : null}

                            {turnNotes.length > 0 ? (
                              <div>
                                <strong style={{ color: 'var(--text-primary)' }}>Field Note Updates:</strong>
                                <ul style={{ margin: '3px 0 0', paddingInlineStart: '18px', color: 'var(--text-secondary)' }}>
                                  {turnNotes.map((n, nIdx) => (
                                    <li key={nIdx}>
                                      {n.topic ? <strong>{n.topic}: </strong> : null}
                                      {n.observation}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            ) : null}

                            {turnBugs.length > 0 ? (
                              <div>
                                <strong style={{ color: 'var(--danger)' }}>Bugs Logged:</strong>
                                <ul style={{ margin: '3px 0 0', paddingInlineStart: '18px', color: 'var(--text-secondary)' }}>
                                  {turnBugs.map((b) => (
                                    <li key={b.key}>
                                      <strong>[{b.severity.toUpperCase()}] {b.summary}</strong>
                                      {b.details ? ` — ${b.details}` : ''}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            ) : null}

                            {t.audio_clip_name ? (
                              <audio
                                controls
                                src={buildApiUrl(
                                  `/api/playtest/sessions/${encodeURIComponent(effectiveSessionId)}/media/${encodeURIComponent(t.audio_clip_name)}`,
                                )}
                                style={{ inlineSize: '100%' }}
                              />
                            ) : null}

                            {Boolean((t.append_log_line && t.append_log_line.trim() !== cleanObs) || t.action) ? (
                              <details className="playtest-turn-inspector">
                                <summary>Raw Prompt Tape &amp; Motor JSON</summary>
                                <pre
                                  tabIndex={0}
                                  style={{
                                    margin: '4px 0 0',
                                    padding: '8px',
                                    borderRadius: 'var(--radius-sm)',
                                    background: 'var(--bg-surface)',
                                    border: '1px solid var(--border-subtle)',
                                    fontSize: '0.74rem',
                                    whiteSpace: 'pre-wrap',
                                    overflowWrap: 'anywhere',
                                  }}
                                >
                                  <code>
                                    {[
                                      t.append_log_line && t.append_log_line.trim() !== cleanObs
                                        ? `Tape: ${t.append_log_line}`
                                        : '',
                                      t.action ? `Action JSON:\n${JSON.stringify(t.action, null, 2)}` : '',
                                    ]
                                      .filter(Boolean)
                                      .join('\n\n')}
                                  </code>
                                </pre>
                              </details>
                            ) : null}
                          </div>
                        </details>
                      </article>
                    );
                  })
                )}
              </div>
            ) : null}

            {inspectorTab === 'report' ? (
              <div className="playtest-tab-scroll">
                {canDispatchFullReport ? (
                  <form
                    className="playtest-report-dispatch-form"
                    method="post"
                    action={buildApiUrl(
                      `/api/playtest/sessions/${encodeURIComponent(effectiveSessionId)}/send-report-to-council`,
                    )}
                    onSubmit={(e) => {
                      e.preventDefault();
                      if (!sendReportMut.isPending) {
                        sendReportMut.mutate();
                      }
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        flexWrap: 'wrap',
                        gap: '6px',
                      }}
                    >
                      <div style={{ fontWeight: 700, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                        Dispatch Full Playtest Report to Council
                      </div>
                      {effectiveSession?.initiated_by === 'council' ? (
                        <span className="badge badge-purple">Auto-Delivered by Council Gate</span>
                      ) : reportSentAtUtc ? (
                        <span className="badge badge-ok">
                          Sent to Council · {new Date(reportSentAtUtc).toLocaleTimeString()}
                        </span>
                      ) : (
                        <span className="badge badge-accent">Manual Run · Ready to Send</span>
                      )}
                    </div>

                    <div style={{ fontSize: '0.79rem', color: 'var(--text-secondary)' }}>
                      Includes full Experience Report
                      {report?.overall_fun_rating !== undefined ? ` (${report.overall_fun_rating}/10)` : ''},{' '}
                      {parsedBugs.length} spotted bug{parsedBugs.length === 1 ? '' : 's'},{' '}
                      {parsedQuestions.open.length + parsedQuestions.answered.length} question
                      {parsedQuestions.open.length + parsedQuestions.answered.length === 1 ? '' : 's'},{' '}
                      {parsedFieldNotes.length} field note{parsedFieldNotes.length === 1 ? '' : 's'},{' '}
                      {parsedRulebook.length} rule{parsedRulebook.length === 1 ? '' : 's'}, and representative player
                      screenshots.
                    </div>

                    <div className="playtest-report-field-group">
                      <label htmlFor={reportNoteInputId} className="playtest-report-field-label">
                        Producer Direction for Council (optional)
                      </label>
                      <input
                        id={reportNoteInputId}
                        name="producer_note"
                        type="text"
                        className="input-control"
                        placeholder="e.g., Prioritize fixing onboarding clarity and UI overlap before expanding scope"
                        value={reportProducerNote}
                        onChange={(e) => setReportProducerNote(e.target.value)}
                      />
                    </div>

                    {pendingSessionWishes.length > 0 ? (
                      <label htmlFor={includeWishesInputId} className="playtest-report-checkbox-row">
                        <input
                          id={includeWishesInputId}
                          name="include_pending_wishes"
                          type="checkbox"
                          checked={includePendingWishes}
                          onChange={(e) => setIncludePendingWishes(e.target.checked)}
                        />
                        <span>
                          Also approve &amp; include {pendingSessionWishes.length} pending player wish
                          {pendingSessionWishes.length === 1 ? '' : 'es'}
                        </span>
                      </label>
                    ) : null}

                    {lastReportDispatchStatus?.sessionId === effectiveSessionId ? (
                      <div
                        role="status"
                        aria-live="polite"
                        style={{
                          padding: '6px 8px',
                          borderRadius: 'var(--radius-sm)',
                          background: 'color-mix(in srgb, var(--ok) 14%, var(--bg-surface))',
                          border: '1px solid color-mix(in srgb, var(--ok) 40%, var(--border-subtle))',
                          fontSize: '0.8rem',
                          color: 'var(--text-primary)',
                        }}
                      >
                        {lastReportDispatchStatus.status === 'council_resumed'
                          ? '✓ Full report sent — Specialist Council resumed in Writers’ Room to create & execute tasks.'
                          : '✓ Full report queued for the active Specialist Council session in Writers’ Room.'}
                      </div>
                    ) : null}

                    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                      <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={sendReportMut.isPending}
                      >
                        <Sparkles size={13} aria-hidden="true" />
                        {sendReportMut.isPending
                          ? 'Sending Report...'
                          : reportSentAtUtc
                            ? 'Re-send Full Report to Council'
                            : 'Send Full Report to Council'}
                      </button>
                    </div>
                  </form>
                ) : null}

                {report ? (
                  <>
                    {report.overall_fun_rating !== undefined ? (
                      <div style={{ fontWeight: 800, fontSize: '1.09rem', color: 'var(--ok)' }}>
                        Overall Fun Rating: {report.overall_fun_rating} / 10
                      </div>
                    ) : null}
                    <DeliberationDisclosure deliberation={report.deliberation} defaultOpen />
                    {report.overall_experience_summary ? (
                      <p style={{ margin: 0, fontSize: '0.91rem', color: 'var(--text-primary)' }}>
                        {report.overall_experience_summary}
                      </p>
                    ) : null}
                    {report.visual_and_ui_defects_observed && report.visual_and_ui_defects_observed.length > 0 ? (
                      <div
                        style={{
                          padding: '8px 10px',
                          borderRadius: 'var(--radius-sm)',
                          background: 'var(--bg-canvas)',
                          border: '1px solid var(--danger, var(--warn))',
                        }}
                      >
                        <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--danger, var(--warn))' }}>
                          Visual &amp; UI Defects Observed ({report.visual_and_ui_defects_observed.length})
                        </div>
                        <ul
                          style={{
                            margin: '4px 0 0',
                            paddingInlineStart: '18px',
                            fontSize: '0.86rem',
                            color: 'var(--text-secondary)',
                            display: 'grid',
                            gap: '3px',
                          }}
                        >
                          {report.visual_and_ui_defects_observed.map((item, idx) => (
                            <li key={idx}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {(
                      [
                        ['Graphics & Visuals', report.graphics_and_visuals],
                        ['Gameplay & Feel', report.gameplay_and_feel],
                        ['Sound & Audio', report.sound_and_audio],
                        ['Intuitiveness & Onboarding', report.intuitiveness_and_onboarding],
                        ['Goals, Pacing & Accomplishment', report.goals_pacing_and_accomplishment],
                        ['World & Atmosphere', report.world_and_atmosphere],
                      ] as const
                    )
                      .filter(([, val]) => Boolean(val))
                      .map(([label, val]) => (
                        <div
                          key={label}
                          style={{
                            padding: '8px 10px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-canvas)',
                            border: '1px solid var(--border-subtle)',
                          }}
                        >
                          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent)' }}>{label}</div>
                          <div style={{ fontSize: '0.87rem', color: 'var(--text-secondary)', marginBlockStart: '2px' }}>
                            {val}
                          </div>
                        </div>
                      ))}
                    {(
                      [
                        ['Key Highlights', report.key_highlights, 'var(--ok)'],
                        ['Key Pain Points', report.key_pain_points, 'var(--warn)'],
                        ['Bugs & Glitches Observed', report.bugs_and_glitches_observed, 'var(--danger, var(--warn))'],
                        [
                          'Unanswered Questions & Mysteries',
                          report.unanswered_questions_and_mysteries,
                          'var(--text-secondary)',
                        ],
                      ] as const
                    )
                      .filter(([, list]) => Array.isArray(list) && list.length > 0)
                      .map(([label, list, color]) => (
                        <div
                          key={label}
                          style={{
                            padding: '8px 10px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-canvas)',
                            border: '1px solid var(--border-subtle)',
                          }}
                        >
                          <div style={{ fontSize: '0.8rem', fontWeight: 700, color }}>
                            {label} ({list!.length})
                          </div>
                          <ul
                            style={{
                              margin: '4px 0 0',
                              paddingInlineStart: '18px',
                              fontSize: '0.86rem',
                              color: 'var(--text-secondary)',
                              display: 'grid',
                              gap: '3px',
                            }}
                          >
                            {list!.map((item, idx) => (
                              <li key={idx}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ))}
                  </>
                ) : (
                  <div style={{ color: 'var(--text-muted)', padding: '16px' }}>
                    Experience report synthesizes automatically when the session finishes.
                  </div>
                )}
              </div>
            ) : null}

            {inspectorTab === 'rulebook' ? (
              <div className="playtest-tab-scroll" style={{ fontSize: '0.86rem' }}>
                {/* 1. Bugs & Glitches Spotted */}
                {parsedBugs.length > 0 ? (
                  <div className="playtest-notebook-section" style={{ borderColor: 'color-mix(in srgb, var(--danger) 42%, var(--border-subtle))' }}>
                    <div style={{ fontWeight: 700, color: 'var(--danger)' }}>
                      Bugs &amp; Glitches Spotted ({parsedBugs.length})
                    </div>
                    <div style={{ display: 'grid', gap: '6px' }}>
                      {parsedBugs.map((bug) => {
                        const matchingTurnIdx =
                          bug.turnIndex !== undefined
                            ? turns.findIndex((t) => t.turn_index === bug.turnIndex)
                            : -1;
                        return (
                          <div
                            key={bug.key}
                            style={{
                              padding: '7px 9px',
                              borderRadius: 'var(--radius-sm)',
                              background: 'var(--bg-surface)',
                              border: '1px solid var(--border-subtle)',
                              display: 'grid',
                              gap: '3px',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '5px' }}>
                              <span
                                className={`badge ${
                                  bug.severity.toLowerCase() === 'critical' || bug.severity.toLowerCase() === 'major'
                                    ? 'badge-danger'
                                    : 'badge-warn'
                                }`}
                              >
                                {bug.severity}
                              </span>
                              <span className="badge">{bug.category.replace(/_/g, ' ')}</span>
                              {bug.turnIndex !== undefined ? (
                                matchingTurnIdx >= 0 ? (
                                  <button
                                    type="button"
                                    className="pill"
                                    onClick={() => seekToTurn(turns[matchingTurnIdx], matchingTurnIdx)}
                                  >
                                    Turn #{bug.turnIndex}
                                  </button>
                                ) : (
                                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    Turn #{bug.turnIndex}
                                  </span>
                                )
                              ) : null}
                            </div>
                            <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{bug.summary}</div>
                            {bug.details ? (
                              <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>{bug.details}</div>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : null}

                {/* 2. Confirmed Player Rulebook */}
                <div className="playtest-notebook-section">
                  <div style={{ fontWeight: 700, color: 'var(--accent)' }}>
                    Inferred Player Rulebook ({parsedRulebook.length})
                  </div>
                  {parsedRulebook.length === 0 ? (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                      No controls or rules confirmed yet.
                    </div>
                  ) : (
                    <ul
                      style={{
                        margin: 0,
                        paddingInlineStart: '18px',
                        color: 'var(--text-secondary)',
                        display: 'grid',
                        gap: '5px',
                      }}
                    >
                      {parsedRulebook.map((r, i) => (
                        <li key={i}>
                          {r.topic ? <strong style={{ color: 'var(--text-primary)' }}>{r.topic}: </strong> : null}
                          {r.behavior}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                {/* 3. Open Mysteries (Unanswered Questions) */}
                <div className="playtest-notebook-section">
                  <div style={{ fontWeight: 700, color: 'var(--warn)' }}>
                    Open Questions &amp; Mysteries ({parsedQuestions.open.length})
                  </div>
                  {parsedQuestions.open.length === 0 ? (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                      No unresolved questions right now.
                    </div>
                  ) : (
                    <ul
                      style={{
                        margin: 0,
                        paddingInlineStart: '18px',
                        color: 'var(--text-primary)',
                        display: 'grid',
                        gap: '5px',
                      }}
                    >
                      {parsedQuestions.open.map((q) => (
                        <li key={q.key}>
                          <span>{q.question}</span>
                          {q.askedTurn !== undefined ? (
                            <span
                              style={{
                                marginInlineStart: '6px',
                                fontFamily: 'var(--font-mono)',
                                fontSize: '0.74rem',
                                color: 'var(--text-muted)',
                              }}
                            >
                              (asked Turn #{q.askedTurn})
                            </span>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                {/* 4. Answered Questions (Collapsible with full Q → Answer pairs, zero snake_case clutter) */}
                {parsedQuestions.answered.length > 0 ? (
                  <details
                    className="playtest-notebook-disclosure"
                    open={parsedQuestions.open.length === 0 && parsedQuestions.answered.length <= 4}
                  >
                    <summary>
                      Answered Questions ({parsedQuestions.answered.length} solved)
                    </summary>
                    <div style={{ display: 'grid', gap: '7px' }}>
                      {parsedQuestions.answered.map((q) => (
                        <div
                          key={q.key}
                          style={{
                            padding: '6px 8px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-surface)',
                            border: '1px solid var(--border-subtle)',
                            display: 'grid',
                            gap: '3px',
                          }}
                        >
                          <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                            <strong>Q:</strong> {q.question}
                          </div>
                          {q.answer ? (
                            <div style={{ color: 'var(--ok)', fontSize: '0.84rem', fontWeight: 600 }}>
                              ✓ {q.answer}
                              {q.answeredTurn !== undefined ? (
                                <span
                                  style={{
                                    marginInlineStart: '6px',
                                    fontFamily: 'var(--font-mono)',
                                    fontSize: '0.73rem',
                                    fontWeight: 400,
                                    color: 'var(--text-muted)',
                                  }}
                                >
                                  (Turn #{q.answeredTurn})
                                </span>
                              ) : null}
                            </div>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  </details>
                ) : null}

                {/* 5. Screen Field Notes */}
                <div className="playtest-notebook-section">
                  <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                    Screen Field Notes ({parsedFieldNotes.length})
                  </div>
                  {parsedFieldNotes.length === 0 ? (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                      No screen field notes recorded yet.
                    </div>
                  ) : (
                    <ul
                      style={{
                        margin: 0,
                        paddingInlineStart: '18px',
                        color: 'var(--text-secondary)',
                        display: 'grid',
                        gap: '5px',
                      }}
                    >
                      {parsedFieldNotes.map((n, i) => (
                        <li key={i}>
                          {n.topic ? <strong style={{ color: 'var(--text-primary)' }}>{n.topic}: </strong> : null}
                          {n.observation}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            ) : null}

            {inspectorTab === 'wishes' ? (
              <div className="playtest-tab-scroll">
                {wishes.length === 0 ? (
                  <div style={{ color: 'var(--text-muted)', padding: '16px' }}>
                    No player wishes recorded for this session.
                  </div>
                ) : (
                  <>
                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                      <input
                        type="text"
                        className="input-control"
                        style={{ flex: '1 1 auto' }}
                        placeholder="Optional Producer note for Council..."
                        value={producerNote}
                        onChange={(e) => setProducerNote(e.target.value)}
                      />
                      <button
                        type="button"
                        className="btn btn-primary"
                        disabled={selectedWishIds.size === 0 || sendWishesMut.isPending}
                        onClick={() => sendWishesMut.mutate(Array.from(selectedWishIds))}
                      >
                        <Send size={12} aria-hidden="true" /> Send ({selectedWishIds.size})
                      </button>
                    </div>

                    {wishes.map((w) => {
                      const checked = selectedWishIds.has(w.wish_id);
                      const isPending = w.status === 'pending';
                      return (
                        <article
                          key={w.wish_id}
                          style={{
                            padding: '10px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-canvas)',
                            border: '1px solid var(--border-subtle)',
                            display: 'grid',
                            gap: '6px',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
                            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontWeight: 700, fontSize: '0.87rem' }}>
                              {isPending ? (
                                <input
                                  type="checkbox"
                                  checked={checked}
                                  onChange={(e) => {
                                    setSelectedWishIds((prev) => {
                                      const next = new Set(prev);
                                      if (e.target.checked) next.add(w.wish_id);
                                      else next.delete(w.wish_id);
                                      return next;
                                    });
                                  }}
                                />
                              ) : null}
                              <span>{w.title || w.category || 'Player Wish'}</span>
                            </label>
                            <span className={`badge ${w.status === 'sent_to_council' ? 'badge-ok' : ''}`}>
                              {w.status}
                            </span>
                          </div>

                          <DeliberationDisclosure deliberation={w.deliberation || w.rationale} />

                          <div style={{ fontSize: '0.87rem', color: 'var(--text-secondary)' }}>
                            {w.wish_text || w.description}
                          </div>

                          {isPending ? (
                            <div style={{ display: 'flex', gap: '6px', justifyContent: 'flex-end' }}>
                              <button
                                type="button"
                                className="btn btn-ghost"
                                onClick={() => dismissWishMut.mutate(w.wish_id)}
                              >
                                <Trash2 size={12} aria-hidden="true" /> Dismiss
                              </button>
                              <button
                                type="button"
                                className="btn btn-primary"
                                disabled={sendWishesMut.isPending}
                                onClick={() => sendWishesMut.mutate([w.wish_id])}
                              >
                                <Sparkles size={12} aria-hidden="true" /> Send to Council
                              </button>
                            </div>
                          ) : null}
                        </article>
                      );
                    })}
                  </>
                )}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
