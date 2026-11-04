import {
  type FormEvent,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ArrowDown,
  BookOpen,
  CheckCircle2,
  Gamepad2,
  Kanban,
  Loader2,
  PanelLeft,
  Pin,
  Plus,
  Rocket,
  Send,
  Sparkles,
  X,
} from 'lucide-react';
import { studioApi } from '../../api/client';
import type {
  CoProducerActionProposal,
  CoProducerLiveStep,
  CoProducerMessage,
  CoProducerThread,
  WorkspaceId,
} from '../../api/types';
import { ErrorBanner } from '../../components/ErrorBanner';
import type { LightboxMedia } from '../../components/MediaLightbox';
import { FormattedRichBlocks } from '../../components/RichTextBlocks';
import {
  buildUnifiedCoProducerSessions,
  formatConversationTimestamp,
  isSameStudioSession,
  type StudioSessionSelection,
  type UnifiedCoProducerSession,
} from '../../domain/coproducerSessions';
import {
  CoProducerActionProposalCard,
  CoProducerMediaCardView,
  CoProducerTurnObservability,
  normalizeTargetWorkspace,
} from './CoProducerCards';

export interface CoProducerViewProps {
  cartridgeId: string;
  hasAnyCartridge: boolean;
  gameTitle?: string;
  orchestratorState?: string;
  boardCounts: { approved: number; total: number };
  liveStepsByThread: Record<string, CoProducerLiveStep[]>;
  onSelectCartridge: (cartridgeId: string) => void;
  onSelectWorkspace: (ws: WorkspaceId) => void;
  onOpenLightbox: (media: LightboxMedia) => void;
  requestedDraftThreadId?: string | null;
  onConsumeRequestedDraftThread?: () => void;
  defaultPauseForGreenlight?: boolean;
  onRunAutonomousBuild?: (params: {
    creativePrompt: string;
    pauseForGreenlight: boolean;
  }) => Promise<void>;
  runningAutonomousBuild?: boolean;
  sessions?: UnifiedCoProducerSession[];
  selectedSession?: StudioSessionSelection | null;
  onSelectSession?: (next: StudioSessionSelection) => void;
  onStartNewGame?: () => void;
  startingNewGame?: boolean;
}

interface OptimisticUserTurn {
  clientId: string;
  threadId: string;
  text: string;
  sketchPaths: string[];
  createdAtUtc: string;
}

const DRAFT_STARTER_PROMPTS = [
  'Pitch 3 wildly distinct playable game concepts tuned to my Player DNA',
  'Help me shape a fresh game idea from scratch and sketch a 16:9 concept mockup',
  'Run a Divergent Pitch Jam with the Specialist Council to surface unexpected genre collisions',
];

const CARTRIDGE_STARTER_PROMPTS = [
  'Give me a concise showrunner briefing on where the game stands right now',
  'Inspect recent playtest sessions and critique the moment-to-moment game feel',
  'Explore a bold design pivot or next-slice expansion with the Specialist Council',
  'Sketch an art or audio variation for the current build so we can audition it together',
];

const SCROLL_BOTTOM_THRESHOLD_PX = 80;

export function CoProducerView({
  cartridgeId,
  hasAnyCartridge,
  gameTitle,
  orchestratorState,
  boardCounts,
  liveStepsByThread,
  onSelectCartridge,
  onSelectWorkspace,
  onOpenLightbox,
  requestedDraftThreadId = null,
  onConsumeRequestedDraftThread,
  defaultPauseForGreenlight = false,
  onRunAutonomousBuild,
  runningAutonomousBuild = false,
  sessions: externalSessions,
  selectedSession: externalSelectedSession,
  onSelectSession: externalOnSelectSession,
  onStartNewGame: externalOnStartNewGame,
  startingNewGame = false,
}: CoProducerViewProps) {
  const queryClient = useQueryClient();
  const composerInputId = useId();
  const draftHandoffSelectId = useId();
  const greenlightCheckboxId = useId();
  const headerGreenlightCheckboxId = useId();

  const [internalSession, setInternalSession] = useState<StudioSessionSelection | null>(() => {
    if (requestedDraftThreadId) {
      return { kind: 'draft', threadId: requestedDraftThreadId };
    }
    if (hasAnyCartridge && cartridgeId) {
      return { kind: 'cartridge', cartridgeId };
    }
    return null;
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [composerText, setComposerText] = useState('');
  const [selectedSketchPaths, setSelectedSketchPaths] = useState<string[]>([]);
  const [optimisticUserTurn, setOptimisticUserTurn] = useState<OptimisticUserTurn | null>(null);
  const [pauseForGreenlight, setPauseForGreenlight] = useState<boolean>(
    Boolean(defaultPauseForGreenlight),
  );
  const [handoffModeOverride, setHandoffModeOverride] = useState<
    'refined_concept' | 'loose_spark' | null
  >(null);
  const [localError, setLocalError] = useState<unknown>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  const [showJumpToCurrent, setShowJumpToCurrent] = useState(false);

  const feedContainerRef = useRef<HTMLDivElement | null>(null);
  const feedContentRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const isPinnedToBottomRef = useRef<boolean>(true);

  const effectiveSelectedSession =
    externalSelectedSession !== undefined ? externalSelectedSession : internalSession;

  const commitSessionSelection = useCallback(
    (next: StudioSessionSelection) => {
      isPinnedToBottomRef.current = true;
      setShowJumpToCurrent(false);
      setLocalError(null);
      if (externalOnSelectSession) {
        externalOnSelectSession(next);
      } else {
        setInternalSession(next);
        if (next.kind === 'cartridge') {
          onSelectCartridge(next.cartridgeId);
        }
      }
    },
    [externalOnSelectSession, onSelectCartridge],
  );

  useEffect(() => {
    setPauseForGreenlight(Boolean(defaultPauseForGreenlight));
  }, [defaultPauseForGreenlight]);

  useEffect(() => {
    if (!requestedDraftThreadId) return;
    setInternalSession({ kind: 'draft', threadId: requestedDraftThreadId });
    setComposerText('');
    setSelectedSketchPaths([]);
    setLocalError(null);
    isPinnedToBottomRef.current = true;
    setShowJumpToCurrent(false);
    onConsumeRequestedDraftThread?.();
    if (composerRef.current && typeof composerRef.current.focus === 'function') {
      composerRef.current.focus();
    }
  }, [requestedDraftThreadId, onConsumeRequestedDraftThread]);

  const effectiveCartridgeParam =
    hasAnyCartridge && cartridgeId && cartridgeId !== 'active' ? cartridgeId : undefined;

  const threadsQuery = useQuery({
    queryKey: ['coproducer-threads', effectiveCartridgeParam ?? 'none'],
    queryFn: () => studioApi.getCoProducerThreads(effectiveCartridgeParam),
  });

  const draftThreads: CoProducerThread[] = useMemo(
    () =>
      Array.isArray(threadsQuery.data?.draft_threads) ? threadsQuery.data.draft_threads : [],
    [threadsQuery.data?.draft_threads],
  );

  const cartridgeThreads: CoProducerThread[] = useMemo(
    () =>
      Array.isArray(threadsQuery.data?.cartridge_threads)
        ? threadsQuery.data.cartridge_threads
        : [],
    [threadsQuery.data?.cartridge_threads],
  );

  const activeCartridgeThread: CoProducerThread | null =
    threadsQuery.data?.active_cartridge_thread ?? null;

  const resolvedSessions: UnifiedCoProducerSession[] = useMemo(() => {
    if (externalSessions) {
      return externalSessions;
    }
    return buildUnifiedCoProducerSessions({
      games: [],
      threadsData: threadsQuery.data,
      fallbackCartridge: hasAnyCartridge
        ? {
            cartridgeId,
            title: gameTitle,
            approved: boardCounts.approved,
            total: boardCounts.total,
            status: orchestratorState,
          }
        : null,
    });
  }, [
    externalSessions,
    threadsQuery.data,
    hasAnyCartridge,
    cartridgeId,
    gameTitle,
    boardCounts.approved,
    boardCounts.total,
    orchestratorState,
  ]);

  // Ensure standalone CoProducerView binds an initial session when external selectedSession is omitted
  useEffect(() => {
    if (externalSelectedSession !== undefined) return;
    if (internalSession !== null) return;
    const firstGame = resolvedSessions.find((s) => s.kind === 'cartridge');
    if (firstGame) {
      setInternalSession(firstGame.selection);
      return;
    }
    const firstDraft = resolvedSessions.find((s) => s.kind === 'draft');
    if (firstDraft) {
      setInternalSession(firstDraft.selection);
    }
  }, [externalSelectedSession, internalSession, resolvedSessions]);

  const isDraftView = useMemo(() => {
    if (effectiveSelectedSession) {
      return effectiveSelectedSession.kind === 'draft';
    }
    return !hasAnyCartridge;
  }, [effectiveSelectedSession, hasAnyCartridge]);

  const activeSessionCartridgeId =
    effectiveSelectedSession?.kind === 'cartridge'
      ? effectiveSelectedSession.cartridgeId
      : cartridgeId;

  const activeThreadId = useMemo(() => {
    if (effectiveSelectedSession?.kind === 'draft') {
      if (effectiveSelectedSession.threadId === '__creating__') {
        return '';
      }
      return effectiveSelectedSession.threadId || draftThreads[0]?.thread_id || '';
    }
    if (!hasAnyCartridge) {
      return draftThreads[0]?.thread_id || '';
    }
    const matchedCartridgeThread =
      (activeCartridgeThread?.cartridge_id === activeSessionCartridgeId
        ? activeCartridgeThread
        : null) ||
      cartridgeThreads.find((ct) => ct.cartridge_id === activeSessionCartridgeId) ||
      null;
    if (matchedCartridgeThread?.thread_id) {
      return matchedCartridgeThread.thread_id;
    }
    if (activeSessionCartridgeId && activeSessionCartridgeId !== 'active') {
      return activeSessionCartridgeId;
    }
    if (activeCartridgeThread?.thread_id) {
      return activeCartridgeThread.thread_id;
    }
    if (hasAnyCartridge && activeSessionCartridgeId === 'active') {
      return 'active';
    }
    return '';
  }, [
    effectiveSelectedSession,
    hasAnyCartridge,
    draftThreads,
    activeCartridgeThread,
    cartridgeThreads,
    activeSessionCartridgeId,
  ]);

  const threadDetailQuery = useQuery({
    queryKey: ['coproducer-thread', activeThreadId],
    queryFn: () => studioApi.getCoProducerThread(activeThreadId),
    enabled: Boolean(activeThreadId),
  });

  const activeThread: CoProducerThread | null =
    threadDetailQuery.data?.thread ??
    (isDraftView
      ? draftThreads.find((d) => d.thread_id === activeThreadId) ?? null
      : (activeCartridgeThread?.cartridge_id === activeSessionCartridgeId
          ? activeCartridgeThread
          : cartridgeThreads.find((ct) => ct.cartridge_id === activeSessionCartridgeId)) ??
        null);

  const serverMessages: CoProducerMessage[] = useMemo(
    () =>
      Array.isArray(threadDetailQuery.data?.messages) ? threadDetailQuery.data.messages : [],
    [threadDetailQuery.data?.messages],
  );

  const canonicalThreadId = activeThread?.thread_id || activeThreadId;

  // Merge optimistic user turn synchronously into conversation history without duplication
  const displayedMessages: CoProducerMessage[] = useMemo(() => {
    if (!optimisticUserTurn) {
      return serverMessages;
    }
    const matchesCurrentThread =
      optimisticUserTurn.threadId === '__pending_draft__' ||
      optimisticUserTurn.threadId === canonicalThreadId ||
      optimisticUserTurn.threadId === activeThreadId;
    if (!matchesCurrentThread) {
      return serverMessages;
    }
    const alreadyPersisted = serverMessages.some(
      (m) =>
        m.sender_role === 'user' &&
        m.content_markdown.trim() === optimisticUserTurn.text.trim(),
    );
    if (alreadyPersisted) {
      return serverMessages;
    }
    const optimisticMsg: CoProducerMessage = {
      deliberation: 'Executive Producer prompt dispatched to Co-Producer.',
      message_id: optimisticUserTurn.clientId,
      thread_id: canonicalThreadId || optimisticUserTurn.threadId,
      turn_index: serverMessages.length + 1,
      sender_role: 'user',
      content_markdown: optimisticUserTurn.text,
      media_cards: [],
      consultation_cards: [],
      action_proposals: [],
      tool_trace: [],
      prompt_trace: [],
      created_at_utc: optimisticUserTurn.createdAtUtc,
    };
    return [...serverMessages, optimisticMsg];
  }, [serverMessages, optimisticUserTurn, canonicalThreadId, activeThreadId]);

  const liveSteps: CoProducerLiveStep[] = canonicalThreadId
    ? liveStepsByThread[canonicalThreadId] ?? []
    : [];

  const scrollFeedToBottom = useCallback(() => {
    const el = feedContainerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    isPinnedToBottomRef.current = true;
    setShowJumpToCurrent(false);
  }, []);

  const handleFeedScroll = useCallback(() => {
    const el = feedContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const pinned = distanceFromBottom <= SCROLL_BOTTOM_THRESHOLD_PX;
    isPinnedToBottomRef.current = pinned;
    setShowJumpToCurrent(!pinned);
  }, []);

  // Always jump straight to the current place in the conversation on open & session switch
  useLayoutEffect(() => {
    isPinnedToBottomRef.current = true;
    scrollFeedToBottom();
  }, [canonicalThreadId, scrollFeedToBottom]);

  const lastMessageId =
    displayedMessages.length > 0
      ? displayedMessages[displayedMessages.length - 1].message_id
      : '';

  const internalCreateDraftMut = useMutation({
    mutationFn: () => studioApi.createCoProducerDraft('New Game Concept'),
    onSuccess: (res) => {
      setLocalError(null);
      if (res?.thread?.thread_id) {
        queryClient.setQueryData(['coproducer-thread', res.thread.thread_id], res);
        commitSessionSelection({ kind: 'draft', threadId: res.thread.thread_id });
      }
      void queryClient.invalidateQueries({ queryKey: ['coproducer-threads'] });
    },
    onError: (err) => setLocalError(err),
  });

  const handleCreateNewGameSession = () => {
    setComposerText('');
    setSelectedSketchPaths([]);
    setLocalError(null);
    isPinnedToBottomRef.current = true;
    setShowJumpToCurrent(false);
    if (externalOnStartNewGame) {
      externalOnStartNewGame();
    } else {
      internalCreateDraftMut.mutate();
    }
  };

  const sendMut = useMutation({
    mutationFn: async (params: { promptText: string; sketchPaths: string[] }) => {
      let targetThreadId = activeThread?.thread_id || activeThreadId;
      if (!targetThreadId) {
        const draftRes = await studioApi.createCoProducerDraft('New Game Concept');
        targetThreadId = draftRes.thread.thread_id;
        queryClient.setQueryData(['coproducer-thread', targetThreadId], draftRes);
        commitSessionSelection({ kind: 'draft', threadId: targetThreadId });
        setOptimisticUserTurn((prev) =>
          prev ? { ...prev, threadId: targetThreadId } : null,
        );
      }
      return studioApi.sendCoProducerMessage(targetThreadId, {
        message: params.promptText,
        selectedSketchPaths: params.sketchPaths,
      });
    },
    onSuccess: async (res) => {
      setLocalError(null);
      const tid = res?.thread?.thread_id || canonicalThreadId;
      if (tid && res?.user_message && res?.assistant_message) {
        queryClient.setQueryData(
          ['coproducer-thread', tid],
          (
            prev:
              | {
                  deliberation?: string;
                  thread?: CoProducerThread;
                  messages?: CoProducerMessage[];
                }
              | undefined,
          ) => {
            const existing = Array.isArray(prev?.messages) ? prev.messages : [];
            const seen = new Set(existing.map((m) => m.message_id));
            const merged = [...existing];
            if (!seen.has(res.user_message.message_id)) {
              merged.push(res.user_message);
              seen.add(res.user_message.message_id);
            }
            if (!seen.has(res.assistant_message.message_id)) {
              merged.push(res.assistant_message);
            }
            return {
              deliberation: res.deliberation || prev?.deliberation || 'ok',
              thread: res.thread || prev?.thread,
              messages: merged,
            };
          },
        );
        setOptimisticUserTurn(null);
      } else if (tid) {
        await queryClient.invalidateQueries({ queryKey: ['coproducer-thread', tid] });
        setOptimisticUserTurn(null);
      } else {
        setOptimisticUserTurn(null);
      }
      void queryClient.invalidateQueries({ queryKey: ['coproducer-threads'] });
    },
    onError: (err, vars) => {
      setOptimisticUserTurn(null);
      setLocalError(err);
      setComposerText((prev) => prev || vars.promptText);
      setSelectedSketchPaths((prev) => (prev.length > 0 ? prev : vars.sketchPaths));
      if (canonicalThreadId) {
        void queryClient.invalidateQueries({
          queryKey: ['coproducer-thread', canonicalThreadId],
        });
      }
    },
  });

  // Maintain bottom lock whenever new turns, optimistic user messages, or live step updates arrive
  useLayoutEffect(() => {
    if (isPinnedToBottomRef.current) {
      scrollFeedToBottom();
    }
  }, [
    displayedMessages.length,
    lastMessageId,
    sendMut.isPending,
    liveSteps.length,
    scrollFeedToBottom,
  ]);

  // Keep feed pinned when media tiles or markdown blocks expand layout height
  useEffect(() => {
    if (typeof ResizeObserver === 'undefined') return undefined;
    const container = feedContainerRef.current;
    const content = feedContentRef.current;
    if (!container || !content) return undefined;

    const ro = new ResizeObserver(() => {
      if (isPinnedToBottomRef.current && feedContainerRef.current) {
        feedContainerRef.current.scrollTop = feedContainerRef.current.scrollHeight;
      }
    });
    ro.observe(content);
    ro.observe(container);
    return () => ro.disconnect();
  }, [canonicalThreadId]);

  const executeActionMut = useMutation({
    mutationFn: (params: {
      messageId?: string;
      proposal?: CoProducerActionProposal;
      directActionKind?: 'commission_game';
      carryOverAssetPaths?: string[];
      dismiss?: boolean;
      overrideBrief?: string;
      overrideHandoffMode?: 'loose_spark' | 'refined_concept';
      overridePauseForGreenlight?: boolean;
    }) =>
      studioApi.executeCoProducerAction({
        threadId: activeThread?.thread_id || activeThreadId,
        messageId: params.messageId,
        proposalId: params.proposal?.proposal_id,
        directActionKind: params.directActionKind,
        carryOverAssetPaths: params.carryOverAssetPaths,
        dismiss: params.dismiss,
        overrideBrief: params.overrideBrief,
        overrideHandoffMode: params.overrideHandoffMode,
        overridePauseForGreenlight: params.overridePauseForGreenlight,
      }),
    onSuccess: (res, vars) => {
      setLocalError(null);
      void queryClient.invalidateQueries({ queryKey: ['coproducer-threads'] });
      void queryClient.invalidateQueries({ queryKey: ['coproducer-thread'] });
      void queryClient.invalidateQueries({ queryKey: ['cartridges'] });
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
      void queryClient.invalidateQueries({ queryKey: ['game-memories'] });
      void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });

      if (vars.dismiss) {
        return;
      }

      const resolvedActionKind =
        vars.proposal?.action_kind || vars.directActionKind || res?.action_kind || '';
      const targetCartridgeId =
        typeof res?.result?.target_cartridge_id === 'string'
          ? res.result.target_cartridge_id
          : '';
      if (
        targetCartridgeId &&
        (resolvedActionKind === 'commission_game' || resolvedActionKind === 'pivot_fork_cartridge')
      ) {
        commitSessionSelection({ kind: 'cartridge', cartridgeId: targetCartridgeId });
        onSelectCartridge(targetCartridgeId);
        setActionNotice(`Switched to game session ${targetCartridgeId}`);
        window.setTimeout(() => setActionNotice(null), 4000);
        return;
      }

      const recWs =
        typeof res?.result?.recommended_workspace === 'string'
          ? normalizeTargetWorkspace(res.result.recommended_workspace)
          : null;
      if (recWs && recWs !== 'coproducer' && resolvedActionKind === 'start_blind_playtest') {
        onSelectWorkspace(recWs);
        return;
      }

      setActionNotice(`Executed "${vars.proposal?.title || resolvedActionKind}"`);
      window.setTimeout(() => setActionNotice(null), 3500);
    },
    onError: (err) => setLocalError(err),
  });

  const handleToggleSketchPin = (relPath: string) => {
    setSelectedSketchPaths((prev) =>
      prev.includes(relPath) ? prev.filter((p) => p !== relPath) : [...prev, relPath],
    );
  };

  const submitPrompt = (rawText: string) => {
    const trimmed = rawText.trim();
    if (!trimmed || sendMut.isPending) return;
    const sketchSnapshot = [...selectedSketchPaths];

    // Move message to history & clear input box synchronously at t = 0ms
    setComposerText('');
    setSelectedSketchPaths([]);
    setLocalError(null);
    isPinnedToBottomRef.current = true;
    setShowJumpToCurrent(false);
    setOptimisticUserTurn({
      clientId: `opt_user_${Date.now()}`,
      threadId: canonicalThreadId || '__pending_draft__',
      text: trimmed,
      sketchPaths: sketchSnapshot,
      createdAtUtc: new Date().toISOString(),
    });

    sendMut.mutate({
      promptText: trimmed,
      sketchPaths: sketchSnapshot,
    });
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    submitPrompt(composerText);
  };

  const conceptBrief = activeThread?.concept_brief ?? null;
  const effectiveDraftHandoffMode: 'refined_concept' | 'loose_spark' =
    handoffModeOverride ?? (conceptBrief?.working_title ? 'refined_concept' : 'loose_spark');

  const pendingCommissionProposal = useMemo(() => {
    for (let i = displayedMessages.length - 1; i >= 0; i -= 1) {
      const m = displayedMessages[i];
      const props = Array.isArray(m.action_proposals) ? m.action_proposals : [];
      const found = props.find(
        (p) => p.action_kind === 'commission_game' && p.status === 'pending',
      );
      if (found) {
        return { messageId: m.message_id, proposal: found };
      }
    }
    return null;
  }, [displayedMessages]);

  const handleCommissionActiveDraft = async () => {
    if (!canonicalThreadId || executeActionMut.isPending) return;
    if (pendingCommissionProposal) {
      await executeActionMut.mutateAsync({
        messageId: pendingCommissionProposal.messageId,
        proposal: pendingCommissionProposal.proposal,
        overrideHandoffMode: effectiveDraftHandoffMode,
        overridePauseForGreenlight: pauseForGreenlight,
      });
      return;
    }
    await executeActionMut.mutateAsync({
      directActionKind: 'commission_game',
      carryOverAssetPaths: selectedSketchPaths,
      overrideHandoffMode: effectiveDraftHandoffMode,
      overridePauseForGreenlight: pauseForGreenlight,
    });
  };

  const activeSessionSummary = useMemo(() => {
    if (effectiveSelectedSession) {
      const match = resolvedSessions.find((s) =>
        isSameStudioSession(s.selection, effectiveSelectedSession),
      );
      if (match) return match;
    }
    return resolvedSessions[0] ?? null;
  }, [effectiveSelectedSession, resolvedSessions]);

  const headerTitle = isDraftView
    ? conceptBrief?.working_title ||
      activeThread?.title ||
      activeSessionSummary?.title ||
      'New Game Concept'
    : activeSessionSummary?.title || gameTitle || activeSessionCartridgeId || 'Game Session';

  const starterPrompts = isDraftView ? DRAFT_STARTER_PROMPTS : CARTRIDGE_STARTER_PROMPTS;
  const mediaScopeCartridgeId = isDraftView ? undefined : activeSessionCartridgeId;
  const surfacedError = localError || threadsQuery.error || threadDetailQuery.error;
  const creatingNewSession = startingNewGame || internalCreateDraftMut.isPending;

  return (
    <div className="coproducer-shell" data-sidebar-open={sidebarOpen ? 'true' : 'false'}>
      {/* Left Game & Concept Sessions Rail */}
      {sidebarOpen ? (
        <aside className="coproducer-sidebar" aria-label="Game Sessions">
          <div className="coproducer-sidebar-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Sparkles size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
              <strong style={{ fontSize: '0.86rem' }}>Sessions</strong>
              {resolvedSessions.length > 0 ? (
                <span className="tab-count">{resolvedSessions.length}</span>
              ) : null}
            </div>

            <button
              type="button"
              className="btn btn-primary"
              onClick={handleCreateNewGameSession}
              disabled={creatingNewSession}
              title="Start a brand-new game concept session with your Creative Co-Producer"
            >
              <Plus size={13} aria-hidden="true" />
              <span>{creatingNewSession ? 'Opening...' : 'New Game'}</span>
            </button>
          </div>

          <div className="coproducer-session-list" role="list">
            {resolvedSessions.length === 0 ? (
              <div
                style={{
                  padding: '14px 12px',
                  fontSize: '0.82rem',
                  color: 'var(--text-muted)',
                  lineHeight: 1.45,
                }}
              >
                No game sessions yet. Send a message or click <strong>New Game</strong> to begin.
              </div>
            ) : (
              resolvedSessions.map((session) => {
                const selected = effectiveSelectedSession
                  ? isSameStudioSession(session.selection, effectiveSelectedSession)
                  : session.key === activeSessionSummary?.key;
                const toneBadgeClass =
                  session.statusTone === 'ok'
                    ? 'badge badge-ok'
                    : session.statusTone === 'purple'
                      ? 'badge badge-purple'
                      : session.statusTone === 'warn'
                        ? 'badge badge-warn'
                        : 'badge badge-accent';

                return (
                  <div key={session.key} role="listitem">
                    <button
                      type="button"
                      className="coproducer-session-item"
                      data-selected={selected ? 'true' : 'false'}
                      aria-current={selected ? 'true' : undefined}
                      onClick={() => commitSessionSelection(session.selection)}
                    >
                      <div className="coproducer-session-item-top">
                        <span className="coproducer-session-item-title">
                          {session.kind === 'draft' ? (
                            <Sparkles
                              size={12}
                              style={{ color: 'var(--purple)', flexShrink: 0 }}
                            />
                          ) : (
                            <Gamepad2
                              size={12}
                              style={{ color: 'var(--accent)', flexShrink: 0 }}
                            />
                          )}
                          <span>
                            {session.isPromotedActive ? '★ ' : ''}
                            {session.title}
                          </span>
                        </span>
                        <span className={toneBadgeClass}>{session.statusLabel}</span>
                      </div>

                      <div className="coproducer-session-item-meta">
                        <span>{session.subtitle}</span>
                        {session.boardProgress ? <span>· {session.boardProgress}</span> : null}
                        <span>
                          ·{' '}
                          {session.messageCount > 0
                            ? `${session.messageCount} msg${session.messageCount === 1 ? '' : 's'}`
                            : 'New'}
                        </span>
                      </div>
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </aside>
      ) : null}

      {/* Main Co-Producer Conversation Pane */}
      <div className="coproducer-main">
        {/* Session Context & Quick Actions Header */}
        <div className="coproducer-session-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => setSidebarOpen((prev) => !prev)}
              aria-expanded={sidebarOpen}
              aria-label="Toggle Game Sessions sidebar"
              title={sidebarOpen ? 'Hide Sessions sidebar' : 'Show Sessions sidebar'}
              style={{ paddingInline: '7px' }}
            >
              <PanelLeft size={14} />
            </button>

            {!sidebarOpen ? (
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleCreateNewGameSession}
                disabled={creatingNewSession}
              >
                <Plus size={13} aria-hidden="true" />
                <span>{creatingNewSession ? 'Opening...' : 'New Game'}</span>
              </button>
            ) : null}

            <strong style={{ fontSize: '0.94rem' }}>{headerTitle}</strong>

            {isDraftView ? (
              <span className="badge badge-purple">New Game Draft</span>
            ) : orchestratorState ? (
              <span className="badge badge-accent">{orchestratorState}</span>
            ) : null}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            {isDraftView && displayedMessages.length > 0 && canonicalThreadId ? (
              <>
                <label htmlFor={draftHandoffSelectId} className="visually-hidden">
                  Council Handoff Mode
                </label>
                <select
                  id={draftHandoffSelectId}
                  name="handoff_mode"
                  className="select-control"
                  value={effectiveDraftHandoffMode}
                  onChange={(e) =>
                    setHandoffModeOverride(
                      e.target.value === 'loose_spark' ? 'loose_spark' : 'refined_concept',
                    )
                  }
                  title="Choose whether the Council locks your co-designed pillars directly or runs an open genesis jam"
                >
                  <option value="refined_concept">Refined Concept (Lock Pillars)</option>
                  <option value="loose_spark">Loose Spark (Council Jam)</option>
                </select>

                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                  <input
                    id={headerGreenlightCheckboxId}
                    name="pause_for_greenlight"
                    type="checkbox"
                    checked={pauseForGreenlight}
                    onChange={(e) => setPauseForGreenlight(e.target.checked)}
                    style={{ accentColor: 'var(--accent)' }}
                  />
                  <label
                    htmlFor={headerGreenlightCheckboxId}
                    style={{
                      fontSize: '0.8rem',
                      color: 'var(--text-secondary)',
                      cursor: 'pointer',
                    }}
                    title="Pause at Greenlight Gate after Game Bible locks before code synthesis starts"
                  >
                    Greenlight Gate
                  </label>
                </div>

                <button
                  type="button"
                  className="btn btn-success"
                  disabled={executeActionMut.isPending || sendMut.isPending}
                  onClick={() => void handleCommissionActiveDraft()}
                  title="Graduate this draft session and its sketches into a live cartridge build"
                >
                  <Rocket size={13} aria-hidden="true" />
                  {executeActionMut.isPending ? 'Starting Build...' : 'Start Game Build'}
                </button>
              </>
            ) : null}

            {!isDraftView ? (
              <>
                <button
                  type="button"
                  className="btn"
                  onClick={() => onSelectWorkspace('board')}
                  title="Inspect Production Board tasks for this game session"
                >
                  <Kanban size={13} />
                  <span>
                    Board
                    {boardCounts.total > 0 ? ` (${boardCounts.approved}/${boardCounts.total})` : ''}
                  </span>
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => onSelectWorkspace('bible')}
                  title="Inspect the full Living Game Bible for this game session"
                >
                  <BookOpen size={13} />
                  <span>Game Bible</span>
                </button>
              </>
            ) : null}
          </div>
        </div>

        {/* Collapsible Living Concept Brief Strip */}
        {conceptBrief && (conceptBrief.working_title || conceptBrief.elevator_pitch) ? (
          <div className="coproducer-brief-strip">
            <details>
              <summary>
                <span>
                  <span className="badge badge-purple" style={{ marginInlineEnd: '8px' }}>
                    Living Concept Brief
                  </span>
                  {conceptBrief.working_title || 'Working Concept'}
                  {conceptBrief.core_genre ? ` · ${conceptBrief.core_genre}` : ''}
                </span>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                  Shaped Pillars & Non-Goals
                </span>
              </summary>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))',
                  gap: '8px',
                  marginBlockStart: '8px',
                  fontSize: '0.82rem',
                }}
              >
                {conceptBrief.elevator_pitch ? (
                  <div>
                    <strong>Elevator Pitch: </strong>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {conceptBrief.elevator_pitch}
                    </span>
                  </div>
                ) : null}
                {Array.isArray(conceptBrief.core_player_verbs) &&
                conceptBrief.core_player_verbs.length > 0 ? (
                  <div>
                    <strong>Core Player Verbs: </strong>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {conceptBrief.core_player_verbs.join(' · ')}
                    </span>
                  </div>
                ) : null}
                {conceptBrief.visual_style_direction ? (
                  <div>
                    <strong>Art Direction: </strong>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {conceptBrief.visual_style_direction}
                    </span>
                  </div>
                ) : null}
                {conceptBrief.audio_mood_direction ? (
                  <div>
                    <strong>Audio Mood: </strong>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {conceptBrief.audio_mood_direction}
                    </span>
                  </div>
                ) : null}
                {Array.isArray(conceptBrief.explicit_non_goals) &&
                conceptBrief.explicit_non_goals.length > 0 ? (
                  <div>
                    <strong>Explicit Non-Goals: </strong>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {conceptBrief.explicit_non_goals.join(' · ')}
                    </span>
                  </div>
                ) : null}
              </div>
            </details>
          </div>
        ) : null}

        {surfacedError ? (
          <div style={{ padding: '10px 16px 0', flexShrink: 0 }}>
            <ErrorBanner
              error={surfacedError}
              title="Co-Producer Error"
              onDismiss={() => setLocalError(null)}
            />
          </div>
        ) : null}

        {actionNotice ? (
          <div
            role="status"
            style={{
              margin: '10px 16px 0',
              padding: '8px 12px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid color-mix(in srgb, var(--ok) 45%, var(--border-subtle))',
              background: 'color-mix(in srgb, var(--ok) 10%, var(--bg-surface))',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '0.85rem',
              color: 'var(--ok)',
              fontWeight: 600,
              flexShrink: 0,
            }}
          >
            <CheckCircle2 size={15} />
            <span>{actionNotice}</span>
          </div>
        ) : null}

        {/* Viewport-Bounded Conversation Stream (Always Pinned to Current Place) */}
        <div className="coproducer-feed-stage">
          <div
            ref={feedContainerRef}
            className="coproducer-feed"
            onScroll={handleFeedScroll}
          >
            <div ref={feedContentRef} className="coproducer-feed-inner">
              {displayedMessages.length === 0 && !sendMut.isPending ? (
                <div className="coproducer-empty-card">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Sparkles size={18} style={{ color: 'var(--accent)' }} />
                      <h2 style={{ margin: 0, fontSize: '1.06rem' }}>
                        {isDraftView
                          ? 'Shape a New Game Idea with Your Creative Co-Producer'
                          : `Co-Producer Session · ${headerTitle}`}
                      </h2>
                    </div>
                    <p style={{ margin: 0, fontSize: '0.89rem', color: 'var(--text-secondary)' }}>
                      {isDraftView
                        ? 'Brainstorm boldly, summon the Specialist Council for contrasting pitches, sketch 16:9 mockups or audio auditions inline, and hand off either a loose spark or a refined concept directly into production.'
                        : 'No conversation turns yet for this game session. Ask for a concise status check, watch playtest replays, or explore a design pivot together—or open Board, Game Bible, and Playtester above for full studio detail.'}
                    </p>
                  </div>

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {starterPrompts.map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        className="btn"
                        style={{ whiteSpace: 'normal', textAlign: 'start', paddingBlock: '7px' }}
                        onClick={() => submitPrompt(prompt)}
                      >
                        <Sparkles size={13} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                        <span>{prompt}</span>
                      </button>
                    ))}
                  </div>

                  {isDraftView && onRunAutonomousBuild ? (
                    <div
                      style={{
                        borderBlockStart: '1px solid var(--border-subtle)',
                        paddingBlockStart: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '12px',
                        flexWrap: 'wrap',
                      }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                          <strong style={{ color: 'var(--text-primary)' }}>
                            Prefer a hands-off run?
                          </strong>{' '}
                          Launch autonomous Specialist Council genesis directly using any optional
                          seed typed below (or pure Player DNA if blank).
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <input
                            id={greenlightCheckboxId}
                            name="pause_for_greenlight"
                            type="checkbox"
                            checked={pauseForGreenlight}
                            onChange={(e) => setPauseForGreenlight(e.target.checked)}
                            style={{ accentColor: 'var(--accent)' }}
                          />
                          <label
                            htmlFor={greenlightCheckboxId}
                            style={{
                              fontSize: '0.83rem',
                              color: 'var(--text-secondary)',
                              cursor: 'pointer',
                            }}
                          >
                            Pause at <strong>Greenlight Gate</strong> after Game Bible locks before
                            code synthesis starts
                          </label>
                        </div>
                      </div>

                      <button
                        type="button"
                        className="btn"
                        disabled={runningAutonomousBuild}
                        onClick={() => {
                          void onRunAutonomousBuild({
                            creativePrompt: composerText.trim(),
                            pauseForGreenlight,
                          })
                            .then(() => {
                              setComposerText('');
                            })
                            .catch(() => {
                              // Surfaced via ErrorBanner while preserving composer seed
                            });
                        }}
                      >
                        <Rocket size={13} aria-hidden="true" />
                        {runningAutonomousBuild ? 'Dispatching...' : 'Run Autonomous Build'}
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : null}

              {displayedMessages.map((msg, msgIdx) => {
                const isLatestMessage = msgIdx === displayedMessages.length - 1;
                const mediaCards = Array.isArray(msg.media_cards) ? msg.media_cards : [];
                const primaryMediaCards = mediaCards.filter((c) => c.show_by_default !== false);
                const backgroundMediaCards = mediaCards.filter((c) => c.show_by_default === false);
                const consultationCards = Array.isArray(msg.consultation_cards)
                  ? msg.consultation_cards
                  : [];
                const actionProposals = Array.isArray(msg.action_proposals)
                  ? msg.action_proposals
                  : [];
                const formattedTime = formatConversationTimestamp(msg.created_at_utc);
                const isOptimisticPending =
                  sendMut.isPending && msg.message_id === optimisticUserTurn?.clientId;

                if (msg.sender_role === 'user') {
                  return (
                    <article
                      key={msg.message_id}
                      className="coproducer-msg"
                      data-role="user"
                      aria-label="Executive Producer message"
                    >
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: '8px',
                          fontSize: '0.76rem',
                          color: 'var(--text-muted)',
                        }}
                      >
                        <strong style={{ color: 'var(--accent)' }}>You</strong>
                        <span title={msg.created_at_utc}>
                          {isOptimisticPending ? 'Sent · awaiting reply' : formattedTime}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.92rem' }}>
                        <FormattedRichBlocks text={msg.content_markdown} />
                      </div>
                    </article>
                  );
                }

                const isMilestone = msg.sender_role === 'milestone_notice';

                return (
                  <article
                    key={msg.message_id}
                    className="coproducer-msg"
                    data-role={isMilestone ? 'system_milestone' : 'assistant'}
                    aria-label={isMilestone ? 'Studio Milestone Notice' : 'Co-Producer response'}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '8px',
                        flexWrap: 'wrap',
                      }}
                    >
                      <span className={isMilestone ? 'badge badge-ok' : 'badge badge-accent'}>
                        {isMilestone ? 'Studio Milestone' : 'Co-Producer'}
                      </span>
                      <span
                        style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}
                        title={msg.created_at_utc}
                      >
                        {formattedTime}
                      </span>
                    </div>

                    {/* Focused Spoken Response (shown first without clutter) */}
                    <div style={{ fontSize: '0.94rem', lineHeight: 1.56 }}>
                      <FormattedRichBlocks text={msg.content_markdown} />
                    </div>

                    {/* Curated Multimodal Media Cards Co-Producer Explicitly Wants Shown */}
                    {primaryMediaCards.map((card) => (
                      <CoProducerMediaCardView
                        key={card.card_id || card.title}
                        card={card}
                        cartridgeId={mediaScopeCartridgeId}
                        isLatestMessage={isLatestMessage}
                        selectedSketchPaths={selectedSketchPaths}
                        onToggleSketchPin={handleToggleSketchPin}
                        onOpenLightbox={onOpenLightbox}
                        onSelectWorkspace={onSelectWorkspace}
                      />
                    ))}

                    {/* Explicit 1-Click Action Proposal Cards */}
                    {actionProposals.map((proposal) => (
                      <CoProducerActionProposalCard
                        key={proposal.proposal_id}
                        messageId={msg.message_id}
                        proposal={proposal}
                        busy={executeActionMut.isPending}
                        onExecute={async (p) => {
                          await executeActionMut.mutateAsync(p);
                        }}
                      />
                    ))}

                    {/* Single Unified Observability Drill-Down ("Behind the scenes") */}
                    <CoProducerTurnObservability
                      deliberation={msg.deliberation}
                      deliberationLabel={
                        isMilestone
                          ? 'Milestone Epistemic Rationale'
                          : 'Co-Producer Epistemic Rationale'
                      }
                      consultationCards={consultationCards}
                      backgroundMediaCards={backgroundMediaCards}
                      primaryMediaCards={primaryMediaCards}
                      actionProposals={actionProposals}
                      toolTrace={msg.tool_trace}
                      promptTraces={msg.prompt_trace}
                      tokenUsage={msg.token_usage}
                      cartridgeId={mediaScopeCartridgeId}
                      selectedSketchPaths={selectedSketchPaths}
                      onToggleSketchPin={handleToggleSketchPin}
                      onOpenLightbox={onOpenLightbox}
                      onSelectWorkspace={onSelectWorkspace}
                      onPrefillPrompt={(txt) => {
                        setComposerText(txt);
                        composerRef.current?.focus();
                      }}
                    />
                  </article>
                );
              })}

              {/* Compact Live Thinking Status + Optional Step Drill-Down */}
              {sendMut.isPending || liveSteps.length > 0 ? (
                <div
                  role="status"
                  aria-live="polite"
                  className="coproducer-thinking-row"
                >
                  <div className="coproducer-thinking-headline">
                    <Loader2 size={14} className="coproducer-spin" />
                    <span>
                      {liveSteps.length > 0
                        ? liveSteps[liveSteps.length - 1].statusHeadline
                        : 'Co-Producer is thinking...'}
                    </span>
                  </div>

                  {liveSteps.length > 0 ? (
                    <details className="coproducer-live-steps-disclosure">
                      <summary>
                        {liveSteps.length} live step{liveSteps.length === 1 ? '' : 's'}
                      </summary>
                      <div className="coproducer-live-steps-list">
                        {liveSteps.map((step, sIdx) => (
                          <div
                            key={`${step.hopIndex}-${step.toolName}-${sIdx}`}
                            style={{
                              color: step.completed ? 'var(--ok)' : 'var(--text-secondary)',
                            }}
                          >
                            {step.completed ? '✓' : '→'} Hop #{step.hopIndex + 1} [{step.toolName}]:{' '}
                            {step.summary || step.statusHeadline}
                          </div>
                        ))}
                      </div>
                    </details>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>

          {showJumpToCurrent && displayedMessages.length > 0 ? (
            <button
              type="button"
              className="btn coproducer-jump-latest"
              onClick={scrollFeedToBottom}
            >
              <ArrowDown size={13} />
              <span>Jump to current</span>
            </button>
          ) : null}
        </div>

        {/* Docked Showrunner Composer */}
        <form className="coproducer-composer" onSubmit={handleSubmit}>
          {selectedSketchPaths.length > 0 ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent)' }}>
                <Pin
                  size={12}
                  style={{ display: 'inline', verticalAlign: 'middle', marginInlineEnd: '4px' }}
                />
                Pinned Sketch Refs:
              </span>
              {selectedSketchPaths.map((p) => (
                <span key={p} className="badge badge-accent" style={{ textTransform: 'none' }}>
                  {p.split('/').pop() || p}
                  <button
                    type="button"
                    onClick={() => handleToggleSketchPin(p)}
                    aria-label={`Unpin ${p}`}
                    style={{
                      border: 'none',
                      background: 'transparent',
                      padding: 0,
                      marginInlineStart: '4px',
                      cursor: 'pointer',
                      display: 'inline-flex',
                    }}
                  >
                    <X size={11} />
                  </button>
                </span>
              ))}
            </div>
          ) : null}

          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px', inlineSize: '100%' }}>
            <label htmlFor={composerInputId} className="visually-hidden">
              Message Creative Co-Producer
            </label>
            <textarea
              ref={composerRef}
              id={composerInputId}
              name="message"
              className="textarea-control coproducer-composer-input"
              rows={1}
              placeholder={
                isDraftView
                  ? 'Describe your game idea, ask for contrasting pitches, or request a 16:9 concept sketch...'
                  : 'Message Co-Producer about progress, playtest feel, art/audio variations, or pivots...'
              }
              value={composerText}
              onChange={(e) => setComposerText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (e.nativeEvent.isComposing || e.keyCode === 229) {
                    return;
                  }
                  submitPrompt(composerText);
                }
              }}
            />
            <button
              type="submit"
              className="btn btn-primary"
              disabled={!composerText.trim() || sendMut.isPending}
              style={{ minBlockSize: '42px', paddingInline: '16px' }}
            >
              <Send size={14} />
              <span>{sendMut.isPending ? 'Thinking...' : 'Send'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
