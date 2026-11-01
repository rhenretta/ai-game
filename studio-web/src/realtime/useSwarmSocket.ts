import { useCallback, useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { buildWsUrl, studioApi } from '../api/client';
import type { ChronologyStepLogEntry, SwarmStreamEvent } from '../api/types';

export interface LiveItemStep {
  itemId: string;
  agentName: string;
  stepSummary: string;
  eventType: string;
  timestampUtc: string;
  turnIndex?: number;
  specialistAgent?: string;
  brief?: string;
  startedAtUtc?: string;
  stepLog?: ChronologyStepLogEntry[];
}

export interface LivePlaytestFrame {
  sessionId: string;
  tick: number;
  frameName: string;
  cursorX?: number;
  cursorY?: number;
  timestampUtc: string;
  seq: number;
}

export interface SwarmRealtimeState {
  connected: boolean;
  lastEventAt: string | null;
  recentEvents: SwarmStreamEvent[];
  liveStepsByItem: Record<string, LiveItemStep>;
  livePlaytestFrames: Record<string, LivePlaytestFrame>;
  ensureHistorySeeded: (cartridgeId?: string) => void;
}

const MAX_RECENT_EVENTS = 100;
const MAX_STEP_LOG_PER_TURN = 40;

const BIBLE_AND_BOARD_EVENTS = new Set([
  'game_bible_updated',
  'writers_room_utterance',
  'concept_evolution_logged',
  'bible_mockup_generated',
  'backlog_item_claimed',
  'backlog_item_yielded',
  'backlog_item_proof_ready',
  'backlog_item_submitted',
  'backlog_item_approved',
  'backlog_item_failed',
  'backlog_item_blocked',
  'backlog_item_changes_requested',
  'backlog_updated',
  'genesis_triggered',
  'checkpoint_verified',
  'cartridge_promoted',
  'slice_verified',
  'slice_assembly_check_completed',
  'studio_done',
  'studio_resumed',
  'studio_paused',
  'studio_stopped',
  'studio_process_updated',
  'studio_settings_updated',
  'build_aborted',
  'game_bible_locked',
  'council_verdict',
  'qa_gate_result',
  'state_change',
  'slice_completed',
  'review_round_completed',
  'council_blind_playtest_deliberation_completed',
  'playtest_council_verdict_recorded',
  'greenlight_decision',
  'error',
]);

const CHRONOLOGY_INVALIDATION_EVENTS = new Set([
  'backlog_item_lead_planning',
  'backlog_item_lead_turn',
  'backlog_item_specialist_started',
  'backlog_item_specialist_completed',
  'backlog_item_proof_ready',
  'backlog_item_submitted',
  'backlog_item_approved',
  'backlog_item_changes_requested',
]);

function compactSingleLine(raw: string, maxChars = 140): string {
  const normalized = raw.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxChars) return normalized;
  const sliced = normalized.slice(0, maxChars - 1);
  const lastSpace = sliced.lastIndexOf(' ');
  return `${(lastSpace > maxChars * 0.6 ? sliced.slice(0, lastSpace) : sliced).trimEnd()}…`;
}

function shortRoleLabel(rawRole: string): string {
  const stripped = rawRole.includes(':') ? rawRole.slice(rawRole.lastIndexOf(':') + 1) : rawRole;
  return stripped.trim() || 'specialist';
}

export function deriveLiveItemUpdate(
  prev: LiveItemStep | undefined,
  ev: SwarmStreamEvent,
  ts: string,
): LiveItemStep | 'clear' | null {
  const payload = ev.payload && typeof ev.payload === 'object' ? ev.payload : {};
  const itemId = typeof payload.item_id === 'string' ? payload.item_id.trim() : '';
  if (!itemId) return null;

  const evType = String(ev.event_type || '').toLowerCase();
  if (
    evType === 'backlog_item_yielded' ||
    evType === 'backlog_item_proof_ready' ||
    evType === 'backlog_item_approved' ||
    evType === 'backlog_item_changes_requested' ||
    evType === 'backlog_item_failed'
  ) {
    return 'clear';
  }

  const rawTurnIdx =
    typeof payload.turn_index === 'number'
      ? payload.turn_index + 1
      : typeof payload.resumed_turn_index === 'number'
        ? payload.resumed_turn_index + 1
        : prev?.turnIndex ?? 1;

  const sameTurn = prev?.turnIndex === rawTurnIdx;
  const baseSteps: ChronologyStepLogEntry[] =
    sameTurn && Array.isArray(prev?.stepLog) ? [...prev.stepLog] : [];
  const startedAtUtc = sameTurn && prev?.startedAtUtc ? prev.startedAtUtc : ts;

  if (evType === 'context_curator_packed' || evType === 'context_curator_question_researched') {
    // Ignore non-board context curations (e.g. slice_reflection_* or council-level calls)
    // that were never claimed as an active Production Board item.
    if (!prev) return null;
    const rawTarget =
      (typeof payload.target_agent_role === 'string' && payload.target_agent_role) ||
      prev.specialistAgent ||
      prev.agentName ||
      'specialist';
    const roleLabel = shortRoleLabel(rawTarget);

    if (evType === 'context_curator_packed') {
      const chunkCount = Array.isArray(payload.selected_chunk_ids)
        ? payload.selected_chunk_ids.length
        : 0;
      const imgCount = Array.isArray(payload.selected_image_paths)
        ? payload.selected_image_paths.length
        : 0;
      const summary = `Curated ${chunkCount} context chunk${chunkCount === 1 ? '' : 's'}${
        imgCount > 0 ? ` + ${imgCount} visual ref${imgCount === 1 ? '' : 's'}` : ''
      } for ${roleLabel}`;
      baseSteps.push({
        step_type: 'context_curated',
        specialist_agent: prev.specialistAgent || 'context_curator',
        message: summary,
        timestamp_utc: ts,
      });
      return {
        ...prev,
        stepSummary: `Turn #${rawTurnIdx}: ${summary}`,
        eventType: ev.event_type,
        timestampUtc: ts,
        startedAtUtc,
        stepLog: baseSteps.slice(-MAX_STEP_LOG_PER_TURN),
      };
    }

    const question =
      typeof payload.question === 'string' ? compactSingleLine(payload.question, 95) : '';
    const qSummary = question
      ? `Researched question for ${roleLabel}: "${question}"`
      : `Researched codebase question for ${roleLabel}`;
    baseSteps.push({
      step_type: 'question_researched',
      specialist_agent: prev.specialistAgent || 'context_curator',
      message: qSummary,
      timestamp_utc: ts,
    });
    return {
      ...prev,
      stepSummary: `Turn #${rawTurnIdx}: ${qSummary}`,
      eventType: ev.event_type,
      timestampUtc: ts,
      startedAtUtc,
      stepLog: baseSteps.slice(-MAX_STEP_LOG_PER_TURN),
    };
  }

  // Only process Production Board item execution events (`backlog_item_*`); never allow unrelated
  // telemetry carrying an `item_id` field to leak raw `ev.deliberation` into `liveStepsByItem`.
  if (!evType.startsWith('backlog_item_')) {
    return null;
  }

  if (evType === 'backlog_item_claimed') {
    const ownerRole =
      (typeof payload.owner_role === 'string' && payload.owner_role) || ev.agent_name || 'TaskLead';
    const resumed = Boolean(payload.resumed) || Number(payload.resumed_turn_index || 0) > 0;
    const summary = resumed
      ? `Resuming at Turn #${rawTurnIdx}...`
      : `Initializing Task Lead (${ownerRole})...`;
    return {
      itemId,
      agentName: ownerRole,
      specialistAgent: 'task_lead',
      turnIndex: rawTurnIdx,
      stepSummary: summary,
      eventType: ev.event_type,
      timestampUtc: ts,
      startedAtUtc: ts,
      stepLog: [],
    };
  }

  if (evType === 'backlog_item_lead_planning') {
    const msg =
      (typeof payload.status_text === 'string' && payload.status_text) ||
      `Task Lead planning Turn #${rawTurnIdx} strategy`;
    baseSteps.push({
      step_type: 'planning_turn',
      specialist_agent: 'task_lead',
      message: compactSingleLine(msg, 160),
      timestamp_utc: ts,
    });
    return {
      itemId,
      agentName: 'task_lead',
      specialistAgent: 'task_lead',
      turnIndex: rawTurnIdx,
      stepSummary: `Turn #${rawTurnIdx}: Deliberating strategy & selecting specialist...`,
      eventType: ev.event_type,
      timestampUtc: ts,
      startedAtUtc,
      stepLog: baseSteps.slice(-MAX_STEP_LOG_PER_TURN),
    };
  }

  if (evType === 'backlog_item_lead_turn') {
    const spec =
      (typeof payload.specialist_agent === 'string' && payload.specialist_agent) || 'task_lead';
    const brief = typeof payload.brief === 'string' ? payload.brief : '';
    const strategy =
      (typeof payload.current_strategy === 'string' && payload.current_strategy) || brief;
    const shortStrategy = strategy ? compactSingleLine(strategy, 120) : '';
    const msg = `Task Lead delegated Turn #${rawTurnIdx} to ${spec}${shortStrategy ? `: ${shortStrategy}` : ''}`;
    baseSteps.push({
      step_type: 'lead_decision',
      specialist_agent: spec,
      message: msg,
      timestamp_utc: ts,
    });
    return {
      itemId,
      agentName: spec,
      specialistAgent: spec,
      turnIndex: rawTurnIdx,
      brief: brief || prev?.brief,
      stepSummary: `Turn #${rawTurnIdx} (${spec}): ${shortStrategy || 'Dispatching specialist...'}`,
      eventType: ev.event_type,
      timestampUtc: ts,
      startedAtUtc,
      stepLog: baseSteps.slice(-MAX_STEP_LOG_PER_TURN),
    };
  }

  if (evType === 'backlog_item_specialist_started') {
    const spec =
      (typeof payload.specialist_agent === 'string' && payload.specialist_agent) ||
      prev?.specialistAgent ||
      'specialist';
    const brief =
      (typeof payload.brief === 'string' && payload.brief) || prev?.brief || '';
    const shortBrief = brief ? compactSingleLine(brief, 120) : '';
    const msg = `${spec} started Turn #${rawTurnIdx}${shortBrief ? `: ${shortBrief}` : ''}`;
    baseSteps.push({
      step_type: 'specialist_started',
      specialist_agent: spec,
      message: msg,
      timestamp_utc: ts,
    });
    return {
      itemId,
      agentName: spec,
      specialistAgent: spec,
      turnIndex: rawTurnIdx,
      brief,
      stepSummary: `Turn #${rawTurnIdx} (${spec}): ${shortBrief || 'Synthesizing & verifying...'}`,
      eventType: ev.event_type,
      timestampUtc: ts,
      startedAtUtc: ts,
      stepLog: baseSteps.slice(-MAX_STEP_LOG_PER_TURN),
    };
  }

  if (evType === 'backlog_item_substep') {
    const spec =
      (typeof payload.specialist_agent === 'string' && payload.specialist_agent) ||
      prev?.specialistAgent ||
      ev.agent_name ||
      'specialist';
    const stepType =
      (typeof payload.step_type === 'string' && payload.step_type) || 'substep';
    const rawMsg =
      (typeof payload.message === 'string' && payload.message) ||
      (typeof payload.step_summary === 'string' && payload.step_summary) ||
      'Executing step...';
    const shortMsg = compactSingleLine(rawMsg, 125);
    const details =
      payload.details && typeof payload.details === 'object' && !Array.isArray(payload.details)
        ? (payload.details as Record<string, unknown>)
        : undefined;
    baseSteps.push({
      step_type: stepType,
      specialist_agent: spec,
      message: shortMsg,
      details,
      timestamp_utc: ts,
    });
    return {
      itemId,
      agentName: spec,
      specialistAgent: spec,
      turnIndex: rawTurnIdx,
      brief: prev?.brief,
      stepSummary: `Turn #${rawTurnIdx} (${spec}): ${shortMsg}`,
      eventType: ev.event_type,
      timestampUtc: ts,
      startedAtUtc,
      stepLog: baseSteps.slice(-MAX_STEP_LOG_PER_TURN),
    };
  }

  if (evType === 'backlog_item_specialist_completed') {
    const spec =
      (typeof payload.specialist_agent === 'string' && payload.specialist_agent) ||
      prev?.specialistAgent ||
      'specialist';
    const summary = typeof payload.summary === 'string' ? compactSingleLine(payload.summary, 120) : '';
    baseSteps.push({
      step_type: 'specialist_completed',
      specialist_agent: spec,
      message: `${spec} completed Turn #${rawTurnIdx}${summary ? `: ${summary}` : ''}`,
      timestamp_utc: ts,
    });
    return {
      itemId,
      agentName: spec,
      specialistAgent: spec,
      turnIndex: rawTurnIdx,
      brief: prev?.brief,
      stepSummary: `Turn #${rawTurnIdx}: ${spec} finished — Task Lead evaluating proof...`,
      eventType: ev.event_type,
      timestampUtc: ts,
      startedAtUtc,
      stepLog: baseSteps.slice(-MAX_STEP_LOG_PER_TURN),
    };
  }

  const fallbackText =
    (typeof payload.step_summary === 'string' && payload.step_summary) ||
    (typeof payload.message === 'string' && payload.message) ||
    (typeof payload.summary === 'string' && payload.summary) ||
    ev.event_type.replace(/^backlog_item_/, '').replace(/_/g, ' ');

  return {
    itemId,
    agentName: prev?.specialistAgent || ev.agent_name || 'TaskWorker',
    specialistAgent: prev?.specialistAgent,
    turnIndex: prev?.turnIndex,
    brief: prev?.brief,
    stepSummary: compactSingleLine(fallbackText, 130),
    eventType: ev.event_type,
    timestampUtc: ts,
    startedAtUtc,
    stepLog: baseSteps,
  };
}

export interface SwarmSocketOptions {
  seedHistoryWhenActive?: boolean;
  cartridgeId?: string;
}

/**
 * Single WebSocket connection to `/ws/swarm` replacing the legacy 143.7 MB 2.5s polling storm.
 * Debounces React Query invalidations when board/bible/playtest delta events arrive, and optionally
 * seeds a bounded 50-event tail once when attaching to an already-running build.
 */
export function useSwarmSocket(options?: SwarmSocketOptions): SwarmRealtimeState {
  const queryClient = useQueryClient();
  const [connected, setConnected] = useState(false);
  const [reconnectEpoch, setReconnectEpoch] = useState(0);
  const [lastEventAt, setLastEventAt] = useState<string | null>(null);
  const [recentEvents, setRecentEvents] = useState<SwarmStreamEvent[]>([]);
  const [liveStepsByItem, setLiveStepsByItem] = useState<Record<string, LiveItemStep>>({});
  const [livePlaytestFrames, setLivePlaytestFrames] = useState<Record<string, LivePlaytestFrame>>({});
  const invalidateTimerRef = useRef<number | null>(null);
  const chronologyInvalidateTimerRef = useRef<number | null>(null);
  const seededEpochRef = useRef<string>('');
  const unmountedRef = useRef(false);

  useEffect(() => {
    let unmounted = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let attempt = 0;
    let hasConnectedOnce = false;
    let disconnectedSinceOpen = false;

    const invalidateAllLiveQueries = () => {
      if (invalidateTimerRef.current !== null) {
        window.clearTimeout(invalidateTimerRef.current);
        invalidateTimerRef.current = null;
      }
      if (chronologyInvalidateTimerRef.current !== null) {
        window.clearTimeout(chronologyInvalidateTimerRef.current);
        chronologyInvalidateTimerRef.current = null;
      }
      void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
      void queryClient.invalidateQueries({ queryKey: ['cartridges'] });
      void queryClient.invalidateQueries({ queryKey: ['studio-status'] });
      void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });
      void queryClient.invalidateQueries({ queryKey: ['playtest-session-detail'] });
      void queryClient.invalidateQueries({ queryKey: ['board-chronology'] });
      void queryClient.invalidateQueries({ queryKey: ['artifacts'] });
      void queryClient.invalidateQueries({ queryKey: ['game-memories'] });
    };

    const scheduleBibleRefresh = () => {
      if (invalidateTimerRef.current !== null) return;
      invalidateTimerRef.current = window.setTimeout(() => {
        invalidateTimerRef.current = null;
        void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
        void queryClient.invalidateQueries({ queryKey: ['cartridges'] });
      }, 450);
    };

    const scheduleChronologyRefresh = () => {
      if (chronologyInvalidateTimerRef.current !== null) return;
      chronologyInvalidateTimerRef.current = window.setTimeout(() => {
        chronologyInvalidateTimerRef.current = null;
        void queryClient.invalidateQueries({ queryKey: ['board-chronology'] });
      }, 650);
    };

    const connect = () => {
      if (unmounted) return;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      try {
        ws = new WebSocket(buildWsUrl('/ws/swarm'));
      } catch {
        disconnectedSinceOpen = true;
        setConnected(false);
        reconnectTimer = window.setTimeout(connect, Math.min(8000, 1000 * 2 ** attempt));
        attempt += 1;
        return;
      }

      ws.onopen = () => {
        if (unmounted) return;
        attempt = 0;
        setConnected(true);
        if (hasConnectedOnce || disconnectedSinceOpen) {
          disconnectedSinceOpen = false;
          setReconnectEpoch((n) => n + 1);
          invalidateAllLiveQueries();
        }
        hasConnectedOnce = true;
      };

      ws.onmessage = (msgEvent) => {
        if (unmounted) return;
        let parsed: SwarmStreamEvent | null = null;
        try {
          parsed = JSON.parse(String(msgEvent.data)) as SwarmStreamEvent;
        } catch {
          return;
        }
        if (!parsed || typeof parsed !== 'object') return;
        const evTypeLower = String(parsed.event_type || '').toLowerCase();
        if (evTypeLower === 'echo' || evTypeLower === 'connected') return;

        const ts = parsed.timestamp_utc || new Date().toISOString();
        setLastEventAt(ts);
        setRecentEvents((prev) => [parsed, ...prev].slice(0, MAX_RECENT_EVENTS));

        const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
        const itemId = typeof payload.item_id === 'string' ? payload.item_id.trim() : '';
        if (itemId) {
          setLiveStepsByItem((prev) => {
            const next = deriveLiveItemUpdate(prev[itemId], parsed, ts);
            if (next === null) return prev;
            if (next === 'clear') {
              if (!(itemId in prev)) return prev;
              const copy = { ...prev };
              delete copy[itemId];
              return copy;
            }
            return { ...prev, [itemId]: next };
          });
        }

        if (BIBLE_AND_BOARD_EVENTS.has(evTypeLower)) {
          scheduleBibleRefresh();
        }
        if (CHRONOLOGY_INVALIDATION_EVENTS.has(evTypeLower)) {
          scheduleChronologyRefresh();
        }
        if (evTypeLower.includes('artifact') || evTypeLower === 'backlog_item_approved') {
          void queryClient.invalidateQueries({ queryKey: ['artifacts'] });
        }
        if (evTypeLower === 'game_memory_updated' || evTypeLower === 'slice_verified') {
          void queryClient.invalidateQueries({ queryKey: ['game-memories'] });
        }
        if (evTypeLower === 'playtest_frame_update' || evTypeLower === 'playtest_turn_completed') {
          const sessionId =
            typeof payload.session_id === 'string' ? payload.session_id.trim() : '';
          if (sessionId) {
            const frameName =
              (typeof payload.frame_name === 'string' && payload.frame_name.trim()) ||
              'live_preview.jpg';
            const tick =
              typeof payload.tick === 'number'
                ? payload.tick
                : typeof payload.sim_tick === 'number'
                  ? payload.sim_tick
                  : 0;
            const cursorX = typeof payload.cursor_x === 'number' ? payload.cursor_x : undefined;
            const cursorY = typeof payload.cursor_y === 'number' ? payload.cursor_y : undefined;
            setLivePlaytestFrames((prev) => ({
              ...prev,
              [sessionId]: {
                sessionId,
                tick,
                frameName,
                cursorX: cursorX ?? prev[sessionId]?.cursorX,
                cursorY: cursorY ?? prev[sessionId]?.cursorY,
                timestampUtc: ts,
                seq: (prev[sessionId]?.seq ?? 0) + 1,
              },
            }));
          }
        }
        if (
          (evTypeLower.includes('playtest') && evTypeLower !== 'playtest_frame_update') ||
          evTypeLower === 'council_playtest_review_completed'
        ) {
          void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });
          void queryClient.invalidateQueries({ queryKey: ['playtest-session-detail'] });
        }
      };

      ws.onclose = () => {
        if (unmounted) return;
        disconnectedSinceOpen = true;
        setConnected(false);
        setLiveStepsByItem((prev) => (Object.keys(prev).length === 0 ? prev : {}));
        if (hasConnectedOnce) {
          scheduleBibleRefresh();
        }
        reconnectTimer = window.setTimeout(connect, Math.min(8000, 1000 * 2 ** Math.min(attempt, 3)));
        attempt += 1;
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    const handleVisibilityOrOnline = () => {
      if (unmounted) return;
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        return;
      }
      if (!ws || (ws.readyState !== WebSocket.OPEN && ws.readyState !== WebSocket.CONNECTING)) {
        attempt = 0;
        connect();
      } else if (hasConnectedOnce) {
        void queryClient.invalidateQueries({ queryKey: ['game-bible'] });
        void queryClient.invalidateQueries({ queryKey: ['cartridges'] });
        void queryClient.invalidateQueries({ queryKey: ['playtest-sessions'] });
      }
    };

    unmountedRef.current = false;
    connect();
    document.addEventListener('visibilitychange', handleVisibilityOrOnline);
    window.addEventListener('online', handleVisibilityOrOnline);

    return () => {
      unmounted = true;
      unmountedRef.current = true;
      document.removeEventListener('visibilitychange', handleVisibilityOrOnline);
      window.removeEventListener('online', handleVisibilityOrOnline);
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      if (invalidateTimerRef.current !== null) window.clearTimeout(invalidateTimerRef.current);
      if (chronologyInvalidateTimerRef.current !== null) {
        window.clearTimeout(chronologyInvalidateTimerRef.current);
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [queryClient]);

  const ensureHistorySeeded = useCallback(
    (cartridgeId?: string) => {
      const normalizedCid = cartridgeId && cartridgeId !== 'active' ? cartridgeId : undefined;
      const epochKey = `${reconnectEpoch}:${normalizedCid ?? 'active'}`;
      if (seededEpochRef.current === epochKey) return;
      seededEpochRef.current = epochKey;

      void studioApi
        .getRecentEvents({ limit: 50, cartridgeId: normalizedCid })
        .then((res) => {
          if (unmountedRef.current || !Array.isArray(res.events) || res.events.length === 0) return;
          const chronological = res.events.filter(
            (ev): ev is SwarmStreamEvent =>
              Boolean(ev && typeof ev === 'object' && ev.event_type && ev.event_type !== 'connected'),
          );
          if (chronological.length === 0) return;

          const newestFirst = [...chronological].reverse();
          setLastEventAt((prev) => prev || newestFirst[0]?.timestamp_utc || null);
          setRecentEvents((prev) => {
            const seen = new Set<string>();
            const merged: SwarmStreamEvent[] = [];
            for (const ev of [...prev, ...newestFirst]) {
              const key =
                ev.event_id || `${ev.timestamp_utc}|${ev.event_type}|${ev.agent_name}`;
              if (seen.has(key)) continue;
              seen.add(key);
              merged.push(ev);
            }
            merged.sort((a, b) => (b.timestamp_utc || '').localeCompare(a.timestamp_utc || ''));
            return merged.slice(0, MAX_RECENT_EVENTS);
          });

          setLiveStepsByItem((prev) => {
            const nextMap: Record<string, LiveItemStep> = {};
            for (const ev of chronological) {
              const payload = ev.payload && typeof ev.payload === 'object' ? ev.payload : {};
              const itemId = typeof payload.item_id === 'string' ? payload.item_id.trim() : '';
              if (!itemId) continue;
              const ts = ev.timestamp_utc || new Date().toISOString();
              const derived = deriveLiveItemUpdate(nextMap[itemId], ev, ts);
              if (derived === 'clear') {
                delete nextMap[itemId];
              } else if (derived !== null) {
                nextMap[itemId] = derived;
              }
            }
            return { ...nextMap, ...prev };
          });
        })
        .catch(() => {
          // Ignore seed errors; live WebSocket delta stream remains authoritative
        });
    },
    [reconnectEpoch],
  );

  const seedActive = Boolean(options?.seedHistoryWhenActive);
  const seedCartridgeId = options?.cartridgeId;

  useEffect(() => {
    if (!seedActive) return;
    ensureHistorySeeded(seedCartridgeId);
  }, [seedActive, seedCartridgeId, ensureHistorySeeded]);

  return {
    connected,
    lastEventAt,
    recentEvents,
    liveStepsByItem,
    livePlaytestFrames,
    ensureHistorySeeded,
  };
}
