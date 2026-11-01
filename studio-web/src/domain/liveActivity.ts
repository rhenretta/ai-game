import type {
  GameBibleResponse,
  PlaytestSessionSummary,
  SwarmStreamEvent,
  WorkspaceId,
} from '../api/types';
import type { LiveItemStep } from '../realtime/useSwarmSocket';
import { isWorkingBoardState } from './boardCards';
import { getRoleMetadata } from './roles';

export type StudioPipelinePhase =
  | 'idle'
  | 'pausing'
  | 'awaiting_greenlight'
  | 'writers_room'
  | 'board_synthesis'
  | 'council_review'
  | 'slice_assembly_check'
  | 'blind_playtest'
  | 'qa_verification'
  | 'qa_repair'
  | 'promoting';

export interface LiveAiCallSummary {
  callNumber?: number;
  agentLabel: string;
  stepLabel: string;
  totalTokens: number;
  thoughtTokens: number;
  timestampUtc: string;
}

export interface LiveStudioActivity {
  isActive: boolean;
  phase: StudioPipelinePhase;
  phaseBadge: string;
  stageTitle: string;
  activeAgentLabel: string;
  headline: string;
  detail?: string;
  whyPostBoardActive?: string;
  progressValue?: number;
  progressMax?: number;
  suggestedWorkspace?: { id: WorkspaceId; label: string };
  latestAiCall?: LiveAiCallSummary;
  lastActivityUtc: string | null;
}

export interface FormattedSwarmEventSummary {
  category: 'stage' | 'board' | 'ai';
  agentLabel: string;
  title: string;
  subtitle?: string;
  badgeText?: string;
  badgeTone: 'accent' | 'ok' | 'warn' | 'danger' | 'neutral';
}

function asRecord(val: unknown): Record<string, unknown> {
  return val && typeof val === 'object' && !Array.isArray(val)
    ? (val as Record<string, unknown>)
    : {};
}

function asString(val: unknown): string {
  return typeof val === 'string' ? val.trim() : '';
}

function asNumber(val: unknown): number | undefined {
  return typeof val === 'number' && Number.isFinite(val) ? val : undefined;
}

export function formatTokenCount(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens <= 0) return '0 tok';
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(2)}M tok`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}k tok`;
  return `${Math.round(tokens)} tok`;
}

export function formatRelativeSeconds(isoUtc: string | null | undefined, nowMs = Date.now()): string {
  if (!isoUtc) return '';
  const tsMs = Date.parse(isoUtc);
  if (!Number.isFinite(tsMs)) return '';
  const diffSec = Math.max(0, Math.round((nowMs - tsMs) / 1000));
  if (diffSec < 5) return 'just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  const mins = Math.floor(diffSec / 60);
  const secs = diffSec % 60;
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s ago` : `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  return `${hrs}h ${mins % 60}m ago`;
}

function parseAiCallLabel(rawLabel: string, fallbackAgent: string): { agentLabel: string; stepLabel: string } {
  const cleaned = rawLabel.trim();
  if (!cleaned || cleaned === 'genai.models.generate_content') {
    const meta = getRoleMetadata(fallbackAgent === 'genai' ? 'StudioAgent' : fallbackAgent);
    return {
      agentLabel: fallbackAgent === 'genai' ? 'Gemini Gateway' : meta.label,
      stepLabel: 'generate_content',
    };
  }
  const dotIdx = cleaned.indexOf('.');
  const rawRole = dotIdx > 0 ? cleaned.slice(0, dotIdx) : fallbackAgent;
  const rest = dotIdx > 0 ? cleaned.slice(dotIdx + 1) : cleaned;
  // Extract bracket suffix like `[candidate_xyz:iter2:hop1]` -> `iter2:hop1`
  const bracketMatch = rest.match(/^([^\[]+)\[([^\]]+)\]$/);
  let stepLabel = rest;
  if (bracketMatch) {
    const methodPart = bracketMatch[1];
    const parts = bracketMatch[2].split(':').filter((p) => !p.startsWith('candidate_'));
    stepLabel = parts.length > 0 ? `${methodPart} (${parts.join(':')})` : methodPart;
  }
  const roleMeta = getRoleMetadata(rawRole);
  return {
    agentLabel: rawRole === 'genai' ? 'Gemini Gateway' : roleMeta.label,
    stepLabel,
  };
}

function extractLatestAiCall(recentEvents: SwarmStreamEvent[]): LiveAiCallSummary | undefined {
  for (const ev of recentEvents) {
    if (String(ev.event_type || '').toLowerCase() !== 'studio_live_activity') continue;
    const pl = asRecord(ev.payload);
    if (pl.phase !== 'llm_gateway_call') continue;
    const stats = asRecord(pl.token_stats);
    const totalTokens = asNumber(stats.total_tokens) ?? 0;
    const thoughtTokens = asNumber(stats.thought_tokens) ?? 0;
    const rawLabel = asString(pl.label);
    const parsed = parseAiCallLabel(rawLabel, ev.agent_name || 'StudioAgent');
    return {
      callNumber: asNumber(pl.total_calls),
      agentLabel: parsed.agentLabel,
      stepLabel: parsed.stepLabel,
      totalTokens,
      thoughtTokens,
      timestampUtc: ev.timestamp_utc,
    };
  }
  return undefined;
}

interface QaFailureContext {
  attempt?: number;
  maxRetries?: number;
  failureGate: string;
  failureReason: string;
  remedyDirective: string;
}

function findLatestQaFailure(recentEvents: SwarmStreamEvent[]): QaFailureContext | undefined {
  for (const ev of recentEvents) {
    const et = String(ev.event_type || '').toUpperCase();
    if (et === 'CARTRIDGE_PROMOTED' || et === 'CHECKPOINT_VERIFIED') return undefined;
    if (et === 'QA_GATE_RESULT') {
      const pl = asRecord(ev.payload);
      if (pl.promoted === true) return undefined;
      return {
        attempt: asNumber(pl.attempt),
        maxRetries: asNumber(pl.max_retries),
        failureGate: asString(pl.failure_gate) || 'Quality Gate',
        failureReason: asString(pl.failure_reason) || asString(ev.deliberation),
        remedyDirective: asString(pl.remedy_directive),
      };
    }
  }
  return undefined;
}

/**
 * Pure structured derivation of what the Autonomous Game Studio is currently executing,
 * bridging post-board phases (Slice Assembly Check, Blind Playtest, 5-Gate QA Verification,
 * SWE Self-Healing Repair, Checkpoint Promotion) and live LLM gateway heartbeats.
 */
export function deriveLiveStudioActivity(params: {
  bible: GameBibleResponse | undefined;
  recentEvents: SwarmStreamEvent[];
  liveStepsByItem: Record<string, LiveItemStep>;
  playtestSessions?: PlaytestSessionSummary[];
}): LiveStudioActivity {
  const { bible, recentEvents, liveStepsByItem, playtestSessions = [] } = params;
  const proc = bible?.process_status;
  const processState = proc?.process_state ?? 'stopped';
  const isRunning = Boolean(proc?.build_actively_running || processState === 'running');
  const isPausing = processState === 'pausing';
  const isAwaitingGreenlight =
    processState === 'awaiting_greenlight' || bible?.orchestrator_state === 'AWAITING_GREENLIGHT';

  const latestAiCall = extractLatestAiCall(recentEvents);
  const lastActivityUtc =
    recentEvents[0]?.timestamp_utc || proc?.last_event_timestamp_utc || bible?.updated_at_utc || null;

  if (!isRunning && !isPausing && !isAwaitingGreenlight) {
    return {
      isActive: false,
      phase: 'idle',
      phaseBadge: 'Idle',
      stageTitle: 'Studio Idle',
      activeAgentLabel: 'Studio',
      headline: 'No active build thread is running.',
      latestAiCall,
      lastActivityUtc,
    };
  }

  if (isAwaitingGreenlight) {
    return {
      isActive: true,
      phase: 'awaiting_greenlight',
      phaseBadge: 'Awaiting Greenlight',
      stageTitle: 'Waiting for Executive Producer Greenlight',
      activeAgentLabel: 'Executive Producer',
      headline: 'Living Game Bible drafted — review and click Greenlight Bible or send steering.',
      suggestedWorkspace: { id: 'writers-room', label: "Open Writers' Room" },
      latestAiCall,
      lastActivityUtc,
    };
  }

  if (isPausing) {
    const workers = proc?.scheduler_running_workers ?? 0;
    return {
      isActive: true,
      phase: 'pausing',
      phaseBadge: 'Pausing',
      stageTitle: 'Draining In-Flight Turn Before Pausing',
      activeAgentLabel: 'Board Scheduler',
      headline:
        workers > 0
          ? `Waiting for ${workers} active worker${workers === 1 ? '' : 's'} to reach a clean turn checkpoint...`
          : 'Completing current step before pausing build...',
      latestAiCall,
      lastActivityUtc,
    };
  }

  // Inspect board state
  const boardItems = (bible?.production_board?.items ?? []).filter((i) => i.state !== 'cancelled');
  const totalBoard = boardItems.length;
  const approvedBoard = boardItems.filter((i) => i.state === 'approved').length;
  const workingBoardItems = boardItems.filter((i) => isWorkingBoardState(i.state));
  const awaitingReviewItems = boardItems.filter((i) => i.state === 'awaiting_review');
  const allBoardApproved = totalBoard > 0 && approvedBoard === totalBoard;

  // Check for actively running blind playtest session
  const livePlaytest = playtestSessions.find(
    (s) => s.status === 'running' || s.status === 'starting',
  );
  if (livePlaytest) {
    const turnCount = livePlaytest.turn_count ?? 0;
    const planSuffix = livePlaytest.current_plan ? ` — ${livePlaytest.current_plan}` : '';
    return {
      isActive: true,
      phase: 'blind_playtest',
      phaseBadge: `Playtest · T${turnCount}`,
      stageTitle: 'Council Blind AI Playtest',
      activeAgentLabel: 'Blind Playtester',
      headline: `Playing '${livePlaytest.game_title || bible?.game_title || 'candidate'}' unguided (Turn ${turnCount})${planSuffix}`,
      whyPostBoardActive: allBoardApproved
        ? `All ${approvedBoard}/${totalBoard} Production Board items are approved — Blind Playtester is now verifying the integrated slice.`
        : undefined,
      suggestedWorkspace: { id: 'playtest', label: 'Watch Live Playtest' },
      latestAiCall,
      lastActivityUtc,
    };
  }

  // Scan recentEvents (newest-first) for the most recent phase-defining event (skipping pure llm_gateway_call heartbeats)
  const qaFailure = findLatestQaFailure(recentEvents);

  for (const ev of recentEvents) {
    const rawType = String(ev.event_type || '');
    const lowerType = rawType.toLowerCase();
    const pl = asRecord(ev.payload);

    if (lowerType === 'studio_live_activity') {
      if (pl.stage === 'supervisor_gate_evaluation') {
        const att = asNumber(pl.attempt);
        const maxAtt = asNumber(pl.max_retries);
        const attSuffix = att ? ` (Attempt ${att}${maxAtt ? `/${maxAtt}` : ''})` : '';
        return {
          isActive: true,
          phase: 'qa_verification',
          phaseBadge: att ? `QA Gates · ${att}/${maxAtt ?? 3}` : 'QA Gates',
          stageTitle: `5-Gate QA Verification${attSuffix}`,
          activeAgentLabel: 'QA Build Supervisor',
          headline:
            asString(ev.deliberation) ||
            `Evaluating candidate cartridge across headless Godot 4 Quality Gates${attSuffix}...`,
          whyPostBoardActive: allBoardApproved
            ? `All ${approvedBoard}/${totalBoard} board tasks are approved and Council deliberation concluded — running automated Quality Gates before slice promotion.`
            : undefined,
          progressValue: att,
          progressMax: maxAtt,
          latestAiCall,
          lastActivityUtc,
        };
      }
      continue;
    }

    // 1. Automated QA Self-Healing Repair (`AGENT_STREAM` with `phase === 'REPAIR'`)
    if (lowerType === 'agent_stream' && String(pl.phase || '').toUpperCase() === 'REPAIR') {
      const gate = asString(pl.failure_gate) || qaFailure?.failureGate || 'Quality Gate';
      const remedy = asString(pl.remedy_directive) || qaFailure?.remedyDirective;
      const att = qaFailure?.attempt;
      const maxAtt = qaFailure?.maxRetries;
      const attLabel = att ? ` (Attempt ${att}${maxAtt ? `/${maxAtt}` : ''})` : '';
      const reasonSuffix = qaFailure?.failureReason ? ` — ${qaFailure.failureReason}` : '';
      return {
        isActive: true,
        phase: 'qa_repair',
        phaseBadge: att ? `QA Repair · ${att}/${maxAtt ?? 3}` : 'QA Repair',
        stageTitle: `QA Self-Healing Repair${attLabel}`,
        activeAgentLabel: getRoleMetadata(ev.agent_name || 'LeadGameplaySWEAgent').label,
        headline: `Optimizing & repairing ${gate}${reasonSuffix}`,
        detail: remedy || undefined,
        whyPostBoardActive: allBoardApproved
          ? `All ${approvedBoard}/${totalBoard} board tasks are approved and Council signed off — Lead Gameplay SWE is resolving ${gate} before final promotion.`
          : undefined,
        progressValue: att,
        progressMax: maxAtt,
        latestAiCall,
        lastActivityUtc,
      };
    }

    // 2. QA Gate Result / Verification State
    if (lowerType === 'qa_gate_result') {
      if (pl.promoted === true) {
        return {
          isActive: true,
          phase: 'promoting',
          phaseBadge: 'Promoting',
          stageTitle: 'Checkpoint & Cartridge Promotion',
          activeAgentLabel: 'QA Build Supervisor',
          headline: 'All Quality Gates passed — sealing checkpoint and promoting cartridge...',
          latestAiCall,
          lastActivityUtc,
        };
      }
      const gate = asString(pl.failure_gate) || 'Quality Gate';
      const reason = asString(pl.failure_reason) || asString(ev.deliberation);
      const att = asNumber(pl.attempt);
      const maxAtt = asNumber(pl.max_retries);
      const attLabel = att ? ` (Attempt ${att}${maxAtt ? `/${maxAtt}` : ''})` : '';
      return {
        isActive: true,
        phase: 'qa_repair',
        phaseBadge: att ? `QA Repair · ${att}/${maxAtt ?? 3}` : 'QA Repair',
        stageTitle: `QA Gate Retry${attLabel}`,
        activeAgentLabel: 'QA Build Supervisor',
        headline: `${gate} failed${attLabel}: ${reason}`,
        detail: asString(pl.remedy_directive) || undefined,
        whyPostBoardActive: allBoardApproved
          ? `All ${approvedBoard}/${totalBoard} board tasks are approved — running QA Gate verification & self-healing repair.`
          : undefined,
        progressValue: att,
        progressMax: maxAtt,
        latestAiCall,
        lastActivityUtc,
      };
    }

    if (lowerType === 'state_change') {
      const st = String(pl.lifecycle_state || pl.state || '').toUpperCase();
      if (st === 'VERIFYING') {
        return {
          isActive: true,
          phase: 'qa_verification',
          phaseBadge: 'QA Gates',
          stageTitle: '5-Gate QA Verification',
          activeAgentLabel: 'QA Build Supervisor',
          headline:
            asString(ev.deliberation) ||
            'Evaluating integrated candidate cartridge across 5 headless Godot Quality Gates...',
          whyPostBoardActive: allBoardApproved
            ? `All ${approvedBoard}/${totalBoard} board tasks are approved and Council signed off — running final Quality Gates before slice promotion.`
            : undefined,
          latestAiCall,
          lastActivityUtc,
        };
      }
      if (st === 'PROMOTED') {
        return {
          isActive: true,
          phase: 'promoting',
          phaseBadge: 'Promoting',
          stageTitle: 'Promoting Verified Slice',
          activeAgentLabel: 'QA Build Supervisor',
          headline: 'Promoting verified build to active cartridge...',
          latestAiCall,
          lastActivityUtc,
        };
      }
    }

    // 3. Structured `writers_room_activity` stage updates (slice_assembly_check, slice_blind_playtest, proof_review, deliberation)
    if (lowerType === 'writers_room_activity') {
      const statusText = asString(pl.status_text) || asString(ev.deliberation);
      const roleMeta = getRoleMetadata(asString(pl.agent_role) || ev.agent_name);
      const delibText = asString(ev.deliberation);

      if (pl.phase === 'slice_qa_gate_check' || delibText.includes('(slice_qa_gate_check)')) {
        return {
          isActive: true,
          phase: 'qa_verification',
          phaseBadge: 'QA Gates',
          stageTitle: 'Slice 5-Gate QA Verification',
          activeAgentLabel: roleMeta.label,
          headline:
            statusText ||
            'Running 5-Gate QA verification (Syntax, Assets, Headless Sim, 60 FPS Frame Budget, Telemetry)...',
          whyPostBoardActive: allBoardApproved
            ? `All ${approvedBoard}/${totalBoard} board tasks are approved — running 5-Gate QA verification (any gate defect will spawn a new Production Board task).`
            : undefined,
          latestAiCall,
          lastActivityUtc,
        };
      }

      if (delibText.includes('(slice_assembly_check)')) {
        return {
          isActive: true,
          phase: 'slice_assembly_check',
          phaseBadge: 'Assembly Check',
          stageTitle: 'Slice Assembly & Boot Sanity Check',
          activeAgentLabel: roleMeta.label,
          headline:
            statusText ||
            'Verifying entrypoint reachability and zero-input boot sanity before Blind Playtest...',
          whyPostBoardActive: allBoardApproved
            ? `All ${approvedBoard}/${totalBoard} board tasks are approved — verifying full scene assembly before Blind Playtest.`
            : undefined,
          latestAiCall,
          lastActivityUtc,
        };
      }

      if (delibText.includes('(slice_blind_playtest)')) {
        return {
          isActive: true,
          phase: 'blind_playtest',
          phaseBadge: 'Playtest Review',
          stageTitle: 'Council Blind Playtest Evaluation',
          activeAgentLabel: roleMeta.label,
          headline:
            statusText ||
            'Council specialists evaluating the Blind Playtester Experience Report and action tape...',
          whyPostBoardActive: allBoardApproved
            ? `All ${approvedBoard}/${totalBoard} board tasks are approved — Council is auditing Blind Playtest telemetry.`
            : undefined,
          suggestedWorkspace: { id: 'playtest', label: 'Inspect Playtest' },
          latestAiCall,
          lastActivityUtc,
        };
      }

      if (delibText.includes('(proof_review)')) {
        return {
          isActive: true,
          phase: 'council_review',
          phaseBadge: 'Proof Review',
          stageTitle: 'Council Proof Inspection',
          activeAgentLabel: roleMeta.label,
          headline:
            statusText ||
            'Council review panel inspecting rendered proof filmstrips and runtime state assertions...',
          suggestedWorkspace: { id: 'board', label: 'View Board' },
          latestAiCall,
          lastActivityUtc,
        };
      }

      return {
        isActive: true,
        phase: 'writers_room',
        phaseBadge: 'Council',
        stageTitle: "Council Writers' Room Deliberation",
        activeAgentLabel: roleMeta.label,
        headline: statusText || `${roleMeta.label} is deliberating in the Writers' Room...`,
        suggestedWorkspace: { id: 'writers-room', label: "Open Writers' Room" },
        latestAiCall,
        lastActivityUtc,
      };
    }

    // 4. Playtest lifecycle events before verdict finishes
    if (
      lowerType === 'playtest_report_ready' ||
      lowerType === 'playtest_video_ready' ||
      lowerType === 'playtest_turn_completed'
    ) {
      return {
        isActive: true,
        phase: 'blind_playtest',
        phaseBadge: 'Playtest Review',
        stageTitle: 'Council Blind Playtest Evaluation',
        activeAgentLabel: 'Council Review Panel',
        headline:
          lowerType === 'playtest_turn_completed'
            ? `Blind Playtester executing live session turns...`
            : `Blind Playtest session finished — Council specialists deliberating on Experience Report...`,
        suggestedWorkspace: { id: 'playtest', label: 'Inspect Playtest' },
        latestAiCall,
        lastActivityUtc,
      };
    }

    // 5. Board work item execution events
    if (lowerType.startsWith('backlog_item_') && workingBoardItems.length > 0) {
      break;
    }

    // 6. Writers' Room utterances / roadmap updates
    if (
      lowerType === 'writers_room_utterance' ||
      lowerType === 'concept_evolution_logged' ||
      lowerType === 'bible_mockup_generated' ||
      lowerType === 'slice_strategy_synthesized'
    ) {
      if (workingBoardItems.length === 0) {
        const speakerMeta = getRoleMetadata(ev.agent_name);
        return {
          isActive: true,
          phase: 'writers_room',
          phaseBadge: 'Council',
          stageTitle: "Council Writers' Room Deliberation",
          activeAgentLabel: speakerMeta.label,
          headline: `Council deliberating after ${speakerMeta.label}'s turn (${approvedBoard}/${totalBoard} board tasks approved)...`,
          suggestedWorkspace: { id: 'writers-room', label: "Open Writers' Room" },
          latestAiCall,
          lastActivityUtc,
        };
      }
      break;
    }
  }

  // Fallback to live board item state or orchestrator state
  const finishedItemIds = new Set(
    boardItems.filter((i) => i.state === 'approved').map((i) => i.item_id),
  );
  const liveSteps = Object.values(liveStepsByItem).filter((s) => !finishedItemIds.has(s.itemId));
  if (liveSteps.length > 0 || workingBoardItems.length > 0) {
    const workers = proc?.scheduler_running_workers || workingBoardItems.length || liveSteps.length;
    if (awaitingReviewItems.length > 0 && liveSteps.length === 0) {
      return {
        isActive: true,
        phase: 'council_review',
        phaseBadge: `Review · ${awaitingReviewItems.length}`,
        stageTitle: 'Council Proof Inspection',
        activeAgentLabel: 'Council Review Panel',
        headline: `Reviewing proof artifacts for ${awaitingReviewItems.length} submitted task${awaitingReviewItems.length === 1 ? '' : 's'} (${approvedBoard}/${totalBoard} approved)...`,
        progressValue: approvedBoard,
        progressMax: totalBoard > 0 ? totalBoard : undefined,
        suggestedWorkspace: { id: 'board', label: 'View Board' },
        latestAiCall,
        lastActivityUtc,
      };
    }
    const primaryStep = liveSteps[0];
    const leadItem = workingBoardItems[0];
    const leadRole =
      primaryStep?.specialistAgent ||
      primaryStep?.agentName ||
      leadItem?.yielded_turn_info?.specialist_agent ||
      leadItem?.raised_by_role;
    const agentLabel = leadRole ? getRoleMetadata(leadRole).label : 'Task Pod';
    const headline = primaryStep
      ? `${leadItem ? `[${leadItem.title}] ` : ''}${primaryStep.stepSummary}`
      : `Executing ${workingBoardItems.length} active board task${workingBoardItems.length === 1 ? '' : 's'}: ${workingBoardItems.map((i) => i.title).slice(0, 2).join(', ')}`;
    return {
      isActive: true,
      phase: 'board_synthesis',
      phaseBadge: `${workers}w · ${approvedBoard}/${totalBoard}`,
      stageTitle: 'Production Board Synthesis',
      activeAgentLabel: agentLabel,
      headline,
      progressValue: approvedBoard,
      progressMax: totalBoard > 0 ? totalBoard : undefined,
      suggestedWorkspace: { id: 'board', label: 'View Board' },
      latestAiCall,
      lastActivityUtc,
    };
  }

  if (allBoardApproved) {
    return {
      isActive: true,
      phase: 'qa_verification',
      phaseBadge: 'Verifying Slice',
      stageTitle: 'Post-Board Slice Verification & Council Gate',
      activeAgentLabel: latestAiCall?.agentLabel || 'Studio Orchestrator',
      headline: `All ${approvedBoard}/${totalBoard} board tasks are approved — running slice verification, QA gates, or Council sign-off...`,
      whyPostBoardActive: `All ${approvedBoard}/${totalBoard} board tasks are approved — active AI calls belong to post-board verification.`,
      latestAiCall,
      lastActivityUtc,
    };
  }

  return {
    isActive: true,
    phase: 'writers_room',
    phaseBadge: proc?.scheduler_running_workers ? `${proc.scheduler_running_workers}w` : 'Active',
    stageTitle: 'Studio Orchestration Active',
    activeAgentLabel: latestAiCall?.agentLabel || 'Studio Council',
    headline: latestAiCall
      ? `${latestAiCall.agentLabel} executing ${latestAiCall.stepLabel}...`
      : 'Convening Studio Council and scheduling next slice work...',
    latestAiCall,
    lastActivityUtc,
  };
}

/**
 * Formats any raw `/ws/swarm` or `/api/events/history` event into a scannable row summary for `LivePulseModal`.
 */
export function summarizeSwarmEvent(ev: SwarmStreamEvent): FormattedSwarmEventSummary {
  const lower = String(ev.event_type || '').toLowerCase();
  const pl = asRecord(ev.payload);

  if (lower === 'studio_live_activity' && pl.phase === 'llm_gateway_call') {
    const stats = asRecord(pl.token_stats);
    const totalTok = asNumber(stats.total_tokens) ?? 0;
    const thoughtTok = asNumber(stats.thought_tokens) ?? 0;
    const callNum = asNumber(pl.total_calls);
    const parsed = parseAiCallLabel(asString(pl.label), ev.agent_name || 'StudioAgent');
    return {
      category: 'ai',
      agentLabel: parsed.agentLabel,
      title: `AI Call${callNum ? ` #${callNum}` : ''}: ${parsed.stepLabel}`,
      subtitle:
        thoughtTok > 0
          ? `${formatTokenCount(totalTok)} total (${formatTokenCount(thoughtTok)} thinking)`
          : `${formatTokenCount(totalTok)} total`,
      badgeText: formatTokenCount(totalTok),
      badgeTone: 'neutral',
    };
  }

  if (lower === 'slice_qa_gate_check_completed') {
    const passed = pl.passed === true;
    const gate = asString(pl.failure_gate) || '5-Gate QA';
    const spawned = Array.isArray(pl.spawned_item_ids)
      ? pl.spawned_item_ids.map((id) => String(id)).filter(Boolean)
      : [];
    return {
      category: 'stage',
      agentLabel: 'QA Build Supervisor',
      title: passed
        ? 'Slice passed all 5 Quality Gates'
        : `${gate} spawned board task${spawned.length ? `: ${spawned.join(', ')}` : ''}`,
      subtitle: passed ? asString(ev.deliberation) : asString(pl.failure_reason) || asString(ev.deliberation),
      badgeText: passed ? 'QA PASSED' : 'TASK SPAWNED',
      badgeTone: passed ? 'ok' : 'warn',
    };
  }

  if (lower === 'qa_gate_result') {
    const promoted = pl.promoted === true;
    const gate = asString(pl.failure_gate) || '5-Gate QA';
    const att = asNumber(pl.attempt);
    const maxAtt = asNumber(pl.max_retries);
    return {
      category: 'stage',
      agentLabel: 'QA Build Supervisor',
      title: promoted
        ? 'Passed all Quality Gates — Candidate Promoted'
        : `${gate} rejected${att ? ` (Attempt ${att}/${maxAtt ?? 3})` : ''}`,
      subtitle: asString(pl.failure_reason) || asString(ev.deliberation),
      badgeText: promoted ? 'PASSED' : 'GATE RETRY',
      badgeTone: promoted ? 'ok' : 'warn',
    };
  }

  if (lower === 'agent_stream' && String(pl.phase || '').toUpperCase() === 'REPAIR') {
    const gate = asString(pl.failure_gate) || 'Quality Gate';
    return {
      category: 'stage',
      agentLabel: getRoleMetadata(ev.agent_name).label,
      title: `Self-Healing Repair started for ${gate}`,
      subtitle: asString(pl.remedy_directive) || asString(ev.deliberation),
      badgeText: 'REPAIR',
      badgeTone: 'warn',
    };
  }

  if (lower === 'writers_room_activity') {
    const roleMeta = getRoleMetadata(asString(pl.agent_role) || ev.agent_name);
    return {
      category: 'stage',
      agentLabel: roleMeta.label,
      title: asString(pl.status_text) || asString(ev.deliberation),
      badgeText: 'LIVE STAGE',
      badgeTone: 'accent',
    };
  }

  if (lower.startsWith('backlog_item_')) {
    const itemId = asString(pl.item_id);
    const action = lower.replace('backlog_item_', '').replace(/_/g, ' ');
    return {
      category: 'board',
      agentLabel: getRoleMetadata(ev.agent_name).label,
      title: `${itemId ? `[${itemId}] ` : ''}${action}`,
      subtitle: asString(pl.message) || asString(pl.summary) || asString(pl.brief) || undefined,
      badgeText: action.toUpperCase(),
      badgeTone:
        lower === 'backlog_item_approved'
          ? 'ok'
          : lower === 'backlog_item_failed' || lower === 'backlog_item_changes_requested'
            ? 'warn'
            : 'accent',
    };
  }

  return {
    category: lower.includes('playtest') || lower.includes('slice') || lower === 'state_change' ? 'stage' : 'board',
    agentLabel: getRoleMetadata(ev.agent_name).label,
    title: asString(pl.status_text) || asString(ev.deliberation) || ev.event_type,
    badgeText: ev.event_type,
    badgeTone: lower === 'error' ? 'danger' : lower.includes('promoted') || lower.includes('verified') ? 'ok' : 'accent',
  };
}
