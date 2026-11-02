import { useEffect, useState } from 'react';
import { ArrowRight, Cpu, Radio, ShieldCheck, Wrench } from 'lucide-react';
import type { WorkspaceId } from '../../api/types';
import {
  formatRelativeSeconds,
  formatTokenCount,
  formatUsdCost,
  type LiveStudioActivity,
} from '../../domain/liveActivity';

export interface LiveActivityBannerProps {
  activity: LiveStudioActivity;
  activeWorkspace: WorkspaceId;
  onSelectWorkspace: (ws: WorkspaceId) => void;
  onOpenLivePulse: () => void;
}

export function LiveActivityBanner({
  activity,
  activeWorkspace,
  onSelectWorkspace,
  onOpenLivePulse,
}: LiveActivityBannerProps) {
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!activity.isActive) return;
    setNowMs(Date.now());
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [activity.isActive, activity.lastActivityUtc, activity.latestAiCall?.timestampUtc]);

  if (!activity.isActive) return null;

  const hasDeterminateProgress =
    typeof activity.progressValue === 'number' &&
    typeof activity.progressMax === 'number' &&
    activity.progressMax > 0;

  const aiCall = activity.latestAiCall;
  const aiAge = aiCall ? formatRelativeSeconds(aiCall.timestampUtc, nowMs) : '';
  const eventAge = formatRelativeSeconds(activity.lastActivityUtc, nowMs);

  const stageBadgeCls =
    activity.phase === 'qa_repair' || activity.phase === 'pausing'
      ? 'badge badge-warn'
      : activity.phase === 'promoting' || activity.phase === 'awaiting_greenlight'
        ? 'badge badge-ok'
        : 'badge badge-accent';

  const StageIcon =
    activity.phase === 'qa_repair'
      ? Wrench
      : activity.phase === 'qa_verification' || activity.phase === 'promoting'
        ? ShieldCheck
        : Radio;

  const secondaryNote = activity.whyPostBoardActive || activity.detail;
  const fullHoverTooltip = secondaryNote
    ? `${activity.headline}\n${secondaryNote}`
    : activity.headline;

  return (
    <section
      className="live-activity-banner"
      data-phase={activity.phase}
      aria-label="Live Studio Pipeline Activity"
    >
      <div className="live-activity-banner-row">
        <div className="live-activity-primary">
          <span className="live-pulse-dot" aria-hidden="true" />
          <span className={stageBadgeCls} style={{ flexShrink: 0 }}>
            <StageIcon size={11} aria-hidden="true" />
            {activity.stageTitle}
          </span>
          <span className="badge" style={{ textTransform: 'none', fontWeight: 600, flexShrink: 0 }}>
            {activity.activeAgentLabel}
          </span>

          {/* Polite live region restricted strictly to concise 1-line stage/headline transitions */}
          <div
            role="status"
            aria-live="polite"
            className="live-activity-status"
            title={fullHoverTooltip}
          >
            <span className="visually-hidden">
              {activity.stageTitle} ({activity.activeAgentLabel}):{' '}
            </span>
            <span className="live-activity-headline">{activity.headline}</span>
          </div>
        </div>

        <div className="live-activity-telemetry">
          {aiCall ? (
            <span
              className="live-ai-heartbeat-pill"
              title={[
                `Latest Gemini API call (#${aiCall.callNumber ?? '?'}) by ${aiCall.agentLabel}`,
                `${aiCall.totalTokens.toLocaleString()} total tokens (${aiCall.thoughtTokens.toLocaleString()} thinking${
                  aiCall.cachedTokens ? `, ${aiCall.cachedTokens.toLocaleString()} cached` : ''
                })`,
                aiCall.imageCount ? `${aiCall.imageCount} image(s) synthesized` : '',
                aiCall.estimatedCallCostUsd !== undefined
                  ? `Call est: ${formatUsdCost(aiCall.estimatedCallCostUsd)}${
                      aiCall.sessionEstimatedCostUsd !== undefined
                        ? ` · Session est: ${formatUsdCost(aiCall.sessionEstimatedCostUsd)}`
                        : ''
                    }${
                      aiCall.sessionCacheHitPct !== undefined && aiCall.sessionCacheHitPct > 0
                        ? ` (${Math.round(aiCall.sessionCacheHitPct)}% input cached)`
                        : ''
                    }`
                  : '',
              ]
                .filter(Boolean)
                .join(' · ')}
            >
              <Cpu size={12} style={{ color: 'var(--accent)', flexShrink: 0 }} aria-hidden="true" />
              <span>
                {aiCall.callNumber ? `AI #${aiCall.callNumber}` : 'AI Call'} · {aiCall.stepLabel} ·{' '}
                {aiCall.totalTokens > 0
                  ? formatTokenCount(aiCall.totalTokens)
                  : aiCall.imageCount && aiCall.imageCount > 0
                    ? `${aiCall.imageCount} img`
                    : formatTokenCount(aiCall.totalTokens)}
                {aiCall.sessionCacheHitPct !== undefined && aiCall.sessionCacheHitPct > 0
                  ? ` (${Math.round(aiCall.sessionCacheHitPct)}% cached)`
                  : ''}
                {aiCall.sessionEstimatedCostUsd !== undefined && aiCall.sessionEstimatedCostUsd > 0
                  ? ` · ${formatUsdCost(aiCall.sessionEstimatedCostUsd)}`
                  : ''}
                {aiAge ? ` · ${aiAge}` : ''}
              </span>
            </span>
          ) : eventAge ? (
            <span className="live-ai-heartbeat-pill">
              <Radio size={11} style={{ color: 'var(--ok)', flexShrink: 0 }} aria-hidden="true" />
              <span>Last event {eventAge}</span>
            </span>
          ) : null}

          {activity.suggestedWorkspace && activity.suggestedWorkspace.id !== activeWorkspace ? (
            <button
              type="button"
              className="btn"
              onClick={() => onSelectWorkspace(activity.suggestedWorkspace!.id)}
              style={{ fontSize: '0.78rem', paddingBlock: '3px', paddingInline: '8px' }}
            >
              <span>{activity.suggestedWorkspace.label}</span>
              <ArrowRight size={12} aria-hidden="true" />
            </button>
          ) : null}

          <button
            type="button"
            className="btn"
            onClick={onOpenLivePulse}
            style={{ fontSize: '0.78rem', paddingBlock: '3px', paddingInline: '8px' }}
          >
            <Radio size={12} style={{ color: 'var(--accent)' }} aria-hidden="true" />
            <span>Inspect Stream</span>
          </button>
        </div>
      </div>

      <progress
        className="studio-live-progress"
        aria-label={activity.stageTitle}
        value={hasDeterminateProgress ? activity.progressValue : undefined}
        max={hasDeterminateProgress ? activity.progressMax : undefined}
      />
    </section>
  );
}
