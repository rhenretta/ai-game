import { useState } from 'react';
import {
  Check,
  Compass,
  ExternalLink,
  Eye,
  Film,
  GitBranch,
  Image as ImageIcon,
  Pin,
  Play,
  Sparkles,
  Terminal,
  Users,
  Volume2,
  X,
  ZoomIn,
} from 'lucide-react';
import { resolveMediaUrl } from '../../api/client';
import type {
  CoProducerActionProposal,
  CoProducerConsultationCard,
  CoProducerMediaCard,
  CoProducerMediaItem,
  CoProducerPromptTraceEntry,
  CoProducerToolTraceEntry,
  WorkspaceId,
} from '../../api/types';
import { CodeDiffView } from '../../components/CodeDiffView';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';
import type { LightboxMedia } from '../../components/MediaLightbox';
import { PromptInspector } from '../../components/PromptInspector';
import { FormattedRichBlocks } from '../../components/RichTextBlocks';
import { getRoleMetadata } from '../../domain/roles';

export function normalizeTargetWorkspace(raw?: string | null): WorkspaceId | null {
  const cleaned = (raw ?? '').trim().toLowerCase().replace(/_/g, '-');
  switch (cleaned) {
    case 'coproducer':
    case 'co-producer':
      return 'coproducer';
    case 'board':
    case 'production-board':
      return 'board';
    case 'writers-room':
    case 'writersroom':
    case 'council':
      return 'writers-room';
    case 'bible':
    case 'game-bible':
      return 'bible';
    case 'memory':
    case 'memories':
    case 'game-memory':
      return 'memory';
    case 'assets':
    case 'assets-and-code':
    case 'code':
      return 'assets';
    case 'playtest':
    case 'playtester':
    case 'playtest-lab':
      return 'playtest';
    case 'dna':
    case 'player-dna':
    case 'steam-dna':
      return 'dna';
    default:
      return null;
  }
}

function mediaCardBadgeLabel(kind: CoProducerMediaCard['card_kind']): {
  label: string;
  cls: string;
} {
  switch (kind) {
    case 'sketch_showcase':
      return { label: 'Concept Sketch', cls: 'badge badge-purple' };
    case 'playtest_video':
      return { label: 'Playtest Broadcast', cls: 'badge badge-accent' };
    case 'qa_clip':
      return { label: 'QA Proof Reel', cls: 'badge badge-ok' };
    case 'progress_summary':
      return { label: 'Studio Progress', cls: 'badge badge-accent' };
    case 'audio_clips':
      return { label: 'Audio Showcase', cls: 'badge badge-purple' };
    case 'code_diff':
      return { label: 'Code Diff', cls: 'badge' };
    case 'image_gallery':
    default:
      return { label: 'Studio Assets', cls: 'badge badge-accent' };
  }
}

function isSpriteLikeImage(item: CoProducerMediaItem, cardKind: string): boolean {
  const pathLower = (item.artifact_path || '').toLowerCase();
  const labelLower = (item.label || '').toLowerCase();
  if (pathLower.includes('sprite') || labelLower.includes('sprite')) return true;
  if (
    cardKind === 'sketch_showcase' &&
    item.width &&
    item.height &&
    Math.abs(item.width - item.height) <= 16
  ) {
    return true;
  }
  return pathLower.includes('/assets/sprites/');
}

export interface CoProducerMediaCardViewProps {
  card: CoProducerMediaCard;
  cartridgeId?: string;
  isLatestMessage: boolean;
  selectedSketchPaths: string[];
  onToggleSketchPin: (relPath: string) => void;
  onOpenLightbox: (media: LightboxMedia) => void;
  onSelectWorkspace: (ws: WorkspaceId) => void;
}

export function CoProducerMediaCardView({
  card,
  cartridgeId,
  isLatestMessage,
  selectedSketchPaths,
  onToggleSketchPin,
  onOpenLightbox,
  onSelectWorkspace,
}: CoProducerMediaCardViewProps) {
  const badge = mediaCardBadgeLabel(card.card_kind);
  // Progressive HTML media lazy-loading (modern-web-guidance: lazy-load-video-and-audio)
  // Applied to offscreen historical turns; omitted on the active latest turn's media players.
  const lazyMediaAttrs: Record<string, unknown> = isLatestMessage ? {} : { loading: 'lazy' };
  const deepLinkWs = normalizeTargetWorkspace(card.deep_link_workspace);

  return (
    <section className="coproducer-media-card" aria-label={card.title}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span className={badge.cls}>{badge.label}</span>
          <strong style={{ fontSize: '0.94rem' }}>{card.title}</strong>
        </div>

        {deepLinkWs ? (
          <button
            type="button"
            className="btn"
            onClick={() => onSelectWorkspace(deepLinkWs)}
            title={
              card.deep_link_target_id
                ? `Inspect ${card.deep_link_target_id} in ${deepLinkWs} tab`
                : `Open ${deepLinkWs} tab`
            }
          >
            <ExternalLink size={13} />
            Open {deepLinkWs}
            {card.deep_link_target_id ? ` (${card.deep_link_target_id})` : ''}
          </button>
        ) : null}
      </div>

      {card.subtitle ? (
        <div style={{ fontSize: '0.86rem', color: 'var(--text-secondary)' }}>
          {card.subtitle}
        </div>
      ) : null}

      {Array.isArray(card.items) && card.items.length > 0 ? (
        <div className="coproducer-media-grid">
          {card.items.map((item, idx) => {
            if (item.media_kind === 'metric') {
              return (
                <div
                  key={`${card.card_id || 'mc'}-metric-${item.item_id || idx}`}
                  className="coproducer-media-tile"
                >
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                    {item.label}
                  </div>
                  <div
                    style={{
                      fontSize: '1.15rem',
                      fontWeight: 700,
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--accent)',
                    }}
                  >
                    {item.metric_value}
                  </div>
                  {item.caption ? (
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      {item.caption}
                    </div>
                  ) : null}
                </div>
              );
            }

            if (item.media_kind === 'code_diff') {
              return (
                <div
                  key={`${card.card_id || 'mc'}-diff-${item.item_id || idx}`}
                  style={{ gridColumn: '1 / -1' }}
                  className="coproducer-media-tile"
                >
                  <CodeDiffView
                    title={item.label || item.artifact_path}
                    diffText={item.diff_unified || ''}
                  />
                  {item.caption ? (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      {item.caption}
                    </div>
                  ) : null}
                </div>
              );
            }

            const rawPath = item.artifact_path || '';
            const resolvedUrl = resolveMediaUrl(rawPath, cartridgeId);

            if (item.media_kind === 'video') {
              const videoW =
                item.width && item.width > 0
                  ? item.width
                  : card.card_kind === 'playtest_video'
                    ? 1600
                    : 1280;
              const videoH = item.height && item.height > 0 ? item.height : 720;
              return (
                <div
                  key={`${card.card_id || 'mc'}-vid-${item.item_id || idx}`}
                  className="coproducer-media-tile"
                  style={card.items.length === 1 ? { gridColumn: '1 / -1' } : undefined}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '0.84rem',
                      fontWeight: 600,
                    }}
                  >
                    <Film size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                    <span>{item.label}</span>
                  </div>
                  <video
                    controls
                    preload="metadata"
                    playsInline
                    width={videoW}
                    height={videoH}
                    src={resolvedUrl}
                    {...lazyMediaAttrs}
                  />
                  {item.caption ? (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {item.caption}
                    </div>
                  ) : null}
                </div>
              );
            }

            if (item.media_kind === 'audio') {
              return (
                <div
                  key={`${card.card_id || 'mc'}-aud-${item.item_id || idx}`}
                  className="coproducer-media-tile"
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '0.84rem',
                      fontWeight: 600,
                    }}
                  >
                    <Volume2 size={14} style={{ color: 'var(--purple)', flexShrink: 0 }} />
                    <span>{item.label}</span>
                  </div>
                  <audio
                    controls
                    preload="metadata"
                    src={resolvedUrl}
                    style={{ inlineSize: '100%' }}
                    {...lazyMediaAttrs}
                  />
                  {item.caption ? (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {item.caption}
                    </div>
                  ) : null}
                </div>
              );
            }

            // Image / Sprite / Filmstrip
            const spriteMode = isSpriteLikeImage(item, card.card_kind);
            const imgW = item.width && item.width > 0 ? item.width : spriteMode ? 512 : 1280;
            const imgH = item.height && item.height > 0 ? item.height : spriteMode ? 512 : 720;
            const isPinned = Boolean(rawPath && selectedSketchPaths.includes(rawPath));
            const canPin =
              Boolean(rawPath) &&
              (card.card_kind === 'sketch_showcase' ||
                rawPath.includes('sketches/') ||
                rawPath.includes('_coproducer_drafts/'));

            return (
              <div
                key={`${card.card_id || 'mc'}-img-${item.item_id || idx}`}
                className="coproducer-media-tile"
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '6px',
                  }}
                >
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '5px',
                      fontSize: '0.84rem',
                      fontWeight: 600,
                    }}
                  >
                    <ImageIcon size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                    {item.label}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {canPin ? (
                      <button
                        type="button"
                        className={isPinned ? 'btn btn-primary' : 'btn'}
                        onClick={() => onToggleSketchPin(rawPath)}
                        aria-pressed={isPinned}
                        title="Pin this sketch as a visual reference for follow-up revisions"
                      >
                        <Pin size={12} />
                        {isPinned ? 'Pinned Ref' : 'Pin Ref'}
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="btn"
                      onClick={() =>
                        onOpenLightbox({
                          url: resolvedUrl,
                          title: item.label || card.title,
                          subtitle: item.caption || card.subtitle,
                          deliberation: item.deliberation || card.deliberation,
                        })
                      }
                      title="Inspect full-resolution image"
                    >
                      <ZoomIn size={12} />
                      Zoom
                    </button>
                  </div>
                </div>

                <div
                  className={spriteMode ? 'coproducer-sprite-plate' : undefined}
                  style={{ cursor: 'pointer' }}
                  onClick={() =>
                    onOpenLightbox({
                      url: resolvedUrl,
                      title: item.label || card.title,
                      subtitle: item.caption || card.subtitle,
                      deliberation: item.deliberation || card.deliberation,
                    })
                  }
                >
                  <img
                    src={resolvedUrl}
                    alt={item.label || card.title}
                    width={imgW}
                    height={imgH}
                    loading={isLatestMessage ? undefined : 'lazy'}
                    style={
                      spriteMode
                        ? {
                            maxBlockSize: '220px',
                            inlineSize: 'auto',
                            objectFit: 'contain',
                            imageRendering: 'pixelated',
                          }
                        : {
                            inlineSize: '100%',
                            blockSize: 'auto',
                            maxBlockSize: '340px',
                            objectFit: 'cover',
                            borderRadius: 'var(--radius-sm)',
                          }
                    }
                  />
                </div>

                {item.caption ? (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    {item.caption}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}

export interface CoProducerConsultationCardViewProps {
  card: CoProducerConsultationCard;
  defaultOpen?: boolean;
  onPrefillPrompt?: (text: string) => void;
}

export function CoProducerConsultationCardView({
  card,
  defaultOpen = false,
  onPrefillPrompt,
}: CoProducerConsultationCardViewProps) {
  const roles = Array.isArray(card.participating_roles) ? card.participating_roles : [];
  const kindMeta =
    card.consultation_kind === 'divergent_pitch_round'
      ? { label: 'Divergent Pitch Jam', cls: 'badge badge-purple', icon: Sparkles }
      : card.consultation_kind === 'specialist_probe'
        ? {
            label: `Specialist Probe · ${roles[0] || 'engineer'}`,
            cls: 'badge badge-accent',
            icon: Compass,
          }
        : card.consultation_kind === 'genre_foundation'
          ? {
              label: 'Genre Craft & Fun Foundation',
              cls: 'badge badge-ok',
              icon: Sparkles,
            }
          : { label: 'Council Huddle', cls: 'badge badge-purple', icon: Users };

  const Icon = kindMeta.icon;
  const huddleTurns = Array.isArray(card.specialist_turns) ? card.specialist_turns : [];
  const pitches = Array.isArray(card.divergent_pitches) ? card.divergent_pitches : [];
  const proofPaths = Array.isArray(card.proof_media_paths) ? card.proof_media_paths : [];

  return (
    <details className="coproducer-consultation-card" open={defaultOpen}>
      <summary>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span className={kindMeta.cls}>
            <Icon size={11} />
            {kindMeta.label}
          </span>
          <strong style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>
            {card.synthesis_headline}
          </strong>
        </span>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          Consultation Notes
        </span>
      </summary>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBlockStart: '10px' }}>
        <DeliberationDisclosure
          deliberation={card.deliberation}
          label="Backchannel Consultation Rationale"
        />

        {card.question_posed ? (
          <div style={{ fontSize: '0.83rem', color: 'var(--text-secondary)' }}>
            <strong>Question Dispatched: </strong>
            <span>{card.question_posed}</span>
          </div>
        ) : null}

        {huddleTurns.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {huddleTurns.map((turn, idx) => {
              const roleMeta = getRoleMetadata(turn.speaker_role);
              return (
                <div
                  key={`${card.card_id}-turn-${idx}`}
                  style={{
                    padding: '9px 11px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-surface)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '6px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                    <span className="badge badge-accent">{roleMeta.label}</span>
                    {turn.key_recommendation ? (
                      <strong style={{ fontSize: '0.82rem' }}>{turn.key_recommendation}</strong>
                    ) : null}
                  </div>
                  <DeliberationDisclosure
                    deliberation={turn.deliberation}
                    label={`${roleMeta.label} Rationale`}
                  />
                  <div style={{ fontSize: '0.86rem' }}>
                    <FormattedRichBlocks text={turn.spoken_quote} />
                  </div>
                  {turn.creative_angle_or_risk ? (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      <strong>Angle / Risk: </strong>
                      {turn.creative_angle_or_risk}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : null}

        {pitches.length > 0 ? (
          <div className="coproducer-pitch-grid">
            {pitches.map((pitch) => (
              <div
                key={pitch.pitch_id}
                style={{
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-surface)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '6px',
                  }}
                >
                  <strong style={{ fontSize: '0.9rem', color: 'var(--accent)' }}>
                    {pitch.working_title}
                  </strong>
                  <span className="badge badge-purple">{pitch.pitch_id}</span>
                </div>
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  {pitch.core_genre}
                </div>
                <DeliberationDisclosure
                  deliberation={pitch.deliberation}
                  label="Pitch Rationale"
                />
                <div style={{ fontSize: '0.84rem' }}>
                  <strong>Hook: </strong>
                  {pitch.hook_summary}
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  <strong>Core Loop: </strong>
                  {pitch.core_loop_sketch}
                </div>
                {pitch.visual_and_audio_vibe ? (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {pitch.visual_and_audio_vibe}
                  </div>
                ) : null}
                {onPrefillPrompt ? (
                  <button
                    type="button"
                    className="btn"
                    style={{ marginBlockStart: 'auto', alignSelf: 'flex-start' }}
                    onClick={() =>
                      onPrefillPrompt(
                        `Let's explore "${pitch.working_title}" (${pitch.core_genre}) — refine the core loop and sketch a 16:9 mockup for it.`,
                      )
                    }
                  >
                    <Sparkles size={12} />
                    Explore This Direction
                  </button>
                ) : null}
              </div>
            ))}
          </div>
        ) : null}

        {card.probe_findings_markdown ? (
          <div
            style={{
              padding: '9px 11px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-subtle)',
              background: 'var(--bg-surface)',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
            }}
          >
            {proofPaths.length > 0 ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {proofPaths.map((p) => (
                  <code key={p} className="studio-inline-code" style={{ fontSize: '0.76rem' }}>
                    {p}
                  </code>
                ))}
              </div>
            ) : null}
            <div style={{ fontSize: '0.86rem' }}>
              <FormattedRichBlocks text={card.probe_findings_markdown} />
            </div>
          </div>
        ) : null}
      </div>
    </details>
  );
}

function actionScopeMeta(proposal: CoProducerActionProposal): {
  badge: string;
  cls: string;
} {
  switch (proposal.action_kind) {
    case 'commission_game':
      return proposal.handoff_mode === 'refined_concept'
        ? {
            badge: 'Commission Game · Refined Concept (Skips Blind Genesis)',
            cls: 'badge badge-ok',
          }
        : {
            badge: 'Commission Game · Loose Spark (Writers’ Room Jam)',
            cls: 'badge badge-purple',
          };
    case 'greenlight_bible':
      return { badge: 'Greenlight Game Bible', cls: 'badge badge-ok' };
    case 'trigger_next_slice':
      return { badge: 'Commission Next Vertical Slice', cls: 'badge badge-accent' };
    case 'pivot_course_correction':
      return { badge: 'Tier 1 · Course Correction', cls: 'badge badge-accent' };
    case 'pivot_structural':
      return { badge: 'Tier 2 · Structural Pivot & Replan', cls: 'badge badge-warn' };
    case 'pivot_fork_cartridge':
      return { badge: 'Tier 3 · Fork New Cartridge', cls: 'badge badge-purple' };
    case 'start_blind_playtest':
      return { badge: 'Autonomous Playtest Session', cls: 'badge badge-accent' };
    case 'launch_game':
      return { badge: 'Launch Native Godot Window', cls: 'badge badge-accent' };
    case 'pause_build':
      return { badge: 'Pause Background Workers', cls: 'badge badge-warn' };
    case 'resume_build':
      return { badge: 'Resume Studio Workers', cls: 'badge badge-ok' };
    case 'record_game_memory':
    default:
      return { badge: 'Durable Studio Memory', cls: 'badge badge-purple' };
  }
}

export interface CoProducerActionProposalCardProps {
  messageId: string;
  proposal: CoProducerActionProposal;
  busy: boolean;
  onExecute: (params: {
    messageId: string;
    proposal: CoProducerActionProposal;
    dismiss?: boolean;
    overrideBrief?: string;
    overrideHandoffMode?: 'loose_spark' | 'refined_concept';
    overridePauseForGreenlight?: boolean;
  }) => Promise<void>;
}

export function CoProducerActionProposalCard({
  messageId,
  proposal,
  busy,
  onExecute,
}: CoProducerActionProposalCardProps) {
  const meta = actionScopeMeta(proposal);
  const [editableBriefText, setEditableBriefText] = useState<string>(
    proposal.steering_or_seed_brief ||
      proposal.concept_brief?.elevator_pitch ||
      proposal.summary ||
      '',
  );
  const [handoffMode, setHandoffMode] = useState<'loose_spark' | 'refined_concept'>(
    proposal.handoff_mode === 'loose_spark' ? 'loose_spark' : 'refined_concept',
  );
  const [pauseForGreenlight, setPauseForGreenlight] = useState<boolean>(
    Boolean(proposal.pause_for_greenlight),
  );

  const cancelIds = Array.isArray(proposal.cancel_item_ids) ? proposal.cancel_item_ids : [];
  const carryoverPaths = Array.isArray(proposal.carry_over_asset_paths)
    ? proposal.carry_over_asset_paths
    : [];
  const memoryMuts = Array.isArray(proposal.memory_mutations) ? proposal.memory_mutations : [];
  const isCommissionOrFork =
    proposal.action_kind === 'commission_game' || proposal.action_kind === 'pivot_fork_cartridge';

  return (
    <div className="coproducer-action-card" data-state={proposal.status}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span className={meta.cls}>{meta.badge}</span>
          <strong style={{ fontSize: '0.92rem' }}>{proposal.title}</strong>
        </div>

        {proposal.status === 'executed' ? (
          <span className="badge badge-ok">
            <Check size={11} /> Executed
          </span>
        ) : proposal.status === 'dismissed' ? (
          <span className="badge">Dismissed</span>
        ) : (
          <span className="badge badge-accent">Awaiting Confirmation</span>
        )}
      </div>

      <div style={{ fontSize: '0.87rem', color: 'var(--text-secondary)' }}>{proposal.summary}</div>

      {cancelIds.length > 0 ? (
        <div style={{ fontSize: '0.8rem', color: 'var(--warn)' }}>
          <strong>Board Items to Cancel & Rewire: </strong>
          {cancelIds.map((id) => (
            <code key={id} className="studio-inline-code" style={{ marginInlineEnd: '5px' }}>
              {id}
            </code>
          ))}
        </div>
      ) : null}

      {memoryMuts.length > 0 ? (
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          <strong>Memory Bank Updates ({memoryMuts.length}): </strong>
          {memoryMuts.map((m, idx) => (
            <span key={idx} style={{ marginInlineEnd: '8px' }}>
              [{m.operation}] {m.title || m.target_memory_id}
            </span>
          ))}
        </div>
      ) : null}

      {carryoverPaths.length > 0 ? (
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          <GitBranch
            size={12}
            style={{ display: 'inline', verticalAlign: 'middle', marginInlineEnd: '4px' }}
          />
          <strong>Carryover Assets ({carryoverPaths.length}): </strong>
          {carryoverPaths.map((p) => (
            <code key={p} className="studio-inline-code" style={{ marginInlineEnd: '5px' }}>
              {p}
            </code>
          ))}
        </div>
      ) : null}

      {proposal.status === 'pending' ? (
        <details
          style={{
            padding: '8px 10px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)',
            background: 'var(--bg-canvas)',
          }}
        >
          <summary
            style={{
              cursor: 'pointer',
              fontSize: '0.8rem',
              fontWeight: 600,
              color: 'var(--text-secondary)',
            }}
          >
            Customize Handoff Options
          </summary>

          <div style={{ display: 'grid', gap: '8px', marginBlockStart: '8px' }}>
            {isCommissionOrFork ? (
              <label style={{ display: 'grid', gap: '3px', fontSize: '0.8rem' }}>
                <span style={{ fontWeight: 600 }}>Council Handoff Mode</span>
                <select
                  className="select-control"
                  value={handoffMode}
                  onChange={(e) =>
                    setHandoffMode(
                      e.target.value === 'loose_spark' ? 'loose_spark' : 'refined_concept',
                    )
                  }
                >
                  <option value="refined_concept">
                    Refined Concept (Lock co-designed pillars & skip blind genesis)
                  </option>
                  <option value="loose_spark">
                    Loose Spark (Let Specialist Council run divergent genesis jam)
                  </option>
                </select>
              </label>
            ) : null}

            <label style={{ display: 'grid', gap: '3px', fontSize: '0.8rem' }}>
              <span style={{ fontWeight: 600 }}>Seed Brief / Council Directive</span>
              <textarea
                rows={3}
                className="textarea-control"
                value={editableBriefText}
                onChange={(e) => setEditableBriefText(e.target.value)}
              />
            </label>

            {isCommissionOrFork ? (
              <label
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '7px',
                  fontSize: '0.82rem',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={pauseForGreenlight}
                  onChange={(e) => setPauseForGreenlight(e.target.checked)}
                />
                <span>Pause for manual Bible greenlight before Slice 1 tasks start</span>
              </label>
            ) : null}
          </div>
        </details>
      ) : null}

      {proposal.status === 'pending' ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() =>
              void onExecute({
                messageId,
                proposal,
                dismiss: false,
                overrideBrief: editableBriefText,
                overrideHandoffMode: isCommissionOrFork ? handoffMode : undefined,
                overridePauseForGreenlight: isCommissionOrFork ? pauseForGreenlight : undefined,
              })
            }
          >
            <Play size={13} />
            {busy ? 'Executing...' : proposal.title}
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={busy}
            onClick={() =>
              void onExecute({
                messageId,
                proposal,
                dismiss: true,
              })
            }
          >
            <X size={13} />
            Dismiss
          </button>
        </div>
      ) : proposal.execution_result ? (
        <div
          style={{
            fontSize: '0.8rem',
            fontFamily: 'var(--font-mono)',
            color: proposal.status === 'executed' ? 'var(--ok)' : 'var(--text-muted)',
          }}
        >
          {typeof proposal.execution_result.target_cartridge_id === 'string'
            ? `Cartridge: ${proposal.execution_result.target_cartridge_id}`
            : JSON.stringify(proposal.execution_result)}
        </div>
      ) : null}
    </div>
  );
}

export interface CoProducerTurnObservabilityProps {
  deliberation?: string;
  deliberationLabel: string;
  consultationCards: CoProducerConsultationCard[];
  backgroundMediaCards: CoProducerMediaCard[];
  primaryMediaCards: CoProducerMediaCard[];
  actionProposals: CoProducerActionProposal[];
  toolTrace: CoProducerToolTraceEntry[];
  promptTraces: CoProducerPromptTraceEntry[];
  tokenUsage?: Record<string, number>;
  cartridgeId?: string;
  selectedSketchPaths: string[];
  onToggleSketchPin: (relPath: string) => void;
  onOpenLightbox: (media: LightboxMedia) => void;
  onSelectWorkspace: (ws: WorkspaceId) => void;
  onPrefillPrompt?: (text: string) => void;
}

/**
 * Single consolidated progressive-disclosure drawer ("Behind the scenes") that keeps the
 * default Co-Producer reply crisp while preserving 100% Rule 4 epistemic & tool observability.
 */
export function CoProducerTurnObservability({
  deliberation,
  deliberationLabel,
  consultationCards,
  backgroundMediaCards,
  primaryMediaCards,
  actionProposals,
  toolTrace,
  promptTraces,
  tokenUsage,
  cartridgeId,
  selectedSketchPaths,
  onToggleSketchPin,
  onOpenLightbox,
  onSelectWorkspace,
  onPrefillPrompt,
}: CoProducerTurnObservabilityProps) {
  const safeTools = Array.isArray(toolTrace) ? toolTrace : [];
  const safePrompts = Array.isArray(promptTraces) ? promptTraces : [];
  const totalTokens = Number(tokenUsage?.total_tokens ?? 0);
  const trimmedDelib = (deliberation ?? '').trim();

  const mediaRationales = primaryMediaCards.filter((c) => Boolean((c.deliberation ?? '').trim()));
  const actionRationales = actionProposals.filter((p) => Boolean((p.deliberation ?? '').trim()));

  const hasAnyBackstage =
    Boolean(trimmedDelib) ||
    consultationCards.length > 0 ||
    backgroundMediaCards.length > 0 ||
    mediaRationales.length > 0 ||
    actionRationales.length > 0 ||
    safeTools.length > 0 ||
    safePrompts.length > 0 ||
    totalTokens > 0;

  if (!hasAnyBackstage) {
    return null;
  }

  const summaryParts: string[] = [];
  if (consultationCards.length > 0) {
    summaryParts.push(
      consultationCards.length === 1
        ? consultationCards[0].synthesis_headline || '1 consultation'
        : `${consultationCards.length} consultations`,
    );
  }
  if (backgroundMediaCards.length > 0) {
    summaryParts.push(
      `${backgroundMediaCards.length} inspected card${backgroundMediaCards.length === 1 ? '' : 's'}`,
    );
  }
  if (safeTools.length > 0) {
    summaryParts.push(`${safeTools.length} tool step${safeTools.length === 1 ? '' : 's'}`);
  }
  if (summaryParts.length === 0) {
    summaryParts.push('Reasoning & traces');
  }

  return (
    <details className="coproducer-behind-scenes">
      <summary>
        <span className="coproducer-behind-scenes-label">
          <Eye size={12} aria-hidden="true" />
          <span>Behind the scenes</span>
          <span className="coproducer-behind-scenes-meta">· {summaryParts.join(' · ')}</span>
        </span>
        {totalTokens > 0 ? (
          <span className="coproducer-behind-scenes-tokens">
            {totalTokens.toLocaleString()} tok
          </span>
        ) : null}
      </summary>

      <div className="coproducer-behind-scenes-body">
        <DeliberationDisclosure
          deliberation={trimmedDelib}
          label={deliberationLabel}
          defaultOpen
        />

        {consultationCards.map((cCard) => (
          <CoProducerConsultationCardView
            key={cCard.card_id}
            card={cCard}
            defaultOpen={consultationCards.length === 1}
            onPrefillPrompt={onPrefillPrompt}
          />
        ))}

        {backgroundMediaCards.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)' }}>
              Inspected Studio Evidence (Available on Demand)
            </div>
            {backgroundMediaCards.map((card) => (
              <CoProducerMediaCardView
                key={card.card_id || card.title}
                card={card}
                cartridgeId={cartridgeId}
                isLatestMessage={false}
                selectedSketchPaths={selectedSketchPaths}
                onToggleSketchPin={onToggleSketchPin}
                onOpenLightbox={onOpenLightbox}
                onSelectWorkspace={onSelectWorkspace}
              />
            ))}
          </div>
        ) : null}

        {mediaRationales.map((card, idx) => (
          <DeliberationDisclosure
            key={`media-rat-${card.card_id || idx}`}
            deliberation={card.deliberation}
            label={`Media Curation Rationale · ${card.title}`}
          />
        ))}

        {actionRationales.map((prop, idx) => (
          <DeliberationDisclosure
            key={`act-rat-${prop.proposal_id || idx}`}
            deliberation={prop.deliberation}
            label={`Action Proposal Rationale · ${prop.title}`}
          />
        ))}

        {safeTools.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                fontSize: '0.78rem',
                fontWeight: 700,
                color: 'var(--text-muted)',
              }}
            >
              <Terminal size={12} />
              Tool Trace ({safeTools.length})
            </div>
            {safeTools.map((step, idx) => (
              <div
                key={`step-${idx}-${step.tool_name}`}
                style={{
                  padding: '6px 8px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-surface)',
                  fontSize: '0.8rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                }}
              >
                <div>
                  <strong style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>
                    #{idx + 1} {step.tool_name}
                  </strong>
                  {step.query_or_question ? ` — ${step.query_or_question}` : ''}
                </div>
                <DeliberationDisclosure deliberation={step.deliberation} label="Tool Rationale" />
                {step.observation_excerpt ? (
                  <div
                    style={{
                      color: 'var(--text-secondary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.76rem',
                      whiteSpace: 'pre-wrap',
                    }}
                  >
                    {step.observation_excerpt}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        ) : null}

        {safePrompts.map((pt, idx) => (
          <PromptInspector
            key={`prompt-${idx}`}
            label={`${pt.step_label || 'prompt'}${pt.model ? ` · ${pt.model}` : ''}${
              pt.temperature !== undefined ? ` · temp=${pt.temperature}` : ''
            }`}
            rawContent={pt.prompt_excerpt}
          />
        ))}
      </div>
    </details>
  );
}
