import { useState } from 'react';
import {
  Activity,
  Check,
  CheckCircle2,
  Copy,
  FileCode,
  GitCommit,
  HelpCircle,
  Image as ImageIcon,
  Layers,
  AlertTriangle,
  Sparkles,
} from 'lucide-react';
import type {
  ChronologyCodeDiff,
  ChronologyContextPack,
  ChronologyFileEntry,
  ChronologySelfInquiry,
  ChronologyTurn,
} from '../../api/types';
import {
  coercePromptString,
  normalizeProofArtifacts,
  normalizeStepLog,
} from '../../domain/promptParser';
import { appendProofVersionParam } from '../../domain/boardCards';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';
import { PromptInspector } from '../../components/PromptInspector';
import { CodeDiffView } from '../../components/CodeDiffView';
import type { LightboxMedia } from '../../components/MediaLightbox';

export interface TurnDetailCardProps {
  turn: Partial<ChronologyTurn>;
  fallbackIndex: number;
  cartridgeId: string;
  itemTitle: string;
  isActiveTurn?: boolean;
  isLatestTurn?: boolean;
  deferOffscreen?: boolean;
  contextPacks: ChronologyContextPack[];
  fileContents: Record<string, ChronologyFileEntry | string>;
  onOpenLightbox: (media: LightboxMedia) => void;
}

export function formatSpecialistLabel(rawAgent: string | undefined): string {
  const s = (rawAgent || 'worker').trim();
  const lower = s.toLowerCase();
  if (lower === 'swe_engineer') return 'SWE Engineer';
  if (lower === 'qa_test_engineer') return 'QA Test Engineer';
  if (lower === 'visual_artist') return 'Visual Artist';
  if (lower === 'audio_engineer') return 'Audio Engineer';
  if (lower === 'task_lead') return 'Task Lead';
  if (lower === 'board_task_spawner') return 'Board Task Spawner';
  return s
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function extractDiffEntries(rawDiffs: unknown): Array<{ label: string; diff: string }> {
  if (!rawDiffs) return [];
  if (typeof rawDiffs === 'string' && rawDiffs.trim()) {
    return [{ label: 'Unified Diff', diff: rawDiffs }];
  }
  if (Array.isArray(rawDiffs)) {
    return rawDiffs
      .map((entry, idx) => {
        if (typeof entry === 'string') {
          return { label: `Diff #${idx + 1}`, diff: entry };
        }
        const obj = entry as ChronologyCodeDiff;
        return {
          label: obj.file || obj.path || obj.module || `Diff #${idx + 1}`,
          diff: obj.diff || '',
        };
      })
      .filter((d) => d.diff.trim().length > 0);
  }
  if (typeof rawDiffs === 'object') {
    return Object.entries(rawDiffs as Record<string, unknown>)
      .filter(([, val]) => typeof val === 'string' && val.trim().length > 0)
      .map(([file, val]) => ({ label: file, diff: String(val) }));
  }
  return [];
}

function SelfInquiryList({
  inquiries,
  label,
  defaultRole,
}: {
  inquiries: ChronologySelfInquiry[];
  label: string;
  defaultRole: string;
}) {
  if (inquiries.length === 0) return null;
  return (
    <details
      style={{
        borderRadius: 'var(--radius-sm)',
        border: '1px solid color-mix(in srgb, var(--warn) 40%, var(--border-subtle))',
        borderInlineStart: '3px solid var(--warn)',
        background: 'color-mix(in srgb, var(--warn) 6%, var(--bg-canvas))',
        padding: '6px 10px',
      }}
    >
      <summary
        style={{
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '8px',
          fontSize: '0.84rem',
          fontWeight: 700,
          color: 'var(--warn)',
          userSelect: 'none',
        }}
      >
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
          <HelpCircle size={12} /> {label}
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
          {inquiries.length} Q&amp;A{inquiries.length === 1 ? '' : 's'}
        </span>
      </summary>
      <div style={{ display: 'grid', gap: '7px', marginBlockStart: '8px' }}>
        {inquiries.map((qa, idx) => (
          <div
            key={idx}
            style={{
              padding: '7px 9px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              display: 'grid',
              gap: '5px',
              fontSize: '0.84rem',
            }}
          >
            <DeliberationDisclosure deliberation={qa.deliberation} label="Self-Inquiry Rationale" />
            <div style={{ fontWeight: 700, color: 'var(--warn)' }}>
              Q ({formatSpecialistLabel(qa.agent_role || defaultRole)}): {qa.question}
            </div>
            <div style={{ color: 'var(--text-primary)', whiteSpace: 'pre-wrap' }}>
              <strong style={{ color: 'var(--ok)' }}>Self-Answer:</strong> {qa.answer}
            </div>
            {qa.researched_chunk_ids && qa.researched_chunk_ids.length > 0 ? (
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                Researched chunks:{' '}
                <code style={{ color: 'var(--accent)' }}>{qa.researched_chunk_ids.join(', ')}</code>
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </details>
  );
}

export function TurnDetailCard({
  turn,
  fallbackIndex,
  cartridgeId,
  itemTitle,
  isActiveTurn = false,
  isLatestTurn = false,
  deferOffscreen = false,
  contextPacks,
  fileContents,
  onOpenLightbox,
}: TurnDetailCardProps) {
  const [copiedSummary, setCopiedSummary] = useState(false);
  const [brokenProofUrls, setBrokenProofUrls] = useState<Record<string, true>>({});

  const turnNumber = (typeof turn.turn_index === 'number' ? turn.turn_index : fallbackIndex) + 1;
  const specialistRaw = turn.specialist_agent || (isActiveTurn ? 'task_lead' : 'worker');
  const specialistLabel = formatSpecialistLabel(specialistRaw);
  const passed = turn.verification_passed !== false;
  const rawTs = turn.timestamp_utc || turn.started_at_utc || '';
  const proofVersionToken = rawTs || `t${turnNumber}`;

  const steps = normalizeStepLog(turn.step_log);
  const diffs = extractDiffEntries(turn.code_diffs);
  const diffFilesSet = new Set(diffs.map((d) => d.label));
  const touchedFiles = (turn.touched_files ?? []).filter(Boolean);
  const proofs = normalizeProofArtifacts(turn.proof_urls, turn.proof_images, cartridgeId).map(
    (p) => ({
      ...p,
      url: appendProofVersionParam(p.url, proofVersionToken),
    }),
  );
  const imageProofs = proofs.filter((p) => !p.isAudio);
  const audioProofs = proofs.filter((p) => p.isAudio);

  const leadInquiries = Array.isArray(turn.lead_self_inquiries) ? turn.lead_self_inquiries : [];
  const specInquiries = Array.isArray(turn.self_inquiries) ? turn.self_inquiries : [];
  const totalInquiries = leadInquiries.length + specInquiries.length;

  const leadPromptStr = coercePromptString(turn.lead_prompt_text);
  const leadContextStr = coercePromptString(turn.lead_curated_context);
  const specPromptStr = coercePromptString(turn.prompt_text);
  const specContextStr = coercePromptString(turn.curated_context);

  const matchingPack = contextPacks.find(
    (cp) => cp.target_agent_role?.toLowerCase() === specialistRaw.toLowerCase(),
  );

  const hasLeadSection = Boolean(
    turn.brief ||
      turn.lead_deliberation ||
      leadInquiries.length > 0 ||
      leadContextStr ||
      leadPromptStr,
  );

  const hasSpecialistDelib = Boolean(
    turn.deliberation && turn.deliberation !== turn.lead_deliberation,
  );

  const timeShort = rawTs.includes('T') ? rawTs.split('T')[1].replace('Z', '').slice(0, 8) : rawTs;

  const summaryText = (turn.summary ?? '').trim();

  const copySummaryText = async () => {
    if (!summaryText || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(summaryText);
      setCopiedSummary(true);
      window.setTimeout(() => setCopiedSummary(false), 1500);
    } catch {
      // Ignore clipboard denial
    }
  };

  return (
    <article
      className={deferOffscreen ? 'deferred-chronology-turn' : undefined}
      style={{
        padding: '11px 12px',
        borderRadius: 'var(--radius-md)',
        background: isActiveTurn
          ? 'color-mix(in srgb, var(--accent) 7%, var(--bg-canvas))'
          : 'var(--bg-canvas)',
        border: isActiveTurn
          ? '1.5px solid var(--accent)'
          : passed
            ? '1px solid var(--border-subtle)'
            : '1px solid color-mix(in srgb, var(--warn) 55%, var(--border-subtle))',
        borderInlineStart: isActiveTurn
          ? '4px solid var(--accent)'
          : passed
            ? '3px solid var(--accent)'
            : '3px solid var(--warn)',
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1fr)',
        gap: '9px',
        minInlineSize: 0,
        overflow: 'clip',
      }}
    >
      {/* Turn Header */}
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '8px',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 700, fontSize: '0.91rem', color: 'var(--text-primary)' }}>
            Turn #{turnNumber}
          </span>
          <span className="badge badge-accent" style={{ textTransform: 'none' }}>
            {specialistLabel}
          </span>
          {isActiveTurn ? (
            <span className="badge badge-ok">
              <Sparkles size={10} /> Live In Progress
            </span>
          ) : passed ? (
            <span className="badge badge-ok">
              <CheckCircle2 size={10} /> Verified
            </span>
          ) : (
            <span className="badge badge-warn">
              <AlertTriangle size={10} /> Needs Fix
            </span>
          )}
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            flexWrap: 'wrap',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.76rem',
            color: 'var(--text-muted)',
          }}
        >
          {steps.length > 0 ? <span>{steps.length} step{steps.length === 1 ? '' : 's'}</span> : null}
          {totalInquiries > 0 ? <span>· {totalInquiries} Q&amp;A</span> : null}
          {diffs.length > 0 ? <span>· {diffs.length} diff{diffs.length === 1 ? '' : 's'}</span> : null}
          {proofs.length > 0 ? <span>· {proofs.length} proof{proofs.length === 1 ? '' : 's'}</span> : null}
          {timeShort ? <span>· {timeShort}</span> : null}
        </div>
      </header>

      {/* 1. Task Lead Planning, Delegation Brief & Lead Epistemics */}
      {hasLeadSection ? (
        <section
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr)',
            gap: '6px',
            padding: '8px 10px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
          }}
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
            <span
              style={{
                fontSize: '0.76rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.03em',
                color: 'var(--purple)',
              }}
            >
              1. Task Lead Planning &amp; Delegation
            </span>
            {(leadPromptStr || leadContextStr) ? (
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                {leadPromptStr ? `Prompt ${(leadPromptStr.length / 1000).toFixed(1)}k` : ''}
                {leadPromptStr && leadContextStr ? ' · ' : ''}
                {leadContextStr ? `Context ${(leadContextStr.length / 1000).toFixed(1)}k` : ''}
              </span>
            ) : null}
          </div>

          <DeliberationDisclosure
            deliberation={turn.lead_deliberation}
            label="Task Lead Deliberation (Planning Rationale)"
          />

          {turn.brief ? (
            turn.brief.length > 320 && !isActiveTurn ? (
              <details
                style={{
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-canvas)',
                  padding: '6px 9px',
                }}
              >
                <summary style={{ cursor: 'pointer', fontSize: '0.84rem', color: 'var(--text-primary)' }}>
                  <strong style={{ color: 'var(--accent)' }}>Delegation Brief: </strong>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {turn.brief.replace(/\s+/g, ' ').slice(0, 140)}…
                  </span>
                </summary>
                <div
                  style={{
                    marginBlockStart: '6px',
                    fontSize: '0.84rem',
                    color: 'var(--text-primary)',
                    whiteSpace: 'pre-wrap',
                    overflowWrap: 'anywhere',
                  }}
                >
                  {turn.brief}
                </div>
              </details>
            ) : (
              <div
                style={{
                  padding: '6px 9px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-canvas)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '0.84rem',
                  whiteSpace: 'pre-wrap',
                  overflowWrap: 'anywhere',
                }}
              >
                <strong style={{ color: 'var(--accent)' }}>Delegation Brief: </strong>
                {turn.brief}
              </div>
            )
          ) : null}

          <SelfInquiryList
            inquiries={leadInquiries}
            label="Task Lead Self-Inquiry Q&A"
            defaultRole="task_lead"
          />

          {leadContextStr ? (
            <PromptInspector
              label="Task Lead Curated Context"
              rawContent={leadContextStr}
              accentColor="var(--purple)"
            />
          ) : null}

          {leadPromptStr ? (
            <PromptInspector
              label="Task Lead Full Planning Prompt"
              rawContent={leadPromptStr}
              accentColor="var(--warn)"
            />
          ) : null}
        </section>
      ) : null}

      {/* 2. Step-by-Step Activity Timeline */}
      {steps.length > 0 ? (
        <details
          open={isActiveTurn || isLatestTurn || !passed}
          style={{
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)',
            background: 'var(--bg-surface)',
            padding: '7px 10px',
          }}
        >
          <summary
            style={{
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '8px',
              fontSize: '0.84rem',
              fontWeight: 700,
              color: 'var(--ok)',
              userSelect: 'none',
            }}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
              <Activity size={12} /> 2. Step-by-Step Turn Execution ({steps.length} step{steps.length === 1 ? '' : 's'})
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.73rem', color: 'var(--text-muted)' }}>
              {steps[steps.length - 1]?.label}
            </span>
          </summary>

          <div
            style={{
              display: 'grid',
              gap: '5px',
              marginBlockStart: '8px',
              maxBlockSize: '320px',
              overflowY: 'auto',
            }}
          >
            {steps.map((step) => (
              <div
                key={step.index}
                style={{
                  display: 'grid',
                  gap: '4px',
                  padding: '6px 8px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-canvas)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '0.84rem',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '6px',
                    flexWrap: 'wrap',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '5px', flexWrap: 'wrap' }}>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.73rem',
                        color: 'var(--text-muted)',
                      }}
                    >
                      #{step.index}
                    </span>
                    <span className={`badge ${step.badgeClass}`} style={{ fontSize: '0.69rem' }}>
                      {step.label}
                    </span>
                    {step.specialistAgent ? (
                      <span
                        style={{
                          fontSize: '0.76rem',
                          fontWeight: 600,
                          color: 'var(--text-secondary)',
                        }}
                      >
                        {formatSpecialistLabel(step.specialistAgent)}
                      </span>
                    ) : null}
                  </div>
                  <div
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.73rem',
                      color: 'var(--text-muted)',
                    }}
                  >
                    {step.elapsedLabel ? (
                      <span style={{ color: 'var(--accent)' }}>{step.elapsedLabel}</span>
                    ) : null}
                    {step.timestampShort ? <span>{step.timestampShort}</span> : null}
                  </div>
                </div>

                <div style={{ color: 'var(--text-primary)', overflowWrap: 'anywhere' }}>
                  {step.message}
                </div>

                {step.detailsEntries.length > 0 ? (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                    {step.detailsEntries.map((de) => (
                      <span
                        key={de.key}
                        className="badge"
                        style={{ textTransform: 'none', fontFamily: 'var(--font-mono)', fontWeight: 500 }}
                        title={`${de.key}: ${de.value}`}
                      >
                        {de.key}: {de.value}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </details>
      ) : isActiveTurn ? (
        <div style={{ fontSize: '0.84rem', color: 'var(--accent)', padding: '4px 2px' }}>
          Initializing turn context and waiting for first specialist sub-step...
        </div>
      ) : null}

      {/* 3. Specialist Self-Inquiries, Curated Context & Full Prompt */}
      {(specInquiries.length > 0 || specContextStr || matchingPack || specPromptStr) ? (
        <section
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr)',
            gap: '6px',
            padding: '8px 10px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
          }}
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
            <span
              style={{
                fontSize: '0.76rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.03em',
                color: 'var(--accent)',
              }}
            >
              3. Specialist Context, Self-Inquiries &amp; Full Prompt ({specialistLabel})
            </span>
          </div>

          <SelfInquiryList
            inquiries={specInquiries}
            label={`${specialistLabel} Self-Inquiry Q&A`}
            defaultRole={specialistRaw}
          />

          {matchingPack &&
          (matchingPack.role_tailored_synthesis ||
            (matchingPack.selected_chunk_ids && matchingPack.selected_chunk_ids.length > 0)) ? (
            <details
              style={{
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-subtle)',
                borderInlineStart: '3px solid var(--purple)',
                background: 'var(--bg-canvas)',
                padding: '6px 10px',
              }}
            >
              <summary
                style={{
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '8px',
                  fontSize: '0.84rem',
                  fontWeight: 700,
                  color: 'var(--purple)',
                }}
              >
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                  <Layers size={12} /> Context Curator Pack ({matchingPack.target_agent_role})
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                  {(matchingPack.selected_chunk_ids ?? []).length} / {matchingPack.total_indexed_chunks ?? 0} chunks
                </span>
              </summary>
              <div style={{ display: 'grid', gap: '6px', marginBlockStart: '7px', fontSize: '0.84rem' }}>
                <DeliberationDisclosure
                  deliberation={matchingPack.deliberation}
                  label="Context Curator Selection Rationale"
                />
                {matchingPack.role_tailored_synthesis ? (
                  <div style={{ whiteSpace: 'pre-wrap', color: 'var(--text-primary)' }}>
                    <strong>Role-Tailored Synthesis: </strong>
                    {matchingPack.role_tailored_synthesis}
                  </div>
                ) : null}
                {matchingPack.search_queries_used && matchingPack.search_queries_used.length > 0 ? (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Queries: <code>{matchingPack.search_queries_used.join(' · ')}</code>
                  </div>
                ) : null}
                {matchingPack.selected_chunk_ids && matchingPack.selected_chunk_ids.length > 0 ? (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                    {matchingPack.selected_chunk_ids.map((cid) => (
                      <span
                        key={cid}
                        className="badge"
                        style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}
                      >
                        {cid}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            </details>
          ) : null}

          {specContextStr ? (
            <PromptInspector
              label={`Curated Context Block (${specialistLabel})`}
              rawContent={specContextStr}
              accentColor="var(--accent)"
            />
          ) : null}

          {specPromptStr ? (
            <PromptInspector
              label={`Full Model Prompt (${specialistLabel})`}
              rawContent={specPromptStr}
              accentColor="var(--warn)"
            />
          ) : null}
        </section>
      ) : null}

      {/* 4. Specialist Response, Verification Outcome, Diffs & Proofs */}
      {(hasSpecialistDelib || summaryText || touchedFiles.length > 0 || diffs.length > 0 || proofs.length > 0) ? (
        <section
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr)',
            gap: '7px',
            padding: '8px 10px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
          }}
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
            <span
              style={{
                fontSize: '0.76rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.03em',
                color: passed ? 'var(--ok)' : 'var(--warn)',
              }}
            >
              4. Specialist Response, Diffs &amp; Emitted Proofs
            </span>
            {summaryText ? (
              <button
                type="button"
                className="btn"
                style={{ padding: '1px 7px', fontSize: '0.76rem' }}
                onClick={() => void copySummaryText()}
              >
                {copiedSummary ? <Check size={10} style={{ color: 'var(--ok)' }} /> : <Copy size={10} />}
                {copiedSummary ? 'Copied' : 'Copy Response'}
              </button>
            ) : null}
          </div>

          {hasSpecialistDelib ? (
            <DeliberationDisclosure
              deliberation={turn.deliberation}
              label={`${specialistLabel} Deliberation`}
              defaultOpen={!passed}
            />
          ) : null}

          {summaryText ? (
            summaryText.length > 360 && passed ? (
              <details
                open={isLatestTurn}
                style={{
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-canvas)',
                  padding: '6px 9px',
                }}
              >
                <summary
                  style={{
                    cursor: 'pointer',
                    fontSize: '0.84rem',
                    color: 'var(--text-primary)',
                  }}
                >
                  <strong style={{ color: 'var(--ok)' }}>Turn Response / QA Output: </strong>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {summaryText.replace(/\s+/g, ' ').slice(0, 145)}…
                  </span>{' '}
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.73rem', color: 'var(--accent)' }}>
                    ({summaryText.length.toLocaleString()} chars)
                  </span>
                </summary>
                <pre tabIndex={0} className="prompt-pre" style={{ marginBlockStart: '6px' }}>
                  <code>{summaryText}</code>
                </pre>
              </details>
            ) : (
              <pre
                tabIndex={0}
                className="prompt-pre"
                style={{
                  borderColor: passed ? 'var(--border-subtle)' : 'color-mix(in srgb, var(--err) 45%, var(--border-subtle))',
                  color: passed ? 'var(--text-primary)' : 'var(--err)',
                }}
              >
                <code>{summaryText}</code>
              </pre>
            )
          ) : null}

          {touchedFiles.length > 0 ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
              {touchedFiles.map((f) => (
                <span
                  key={f}
                  className="badge"
                  style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}
                >
                  <GitCommit size={10} /> {f}
                </span>
              ))}
            </div>
          ) : null}

          {diffs.length > 0 ? (
            <div style={{ display: 'grid', gap: '5px' }}>
              {diffs.map((d, di) => (
                <details
                  key={di}
                  style={{
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-canvas)',
                    padding: '5px 8px',
                  }}
                >
                  <summary
                    style={{
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 600,
                      color: 'var(--ok)',
                    }}
                  >
                    ± Unified Code Diff: {d.label}
                  </summary>
                  <div style={{ marginBlockStart: '6px' }}>
                    <CodeDiffView title={d.label} diffText={d.diff} />
                  </div>
                </details>
              ))}
            </div>
          ) : null}

          {/* Full Module Source fallback for touched files without inline diff */}
          {touchedFiles
            .filter((f) => !diffFilesSet.has(f) && fileContents[f])
            .map((f) => {
              const entry = fileContents[f];
              const contentStr =
                typeof entry === 'string'
                  ? entry
                  : !entry.is_binary && entry.content
                    ? entry.content
                    : '';
              if (!contentStr) return null;
              const lineTotal =
                typeof entry === 'object' && entry.line_count
                  ? entry.line_count
                  : contentStr.split('\n').length;
              return (
                <details
                  key={`src_${f}`}
                  style={{
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-canvas)',
                    padding: '5px 8px',
                  }}
                >
                  <summary
                    style={{
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '5px',
                      fontSize: '0.8rem',
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--accent)',
                    }}
                  >
                    <FileCode size={11} /> Module Source: {f} ({lineTotal} lines)
                  </summary>
                  <pre tabIndex={0} className="prompt-pre" style={{ marginBlockStart: '6px' }}>
                    <code>{contentStr}</code>
                  </pre>
                </details>
              );
            })}

          {/* Emitted Proof Strip */}
          {proofs.length > 0 ? (
            <div style={{ display: 'grid', gap: '6px' }}>
              {imageProofs.length > 0 ? (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
                    gap: '7px',
                  }}
                >
                  {imageProofs.map((pr, pIdx) => {
                    const pairedAudio = pr.isWaveform ? audioProofs[0]?.url : undefined;
                    if (brokenProofUrls[pr.url]) {
                      return (
                        <div
                          key={pIdx}
                          style={{
                            padding: '8px 10px',
                            border: '1px dashed var(--border-subtle)',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-canvas)',
                            color: 'var(--text-muted)',
                            fontSize: '0.76rem',
                            fontFamily: 'var(--font-mono)',
                            display: 'grid',
                            gap: '3px',
                            alignContent: 'center',
                            overflowWrap: 'anywhere',
                          }}
                        >
                          <span style={{ color: 'var(--warn)', fontWeight: 600 }}>
                            Superseded on retry
                          </span>
                          <span>{pr.filename}</span>
                        </div>
                      );
                    }
                    return (
                      <button
                        key={pIdx}
                        type="button"
                        onClick={() =>
                          onOpenLightbox({
                            url: pr.url,
                            title: `${itemTitle} — Turn #${turnNumber} (${specialistLabel})`,
                            subtitle: pr.filename,
                            deliberation: turn.deliberation || turn.lead_deliberation,
                            audioUrl: pairedAudio,
                          })
                        }
                        style={{
                          padding: 0,
                          border: '1px solid var(--border-subtle)',
                          borderRadius: 'var(--radius-sm)',
                          background: '#05080f',
                          cursor: 'zoom-in',
                          overflow: 'clip',
                          textAlign: 'start',
                        }}
                      >
                        <img
                          src={pr.url}
                          alt={pr.filename}
                          width={640}
                          height={360}
                          loading="lazy"
                          onError={() =>
                            setBrokenProofUrls((prev) =>
                              prev[pr.url] ? prev : { ...prev, [pr.url]: true },
                            )
                          }
                          style={{ aspectRatio: '16 / 9', inlineSize: '100%', objectFit: 'cover' }}
                        />
                        <div
                          style={{
                            padding: '3px 6px',
                            fontSize: '0.73rem',
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--text-secondary)',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                          }}
                        >
                          <ImageIcon size={10} style={{ flexShrink: 0 }} />
                          {pr.filename}
                        </div>
                      </button>
                    );
                  })}
                </div>
              ) : null}

              {audioProofs.map((aud, aIdx) => (
                <div
                  key={`aud_${aIdx}`}
                  style={{
                    padding: '6px 8px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-canvas)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <div style={{ fontSize: '0.76rem', fontFamily: 'var(--font-mono)', marginBlockEnd: '4px' }}>
                    {aud.filename}
                  </div>
                  <audio controls preload="metadata" src={aud.url} style={{ inlineSize: '100%', blockSize: '28px' }} />
                </div>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}
    </article>
  );
}
