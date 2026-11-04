import { resolveMediaUrl } from '../../api/client';
import type { GameBibleResponse } from '../../api/types';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';
import type { LightboxMedia } from '../../components/MediaLightbox';

export interface GameBibleViewProps {
  bible: GameBibleResponse | undefined;
  cartridgeId: string;
  onOpenLightbox: (media: LightboxMedia) => void;
}

function renderKeyValueObject(
  raw: Record<string, unknown> | Array<Record<string, unknown>> | string | null | undefined,
  headingLabel?: string,
) {
  if (!raw) return null;

  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    return (
      <div
        style={{
          padding: '8px 10px',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-canvas)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        {headingLabel ? (
          <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--accent)', marginBlockEnd: '4px' }}>
            {headingLabel}
          </div>
        ) : null}
        <div style={{ fontSize: '0.91rem', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
          {trimmed}
        </div>
      </div>
    );
  }

  if (Array.isArray(raw)) {
    if (raw.length === 0) return null;
    return (
      <div style={{ display: 'grid', gap: '8px' }}>
        {headingLabel ? (
          <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--accent)' }}>
            {headingLabel}
          </div>
        ) : null}
        {raw.map((entry, idx) => (
          <div key={idx}>{renderKeyValueObject(entry as Record<string, unknown>)}</div>
        ))}
      </div>
    );
  }

  const obj = raw;
  const delib = typeof obj.deliberation === 'string' ? obj.deliberation : undefined;
  const entries = Object.entries(obj).filter(([k, v]) => k !== 'deliberation' && v !== null && v !== undefined && v !== '');
  if (!delib && entries.length === 0) return null;

  return (
    <div style={{ display: 'grid', gap: '10px' }}>
      <DeliberationDisclosure deliberation={delib} />
      <div style={{ display: 'grid', gap: '8px' }}>
        {entries.map(([key, val]) => (
          <div
            key={key}
            style={{
              padding: '8px 10px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-canvas)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--accent)', marginBlockEnd: '4px' }}>
              {key.replace(/_/g, ' ')}
            </div>
            {Array.isArray(val) ? (
              <ul style={{ margin: 0, paddingInlineStart: '18px', display: 'grid', gap: '4px', fontSize: '0.87rem', color: 'var(--text-secondary)' }}>
                {val.map((item, idx) => (
                  <li key={idx}>
                    {typeof item === 'string'
                      ? item
                      : typeof item === 'object' && item !== null
                        ? Object.entries(item as Record<string, unknown>)
                            .filter(([ik]) => ik !== 'deliberation')
                            .map(([ik, iv]) => `${ik}: ${typeof iv === 'string' ? iv : JSON.stringify(iv)}`)
                            .join(' · ')
                        : String(item)}
                  </li>
                ))}
              </ul>
            ) : typeof val === 'object' ? (
              <pre style={{ margin: 0, fontSize: '0.8rem', whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
                {JSON.stringify(val, null, 2)}
              </pre>
            ) : (
              <div style={{ fontSize: '0.91rem', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
                {String(val)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function GameBibleView({ bible, cartridgeId, onOpenLightbox }: GameBibleViewProps) {
  if (!bible || !bible.has_bible) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
        No Living Game Bible synthesized yet. Click <strong>New Game</strong> to shape a concept in Co-Producer or launch an autonomous Council build.
      </div>
    );
  }

  const mockups = bible.visual_mockups ?? [];
  const palette = bible.color_palette_hex ?? mockups[0]?.color_palette_hex ?? [];
  const genreFoundation = bible.genre_craft_foundation ?? null;
  const foundationalPillars = genreFoundation?.foundational_pillars ?? [];
  const uxContracts = genreFoundation?.essential_spatial_and_ux_contracts ?? [];
  const antiPatterns = genreFoundation?.genre_killing_anti_patterns ?? [];
  const pillars = bible.gameplay_pillars ?? [];
  const characters = bible.narrative_bible?.characters ?? [];
  const barks = bible.narrative_bible?.dialogue_barks ?? [];
  const storyboard = bible.experience_storyboard ?? [];
  const evolution = bible.concept_evolution ?? bible.concept_evolution_log ?? [];

  const sections = [
    { id: 'bible-vision', label: 'Vision & Hook' },
    ...(genreFoundation ? [{ id: 'bible-genre-foundation', label: `Genre & Fun (${foundationalPillars.length})` }] : []),
    { id: 'bible-mockups', label: `Concept Mockups (${mockups.length})` },
    { id: 'bible-loop', label: 'Core Loop & Pillars' },
    { id: 'bible-onboarding', label: 'Onboarding & Goals' },
    { id: 'bible-visuals', label: 'Palette, Visuals & Audio' },
    { id: 'bible-cast', label: `World & Cast (${characters.length})` },
    { id: 'bible-storyboard', label: `Storyboard & Evolution (${storyboard.length})` },
  ];

  return (
    <div className="bible-split-layout">
      {/* Left Sticky Section Navigation */}
      <nav
        aria-label="Game Bible Sections"
        className="scroll-pane"
        style={{
          padding: '14px 10px',
          borderInlineEnd: '1px solid var(--border-subtle)',
          background: 'var(--bg-surface)',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
        }}
      >
        <div style={{ fontSize: '0.76rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', padding: '4px 8px' }}>
          Living Game Bible
        </div>
        {sections.map((sec) => (
          <a
            key={sec.id}
            href={`#${sec.id}`}
            style={{
              padding: '6px 8px',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-secondary)',
              textDecoration: 'none',
              fontSize: '0.87rem',
              fontWeight: 600,
            }}
          >
            {sec.label}
          </a>
        ))}
      </nav>

      {/* Consolidated Specification Reader */}
      <div className="scroll-pane" style={{ padding: '18px 24px' }}>
        <div style={{ maxInlineSize: '960px', display: 'grid', gap: '24px' }}>
          {/* 1. Vision & Hook */}
          <section id="bible-vision" style={{ display: 'grid', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span className="badge badge-accent">{bible.core_genre}</span>
              <span className="badge badge-ok">{bible.status}</span>
              <code style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{bible.cartridge_id}</code>
            </div>
            <h1 style={{ margin: 0, fontSize: '1.6rem', fontWeight: 800 }}>{bible.game_title}</h1>
            <DeliberationDisclosure deliberation={bible.deliberation} label="Bible Synthesis Rationale" />

            {(bible.elevator_pitch || bible.creative_vision_summary) ? (
              <p style={{ margin: 0, fontSize: '1.02rem', lineHeight: 1.6, color: 'var(--text-primary)' }}>
                {bible.elevator_pitch || bible.creative_vision_summary}
              </p>
            ) : null}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '10px' }}>
              {bible.first_60_seconds_hook ? (
                <div style={{ padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--ok)', marginBlockEnd: '4px' }}>
                    First 60 Seconds Hook
                  </div>
                  <div style={{ fontSize: '0.91rem', color: 'var(--text-secondary)' }}>{bible.first_60_seconds_hook}</div>
                </div>
              ) : null}
              {bible.diegetic_ui_philosophy ? (
                <div style={{ padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--accent)', marginBlockEnd: '4px' }}>
                    Diegetic UI Philosophy
                  </div>
                  <div style={{ fontSize: '0.91rem', color: 'var(--text-secondary)' }}>{bible.diegetic_ui_philosophy}</div>
                </div>
              ) : null}
            </div>
          </section>

          {/* 1b. Genre Craft & Fun Foundation (Curated Genesis & Evolved Pillars) */}
          {genreFoundation ? (
            <section
              id="bible-genre-foundation"
              style={{
                display: 'grid',
                gap: '12px',
                padding: '14px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '1.16rem', fontWeight: 700 }}>
                  Genre Craft &amp; Fun Foundation
                </h2>
                <span className="badge badge-ok">
                  {genreFoundation.genre_and_subgenre_archetype}
                </span>
              </div>

              <DeliberationDisclosure
                deliberation={genreFoundation.deliberation}
                label="Genre Psychology & First-Principles Fun Rationale"
              />

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '10px' }}>
                <div style={{ padding: '10px 12px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--ok)', marginBlockEnd: '4px' }}>
                    Core Fun Thesis
                  </div>
                  <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.55, color: 'var(--text-primary)' }}>
                    {genreFoundation.core_fun_thesis}
                  </p>
                </div>

                <div style={{ padding: '10px 12px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--accent)', marginBlockEnd: '4px' }}>
                    Bootstrap-to-Mastery Arc (t=0 Unsolved &rarr; Earned)
                  </div>
                  <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.55, color: 'var(--text-secondary)' }}>
                    {genreFoundation.bootstrap_to_mastery_arc}
                  </p>
                </div>
              </div>

              {foundationalPillars.length > 0 ? (
                <div style={{ display: 'grid', gap: '8px' }}>
                  <h3 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700 }}>
                    Evolvable Genre-Craft Pillars ({foundationalPillars.length})
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '10px' }}>
                    {foundationalPillars.map((pillar) => (
                      <article
                        key={pillar.pillar_id || pillar.title}
                        style={{
                          padding: '10px 12px',
                          borderRadius: 'var(--radius-sm)',
                          background: 'var(--bg-canvas)',
                          border: '1px solid var(--border-subtle)',
                          display: 'grid',
                          gap: '6px',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '6px', flexWrap: 'wrap' }}>
                          <strong style={{ fontSize: '0.92rem', color: 'var(--text-primary)' }}>{pillar.title}</strong>
                          <code style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>{pillar.pillar_id}</code>
                        </div>
                        <div style={{ fontSize: '0.86rem', color: 'var(--text-primary)' }}>
                          <strong>Why It&apos;s Fun: </strong>
                          {pillar.why_it_creates_fun}
                        </div>
                        <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                          <strong>Systemic Expression: </strong>
                          {pillar.systemic_mechanics_expression}
                        </div>
                        <div style={{ fontSize: '0.82rem', color: 'var(--warn)' }}>
                          <strong>Avoid Anti-Pattern: </strong>
                          {pillar.anti_pattern_to_avoid}
                        </div>
                        <DeliberationDisclosure
                          deliberation={pillar.deliberation}
                          label="Pillar Rationale"
                        />
                      </article>
                    ))}
                  </div>
                </div>
              ) : null}

              {uxContracts.length > 0 || antiPatterns.length > 0 ? (
                <details style={{ padding: '8px 10px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)' }}>
                  <summary style={{ cursor: 'pointer', fontSize: '0.86rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Spatial / UX Precision Contracts ({uxContracts.length}) &amp; Genre-Killing Anti-Patterns ({antiPatterns.length})
                  </summary>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '12px', marginBlockStart: '10px' }}>
                    {uxContracts.length > 0 ? (
                      <div>
                        <div style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--accent)', marginBlockEnd: '6px' }}>
                          Essential Spatial &amp; UX Contracts
                        </div>
                        <ul style={{ margin: 0, paddingInlineStart: '18px', fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'grid', gap: '4px' }}>
                          {uxContracts.map((item, idx) => (
                            <li key={idx}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {antiPatterns.length > 0 ? (
                      <div>
                        <div style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--warn)', marginBlockEnd: '6px' }}>
                          Genre-Killing Anti-Patterns (Forbidden)
                        </div>
                        <ul style={{ margin: 0, paddingInlineStart: '18px', fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'grid', gap: '4px' }}>
                          {antiPatterns.map((item, idx) => (
                            <li key={idx}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                </details>
              ) : null}
            </section>
          ) : null}

          {/* 2. 16:9 Concept Mockups */}
          {mockups.length > 0 ? (
            <section id="bible-mockups" style={{ display: 'grid', gap: '12px' }}>
              <h2 style={{ margin: 0, fontSize: '1.16rem', fontWeight: 700 }}>16:9 Visual Concept Mockups</h2>
              {mockups.map((m) => {
                const imgUrl = resolveMediaUrl(m.image_path || m.url, cartridgeId);
                return (
                  <article
                    key={m.mockup_id}
                    style={{
                      padding: '14px',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-surface)',
                      border: '1px solid var(--border-subtle)',
                      display: 'grid',
                      gap: '10px',
                    }}
                  >
                    <h3 style={{ margin: 0, fontSize: '1.02rem' }}>{m.title}</h3>
                    <DeliberationDisclosure deliberation={m.deliberation} />
                    {imgUrl ? (
                      <img
                        src={imgUrl}
                        alt={m.title}
                        loading="lazy"
                        onClick={() =>
                          onOpenLightbox({
                            url: imgUrl,
                            title: m.title,
                            subtitle: m.camera_and_spatial_paradigm,
                            deliberation: m.deliberation,
                          })
                        }
                        style={{
                          aspectRatio: '16 / 9',
                          inlineSize: '100%',
                          objectFit: 'cover',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--border-subtle)',
                          cursor: 'zoom-in',
                        }}
                      />
                    ) : null}
                    <p style={{ margin: 0, fontSize: '0.91rem', color: 'var(--text-secondary)' }}>
                      {m.scene_description}
                    </p>
                    {m.ui_callout_notes && m.ui_callout_notes.length > 0 ? (
                      <ul style={{ margin: 0, paddingInlineStart: '18px', fontSize: '0.87rem', color: 'var(--text-secondary)' }}>
                        {m.ui_callout_notes.map((note, idx) => (
                          <li key={idx}>{note}</li>
                        ))}
                      </ul>
                    ) : null}
                  </article>
                );
              })}
            </section>
          ) : null}

          {/* 3. Core Loop & Gameplay Pillars */}
          <section id="bible-loop" style={{ display: 'grid', gap: '12px', padding: '14px', borderRadius: 'var(--radius-md)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <h2 style={{ margin: 0, fontSize: '1.16rem', fontWeight: 700 }}>Core Loop &amp; Gameplay Pillars</h2>
            {pillars.length > 0 ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '8px' }}>
                {pillars.map((p, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '10px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-canvas)',
                      border: '1px solid var(--border-subtle)',
                      fontSize: '0.87rem',
                    }}
                  >
                    {typeof p === 'string'
                      ? p
                      : `${String(p.pillar_name || p.title || `Pillar #${idx + 1}`)}: ${String(p.description || p.mechanical_expression || JSON.stringify(p))}`}
                  </div>
                ))}
              </div>
            ) : null}
            {renderKeyValueObject(bible.core_loop)}
            {renderKeyValueObject(bible.engagement_plan)}
          </section>

          {/* 4. Onboarding, Goal Hierarchy & Level Strategy */}
          <section id="bible-onboarding" style={{ display: 'grid', gap: '12px', padding: '14px', borderRadius: 'var(--radius-md)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <h2 style={{ margin: 0, fontSize: '1.16rem', fontWeight: 700 }}>Onboarding, Goals &amp; Level Generation</h2>
            {renderKeyValueObject(bible.onboarding_plan)}
            {renderKeyValueObject(bible.goal_hierarchy)}
            {renderKeyValueObject(bible.level_generation_strategy)}
          </section>

          {/* 5. Per-Game Color Palette, Visual Style & Audio Direction */}
          <section id="bible-visuals" style={{ display: 'grid', gap: '12px', padding: '14px', borderRadius: 'var(--radius-md)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <h2 style={{ margin: 0, fontSize: '1.16rem', fontWeight: 700 }}>Game Color Palette, Visual Style &amp; Audio Direction</h2>
            {palette.length > 0 ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {palette.map((hex) => (
                  <div
                    key={hex}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '5px 10px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-canvas)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    <span
                      style={{
                        inlineSize: '18px',
                        blockSize: '18px',
                        borderRadius: '4px',
                        background: hex,
                        border: '1px solid rgba(255,255,255,0.2)',
                      }}
                    />
                    <code style={{ fontSize: '0.84rem' }}>{hex}</code>
                  </div>
                ))}
              </div>
            ) : null}
            {bible.camera_and_spatial_paradigm ? (
              <div style={{ fontSize: '0.91rem', color: 'var(--text-secondary)' }}>
                <strong>Camera &amp; Spatial Paradigm:</strong> {bible.camera_and_spatial_paradigm}
              </div>
            ) : null}
            {renderKeyValueObject(bible.visual_style_guide, 'Visual Style Guide')}
            {renderKeyValueObject(bible.audio_direction_guide, 'Audio Direction Guide')}
          </section>

          {/* 6. World, Characters & Voice Barks */}
          <section id="bible-cast" style={{ display: 'grid', gap: '12px', padding: '14px', borderRadius: 'var(--radius-md)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <h2 style={{ margin: 0, fontSize: '1.16rem', fontWeight: 700 }}>World Lore, Characters &amp; Voice Barks</h2>
            {bible.narrative_bible?.world_title ? (
              <div style={{ fontSize: '0.98rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {bible.narrative_bible.world_title}
              </div>
            ) : null}
            {bible.narrative_bible?.setting_summary ? (
              <p style={{ margin: 0, fontSize: '0.91rem', color: 'var(--text-secondary)' }}>
                {bible.narrative_bible.setting_summary}
              </p>
            ) : null}
            {bible.narrative_bible?.lore_premise ? (
              <p style={{ margin: 0, fontSize: '0.91rem', color: 'var(--text-secondary)' }}>
                {bible.narrative_bible.lore_premise}
              </p>
            ) : null}

            {characters.length > 0 ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '10px' }}>
                {characters.map((c) => (
                  <div
                    key={c.character_id}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-canvas)',
                      border: '1px solid var(--border-subtle)',
                      display: 'grid',
                      gap: '5px',
                    }}
                  >
                    <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{c.name}</div>
                    {c.role ? <div style={{ fontSize: '0.84rem', color: 'var(--accent)' }}>{c.role}</div> : null}
                    <DeliberationDisclosure deliberation={c.deliberation} />
                    {c.backstory ? (
                      <div style={{ fontSize: '0.87rem', color: 'var(--text-secondary)' }}>{c.backstory}</div>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : null}

            {barks.length > 0 ? (
              <div style={{ display: 'grid', gap: '6px' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                  Dialogue &amp; Acoustic Barks ({barks.length})
                </div>
                {barks.slice(0, 12).map((b, idx) => (
                  <div
                    key={b.bark_id || idx}
                    style={{
                      padding: '8px 10px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-canvas)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '10px',
                      flexWrap: 'wrap',
                    }}
                  >
                    <div style={{ fontSize: '0.87rem' }}>
                      <strong>{b.speaker_name || b.speaker_id || 'Voice'}:</strong> “{b.line || b.text}”
                    </div>
                    {(b.audio_file || b.voice_clip) ? (
                      <audio controls src={resolveMediaUrl(b.audio_file || b.voice_clip, cartridgeId)} style={{ blockSize: '28px' }} />
                    ) : null}
                  </div>
                ))}
              </div>
            ) : null}
          </section>

          {/* 7. Storyboard & Concept Evolution */}
          {(storyboard.length > 0 || evolution.length > 0) ? (
            <section id="bible-storyboard" style={{ display: 'grid', gap: '12px', padding: '14px', borderRadius: 'var(--radius-md)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
              <h2 style={{ margin: 0, fontSize: '1.16rem', fontWeight: 700 }}>Experience Storyboard &amp; Concept Evolution</h2>
              {storyboard.map((beat) => (
                <div
                  key={beat.beat_index}
                  style={{
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-canvas)',
                    border: '1px solid var(--border-subtle)',
                    display: 'grid',
                    gap: '4px',
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: '0.91rem', color: 'var(--accent)' }}>
                    Beat #{beat.beat_index}: {beat.phase_title}
                  </div>
                  <DeliberationDisclosure deliberation={beat.deliberation} />
                  {beat.sensory_and_camera_staging ? (
                    <div style={{ fontSize: '0.87rem', color: 'var(--text-secondary)' }}>{beat.sensory_and_camera_staging}</div>
                  ) : null}
                </div>
              ))}
              {evolution.length > 0 ? (
                <details>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, fontSize: '0.87rem', color: 'var(--text-secondary)' }}>
                    Concept Pivot Log ({evolution.length})
                  </summary>
                  <div style={{ display: 'grid', gap: '8px', marginBlockStart: '8px' }}>
                    {evolution.map((ev, idx) => (
                      <div
                        key={ev.entry_id || idx}
                        style={{
                          padding: '8px 10px',
                          borderRadius: 'var(--radius-sm)',
                          background: 'var(--bg-canvas)',
                          border: '1px solid var(--border-subtle)',
                          fontSize: '0.87rem',
                        }}
                      >
                        <strong>{ev.author_role || 'Council'}:</strong>{' '}
                        {ev.evolved_premise || ev.hypothesis || ev.pivot_reason || ev.critique_or_pivot_reason}
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}
