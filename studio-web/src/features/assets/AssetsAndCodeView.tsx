import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { FileCode2, Film, Image as ImageIcon, Music, Sparkles } from 'lucide-react';
import { resolveMediaUrl, studioApi } from '../../api/client';
import type { GameBibleResponse } from '../../api/types';
import { ErrorBanner } from '../../components/ErrorBanner';
import type { LightboxMedia } from '../../components/MediaLightbox';

export type AssetCategory = 'sprites' | 'mockups' | 'proofs' | 'audio' | 'code';

export interface AssetsAndCodeViewProps {
  cartridgeId: string;
  bible: GameBibleResponse | undefined;
  onOpenLightbox: (media: LightboxMedia) => void;
}

export function AssetsAndCodeView({ cartridgeId, bible, onOpenLightbox }: AssetsAndCodeViewProps) {
  const [category, setCategory] = useState<AssetCategory>('sprites');
  const [selectedModulePath, setSelectedModulePath] = useState<string>('');

  const artifactsQuery = useQuery({
    queryKey: ['artifacts', cartridgeId],
    queryFn: () => studioApi.getArtifacts(cartridgeId),
  });

  // Structural provenance categorization (Rule 2: zero keyword/regex guessing!)
  const categorized = useMemo(() => {
    const boardItems = bible?.production_board?.items ?? [];
    const proofUrlMap = new Map<string, { url: string; itemTitle: string }>();
    const moduleSet = new Set<string>(artifactsQuery.data?.code_modules ?? []);

    for (const item of boardItems) {
      for (const mod of item.target_modules ?? []) {
        if (mod && !mod.endsWith('.png') && !mod.endsWith('.wav') && !mod.endsWith('.mp3')) {
          moduleSet.add(mod);
        }
      }
      for (const u of item.latest_proof_urls ?? []) {
        if (u && !u.toLowerCase().endsWith('.wav') && !u.toLowerCase().endsWith('.mp3')) {
          const fname = u.split('/').pop()?.split('?')[0] ?? u;
          proofUrlMap.set(fname, { url: u, itemTitle: item.title });
        }
      }
    }

    const mockupFiles = new Set<string>();
    const mockups = (bible?.visual_mockups ?? []).map((m) => {
      const p = m.image_path || m.url || '';
      const fname = p.split('/').pop()?.split('?')[0] ?? '';
      if (fname) mockupFiles.add(fname);
      return m;
    });

    const uniqueSpriteMap = new Map<string, NonNullable<typeof artifactsQuery.data>['sprites'][number]>();
    for (const s of artifactsQuery.data?.sprites ?? []) {
      if (proofUrlMap.has(s.filename) || mockupFiles.has(s.filename)) continue;
      const key = s.url || s.filename;
      if (!uniqueSpriteMap.has(key)) uniqueSpriteMap.set(key, s);
    }
    const gameSprites = Array.from(uniqueSpriteMap.values());

    const audioClips = artifactsQuery.data?.audio_clips ?? [];
    // Deduplicate audio clips by filename
    const uniqueAudioMap = new Map<string, (typeof audioClips)[number]>();
    for (const a of audioClips) {
      if (!uniqueAudioMap.has(a.filename)) uniqueAudioMap.set(a.filename, a);
    }

    const allModules = Array.from(moduleSet).sort();

    return {
      gameSprites,
      mockups,
      proofCaptures: Array.from(proofUrlMap.entries()).map(([filename, meta]) => ({
        filename,
        url: meta.url,
        itemTitle: meta.itemTitle,
      })),
      audioClips: Array.from(uniqueAudioMap.values()),
      allModules,
    };
  }, [artifactsQuery.data, bible]);

  const activeModule = selectedModulePath || categorized.allModules[0] || '';

  const moduleContentQuery = useQuery({
    queryKey: ['artifact-content', cartridgeId, activeModule],
    queryFn: () => studioApi.getArtifactContent({ path: activeModule, cartridgeId }),
    enabled: category === 'code' && Boolean(activeModule),
  });

  const tabs: Array<{ id: AssetCategory; label: string; count: number; Icon: typeof ImageIcon }> = [
    { id: 'sprites', label: 'Visual Assets', count: categorized.gameSprites.length, Icon: ImageIcon },
    { id: 'mockups', label: 'Concept Mockups', count: categorized.mockups.length, Icon: Sparkles },
    { id: 'proofs', label: 'QA Proof Captures', count: categorized.proofCaptures.length, Icon: Film },
    { id: 'audio', label: 'Audio Tracks', count: categorized.audioClips.length, Icon: Music },
    { id: 'code', label: 'GDScript & Scenes', count: categorized.allModules.length, Icon: FileCode2 },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateRows: 'auto 1fr', blockSize: '100%', minBlockSize: 0 }}>
      <div
        style={{
          padding: '10px 14px',
          borderBlockEnd: '1px solid var(--border-subtle)',
          background: 'var(--bg-surface)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div className="pill-list" role="tablist" aria-label="Asset Categories">
          {tabs.map((t) => {
            const Icon = t.Icon;
            return (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={category === t.id}
                className={`pill${category === t.id ? ' is-active' : ''}`}
                onClick={() => setCategory(t.id)}
              >
                <Icon size={12} />
                <span>{t.label}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem' }}>{t.count}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="scroll-pane" style={{ padding: '14px' }}>
        <ErrorBanner error={artifactsQuery.error} onRetry={() => void artifactsQuery.refetch()} />

        {category === 'sprites' ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '10px' }}>
            {categorized.gameSprites.map((sp, idx) => {
              const url = resolveMediaUrl(sp.url, cartridgeId);
              return (
                <article
                  key={`${sp.filename}-${idx}`}
                  className={idx >= 16 ? 'deferred-asset-card' : undefined}
                  style={{
                    padding: '10px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border-subtle)',
                    display: 'grid',
                    gap: '8px',
                  }}
                >
                  <button
                    type="button"
                    onClick={() =>
                      onOpenLightbox({
                        url,
                        title: sp.filename,
                        subtitle: `${(sp.size_bytes / 1024).toFixed(1)} KB`,
                      })
                    }
                    style={{
                      padding: '8px',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-sm)',
                      background: '#05080f',
                      cursor: 'zoom-in',
                      display: 'grid',
                      placeItems: 'center',
                      blockSize: '130px',
                    }}
                  >
                    <img
                      src={url}
                      alt={sp.filename}
                      loading="lazy"
                      style={{ maxBlockSize: '112px', objectFit: 'contain', imageRendering: 'pixelated' }}
                    />
                  </button>
                  <div style={{ fontSize: '0.84rem', fontWeight: 600, overflowWrap: 'anywhere' }}>{sp.filename}</div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {(sp.size_bytes / 1024).toFixed(1)} KB
                  </div>
                </article>
              );
            })}
          </div>
        ) : null}

        {category === 'mockups' ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '12px' }}>
            {categorized.mockups.map((m) => {
              const url = resolveMediaUrl(m.image_path || m.url, cartridgeId);
              return (
                <article
                  key={m.mockup_id}
                  style={{
                    padding: '12px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border-subtle)',
                    display: 'grid',
                    gap: '8px',
                  }}
                >
                  {url ? (
                    <img
                      src={url}
                      alt={m.title}
                      loading="lazy"
                      onClick={() => onOpenLightbox({ url, title: m.title, deliberation: m.deliberation })}
                      style={{
                        aspectRatio: '16 / 9',
                        inlineSize: '100%',
                        objectFit: 'cover',
                        borderRadius: 'var(--radius-sm)',
                        cursor: 'zoom-in',
                      }}
                    />
                  ) : null}
                  <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{m.title}</div>
                  <div className="clamp-3" style={{ fontSize: '0.87rem', color: 'var(--text-secondary)' }}>
                    {m.scene_description}
                  </div>
                </article>
              );
            })}
          </div>
        ) : null}

        {category === 'proofs' ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '10px' }}>
            {categorized.proofCaptures.map((p, idx) => {
              const url = resolveMediaUrl(p.url, cartridgeId);
              return (
                <article
                  key={`${p.filename}-${idx}`}
                  className={idx >= 12 ? 'deferred-asset-card' : undefined}
                  style={{
                    padding: '10px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border-subtle)',
                    display: 'grid',
                    gap: '6px',
                  }}
                >
                  <img
                    src={url}
                    alt={p.filename}
                    loading="lazy"
                    onClick={() => onOpenLightbox({ url, title: p.itemTitle, subtitle: p.filename })}
                    style={{
                      aspectRatio: '16 / 9',
                      inlineSize: '100%',
                      objectFit: 'cover',
                      borderRadius: 'var(--radius-sm)',
                      cursor: 'zoom-in',
                    }}
                  />
                  <div style={{ fontWeight: 600, fontSize: '0.87rem' }}>{p.itemTitle}</div>
                  <code style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>{p.filename}</code>
                </article>
              );
            })}
          </div>
        ) : null}

        {category === 'audio' ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '10px' }}>
            {categorized.audioClips.map((clip) => {
              const url = resolveMediaUrl(clip.url, cartridgeId);
              return (
                <article
                  key={clip.filename}
                  style={{
                    padding: '12px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border-subtle)',
                    display: 'grid',
                    gap: '8px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.87rem', fontFamily: 'var(--font-mono)' }}>
                      {clip.filename}
                    </span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      {(clip.size_bytes / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <audio controls preload="none" src={url} style={{ inlineSize: '100%' }} />
                </article>
              );
            })}
          </div>
        ) : null}

        {category === 'code' ? (
          <div className="assets-code-grid">
            <div
              style={{
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                padding: '8px',
                display: 'flex',
                flexDirection: 'column',
                gap: '3px',
              }}
            >
              {categorized.allModules.map((mod) => {
                const isSel = mod === activeModule;
                return (
                  <button
                    key={mod}
                    type="button"
                    onClick={() => setSelectedModulePath(mod)}
                    style={{
                      textAlign: 'left',
                      padding: '6px 8px',
                      borderRadius: 'var(--radius-sm)',
                      border: `1px solid ${isSel ? 'var(--accent)' : 'transparent'}`,
                      background: isSel ? 'color-mix(in srgb, var(--accent) 14%, var(--bg-elevated))' : 'transparent',
                      color: isSel ? 'var(--text-primary)' : 'var(--text-secondary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.84rem',
                      cursor: 'pointer',
                      overflowWrap: 'anywhere',
                    }}
                  >
                    {mod}
                  </button>
                );
              })}
            </div>

            <div style={{ minInlineSize: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <ErrorBanner error={moduleContentQuery.error} onRetry={() => void moduleContentQuery.refetch()} />
              {moduleContentQuery.isLoading ? (
                <div style={{ padding: '20px', color: 'var(--text-muted)' }}>Loading {activeModule}...</div>
              ) : moduleContentQuery.data?.content !== undefined ? (
                <div style={{ display: 'grid', gap: '6px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <code style={{ fontWeight: 700, color: 'var(--accent)' }}>{activeModule}</code>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {(moduleContentQuery.data.size_bytes / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <pre className="diff-block" style={{ maxBlockSize: '72dvh', overflowY: 'auto' }}>
                    {moduleContentQuery.data.content}
                  </pre>
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
