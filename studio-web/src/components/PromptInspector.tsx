import { useMemo, useState } from 'react';
import { Check, Copy, FileText, FolderTree, Search } from 'lucide-react';
import { coercePromptString, countLinesFast, parsePromptIntoSections } from '../domain/promptParser';

export interface PromptInspectorProps {
  label: string;
  rawContent: unknown;
  accentColor?: string;
  defaultOpen?: boolean;
}

export function PromptInspector({
  label,
  rawContent,
  accentColor = 'var(--accent)',
  defaultOpen = false,
}: PromptInspectorProps) {
  const fullText = useMemo(() => coercePromptString(rawContent), [rawContent]);
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [rawMode, setRawMode] = useState(false);
  const [expandGeneration, setExpandGeneration] = useState<{ gen: number; open: boolean }>({
    gen: 0,
    open: false,
  });
  const [filterQuery, setFilterQuery] = useState('');
  const [copied, setCopied] = useState(false);

  const lineCount = useMemo(() => countLinesFast(fullText), [fullText]);

  // Lazily parse multi-hundred-KB prompt strings only when expanded
  const sections = useMemo(
    () => (isOpen && fullText ? parsePromptIntoSections(fullText) : []),
    [isOpen, fullText],
  );

  const filteredSections = useMemo(() => {
    const q = filterQuery.trim().toLowerCase();
    if (!q) return sections;
    return sections.filter(
      (s) =>
        s.title.toLowerCase().includes(q) ||
        s.content.toLowerCase().includes(q),
    );
  }, [sections, filterQuery]);

  if (!fullText.trim()) return null;

  const handleCopy = async () => {
    if (!navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(fullText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // Ignore clipboard permission denial
    }
  };

  return (
    <details
      open={isOpen}
      onToggle={(e) => setIsOpen(e.currentTarget.open)}
      style={{
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border-subtle)',
        borderInlineStart: `3px solid ${accentColor}`,
        background: 'var(--bg-canvas)',
        padding: '6px 10px',
      }}
    >
      <summary
        onClick={(e) => {
          e.preventDefault();
          setIsOpen((prev) => !prev);
        }}
        style={{
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '8px',
          flexWrap: 'wrap',
          fontSize: '0.84rem',
          fontWeight: 700,
          color: accentColor,
          userSelect: 'none',
        }}
      >
        <span>{label}</span>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.76rem',
            fontWeight: 500,
            color: 'var(--text-muted)',
          }}
        >
          {lineCount.toLocaleString()} lines · {fullText.length.toLocaleString()} chars (untruncated)
        </span>
      </summary>

      {isOpen ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '8px', marginBlockStart: '8px' }}>
          {/* Inspector Toolbar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '6px',
              flexWrap: 'wrap',
              padding: '5px 7px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px', flexWrap: 'wrap' }}>
              {!rawMode && sections.length > 1 ? (
                <>
                  <button
                    type="button"
                    className="btn"
                    style={{ padding: '2px 7px', fontSize: '0.8rem' }}
                    onClick={() => setExpandGeneration((prev) => ({ gen: prev.gen + 1, open: true }))}
                  >
                    Expand All ({filteredSections.length})
                  </button>
                  <button
                    type="button"
                    className="btn"
                    style={{ padding: '2px 7px', fontSize: '0.8rem' }}
                    onClick={() => setExpandGeneration((prev) => ({ gen: prev.gen + 1, open: false }))}
                  >
                    Collapse All
                  </button>
                </>
              ) : null}

              <button
                type="button"
                className={`btn${rawMode ? ' btn-primary' : ''}`}
                style={{ padding: '2px 7px', fontSize: '0.8rem' }}
                onClick={() => setRawMode((v) => !v)}
              >
                {rawMode ? <FolderTree size={11} /> : <FileText size={11} />}
                {rawMode ? `Structured Sections (${sections.length})` : 'Raw Full Text'}
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
              {!rawMode && sections.length > 1 ? (
                <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
                  <Search
                    size={11}
                    style={{
                      position: 'absolute',
                      insetInlineStart: '7px',
                      color: 'var(--text-muted)',
                      pointerEvents: 'none',
                    }}
                  />
                  <input
                    type="search"
                    className="input-control"
                    style={{ paddingInlineStart: '22px', paddingBlock: '2px', fontSize: '0.8rem', inlineSize: '165px' }}
                    placeholder="Filter prompt..."
                    value={filterQuery}
                    onChange={(e) => setFilterQuery(e.target.value)}
                  />
                </div>
              ) : null}

              <button
                type="button"
                className="btn"
                style={{ padding: '2px 8px', fontSize: '0.8rem' }}
                onClick={() => void handleCopy()}
              >
                {copied ? <Check size={11} style={{ color: 'var(--ok)' }} /> : <Copy size={11} />}
                {copied ? 'Copied!' : 'Copy Full'}
              </button>
            </div>
          </div>

          {rawMode || sections.length <= 1 ? (
            <pre tabIndex={0} className="prompt-pre" style={{ maxBlockSize: '560px' }}>
              <code>{fullText}</code>
            </pre>
          ) : filteredSections.length === 0 ? (
            <div style={{ padding: '10px', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
              No prompt sections matched &ldquo;{filterQuery}&rdquo;.
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '5px' }}>
              {filteredSections.map((sec) => {
                const sectionOpenDefault =
                  expandGeneration.gen > 0 ? expandGeneration.open : filterQuery.trim().length > 0;
                return (
                  <details
                    key={`${sec.key}_${expandGeneration.gen}`}
                    open={sectionOpenDefault}
                    style={{
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)',
                      background: 'var(--bg-surface)',
                      padding: '5px 8px',
                    }}
                  >
                    <summary
                      style={{
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'baseline',
                        justifyContent: 'space-between',
                        gap: '8px',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        color: 'var(--text-primary)',
                      }}
                    >
                      <span style={{ minInlineSize: 0, overflowWrap: 'anywhere' }}>
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.73rem',
                            color: accentColor,
                            marginInlineEnd: '6px',
                          }}
                        >
                          #{sec.index}
                        </span>
                        {sec.title}
                      </span>
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.73rem',
                          color: 'var(--text-muted)',
                          whiteSpace: 'nowrap',
                          flexShrink: 0,
                        }}
                      >
                        {sec.subs.length > 0 ? `${sec.subs.length} blocks · ` : ''}
                        {sec.lineCount.toLocaleString()}L · {sec.charCount.toLocaleString()}c
                      </span>
                    </summary>

                    <div style={{ display: 'grid', gap: '5px', marginBlockStart: '6px' }}>
                      {sec.subs.length > 0 ? (
                        <>
                          {sec.preamble ? (
                            <pre tabIndex={0} className="prompt-pre" style={{ maxBlockSize: '200px' }}>
                              <code>{sec.preamble}</code>
                            </pre>
                          ) : null}
                          {sec.subs.map((sub) => (
                            <details
                              key={`${sub.key}_${expandGeneration.gen}`}
                              open={expandGeneration.gen > 0 ? expandGeneration.open : false}
                              style={{
                                borderRadius: 'var(--radius-sm)',
                                border: '1px solid var(--border-subtle)',
                                background: 'var(--bg-canvas)',
                                padding: '4px 8px',
                              }}
                            >
                              <summary
                                style={{
                                  cursor: 'pointer',
                                  display: 'flex',
                                  alignItems: 'baseline',
                                  justifyContent: 'space-between',
                                  gap: '8px',
                                  fontSize: '0.76rem',
                                  fontWeight: 600,
                                  color: 'var(--text-secondary)',
                                }}
                              >
                                <span style={{ minInlineSize: 0, overflowWrap: 'anywhere' }}>{sub.title}</span>
                                <span
                                  style={{
                                    fontFamily: 'var(--font-mono)',
                                    fontSize: '0.69rem',
                                    color: 'var(--text-muted)',
                                    whiteSpace: 'nowrap',
                                    flexShrink: 0,
                                  }}
                                >
                                  {sub.meta}
                                </span>
                              </summary>
                              <pre tabIndex={0} className="prompt-pre" style={{ marginBlockStart: '5px' }}>
                                <code>{sub.content}</code>
                              </pre>
                            </details>
                          ))}
                        </>
                      ) : (
                        <pre tabIndex={0} className="prompt-pre">
                          <code>{sec.content}</code>
                        </pre>
                      )}
                    </div>
                  </details>
                );
              })}
            </div>
          )}
        </div>
      ) : null}
    </details>
  );
}
