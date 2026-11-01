import { AlertTriangle, RefreshCw, X } from 'lucide-react';
import { StudioApiError } from '../api/client';
import { splitErrorHeadlineAndTraceback } from '../domain/errors';

export interface ErrorBannerProps {
  error: unknown;
  title?: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}

/**
 * Rule 5: Unmasked epistemic error surface — preserves authentic HTTP status, endpoint, and server trace.
 */
export function ErrorBanner({ error, title = 'Studio Error', onRetry, onDismiss }: ErrorBannerProps) {
  if (!error) return null;

  const isStudioErr = error instanceof StudioApiError;
  const message = error instanceof Error ? error.message : String(error);
  const { headline, stackTrace } = splitErrorHeadlineAndTraceback(message);

  return (
    <div
      role="alert"
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '10px',
        padding: '10px 14px',
        background: 'color-mix(in srgb, var(--err) 12%, var(--bg-surface))',
        border: '1px solid color-mix(in srgb, var(--err) 45%, var(--border-subtle))',
        borderRadius: 'var(--radius-md)',
        color: 'var(--text-primary)',
      }}
    >
      <AlertTriangle size={16} style={{ color: 'var(--err)', flexShrink: 0, marginBlockStart: '2px' }} />
      <div style={{ flex: '1 1 auto', minInlineSize: 0 }}>
        <div style={{ fontWeight: 700, fontSize: '0.87rem', color: 'var(--err)' }}>
          {title}
          {isStudioErr && error.status > 0 ? ` (HTTP ${error.status} · ${error.endpoint})` : ''}
        </div>
        {isStudioErr && error.deliberation && error.deliberation !== message ? (
          <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginBlockStart: '2px' }}>
            {error.deliberation}
          </div>
        ) : null}
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.84rem',
            marginBlockStart: '4px',
            whiteSpace: 'pre-wrap',
            overflowWrap: 'anywhere',
          }}
        >
          {headline}
        </div>
        {stackTrace ? (
          <details style={{ marginBlockStart: '6px' }}>
            <summary
              style={{
                cursor: 'pointer',
                fontSize: '0.8rem',
                fontWeight: 600,
                color: 'var(--warn)',
                userSelect: 'none',
              }}
            >
              View Full Diagnostic Stack Trace
            </summary>
            <pre tabIndex={0} className="prompt-pre" style={{ marginBlockStart: '6px', maxBlockSize: '240px', color: 'var(--err)' }}>
              <code>{stackTrace}</code>
            </pre>
          </details>
        ) : null}
      </div>
      <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
        {onRetry ? (
          <button type="button" className="btn btn-danger" onClick={onRetry}>
            <RefreshCw size={13} /> Retry
          </button>
        ) : null}
        {onDismiss ? (
          <button
            type="button"
            className="btn btn-ghost"
            onClick={onDismiss}
            aria-label="Dismiss error"
            title="Dismiss error"
          >
            <X size={13} /> Dismiss
          </button>
        ) : null}
      </div>
    </div>
  );
}
