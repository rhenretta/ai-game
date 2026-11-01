import { type ReactNode, useEffect, useId, useRef } from 'react';
import { X } from 'lucide-react';

export interface ModalDialogProps {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: ReactNode;
  wide?: boolean;
  children: ReactNode;
  footer?: ReactNode;
}

/**
 * Accessible Top-Layer Modal Dialog implementing `modern-web-guidance` (`light-dismiss-a-dialog`):
 * - Uses declarative `closedby="any"` + `.showModal()` for top-layer focus trapping & Esc handling
 * - Includes mandatory `'closedBy' in HTMLDialogElement.prototype` click bounding-rect fallback for Safari
 */
export function ModalDialog({
  open,
  onClose,
  title,
  subtitle,
  wide = false,
  children,
  footer,
}: ModalDialogProps) {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (open) {
      if (!dialog.open && typeof dialog.showModal === 'function') {
        dialog.showModal();
      }
    } else if (dialog.open && typeof dialog.close === 'function') {
      dialog.close();
    }
  }, [open]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    const handleNativeClose = () => {
      onClose();
    };
    dialog.addEventListener('close', handleNativeClose);

    // Fallback for browsers without <dialog closedby> support (e.g. Safari)
    let cleanupFallback: (() => void) | undefined;
    if (typeof HTMLDialogElement !== 'undefined' && !('closedBy' in HTMLDialogElement.prototype)) {
      const handleBackdropClick = (event: MouseEvent) => {
        if (event.target !== dialog) return;
        const rect = dialog.getBoundingClientRect();
        const isDialogContent =
          rect.top <= event.clientY &&
          event.clientY <= rect.top + rect.height &&
          rect.left <= event.clientX &&
          event.clientX <= rect.left + rect.width;
        if (isDialogContent) return;
        dialog.close();
      };
      dialog.addEventListener('click', handleBackdropClick);
      cleanupFallback = () => dialog.removeEventListener('click', handleBackdropClick);
    }

    return () => {
      dialog.removeEventListener('close', handleNativeClose);
      cleanupFallback?.();
    };
  }, [onClose]);

  return (
    <dialog
      ref={dialogRef}
      className={`studio-dialog${wide ? ' dialog-wide' : ''}`}
      aria-labelledby={titleId}
      {...({ closedby: 'any' } as Record<string, string>)}
    >
      {open ? (
        <div className="dialog-inner">
          <div className="dialog-header">
            <div style={{ minInlineSize: 0 }}>
              <h2
                id={titleId}
                style={{ margin: 0, fontSize: '1.02rem', fontWeight: 700, color: 'var(--text-primary)' }}
              >
                {title}
              </h2>
              {subtitle ? (
                <div style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginBlockStart: '2px' }}>
                  {subtitle}
                </div>
              ) : null}
            </div>
            <button type="button" className="btn btn-ghost" onClick={onClose} aria-label="Close dialog">
              <X size={16} />
            </button>
          </div>
          <div className="dialog-body">{children}</div>
          {footer ? <div className="dialog-footer">{footer}</div> : null}
        </div>
      ) : null}
    </dialog>
  );
}
