import { useState } from 'react';
import { ZoomIn, ZoomOut } from 'lucide-react';
import { ModalDialog } from './ModalDialog';
import { DeliberationDisclosure } from './DeliberationDisclosure';

export interface LightboxMedia {
  url: string;
  title: string;
  subtitle?: string;
  deliberation?: string;
  audioUrl?: string;
}

export interface MediaLightboxProps {
  media: LightboxMedia | null;
  onClose: () => void;
}

export function MediaLightbox({ media, onClose }: MediaLightboxProps) {
  const [pixelZoom, setPixelZoom] = useState(false);

  return (
    <ModalDialog
      open={Boolean(media)}
      onClose={() => {
        setPixelZoom(false);
        onClose();
      }}
      title={media?.title ?? 'Artifact Preview'}
      subtitle={media?.subtitle}
      wide
      footer={
        media ? (
          <>
            <button type="button" className="btn" onClick={() => setPixelZoom((z) => !z)}>
              {pixelZoom ? <ZoomOut size={14} /> : <ZoomIn size={14} />}
              {pixelZoom ? 'Fit to Window' : '1:1 Pixel Zoom'}
            </button>
            <a
              href={media.url}
              target="_blank"
              rel="noreferrer"
              className="btn btn-primary"
              style={{ textDecoration: 'none' }}
            >
              Open Raw Image
            </a>
          </>
        ) : undefined
      }
    >
      {media ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <DeliberationDisclosure deliberation={media.deliberation} defaultOpen />
          {media.audioUrl ? (
            <audio controls src={media.audioUrl} style={{ inlineSize: '100%' }} />
          ) : null}
          <div
            className="lightbox-pan-viewport"
            tabIndex={pixelZoom ? 0 : undefined}
            role={pixelZoom ? 'region' : undefined}
            aria-label={pixelZoom ? `Zoomed preview: ${media.title}` : undefined}
          >
            <img
              src={media.url}
              alt={media.title}
              style={
                pixelZoom
                  ? { maxInlineSize: 'none', imageRendering: 'pixelated' }
                  : { maxBlockSize: '72dvh', objectFit: 'contain' }
              }
            />
          </div>
        </div>
      ) : null}
    </ModalDialog>
  );
}
