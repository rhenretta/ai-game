import { type FormEvent, useState } from 'react';
import { Sparkles } from 'lucide-react';
import { ModalDialog } from '../../components/ModalDialog';

export interface NewGameModalProps {
  open: boolean;
  onClose: () => void;
  defaultPauseForGreenlight: boolean;
  onSubmit: (params: { creativePrompt: string; pauseForGreenlight: boolean }) => Promise<void>;
  busy: boolean;
}

export function NewGameModal({
  open,
  onClose,
  defaultPauseForGreenlight,
  onSubmit,
  busy,
}: NewGameModalProps) {
  const [prompt, setPrompt] = useState('');
  const [pauseForGreenlight, setPauseForGreenlight] = useState(defaultPauseForGreenlight);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    await onSubmit({ creativePrompt: prompt.trim(), pauseForGreenlight });
    setPrompt('');
    onClose();
  };

  return (
    <ModalDialog
      open={open}
      onClose={onClose}
      title="Commission New Game from Scratch"
      subtitle="Convenes the 10-specialist Council grounded in your Steam Player DNA. Existing cartridges remain preserved."
      footer={
        <>
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            type="submit"
            form="new-game-form"
            className="btn btn-primary"
            disabled={busy}
          >
            <Sparkles size={14} />
            {busy ? 'Dispatching...' : 'Start Genesis Build'}
          </button>
        </>
      }
    >
      <form id="new-game-form" onSubmit={(e) => void handleSubmit(e)} style={{ display: 'grid', gap: '12px' }}>
        <label style={{ display: 'grid', gap: '5px' }}>
          <span style={{ fontWeight: 600, fontSize: '0.87rem' }}>
            Optional Creative Seed / High-Concept Direction
          </span>
          <textarea
            className="textarea-control"
            rows={4}
            placeholder="Leave blank for pure Player-DNA autonomous ideation, or describe a mood, mechanic, or premise..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={pauseForGreenlight}
            onChange={(e) => setPauseForGreenlight(e.target.checked)}
          />
          <span style={{ fontSize: '0.87rem', color: 'var(--text-secondary)' }}>
            Pause at <strong>Greenlight Gate</strong> after Game Bible locks before code synthesis starts
          </span>
        </label>
      </form>
    </ModalDialog>
  );
}
