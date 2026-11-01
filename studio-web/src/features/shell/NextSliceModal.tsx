import { type FormEvent, useState } from 'react';
import { Rocket } from 'lucide-react';
import { ModalDialog } from '../../components/ModalDialog';

export interface NextSliceModalProps {
  open: boolean;
  onClose: () => void;
  gameTitle: string;
  currentSliceNumber: number;
  onSubmit: (feedback: string) => Promise<void>;
  busy: boolean;
}

export function NextSliceModal({
  open,
  onClose,
  gameTitle,
  currentSliceNumber,
  onSubmit,
  busy,
}: NextSliceModalProps) {
  const [feedback, setFeedback] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    await onSubmit(feedback.trim());
    setFeedback('');
    onClose();
  };

  return (
    <ModalDialog
      open={open}
      onClose={onClose}
      title={`Iterate / Resume Build — ${gameTitle || 'Selected Game'}`}
      subtitle={`Advances production beyond Slice #${currentSliceNumber} (or resumes any incomplete board items).`}
      footer={
        <>
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" form="next-slice-form" className="btn btn-primary" disabled={busy}>
            <Rocket size={14} />
            {busy ? 'Dispatching...' : 'Dispatch Council & Workers'}
          </button>
        </>
      }
    >
      <form id="next-slice-form" onSubmit={(e) => void handleSubmit(e)} style={{ display: 'grid', gap: '12px' }}>
        <label style={{ display: 'grid', gap: '5px' }}>
          <span style={{ fontWeight: 600, fontSize: '0.87rem' }}>
            Executive Producer Notes for Next Vertical Slice (Optional)
          </span>
          <textarea
            className="textarea-control"
            rows={4}
            placeholder="Leave blank for autonomous Council slice reflection & roadmap progression, or steer what this iteration should focus on..."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
          />
        </label>
      </form>
    </ModalDialog>
  );
}
