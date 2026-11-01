export interface DeliberationDisclosureProps {
  deliberation?: string | null;
  label?: string;
  defaultOpen?: boolean;
}

/**
 * Rule 4: Rationale-First Epistemic Observability block.
 * Renders pre-decision reasoning (`deliberation`) first before verdicts/scores/copy,
 * collapsible when `defaultOpen={false}` so card layouts stay dense and scannable.
 */
export function DeliberationDisclosure({
  deliberation,
  label = 'Epistemic Rationale',
  defaultOpen = false,
}: DeliberationDisclosureProps) {
  const text = (deliberation ?? '').trim();
  if (!text) return null;

  return (
    <details className="deliberation-box" open={defaultOpen}>
      <summary className="deliberation-summary">{label}</summary>
      <div style={{ marginBlockStart: '5px', whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{text}</div>
    </details>
  );
}
