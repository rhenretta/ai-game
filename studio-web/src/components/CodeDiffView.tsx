export interface CodeDiffViewProps {
  diffText: string;
  title?: string;
}

export function CodeDiffView({ diffText, title }: CodeDiffViewProps) {
  const lines = diffText.split('\n');
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {title ? (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--accent)', fontWeight: 600 }}>
          {title}
        </div>
      ) : null}
      <pre className="diff-block">
        {lines.map((line, idx) => {
          let cls = '';
          if (line.startsWith('@@')) cls = 'diff-line-hunk';
          else if (line.startsWith('+') && !line.startsWith('+++')) cls = 'diff-line-add';
          else if (line.startsWith('-') && !line.startsWith('---')) cls = 'diff-line-del';
          return (
            <span key={idx} className={cls || undefined}>
              {line}
              {'\n'}
            </span>
          );
        })}
      </pre>
    </div>
  );
}
