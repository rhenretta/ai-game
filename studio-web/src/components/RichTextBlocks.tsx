import { Fragment } from 'react';
import {
  parseStructuredBlocks,
  tokenizeInlineFormatting,
  type InlineToken,
  type StructuredTextBlock,
} from '../domain/utterances';

export function InlineTokensView({ tokens }: { tokens: InlineToken[] }) {
  return (
    <>
      {tokens.map((tok, i) => {
        if (tok.type === 'code') {
          return (
            <code key={i} className="studio-inline-code">
              {tok.value}
            </code>
          );
        }
        if (tok.type === 'bold') {
          return <strong key={i}>{tok.value}</strong>;
        }
        return <Fragment key={i}>{tok.value}</Fragment>;
      })}
    </>
  );
}

export function FormattedInlineText({ text }: { text: string }) {
  if (!text) return null;
  return <InlineTokensView tokens={tokenizeInlineFormatting(text)} />;
}

export function StructuredBlocksView({
  blocks,
  className = 'studio-rich-blocks',
  listClassName = 'studio-rich-list',
}: {
  blocks: StructuredTextBlock[];
  className?: string;
  listClassName?: string;
}) {
  if (blocks.length === 0) return null;
  return (
    <div className={className}>
      {blocks.map((block, bIdx) => {
        if (block.type === 'ordered-list') {
          return (
            <ol key={bIdx} className={listClassName}>
              {block.items.map((itemTokens, liIdx) => (
                <li key={liIdx}>
                  <InlineTokensView tokens={itemTokens} />
                </li>
              ))}
            </ol>
          );
        }
        if (block.type === 'unordered-list') {
          return (
            <ul key={bIdx} className={listClassName}>
              {block.items.map((itemTokens, liIdx) => (
                <li key={liIdx}>
                  <InlineTokensView tokens={itemTokens} />
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={bIdx}>
            <InlineTokensView tokens={block.tokens} />
          </p>
        );
      })}
    </div>
  );
}

export function FormattedRichBlocks({
  text,
  className = 'studio-rich-blocks',
}: {
  text: string;
  className?: string;
}) {
  if (!text || !text.trim()) return null;
  const blocks = parseStructuredBlocks(text);
  return <StructuredBlocksView blocks={blocks} className={className} />;
}
