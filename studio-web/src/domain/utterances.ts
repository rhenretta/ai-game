import type { WritersRoomUtterance } from '../api/types';

export type BadgeTone = 'ok' | 'accent' | 'warn' | 'neutral';

export interface FormattedBibleBadge {
  label: string;
  tone: BadgeTone;
  title?: string;
}

export type InlineToken =
  | { type: 'text'; value: string }
  | { type: 'code'; value: string }
  | { type: 'bold'; value: string };

export type StructuredTextBlock =
  | { type: 'paragraph'; tokens: InlineToken[] }
  | { type: 'ordered-list'; items: InlineToken[][] }
  | { type: 'unordered-list'; items: InlineToken[][] };

export interface FormattedUtterance {
  turnKindLabel: string | null;
  distinctDeliberation: string | undefined;
  leadBlocks: StructuredTextBlock[];
  detailBlocks: StructuredTextBlock[];
  hasExpandableDetail: boolean;
  detailWordCount: number;
  badges: FormattedBibleBadge[];
}

interface ExtractedItemRef {
  itemId: string;
  title?: string;
}

/**
 * Extracts `item_id` (and optional `title`) pairs from serialized Python `WorkItem(...)` or stringified `WorkItem` objects
 * so historical transcripts never leak raw multi-kilobyte Python object reprs into the UI.
 */
export function extractSerializedWorkItemRefs(raw: string): ExtractedItemRef[] {
  const refs: ExtractedItemRef[] = [];
  const seen = new Set<string>();
  const marker = 'item_id=';
  let searchFrom = 0;

  while (searchFrom < raw.length) {
    const idx = raw.indexOf(marker, searchFrom);
    if (idx < 0) break;
    const quotePos = idx + marker.length;
    const quoteChar = raw[quotePos];
    if (quoteChar !== "'" && quoteChar !== '"') {
      searchFrom = quotePos + 1;
      continue;
    }
    const endQuote = raw.indexOf(quoteChar, quotePos + 1);
    if (endQuote < 0) break;
    const itemId = raw.slice(quotePos + 1, endQuote).trim();

    let title: string | undefined;
    const nextItemIdx = raw.indexOf(marker, endQuote + 1);
    const windowEnd = nextItemIdx >= 0 ? nextItemIdx : Math.min(raw.length, endQuote + 320);
    const windowSlice = raw.slice(endQuote + 1, windowEnd);
    const titleMarker = 'title=';
    const tIdx = windowSlice.indexOf(titleMarker);
    if (tIdx >= 0) {
      const tQuotePos = tIdx + titleMarker.length;
      const tQuote = windowSlice[tQuotePos];
      if (tQuote === "'" || tQuote === '"') {
        let cursor = tQuotePos + 1;
        while (cursor < windowSlice.length) {
          if (windowSlice[cursor] === tQuote && windowSlice[cursor - 1] !== '\\') {
            title = windowSlice.slice(tQuotePos + 1, cursor).replace(/\\'/g, "'").replace(/\\"/g, '"').trim();
            break;
          }
          cursor += 1;
        }
      }
    }

    if (itemId && !seen.has(itemId)) {
      seen.add(itemId);
      refs.push({ itemId, title });
    }
    searchFrom = endQuote + 1;
  }

  return refs;
}

function extractBracketedQuotedIds(raw: string): string[] {
  const openIdx = raw.indexOf('[');
  const closeIdx = raw.lastIndexOf(']');
  if (openIdx < 0 || closeIdx <= openIdx) return [];
  const inner = raw.slice(openIdx + 1, closeIdx);
  const ids: string[] = [];
  let i = 0;
  while (i < inner.length) {
    const ch = inner[i];
    if (ch === "'" || ch === '"') {
      const end = inner.indexOf(ch, i + 1);
      if (end < 0) break;
      const candidate = inner.slice(i + 1, end).trim();
      if (candidate && !candidate.includes(' ')) {
        ids.push(candidate);
      }
      i = end + 1;
    } else {
      i += 1;
    }
  }
  return ids;
}

export function normalizeUtteranceBadges(rawBadges: unknown): FormattedBibleBadge[] {
  if (!Array.isArray(rawBadges)) return [];
  const out: FormattedBibleBadge[] = [];
  const seenLabels = new Set<string>();

  const pushBadge = (badge: FormattedBibleBadge) => {
    const key = `${badge.tone}:${badge.label}`;
    if (seenLabels.has(key)) return;
    seenLabels.add(key);
    out.push(badge);
  };

  for (const entry of rawBadges) {
    const raw = String(entry ?? '').trim();
    if (!raw) continue;

    if (raw.includes('WorkItem(') || raw.includes('item_id=')) {
      const refs = extractSerializedWorkItemRefs(raw);
      if (refs.length > 0) {
        for (const ref of refs) {
          pushBadge({
            label: `Shaped: ${ref.itemId}`,
            tone: 'accent',
            title: ref.title ? `${ref.itemId} — ${ref.title}` : ref.itemId,
          });
        }
        continue;
      }
    }

    if (raw.startsWith('Shaped/split ')) {
      const ids = extractBracketedQuotedIds(raw);
      if (ids.length > 0) {
        for (const id of ids) {
          pushBadge({ label: `Shaped: ${id}`, tone: 'accent', title: raw });
        }
        continue;
      }
    }

    if (raw.startsWith('Raised ')) {
      const ids = extractBracketedQuotedIds(raw);
      if (ids.length > 0) {
        for (const id of ids) {
          pushBadge({ label: `+ ${id}`, tone: 'ok', title: raw });
        }
        continue;
      }
    }

    if (raw.startsWith('backlog:+')) {
      const id = raw.slice('backlog:+'.length).trim();
      if (id) {
        pushBadge({ label: `+ ${id}`, tone: 'ok' });
        continue;
      }
    }

    if (raw.startsWith('review:')) {
      const parts = raw.split(':');
      const itemId = parts[1] ?? 'item';
      const verdict = parts[2] ?? '';
      if (verdict === 'approve') {
        pushBadge({ label: `Approved: ${itemId}`, tone: 'ok' });
      } else {
        pushBadge({ label: `Revise: ${itemId}`, tone: 'warn' });
      }
      continue;
    }

    if (raw.startsWith('updated:')) {
      const field = raw.slice('updated:'.length).replace(/_/g, ' ').trim();
      pushBadge({ label: `Updated ${field}`, tone: 'neutral' });
      continue;
    }

    if (raw.startsWith('next_slice_strategy:')) {
      const sid = raw.slice('next_slice_strategy:'.length).trim();
      pushBadge({ label: `Strategy: ${sid}`, tone: 'accent' });
      continue;
    }

    if (raw.startsWith('slice_reflection:')) {
      const scoreIdx = raw.indexOf(':score_');
      if (scoreIdx >= 0) {
        const score = raw.slice(scoreIdx + ':score_'.length).trim();
        pushBadge({ label: `Quality ${score}`, tone: 'accent', title: raw });
        continue;
      }
    }

    if (raw.startsWith('slice_reflection_capture:')) {
      const sid = raw.slice('slice_reflection_capture:'.length).trim();
      pushBadge({ label: `State Capture: ${sid}`, tone: 'neutral' });
      continue;
    }

    if (raw.startsWith('Strategized ') && raw.includes('[')) {
      const ids = extractBracketedQuotedIds(raw);
      if (ids.length > 0) {
        for (const id of ids) {
          pushBadge({ label: `Slice: ${id}`, tone: 'accent' });
        }
        continue;
      }
    }

    // Never allow a single badge string to stretch across multiple lines
    if (raw.length > 68) {
      pushBadge({
        label: `${raw.slice(0, 65).trimEnd()}…`,
        tone: raw.startsWith('Director Steering Challenge') ? 'warn' : 'neutral',
        title: raw,
      });
    } else {
      pushBadge({
        label: raw,
        tone:
          raw.includes('Executive Producer') || raw.includes('Blind Playtester')
            ? 'warn'
            : raw.includes('Locked Living Game Bible')
              ? 'ok'
              : 'neutral',
      });
    }
  }

  return out;
}

function sanitizeSerializedWorkItemsInDialogue(rawText: string): string {
  const marker = 'Shaped/refined board task(s):';
  const mIdx = rawText.indexOf(marker);
  if (mIdx < 0) return rawText;

  const after = rawText.slice(mIdx + marker.length);
  if (!after.includes('item_id=') && !after.includes('WorkItem(')) {
    return rawText;
  }

  const refs = extractSerializedWorkItemRefs(after);
  const before = rawText.slice(0, mIdx).trimEnd();
  if (refs.length === 0) {
    return before;
  }

  const cleanRefs = refs
    .map((r) => (r.title ? `\`${r.itemId}\` (${r.title})` : `\`${r.itemId}\``))
    .join(', ');
  return `${before}\n\nShaped/refined board task(s): ${cleanRefs}.`.trim();
}

export function tokenizeInlineFormatting(text: string): InlineToken[] {
  const tokens: InlineToken[] = [];
  let cursor = 0;

  const appendText = (val: string) => {
    if (!val) return;
    const prev = tokens[tokens.length - 1];
    if (prev && prev.type === 'text') {
      prev.value += val;
    } else {
      tokens.push({ type: 'text', value: val });
    }
  };

  while (cursor < text.length) {
    if (text[cursor] === '`') {
      const endTick = text.indexOf('`', cursor + 1);
      if (endTick > cursor + 1 && endTick - cursor <= 140) {
        tokens.push({ type: 'code', value: text.slice(cursor + 1, endTick) });
        cursor = endTick + 1;
        continue;
      }
    }

    if (text[cursor] === '*' && text[cursor + 1] === '*') {
      const endBold = text.indexOf('**', cursor + 2);
      if (endBold > cursor + 2 && endBold - cursor <= 200) {
        tokens.push({ type: 'bold', value: text.slice(cursor + 2, endBold) });
        cursor = endBold + 2;
        continue;
      }
    }

    const nextTick = text.indexOf('`', cursor + 1);
    const nextBold = text.indexOf('**', cursor + 1);
    let nextSpecial = text.length;
    if (nextTick >= 0 && nextTick < nextSpecial) nextSpecial = nextTick;
    if (nextBold >= 0 && nextBold < nextSpecial) nextSpecial = nextBold;

    appendText(text.slice(cursor, nextSpecial));
    cursor = nextSpecial;
  }

  return tokens;
}

function buildBacktickMask(text: string): Uint8Array {
  const mask = new Uint8Array(text.length);
  let openIdx = -1;
  for (let i = 0; i < text.length; i += 1) {
    if (text[i] === '`') {
      if (openIdx < 0) {
        openIdx = i;
      } else if (i - openIdx <= 180) {
        for (let k = openIdx; k <= i; k += 1) {
          mask[k] = 1;
        }
        openIdx = -1;
      } else {
        openIdx = i;
      }
    }
  }
  return mask;
}

function hasValidMarkerBoundaryBefore(
  text: string,
  idx: number,
  num: number,
  delim: '.' | ')',
): boolean {
  if (idx === 0) return num === 1;
  const prevChar = text[idx - 1];
  if (!/\s/.test(prevChar)) return false;

  // Walk backward across horizontal whitespace to inspect whether marker starts a new line
  // or follows sentence/clause punctuation rather than a mid-sentence word like `at least 1.`
  let scan = idx - 1;
  while (scan >= 0 && (text[scan] === ' ' || text[scan] === '\t')) {
    scan -= 1;
  }
  if (scan < 0 || text[scan] === '\n') {
    return true;
  }

  const prevNonWs = text[scan];
  if (delim === '.') {
    if (num === 1) {
      return /[.:;!?—\-)\]]/.test(prevNonWs);
    }
    return /[.,;:!?)\]`'"—\-]/.test(prevNonWs);
  }

  // For `N)` markers, ensure we are not closing an inline parenthetical like `(see step 2) next`
  const lineStart = text.lastIndexOf('\n', idx - 1) + 1;
  const recentWindow = text.slice(Math.max(lineStart, idx - 28), idx);
  const lastOpenParen = recentWindow.lastIndexOf('(');
  const lastCloseParen = recentWindow.lastIndexOf(')');
  if (lastOpenParen >= 0 && lastOpenParen > lastCloseParen) {
    return false;
  }
  return true;
}

function findNextSequentialMarker(
  text: string,
  mask: Uint8Array,
  num: number,
  delim: '.' | ')',
  searchFrom: number,
): { markerStart: number; contentStart: number } | null {
  const token = `${num}${delim}`;
  let pos = searchFrom;
  while (pos < text.length) {
    const idx = text.indexOf(token, pos);
    if (idx < 0) return null;
    const afterIdx = idx + token.length;
    if (
      mask[idx] === 0 &&
      afterIdx < text.length &&
      /\s/.test(text[afterIdx]) &&
      hasValidMarkerBoundaryBefore(text, idx, num, delim)
    ) {
      let contentStart = afterIdx;
      while (contentStart < text.length && (text[contentStart] === ' ' || text[contentStart] === '\t')) {
        contentStart += 1;
      }
      if (contentStart < text.length && text[contentStart] !== '\n') {
        return { markerStart: idx, contentStart };
      }
    }
    pos = idx + 1;
  }
  return null;
}

/**
 * Splits inline or multi-line `1. ... 2. ... 3. ...` or `1) ... 2) ... 3) ...` enumerations
 * into a clean `{ preamble, items }` structure.
 *
 * Strictly enforces monotonic sequence (`1 -> 2 -> 3 ...`), skips backticked code spans,
 * and requires sentence/clause or newline boundaries before `N.` markers so prose ending in
 * numbers (e.g. `increases from 0 to at least 1. 2. Ensure...`) never splits mid-sentence.
 */
export function splitSequentialNumberedEnumeration(paragraph: string): {
  preamble: string;
  items: string[];
} | null {
  const mask = buildBacktickMask(paragraph);

  const candidates: Array<{ markerStart: number; contentStart: number; delim: '.' | ')' }> = [];
  for (const delim of ['.', ')'] as const) {
    let searchFrom = 0;
    while (searchFrom < paragraph.length) {
      const m1 = findNextSequentialMarker(paragraph, mask, 1, delim, searchFrom);
      if (!m1) break;
      candidates.push({ ...m1, delim });
      searchFrom = m1.markerStart + 2;
    }
  }
  candidates.sort((a, b) => a.markerStart - b.markerStart);

  for (const first of candidates) {
    const second = findNextSequentialMarker(
      paragraph,
      mask,
      2,
      first.delim,
      first.contentStart,
    );
    if (!second) continue;

    const firstBody = paragraph
      .slice(first.contentStart, second.markerStart)
      .trim()
      .replace(/;\s*$/, '');
    if (!firstBody) continue;

    const preamble = paragraph.slice(0, first.markerStart).trim();
    const items: string[] = [firstBody];

    let curContentStart = second.contentStart;
    let nextExpected = 3;
    while (curContentStart < paragraph.length) {
      const nxt = findNextSequentialMarker(
        paragraph,
        mask,
        nextExpected,
        first.delim,
        curContentStart,
      );
      if (nxt) {
        const segBody = paragraph
          .slice(curContentStart, nxt.markerStart)
          .trim()
          .replace(/;\s*$/, '');
        if (segBody) {
          items.push(segBody);
          curContentStart = nxt.contentStart;
          nextExpected += 1;
          continue;
        }
      }
      const tailBody = paragraph.slice(curContentStart).trim();
      if (tailBody) {
        items.push(tailBody);
      }
      break;
    }

    if (items.length >= 2) {
      return { preamble, items };
    }
  }

  return null;
}

export function stripLeadingSingleListMarker(line: string): string {
  return line.replace(/^(?:1[.)]|[-*•])\s+/, '').trim();
}

export function parseStructuredBlocks(rawText: string): StructuredTextBlock[] {
  const normalized = rawText.replace(/\r\n/g, '\n').trim();
  if (!normalized) return [];

  const blocks: StructuredTextBlock[] = [];
  const rawParagraphs = normalized.split(/\n{2,}/);

  for (const rawPara of rawParagraphs) {
    const lines = rawPara
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length === 0) continue;

    // Check for multi-line unordered bullet list (optionally preceded by a 1-line preamble)
    const firstBulletIdx = lines.findIndex((l) => /^[-*•]\s+/.test(l));
    if (
      firstBulletIdx >= 0 &&
      lines.length - firstBulletIdx >= 2 &&
      lines.slice(firstBulletIdx).every((l) => /^[-*•]\s+/.test(l))
    ) {
      if (firstBulletIdx > 0) {
        blocks.push({
          type: 'paragraph',
          tokens: tokenizeInlineFormatting(lines.slice(0, firstBulletIdx).join(' ')),
        });
      }
      blocks.push({
        type: 'unordered-list',
        items: lines
          .slice(firstBulletIdx)
          .map((l) => tokenizeInlineFormatting(l.replace(/^[-*•]\s+/, ''))),
      });
      continue;
    }

    // Check for sequential numbered enumeration (supports both multi-line and inline `1. ... 2. ...` / `1) ... 2) ...`)
    const multiLineJoined = lines.join('\n');
    const seqEnum = splitSequentialNumberedEnumeration(multiLineJoined);
    if (seqEnum) {
      if (seqEnum.preamble) {
        blocks.push({
          type: 'paragraph',
          tokens: tokenizeInlineFormatting(seqEnum.preamble.replace(/\s*\n\s*/g, ' ')),
        });
      }
      blocks.push({
        type: 'ordered-list',
        items: seqEnum.items.map((it) =>
          tokenizeInlineFormatting(it.replace(/\s*\n\s*/g, ' ')),
        ),
      });
      continue;
    }

    const joined = stripLeadingSingleListMarker(lines.join(' '));
    if (joined) {
      blocks.push({
        type: 'paragraph',
        tokens: tokenizeInlineFormatting(joined),
      });
    }
  }

  return blocks;
}

function findLeadSummarySplitIndex(text: string): number {
  const maxLeadChars = 280;
  const minLeadChars = 95;
  if (text.length <= 320) return -1;

  // Prefer natural paragraph break if one appears early
  const paraBreak = text.indexOf('\n\n');
  if (paraBreak >= 60 && paraBreak <= 310) {
    return paraBreak;
  }

  // Prefer breaking right before an inline `1) ` enumeration
  const enumStart = text.search(/\s1\)\s+/);
  if (enumStart >= 60 && enumStart <= 310) {
    return enumStart;
  }

  let inBackticks = false;
  let firstGoodBoundary = -1;
  let secondBoundary = -1;

  for (let i = 0; i < Math.min(text.length - 2, maxLeadChars + 40); i += 1) {
    const ch = text[i];
    if (ch === '`') {
      inBackticks = !inBackticks;
      continue;
    }
    if (inBackticks) continue;

    if ((ch === '.' || ch === '!' || ch === '?') && /\s/.test(text[i + 1] ?? '')) {
      // Skip decimals like `0.68` or abbreviations like `e.g.`
      const prevChar = i > 0 ? text[i - 1] : '';
      const nextChar = text[i + 2] ?? '';
      if (/\d/.test(prevChar) && /\d/.test(text[i + 1] ?? '')) continue;
      if (nextChar && /[a-z]/.test(nextChar)) continue;

      const splitAt = i + 1;
      if (splitAt >= minLeadChars && firstGoodBoundary < 0) {
        firstGoodBoundary = splitAt;
      } else if (splitAt > firstGoodBoundary && splitAt <= maxLeadChars) {
        secondBoundary = splitAt;
        break;
      }
    }
  }

  if (secondBoundary > 0) return secondBoundary;
  if (firstGoodBoundary > 0) return firstGoodBoundary;

  // Fallback: clean word boundary around 240 chars outside backticks
  const spaceIdx = text.lastIndexOf(' ', 240);
  return spaceIdx >= 100 ? spaceIdx : -1;
}

function inferTurnKindLabel(u: WritersRoomUtterance): string | null {
  if (u.speaker_role === 'ExecutiveProducer' || u.speaker_role === 'UserSteering') {
    return 'Producer Direction';
  }
  const uid = u.utterance_id ?? '';
  if (uid.includes('_task_shaping')) return 'Task Shaping';
  if (uid.includes('_review')) return 'Proof Review';
  if (uid.includes('_reflection')) return 'Slice Reflection';
  if (uid.includes('_strategy')) return 'Slice Strategy';
  if (uid.includes('_blind_')) return 'Playtest Debrief';
  return null;
}

export function formatWritersRoomUtterance(u: WritersRoomUtterance): FormattedUtterance {
  const rawSpoken = (u.spoken_dialogue || u.message || '').trim();
  const rawDelib = (u.deliberation || '').trim();
  const cleanedSpoken = sanitizeSerializedWorkItemsInDialogue(rawSpoken);

  // Deduplicate deliberation when `_task_shaping` copied `deliberation` verbatim as the prefix of `spoken_dialogue`
  const isDuplicateDelib =
    Boolean(rawDelib) &&
    (cleanedSpoken === rawDelib || cleanedSpoken.startsWith(rawDelib.slice(0, Math.min(160, rawDelib.length))));

  const distinctDeliberation = rawDelib && !isDuplicateDelib ? rawDelib : undefined;
  const isProducer = u.speaker_role === 'ExecutiveProducer' || u.speaker_role === 'UserSteering';

  let leadText = cleanedSpoken;
  let detailText = '';

  if (!isProducer) {
    const splitIdx = findLeadSummarySplitIndex(cleanedSpoken);
    if (splitIdx > 0) {
      leadText = cleanedSpoken.slice(0, splitIdx).trim();
      detailText = cleanedSpoken.slice(splitIdx).trim();
    }
  }

  // Only fold if the remaining detail is meaningful (not just a 3-word tail)
  if (detailText.length > 0 && detailText.length < 75) {
    leadText = cleanedSpoken;
    detailText = '';
  }

  const detailWordCount = detailText ? detailText.split(/\s+/).filter(Boolean).length : 0;

  return {
    turnKindLabel: inferTurnKindLabel(u),
    distinctDeliberation,
    leadBlocks: parseStructuredBlocks(leadText),
    detailBlocks: detailText ? parseStructuredBlocks(detailText) : [],
    hasExpandableDetail: detailText.length > 0,
    detailWordCount,
    badges: normalizeUtteranceBadges(u.bible_updates_summary),
  };
}
