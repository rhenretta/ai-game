import { resolveMediaUrl } from '../api/client';
import type { ChronologyProofArtifact, ChronologyStepLogEntry } from '../api/types';

export interface PromptSubsection {
  key: string;
  title: string;
  meta: string;
  content: string;
}

export interface PromptSection {
  key: string;
  index: number;
  title: string;
  lineCount: number;
  charCount: number;
  content: string;
  preamble: string;
  subs: PromptSubsection[];
}

export interface NormalizedProofItem {
  filename: string;
  path: string;
  url: string;
  isAudio: boolean;
  isWaveform: boolean;
}

export interface NormalizedStepEntry {
  index: number;
  stepType: string;
  label: string;
  badgeClass: string;
  specialistAgent: string;
  message: string;
  timestampShort: string;
  elapsedLabel: string;
  detailsEntries: Array<{ key: string; value: string }>;
}

export function coercePromptString(raw: unknown): string {
  if (!raw) return '';
  if (typeof raw === 'string') return raw;
  try {
    return JSON.stringify(raw, null, 2);
  } catch {
    return String(raw);
  }
}

export function countLinesFast(text: string): number {
  if (!text) return 0;
  let count = 1;
  for (let i = 0; i < text.length; i += 1) {
    if (text.charCodeAt(i) === 10) count += 1;
  }
  return count;
}

function parseNestedPromptSubsections(sectionLines: string[]): {
  preamble: string;
  subs: PromptSubsection[];
} {
  const joinedLen = sectionLines.reduce((acc, l) => acc + l.length + 1, 0);

  // 1. Curated Studio Context chunks: [namespace:id] (category) Title
  const chunkRe = /^\[([a-z0-9_.-]+:[^\]]+)\]\s*(?:\(([^)]+)\))?\s*(.*)$/i;
  const chunkHits: Array<{ idx: number; id: string; cat: string; rest: string }> = [];
  for (let i = 0; i < sectionLines.length; i += 1) {
    const m = sectionLines[i].match(chunkRe);
    if (m) {
      chunkHits.push({ idx: i, id: m[1], cat: m[2] || 'context', rest: m[3] || '' });
    }
  }
  if (chunkHits.length >= 2) {
    const preamble = sectionLines.slice(0, chunkHits[0].idx).join('\n').trim();
    const subs = chunkHits.map((hit, hIdx) => {
      const endIdx = hIdx + 1 < chunkHits.length ? chunkHits[hIdx + 1].idx : sectionLines.length;
      const bodyLines = sectionLines.slice(hit.idx, endIdx);
      const bodyText = bodyLines.join('\n');
      const cleanRest = hit.rest.trim();
      const shortRest = cleanRest.length > 90 ? `${cleanRest.slice(0, 90)}…` : cleanRest;
      return {
        key: `chunk_${hIdx}_${hit.id}`,
        title: `[${hit.id}]${shortRest ? ` ${shortRest}` : ''}`,
        meta: `${hit.cat} · ${bodyLines.length} lines · ${bodyText.length.toLocaleString()} chars`,
        content: bodyText,
      };
    });
    return { preamble, subs };
  }

  // 2. Existing Cartridge Code files: --- Existing <filename> ... --- or === FILE: ... ===
  const fileRe = /^(?:---\s*(Existing\s+.+?)\s*---|===\s*(FILE:\s*.+?)\s*===)\s*$/;
  const fileHits: Array<{ idx: number; label: string }> = [];
  for (let i = 0; i < sectionLines.length; i += 1) {
    const m = sectionLines[i].match(fileRe);
    if (m) {
      fileHits.push({ idx: i, label: (m[1] || m[2] || '').trim() });
    }
  }
  if (fileHits.length >= 1) {
    const preamble = sectionLines.slice(0, fileHits[0].idx).join('\n').trim();
    const subs = fileHits.map((hit, fIdx) => {
      const endIdx = fIdx + 1 < fileHits.length ? fileHits[fIdx + 1].idx : sectionLines.length;
      const bodyLines = sectionLines.slice(hit.idx, endIdx);
      const bodyText = bodyLines.join('\n');
      return {
        key: `file_${fIdx}`,
        title: hit.label,
        meta: `${bodyLines.length} lines · ${bodyText.length.toLocaleString()} chars`,
        content: bodyText,
      };
    });
    return { preamble, subs };
  }

  // 3. Large numbered specification or instruction blocks (1. ..., 2. ..., etc.)
  if (joinedLen > 2200) {
    const numRe = /^(\d+)\.\s+([^\n]+)/;
    const numHits: Array<{ idx: number; num: string; label: string }> = [];
    for (let i = 0; i < sectionLines.length; i += 1) {
      const m = sectionLines[i].match(numRe);
      if (m) {
        const rawRest = m[2].trim();
        const colonIdx = rawRest.indexOf(':');
        const label =
          colonIdx > 2 && colonIdx <= 95
            ? rawRest.slice(0, colonIdx).trim()
            : `${rawRest.slice(0, 88).trim()}${rawRest.length > 88 ? '…' : ''}`;
        numHits.push({ idx: i, num: m[1], label });
      }
    }
    if (numHits.length >= 3) {
      const preamble = sectionLines.slice(0, numHits[0].idx).join('\n').trim();
      const subs = numHits.map((hit, nIdx) => {
        const endIdx = nIdx + 1 < numHits.length ? numHits[nIdx + 1].idx : sectionLines.length;
        const bodyLines = sectionLines.slice(hit.idx, endIdx);
        const bodyText = bodyLines.join('\n');
        return {
          key: `rule_${hit.num}_${nIdx}`,
          title: `§${hit.num}. ${hit.label}`,
          meta: `${bodyLines.length} lines · ${bodyText.length.toLocaleString()} chars`,
          content: bodyText,
        };
      });
      return { preamble, subs };
    }
  }

  return { preamble: '', subs: [] };
}

/**
 * Splits large raw prompts and curated context payloads into collapsible structural sections
 * (`=== BANNER ===`, `SECTION HEADER:`, embedded file snapshots, and `[namespace:id]` context chunks)
 * while preserving 100% of the untruncated text.
 */
export function parsePromptIntoSections(rawText: string): PromptSection[] {
  const text = String(rawText || '');
  if (!text) return [];

  const lines = text.split('\n');
  type RawSectionAccum = { title: string; lines: string[] };
  const rawSections: RawSectionAccum[] = [];
  let current: RawSectionAccum | null = null;
  let inBannerBlock = false;
  let inEmbeddedFiles = false;
  let forceSplitNext = false;

  const pushNewSection = (title: string, firstLine: string): RawSectionAccum => {
    const nextSec: RawSectionAccum = { title: String(title || 'Section').trim(), lines: [firstLine] };
    rawSections.push(nextSec);
    return nextSec;
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();
    const atColZero = line.length > 0 && line[0] !== ' ' && line[0] !== '\t';

    if (atColZero) {
      if (/^===\s*END\b/i.test(trimmed)) {
        if (!current) {
          current = pushNewSection('Context Footer', line);
        } else {
          current.lines.push(line);
        }
        inBannerBlock = false;
        forceSplitNext = true;
        continue;
      }

      const bannerMatch = !inBannerBlock ? trimmed.match(/^===\s*(.+?)\s*===$/) : null;
      if (bannerMatch && !/^FILE:/i.test(bannerMatch[1])) {
        const bTitle = bannerMatch[1].trim();
        inBannerBlock = /CURATED STUDIO CONTEXT/i.test(bTitle);
        inEmbeddedFiles = false;
        forceSplitNext = false;
        current = pushNewSection(bTitle, line);
        continue;
      }

      if (!inBannerBlock) {
        if (/^You are the\b/i.test(trimmed)) {
          inEmbeddedFiles = false;
          forceSplitNext = false;
          const preview = trimmed.length > 72 ? `${trimmed.slice(0, 72)}…` : trimmed;
          current = pushNewSection(`Role Directive — ${preview}`, line);
          continue;
        }

        const upperHeaderMatch = trimmed.match(/^([A-Z][A-Z0-9 _/&().,'"#—+-]{3,}):(?:\s+(.*))?$/);
        if (upperHeaderMatch) {
          const hdrName = upperHeaderMatch[1].trim();
          const breaksFileDump = /^(CRITICAL|PRIOR|PREVIOUS|COLLECTED|ACTION|SELF-INQUIRY|RUNTIME|OUTPUT|RETURN)/i.test(
            hdrName,
          );
          if (!inEmbeddedFiles || breaksFileDump) {
            inEmbeddedFiles = /^(EXISTING CARTRIDGE CODE|CARTRIDGE SYMBOLS)/i.test(hdrName);
            forceSplitNext = false;
            current = pushNewSection(hdrName, line);
            continue;
          }
        }

        if (forceSplitNext && trimmed.length > 0) {
          forceSplitNext = false;
          const preview = trimmed.length > 64 ? `${trimmed.slice(0, 64)}…` : trimmed;
          current = pushNewSection(`Continuation (${preview})`, line);
          continue;
        }
      }
    }

    if (!current) {
      const firstLabel = trimmed
        ? `${trimmed.slice(0, 64)}${trimmed.length > 64 ? '…' : ''}`
        : 'Preamble';
      current = pushNewSection(firstLabel, line);
    } else {
      current.lines.push(line);
    }
  }

  // Merge consecutive 1-line metadata headers so the section list stays clean
  const merged: Array<{ title: string; firstTitle: string; isCompactMeta: boolean; lines: string[] }> = [];
  for (const sec of rawSections) {
    const nonEmptyCount = sec.lines.filter((l) => l.trim().length > 0).length;
    const totalChars = sec.lines.reduce((acc, l) => acc + l.length + 1, 0);
    const prev = merged.length > 0 ? merged[merged.length - 1] : null;
    if (prev && prev.isCompactMeta && nonEmptyCount <= 1 && totalChars < 260) {
      prev.lines.push(...sec.lines);
      prev.title = `${prev.firstTitle} + Metadata`;
    } else {
      merged.push({
        title: sec.title,
        firstTitle: sec.title,
        isCompactMeta: nonEmptyCount <= 1 && totalChars < 260,
        lines: sec.lines,
      });
    }
  }

  return merged.map((sec, idx) => {
    const content = sec.lines.join('\n');
    const { preamble, subs } = parseNestedPromptSubsections(sec.lines);
    return {
      key: `sec_${idx}`,
      index: idx + 1,
      title: sec.title,
      lineCount: sec.lines.length,
      charCount: content.length,
      content,
      preamble,
      subs,
    };
  });
}

function extractFilenameFromUrlOrPath(raw: string): string {
  const trimmed = raw.trim();
  const qIdx = trimmed.indexOf('?');
  if (qIdx >= 0) {
    try {
      const params = new URLSearchParams(trimmed.slice(qIdx + 1));
      const pathParam = params.get('path');
      if (pathParam) {
        return pathParam.split('/').pop() || pathParam;
      }
    } catch {
      // Fall through to pathname extraction
    }
  }
  return trimmed.split('?')[0].split('/').pop() || trimmed;
}

export function normalizeProofArtifacts(
  rawProofs: Array<ChronologyProofArtifact | string> | undefined,
  fallbackImages: string[] | undefined,
  cartridgeId: string,
): NormalizedProofItem[] {
  const source: Array<ChronologyProofArtifact | string> =
    rawProofs && rawProofs.length > 0 ? rawProofs : (fallbackImages ?? []);

  const out: NormalizedProofItem[] = [];
  for (const item of source) {
    if (!item) continue;
    if (typeof item === 'string') {
      const trimmed = item.trim();
      if (!trimmed) continue;
      const filename = extractFilenameFromUrlOrPath(trimmed);
      const lower = filename.toLowerCase();
      out.push({
        filename,
        path: trimmed,
        url: resolveMediaUrl(trimmed, cartridgeId),
        isAudio: lower.endsWith('.wav') || lower.endsWith('.mp3') || lower.endsWith('.ogg'),
        isWaveform: lower.startsWith('waveform_'),
      });
    } else if (typeof item === 'object') {
      const rawUrl = item.url || item.path || '';
      if (!rawUrl) continue;
      const filename = item.filename || extractFilenameFromUrlOrPath(rawUrl);
      const lower = filename.toLowerCase();
      out.push({
        filename,
        path: item.path || rawUrl,
        url: resolveMediaUrl(rawUrl, cartridgeId),
        isAudio: Boolean(
          item.is_audio || lower.endsWith('.wav') || lower.endsWith('.mp3') || lower.endsWith('.ogg'),
        ),
        isWaveform: lower.startsWith('waveform_'),
      });
    }
  }
  return out;
}

function classifyStepBadge(stepType: string): { label: string; badgeClass: string } {
  const s = stepType.toLowerCase();
  if (s.includes('error') || s.includes('fail')) return { label: 'ERROR', badgeClass: 'badge-err' };
  if (s.includes('test') || s.includes('verify')) return { label: 'VERIFY / TEST', badgeClass: 'badge-ok' };
  if (s.includes('inquiry') || s.includes('question')) return { label: 'SELF-INQUIRY', badgeClass: 'badge-warn' };
  if (s.includes('context') || s.includes('curat')) return { label: 'CURATING CONTEXT', badgeClass: 'badge-purple' };
  if (s.includes('plan')) return { label: 'PLANNING', badgeClass: 'badge-purple' };
  if (s.includes('render') || s.includes('scene') || s.includes('proof')) {
    return { label: 'RENDER / PROOF', badgeClass: 'badge-warn' };
  }
  if (s.includes('edit') || s.includes('patch') || s.includes('write')) {
    return { label: 'EDITING FILE', badgeClass: 'badge-accent' };
  }
  if (s.includes('synth') || s.includes('asset') || s.includes('generat')) {
    return { label: 'SYNTHESIZING', badgeClass: 'badge-accent' };
  }
  return { label: (stepType || 'STEP').replace(/_/g, ' ').toUpperCase(), badgeClass: '' };
}

export function normalizeStepLog(
  rawSteps: Array<ChronologyStepLogEntry | Record<string, unknown> | string> | undefined,
): NormalizedStepEntry[] {
  if (!Array.isArray(rawSteps) || rawSteps.length === 0) return [];

  let prevMs: number | null = null;
  return rawSteps.map((raw, idx) => {
    if (typeof raw === 'string') {
      return {
        index: idx + 1,
        stepType: 'step',
        label: 'STEP',
        badgeClass: '',
        specialistAgent: '',
        message: raw,
        timestampShort: '',
        elapsedLabel: '',
        detailsEntries: [],
      };
    }

    const obj = raw as Record<string, unknown>;
    const stepType = String(obj.step_type ?? 'step');
    const { label, badgeClass } = classifyStepBadge(stepType);
    const tsRaw = typeof obj.timestamp_utc === 'string' ? obj.timestamp_utc : '';
    let timestampShort = '';
    let elapsedLabel = '';

    if (tsRaw) {
      const timePart = tsRaw.split('T')[1];
      if (timePart) {
        timestampShort = timePart.replace('Z', '').slice(0, 8);
      }
      const parsedMs = Date.parse(tsRaw);
      if (!Number.isNaN(parsedMs)) {
        if (prevMs !== null && parsedMs >= prevMs) {
          const deltaSec = (parsedMs - prevMs) / 1000;
          if (deltaSec >= 0.1) {
            if (deltaSec >= 60) {
              const mins = Math.floor(deltaSec / 60);
              const secs = Math.round(deltaSec % 60);
              elapsedLabel = `+${mins}m ${String(secs).padStart(2, '0')}s`;
            } else {
              elapsedLabel = `+${deltaSec.toFixed(1)}s`;
            }
          }
        }
        prevMs = parsedMs;
      }
    }

    const detailsEntries: Array<{ key: string; value: string }> = [];
    if (obj.details && typeof obj.details === 'object' && !Array.isArray(obj.details)) {
      for (const [k, v] of Object.entries(obj.details as Record<string, unknown>)) {
        if (v === null || v === undefined || v === '') continue;
        if (Array.isArray(v)) {
          if (v.length > 0) detailsEntries.push({ key: k, value: v.map(String).join(', ') });
        } else if (typeof v === 'object') {
          detailsEntries.push({ key: k, value: JSON.stringify(v) });
        } else {
          detailsEntries.push({ key: k, value: String(v) });
        }
      }
    }

    return {
      index: idx + 1,
      stepType,
      label,
      badgeClass,
      specialistAgent: typeof obj.specialist_agent === 'string' ? obj.specialist_agent : '',
      message: typeof obj.message === 'string' ? obj.message : JSON.stringify(obj),
      timestampShort,
      elapsedLabel,
      detailsEntries,
    };
  });
}
