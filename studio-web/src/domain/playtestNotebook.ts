import type { PlaytestTurn } from '../api/types';

export interface PlaytestRulebookEntry {
  topic: string;
  behavior: string;
}

export interface PlaytestFieldNoteEntry {
  topic: string;
  observation: string;
}

export interface PlaytestQuestionEntry {
  key: string;
  question: string;
  status: 'open' | 'answered';
  answer: string;
  askedTurn?: number;
  answeredTurn?: number;
}

export interface PlaytestBugEntry {
  key: string;
  severity: string;
  category: string;
  summary: string;
  details: string;
  turnIndex?: number;
  screenshotName?: string;
}

export interface FormattedMotorAction {
  summary: string;
  chips: string[];
  mouseButtonAction: string;
  hasExplicitMouse: boolean;
  isDrag: boolean;
  targetX?: number;
  targetY?: number;
  dragFromX?: number;
  dragFromY?: number;
  durationMs?: number;
  inspectHighResNext: boolean;
  actionDeliberation: string;
}

function asRecord(val: unknown): Record<string, unknown> | null {
  return val !== null && typeof val === 'object' && !Array.isArray(val)
    ? (val as Record<string, unknown>)
    : null;
}

function asNonEmptyString(val: unknown): string {
  return typeof val === 'string' ? val.trim() : '';
}

function asFiniteNumber(val: unknown): number | undefined {
  if (typeof val === 'number' && Number.isFinite(val)) return val;
  if (typeof val === 'string' && val.trim() !== '') {
    const parsed = Number(val);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function humanizeIdentifier(rawKey: string): string {
  const trimmed = rawKey.trim();
  if (!trimmed) return '';
  if (trimmed.includes(' ') || !trimmed.includes('_')) return trimmed;
  return trimmed
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

const MOUSE_ACTION_LABELS: Record<string, string> = {
  left_click: 'Left Click',
  right_click: 'Right Click',
  double_click: 'Double Click',
  drag_left: 'Drag',
  hold_left: 'Hold Left',
  release_left: 'Release Left',
  wheel_up: 'Scroll Up',
  wheel_down: 'Scroll Down',
};

/**
 * Formats a turn's motor action into crisp human-readable chips and a summary string
 * without exposing raw JSON payloads front-and-center.
 */
export function formatPlaytestMotorAction(turn: PlaytestTurn | undefined): FormattedMotorAction {
  if (!turn) {
    return {
      summary: 'Awaiting turn',
      chips: [],
      mouseButtonAction: 'none',
      hasExplicitMouse: false,
      isDrag: false,
      inspectHighResNext: false,
      actionDeliberation: '',
    };
  }

  const act = asRecord(turn.action) ?? {};
  const px = turn.action_px;

  const btnRaw = asNonEmptyString(px?.mouse_button_action ?? act.mouse_button_action) || 'none';
  const holdKeys = Array.isArray(px?.hold_keys)
    ? px.hold_keys.map(String).filter(Boolean)
    : Array.isArray(act.hold_keys)
      ? act.hold_keys.map(String).filter(Boolean)
      : [];
  const tapKeys = Array.isArray(px?.tap_keys)
    ? px.tap_keys.map(String).filter(Boolean)
    : Array.isArray(act.tap_keys)
      ? act.tap_keys.map(String).filter(Boolean)
      : [];

  const durationMs = asFiniteNumber(px?.duration_ms ?? act.duration_ms);
  const mxNorm = asFiniteNumber(act.mouse_x);
  const myNorm = asFiniteNumber(act.mouse_y);

  const targetX =
    asFiniteNumber(px?.target_x ?? px?.x) ??
    (mxNorm !== undefined ? Math.round((mxNorm / 1000) * 1280) : turn.cursor_end_x);
  const targetY =
    asFiniteNumber(px?.target_y ?? px?.y) ??
    (myNorm !== undefined ? Math.round((myNorm / 1000) * 720) : turn.cursor_end_y);
  const dragFromX = asFiniteNumber(px?.drag_from_x) ?? turn.cursor_start_x;
  const dragFromY = asFiniteNumber(px?.drag_from_y) ?? turn.cursor_start_y;

  const hasExplicitMouse =
    Boolean(px?.has_explicit_mouse) || mxNorm !== undefined || myNorm !== undefined || btnRaw !== 'none';
  const isDrag = btnRaw === 'drag_left';
  const inspectHighResNext = Boolean(act.inspect_high_res_next);
  const actionDeliberation = asNonEmptyString(act.deliberation);

  const chips: string[] = [];

  if (btnRaw !== 'none') {
    const label = MOUSE_ACTION_LABELS[btnRaw] ?? humanizeIdentifier(btnRaw);
    if (isDrag && dragFromX !== undefined && dragFromY !== undefined && targetX !== undefined && targetY !== undefined) {
      chips.push(
        `${label} (${Math.round(dragFromX)}, ${Math.round(dragFromY)}) → (${Math.round(targetX)}, ${Math.round(targetY)})`,
      );
    } else if (targetX !== undefined && targetY !== undefined) {
      chips.push(`${label} (${Math.round(targetX)}, ${Math.round(targetY)})`);
    } else {
      chips.push(label);
    }
  } else if (mxNorm !== undefined && myNorm !== undefined && targetX !== undefined && targetY !== undefined) {
    chips.push(`Move (${Math.round(targetX)}, ${Math.round(targetY)})`);
  }

  if (tapKeys.length > 0) {
    chips.push(`Tap [${tapKeys.join(' + ')}]`);
  }

  if (holdKeys.length > 0) {
    const durSuffix = durationMs ? ` ${durationMs}ms` : '';
    chips.push(`Hold [${holdKeys.join(' + ')}]${durSuffix}`);
  }

  if (chips.length === 0) {
    chips.push(durationMs ? `Observe (${durationMs}ms)` : 'Observe');
  }

  if (inspectHighResNext) {
    chips.push('High-Res Inspect');
  }

  return {
    summary: chips.join(' · '),
    chips,
    mouseButtonAction: btnRaw,
    hasExplicitMouse,
    isDrag,
    targetX,
    targetY,
    dragFromX,
    dragFromY,
    durationMs,
    inspectHighResNext,
    actionDeliberation,
  };
}

/**
 * Strips the internal prompt-tape header (`[Turn 120 | t=89.8s | ...]`) and trailing
 * serialized notebook blocks (`| NOTE+[...] | QUESTION+[...] | RULEBOOK+[...]`) from
 * `append_log_line`, returning only the concise human-readable turn observation.
 */
export function extractCleanTurnObservation(appendLogLine?: string | null): string {
  const raw = asNonEmptyString(appendLogLine);
  if (!raw) return '';

  let cleaned = raw;
  if (cleaned.startsWith('[Turn ')) {
    const closeBracket = cleaned.indexOf(']');
    if (closeBracket !== -1) {
      cleaned = cleaned.slice(closeBracket + 1).trim();
    }
  }

  const suffixMarkers = [
    ' | NOTE+[',
    ' | QUESTION+[',
    ' | RULEBOOK+[',
    ' | BUG+[',
    ' | IMPRESSION+[',
    ' | WISH+[',
    ' | MOOD+[',
  ];

  let cutIndex = cleaned.length;
  for (const marker of suffixMarkers) {
    const idx = cleaned.indexOf(marker);
    if (idx !== -1 && idx < cutIndex) {
      cutIndex = idx;
    }
  }

  return cleaned.slice(0, cutIndex).trim();
}

export function parseRulebookEntries(
  raw: Array<Record<string, unknown> | string> | Record<string, unknown> | undefined | null,
): PlaytestRulebookEntry[] {
  if (!raw) return [];
  if (Array.isArray(raw)) {
    return raw
      .map((item): PlaytestRulebookEntry | null => {
        if (typeof item === 'string') {
          const text = item.trim();
          return text ? { topic: '', behavior: text } : null;
        }
        const rec = asRecord(item);
        if (!rec) return null;
        const topic =
          asNonEmptyString(rec.key_or_topic) ||
          asNonEmptyString(rec.key) ||
          asNonEmptyString(rec.title);
        const behavior =
          asNonEmptyString(rec.behavior) ||
          asNonEmptyString(rec.rule) ||
          asNonEmptyString(rec.description);
        if (!topic && !behavior) return null;
        return { topic, behavior };
      })
      .filter((x): x is PlaytestRulebookEntry => x !== null);
  }

  const obj = asRecord(raw);
  if (!obj) return [];
  return Object.entries(obj)
    .map(([k, v]): PlaytestRulebookEntry | null => {
      if (typeof v === 'string') {
        return { topic: k, behavior: v.trim() };
      }
      const rec = asRecord(v);
      if (!rec) return null;
      const behavior =
        asNonEmptyString(rec.behavior) ||
        asNonEmptyString(rec.rule) ||
        asNonEmptyString(rec.description);
      return behavior ? { topic: asNonEmptyString(rec.key_or_topic) || k, behavior } : null;
    })
    .filter((x): x is PlaytestRulebookEntry => x !== null);
}

export function parseFieldNoteEntries(
  raw: Array<Record<string, unknown> | string> | Record<string, unknown> | undefined | null,
): PlaytestFieldNoteEntry[] {
  if (!raw) return [];
  if (Array.isArray(raw)) {
    return raw
      .map((item): PlaytestFieldNoteEntry | null => {
        if (typeof item === 'string') {
          const text = item.trim();
          return text ? { topic: '', observation: text } : null;
        }
        const rec = asRecord(item);
        if (!rec) return null;
        const topic =
          asNonEmptyString(rec.topic) ||
          asNonEmptyString(rec.key) ||
          asNonEmptyString(rec.title);
        const observation =
          asNonEmptyString(rec.observation) ||
          asNonEmptyString(rec.note) ||
          asNonEmptyString(rec.description);
        if (!topic && !observation) return null;
        return { topic, observation };
      })
      .filter((x): x is PlaytestFieldNoteEntry => x !== null);
  }

  const obj = asRecord(raw);
  if (!obj) return [];
  return Object.entries(obj)
    .map(([k, v]): PlaytestFieldNoteEntry | null => {
      if (typeof v === 'string') {
        return { topic: k, observation: v.trim() };
      }
      const rec = asRecord(v);
      if (!rec) return null;
      const observation =
        asNonEmptyString(rec.observation) ||
        asNonEmptyString(rec.note) ||
        asNonEmptyString(rec.description);
      return observation ? { topic: asNonEmptyString(rec.topic) || k, observation } : null;
    })
    .filter((x): x is PlaytestFieldNoteEntry => x !== null);
}

export function parseQuestionEntries(
  raw: Array<Record<string, unknown> | string> | Record<string, unknown> | undefined | null,
): { open: PlaytestQuestionEntry[]; answered: PlaytestQuestionEntry[] } {
  const all: PlaytestQuestionEntry[] = [];
  if (!raw) return { open: [], answered: [] };

  const pushEntry = (fallbackKey: string, val: unknown) => {
    if (typeof val === 'string') {
      const q = val.trim();
      if (q) {
        all.push({ key: fallbackKey, question: q, status: 'open', answer: '' });
      }
      return;
    }
    const rec = asRecord(val);
    if (!rec) return;
    const key = asNonEmptyString(rec.question_key) || fallbackKey;
    const question =
      asNonEmptyString(rec.question) ||
      asNonEmptyString(rec.description) ||
      humanizeIdentifier(key);
    if (!question) return;
    const answer = asNonEmptyString(rec.answer);
    const rawStatus = asNonEmptyString(rec.status).toLowerCase();
    const status: 'open' | 'answered' =
      rawStatus === 'answered' || (rawStatus !== 'open' && Boolean(answer))
        ? 'answered'
        : 'open';
    const askedTurn = asFiniteNumber(rec.asked_turn);
    const answeredTurn = asFiniteNumber(rec.answered_turn);
    all.push({
      key,
      question,
      status,
      answer,
      askedTurn,
      answeredTurn,
    });
  };

  if (Array.isArray(raw)) {
    raw.forEach((item, idx) => pushEntry(`q_${idx + 1}`, item));
  } else {
    const obj = asRecord(raw);
    if (obj) {
      for (const [k, v] of Object.entries(obj)) {
        pushEntry(k, v);
      }
    }
  }

  const open = all
    .filter((q) => q.status === 'open')
    .sort((a, b) => (b.askedTurn ?? 0) - (a.askedTurn ?? 0));
  const answered = all
    .filter((q) => q.status === 'answered')
    .sort((a, b) => (b.answeredTurn ?? b.askedTurn ?? 0) - (a.answeredTurn ?? a.askedTurn ?? 0));

  return { open, answered };
}

export function parseBugEntries(
  raw: Array<Record<string, unknown> | string> | Record<string, unknown> | undefined | null,
): PlaytestBugEntry[] {
  const bugs: PlaytestBugEntry[] = [];
  if (!raw) return bugs;

  const pushBug = (fallbackKey: string, val: unknown) => {
    if (typeof val === 'string') {
      const text = val.trim();
      if (text) {
        bugs.push({
          key: fallbackKey,
          severity: 'major',
          category: 'bug',
          summary: text,
          details: '',
        });
      }
      return;
    }
    const rec = asRecord(val);
    if (!rec) return;
    const key = asNonEmptyString(rec.bug_key) || fallbackKey;
    const severity = asNonEmptyString(rec.severity) || 'major';
    const category = asNonEmptyString(rec.category) || 'bug';
    const summary =
      asNonEmptyString(rec.summary) ||
      asNonEmptyString(rec.title) ||
      humanizeIdentifier(key);
    const details =
      asNonEmptyString(rec.details) ||
      asNonEmptyString(rec.description);
    const turnIndex = asFiniteNumber(rec.turn_index);
    const screenshotName = asNonEmptyString(rec.screenshot_name) || undefined;
    if (!summary && !details) return;
    bugs.push({
      key,
      severity,
      category,
      summary: summary || details,
      details: summary ? details : '',
      turnIndex,
      screenshotName,
    });
  };

  if (Array.isArray(raw)) {
    raw.forEach((item, idx) => pushBug(`bug_${idx + 1}`, item));
  } else {
    const obj = asRecord(raw);
    if (obj) {
      for (const [k, v] of Object.entries(obj)) {
        pushBug(k, v);
      }
    }
  }

  return bugs.sort((a, b) => (b.turnIndex ?? 0) - (a.turnIndex ?? 0));
}
