import { describe, expect, it } from 'vitest';
import {
  normalizeProofArtifacts,
  normalizeStepLog,
  parsePromptIntoSections,
} from '../src/domain/promptParser';

describe('promptParser', () => {
  it('parses banners, context chunks, and embedded file snapshots without losing characters', () => {
    const raw = [
      'You are the SWE Engineer building a deterministic state module.',
      'ARCHITECTURAL SPECIFICATION:',
      'Implement deterministic tick step.',
      '=== CURATED STUDIO CONTEXT ===',
      'Selected 2 chunks for swe_engineer.',
      '[bible:core_loop] (design) Core Simulation Loop',
      'Tick interval is 16ms fixed.',
      '[code:scripts/store.gd] (code) State Store Contract',
      'signal state_updated(tick: int)',
      '=== END CURATED STUDIO CONTEXT ===',
      'EXISTING CARTRIDGE CODE:',
      '--- Existing scripts/store.gd ---',
      'extends Node',
      'var tick: int = 0',
      '--- Existing scripts/runner.gd ---',
      'extends Node',
      'func step() -> void:',
      '  pass',
      'ACTION REQUIRED:',
      'Return structured patches.',
    ].join('\n');

    const sections = parsePromptIntoSections(raw);
    expect(sections.length).toBeGreaterThanOrEqual(5);

    const curatedSec = sections.find((s) => s.title.includes('CURATED STUDIO CONTEXT'));
    expect(curatedSec).toBeDefined();
    expect(curatedSec?.subs).toHaveLength(2);
    expect(curatedSec?.subs[0].title).toContain('[bible:core_loop]');
    expect(curatedSec?.subs[1].title).toContain('[code:scripts/store.gd]');

    const codeSec = sections.find((s) => s.title === 'EXISTING CARTRIDGE CODE');
    expect(codeSec).toBeDefined();
    expect(codeSec?.subs).toHaveLength(2);
    expect(codeSec?.subs[0].title).toContain('Existing scripts/store.gd');

    // Rejoining section contents preserves 100% of lines
    const reconstructed = sections.map((s) => s.content).join('\n');
    expect(reconstructed).toBe(raw);
  });

  it('normalizes step logs with elapsed deltas and structured details', () => {
    const steps = normalizeStepLog([
      {
        timestamp_utc: '2026-10-31T14:00:00.000Z',
        step_type: 'planning_turn',
        specialist_agent: 'task_lead',
        message: 'Task Lead planning Turn #1',
      },
      {
        timestamp_utc: '2026-10-31T14:00:08.400Z',
        step_type: 'editing_file',
        specialist_agent: 'swe_engineer',
        message: 'Patching file scripts/store.gd',
        details: {
          file: 'scripts/store.gd',
          patches: 2,
          lines: 84,
        },
      },
    ]);

    expect(steps).toHaveLength(2);
    expect(steps[0].label).toBe('PLANNING');
    expect(steps[1].label).toBe('EDITING FILE');
    expect(steps[1].elapsedLabel).toBe('+8.4s');
    expect(steps[1].detailsEntries).toEqual([
      { key: 'file', value: 'scripts/store.gd' },
      { key: 'patches', value: '2' },
      { key: 'lines', value: '84' },
    ]);
  });

  it('normalizes both object-shaped and string-shaped proof artifacts', () => {
    const proofs = normalizeProofArtifacts(
      [
        {
          path: 'artifacts/renders/sheet_01.png',
          filename: 'sheet_01.png',
          url: '/api/artifacts/content?path=sheet_01.png&raw=true',
          is_audio: false,
        },
        '/api/artifacts/content?path=sfx_click.wav&raw=true',
      ],
      undefined,
      'candidate_synthetic_01',
    );

    expect(proofs).toHaveLength(2);
    expect(proofs[0].filename).toBe('sheet_01.png');
    expect(proofs[0].isAudio).toBe(false);
    expect(proofs[1].filename).toBe('sfx_click.wav');
    expect(proofs[1].isAudio).toBe(true);
  });
});
