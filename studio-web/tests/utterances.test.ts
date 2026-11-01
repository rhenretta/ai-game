import { describe, expect, it } from 'vitest';
import {
  formatWritersRoomUtterance,
  normalizeUtteranceBadges,
  parseStructuredBlocks,
} from '../src/domain/utterances';

describe('Writers Room utterance formatting and summarization', () => {
  it('converts serialized Python WorkItem dumps and board tags into compact single-line badges', () => {
    const rawDump =
      "Shaped/split 2 backlog item(s) -> [WorkItem(deliberation='Long internal rationale', item_id='mech_core_grid', kind='mechanic_code', title='Core Grid Solver', spec='Very long spec text'), WorkItem(deliberation='Second rationale', item_id='ui_status_bar', kind='ui_code', title='Status Bar Overlay', spec='Overlay spec')]";

    const badges = normalizeUtteranceBadges([
      "Raised 2 backlog item(s): ['asset_tile_set', 'audio_click_cue']",
      rawDump,
      'review:mech_core_grid:approve',
      'review:ui_status_bar:request_changes',
      'updated:core_loop',
    ]);

    expect(badges).toEqual([
      { label: '+ asset_tile_set', tone: 'ok', title: "Raised 2 backlog item(s): ['asset_tile_set', 'audio_click_cue']" },
      { label: '+ audio_click_cue', tone: 'ok', title: "Raised 2 backlog item(s): ['asset_tile_set', 'audio_click_cue']" },
      { label: 'Shaped: mech_core_grid', tone: 'accent', title: 'mech_core_grid — Core Grid Solver' },
      { label: 'Shaped: ui_status_bar', tone: 'accent', title: 'ui_status_bar — Status Bar Overlay' },
      { label: 'Approved: mech_core_grid', tone: 'ok' },
      { label: 'Revise: ui_status_bar', tone: 'warn' },
      { label: 'Updated core loop', tone: 'neutral' },
    ]);
  });

  it('splits long specialist turns into lead summary + expandable detail while keeping Producer turns unfolded', () => {
    const sharedRationale =
      'Reviewing Stage 1 tasks as Systems Architect to establish clean boundaries between simulation ticks and HUD signals. Every module must communicate strictly through typed Godot signals rather than scene-tree node path coupling.';
    const shapingSpoken =
      `${sharedRationale} Specifically: 1) Decouple tick stepping inside \`scripts/sim_clock.gd\` from camera interpolation; 2) Emit **deterministic** state snapshots at fixed intervals; 3) Route input actions through \`_unhandled_input\`. ` +
      "Shaped/refined board task(s): `deliberation='Internal' item_id='mech_sim_clock' kind='mechanic_code' title='Simulation Clock' spec='Detailed spec'`." ;

    const formattedShaping = formatWritersRoomUtterance({
      utterance_id: 'utt_42_GameplaySystemsEngineer_task_shaping',
      turn_index: 42,
      speaker_role: 'GameplaySystemsEngineer',
      deliberation: sharedRationale,
      spoken_dialogue: shapingSpoken,
      bible_updates_summary: [],
    });

    expect(formattedShaping.turnKindLabel).toBe('Task Shaping');
    // Duplicate deliberation prefix is suppressed so rationale is not shown twice
    expect(formattedShaping.distinctDeliberation).toBeUndefined();
    expect(formattedShaping.hasExpandableDetail).toBe(true);
    expect(formattedShaping.leadBlocks.length).toBeGreaterThan(0);
    expect(formattedShaping.detailBlocks.some((b) => b.type === 'ordered-list')).toBe(true);

    const producerMsg =
      'Focus the next iteration on deep systemic progression, modular crafting chains, and long-horizon automation goals across multiple regions without handing pre-built structures to the player at boot.';
    const formattedProducer = formatWritersRoomUtterance({
      utterance_id: 'utt_exec_01',
      turn_index: 43,
      speaker_role: 'ExecutiveProducer',
      deliberation: 'Executive Producer provided slice direction.',
      spoken_dialogue: producerMsg,
    });

    expect(formattedProducer.turnKindLabel).toBe('Producer Direction');
    expect(formattedProducer.hasExpandableDetail).toBe(false);
  });

  it('parses inline code, bold emphasis, and inline numbered enumerations into structured blocks', () => {
    const blocks = parseStructuredBlocks(
      'Key requirements: 1) Wire `res://main.tscn` to the root controller; 2) Enforce **zero** frame drops under load.',
    );

    expect(blocks).toHaveLength(2);
    expect(blocks[0]).toEqual({
      type: 'paragraph',
      tokens: [{ type: 'text', value: 'Key requirements:' }],
    });
    expect(blocks[1].type).toBe('ordered-list');
    if (blocks[1].type === 'ordered-list') {
      expect(blocks[1].items).toHaveLength(2);
      expect(blocks[1].items[0]).toContainEqual({ type: 'code', value: 'res://main.tscn' });
      expect(blocks[1].items[1]).toContainEqual({ type: 'bold', value: 'zero' });
    }
  });

  it('parses period-delimited inline lists without splitting on sentence-ending cardinal numbers', () => {
    const blocks = parseStructuredBlocks(
      '1. Trigger `_apply_step(id)` and verify output rises from 0 to at least 1. 2. Confirm display label updates to `x1` before stage reaches 2. 3. Validate clean teardown.',
    );

    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe('ordered-list');
    if (blocks[0].type === 'ordered-list') {
      expect(blocks[0].items).toHaveLength(3);
      expect(blocks[0].items[0].some((t) => t.type === 'code' && t.value === '_apply_step(id)')).toBe(true);
      const item0Text = blocks[0].items[0].map((t) => t.value).join('');
      expect(item0Text.endsWith('from 0 to at least 1.')).toBe(true);
    }
  });
});
