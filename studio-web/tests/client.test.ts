import { afterEach, describe, expect, it, vi } from 'vitest';
import { resolveMediaUrl, StudioApiError, studioApi } from '../src/api/client';

describe('studioApi client contracts (Rule 4 & Rule 5)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('serializes `deliberation` strictly as key index 0 on every state-mutating POST request (Rule 4)', async () => {
    const capturedBodies: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
      if (typeof init?.body === 'string') {
        capturedBodies.push(init.body);
      }
      return new Response(JSON.stringify({ deliberation: 'ok', status: 'ok', job_id: 'job_1' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    });

    await studioApi.launchGame('candidate_synthetic_01');
    await studioApi.updateStudioSettings({ pauseForGreenlight: true, paused: false });
    await studioApi.submitGreenlightOrSteering({ approved: false, feedback: 'Tighten input responsiveness.' });
    await studioApi.triggerGenesisBuild({ resumeCurrent: false, creativePrompt: 'Synthetic tactics loop' });
    await studioApi.startPlaytestSession({
      checkpointId: 'active',
      candidateId: 'candidate_synthetic_01',
      maxTurns: 12,
      enableAudio: true,
    });
    await studioApi.controlPlaytestSession('pt_01', 'pause');
    await studioApi.sendWishesToCouncil({ wishIds: ['wish_1'], producerNote: 'Prioritize camera bounds.' });

    expect(capturedBodies).toHaveLength(7);
    for (const raw of capturedBodies) {
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      const keys = Object.keys(parsed);
      expect(keys[0]).toBe('deliberation');
      expect(typeof parsed.deliberation).toBe('string');
      expect((parsed.deliberation as string).length).toBeGreaterThan(10);
    }
  });

  it('surfaces authentic HTTP & epistemic errors via StudioApiError without silent fallback (Rule 5)', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          deliberation: 'Godot headless binary returned non-zero exit code 1.',
          detail: 'Godot launch failed: scene root missing.',
        }),
        { status: 502, statusText: 'Bad Gateway' },
      ),
    );

    await expect(studioApi.launchGame('candidate_synthetic_01')).rejects.toMatchObject({
      name: 'StudioApiError',
      status: 502,
      endpoint: '/api/game/launch',
      message: 'Godot launch failed: scene root missing.',
      deliberation: 'Godot headless binary returned non-zero exit code 1.',
    });
  });

  it('throws StudioApiError on network failure instead of returning mock data (Rule 5)', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new Error('ECONNREFUSED 127.0.0.1:8000'));

    let caught: unknown;
    try {
      await studioApi.getStudioStatus();
    } catch (err) {
      caught = err;
    }

    expect(caught).toBeInstanceOf(StudioApiError);
    expect((caught as StudioApiError).status).toBe(0);
    expect((caught as StudioApiError).message).toContain('ECONNREFUSED');
  });

  it('resolves cartridge-scoped media URLs accurately', () => {
    expect(resolveMediaUrl('assets/sprites/unit_a.png', 'candidate_synthetic_01')).toBe(
      '/api/artifacts/content?path=candidate_synthetic_01%2Fassets%2Fsprites%2Funit_a.png&raw=true',
    );
    expect(resolveMediaUrl('/api/artifacts/content?filename=hud.png&raw=true', 'candidate_synthetic_01')).toBe(
      '/api/artifacts/content?filename=hud.png&raw=true&cartridge=candidate_synthetic_01',
    );
  });
});
