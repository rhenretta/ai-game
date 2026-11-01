import {
  BookOpen,
  Brain,
  Cpu,
  Dna,
  FolderCode,
  Gamepad2,
  History,
  Kanban,
  MessageSquare,
  Pause,
  Play,
  Plus,
  Radio,
  Rocket,
  Star,
} from 'lucide-react';
import type {
  GameBibleResponse,
  WorkspaceId,
} from '../../api/types';
import {
  resolveSliceVerificationState,
  type NormalizedGameCatalogEntry,
} from '../../domain/slices';

export interface TopCommandBarProps {
  games: NormalizedGameCatalogEntry[];
  selectedCartridgeId: string;
  activeCartridgeId: string;
  onSelectCartridge: (cartridgeId: string) => void;
  activeWorkspace: WorkspaceId;
  onSelectWorkspace: (ws: WorkspaceId) => void;
  bible: GameBibleResponse | undefined;
  boardCounts: { approved: number; total: number };
  transcriptCount: number;
  assetsCount: number;
  playtestRunsCount: number;
  pendingWishesCount: number;
  wsConnected: boolean;
  recentEventCount: number;
  onActivateGame: () => void;
  activatingGame: boolean;
  onLaunchGame: () => void;
  launchingGame: boolean;
  launchStatusText: string | null;
  onPauseOrResume: (pause: boolean) => void;
  togglingPause: boolean;
  onApproveGreenlight: () => void;
  hasBuildError?: boolean;
  livePhaseBadge?: string;
  onResumeBuild?: () => void;
  resumingBuild?: boolean;
  uiScalePct?: number;
  onChangeUiScalePct?: (nextPct: number) => void;
  onOpenNextSliceModal: () => void;
  onOpenNewGameModal: () => void;
  onOpenCheckpointsModal: () => void;
  onOpenLivePulseModal: () => void;
}

export const UI_SCALE_MIN_PCT = 90;
export const UI_SCALE_MAX_PCT = 160;
export const UI_SCALE_STEP_PCT = 10;

export function TopCommandBar({
  games,
  selectedCartridgeId,
  activeCartridgeId,
  onSelectCartridge,
  activeWorkspace,
  onSelectWorkspace,
  bible,
  boardCounts,
  transcriptCount,
  assetsCount,
  playtestRunsCount,
  pendingWishesCount,
  wsConnected,
  recentEventCount,
  onActivateGame,
  activatingGame,
  onLaunchGame,
  launchingGame,
  launchStatusText,
  onPauseOrResume,
  togglingPause,
  onApproveGreenlight,
  hasBuildError = false,
  livePhaseBadge,
  onResumeBuild,
  resumingBuild = false,
  uiScalePct = 100,
  onChangeUiScalePct,
  onOpenNextSliceModal,
  onOpenNewGameModal,
  onOpenCheckpointsModal,
  onOpenLivePulseModal,
}: TopCommandBarProps) {
  const proc = bible?.process_status;
  const isRunning = Boolean(proc?.build_actively_running || proc?.process_state === 'running');
  const isPaused = Boolean(proc?.is_paused);
  const isAwaitingGreenlight =
    proc?.process_state === 'awaiting_greenlight' || bible?.orchestrator_state === 'AWAITING_GREENLIGHT';
  const isErrorState = Boolean(hasBuildError || proc?.process_state === 'error');
  const sliceVerification = resolveSliceVerificationState(bible, isErrorState);
  const isSliceVerified =
    bible?.production_board?.items !== undefined
      ? sliceVerification.isSliceVerified
      : boardCounts.total > 0 &&
        boardCounts.approved >= boardCounts.total &&
        !isRunning &&
        !isPaused &&
        !isAwaitingGreenlight &&
        !isErrorState &&
        !sliceVerification.hasUnfinishedExplicitSlice;

  const effectiveCartridge = selectedCartridgeId || bible?.active_candidate_id || activeCartridgeId || 'active';
  const canSetActive =
    Boolean(effectiveCartridge) && effectiveCartridge !== 'active' && effectiveCartridge !== activeCartridgeId;

  const tabs: Array<{ id: WorkspaceId; label: string; icon: typeof Kanban; badge?: string | number; warnBadge?: number }> = [
    {
      id: 'board',
      label: 'Board',
      icon: Kanban,
      badge: boardCounts.total > 0 ? `${boardCounts.approved}/${boardCounts.total}` : undefined,
    },
    {
      id: 'writers-room',
      label: "Writers' Room",
      icon: MessageSquare,
      badge: transcriptCount > 0 ? transcriptCount : undefined,
    },
    {
      id: 'bible',
      label: 'Game Bible',
      icon: BookOpen,
    },
    {
      id: 'memory',
      label: 'Memory',
      icon: Brain,
    },
    {
      id: 'assets',
      label: 'Assets & Code',
      icon: FolderCode,
      badge: assetsCount > 0 ? assetsCount : undefined,
    },
    {
      id: 'playtest',
      label: 'Playtester',
      icon: Gamepad2,
      badge: playtestRunsCount > 0 ? playtestRunsCount : undefined,
      warnBadge: pendingWishesCount > 0 ? pendingWishesCount : undefined,
    },
    {
      id: 'dna',
      label: 'Player DNA',
      icon: Dna,
    },
  ];

  return (
    <header className="top-command-bar">
      <div className="brand-cluster">
        <Cpu size={16} style={{ color: 'var(--accent)', flexShrink: 0 }} />
        <span className="brand-title">Studio</span>

        <select
          aria-label="Select Game Cartridge"
          className="select-control"
          style={{ maxInlineSize: '260px', fontWeight: 600 }}
          value={selectedCartridgeId}
          onChange={(e) => onSelectCartridge(e.target.value)}
        >
          {games.length === 0 ? (
            <option value="active">{bible?.game_title || 'Active Cartridge'}</option>
          ) : null}
          {games.map((g) => (
            <option key={g.cartridge_id} value={g.cartridge_id}>
              {g.isPromotedActive ? '★ ' : ''}
              {g.game_title}
              {g.total_items > 0 ? ` [${g.approved_items}/${g.total_items}]` : ''}
              {g.hasDuplicateTitle ? ` (${g.shortId})` : ''}
            </option>
          ))}
        </select>

        {canSetActive ? (
          <button
            type="button"
            className="btn"
            onClick={onActivateGame}
            disabled={activatingGame}
            title="Promote selected cartridge to cartridges/active"
          >
            <Star size={13} />
            {activatingGame ? 'Setting...' : 'Set Active'}
          </button>
        ) : null}

        <button
          type="button"
          className="btn btn-success"
          onClick={onLaunchGame}
          disabled={launchingGame}
          title="Launch selected cartridge in a native Godot 4.3 window"
        >
          <Gamepad2 size={14} />
          {launchingGame ? 'Launching...' : launchStatusText || 'Launch'}
        </button>
      </div>

      <nav className="workspace-tabs" role="tablist" aria-label="Studio Workspaces">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const selected = activeWorkspace === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={selected}
              className="ws-tab"
              onClick={() => onSelectWorkspace(tab.id)}
            >
              <Icon size={13} />
              <span>{tab.label}</span>
              {tab.badge !== undefined ? <span className="tab-count">{tab.badge}</span> : null}
              {tab.warnBadge !== undefined ? (
                <span
                  className="tab-count"
                  style={{
                    color: 'var(--warn)',
                    border: '1px solid color-mix(in srgb, var(--warn) 50%, transparent)',
                  }}
                  title={`${tab.warnBadge} pending player wish(es)`}
                >
                  +{tab.warnBadge}
                </span>
              ) : null}
            </button>
          );
        })}
      </nav>

      <div className="command-actions">
        {onChangeUiScalePct ? (
          <div className="ui-scale-control" role="group" aria-label="UI text size">
            <button
              type="button"
              className="ui-scale-btn"
              disabled={uiScalePct <= UI_SCALE_MIN_PCT}
              onClick={() =>
                onChangeUiScalePct(Math.max(UI_SCALE_MIN_PCT, uiScalePct - UI_SCALE_STEP_PCT))
              }
              title="Decrease UI text size"
            >
              A-<span className="visually-hidden"> Decrease text size</span>
            </button>
            <button
              type="button"
              className="ui-scale-btn ui-scale-readout"
              onClick={() => onChangeUiScalePct(100)}
              title="Reset UI text size to 100%"
            >
              {uiScalePct}%<span className="visually-hidden"> Reset text size</span>
            </button>
            <button
              type="button"
              className="ui-scale-btn"
              disabled={uiScalePct >= UI_SCALE_MAX_PCT}
              onClick={() =>
                onChangeUiScalePct(Math.min(UI_SCALE_MAX_PCT, uiScalePct + UI_SCALE_STEP_PCT))
              }
              title="Increase UI text size"
            >
              A+<span className="visually-hidden"> Increase text size</span>
            </button>
          </div>
        ) : null}

        <button
          type="button"
          className="btn"
          onClick={onOpenLivePulseModal}
          title={wsConnected ? 'WebSocket streaming live deltas' : 'WebSocket reconnecting'}
        >
          <Radio size={13} style={{ color: wsConnected ? 'var(--ok)' : 'var(--warn)' }} />
          <span>{wsConnected ? 'Live' : 'Sync'}</span>
          {recentEventCount > 0 ? <span className="tab-count">{recentEventCount}</span> : null}
        </button>

        <button
          type="button"
          className="btn"
          onClick={onOpenCheckpointsModal}
          title="Browse and activate verified slice checkpoints"
        >
          <History size={13} />
          <span>Checkpoints</span>
        </button>

        {isAwaitingGreenlight ? (
          <button type="button" className="btn btn-success" onClick={onApproveGreenlight}>
            <Play size={13} /> Greenlight Bible
          </button>
        ) : isRunning ? (
          <>
            <span className="badge badge-ok">
              Running
              {proc?.scheduler_running_workers
                ? ` · ${proc.scheduler_running_workers}w`
                : livePhaseBadge
                  ? ` · ${livePhaseBadge}`
                  : ''}
            </span>
            <button
              type="button"
              className="btn btn-warn"
              onClick={() => onPauseOrResume(true)}
              disabled={togglingPause}
            >
              <Pause size={13} /> Pause
            </button>
          </>
        ) : isPaused ? (
          <>
            <span className="badge badge-warn">Paused</span>
            <button
              type="button"
              className="btn btn-success"
              onClick={() => onPauseOrResume(false)}
              disabled={togglingPause}
            >
              <Play size={13} /> Resume
            </button>
          </>
        ) : isSliceVerified ? (
          <>
            <span className="badge badge-ok">Slice Verified</span>
            <button type="button" className="btn btn-primary" onClick={onOpenNextSliceModal}>
              <Rocket size={13} aria-hidden="true" /> Next Slice
            </button>
          </>
        ) : bible?.has_bible ? (
          <>
            <span className={`badge ${isErrorState ? 'badge-danger' : 'badge-warn'}`}>
              {isErrorState
                ? 'Build Error'
                : sliceVerification.isAwaitingCouncilPlaytest
                  ? 'Awaiting Playtest'
                  : boardCounts.total > 0
                    ? `Stopped · ${boardCounts.approved}/${boardCounts.total}`
                    : 'Stopped'}
            </span>
            <button
              type="button"
              className="btn btn-success"
              onClick={() => (onResumeBuild ? onResumeBuild() : onOpenNextSliceModal())}
              disabled={resumingBuild}
              title={
                sliceVerification.isAwaitingCouncilPlaytest
                  ? 'All board tasks are approved; click to resume slice into Council Blind Playtest & verification'
                  : 'Resume development from the persisted Living Game Bible and Production Board'
              }
            >
              <Play size={13} aria-hidden="true" /> {resumingBuild ? 'Resuming...' : 'Resume'}
            </button>
          </>
        ) : null}

        <button type="button" className="btn btn-primary" onClick={onOpenNewGameModal}>
          <Plus size={14} /> New Game
        </button>
      </div>
    </header>
  );
}
