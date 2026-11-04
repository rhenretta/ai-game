import { type FormEvent, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Archive,
  Brain,
  CheckCircle2,
  Plus,
  RotateCcw,
  Users,
} from 'lucide-react';
import { studioApi } from '../../api/client';
import type {
  GameMemoryEntry,
  MemoryKind,
  MemoryStatus,
} from '../../api/types';
import { DeliberationDisclosure } from '../../components/DeliberationDisclosure';
import { ErrorBanner } from '../../components/ErrorBanner';
import { getRoleMetadata } from '../../domain/roles';

export interface GameMemoryViewProps {
  cartridgeId: string;
  gameTitle?: string;
  activeSliceId?: string;
}

const COUNCIL_SPECIALIST_BANKS = [
  'CoProducer',
  'CreativeDirector',
  'GameplayDesigner',
  'LevelAndProgressionDesigner',
  'GameplaySystemsEngineer',
  'StoryWorldArchitect',
  'ScriptwriterAgent',
  'CharacterDesigner',
  'VisualArtDirector',
  'UXExperienceDesigner',
  'AudioAtmosphereDesigner',
] as const;

const BOARD_SPECIALIST_BANKS = [
  'task_lead',
  'swe_engineer',
  'qa_test_engineer',
  'visual_artist',
  'audio_designer',
  'script_writer',
] as const;

const KIND_LABELS: Record<MemoryKind, string> = {
  strategic_goal: 'Strategic Goal',
  architectural_decision: 'Architectural Decision',
  lesson_learned: 'Lesson Learned',
  domain_note: 'Domain Note',
};

const STATUS_LABELS: Record<MemoryStatus, string> = {
  active: 'Active',
  fulfilled: 'Fulfilled',
  superseded: 'Superseded',
  archived: 'Archived',
};

function syncControlValidationState(
  control: HTMLInputElement | HTMLTextAreaElement,
  interactedMap: WeakMap<HTMLElement, { hasInteracted: boolean; hasBlurred: boolean }>,
  phase: 'input' | 'blur' | 'submit',
): void {
  const minLenAttr = Number(control.getAttribute('minlength') ?? 0);
  const trimmedLen = control.value.trim().length;
  if (control.required && trimmedLen === 0) {
    control.setCustomValidity('This field is required.');
  } else if (minLenAttr > 0 && trimmedLen < minLenAttr) {
    control.setCustomValidity(`Please enter at least ${minLenAttr} non-whitespace characters.`);
  } else {
    control.setCustomValidity('');
  }

  const prev = interactedMap.get(control) ?? { hasInteracted: false, hasBlurred: false };
  if (phase === 'input') {
    prev.hasInteracted = true;
  } else if (phase === 'blur') {
    prev.hasBlurred = true;
  } else {
    prev.hasInteracted = true;
    prev.hasBlurred = true;
  }
  interactedMap.set(control, prev);

  if (phase === 'submit' || (prev.hasBlurred && prev.hasInteracted)) {
    const valid = control.checkValidity();
    control.classList.toggle('user-invalid-fallback', !valid);
    control.classList.toggle('user-valid-fallback', valid && trimmedLen > 0);
    if (!valid) {
      control.setAttribute('aria-invalid', 'true');
    } else {
      control.setAttribute('aria-invalid', 'false');
    }
  }
}

export function GameMemoryView({
  cartridgeId,
  gameTitle,
  activeSliceId,
}: GameMemoryViewProps) {
  const queryClient = useQueryClient();
  const [selectedBank, setSelectedBank] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<'all' | MemoryStatus>('active');
  const [selectedKind, setSelectedKind] = useState<'all' | MemoryKind>('all');
  const [composerOpen, setComposerOpen] = useState<boolean>(false);
  const [mutationError, setMutationError] = useState<unknown>(null);

  const [deliberation, setDeliberation] = useState<string>('');
  const [targetBank, setTargetBank] = useState<string>('council');
  const [kind, setKind] = useState<MemoryKind>('strategic_goal');
  const [title, setTitle] = useState<string>('');
  const [content, setContent] = useState<string>('');
  const [selectedRoles, setSelectedRoles] = useState<string[]>([]);

  const formRef = useRef<HTMLFormElement | null>(null);
  const interactedControlsRef = useRef<
    WeakMap<HTMLElement, { hasInteracted: boolean; hasBlurred: boolean }>
  >(new WeakMap());

  const memoriesQuery = useQuery({
    queryKey: ['game-memories', cartridgeId, selectedBank, selectedStatus, selectedKind],
    queryFn: () =>
      studioApi.getGameMemories({
        cartridgeId,
        bank: selectedBank,
        status: selectedStatus,
        kind: selectedKind,
      }),
  });

  const mutateMemory = useMutation({
    mutationFn: studioApi.mutateGameMemory,
    onSuccess: () => {
      setMutationError(null);
      void queryClient.invalidateQueries({ queryKey: ['game-memories'] });
    },
    onError: (err) => {
      setMutationError(err);
    },
  });

  const resetFormValidation = () => {
    const form = formRef.current;
    if (!form) return;
    const controls = form.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>(
      'input:not([type="checkbox"]), textarea',
    );
    controls.forEach((ctrl) => {
      interactedControlsRef.current.delete(ctrl);
      ctrl.setCustomValidity('');
      ctrl.classList.remove('user-invalid-fallback', 'user-valid-fallback');
      ctrl.removeAttribute('aria-invalid');
    });
  };

  const handleOpenComposer = () => {
    if (!composerOpen && selectedBank !== 'all') {
      setTargetBank(selectedBank);
    }
    setComposerOpen((open) => !open);
  };

  const handleToggleApplicableRole = (role: string) => {
    setSelectedRoles((prev) =>
      prev.includes(role) ? prev.filter((r) => r !== role) : [...prev, role],
    );
  };

  const handleSubmitRecord = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const controls = Array.from(
      form.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>(
        'input:not([type="checkbox"]), textarea',
      ),
    );
    controls.forEach((ctrl) =>
      syncControlValidationState(ctrl, interactedControlsRef.current, 'submit'),
    );
    if (!form.checkValidity()) {
      const firstInvalid = controls.find((c) => !c.checkValidity());
      firstInvalid?.focus();
      return;
    }

    await mutateMemory.mutateAsync({
      deliberation: deliberation.trim(),
      cartridge: cartridgeId,
      author_role: 'executive_producer',
      operation: 'record',
      target_bank: targetBank,
      kind,
      title: title.trim(),
      content: content.trim(),
      applicable_roles: selectedRoles,
      slice_id: activeSliceId,
    });

    setDeliberation('');
    setTitle('');
    setContent('');
    setSelectedRoles([]);
    resetFormValidation();
    setComposerOpen(false);
  };

  const handleLifecycleAction = (
    entry: GameMemoryEntry,
    operation: 'fulfill' | 'archive' | 'update',
  ) => {
    const actionLabel =
      operation === 'fulfill'
        ? 'Marked durable goal/memory as fulfilled'
        : operation === 'archive'
          ? 'Archived memory entry'
          : 'Restored memory entry to active circulation';
    mutateMemory.mutate({
      deliberation: `${actionLabel} ('${entry.title}') from Mission Control.`,
      cartridge: cartridgeId,
      author_role: 'executive_producer',
      operation,
      target_memory_id: entry.memory_id,
      slice_id: activeSliceId,
      resolution_note:
        operation === 'fulfill'
          ? 'Marked fulfilled by Executive Producer in Mission Control.'
          : operation === 'archive'
            ? 'Archived by Executive Producer in Mission Control.'
            : undefined,
    });
  };

  const data = memoriesQuery.data;
  const entries = data?.entries ?? [];
  const activeBankCounts = data?.bank_counts ?? {};
  const totalBankCounts = data?.total_bank_counts ?? activeBankCounts;
  const countsForNav = selectedStatus === 'active' ? activeBankCounts : totalBankCounts;
  const allBanksList = [...COUNCIL_SPECIALIST_BANKS, ...BOARD_SPECIALIST_BANKS];

  return (
    <div className="memory-workspace-layout">
      <nav className="memory-bank-sidebar" aria-label="Memory banks">
        <div className="memory-sidebar-heading">
          <Brain size={15} />
          <span>Memory Banks</span>
        </div>

        <button
          type="button"
          className="memory-bank-btn"
          aria-pressed={selectedBank === 'all'}
          onClick={() => setSelectedBank('all')}
        >
          <span>All Banks</span>
          <span className="memory-bank-count">
            {selectedStatus === 'active' ? (data?.active_count ?? 0) : (data?.total_count ?? 0)}
          </span>
        </button>

        <div className="memory-sidebar-group-label">Shared Game Bank</div>
        <button
          type="button"
          className="memory-bank-btn"
          aria-pressed={selectedBank === 'council'}
          onClick={() => setSelectedBank('council')}
        >
          <span className="memory-bank-btn-label">
            <Users size={13} />
            <span>Council Shared</span>
          </span>
          <span className="memory-bank-count">{countsForNav.council ?? 0}</span>
        </button>

        <div className="memory-sidebar-group-label">Council Specialists</div>
        {COUNCIL_SPECIALIST_BANKS.map((bank) => {
          const meta = getRoleMetadata(bank);
          return (
            <button
              key={bank}
              type="button"
              className="memory-bank-btn"
              aria-pressed={selectedBank === bank}
              onClick={() => setSelectedBank(bank)}
            >
              <span>{meta.shortLabel}</span>
              <span className="memory-bank-count">{countsForNav[bank] ?? 0}</span>
            </button>
          );
        })}

        <div className="memory-sidebar-group-label">Production Specialists</div>
        {BOARD_SPECIALIST_BANKS.map((bank) => {
          const meta = getRoleMetadata(bank);
          return (
            <button
              key={bank}
              type="button"
              className="memory-bank-btn"
              aria-pressed={selectedBank === bank}
              onClick={() => setSelectedBank(bank)}
            >
              <span>{meta.label}</span>
              <span className="memory-bank-count">{countsForNav[bank] ?? 0}</span>
            </button>
          );
        })}
      </nav>

      <section className="memory-main-column" aria-label="Long-term game memory ledger">
        <header className="memory-toolbar">
          <div className="memory-toolbar-title-group">
            <h2>
              Long-Term Memory
              {gameTitle || data?.game_title ? ` · ${gameTitle || data?.game_title}` : ''}
            </h2>
            <p className="memory-toolbar-subtitle">
              Persistent cross-slice strategic goals, architectural decisions, and operational lessons curated by{' '}
              <code>ContextCuratorAgent</code> and queryable on demand by every studio agent.
            </p>
          </div>

          <button
            type="button"
            className="btn btn-primary"
            aria-expanded={composerOpen}
            onClick={handleOpenComposer}
          >
            <Plus size={14} />
            <span>{composerOpen ? 'Close Composer' : 'Record Memory'}</span>
          </button>
        </header>

        <div className="memory-filter-bar" role="toolbar" aria-label="Filter memories">
          <div className="memory-filter-pills" role="group" aria-label="Lifecycle status">
            {(
              [
                ['active', 'Active'],
                ['fulfilled', 'Fulfilled'],
                ['superseded', 'Superseded'],
                ['archived', 'Archived'],
                ['all', 'All Statuses'],
              ] as const
            ).map(([val, label]) => (
              <button
                key={val}
                type="button"
                className="memory-filter-pill"
                aria-pressed={selectedStatus === val}
                onClick={() => setSelectedStatus(val)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="memory-filter-pills" role="group" aria-label="Memory category">
            {(
              [
                ['all', 'All Kinds'],
                ['strategic_goal', 'Strategic Goals'],
                ['architectural_decision', 'Architecture'],
                ['lesson_learned', 'Lessons Learned'],
                ['domain_note', 'Domain Notes'],
              ] as const
            ).map(([val, label]) => (
              <button
                key={val}
                type="button"
                className="memory-filter-pill"
                aria-pressed={selectedKind === val}
                onClick={() => setSelectedKind(val)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {memoriesQuery.error ? (
          <ErrorBanner
            error={memoriesQuery.error}
            title="Failed Loading Game Memory Ledger"
            onRetry={() => void memoriesQuery.refetch()}
          />
        ) : null}

        {mutationError ? (
          <ErrorBanner
            error={mutationError}
            title="Memory Mutation Failed"
            onDismiss={() => setMutationError(null)}
          />
        ) : null}

        {composerOpen ? (
          <form
            ref={formRef}
            className="memory-composer-form"
            onSubmit={(e) => void handleSubmitRecord(e)}
            noValidate
          >
            <h3>Record Durable Game or Agent Memory</h3>

            <div className="memory-composer-row">
              <div className="memory-field">
                <label htmlFor="memory-target-bank">Target Memory Bank</label>
                <span id="memory-bank-hint" className="memory-field-hint">
                  Choose Council Shared for game-wide direction or a specialist bank for role-specific craft memory.
                </span>
                <select
                  id="memory-target-bank"
                  value={targetBank}
                  aria-describedby="memory-bank-hint"
                  onChange={(e) => setTargetBank(e.target.value)}
                >
                  <option value="council">Council Shared Memory (council)</option>
                  <optgroup label="Council Specialists">
                    {COUNCIL_SPECIALIST_BANKS.map((b) => (
                      <option key={b} value={b}>
                        {getRoleMetadata(b).label} ({b})
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Production Specialists">
                    {BOARD_SPECIALIST_BANKS.map((b) => (
                      <option key={b} value={b}>
                        {getRoleMetadata(b).label} ({b})
                      </option>
                    ))}
                  </optgroup>
                </select>
              </div>

              <div className="memory-field">
                <label htmlFor="memory-kind">Memory Kind</label>
                <span id="memory-kind-hint" className="memory-field-hint">
                  Classify whether this persists a multi-slice goal, architectural rule, operational lesson, or craft note.
                </span>
                <select
                  id="memory-kind"
                  value={kind}
                  aria-describedby="memory-kind-hint"
                  onChange={(e) => setKind(e.target.value as MemoryKind)}
                >
                  <option value="strategic_goal">Strategic Goal</option>
                  <option value="architectural_decision">Architectural Decision</option>
                  <option value="lesson_learned">Lesson Learned</option>
                  <option value="domain_note">Domain Note</option>
                </select>
              </div>
            </div>

            <div className="memory-field">
              <label htmlFor="memory-deliberation">Epistemic Rationale (Rule 4 · Required First)</label>
              <span id="memory-deliberation-hint" className="memory-field-hint">
                Explain why this memory matters across future tasks or vertical slices (minimum 8 characters).
              </span>
              <textarea
                id="memory-deliberation"
                name="deliberation"
                rows={2}
                required
                minLength={8}
                value={deliberation}
                aria-describedby="memory-deliberation-hint"
                aria-errormessage="memory-deliberation-error"
                onChange={(e) => {
                  setDeliberation(e.target.value);
                  syncControlValidationState(e.currentTarget, interactedControlsRef.current, 'input');
                }}
                onBlur={(e) =>
                  syncControlValidationState(e.currentTarget, interactedControlsRef.current, 'blur')
                }
              />
              <div id="memory-deliberation-error" className="memory-field-error" role="alert">
                <span aria-hidden="true">!</span> Please provide an epistemic rationale of at least 8 characters.
              </div>
            </div>

            <div className="memory-field">
              <label htmlFor="memory-title">Memory Headline</label>
              <span id="memory-title-hint" className="memory-field-hint">
                Concise title identifying the goal, decision, or lesson (minimum 4 characters).
              </span>
              <input
                id="memory-title"
                name="title"
                type="text"
                required
                minLength={4}
                value={title}
                aria-describedby="memory-title-hint"
                aria-errormessage="memory-title-error"
                onChange={(e) => {
                  setTitle(e.target.value);
                  syncControlValidationState(e.currentTarget, interactedControlsRef.current, 'input');
                }}
                onBlur={(e) =>
                  syncControlValidationState(e.currentTarget, interactedControlsRef.current, 'blur')
                }
              />
              <div id="memory-title-error" className="memory-field-error" role="alert">
                <span aria-hidden="true">!</span> Please enter a concise headline of at least 4 characters.
              </div>
            </div>

            <div className="memory-field">
              <label htmlFor="memory-content">Actionable Guidance</label>
              <span id="memory-content-hint" className="memory-field-hint">
                Concrete guidance or constraint that ContextCuratorAgent and querying specialists should apply (minimum 8 characters).
              </span>
              <textarea
                id="memory-content"
                name="content"
                rows={3}
                required
                minLength={8}
                value={content}
                aria-describedby="memory-content-hint"
                aria-errormessage="memory-content-error"
                onChange={(e) => {
                  setContent(e.target.value);
                  syncControlValidationState(e.currentTarget, interactedControlsRef.current, 'input');
                }}
                onBlur={(e) =>
                  syncControlValidationState(e.currentTarget, interactedControlsRef.current, 'blur')
                }
              />
              <div id="memory-content-error" className="memory-field-error" role="alert">
                <span aria-hidden="true">!</span> Please enter actionable guidance of at least 8 characters.
              </div>
            </div>

            <fieldset className="memory-roles-fieldset">
              <legend>Proactive Curation Target Roles (Optional)</legend>
              <span className="memory-field-hint">
                Select additional roles that should proactively receive this memory when ContextCuratorAgent builds their context pack.
              </span>
              <div className="memory-role-chips">
                {allBanksList.map((role) => {
                  const checked = selectedRoles.includes(role);
                  return (
                    <label key={role} className="memory-role-choice">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => handleToggleApplicableRole(role)}
                      />
                      <span>{getRoleMetadata(role).shortLabel}</span>
                    </label>
                  );
                })}
              </div>
            </fieldset>

            <div className="memory-composer-actions">
              <button
                type="button"
                className="btn"
                onClick={() => {
                  resetFormValidation();
                  setComposerOpen(false);
                }}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={mutateMemory.isPending}
              >
                {mutateMemory.isPending ? 'Saving...' : 'Save Memory Entry'}
              </button>
            </div>
          </form>
        ) : null}

        {memoriesQuery.isLoading ? (
          <div className="empty-state-box">Loading per-game memory banks...</div>
        ) : entries.length === 0 ? (
          <div className="empty-state-box">
            No memory entries match the active filters yet. Council and Production specialists automatically record
            multi-slice goals, architectural decisions, and lessons learned as slices progress—or use{' '}
            <strong>Record Memory</strong> above to seed guidance directly.
          </div>
        ) : (
          <div className="memory-cards-grid">
            {entries.map((entry) => {
              const bankMeta = getRoleMetadata(entry.memory_bank);
              const authorMeta = getRoleMetadata(entry.author_role);
              return (
                <article
                  key={entry.memory_id}
                  className="memory-card"
                  data-status={entry.status}
                  data-kind={entry.kind}
                >
                  <header className="memory-card-header">
                    <div className="memory-card-badges">
                      <span className="memory-badge memory-badge-bank">{bankMeta.label}</span>
                      <span className="memory-badge" data-kind={entry.kind}>
                        {KIND_LABELS[entry.kind] ?? entry.kind}
                      </span>
                      <span className="memory-badge" data-status={entry.status}>
                        {STATUS_LABELS[entry.status] ?? entry.status}
                      </span>
                    </div>
                    <h3>{entry.title}</h3>
                    <div className="memory-card-meta">
                      <span>Author: {authorMeta.shortLabel}</span>
                      {entry.slice_id ? <span>Slice: {entry.slice_id}</span> : null}
                      {entry.item_id ? <span>Task: {entry.item_id}</span> : null}
                      <span>ID: {entry.memory_id}</span>
                    </div>
                  </header>

                  <div className="memory-card-content">
                    <p className="memory-card-body">{entry.content}</p>

                    {entry.resolution_note ? (
                      <p className="memory-resolution-note">
                        <strong>Resolution:</strong> {entry.resolution_note}
                      </p>
                    ) : null}

                    {entry.superseded_by_id ? (
                      <p className="memory-resolution-note">
                        <strong>Superseded by:</strong> <code>{entry.superseded_by_id}</code>
                      </p>
                    ) : null}

                    {entry.applicable_roles.length > 0 ? (
                      <div className="memory-applicable-list" aria-label="Applicable roles">
                        <span className="memory-applicable-label">Curated for:</span>
                        {entry.applicable_roles.map((r) => (
                          <span key={r} className="memory-role-tag">
                            {getRoleMetadata(r).shortLabel}
                          </span>
                        ))}
                      </div>
                    ) : null}

                    <DeliberationDisclosure
                      deliberation={entry.deliberation}
                      label="Epistemic Rationale"
                    />
                  </div>

                  <footer className="memory-card-footer">
                    {entry.status === 'active' ? (
                      <>
                        <button
                          type="button"
                          className="btn memory-action-btn"
                          disabled={mutateMemory.isPending}
                          onClick={() => handleLifecycleAction(entry, 'fulfill')}
                        >
                          <CheckCircle2 size={13} />
                          <span>Mark Fulfilled</span>
                        </button>
                        <button
                          type="button"
                          className="btn memory-action-btn"
                          disabled={mutateMemory.isPending}
                          onClick={() => handleLifecycleAction(entry, 'archive')}
                        >
                          <Archive size={13} />
                          <span>Archive</span>
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        className="btn memory-action-btn"
                        disabled={mutateMemory.isPending}
                        onClick={() => handleLifecycleAction(entry, 'update')}
                      >
                        <RotateCcw size={13} />
                        <span>Restore Active</span>
                      </button>
                    )}
                  </footer>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
