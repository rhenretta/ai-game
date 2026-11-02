/**
 * Epistemic & Studio Domain Contracts mirroring `engine/ui/models.py` and `engine/ui/server.py`.
 * Honors Rule 4: `deliberation` is declared strictly first (property index 0) across all models.
 */

export type WorkspaceId =
  | 'board'
  | 'writers-room'
  | 'bible'
  | 'memory'
  | 'assets'
  | 'playtest'
  | 'dna';

export interface GameCatalogEntry {
  cartridge_id: string;
  game_title: string;
  core_genre: string;
  status: string;
  has_backlog: boolean;
  approved_items: number;
  total_items: number;
  updated_at_utc: string;
}

export interface CartridgesResponse {
  cartridges: string[];
  games: GameCatalogEntry[];
}

export interface LastBuildError {
  event_type: string;
  agent_name: string;
  item_id?: string;
  message: string;
  timestamp_utc: string;
}

export interface ProcessStatus {
  process_state: 'running' | 'paused' | 'pausing' | 'awaiting_greenlight' | 'stopped' | string;
  is_process_alive: boolean;
  is_paused: boolean;
  pause_reason?: string | null;
  pause_for_greenlight: boolean;
  build_actively_running: boolean;
  lock_held: boolean;
  active_threads: string[];
  daemon_pid?: number | null;
  scheduler_running_workers: number;
  orchestrator_state: string;
  last_event_timestamp_utc?: string | null;
  last_error?: LastBuildError | null;
}

export interface TokenUsageSummary {
  total_calls: number;
  calls_with_cache_hit?: number;
  cache_hit_call_rate_pct?: number;
  total_prompt_tokens: number;
  total_cached_tokens: number;
  uncached_prompt_tokens?: number;
  cached_token_share_pct?: number;
  total_output_tokens: number;
  total_thought_tokens: number;
  total_tokens: number;
  image_generations?: number;
  tts_generations?: number;
  tts_audio_seconds?: number;
  embedding_calls?: number;
  embedding_estimated_tokens?: number;
  playtest_calls?: number;
  playtest_tokens?: number;
  playtest_estimated_cost_usd?: number;
  estimated_cost_usd?: number;
  estimated_cost_usd_2027_std?: number;
  by_label?: Record<string, Record<string, number>>;
}

export interface StudioStatusResponse {
  deliberation: string;
  status: string;
  active_cartridge: string;
  cartridges: string[];
  database_connected: boolean;
  db_event_count: number;
  supervisor_state: string;
  current_episode: number;
  uptime_seconds: number;
  token_usage?: TokenUsageSummary | Record<string, unknown>;
}

export interface VerticalSliceSpec {
  deliberation?: string;
  slice_id: string;
  slice_index: number;
  title: string;
  slice_objective?: string;
  player_acclimation_and_mechanics_focus?: string;
  expansion_from_prior_slice?: string;
  scope_boundaries?: string;
  status?: string;
}

export interface ShapingNote {
  deliberation?: string;
  specialist_role?: string;
  author_role?: string;
  direction?: string;
  note?: string;
  timestamp_utc?: string;
}

export type WorkItemState =
  | 'approved'
  | 'in_progress'
  | 'waiting_for_lock'
  | 'awaiting_review'
  | 'changes_requested'
  | 'ready'
  | 'blocked'
  | 'failed'
  | 'needs_decision'
  | 'cancelled'
  | string;

export interface WorkItemRevision {
  deliberation?: string;
  revision_index?: number;
  worker_notes?: string;
  proof_paths?: string[];
  proof_urls?: string[];
  verdict?: string;
  feedback?: string;
  required_changes?: string[];
  timestamp_utc?: string;
}

export interface YieldedTurnInfo {
  completed_turns?: number;
  next_turn_index?: number;
  specialist_agent?: string;
  target_paths?: string[];
  brief?: string;
}

export interface ProductionBoardItem {
  deliberation: string;
  item_id: string;
  kind: string;
  title: string;
  spec: string;
  proof_spec?: string;
  depends_on: string[];
  target_modules: string[];
  raised_by_role?: string;
  slice_id?: string;
  spawned_by_item_id?: string | null;
  design_revision?: number;
  state: WorkItemState;
  max_revisions?: number;
  spec_resets?: number;
  revisions?: Array<WorkItemRevision | Record<string, unknown>>;
  accepted_compromise?: string | null;
  failure?: string | null;
  failure_count?: number;
  attempt_count?: number;
  priority?: number;
  decision_reason?: string | null;
  shaping_notes?: Array<ShapingNote | string>;
  waiting_on_locks?: string[];
  lock_holders?: string[] | Record<string, string>;
  yielded_turn_info?: YieldedTurnInfo | null;
  latest_proof_urls?: string[];
}

export interface ProductionBoardSummary {
  deliberation: string;
  cartridge_id: string;
  design_revision: number;
  total_items: number;
  counts_by_state: Record<string, number>;
  is_drained: boolean;
  items: ProductionBoardItem[];
}

export interface WritersRoomUtterance {
  deliberation?: string;
  utterance_id?: string;
  turn_index?: number;
  speaker_role: string;
  speaker_display_name?: string;
  spoken_dialogue?: string;
  message?: string;
  addressed_to?: string[];
  referenced_roles?: string[];
  bible_updates_summary?: string[];
  proof_image_paths?: string[];
  artifact_emitted?:
    | string
    | {
        type?: string;
        path?: string;
        url?: string;
        image_path?: string;
        item_ids?: string[];
      }
    | null;
  slice_id?: string;
  timestamp_utc?: string;
}

export interface SceneLayerSpec {
  deliberation?: string;
  layer_id: string;
  depth_order: number;
  scroll_parallax_factor?: [number, number];
  visual_description: string;
  associated_asset_ids?: string[];
}

export interface VisualMockupSpec {
  deliberation?: string;
  mockup_id: string;
  title: string;
  scene_description: string;
  image_path?: string;
  url?: string;
  aspect_ratio?: string;
  camera_and_spatial_paradigm?: string;
  color_palette_hex?: string[];
  scene_layers?: SceneLayerSpec[];
  ui_callout_notes?: string[];
  created_at_utc?: string;
}

export interface StoryboardBeat {
  deliberation?: string;
  beat_index: number;
  phase_title: string;
  player_emotional_state?: string;
  sensory_and_camera_staging?: string;
  diegetic_teaching_moment?: string;
  narrative_integration?: string;
}

export interface ConceptEvolutionEntry {
  deliberation?: string;
  entry_id?: string;
  version?: number;
  turn_index?: number;
  author_role?: string;
  hypothesis?: string;
  previous_premise?: string;
  evolved_premise?: string;
  pivot_reason?: string;
  critique_or_pivot_reason?: string;
  timestamp_utc?: string;
}

export interface NarrativeCharacter {
  deliberation?: string;
  character_id: string;
  name: string;
  role?: string;
  voice_profile?: string;
  personality_traits?: string[];
  backstory?: string;
  portrait_asset?: string;
}

export interface DialogueBark {
  deliberation?: string;
  bark_id?: string;
  speaker_id?: string;
  speaker_name?: string;
  trigger_event?: string;
  line?: string;
  text?: string;
  audio_file?: string;
  voice_clip?: string;
}

export interface NarrativeBible {
  deliberation?: string;
  bible_id?: string;
  world_title?: string;
  setting_summary?: string;
  lore_premise?: string;
  characters?: NarrativeCharacter[];
  cutscenes?: Array<Record<string, unknown>>;
  dialogue_barks?: DialogueBark[];
}

export interface GameBibleResponse {
  deliberation: string;
  has_bible: boolean;
  bible_id?: string | null;
  cartridge_id: string;
  active_candidate_id?: string;
  status: string;
  game_title: string;
  core_genre: string;
  elevator_pitch?: string;
  creative_vision_summary?: string;
  core_fantasy?: string;
  target_player_dna_summary?: string;
  first_60_seconds_hook?: string;
  diegetic_ui_philosophy?: string;
  camera_and_spatial_paradigm?: string;
  color_palette_hex?: string[];
  gameplay_pillars?: Array<string | Record<string, unknown>>;
  core_loop?: Record<string, unknown> | null;
  onboarding_plan?: Record<string, unknown> | null;
  goal_hierarchy?: Record<string, unknown> | null;
  engagement_plan?: Record<string, unknown> | null;
  level_generation_strategy?: Record<string, unknown> | string | null;
  technical_architecture_blueprint?: Record<string, unknown> | Array<Record<string, unknown>> | string | null;
  visual_style_guide?: Record<string, unknown> | string | null;
  audio_direction_guide?: Record<string, unknown> | string | null;
  slice_roadmap?: VerticalSliceSpec[];
  active_slice_index?: number;
  writers_room_transcript?: WritersRoomUtterance[];
  visual_mockups?: VisualMockupSpec[];
  experience_storyboard?: StoryboardBeat[];
  concept_evolution?: ConceptEvolutionEntry[];
  concept_evolution_log?: ConceptEvolutionEntry[];
  concept_fragments?: Array<Record<string, unknown>>;
  concept_candidates?: Array<Record<string, unknown>>;
  world_spec?: Record<string, unknown> | null;
  narrative_bible?: NarrativeBible | null;
  pause_for_greenlight?: boolean;
  build_actively_running?: boolean;
  build_paused?: boolean;
  orchestrator_state?: string;
  process_status?: ProcessStatus;
  production_board?: ProductionBoardSummary;
  updated_at_utc?: string;
}

export interface ChronologyCodeDiff {
  file?: string;
  module?: string;
  path?: string;
  diff?: string;
}

export interface ChronologyStepLogEntry {
  timestamp_utc?: string;
  step_type: string;
  specialist_agent?: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface ChronologySelfInquiry {
  deliberation?: string;
  agent_role?: string;
  question: string;
  answer: string;
  researched_chunk_ids?: string[];
}

export interface ChronologyContextPack {
  deliberation?: string;
  filename?: string;
  target_agent_role: string;
  revision?: number;
  total_indexed_chunks?: number;
  selected_chunk_ids?: string[];
  selected_image_paths?: string[];
  selected_image_urls?: string[];
  role_tailored_synthesis?: string;
  search_queries_used?: string[];
}

export interface ChronologyProofArtifact {
  path: string;
  filename: string;
  url: string;
  is_audio?: boolean;
}

export interface ChronologyReviewerBreakdown {
  reviewer_role: string;
  raw_text: string;
  points: string[];
}

export interface ChronologyCouncilReview {
  deliberation?: string;
  revision_index?: number;
  verdict?: string;
  approved?: boolean;
  summary?: string;
  feedback?: string;
  reviewer_breakdown?: ChronologyReviewerBreakdown[];
  worker_notes?: string;
  proof_artifacts?: Array<ChronologyProofArtifact | string>;
}

export interface ChronologyFileEntry {
  path: string;
  is_binary: boolean;
  content?: string;
  url?: string;
  line_count?: number;
  size_bytes?: number;
}

export interface ChronologyTurn {
  deliberation?: string;
  turn_index: number;
  specialist_agent: string;
  phase?: string;
  brief?: string;
  summary?: string;
  verification_passed?: boolean;
  touched_files?: string[];
  code_diffs?: Array<ChronologyCodeDiff | string> | Record<string, string>;
  step_log?: Array<ChronologyStepLogEntry | Record<string, unknown> | string>;
  curated_context?: Record<string, unknown> | string | null;
  prompt_text?: string;
  self_inquiries?: ChronologySelfInquiry[];
  lead_deliberation?: string;
  lead_prompt_text?: string;
  lead_curated_context?: Record<string, unknown> | string | null;
  lead_self_inquiries?: ChronologySelfInquiry[];
  proof_images?: string[];
  proof_urls?: Array<ChronologyProofArtifact | string>;
  timestamp_utc?: string;
  started_at_utc?: string;
}

export interface ChronologyAttempt {
  attempt_key: string;
  revision_index: number;
  attempt_number: number;
  is_failed_retry: boolean;
  status: string;
  current_strategy?: string;
  proof_strategy?: string;
  submission_summary?: string;
  turns: ChronologyTurn[];
  council_review?: ChronologyCouncilReview | null;
  pending_decision?: Record<string, unknown> | null;
  active_turn?: Partial<ChronologyTurn> | null;
}

export interface BoardItemChronologyResponse {
  deliberation: string;
  cartridge_id: string;
  item_id: string;
  title: string;
  kind: string;
  state: WorkItemState;
  raised_by_role?: string;
  spawned_by_item_id?: string | null;
  shaping_notes?: Array<ShapingNote | string>;
  spec: string;
  proof_spec?: string;
  target_modules: string[];
  depends_on: string[];
  failure?: string | null;
  failure_count?: number;
  spec_resets?: number;
  context_packs?: ChronologyContextPack[];
  self_inquiries?: ChronologySelfInquiry[];
  attempts: ChronologyAttempt[];
  file_contents?: Record<string, ChronologyFileEntry | string>;
}

export interface AssetFileItem {
  filename: string;
  url: string;
  size_bytes: number;
}

export interface ArtifactInventoryResponse {
  deliberation: string;
  exists: boolean;
  manifest: Record<string, unknown>;
  audio_clips: AssetFileItem[];
  sprites: AssetFileItem[];
  code_modules: string[];
}

export interface ArtifactContentResponse {
  is_image?: boolean;
  is_audio?: boolean;
  url?: string;
  content?: string;
  filename: string;
  size_bytes: number;
  mime_type?: string;
}

export interface VerifiedCheckpointRecord {
  deliberation: string;
  checkpoint_id: string;
  candidate_id: string;
  requirement_id: string;
  track_title: string;
  created_at_utc: string;
  verified_tracks: string[];
  cartridge_path: string;
  is_active: boolean;
  files: string[];
}

export interface CheckpointsResponse {
  deliberation: string;
  total_checkpoints: number;
  latest_checkpoint_id?: string | null;
  checkpoints: VerifiedCheckpointRecord[];
}

export interface PlayableTarget {
  target_type: 'active' | 'completed_slice' | 'checkpoint' | string;
  checkpoint_id: string;
  candidate_id?: string;
  slice_id?: string | null;
  label: string;
  game_title: string;
  game_description?: string;
}

export interface PlaytestExperienceReport {
  deliberation?: string;
  overall_experience_summary?: string;
  visual_and_ui_defects_observed?: string[];
  graphics_and_visuals?: string;
  gameplay_and_feel?: string;
  sound_and_audio?: string;
  intuitiveness_and_onboarding?: string;
  goals_pacing_and_accomplishment?: string;
  world_and_atmosphere?: string;
  key_highlights?: string[];
  key_pain_points?: string[];
  bugs_and_glitches_observed?: string[];
  unanswered_questions_and_mysteries?: string[];
  overall_fun_rating?: number;
  dispatched_to_council_at_utc?: string | null;
  dispatched_producer_note?: string | null;
}

export interface PlaytestSessionSummary {
  deliberation: string;
  session_id: string;
  checkpoint_id: string;
  candidate_id?: string;
  slice_id?: string | null;
  slice_index?: number | null;
  slice_title?: string | null;
  game_title: string;
  game_description?: string;
  status: string;
  initiated_by?: 'council' | 'operator' | string;
  end_outcome?: string | null;
  end_summary?: string | null;
  error_message?: string | null;
  turn_count: number;
  sim_ticks_total?: number;
  sim_duration_sec?: number;
  current_plan?: string;
  current_mood?: string;
  prompt_tokens?: number;
  cached_tokens?: number;
  output_tokens?: number;
  has_video?: boolean;
  latest_frame_name?: string | null;
  overall_fun_rating?: number | null;
  overall_experience_summary?: string | null;
  bugs_count?: number;
  wishes_count?: number;
  pending_wishes_count?: number;
  report_sent_to_council_at_utc?: string | null;
  report_producer_note?: string | null;
  matches_active_cartridge?: boolean;
  started_at_utc?: string;
  updated_at_utc?: string;
  ended_at_utc?: string | null;
  rulebook?: Array<Record<string, unknown> | string> | Record<string, unknown>;
  field_notes?: Array<Record<string, unknown> | string> | Record<string, unknown>;
  questions?: Array<Record<string, unknown> | string> | Record<string, unknown>;
  bugs_found?: Array<Record<string, unknown> | string> | Record<string, unknown>;
  impressions?: Record<string, string>;
  session_report?: PlaytestExperienceReport | null;
  compacted_journal?: Array<Record<string, unknown> | string>;
}

export interface PlaytestTurnActionPx {
  target_x?: number;
  target_y?: number;
  drag_from_x?: number;
  drag_from_y?: number;
  has_explicit_mouse?: boolean;
  mouse_button_action?: string;
  hold_keys?: string[];
  tap_keys?: string[];
  duration_ms?: number;
  x?: number;
  y?: number;
  end_x?: number;
  end_y?: number;
}

export interface PlaytestTurn {
  turn_id: string;
  session_id: string;
  turn_index: number;
  sim_tick_start?: number;
  sim_tick_end?: number;
  video_timestamp_sec?: number;
  cursor_start_x?: number;
  cursor_start_y?: number;
  cursor_end_x?: number;
  cursor_end_y?: number;
  current_plan?: string;
  deliberation: string;
  append_log_line?: string;
  mood_state?: string;
  mood_note?: string;
  action?: Record<string, unknown>;
  action_px?: PlaytestTurnActionPx;
  screenshot_name?: string | null;
  audio_clip_name?: string | null;
  field_note_updates?: Array<Record<string, unknown> | string>;
  question_updates?: Array<Record<string, unknown> | string>;
  rulebook_updates?: Array<Record<string, unknown> | string>;
  bug_reports?: Array<Record<string, unknown> | string>;
  impression_updates?: Array<Record<string, unknown> | string> | Record<string, string>;
  prompt_tokens?: number;
  cached_tokens?: number;
  output_tokens?: number;
  thought_tokens?: number;
  created_at_utc?: string;
}

export interface PlaytestWish {
  deliberation?: string;
  wish_id: string;
  session_id: string;
  candidate_id?: string;
  turn_index?: number;
  category?: string;
  priority?: string;
  title?: string;
  wish_text?: string;
  description?: string;
  rationale?: string;
  status: 'pending' | 'sent_to_council' | 'dismissed' | string;
  screenshot_name?: string | null;
  created_at_utc?: string;
}

export interface PlaytestSessionsListResponse {
  deliberation: string;
  sessions: PlaytestSessionSummary[];
  playable_targets: PlayableTarget[];
  games_catalog: Array<Record<string, unknown>>;
  active_cartridge_id: string;
  pending_wishes_count: number;
}

export interface PlaytestSessionDetailResponse {
  deliberation: string;
  session: PlaytestSessionSummary;
  turns: PlaytestTurn[];
  compactions: Array<Record<string, unknown>>;
  wishes: PlaytestWish[];
  token_summary?: Record<string, number>;
}

export interface PlaytestReportDispatchResponse {
  deliberation: string;
  status: 'council_resumed' | 'queued_for_active_council' | string;
  candidate_id: string;
  auto_resumed_job_id?: string | null;
  session: PlaytestSessionSummary;
  wishes: PlaytestWish[];
  proof_image_paths?: string[];
}

export interface SteamTopGame {
  appid?: number;
  name: string;
  playtime_hours: number;
  playtime_share?: number;
  playtime_2weeks_minutes?: number | null;
  playtime_2weeks_hours?: number | null;
  rtime_last_played?: number | null;
  last_played_date?: string;
  days_since_played?: number | null;
  genres?: string[];
  categories?: string[];
  tags?: string[];
  short_description?: string;
  achievements_unlocked?: number;
  achievements_total?: number;
}

export interface SteamActivityWindow {
  window_key?: string;
  label?: string;
  days?: number | null;
  games_count?: number;
  active_games_count?: number;
  total_playtime_hours?: number;
  recent_2weeks_hours?: number;
  top_genres?: string[];
  genre_weights?: Record<string, number>;
  category_weights?: Record<string, number>;
  games?: SteamTopGame[];
  top_games?: SteamTopGame[];
}

export interface SteamDnaResponse {
  deliberation: string;
  player_id: string;
  profile_tag: string;
  total_playtime_hours: number;
  total_games_analyzed: number;
  top_genres: string[];
  genre_weights: Record<string, number>;
  top_categories: string[];
  category_weights: Record<string, number>;
  playtime_distribution: Record<string, number>;
  top_games: SteamTopGame[];
  activity_windows: Record<string, SteamActivityWindow>;
  pacing_preference: string;
  pacing_score: number;
  challenge_tolerance: number;
  thematic_biases: string[];
  thematic_weights: Record<string, number>;
  mechanical_biases: Record<string, number>;
  preferred_palette_mood: string;
  palette_weights: Record<string, number>;
  created_at_utc: string;
  ingestion_provenance: Record<string, unknown>;
  live_writers_room_dna_directive: string;
}

export interface SwarmStreamEvent {
  deliberation: string;
  event_id?: string;
  build_id?: string;
  event_type: string;
  agent_name: string;
  timestamp_utc: string;
  payload: Record<string, unknown>;
}

export interface EventsHistoryResponse {
  deliberation: string;
  events: SwarmStreamEvent[];
  process_status?: ProcessStatus;
}

export type MemoryKind =
  | 'strategic_goal'
  | 'architectural_decision'
  | 'lesson_learned'
  | 'domain_note';

export type MemoryStatus = 'active' | 'fulfilled' | 'superseded' | 'archived';

export type MemoryMutationOperation =
  | 'record'
  | 'update'
  | 'fulfill'
  | 'supersede'
  | 'archive';

export interface GameMemoryEntry {
  deliberation: string;
  memory_id: string;
  memory_bank: string;
  author_role: string;
  kind: MemoryKind;
  status: MemoryStatus;
  title: string;
  content: string;
  applicable_roles: string[];
  slice_id: string;
  item_id: string;
  resolution_note: string;
  superseded_by_id: string;
  created_at_utc: string;
  updated_at_utc: string;
}

export interface GameMemoryListResponse {
  deliberation: string;
  cartridge_id: string;
  game_title: string;
  updated_at_utc: string;
  total_count: number;
  active_count: number;
  bank_counts: Record<string, number>;
  total_bank_counts?: Record<string, number>;
  banks: string[];
  entries: GameMemoryEntry[];
  mutated_entries?: GameMemoryEntry[];
}

export interface GameMemoryMutationRequest {
  deliberation: string;
  cartridge?: string;
  author_role?: string;
  operation: MemoryMutationOperation;
  target_memory_id?: string;
  target_bank?: string;
  kind?: MemoryKind;
  title?: string;
  content?: string;
  applicable_roles?: string[];
  resolution_note?: string;
  slice_id?: string;
  item_id?: string;
}
