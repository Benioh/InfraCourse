// ============================================================================
// Quiz Gate types — pre-patch comprehension check.
// ============================================================================

export type QuizPrereqItem = {
  title: string;
  kind: "read" | "notebook" | "source" | string;
  path: string;
  href?: string;
  estimate_min?: number;
};

export type QuizOption = {
  id: string;
  text: string;
};

export type QuizQuestion = {
  id: string;
  prompt: string;
  options: QuizOption[];
};

export type QuizStatus = {
  mission: string;
  passed: boolean;
  score?: number;
  total?: number;
  passed_at?: string;
  never_attempted?: boolean;
};

export type QuizPayload = {
  mission: string;
  title?: string;
  description?: string;
  pass_threshold?: number;
  prereq_checklist?: QuizPrereqItem[];
  questions: QuizQuestion[];
  total_questions: number;
  status?: QuizStatus | null;
};

export type QuizQuestionResult = {
  id: string;
  your_answer?: string | null;
  expected?: string | null;
  correct: boolean;
  explanation: string;
};

export type QuizSubmissionResult = {
  mission: string;
  score: number;
  total: number;
  score_ratio: number;
  threshold: number;
  passed: boolean;
  results: QuizQuestionResult[];
  submitted_at: string;
};

// ============================================================================
// Patch Track types — first-class citizens, drive the Mission Detail page.
// ============================================================================

export type PatchSummary = {
  passed: number;
  failed: number;
  total: number;
};

export type PatchStatus = {
  mission: string;
  passed: boolean;
  exit_code?: number;
  started_at?: string;
  finished_at?: string;
  duration_s?: number;
  summary?: PatchSummary;
  output?: string;
  output_truncated?: boolean;
  never_run?: boolean;
  error?: string;
};

export type PatchPayload = {
  mission: string;
  title?: string;
  description?: string;
  test_command?: string;
  test_count?: number;
  test_kind?: string;
  task_md_path?: string;
  task_md_content?: string | null;
  starter_file?: string;
  reference_file?: string;
  status?: PatchStatus | null;
  quiz_passed?: boolean;  // ★ gate
  lesson?: LessonPayload;
  lesson_docs?: LessonDocItem[];
  source_reading?: SourceReadingItem[];
  notebooks?: string[];
  mini_infra_targets?: string[];
  tickets?: string[];
  prereq?: string;
  next_lab?: string;
  estimated_minutes?: number;
  no_gpu_friendly?: boolean;
};

// ============================================================================
// Quest (now a thin metadata shell — Patch Track is the real content).
// ============================================================================

export type Quest = {
  id: string;
  title: string;
  act?: string;
  level?: string;
  role?: string;
  priority?: string;
  frameworks?: string[];
  gpu_modes?: Record<string, string>;
  estimated_minutes?: number;
  no_gpu_friendly?: boolean;
  patch?: {
    description?: string;
    task_md?: string;
    starter_file?: string;
    reference_file?: string;
    test_command?: string;
    status_file?: string;
    test_count?: number;
    test_kind?: string;
  };
  lesson?: LessonPayload;
  lesson_docs?: LessonDocItem[];
  source_reading?: SourceReadingItem[];
  notebooks?: string[];
  mini_infra_targets?: string[];
  tickets?: string[];
  prereq?: string;
  next_lab?: string;
  // Legacy fields (rendered if present, ignored if absent during migration):
  datasets?: { name: string; source?: string; required?: boolean }[];
  learning_goals?: string[];
  success_criteria?: string[];
  notebook_bridge?: NotebookBridgeItem[];
  project?: QuestProject;
  commands?: Record<string, string>;
  badge?: { name?: string; pass_score?: number };
};

export type LessonSourceRef = {
  repo_path: string;
  lines?: [number, number] | string;
  title?: string;
  note?: string;
};

export type LessonSection = {
  title: string;
  plain_explanation?: string;
  walkthrough?: string;
  mental_model?: string;
  why_it_matters?: string;
  common_confusions?: string[];
  source_refs?: LessonSourceRef[];
  checkpoint_questions?: string[];
};

export type LessonPayload = {
  opening?: string;
  sections?: LessonSection[];
};

export type LessonDocItem = {
  title: string;
  path: string;
  description?: string;
};

export type SourceReadingHighlight = {
  /** Either [start, end] inclusive 1-indexed numbers, or a string like "L120-L145" / "120-145". */
  lines: [number, number] | string;
  /** Side-by-side annotation explaining what's interesting about this region. Markdown allowed. */
  note: string;
  /** Optional sub-title displayed above the snippet. */
  title?: string;
};

export type SourceReadingItem = {
  title: string;
  repo_path: string;
  focus?: string;
  /** Long-form, beginner-friendly markdown walkthrough for this source file.
   * Rendered between `focus` and `highlights` in the SourceReadingCard.
   * Use for: 入口/生命周期/不变量/常见误解/与本关 patch 的关系。 */
  walkthrough?: string;
  questions?: string[];
  /** Optional curated highlights — when present, the lab page renders the
   * file inline with these regions called out alongside the note text. */
  highlights?: SourceReadingHighlight[];
};

export type StarterPayload = {
  mission: string;
  path: string;
  content: string;
  exists: boolean;
  bytes?: number;
  size?: number;
  language?: string;
};

export type NotebookBridgeItem = {
  notebook: string;
  before?: string;
  after?: string;
};

export type QuestProject = {
  name: string;
  scenario?: string;
  milestones?: string[];
  deliverables?: string[];
  stretch?: string[];
};

export type Ticket = {
  id: string;
  title: string;
  mission?: string;
  severity?: string;
  symptom?: string;
  error_signatures?: string[];
  classification_questions?: string[];
  likely_causes?: string[];
  minimal_checks?: string[];
  minimal_fixes?: string[];
  what_not_to_do?: string[];
  hint_1?: string;
  hint_2?: string;
  solution?: string;
};

export type BadgeRow = {
  mission: string;
  level?: string;
  title?: string;
  passed: boolean;
  test_count?: number | null;
  summary?: PatchSummary | null;
  finished_at?: string | null;
};

// ============================================================================
// Dashboard / Curriculum (now driven by patch_status, not grade.json).
// ============================================================================

export type DashboardMission = {
  id: string;
  title: string;
  level?: string;
  role?: string;
  frameworks?: string[];
  gpu_modes?: Record<string, string>;
  estimated_minutes?: number;
  no_gpu_friendly?: boolean;
  patch_test_command?: string;
  patch_status?: PatchStatus | null;
  status: string;
};

export type DashboardPayload = {
  current_level?: string | null;
  current_role?: string | null;
  active_mission?: string | null;
  completed_missions: number;
  total_missions: number;
  missions: DashboardMission[];
};

export type CurriculumMission = {
  id: string;
  level?: string;
  title?: string;
  act?: string;
  role?: string;
  frameworks?: string[];
  patch_description?: string;
  patch_test_count?: number;
  lesson_section_count?: number;
  source_count: number;
  notebook_count: number;
  ticket_count: number;
  recommended_command?: string | null;
  status: string;
  patch_status?: PatchStatus | null;
};

export type CurriculumPhase = {
  name: string;
  missions: CurriculumMission[];
  frameworks: string[];
  source_count: number;
  notebook_count: number;
  completed_missions: number;
};

export type CurriculumPayload = {
  title: string;
  principle: string;
  stats: {
    total_missions: number;
    completed_missions: number;
    total_source_nodes: number;
    total_notebooks: number;
    total_projects: number;
  };
  current_mission?: CurriculumMission | null;
  self_study_loop: { name: string; duration: string; action: string }[];
  stuck_playbook: { symptom: string; action: string }[];
  completion_checks: string[];
  docs: { title: string; path: string; summary: string }[];
  phases: CurriculumPhase[];
  missions: CurriculumMission[];
};

// ============================================================================
// Notebook / Source viewer (unchanged from previous version)
// ============================================================================

export type NotebookOutput = {
  type: "text" | "image/png";
  body: string;
};

export type NotebookCell = {
  index: number;
  cell_type: "markdown" | "code" | "raw" | string;
  source: string;
  execution_count?: number | null;
  outputs?: NotebookOutput[];
};

export type NotebookPayload = {
  path: string;
  title: string;
  cell_count: number;
  cells: NotebookCell[];
};

export type SourcePayload = {
  path: string;
  language: string;
  size: number;
  line_count: number;
  content: string;
  truncated: boolean;
  reason?: string;
};

export type SourceTreeEntry = {
  name: string;
  path: string;
  type: "dir" | "file" | "binary";
  size: number | null;
};

export type SourceTreePayload = {
  path: string;
  entries: SourceTreeEntry[];
};
