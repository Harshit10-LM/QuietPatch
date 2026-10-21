export type Severity = 'critical' | 'high' | 'medium' | 'low' | 'informational';
export type FindingCategory = 'todo' | 'documentation' | 'configuration' | 'python' | 'javascript' | 'general';
export type VerificationStatus = 'unverified' | 'passed' | 'failed' | 'skipped';
export type PatchStatus = 'pending' | 'approved' | 'rejected' | 'applied' | 'undone';

export interface Finding {
  id: string;
  category: FindingCategory;
  severity: Severity;
  title: string;
  file_path: string;
  line_number?: number | null;
  evidence: string;
  explanation: string;
  suggested_action: string;
  confidence: number;
  verification_status: VerificationStatus;
  ai_assisted: boolean;
  is_demo?: boolean;
  ai_model?: string | null;
  estimated_effort_mins: number;
  created_at: string;
}

export interface ProposedPatch {
  id: string;
  finding_id: string;
  file_path: string;
  change_summary: string;
  original_content: string;
  proposed_content: string;
  diff: string;
  explanation: string;
  risks_assumptions: string;
  suggested_verification: string;
  status: PatchStatus;
  is_demo?: boolean;
  ai_model?: string | null;
  backup_path?: string | null;
  created_at: string;
  applied_at?: string | null;
}

export interface StageRecord {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'partial' | 'failed' | 'skipped';
  duration_seconds: number;
  details?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface MaintenanceRun {
  id: string;
  repository_name: string;
  repository_path: string;
  repository_type: 'local' | 'sample' | 'github';
  status: 'discovering' | 'scanning' | 'analyzing' | 'prioritizing' | 'proposing' | 'completed' | 'failed';
  stages: Record<string, StageRecord>;
  files_scanned: number;
  findings: Finding[];
  patches: ProposedPatch[];
  ai_provider: string;
  ai_model: string;
  duration_seconds: number;
  time_saved_minutes: number;
  created_at: string;
  completed_at?: string | null;
  error?: string | null;
}

export interface TimeSavedSummary {
  total_minutes: number;
  total_hours: number;
  breakdown_by_category: Record<string, number>;
  breakdown_by_status: Record<string, number>;
  assumptions_note: string;
}

export interface MaintenanceReport {
  run_id: string;
  repository_name: string;
  repository_path: string;
  repository_type: string;
  generated_at: string;
  ai_provider: string;
  ai_model: string;
  total_duration_seconds: number;
  files_scanned: number;
  total_findings: number;
  findings_by_severity: Record<string, number>;
  findings_by_category: Record<string, number>;
  fixes_proposed: number;
  fixes_applied: number;
  checks_passed: number;
  checks_failed: number;
  checks_skipped: number;
  time_saved: TimeSavedSummary;
  remaining_work: Array<{
    id: string;
    title: string;
    severity: string;
    file_path: string;
    action: string;
  }>;
  stage_durations: Record<string, number>;
}

export interface PublicConfig {
  app_mode: 'local' | 'hosted';
  ai_provider: 'ollama' | 'gemma_api' | 'demo';
  ollama_base_url: string;
  ollama_model: string;
  ollama_request_timeout_seconds?: number;
  ollama_keep_alive?: string;
  ollama_batch_size?: number;
  gemma_api_model: string;
  has_gemma_api_key: boolean;
  allowed_root: string;
  sentry_configured: boolean;
  sentry_environment: string;
  sentry_sample_rate: number;
  max_repo_files: number;
  max_repo_bytes: number;
  max_scan_seconds: number;
  sample_repo_available: boolean;
}

export interface ProviderStatus {
  ai: {
    available: boolean;
    reachable?: boolean;
    model_installed?: boolean;
    status?: 'connected' | 'model_missing' | 'offline' | 'error' | 'demo' | 'not_configured';
    model: string;
    message: string;
    instruction?: string | null;
    active_provider: string;
    app_mode: string;
    installed_models?: string[];
    privacy_notice?: string;
    is_demo?: boolean;
  };
  observability: {
    status: 'connected' | 'not_configured';
    message: string;
    environment: string;
    sample_rate: number;
    configured: boolean;
  };
}
