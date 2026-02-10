// System Types
export interface System {
  id: string;
  name: string;
  system_type: string;
  serial_number?: string;
  model?: string;
  status: 'active' | 'inactive' | 'maintenance' | 'anomaly_detected' | 'healthy';
  health_score: number;
  discovered_schema?: Record<string, unknown>;
  confirmed_fields?: Record<string, FieldConfirmation>;
  created_at: string;
  anomaly_count?: number;
  last_analysis_at?: string;
}

export interface FieldConfirmation {
  confirmed: boolean;
  type?: string;
  unit?: string;
  meaning?: string;
  corrected?: boolean;
  confirmed_at: string;
}

// Anomaly Types
export type AnomalySeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type AnomalyType = 
  | 'behavioral_deviation'
  | 'threshold_breach'
  | 'pattern_change'
  | 'correlation_break'
  | 'margin_warning'
  | 'predictive_failure';

export interface Anomaly {
  id: string;
  type: AnomalyType;
  severity: AnomalySeverity;
  title: string;
  description: string;
  affected_fields: string[];
  natural_language_explanation: string;
  recommendations: Recommendation[];
  impact_score: number;
  timestamp: string;
}

export interface Recommendation {
  type: string;
  priority: 'high' | 'medium' | 'low';
  action: string;
  rationale?: string;
}

// Engineering Margin Types
export interface EngineeringMargin {
  component: string;
  parameter: string;
  current_value: number;
  design_limit: number;
  margin_percentage: number;
  trend: 'improving' | 'stable' | 'degrading';
  projected_breach_date?: string;
  safety_critical: boolean;
}

// Blind Spot Types
export interface BlindSpot {
  title: string;
  description: string;
  recommended_sensor?: {
    type: string;
    specification: string;
    estimated_cost: number;
  } | null;
  diagnostic_coverage_improvement: number;
}

// Analysis Types
export interface AnalysisResult {
  system_id: string;
  timestamp: string;
  health_score: number | null;
  data_analyzed?: {
    record_count: number;
    source_count: number;
    field_count: number;
  };
  anomalies: Anomaly[];
  engineering_margins: EngineeringMargin[];
  blind_spots: BlindSpot[];
  insights_summary?: string;
  insights?: string[];
  ai_analysis?: {
    ai_powered: boolean;
    agents_used: string[];
    agent_statuses: {
      agent: string;
      status: string;
      findings: number;
      perspective?: string;
      error?: string;
    }[];
    total_findings_raw: number;
    total_anomalies_unified: number;
  };
}

// Impact Radar Types
export interface ImpactRadarData {
  system_id: string;
  timestamp: string;
  total_anomalies: number;
  high_impact_anomalies: number;
  impact_distribution: {
    top_20_percent: {
      anomaly_count: number;
      impact_percentage: number;
    };
    remaining_80_percent: {
      anomaly_count: number;
      impact_percentage: number;
    };
  };
  prioritized_issues: PrioritizedIssue[];
}

export interface PrioritizedIssue {
  rank: number;
  title: string;
  impact_score: number;
  affected_percentage: number;
  recommended_action: string;
}

// Conversation Types
export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  data?: Record<string, unknown>;
}

export interface QueryResponse {
  type: 'explanation' | 'data_query' | 'general';
  query: string;
  response: string;
  evidence?: string[];
  related_data?: Record<string, unknown>;
  summary?: Record<string, unknown>;
}

// ─── Discovered Field Types (v2 — LLM-enriched) ─────────────────────

export interface FieldConfidenceScores {
  type: number;
  unit: number;
  meaning: number;
  overall: number;
}

export interface EngineeringContext {
  typical_range?: { min: number; max: number } | null;
  operating_range_description?: string | null;
  what_high_means?: string | null;
  what_low_means?: string | null;
  safety_critical?: boolean;
  design_limit_hint?: { min: number; max: number } | null;
}

export interface DiscoveredField {
  name: string;
  display_name?: string;
  description?: string;
  type?: string;
  physical_unit?: string;
  physical_unit_full?: string;
  category?: string;
  component?: string | null;
  engineering_context?: EngineeringContext;
  value_interpretation?: {
    assessment?: string;
  };
  confidence?: FieldConfidenceScores | number;
  reasoning?: string;
  source_file?: string;
  sample_values?: unknown[];
  statistics?: {
    min?: number;
    max?: number;
    mean?: number;
    std?: number;
    null_percentage?: number;
  };
  // Legacy compat
  inferred_type?: string;
  inferred_meaning?: string;
  field_category?: string;
}

export interface DetectedComponent {
  name: string;
  role: string;
  fields: string[];
}

export interface SystemIdentification {
  system_type: string;
  system_type_confidence: number;
  system_subtype?: string | null;
  system_description: string;
  domain?: string;
  detected_components?: DetectedComponent[];
  probable_use_case?: string;
  data_characteristics?: {
    temporal_resolution?: string;
    duration_estimate?: string;
    completeness?: string;
  };
}

export interface FieldRelationship {
  fields: string[];
  relationship: string;
  description: string;
  expected_correlation?: string;
  diagnostic_value?: string;
  // Legacy compat
  field_a?: string;
  field_b?: string;
  relationship_type?: string;
  strength?: number;
  confidence?: number;
}

export interface ConfirmationRequest {
  field?: string;
  field_name?: string;
  reason?: string;
  question: string;
  // Legacy compat
  type?: 'field_confirmation' | 'relationship_confirmation';
  inferred_unit?: string;
  inferred_type?: string;
  sample_values?: unknown[];
  options?: string[];
}

export interface SystemRecommendation {
  suggested_name: string;
  suggested_type: string;
  suggested_description: string;
  confidence: number;
  system_subtype?: string | null;
  domain?: string | null;
  detected_components?: DetectedComponent[];
  probable_use_case?: string | null;
  data_characteristics?: {
    temporal_resolution?: string;
    duration_estimate?: string;
    completeness?: string;
  };
  reasoning?: string;
  analysis_summary?: {
    files_analyzed: number;
    total_records: number;
    unique_fields: number;
    ai_powered?: boolean;
  };
}

export interface AnalyzeFilesResponse {
  status: string;
  analysis_id: string;
  files_analyzed: number;
  total_records: number;
  ai_powered: boolean;
  discovered_fields: DiscoveredField[];
  recommendation: SystemRecommendation;
  field_relationships: FieldRelationship[];
  blind_spots: string[];
  confirmation_requests: ConfirmationRequest[];
  file_classification: {
    data_files: string[];
    description_files: string[];
    error_files: string[];
  };
  file_errors: unknown[];
  context_extracted: boolean;
  fields_enriched: number;
  available_system_types: Record<string, string>;
}

// Anomaly Feedback Types
export type FeedbackType = 'relevant' | 'false_positive' | 'already_known';

export interface AnomalyFeedback {
  id: string;
  system_id: string;
  anomaly_id: string;
  anomaly_title: string;
  anomaly_type: string;
  severity: string;
  feedback_type: FeedbackType;
  comment?: string;
  created_at: string;
}

// Schedule / Watchdog Types
export interface Schedule {
  system_id: string;
  enabled: boolean;
  interval: '1h' | '6h' | '12h' | '24h' | '7d';
  created_at?: string;
  updated_at?: string;
  last_run_at?: string;
  last_run_status?: 'success' | 'error' | 'skipped';
  last_error?: string;
  run_count: number;
}

export interface FeedbackSummary {
  system_id: string;
  total_feedback: number;
  by_type: {
    relevant: number;
    false_positive: number;
    already_known: number;
  };
  false_positive_rate: number;
  false_positive_patterns: Array<{
    anomaly_type: string;
    total: number;
    false_positive_count: number;
    false_positive_rate: number;
  }>;
  confidence_score: number | null;
}
