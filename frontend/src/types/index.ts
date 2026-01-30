// System Types
export interface System {
  id: string;
  name: string;
  system_type: string;
  serial_number?: string;
  model?: string;
  status: 'active' | 'inactive' | 'maintenance' | 'anomaly_detected';
  health_score: number;
  discovered_schema?: Record<string, unknown>;
  confirmed_fields?: Record<string, FieldConfirmation>;
  created_at: string;
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

// Discovered Field Types
export interface DiscoveredField {
  name: string;
  inferred_type: string;
  physical_unit?: string;
  inferred_meaning?: string;
  confidence: number;
  sample_values?: unknown[];
  statistics?: {
    min?: number;
    max?: number;
    mean?: number;
    std?: number;
    null_percentage?: number;
  };
}

export interface FieldRelationship {
  field_a: string;
  field_b: string;
  relationship_type: string;
  strength: number;
  description: string;
  confidence: number;
}

export interface ConfirmationRequest {
  type: 'field_confirmation' | 'relationship_confirmation';
  field_name?: string;
  question: string;
  inferred_unit?: string;
  inferred_type?: string;
  sample_values?: unknown[];
  options: string[];
}
