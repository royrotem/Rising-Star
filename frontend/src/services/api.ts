import axios from 'axios';
import type {
  System,
  Schedule,
  AnalysisResult,
  ImpactRadarData,
  QueryResponse,
  DiscoveredField,
  FieldRelationship,
  ConfirmationRequest,
} from '../types';

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Systems API
export const systemsApi = {
  list: async (): Promise<System[]> => {
    const { data } = await api.get('/systems/');
    return data;
  },

  get: async (systemId: string): Promise<System> => {
    const { data } = await api.get(`/systems/${systemId}`);
    return data;
  },

  create: async (system: Partial<System> & { analysis_id?: string }): Promise<System> => {
    const { data } = await api.post('/systems/', system);
    return data;
  },

  ingest: async (
    systemId: string,
    file: File,
    sourceName: string
  ): Promise<{
    status: string;
    source_id: string;
    record_count: number;
    discovered_fields: DiscoveredField[];
    relationships: FieldRelationship[];
    confirmation_requests: ConfirmationRequest[];
  }> => {
    const formData = new FormData();
    formData.append('file', file);

    const encodedSourceName = encodeURIComponent(sourceName);
    const { data } = await api.post(
      `/systems/${systemId}/ingest?source_name=${encodedSourceName}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return data;
  },

  confirmFields: async (
    systemId: string,
    confirmations: Array<{
      field_name: string;
      confirmed_type?: string;
      confirmed_unit?: string;
      confirmed_meaning?: string;
      is_correct: boolean;
    }>
  ): Promise<{ status: string; confirmed_count: number }> => {
    const { data } = await api.post(`/systems/${systemId}/confirm-fields`, confirmations);
    return data;
  },

  getAnalysis: async (systemId: string): Promise<AnalysisResult | null> => {
    try {
      const { data } = await api.get(`/systems/${systemId}/analysis`);
      return data;
    } catch {
      return null;
    }
  },

  analyze: async (
    systemId: string,
    options?: {
      include_anomaly_detection?: boolean;
      include_root_cause?: boolean;
      include_blind_spots?: boolean;
      time_range_hours?: number;
    }
  ): Promise<AnalysisResult> => {
    const { data } = await api.post(`/systems/${systemId}/analyze`, options || {});
    return data;
  },

  query: async (
    systemId: string,
    query: string,
    context?: Record<string, unknown>
  ): Promise<QueryResponse> => {
    const { data } = await api.post(`/systems/${systemId}/query`, { query, context });
    return data;
  },

  getImpactRadar: async (systemId: string): Promise<ImpactRadarData> => {
    const { data } = await api.get(`/systems/${systemId}/impact-radar`);
    return data;
  },

  delete: async (systemId: string): Promise<void> => {
    await api.delete(`/systems/${systemId}`);
  },

  getNextGenSpecs: async (systemId: string): Promise<{
    recommended_improvements: {
      new_sensors: Array<{
        type: string;
        location: string;
        sampling_rate: string;
        rationale: string;
        estimated_cost: number;
      }>;
      data_architecture: Record<string, unknown>;
      connectivity: Record<string, unknown>;
    };
    expected_benefits: Record<string, string>;
  }> => {
    const { data } = await api.get(`/systems/${systemId}/next-gen-specs`);
    return data;
  },
};

// Ground Truth Evaluation API (for demo systems with Kaggle data)
export interface GroundTruthEvaluation {
  system_id: string;
  evaluation: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc_roc: number;
  };
  confusion_matrix: {
    true_positive: number;
    false_positive: number;
    false_negative: number;
    true_negative: number;
  };
  fault_type_breakdown: Record<string, {
    total: number;
    detected: number;
    missed: number;
    recall: number;
  }>;
  roc_curve: Array<{ fpr: number; tpr: number; threshold?: number }>;
  per_row_sample: Array<{
    row_index: number;
    ground_truth_label: number;
    ground_truth_fault: string;
    detected: boolean;
    classification: 'true_positive' | 'false_positive' | 'false_negative' | 'true_negative';
  }>;
  summary: {
    total_records: number;
    total_gt_anomalous: number;
    total_gt_normal: number;
    anomalies_detected_by_system: number;
    rows_flagged: number;
  };
  anomaly_details: Array<{
    id: string;
    title: string;
    severity: string;
    type: string;
    affected_fields: string[];
    confidence: number;
    impact_score: number;
    matched_fault_types?: string[];
    detected_rows: number;
  }>;
}

export const demoApi = {
  evaluate: async (systemId: string): Promise<GroundTruthEvaluation> => {
    const { data } = await api.post(`/systems/demo/${systemId}/evaluate`);
    return data;
  },

  getGroundTruth: async (systemId: string): Promise<{
    system_id: string;
    label_map: Record<number, string>;
    distribution: Record<string, number>;
    total_records: number;
    total_anomalous: number;
    total_normal: number;
    labels: number[];
  }> => {
    const { data } = await api.get(`/systems/demo/${systemId}/ground-truth`);
    return data;
  },
};

// Schedules API (Watchdog Mode)
export const schedulesApi = {
  get: async (systemId: string): Promise<Schedule> => {
    const { data } = await api.get(`/schedules/${systemId}`);
    return data;
  },

  list: async (): Promise<Schedule[]> => {
    const { data } = await api.get('/schedules/');
    return data;
  },

  set: async (systemId: string, config: { enabled: boolean; interval: string }): Promise<Schedule> => {
    const { data } = await api.put(`/schedules/${systemId}`, config);
    return data;
  },

  delete: async (systemId: string): Promise<void> => {
    await api.delete(`/schedules/${systemId}`);
  },
};

// Agents API
export const agentsApi = {
  getStatus: async (): Promise<Record<string, {
    name: string;
    status: string;
    capabilities: string[];
    current_task: string | null;
  }>> => {
    const { data } = await api.get('/agents/status');
    return data;
  },
};

export default api;
