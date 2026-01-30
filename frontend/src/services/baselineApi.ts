const API_BASE = '/api/v1/baselines';

export interface Snapshot {
  id: string;
  timestamp: string;
  health_score: number | null;
  record_count: number;
  field_count: number;
  field_stats: Record<string, FieldSnapshot>;
  anomaly_count: number;
  anomaly_summary: Record<string, number>;
}

export interface FieldSnapshot {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  q25: number;
  q75: number;
  null_pct: number;
}

export interface Baseline {
  system_id: string;
  snapshot_count: number;
  first_snapshot: string;
  last_snapshot: string;
  health_score: { mean: number | null; min: number | null; max: number | null; std: number };
  anomaly_count: { mean: number; min: number; max: number };
  field_baselines: Record<string, {
    mean_of_means: number;
    std_of_means: number;
    avg_std: number;
    range_of_means: [number, number];
    snapshots_with_data: number;
  }>;
}

export interface Deviation {
  field: string;
  current_mean: number;
  baseline_mean: number;
  baseline_std: number;
  z_deviation: number;
  pct_change: number;
  status: 'normal' | 'minor_deviation' | 'significant_deviation' | 'critical_deviation';
}

export interface CompareResult {
  status: string;
  snapshot_count?: number;
  baseline_period?: string;
  critical_deviations?: number;
  significant_deviations?: number;
  fields_compared?: number;
  deviations?: Deviation[];
  message?: string;
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const baselineApi = {
  captureSnapshot: (systemId: string) =>
    request<{ status: string; snapshot: Snapshot }>(
      `${API_BASE}/systems/${systemId}/snapshot`,
      { method: 'POST' },
    ),

  getHistory: (systemId: string, limit = 50) =>
    request<{ system_id: string; count: number; snapshots: Snapshot[] }>(
      `${API_BASE}/systems/${systemId}/history?limit=${limit}`,
    ),

  getBaseline: (systemId: string) =>
    request<Baseline>(`${API_BASE}/systems/${systemId}/baseline`),

  compare: (systemId: string) =>
    request<CompareResult>(`${API_BASE}/systems/${systemId}/compare`),

  clear: (systemId: string) =>
    request<{ status: string }>(`${API_BASE}/systems/${systemId}`, { method: 'DELETE' }),
};
