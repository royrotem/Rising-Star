/**
 * AnomalyExplorer — Interactive anomaly exploration page with ground-truth evaluation.
 *
 * After clicking ANALYZE on a demo system with Kaggle TLM-UAV data, this page:
 * 1. Shows all detected anomalies (as before)
 * 2. Runs ground-truth evaluation to compare detections vs real labels
 * 3. Displays ROC curve, confusion matrix, per-fault-type breakdown
 * 4. Color-codes rows: green=correct, red=missed, orange=false alarm
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Search,
  Filter,
  SortDesc,
  AlertTriangle,
  Activity,
  Loader2,
  ChevronDown,
  ChevronUp,
  Lightbulb,
  CheckCircle,
  Globe,
  Users,
  Eye,
  BarChart3,
  List,
  X,
  Target,
  XCircle,
  MinusCircle,
  Shield,
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi, demoApi } from '../services/api';
import type { GroundTruthEvaluation } from '../services/api';
import type { System } from '../types';
import { FeedbackButtons } from '../components/AnomalyFeedback';
import { getSeverityCardColor, getSeverityDotColor, getSeveritySmallBadge } from '../utils/colors';

// Reuse the AnalysisData shape from SystemDetail
interface AnomalyItem {
  id: string;
  type: string;
  severity: string;
  title: string;
  description: string;
  affected_fields?: string[];
  natural_language_explanation: string;
  possible_causes?: string[];
  recommendations: Array<{ type: string; priority: string; action: string }>;
  impact_score: number;
  confidence?: number;
  value?: Record<string, unknown>;
  expected_range?: [number, number];
  contributing_agents?: string[];
  web_references?: string[];
  agent_perspectives?: Array<{ agent: string; perspective: string }>;
}

type SortKey = 'impact_score' | 'confidence' | 'severity' | 'title';
type SortDir = 'asc' | 'desc';

const SEVERITY_ORDER: Record<string, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
  info: 0,
};

const CLASSIFICATION_COLORS: Record<string, { bg: string; text: string; border: string; label: string }> = {
  true_positive: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Detected Correctly' },
  false_negative: { bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/30', label: 'Missed (FN)' },
  false_positive: { bg: 'bg-orange-500/10', text: 'text-orange-400', border: 'border-orange-500/30', label: 'False Alarm (FP)' },
  true_negative: { bg: 'bg-stone-500/10', text: 'text-stone-400', border: 'border-stone-500/30', label: 'Correct Normal' },
};

// ── Simple SVG ROC Curve ────────────────────────────────────────────────
function RocCurve({ points, auc }: { points: Array<{ fpr: number; tpr: number }>; auc: number }) {
  const w = 280;
  const h = 220;
  const pad = 40;

  const toX = (fpr: number) => pad + fpr * (w - pad - 10);
  const toY = (tpr: number) => h - pad - tpr * (h - pad - 10);

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(p.fpr).toFixed(1)},${toY(p.tpr).toFixed(1)}`)
    .join(' ');

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-[280px]">
      {/* Grid */}
      <line x1={pad} y1={h - pad} x2={w - 10} y2={h - pad} stroke="#57534e" strokeWidth="1" />
      <line x1={pad} y1={10} x2={pad} y2={h - pad} stroke="#57534e" strokeWidth="1" />
      {/* Diagonal reference */}
      <line x1={pad} y1={h - pad} x2={w - 10} y2={10} stroke="#57534e" strokeWidth="0.5" strokeDasharray="4,4" />
      {/* ROC curve */}
      <path d={pathD} fill="none" stroke="#22d3ee" strokeWidth="2.5" strokeLinejoin="round" />
      {/* Fill under curve */}
      <path
        d={`${pathD} L${toX(points[points.length - 1]?.fpr ?? 1)},${toY(0)} L${toX(0)},${toY(0)} Z`}
        fill="#22d3ee"
        fillOpacity="0.08"
      />
      {/* Points */}
      {points.map((p, i) => (
        <circle key={i} cx={toX(p.fpr)} cy={toY(p.tpr)} r="3" fill="#22d3ee" fillOpacity="0.6" />
      ))}
      {/* Labels */}
      <text x={w / 2} y={h - 5} textAnchor="middle" fill="#a8a29e" fontSize="10">FPR</text>
      <text x={12} y={h / 2} textAnchor="middle" fill="#a8a29e" fontSize="10" transform={`rotate(-90,12,${h / 2})`}>TPR</text>
      <text x={w - 10} y={20} textAnchor="end" fill="#22d3ee" fontSize="11" fontWeight="bold">
        AUC = {auc.toFixed(3)}
      </text>
    </svg>
  );
}

// ── Confusion Matrix Visual ─────────────────────────────────────────────
function ConfusionMatrix({ cm }: { cm: GroundTruthEvaluation['confusion_matrix'] }) {
  const total = cm.true_positive + cm.false_positive + cm.false_negative + cm.true_negative;
  const cell = (value: number, color: string) => (
    <div className={clsx('rounded-lg p-3 text-center', color)}>
      <div className="text-lg font-bold tabular-nums">{value.toLocaleString()}</div>
      <div className="text-[10px] opacity-60">{total > 0 ? ((value / total) * 100).toFixed(1) : 0}%</div>
    </div>
  );

  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-xs">
      <div />
      <div className="text-center text-stone-400 pb-1 text-[10px] font-medium">Predicted Anomaly</div>
      <div className="text-center text-stone-400 pb-1 text-[10px] font-medium">Predicted Normal</div>

      <div className="flex items-center text-stone-400 pr-2 text-[10px] font-medium">Actual Anomaly</div>
      {cell(cm.true_positive, 'bg-emerald-500/15 text-emerald-400')}
      {cell(cm.false_negative, 'bg-red-500/15 text-red-400')}

      <div className="flex items-center text-stone-400 pr-2 text-[10px] font-medium">Actual Normal</div>
      {cell(cm.false_positive, 'bg-orange-500/15 text-orange-400')}
      {cell(cm.true_negative, 'bg-stone-500/15 text-stone-300')}
    </div>
  );
}

export default function AnomalyExplorer() {
  const { systemId } = useParams();
  const [system, setSystem] = useState<System | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Ground truth evaluation
  const [evaluation, setEvaluation] = useState<GroundTruthEvaluation | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [rowFilter, setRowFilter] = useState<string | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState<string[]>([]);
  const [typeFilter, setTypeFilter] = useState<string[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>('impact_score');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');

  useEffect(() => {
    if (!systemId) return;
    loadData();
  }, [systemId]);

  const loadData = async () => {
    if (!systemId) return;
    setLoading(true);
    setError(null);
    try {
      const sys = await systemsApi.get(systemId);
      setSystem(sys as System);
      const result = await systemsApi.getAnalysis(systemId);
      if (result && result.anomalies) {
        setAnomalies(result.anomalies as AnomalyItem[]);
      }

      // Auto-run evaluation if this is a demo system with ground truth
      if (sys?.metadata?.has_ground_truth || sys?.metadata?.is_real_data || sys?.is_demo) {
        setEvalLoading(true);
        try {
          const evalResult = await demoApi.evaluate(systemId);
          setEvaluation(evalResult);
          setShowEvaluation(true);
        } catch {
          // Ground truth not available — that's fine
        } finally {
          setEvalLoading(false);
        }
      }
    } catch (e) {
      console.error('Failed to load anomalies:', e);
      setError('Failed to load anomaly data. Run analysis first.');
    } finally {
      setLoading(false);
    }
  };

  const hasGroundTruth = evaluation !== null;

  // Derived data
  const allSeverities = useMemo(() => {
    const set = new Set(anomalies.map((a) => a.severity));
    return Array.from(set).sort((a, b) => (SEVERITY_ORDER[b] ?? 0) - (SEVERITY_ORDER[a] ?? 0));
  }, [anomalies]);

  const allTypes = useMemo(() => {
    const set = new Set(anomalies.map((a) => a.type));
    return Array.from(set).sort();
  }, [anomalies]);

  const filtered = useMemo(() => {
    let items = [...anomalies];

    // Search
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      items = items.filter(
        (a) =>
          a.title.toLowerCase().includes(q) ||
          a.description.toLowerCase().includes(q) ||
          a.natural_language_explanation.toLowerCase().includes(q) ||
          (a.affected_fields || []).some((f) => f.toLowerCase().includes(q))
      );
    }

    // Severity filter
    if (severityFilter.length > 0) {
      items = items.filter((a) => severityFilter.includes(a.severity));
    }

    // Type filter
    if (typeFilter.length > 0) {
      items = items.filter((a) => typeFilter.includes(a.type));
    }

    // Sort
    items.sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case 'impact_score':
          cmp = a.impact_score - b.impact_score;
          break;
        case 'confidence':
          cmp = (a.confidence ?? 0) - (b.confidence ?? 0);
          break;
        case 'severity':
          cmp = (SEVERITY_ORDER[a.severity] ?? 0) - (SEVERITY_ORDER[b.severity] ?? 0);
          break;
        case 'title':
          cmp = a.title.localeCompare(b.title);
          break;
      }
      return sortDir === 'desc' ? -cmp : cmp;
    });

    return items;
  }, [anomalies, searchQuery, severityFilter, typeFilter, sortKey, sortDir]);

  // Severity distribution
  const severityDistribution = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const a of anomalies) {
      counts[a.severity] = (counts[a.severity] || 0) + 1;
    }
    return counts;
  }, [anomalies]);

  const maxSeverityCount = Math.max(1, ...Object.values(severityDistribution));

  // Filtered per-row results
  const filteredRows = useMemo(() => {
    if (!evaluation) return [];
    const rows = evaluation.per_row_sample;
    if (!rowFilter) return rows;
    return rows.filter((r) => r.classification === rowFilter);
  }, [evaluation, rowFilter]);

  const toggleSeverityFilter = (sev: string) => {
    setSeverityFilter((prev) =>
      prev.includes(sev) ? prev.filter((s) => s !== sev) : [...prev, sev]
    );
  };

  const toggleTypeFilter = (type: string) => {
    setTypeFilter((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const clearFilters = () => {
    setSearchQuery('');
    setSeverityFilter([]);
    setTypeFilter([]);
  };

  const hasActiveFilters = searchQuery || severityFilter.length > 0 || typeFilter.length > 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-primary-400 animate-spin mx-auto mb-3" />
          <p className="text-sm text-stone-400">Analyzing anomalies...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 page-enter">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link
          to={`/systems/${systemId}`}
          className="p-2 hover:bg-stone-700 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-stone-400" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Anomaly Explorer
          </h1>
          <p className="text-sm text-stone-400">
            {system?.name || 'System'} — {anomalies.length} anomalies detected
            {hasGroundTruth && (
              <span className="ml-2 text-cyan-400">(Kaggle TLM-UAV — real data with ground truth)</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {hasGroundTruth && (
            <button
              onClick={() => setShowEvaluation(!showEvaluation)}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium transition-colors',
                showEvaluation
                  ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                  : 'bg-stone-700/60 text-stone-400 border border-stone-600/50 hover:text-cyan-400'
              )}
            >
              <Target className="w-3.5 h-3.5" />
              Ground Truth
            </button>
          )}
          <button
            onClick={() => setViewMode('list')}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              viewMode === 'list'
                ? 'bg-primary-500/10 text-primary-400'
                : 'text-stone-400 hover:text-stone-300'
            )}
          >
            <List className="w-4 h-4" />
          </button>
          <button
            onClick={() => setViewMode('grid')}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              viewMode === 'grid'
                ? 'bg-primary-500/10 text-primary-400'
                : 'text-stone-400 hover:text-stone-300'
            )}
          >
            <BarChart3 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6 glass-card p-4 border-red-500/30">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════
          GROUND TRUTH EVALUATION PANEL
          ════════════════════════════════════════════════════════════════════ */}
      {showEvaluation && evaluation && (
        <div className="mb-6 space-y-4 animate-fade-in">
          {/* Metrics Banner */}
          <div className="glass-card p-5 border border-cyan-500/20">
            <div className="flex items-center gap-2 mb-4">
              <Shield className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-cyan-400">
                Ground Truth Evaluation — Kaggle TLM:UAV Dataset
              </h3>
              <span className="ml-auto text-[10px] text-stone-400">
                {evaluation.summary.total_records.toLocaleString()} records
                ({evaluation.summary.total_gt_anomalous.toLocaleString()} anomalous,
                {' '}{evaluation.summary.total_gt_normal.toLocaleString()} normal)
              </span>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-5 gap-3 mb-5">
              {[
                { label: 'Accuracy', value: evaluation.evaluation.accuracy, color: 'text-white' },
                { label: 'Precision', value: evaluation.evaluation.precision, color: 'text-emerald-400' },
                { label: 'Recall', value: evaluation.evaluation.recall, color: 'text-blue-400' },
                { label: 'F1 Score', value: evaluation.evaluation.f1_score, color: 'text-purple-400' },
                { label: 'AUC-ROC', value: evaluation.evaluation.auc_roc, color: 'text-cyan-400' },
              ].map((m) => (
                <div key={m.label} className="bg-stone-800/50 rounded-xl p-3 text-center">
                  <div className={clsx('text-xl font-bold tabular-nums', m.color)}>
                    {(m.value * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-stone-400 mt-0.5">{m.label}</div>
                </div>
              ))}
            </div>

            {/* ROC + Confusion Matrix side by side */}
            <div className="grid grid-cols-2 gap-5">
              <div>
                <h4 className="text-xs font-medium text-stone-400 uppercase tracking-wide mb-2">ROC Curve</h4>
                <RocCurve points={evaluation.roc_curve} auc={evaluation.evaluation.auc_roc} />
              </div>
              <div>
                <h4 className="text-xs font-medium text-stone-400 uppercase tracking-wide mb-2">Confusion Matrix</h4>
                <ConfusionMatrix cm={evaluation.confusion_matrix} />
              </div>
            </div>
          </div>

          {/* Per-Fault-Type Breakdown */}
          <div className="glass-card p-5 border border-cyan-500/10">
            <h3 className="text-xs font-medium text-stone-400 uppercase tracking-wide mb-3 flex items-center gap-2">
              <Target className="w-3.5 h-3.5 text-cyan-400" />
              Per-Fault-Type Detection Rate
            </h3>
            <div className="grid grid-cols-4 gap-3">
              {Object.entries(evaluation.fault_type_breakdown).map(([fault, stats]) => (
                <div key={fault} className="bg-stone-800/50 rounded-xl p-3">
                  <div className="text-xs font-medium text-stone-200 mb-2 capitalize">
                    {fault.replace(/_/g, ' ')}
                  </div>
                  <div className="flex items-end gap-2 mb-1.5">
                    <span className="text-lg font-bold text-white tabular-nums">
                      {(stats.recall * 100).toFixed(0)}%
                    </span>
                    <span className="text-[10px] text-stone-400 mb-0.5">recall</span>
                  </div>
                  {/* Progress bar */}
                  <div className="w-full bg-stone-700/50 rounded-full h-2 mb-1">
                    <div
                      className={clsx(
                        'h-2 rounded-full transition-all',
                        stats.recall >= 0.8 ? 'bg-emerald-500' : stats.recall >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                      )}
                      style={{ width: `${stats.recall * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-stone-400">
                    <span className="text-emerald-400">{stats.detected} detected</span>
                    <span className="text-red-400">{stats.missed} missed</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Per-Row Sample Table */}
          <div className="glass-card p-5 border border-cyan-500/10">
            <div className="flex items-center gap-2 mb-3">
              <h3 className="text-xs font-medium text-stone-400 uppercase tracking-wide flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                Row-Level Classification (sampled)
              </h3>
              {/* Row classification filter */}
              <div className="flex gap-1.5 ml-auto">
                {[
                  { key: null, label: 'All', icon: null },
                  { key: 'true_positive', label: 'TP', icon: CheckCircle },
                  { key: 'false_negative', label: 'FN', icon: XCircle },
                  { key: 'false_positive', label: 'FP', icon: MinusCircle },
                ].map((f) => (
                  <button
                    key={f.key ?? 'all'}
                    onClick={() => setRowFilter(f.key)}
                    className={clsx(
                      'px-2 py-1 rounded-lg text-[10px] font-medium transition-colors',
                      rowFilter === f.key
                        ? 'bg-cyan-500/15 text-cyan-400'
                        : 'text-stone-400 hover:text-stone-200 hover:bg-stone-700/50'
                    )}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="max-h-[300px] overflow-y-auto rounded-lg">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-stone-800 z-10">
                  <tr className="text-stone-400 text-left">
                    <th className="px-3 py-2 font-medium">Row</th>
                    <th className="px-3 py-2 font-medium">Ground Truth</th>
                    <th className="px-3 py-2 font-medium">Detected?</th>
                    <th className="px-3 py-2 font-medium">Classification</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-700/30">
                  {filteredRows.map((row) => {
                    const cls = CLASSIFICATION_COLORS[row.classification] || CLASSIFICATION_COLORS.true_negative;
                    return (
                      <tr key={row.row_index} className={clsx(cls.bg, 'transition-colors')}>
                        <td className="px-3 py-1.5 tabular-nums text-stone-300 font-mono">
                          #{row.row_index}
                        </td>
                        <td className="px-3 py-1.5">
                          <span className={clsx(
                            'px-1.5 py-0.5 rounded text-[10px] font-medium',
                            row.ground_truth_label === 0
                              ? 'bg-stone-600/50 text-stone-300'
                              : 'bg-amber-500/15 text-amber-400'
                          )}>
                            {row.ground_truth_fault}
                          </span>
                        </td>
                        <td className="px-3 py-1.5">
                          {row.detected ? (
                            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                          ) : (
                            <X className="w-3.5 h-3.5 text-stone-500" />
                          )}
                        </td>
                        <td className="px-3 py-1.5">
                          <span className={clsx('px-1.5 py-0.5 rounded text-[10px] font-medium border', cls.text, cls.border)}>
                            {cls.label}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {filteredRows.length === 0 && (
                <div className="text-center py-6 text-stone-400 text-xs">
                  No rows match this filter
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Evaluation loading state */}
      {evalLoading && (
        <div className="mb-6 glass-card p-4 border-cyan-500/20 flex items-center gap-3">
          <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
          <span className="text-sm text-cyan-400">Evaluating against ground truth labels...</span>
        </div>
      )}

      {/* Severity Distribution */}
      {anomalies.length > 0 && (
        <div className="glass-card p-5 mb-6">
          <h3 className="text-xs font-medium text-stone-400 uppercase tracking-wide mb-3 flex items-center gap-2">
            <BarChart3 className="w-3.5 h-3.5" />
            Severity Distribution
          </h3>
          <div className="flex items-end gap-3 h-20">
            {['critical', 'high', 'medium', 'low', 'info'].map((sev) => {
              const count = severityDistribution[sev] || 0;
              const pct = (count / maxSeverityCount) * 100;
              const isFiltered = severityFilter.length > 0 && !severityFilter.includes(sev);
              return (
                <button
                  key={sev}
                  onClick={() => toggleSeverityFilter(sev)}
                  className={clsx(
                    'flex-1 flex flex-col items-center gap-1 group transition-opacity',
                    isFiltered && 'opacity-30'
                  )}
                >
                  <span className="text-xs font-semibold text-white tabular-nums">
                    {count}
                  </span>
                  <div className="w-full relative" style={{ height: '48px' }}>
                    <div
                      className={clsx(
                        'absolute bottom-0 left-0 right-0 rounded-t-md transition-all duration-300',
                        getSeverityDotColor(sev),
                        'group-hover:opacity-80'
                      )}
                      style={{ height: `${Math.max(pct, count > 0 ? 10 : 2)}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-stone-400 capitalize">{sev}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Search & Filters */}
      <div className="flex items-center gap-3 mb-5">
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-stone-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search anomalies..."
            className="w-full pl-9 pr-4 py-2 bg-stone-700/60 border border-stone-600/50 rounded-xl text-sm text-white placeholder-stone-500 focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500 transition-colors"
          />
        </div>

        {/* Type filter dropdown */}
        {allTypes.length > 1 && (
          <div className="relative group">
            <button className="flex items-center gap-2 px-3 py-2 bg-stone-700/60 border border-stone-600/50 rounded-xl text-xs text-stone-400 hover:text-stone-200 transition-colors">
              <Filter className="w-3.5 h-3.5" />
              Type
              {typeFilter.length > 0 && (
                <span className="px-1.5 py-0.5 bg-primary-500/20 text-primary-400 rounded-full text-[10px] font-bold">
                  {typeFilter.length}
                </span>
              )}
            </button>
            <div className="absolute top-full mt-1 right-0 bg-stone-700 border border-stone-600 rounded-xl shadow-xl p-2 min-w-[200px] hidden group-hover:block z-20">
              {allTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => toggleTypeFilter(type)}
                  className={clsx(
                    'w-full text-left px-3 py-1.5 rounded-lg text-xs transition-colors',
                    typeFilter.includes(type)
                      ? 'bg-primary-500/10 text-primary-400'
                      : 'text-stone-400 hover:bg-stone-600/50 hover:text-stone-200'
                  )}
                >
                  {type.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Sort */}
        <div className="relative group">
          <button className="flex items-center gap-2 px-3 py-2 bg-stone-700/60 border border-stone-600/50 rounded-xl text-xs text-stone-400 hover:text-stone-200 transition-colors">
            <SortDesc className="w-3.5 h-3.5" />
            Sort
          </button>
          <div className="absolute top-full mt-1 right-0 bg-stone-700 border border-stone-600 rounded-xl shadow-xl p-2 min-w-[180px] hidden group-hover:block z-20">
            {([
              ['impact_score', 'Impact Score'],
              ['confidence', 'Confidence'],
              ['severity', 'Severity'],
              ['title', 'Title'],
            ] as [SortKey, string][]).map(([key, label]) => (
              <button
                key={key}
                onClick={() => toggleSort(key)}
                className={clsx(
                  'w-full text-left px-3 py-1.5 rounded-lg text-xs transition-colors flex items-center justify-between',
                  sortKey === key
                    ? 'bg-primary-500/10 text-primary-400'
                    : 'text-stone-400 hover:bg-stone-600/50 hover:text-stone-200'
                )}
              >
                {label}
                {sortKey === key && (
                  sortDir === 'desc'
                    ? <ChevronDown className="w-3 h-3" />
                    : <ChevronUp className="w-3 h-3" />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Clear filters */}
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="flex items-center gap-1.5 px-3 py-2 text-xs text-red-400 hover:text-red-300 transition-colors"
          >
            <X className="w-3.5 h-3.5" />
            Clear
          </button>
        )}

        {/* Count */}
        <span className="text-xs text-stone-400 ml-auto tabular-nums">
          {filtered.length} / {anomalies.length}
        </span>
      </div>

      {/* Anomaly List */}
      {filtered.length === 0 ? (
        <div className="text-center py-16">
          <CheckCircle className="w-12 h-12 text-stone-400 mx-auto mb-3" />
          <p className="text-stone-400 font-medium">
            {anomalies.length === 0
              ? 'No anomalies detected'
              : 'No anomalies match your filters'}
          </p>
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="mt-2 text-primary-400 text-sm hover:text-primary-300"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <div className={clsx(
          viewMode === 'grid'
            ? 'grid grid-cols-2 gap-4'
            : 'space-y-3'
        )}>
          {filtered.map((anomaly) => {
            const isExpanded = expandedId === anomaly.id;

            // Find this anomaly's evaluation detail if ground truth is available
            const evalDetail = evaluation?.anomaly_details?.find((d) => d.id === anomaly.id);

            return (
              <div
                key={anomaly.id}
                className={clsx(
                  'glass-card-hover p-4 border-l-4 cursor-pointer',
                  getSeverityCardColor(anomaly.severity),
                  isExpanded && 'ring-1 ring-primary-500/30'
                )}
                onClick={() => setExpandedId(isExpanded ? null : anomaly.id)}
              >
                {/* Header */}
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={clsx(
                        'px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wide border',
                        getSeveritySmallBadge(anomaly.severity)
                      )}>
                        {anomaly.severity}
                      </span>
                      <span className="text-[10px] text-stone-400 font-mono">
                        {anomaly.type.replace(/_/g, ' ')}
                      </span>
                      {/* GT match badge */}
                      {evalDetail && evalDetail.matched_fault_types && (
                        <span className="px-1.5 py-0.5 bg-cyan-500/10 border border-cyan-500/20 rounded text-[10px] text-cyan-400 font-medium">
                          GT: {evalDetail.matched_fault_types.join(', ').replace(/_/g, ' ')}
                        </span>
                      )}
                    </div>
                    <h3 className="font-medium text-sm text-white">{anomaly.title}</h3>
                  </div>
                  <div className="flex flex-col items-end gap-1 flex-shrink-0">
                    <span className="text-xs font-semibold text-stone-300 tabular-nums">
                      {anomaly.impact_score.toFixed(1)}
                    </span>
                    <span className="text-[10px] text-stone-400">impact</span>
                  </div>
                </div>

                <p className="text-xs text-stone-400 leading-relaxed mb-2">
                  {anomaly.description}
                </p>

                {/* Confidence bar */}
                {anomaly.confidence != null && (
                  <div className="flex items-center gap-2 mb-2">
                    <div className="flex-1 bg-stone-700/50 rounded-full h-1.5 max-w-[120px]">
                      <div
                        className="bg-primary-500 h-1.5 rounded-full transition-all"
                        style={{ width: `${anomaly.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-stone-400 tabular-nums">
                      {(anomaly.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                )}

                {/* Affected fields */}
                {anomaly.affected_fields && anomaly.affected_fields.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {anomaly.affected_fields.map((field, idx) => (
                      <span
                        key={idx}
                        className="px-1.5 py-0.5 bg-stone-700/50 rounded text-[10px] text-stone-400 font-mono"
                      >
                        {field}
                      </span>
                    ))}
                  </div>
                )}

                {/* GT evaluation badge for this anomaly */}
                {evalDetail && (
                  <div className="flex items-center gap-2 text-[10px] mt-1">
                    <span className="text-cyan-400/70">
                      Matched {evalDetail.detected_rows.toLocaleString()} rows in ground truth
                    </span>
                  </div>
                )}

                {/* Expanded Details */}
                {isExpanded && (
                  <div
                    className="mt-3 pt-3 border-t border-stone-600/50 space-y-3 animate-fade-in"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {/* AI Explanation */}
                    <div>
                      <h4 className="text-xs font-medium text-primary-400 mb-1.5 flex items-center gap-1.5">
                        <Lightbulb className="w-3.5 h-3.5" />
                        AI Analysis
                      </h4>
                      <p className="text-xs text-stone-300 leading-relaxed">
                        {anomaly.natural_language_explanation}
                      </p>
                    </div>

                    {/* Possible Causes */}
                    {anomaly.possible_causes && anomaly.possible_causes.length > 0 && (
                      <div>
                        <h4 className="text-xs font-medium text-orange-400 mb-1.5 flex items-center gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          Possible Causes
                        </h4>
                        <ul className="space-y-1">
                          {anomaly.possible_causes.map((cause, idx) => (
                            <li key={idx} className="flex items-start gap-1.5 text-xs text-stone-300">
                              <span className="text-orange-400 mt-0.5">-</span>
                              {cause}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Recommendations */}
                    {anomaly.recommendations && anomaly.recommendations.length > 0 && (
                      <div>
                        <h4 className="text-xs font-medium text-emerald-400 mb-1.5 flex items-center gap-1.5">
                          <CheckCircle className="w-3.5 h-3.5" />
                          Recommendations
                        </h4>
                        {anomaly.recommendations.map((rec, idx) => (
                          <div key={idx} className="flex items-start gap-2 text-xs mb-1.5">
                            <span className={clsx(
                              'px-1.5 py-0.5 rounded text-[10px] font-medium flex-shrink-0',
                              rec.priority === 'immediate' || rec.priority === 'high'
                                ? 'bg-red-500/15 text-red-400'
                                : rec.priority === 'medium'
                                ? 'bg-yellow-500/15 text-yellow-400'
                                : 'bg-stone-500/15 text-stone-400'
                            )}>
                              {rec.priority}
                            </span>
                            <span className="text-stone-300">{rec.action}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Contributing Agents */}
                    {anomaly.contributing_agents && anomaly.contributing_agents.length > 0 && (
                      <div>
                        <h4 className="text-xs font-medium text-purple-400 mb-1.5 flex items-center gap-1.5">
                          <Users className="w-3.5 h-3.5" />
                          Contributing Agents
                        </h4>
                        <div className="flex flex-wrap gap-1.5">
                          {anomaly.contributing_agents.map((agent, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-0.5 bg-purple-500/10 border border-purple-500/20 rounded-full text-[10px] text-purple-300"
                            >
                              {agent}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Agent Perspectives */}
                    {anomaly.agent_perspectives && anomaly.agent_perspectives.length > 1 && (
                      <div>
                        <h4 className="text-xs font-medium text-blue-400 mb-1.5 flex items-center gap-1.5">
                          <Eye className="w-3.5 h-3.5" />
                          Agent Perspectives
                        </h4>
                        <div className="space-y-1.5">
                          {anomaly.agent_perspectives.map((p, idx) => (
                            <div key={idx} className="p-2 bg-stone-700/50 rounded-lg">
                              <span className="text-[10px] font-medium text-blue-300">{p.agent}</span>
                              <p className="text-[10px] text-stone-400 mt-0.5">{p.perspective}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Web References */}
                    {anomaly.web_references && anomaly.web_references.length > 0 && (
                      <div>
                        <h4 className="text-xs font-medium text-cyan-400 mb-1.5 flex items-center gap-1.5">
                          <Globe className="w-3.5 h-3.5" />
                          References
                        </h4>
                        <ul className="space-y-0.5">
                          {anomaly.web_references.map((ref, idx) => (
                            <li key={idx}>
                              <a
                                href={ref}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[10px] text-cyan-400 hover:text-cyan-300 underline truncate block"
                              >
                                {ref}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Feedback */}
                    {systemId && (
                      <FeedbackButtons
                        systemId={systemId}
                        anomalyId={anomaly.id}
                        anomalyTitle={anomaly.title}
                        anomalyType={anomaly.type}
                        severity={anomaly.severity}
                      />
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
