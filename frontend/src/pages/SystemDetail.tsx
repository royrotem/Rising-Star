import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Upload,
  MessageSquare,
  TrendingDown,
  ChevronRight,
  RefreshCw,
  Loader2,
  AlertCircle,
  Download,
  Activity,
  Settings2,
  Target,
  Shield,
  CheckCircle,
  X,
  XCircle,
  MinusCircle,
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi, demoApi } from '../services/api';
import type { GroundTruthEvaluation } from '../services/api';
import type { System, AnalysisResult } from '../types';
import { FeedbackButtons, FeedbackSummaryBanner } from '../components/AnomalyFeedback';
import { useAnalysisStream } from '../hooks/useAnalysisStream';
import { AnalysisStreamPanel } from '../components/AnalysisStreamPanel';
import { reportApi } from '../services/reportApi';
import BaselinePanel from '../components/BaselinePanel';
import WatchdogPanel from '../components/WatchdogPanel';
import { getSeverityCardColor } from '../utils/colors';

// ── Classification color map ────────────────────────────────────────────
const CLASSIFICATION_COLORS: Record<string, { bg: string; text: string; border: string; label: string }> = {
  true_positive: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Detected Correctly' },
  false_negative: { bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/30', label: 'Missed (FN)' },
  false_positive: { bg: 'bg-orange-500/10', text: 'text-orange-400', border: 'border-orange-500/30', label: 'False Alarm (FP)' },
  true_negative: { bg: 'bg-stone-500/10', text: 'text-stone-400', border: 'border-stone-500/30', label: 'Correct Normal' },
};

// ── SVG ROC Curve ───────────────────────────────────────────────────────
function RocCurve({ points, auc }: { points: Array<{ fpr: number; tpr: number }>; auc: number }) {
  const w = 280, h = 220, pad = 40;
  const toX = (fpr: number) => pad + fpr * (w - pad - 10);
  const toY = (tpr: number) => h - pad - tpr * (h - pad - 10);
  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(p.fpr).toFixed(1)},${toY(p.tpr).toFixed(1)}`)
    .join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-[280px]">
      <line x1={pad} y1={h - pad} x2={w - 10} y2={h - pad} stroke="#57534e" strokeWidth="1" />
      <line x1={pad} y1={10} x2={pad} y2={h - pad} stroke="#57534e" strokeWidth="1" />
      <line x1={pad} y1={h - pad} x2={w - 10} y2={10} stroke="#57534e" strokeWidth="0.5" strokeDasharray="4,4" />
      <path d={pathD} fill="none" stroke="#22d3ee" strokeWidth="2.5" strokeLinejoin="round" />
      <path
        d={`${pathD} L${toX(points[points.length - 1]?.fpr ?? 1)},${toY(0)} L${toX(0)},${toY(0)} Z`}
        fill="#22d3ee" fillOpacity="0.08"
      />
      {points.map((p, i) => (
        <circle key={i} cx={toX(p.fpr)} cy={toY(p.tpr)} r="3" fill="#22d3ee" fillOpacity="0.6" />
      ))}
      <text x={w / 2} y={h - 5} textAnchor="middle" fill="#a8a29e" fontSize="10">FPR</text>
      <text x={12} y={h / 2} textAnchor="middle" fill="#a8a29e" fontSize="10" transform={`rotate(-90,12,${h / 2})`}>TPR</text>
      <text x={w - 10} y={20} textAnchor="end" fill="#22d3ee" fontSize="11" fontWeight="bold">AUC = {auc.toFixed(3)}</text>
    </svg>
  );
}

// ── Confusion Matrix ────────────────────────────────────────────────────
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

interface DataStatistics {
  total_records: number;
  total_sources: number;
  field_count: number;
  fields: Array<{
    name: string;
    type: string;
    null_count: number;
    unique_count: number;
    min?: number;
    max?: number;
    mean?: number;
    std?: number;
  }>;
}

interface AnalysisData {
  health_score: number | null;
  data_analyzed?: {
    record_count: number;
    source_count: number;
    field_count: number;
  };
  anomalies: Array<{
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
  }>;
  engineering_margins: Array<{
    component: string;
    parameter: string;
    current_value: number;
    design_limit: number;
    lower_limit?: number;
    margin_percentage: number;
    trend: string;
    safety_critical: boolean;
  }>;
  blind_spots: Array<{
    title: string;
    description: string;
    recommended_sensor?: { type: string; specification: string; estimated_cost: number } | null;
    diagnostic_coverage_improvement: number;
  }>;
  insights?: string[];
  insights_summary?: string;
  trend_analysis?: Record<string, {
    direction: string;
    change_percentage: number;
    volatility: string;
  }>;
  recommendations?: Array<{
    type: string;
    priority: string;
    action: string;
    source_anomaly?: string;
  }>;
  ai_analysis?: {
    ai_powered: boolean;
    agents_used: string[];
    agent_statuses: Array<{ agent: string; status: string; findings: number; perspective?: string; error?: string }>;
    total_findings_raw: number;
    total_anomalies_unified: number;
  };
}

function getTrendIcon(trend: string) {
  if (trend === 'degrading') return <TrendingDown className="w-3.5 h-3.5 text-red-400" />;
  if (trend === 'improving') return <TrendingDown className="w-3.5 h-3.5 text-emerald-400 rotate-180" />;
  return <Activity className="w-3.5 h-3.5 text-stone-400" />;
}

export default function SystemDetail() {
  const { systemId } = useParams();
  const navigate = useNavigate();
  const [system, setSystem] = useState<System | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [statistics, setStatistics] = useState<DataStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedAnomaly, setSelectedAnomaly] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const { stream, startStream } = useAnalysisStream();

  // Ground truth evaluation
  const [evaluation, setEvaluation] = useState<GroundTruthEvaluation | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [rowFilter, setRowFilter] = useState<string | null>(null);

  // Agent selection
  const [showAgentConfig, setShowAgentConfig] = useState(false);
  const [availableAgents, setAvailableAgents] = useState<Array<{ name: string; perspective: string }>>([]);
  const [selectedAgents, setSelectedAgents] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (stream.result && !stream.active) {
      const r = stream.result as AnalysisResult & Record<string, unknown>;
      setAnalysis({
        health_score: r.health_score,
        data_analyzed: r.data_analyzed,
        anomalies: (r.anomalies as AnalysisData['anomalies']) || [],
        engineering_margins: (r.engineering_margins as AnalysisData['engineering_margins']) || [],
        blind_spots: (r.blind_spots as AnalysisData['blind_spots']) || [],
        insights_summary: r.insights_summary as string | undefined,
        insights: r.insights as string[] | undefined,
        ai_analysis: r.ai_analysis as AnalysisData['ai_analysis'],
      });
      if (system && r.health_score) {
        setSystem({ ...system, health_score: r.health_score as number });
      }
      setAnalyzing(false);

      // Auto-run ground truth evaluation if this is a demo system
      if (systemId && system?.is_demo) {
        setEvalLoading(true);
        demoApi.evaluate(systemId)
          .then((evalResult) => {
            setEvaluation(evalResult);
            setShowEvaluation(true);
          })
          .catch(() => { /* ground truth not available */ })
          .finally(() => setEvalLoading(false));
      }
    }
  }, [stream.result, stream.active]);

  useEffect(() => {
    if (stream.error && !stream.active) {
      setError(stream.error);
      setAnalyzing(false);
    }
  }, [stream.error, stream.active]);

  useEffect(() => {
    loadSystem();
    // Load available agents
    fetch('/api/v1/systems/available-agents')
      .then((r) => r.ok ? r.json() : null)
      .then((data) => {
        if (data?.agents) {
          setAvailableAgents(data.agents);
          setSelectedAgents(new Set(data.agents.map((a: { name: string }) => a.name)));
        }
      })
      .catch(() => { /* agents list not critical */ });
  }, [systemId]);

  const loadSystem = async () => {
    if (!systemId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await systemsApi.get(systemId);
      setSystem(data as System);

      try {
        const statsResponse = await fetch(`/api/v1/systems/${systemId}/statistics`);
        if (statsResponse.ok) {
          const stats = await statsResponse.json();
          setStatistics(stats);
        }
      } catch {
        // Statistics not available yet
      }

      const saved = await systemsApi.getAnalysis(systemId);
      if (saved) {
        setAnalysis({
          health_score: saved.health_score ?? (data as System).health_score ?? null,
          data_analyzed: saved.data_analyzed,
          anomalies: (saved.anomalies as AnalysisData['anomalies']) || [],
          engineering_margins: (saved.engineering_margins as AnalysisData['engineering_margins']) || [],
          blind_spots: (saved.blind_spots as AnalysisData['blind_spots']) || [],
          insights_summary: saved.insights_summary,
          insights: saved.insights,
          ai_analysis: saved.ai_analysis,
        });
        // If demo system with existing analysis, load ground truth evaluation
        if ((data as System).is_demo && saved.anomalies && saved.anomalies.length > 0) {
          setEvalLoading(true);
          demoApi.evaluate(systemId)
            .then((evalResult) => {
              setEvaluation(evalResult);
              setShowEvaluation(true);
            })
            .catch(() => { /* ground truth not available */ })
            .finally(() => setEvalLoading(false));
        }
      } else {
        setAnalysis({
          health_score: (data as System).health_score || null,
          anomalies: [],
          engineering_margins: [],
          blind_spots: [],
        });
      }
    } catch (error) {
      console.error('Failed to load system:', error);
      setError('Failed to load system. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = () => {
    if (!systemId) return;
    setAnalyzing(true);
    setError(null);
    const agentList = selectedAgents.size < availableAgents.length
      ? Array.from(selectedAgents)
      : undefined;
    startStream(systemId, agentList);
  };

  const toggleAgent = (agentName: string) => {
    setSelectedAgents((prev) => {
      const next = new Set(prev);
      if (next.has(agentName)) {
        if (next.size > 1) next.delete(agentName); // keep at least 1
      } else {
        next.add(agentName);
      }
      return next;
    });
  };

  const toggleAllAgents = () => {
    if (selectedAgents.size === availableAgents.length) {
      // Deselect all except first (must keep at least 1)
      setSelectedAgents(new Set([availableAgents[0]?.name].filter(Boolean)));
    } else {
      setSelectedAgents(new Set(availableAgents.map((a) => a.name)));
    }
  };

  const handleDownloadReport = async () => {
    if (!systemId || downloading) return;
    setDownloading(true);
    try {
      await reportApi.downloadReport(systemId);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Download failed';
      setError(msg);
    } finally {
      setDownloading(false);
    }
  };

  const getProgressWidth = (marginPct: number) => {
    return (100 - marginPct) + '%';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-6 h-6 text-stone-400 animate-spin" />
      </div>
    );
  }

  if (error && !system) {
    return (
      <div className="p-8 text-center">
        <p className="text-stone-400 mb-4">{error}</p>
        <Link to="/systems" className="text-primary-400 hover:text-primary-300 text-sm">
          Back to Systems
        </Link>
      </div>
    );
  }

  if (!system) {
    return (
      <div className="p-8 text-center">
        <p className="text-stone-400">System not found</p>
        <Link to="/systems" className="text-primary-400 hover:text-primary-300 text-sm">
          Back to Systems
        </Link>
      </div>
    );
  }

  const hasData = statistics && statistics.total_records > 0;

  return (
    <div className="p-8 page-enter">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <Link to="/systems" className="p-2 hover:bg-stone-700 rounded-lg transition-colors">
          <ArrowLeft className="w-4 h-4 text-stone-400" />
        </Link>
        <div className="flex-1">
          <h1 className="text-xl font-semibold text-white">{system.name}</h1>
          <p className="text-stone-400 text-sm capitalize">{system.system_type.replace('_', ' ')}</p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to={`/systems/${systemId}/chat`}
            className="flex items-center gap-2 px-3.5 py-2 text-stone-400 hover:text-white hover:bg-stone-700 rounded-lg text-sm transition-colors"
          >
            <MessageSquare className="w-4 h-4" />
            Ask AI
          </Link>
          <button
            onClick={handleDownloadReport}
            disabled={downloading}
            className="flex items-center gap-2 px-3.5 py-2 text-stone-400 hover:text-white hover:bg-stone-700 disabled:opacity-50 rounded-lg text-sm transition-colors"
          >
            <Download className={clsx("w-4 h-4", downloading && "animate-bounce")} />
            PDF
          </button>
          <button
            onClick={() => navigate('/systems/new')}
            className="flex items-center gap-2 px-3.5 py-2 text-stone-400 hover:text-white hover:bg-stone-700 rounded-lg text-sm transition-colors"
          >
            <Upload className="w-4 h-4" />
            Upload
          </button>
          <button
            onClick={() => setShowAgentConfig(!showAgentConfig)}
            className={clsx(
              "flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm transition-colors",
              showAgentConfig
                ? "text-primary-400 bg-primary-500/10"
                : "text-stone-400 hover:text-white hover:bg-stone-700"
            )}
          >
            <Settings2 className="w-4 h-4" />
            Agents
            {selectedAgents.size < availableAgents.length && (
              <span className="px-1.5 py-0.5 bg-primary-500/20 text-primary-400 text-[10px] rounded-full tabular-nums">
                {selectedAgents.size}/{availableAgents.length}
              </span>
            )}
          </button>
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <RefreshCw className={clsx("w-3.5 h-3.5", analyzing && "animate-spin")} />
            {analyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {/* Agent Selection Panel */}
      {showAgentConfig && availableAgents.length > 0 && (
        <div className="mb-6 glass-card p-4 page-enter">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Settings2 className="w-4 h-4 text-stone-400" />
              <p className="text-sm font-medium text-white">AI Agent Configuration</p>
            </div>
            <button
              onClick={toggleAllAgents}
              className="text-xs text-stone-400 hover:text-white transition-colors"
            >
              {selectedAgents.size === availableAgents.length ? 'Deselect All' : 'Select All'}
            </button>
          </div>
          <p className="text-xs text-stone-400 mb-3">
            Choose which AI agents to include in the analysis. Each agent provides a unique analytical perspective.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
            {availableAgents.map((agent) => {
              const isSelected = selectedAgents.has(agent.name);
              return (
                <button
                  key={agent.name}
                  onClick={() => toggleAgent(agent.name)}
                  className={clsx(
                    'p-2.5 rounded-lg text-left transition-all border',
                    isSelected
                      ? 'bg-primary-500/10 border-primary-500/30 hover:bg-primary-500/15'
                      : 'bg-stone-700/30 border-stone-600/30 hover:bg-stone-700/50 opacity-50'
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <div className={clsx(
                      'w-3 h-3 rounded-sm border flex items-center justify-center flex-shrink-0',
                      isSelected
                        ? 'bg-primary-500 border-primary-500'
                        : 'border-stone-500'
                    )}>
                      {isSelected && (
                        <svg className="w-2 h-2 text-white" viewBox="0 0 12 12" fill="none">
                          <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      )}
                    </div>
                    <p className="text-xs font-medium text-stone-200 truncate">{agent.name}</p>
                  </div>
                  <p className="text-[10px] text-stone-400 line-clamp-2 pl-5">{agent.perspective}</p>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-6 bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <p className="text-sm text-stone-300">{error}</p>
        </div>
      )}

      {/* No Data Banner */}
      {!hasData && (
        <div className="mb-8 glass-card p-6">
          <h3 className="text-sm font-medium text-white mb-1">No Data Ingested</h3>
          <p className="text-xs text-stone-400 mb-4">
            Upload telemetry data to enable analysis and anomaly detection.
          </p>
          <button
            onClick={() => navigate('/systems/new')}
            className="px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white text-sm rounded-lg font-medium transition-colors"
          >
            Upload Data
          </button>
        </div>
      )}

      {/* Data Statistics */}
      {hasData && statistics && (
        <div className="grid grid-cols-4 gap-4 mb-8">
          <div className="stat-card">
            <p className="section-header mb-2">Records</p>
            <p className="text-xl font-semibold text-white tabular-nums">{statistics.total_records.toLocaleString()}</p>
          </div>
          <div className="stat-card">
            <p className="section-header mb-2">Sources</p>
            <p className="text-xl font-semibold text-white tabular-nums">{statistics.total_sources}</p>
          </div>
          <div className="stat-card">
            <p className="section-header mb-2">Fields</p>
            <p className="text-xl font-semibold text-white tabular-nums">{statistics.field_count}</p>
          </div>
          <div className="stat-card">
            <p className="section-header mb-2">Status</p>
            <p className="text-xl font-semibold text-white capitalize">{system.status?.replace('_', ' ') || 'Active'}</p>
          </div>
        </div>
      )}

      {/* Health Score */}
      {analysis && (
        <div className="glass-card p-6 mb-6">
          <div className="flex items-start justify-between">
            <div className="flex-1 mr-8">
              <p className="section-header mb-2">System Health</p>
              <p className="text-sm text-stone-400 leading-relaxed whitespace-pre-line">
                {analysis.insights_summary || 'Run analysis to get detailed health insights'}
              </p>
            </div>
            <div className="flex-shrink-0">
              {analysis.health_score !== null ? (
                <div className={clsx(
                  'text-4xl font-semibold tabular-nums',
                  analysis.health_score >= 90 ? 'text-emerald-400' :
                  analysis.health_score >= 70 ? 'text-yellow-400' : 'text-red-400'
                )}>
                  {analysis.health_score.toFixed(0)}%
                </div>
              ) : (
                <div className="text-3xl text-stone-500">--</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* AI Chat link */}
      <Link
        to={`/systems/${systemId}/chat`}
        className="block mb-6 glass-card p-4 hover:border-stone-500 transition-all group"
      >
        <div className="flex items-center gap-3">
          <MessageSquare className="w-4 h-4 text-stone-400 group-hover:text-primary-400 transition-colors" />
          <div className="flex-1">
            <p className="text-sm text-stone-300 group-hover:text-white transition-colors">
              Ask AI about this system
            </p>
            <p className="text-xs text-stone-400 mt-0.5">
              Natural language queries about anomalies, patterns, and recommendations
            </p>
          </div>
          <ChevronRight className="w-4 h-4 text-stone-500 group-hover:text-stone-400 transition-colors" />
        </div>
      </Link>

      {/* Live Analysis Progress */}
      {analyzing && <AnalysisStreamPanel stream={stream} />}

      {/* AI Agents Status */}
      {analysis?.ai_analysis && (
        <div className="glass-card p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <p className="section-header">AI Multi-Agent Analysis</p>
            <span className={clsx(
              'px-2 py-0.5 rounded text-[10px] font-medium',
              analysis.ai_analysis.ai_powered
                ? 'bg-primary-500/10 text-primary-400'
                : 'bg-stone-700 text-stone-400'
            )}>
              {analysis.ai_analysis.ai_powered ? 'LLM Powered' : 'Rule-Based'}
            </span>
          </div>
          <div className="grid grid-cols-4 sm:grid-cols-5 lg:grid-cols-7 gap-2">
            {analysis.ai_analysis.agent_statuses.map((agent, idx) => (
              <div
                key={idx}
                className={clsx(
                  'p-2.5 rounded-lg text-center',
                  agent.status === 'success'
                    ? 'bg-emerald-500/5 border border-emerald-500/15'
                    : 'bg-red-500/5 border border-red-500/15'
                )}
              >
                <p className="text-[11px] font-medium text-stone-300 truncate">{agent.agent}</p>
                <p className="text-[10px] text-stone-400 mt-0.5">
                  {agent.status === 'success' ? `${agent.findings} findings` : 'Error'}
                </p>
              </div>
            ))}
          </div>
          {analysis.ai_analysis.total_findings_raw > 0 && (
            <p className="text-[11px] text-stone-400 mt-3">
              {analysis.ai_analysis.total_findings_raw} raw findings → {analysis.ai_analysis.total_anomalies_unified} unified anomalies
            </p>
          )}
        </div>
      )}

      {/* Watchdog */}
      {systemId && <WatchdogPanel systemId={systemId} />}

      {/* Baseline & Historical */}
      {systemId && <BaselinePanel systemId={systemId} />}

      {/* Key Insights */}
      {analysis && analysis.insights && analysis.insights.length > 0 && (
        <div className="glass-card p-6 mb-6">
          <p className="section-header mb-4">Key Insights</p>
          <div className="space-y-2">
            {analysis.insights.map((insight, idx) => (
              <div key={idx} className={clsx(
                'p-3 rounded-lg text-sm text-stone-300',
                insight.toLowerCase().includes('urgent') || insight.toLowerCase().includes('critical')
                  ? 'bg-red-500/10 border border-red-500/15'
                  : insight.toLowerCase().includes('warning') || insight.toLowerCase().includes('high')
                  ? 'bg-orange-500/10 border border-orange-500/15'
                  : 'bg-stone-600/50'
              )}>
                {insight}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Anomaly Feedback Summary */}
      {systemId && analysis && analysis.anomalies.length > 0 && (
        <FeedbackSummaryBanner systemId={systemId} />
      )}

      {/* ════════════════════════════════════════════════════════════════
          GROUND TRUTH EVALUATION PANEL (Kaggle TLM-UAV)
          ════════════════════════════════════════════════════════════════ */}
      {evalLoading && (
        <div className="mb-6 glass-card p-4 border border-cyan-500/20 flex items-center gap-3">
          <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
          <span className="text-sm text-cyan-400">Evaluating against ground truth labels...</span>
        </div>
      )}

      {evaluation && (
        <div className="mb-6">
          <button
            onClick={() => setShowEvaluation(!showEvaluation)}
            className={clsx(
              'w-full flex items-center gap-2 px-5 py-3 rounded-xl text-sm font-medium transition-colors mb-3',
              showEvaluation
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                : 'glass-card text-stone-400 hover:text-cyan-400'
            )}
          >
            <Target className="w-4 h-4" />
            Ground Truth Evaluation — Kaggle TLM:UAV Dataset
            <span className="ml-auto text-xs tabular-nums">
              Accuracy {(evaluation.evaluation.accuracy * 100).toFixed(1)}% | F1 {(evaluation.evaluation.f1_score * 100).toFixed(1)}%
            </span>
          </button>

          {showEvaluation && (
            <div className="space-y-4 animate-fade-in">
              {/* Metrics Banner */}
              <div className="glass-card p-5 border border-cyan-500/20">
                <div className="flex items-center gap-2 mb-4">
                  <Shield className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-cyan-400">Detection Metrics</h3>
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
                  <div className="flex gap-1.5 ml-auto">
                    {[
                      { key: null as string | null, label: 'All' },
                      { key: 'true_positive', label: 'TP' },
                      { key: 'false_negative', label: 'FN' },
                      { key: 'false_positive', label: 'FP' },
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
                      {(rowFilter
                        ? evaluation.per_row_sample.filter((r) => r.classification === rowFilter)
                        : evaluation.per_row_sample
                      ).map((row) => {
                        const cls = CLASSIFICATION_COLORS[row.classification] || CLASSIFICATION_COLORS.true_negative;
                        return (
                          <tr key={row.row_index} className={clsx(cls.bg, 'transition-colors')}>
                            <td className="px-3 py-1.5 tabular-nums text-stone-300 font-mono">#{row.row_index}</td>
                            <td className="px-3 py-1.5">
                              <span className={clsx(
                                'px-1.5 py-0.5 rounded text-[10px] font-medium',
                                row.ground_truth_label === 0 ? 'bg-stone-600/50 text-stone-300' : 'bg-amber-500/15 text-amber-400'
                              )}>
                                {row.ground_truth_fault}
                              </span>
                            </td>
                            <td className="px-3 py-1.5">
                              {row.detected
                                ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                                : <X className="w-3.5 h-3.5 text-stone-500" />
                              }
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
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        {/* Anomalies */}
        <div className="glass-card">
          <div className="px-6 py-4 border-b border-stone-600/40 flex items-center justify-between">
            <p className="text-sm font-medium text-white">Detected Anomalies</p>
            {analysis && analysis.anomalies.length > 0 && systemId && (
              <Link
                to={`/systems/${systemId}/anomalies`}
                className="text-xs text-stone-400 hover:text-stone-300 transition-colors"
              >
                Explore All
              </Link>
            )}
          </div>
          <div className="p-4 space-y-3">
            {!analysis || analysis.anomalies.length === 0 ? (
              <div className="text-center py-10">
                <p className="text-stone-400 text-sm">
                  {hasData ? 'No anomalies detected' : 'No data to analyze'}
                </p>
                <p className="text-stone-400 text-xs mt-1">
                  {hasData
                    ? 'System operating within normal parameters'
                    : 'Upload data and run analysis'}
                </p>
              </div>
            ) : (
              analysis.anomalies.map((anomaly) => (
                <div
                  key={anomaly.id}
                  className={clsx(
                    'p-4 rounded-lg border-l-[3px] cursor-pointer transition-colors',
                    getSeverityCardColor(anomaly.severity),
                    selectedAnomaly === anomaly.id ? 'ring-1 ring-primary-500/30' : ''
                  )}
                  onClick={() => setSelectedAnomaly(
                    selectedAnomaly === anomaly.id ? null : anomaly.id
                  )}
                >
                  <div className="flex items-start justify-between mb-1.5">
                    <h3 className="text-sm font-medium text-white">{anomaly.title}</h3>
                    <span className="text-xs text-stone-400 tabular-nums ml-3 flex-shrink-0">
                      {anomaly.impact_score.toFixed(1)}
                    </span>
                  </div>
                  <p className="text-xs text-stone-400 leading-relaxed">{anomaly.description}</p>

                  {selectedAnomaly === anomaly.id && (
                    <div className="mt-4 pt-3 border-t border-stone-600/50 space-y-3">
                      {anomaly.confidence && (
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] text-stone-400">Confidence</span>
                          <div className="flex-1 bg-stone-600/50 rounded-full h-1.5 max-w-[80px]">
                            <div
                              className="bg-primary-500 h-1.5 rounded-full"
                              style={{ width: `${anomaly.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-[11px] text-stone-400 tabular-nums">{(anomaly.confidence * 100).toFixed(0)}%</span>
                        </div>
                      )}

                      <div>
                        <p className="text-[11px] text-stone-400 mb-1">Analysis</p>
                        <p className="text-xs text-stone-300 leading-relaxed">{anomaly.natural_language_explanation}</p>
                      </div>

                      {anomaly.possible_causes && anomaly.possible_causes.length > 0 && (
                        <div>
                          <p className="text-[11px] text-stone-400 mb-1">Possible Causes</p>
                          <ul className="space-y-1">
                            {anomaly.possible_causes.map((cause, idx) => (
                              <li key={idx} className="text-xs text-stone-300 pl-3 relative before:content-[''] before:absolute before:left-0 before:top-[7px] before:w-1 before:h-1 before:bg-stone-600 before:rounded-full">
                                {cause}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {anomaly.recommendations && anomaly.recommendations.length > 0 && (
                        <div>
                          <p className="text-[11px] text-stone-400 mb-1">Recommendations</p>
                          {anomaly.recommendations.map((rec, idx) => (
                            <div key={idx} className="flex items-start gap-2 text-xs mb-1.5">
                              <span className={clsx(
                                'px-1.5 py-0.5 rounded text-[10px] font-medium flex-shrink-0',
                                rec.priority === 'immediate' || rec.priority === 'high' ? 'bg-red-500/10 text-red-400' :
                                rec.priority === 'medium' ? 'bg-yellow-500/10 text-yellow-400' :
                                'bg-stone-700 text-stone-400'
                              )}>
                                {rec.priority}
                              </span>
                              <span className="text-stone-300">{rec.action}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {anomaly.contributing_agents && anomaly.contributing_agents.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {anomaly.contributing_agents.map((agent, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-0.5 bg-stone-700 rounded text-[10px] text-stone-400"
                            >
                              {agent}
                            </span>
                          ))}
                        </div>
                      )}

                      {anomaly.agent_perspectives && anomaly.agent_perspectives.length > 1 && (
                        <div>
                          <p className="text-[11px] text-stone-400 mb-1">Perspectives</p>
                          <div className="space-y-1.5">
                            {anomaly.agent_perspectives.map((p, idx) => (
                              <div key={idx} className="p-2 bg-stone-600/50 rounded-lg">
                                <span className="text-[10px] font-medium text-stone-400">{p.agent}</span>
                                <p className="text-[10px] text-stone-400 mt-0.5">{p.perspective}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {anomaly.web_references && anomaly.web_references.length > 0 && (
                        <div>
                          <p className="text-[11px] text-stone-400 mb-1">References</p>
                          <ul className="space-y-0.5">
                            {anomaly.web_references.map((ref, idx) => (
                              <li key={idx}>
                                <a
                                  href={ref}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-[10px] text-primary-400 hover:text-primary-300 underline truncate block"
                                >
                                  {ref}
                                </a>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

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
              ))
            )}
          </div>
        </div>

        {/* Engineering Margins */}
        <div className="glass-card">
          <div className="px-6 py-4 border-b border-stone-600/40">
            <p className="text-sm font-medium text-white">Engineering Margins</p>
          </div>
          <div className="p-4 space-y-3">
            {!analysis || analysis.engineering_margins.length === 0 ? (
              <div className="text-center py-10">
                <p className="text-stone-400 text-sm">
                  {hasData ? 'No margins calculated' : 'No data available'}
                </p>
                <p className="text-stone-400 text-xs mt-1">
                  {hasData
                    ? 'Run analysis to calculate margins'
                    : 'Upload numeric data first'}
                </p>
              </div>
            ) : (
              analysis.engineering_margins.map((margin, idx) => (
                <div key={idx} className="p-3.5 bg-stone-700/40 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <div>
                      <h3 className="text-sm text-white">{margin.component}</h3>
                      <p className="text-xs text-stone-400">{margin.parameter}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      {getTrendIcon(margin.trend)}
                      {margin.safety_critical && (
                        <span className="px-1.5 py-0.5 text-[10px] bg-red-500/10 text-red-400 rounded">
                          Safety
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="mt-2.5">
                    <div className="flex justify-between text-[11px] mb-1.5">
                      <span className="text-stone-400 tabular-nums">{margin.current_value.toFixed(2)}</span>
                      <span className="text-stone-400 tabular-nums">Limit: {margin.design_limit.toFixed(2)}</span>
                    </div>
                    <div className="h-1.5 bg-stone-600/50 rounded-full overflow-hidden">
                      <div
                        className={clsx(
                          'h-full rounded-full',
                          margin.margin_percentage < 15 ? 'bg-red-400' :
                          margin.margin_percentage < 30 ? 'bg-yellow-400' : 'bg-emerald-400'
                        )}
                        style={{ width: getProgressWidth(margin.margin_percentage) }}
                      />
                    </div>
                    <p className="text-[11px] text-stone-400 mt-1 tabular-nums">
                      {margin.margin_percentage.toFixed(1)}% remaining
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Blind Spots */}
        <div className="col-span-2 glass-card">
          <div className="px-6 py-4 border-b border-stone-600/40">
            <p className="text-sm font-medium text-white">Blind Spots & Recommendations</p>
          </div>
          <div className="p-4">
            {!analysis || analysis.blind_spots.length === 0 ? (
              <div className="text-center py-10">
                <p className="text-stone-400 text-sm">No blind spots identified</p>
                <p className="text-stone-400 text-xs mt-1">
                  Run analysis with data to identify gaps
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                {analysis.blind_spots.map((spot, idx) => (
                  <div key={idx} className="p-4 bg-stone-700/40 rounded-lg">
                    <h3 className="text-sm font-medium text-white mb-1.5">{spot.title}</h3>
                    <p className="text-xs text-stone-400 leading-relaxed mb-3">{spot.description}</p>
                    {spot.recommended_sensor && (
                      <div className="p-2.5 bg-stone-700/50 rounded-lg text-xs">
                        <p className="text-stone-400 mb-1.5">Recommended Sensor</p>
                        <div className="grid grid-cols-2 gap-1 text-[11px]">
                          <span className="text-stone-400">Type:</span>
                          <span className="text-stone-300">{spot.recommended_sensor.type}</span>
                          <span className="text-stone-400">Spec:</span>
                          <span className="text-stone-300">{spot.recommended_sensor.specification}</span>
                          <span className="text-stone-400">Cost:</span>
                          <span className="text-stone-300">${spot.recommended_sensor.estimated_cost}</span>
                        </div>
                      </div>
                    )}
                    <p className="text-[11px] text-stone-400 mt-2">
                      Coverage gain: <span className="text-emerald-400 tabular-nums">+{spot.diagnostic_coverage_improvement}%</span>
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Field Statistics */}
        {hasData && statistics && statistics.fields.length > 0 && (
          <div className="col-span-2 glass-card">
            <div className="px-6 py-4 border-b border-stone-600/40">
              <p className="text-sm font-medium text-white">Field Statistics</p>
            </div>
            <div className="p-4 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-stone-400 border-b border-stone-600/40">
                    <th className="pb-2.5 pr-4 font-medium">Field</th>
                    <th className="pb-2.5 pr-4 font-medium">Type</th>
                    <th className="pb-2.5 pr-4 font-medium">Unique</th>
                    <th className="pb-2.5 pr-4 font-medium">Min</th>
                    <th className="pb-2.5 pr-4 font-medium">Max</th>
                    <th className="pb-2.5 pr-4 font-medium">Mean</th>
                    <th className="pb-2.5 font-medium">Std</th>
                  </tr>
                </thead>
                <tbody>
                  {statistics.fields.map((field, idx) => (
                    <tr key={idx} className="border-b border-stone-600/40/50 text-stone-400">
                      <td className="py-2.5 pr-4 font-mono text-stone-300">{field.name}</td>
                      <td className="py-2.5 pr-4">{field.type}</td>
                      <td className="py-2.5 pr-4 tabular-nums">{field.unique_count}</td>
                      <td className="py-2.5 pr-4 tabular-nums">{field.min !== undefined ? field.min.toFixed(2) : '-'}</td>
                      <td className="py-2.5 pr-4 tabular-nums">{field.max !== undefined ? field.max.toFixed(2) : '-'}</td>
                      <td className="py-2.5 pr-4 tabular-nums">{field.mean !== undefined ? field.mean.toFixed(2) : '-'}</td>
                      <td className="py-2.5 tabular-nums">{field.std !== undefined ? field.std.toFixed(2) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
