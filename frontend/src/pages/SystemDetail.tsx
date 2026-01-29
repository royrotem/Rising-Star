import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Upload,
  MessageSquare,
  Activity,
  AlertTriangle,
  TrendingDown,
  Lightbulb,
  ChevronRight,
  RefreshCw,
  Loader2,
  CheckCircle,
  Database,
  FileText,
  AlertCircle,
  Globe,
  Users,
  Brain,
  Cpu,
  Eye,
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';
import type { System, AnalysisResult } from '../types';

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

function getSeverityColor(severity: string) {
  switch (severity) {
    case 'critical': return 'border-red-500 bg-red-500/10';
    case 'high': return 'border-orange-500 bg-orange-500/10';
    case 'medium': return 'border-yellow-500 bg-yellow-500/10';
    case 'low': return 'border-green-500 bg-green-500/10';
    default: return 'border-slate-500 bg-slate-500/10';
  }
}

function getTrendIcon(trend: string) {
  if (trend === 'degrading') return <TrendingDown className="w-4 h-4 text-red-500" />;
  if (trend === 'improving') return <TrendingDown className="w-4 h-4 text-green-500 rotate-180" />;
  return <Activity className="w-4 h-4 text-slate-500" />;
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

  useEffect(() => {
    loadSystem();
  }, [systemId]);

  const loadSystem = async () => {
    if (!systemId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await systemsApi.get(systemId);
      setSystem(data as System);

      // Load statistics
      try {
        const statsResponse = await fetch(`/api/v1/systems/${systemId}/statistics`);
        if (statsResponse.ok) {
          const stats = await statsResponse.json();
          setStatistics(stats);
        }
      } catch {
        console.log('No statistics available');
      }

      // Initialize empty analysis
      setAnalysis({
        health_score: (data as System).health_score || null,
        anomalies: [],
        engineering_margins: [],
        blind_spots: [],
      });
    } catch (error) {
      console.error('Failed to load system:', error);
      setError('Failed to load system. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!systemId) return;
    setAnalyzing(true);
    setError(null);
    try {
      const result = await systemsApi.analyze(systemId);
      if (result) {
        setAnalysis({
          health_score: result.health_score,
          data_analyzed: result.data_analyzed,
          anomalies: result.anomalies || [],
          engineering_margins: result.engineering_margins || [],
          blind_spots: result.blind_spots || [],
          insights_summary: result.insights_summary,
          insights: result.insights,
          ai_analysis: result.ai_analysis,
        });
        if (system && result.health_score) {
          setSystem({ ...system, health_score: result.health_score });
        }
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setError('Analysis failed. Make sure you have uploaded data first.');
    } finally {
      setAnalyzing(false);
    }
  };

  const getProgressWidth = (marginPct: number) => {
    return (100 - marginPct) + '%';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  if (error && !system) {
    return (
      <div className="p-8 text-center">
        <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <p className="text-slate-400 mb-4">{error}</p>
        <Link to="/systems" className="text-primary-400 hover:text-primary-300">
          Back to Systems
        </Link>
      </div>
    );
  }

  if (!system) {
    return (
      <div className="p-8 text-center">
        <p className="text-slate-400">System not found</p>
        <Link to="/systems" className="text-primary-400 hover:text-primary-300">
          Back to Systems
        </Link>
      </div>
    );
  }

  const hasData = statistics && statistics.total_records > 0;

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link to="/systems" className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
          <ArrowLeft className="w-5 h-5 text-slate-400" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">{system.name}</h1>
          <p className="text-slate-400 capitalize">{system.system_type.replace('_', ' ')} | ID: {systemId}</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/systems/new')}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
          >
            <Upload className="w-4 h-4" />
            Upload Data
          </button>
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white rounded-lg font-medium transition-colors"
          >
            <RefreshCw className={clsx("w-4 h-4", analyzing && "animate-spin")} />
            {analyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <p className="text-sm text-slate-300">{error}</p>
          </div>
        </div>
      )}

      {/* No Data Banner */}
      {!hasData && (
        <div className="mb-6 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-6">
          <div className="flex items-start gap-4">
            <Database className="w-8 h-8 text-yellow-400 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-yellow-400 mb-1">No Data Ingested</h3>
              <p className="text-sm text-slate-300 mb-4">
                Upload telemetry data to enable analysis, anomaly detection, and insights.
                The AI will automatically discover your data schema and learn your system's patterns.
              </p>
              <button
                onClick={() => navigate('/systems/new')}
                className="px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-black rounded-lg font-medium transition-colors"
              >
                Upload Data Now
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Data Statistics */}
      {hasData && statistics && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <Database className="w-6 h-6 text-primary-400 mb-2" />
            <p className="text-2xl font-bold text-white">{statistics.total_records.toLocaleString()}</p>
            <p className="text-sm text-slate-400">Total Records</p>
          </div>
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <FileText className="w-6 h-6 text-primary-400 mb-2" />
            <p className="text-2xl font-bold text-white">{statistics.total_sources}</p>
            <p className="text-sm text-slate-400">Data Sources</p>
          </div>
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <Activity className="w-6 h-6 text-primary-400 mb-2" />
            <p className="text-2xl font-bold text-white">{statistics.field_count}</p>
            <p className="text-sm text-slate-400">Fields</p>
          </div>
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <CheckCircle className="w-6 h-6 text-green-400 mb-2" />
            <p className="text-2xl font-bold text-white">{system.status || 'Active'}</p>
            <p className="text-sm text-slate-400">Status</p>
          </div>
        </div>
      )}

      {/* Health Score */}
      {analysis && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white mb-1">System Health</h2>
              <p className="text-slate-400 text-sm whitespace-pre-line">
                {analysis.insights_summary || 'Run analysis to get detailed health insights'}
              </p>
            </div>
            <div className="text-right">
              {analysis.health_score !== null ? (
                <div className={clsx(
                  'text-5xl font-bold',
                  analysis.health_score >= 90 ? 'text-green-500' :
                  analysis.health_score >= 70 ? 'text-yellow-500' : 'text-red-500'
                )}>
                  {analysis.health_score.toFixed(0)}%
                </div>
              ) : (
                <div className="text-3xl text-slate-500">--</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* AI Agents Status */}
      {analysis?.ai_analysis && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              AI Multi-Agent Analysis
            </h2>
            <span className={clsx(
              'px-3 py-1 rounded-full text-xs font-medium',
              analysis.ai_analysis.ai_powered
                ? 'bg-purple-500/20 text-purple-400'
                : 'bg-slate-600/50 text-slate-400'
            )}>
              {analysis.ai_analysis.ai_powered ? 'LLM Powered' : 'Rule-Based Fallback'}
            </span>
          </div>
          <div className="grid grid-cols-5 gap-3">
            {analysis.ai_analysis.agent_statuses.map((agent, idx) => (
              <div
                key={idx}
                className={clsx(
                  'p-3 rounded-lg border text-center',
                  agent.status === 'success'
                    ? 'border-green-500/30 bg-green-500/5'
                    : 'border-red-500/30 bg-red-500/5'
                )}
              >
                <Cpu className={clsx(
                  'w-5 h-5 mx-auto mb-2',
                  agent.status === 'success' ? 'text-green-400' : 'text-red-400'
                )} />
                <p className="text-xs font-medium text-white truncate">{agent.agent}</p>
                <p className="text-xs text-slate-500 mt-1">
                  {agent.status === 'success' ? `${agent.findings} findings` : 'Error'}
                </p>
              </div>
            ))}
          </div>
          {analysis.ai_analysis.total_findings_raw > 0 && (
            <p className="text-xs text-slate-500 mt-3 text-center">
              {analysis.ai_analysis.total_findings_raw} raw findings merged into{' '}
              {analysis.ai_analysis.total_anomalies_unified} unified anomalies
            </p>
          )}
        </div>
      )}

      {/* Key Insights */}
      {analysis && analysis.insights && analysis.insights.length > 0 && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-500" />
            Key Insights
          </h2>
          <div className="space-y-2">
            {analysis.insights.map((insight, idx) => (
              <div key={idx} className={clsx(
                'p-3 rounded-lg flex items-start gap-3',
                insight.toLowerCase().includes('urgent') || insight.toLowerCase().includes('critical')
                  ? 'bg-red-500/10 border border-red-500/30'
                  : insight.toLowerCase().includes('warning') || insight.toLowerCase().includes('high')
                  ? 'bg-orange-500/10 border border-orange-500/30'
                  : 'bg-slate-700/50'
              )}>
                <AlertCircle className={clsx(
                  'w-5 h-5 mt-0.5',
                  insight.toLowerCase().includes('urgent') || insight.toLowerCase().includes('critical')
                    ? 'text-red-400'
                    : insight.toLowerCase().includes('warning') || insight.toLowerCase().includes('high')
                    ? 'text-orange-400'
                    : 'text-slate-400'
                )} />
                <p className="text-sm text-slate-300">{insight}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        {/* Anomalies */}
        <div className="bg-slate-800 rounded-xl border border-slate-700">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-white">Detected Anomalies</h2>
          </div>
          <div className="p-4 space-y-4">
            {!analysis || analysis.anomalies.length === 0 ? (
              <div className="text-center py-8">
                <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-3" />
                <p className="text-slate-300 font-medium">
                  {hasData ? 'No Anomalies Detected' : 'No Data to Analyze'}
                </p>
                <p className="text-slate-500 text-sm">
                  {hasData
                    ? 'System is operating within normal parameters'
                    : 'Upload data and run analysis to detect anomalies'}
                </p>
              </div>
            ) : (
              analysis.anomalies.map((anomaly) => (
                <div
                  key={anomaly.id}
                  className={clsx(
                    'p-4 rounded-lg border-l-4 cursor-pointer transition-colors',
                    getSeverityColor(anomaly.severity),
                    selectedAnomaly === anomaly.id ? 'ring-2 ring-primary-500' : ''
                  )}
                  onClick={() => setSelectedAnomaly(
                    selectedAnomaly === anomaly.id ? null : anomaly.id
                  )}
                >
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-medium text-white">{anomaly.title}</h3>
                    <span className="text-sm font-medium text-slate-400">
                      Impact: {anomaly.impact_score.toFixed(1)}
                    </span>
                  </div>
                  <p className="text-sm text-slate-300 mb-3">{anomaly.description}</p>

                  {selectedAnomaly === anomaly.id && (
                    <div className="mt-4 pt-4 border-t border-slate-600 space-y-4">
                      {/* Confidence indicator */}
                      {anomaly.confidence && (
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-500">Confidence:</span>
                          <div className="flex-1 bg-slate-700 rounded-full h-2 max-w-[100px]">
                            <div
                              className="bg-primary-500 h-2 rounded-full"
                              style={{ width: `${anomaly.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-slate-400">{(anomaly.confidence * 100).toFixed(0)}%</span>
                        </div>
                      )}

                      {/* AI Explanation */}
                      <div>
                        <h4 className="text-sm font-medium text-primary-400 mb-2 flex items-center gap-2">
                          <Lightbulb className="w-4 h-4" />
                          AI Analysis
                        </h4>
                        <p className="text-sm text-slate-300 leading-relaxed">{anomaly.natural_language_explanation}</p>
                      </div>

                      {/* Possible Causes */}
                      {anomaly.possible_causes && anomaly.possible_causes.length > 0 && (
                        <div>
                          <h4 className="text-sm font-medium text-orange-400 mb-2 flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4" />
                            Possible Causes
                          </h4>
                          <ul className="space-y-1">
                            {anomaly.possible_causes.map((cause, idx) => (
                              <li key={idx} className="flex items-start gap-2 text-sm">
                                <span className="text-orange-400 mt-1">•</span>
                                <span className="text-slate-300">{cause}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Recommendations */}
                      {anomaly.recommendations && anomaly.recommendations.length > 0 && (
                        <div>
                          <h4 className="text-sm font-medium text-green-400 mb-2 flex items-center gap-2">
                            <CheckCircle className="w-4 h-4" />
                            Recommended Actions
                          </h4>
                          {anomaly.recommendations.map((rec, idx) => (
                            <div key={idx} className="flex items-start gap-2 text-sm mb-2">
                              <span className={clsx(
                                'px-1.5 py-0.5 rounded text-xs font-medium',
                                rec.priority === 'immediate' || rec.priority === 'high' ? 'bg-red-500/20 text-red-400' :
                                rec.priority === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                'bg-slate-500/20 text-slate-400'
                              )}>
                                {rec.priority}
                              </span>
                              <span className="text-slate-300">{rec.action}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Contributing Agents */}
                      {anomaly.contributing_agents && anomaly.contributing_agents.length > 0 && (
                        <div>
                          <h4 className="text-sm font-medium text-purple-400 mb-2 flex items-center gap-2">
                            <Users className="w-4 h-4" />
                            Contributing AI Agents
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {anomaly.contributing_agents.map((agent, idx) => (
                              <span
                                key={idx}
                                className="px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded-full text-xs text-purple-300"
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
                          <h4 className="text-sm font-medium text-blue-400 mb-2 flex items-center gap-2">
                            <Eye className="w-4 h-4" />
                            Agent Perspectives
                          </h4>
                          <div className="space-y-2">
                            {anomaly.agent_perspectives.map((p, idx) => (
                              <div key={idx} className="p-2 bg-slate-700/50 rounded-lg">
                                <span className="text-xs font-medium text-blue-300">{p.agent}:</span>
                                <p className="text-xs text-slate-400 mt-1">{p.perspective}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Web References */}
                      {anomaly.web_references && anomaly.web_references.length > 0 && (
                        <div>
                          <h4 className="text-sm font-medium text-cyan-400 mb-2 flex items-center gap-2">
                            <Globe className="w-4 h-4" />
                            Web References
                          </h4>
                          <ul className="space-y-1">
                            {anomaly.web_references.map((ref, idx) => (
                              <li key={idx} className="text-xs">
                                <a
                                  href={ref}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-cyan-400 hover:text-cyan-300 underline truncate block"
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
                </div>
              ))
            )}
          </div>
        </div>

        {/* Engineering Margins */}
        <div className="bg-slate-800 rounded-xl border border-slate-700">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary-500" />
            <h2 className="text-lg font-semibold text-white">Engineering Margins</h2>
          </div>
          <div className="p-4 space-y-4">
            {!analysis || analysis.engineering_margins.length === 0 ? (
              <div className="text-center py-8">
                <Activity className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                <p className="text-slate-300 font-medium">
                  {hasData ? 'No Margins Calculated' : 'No Data Available'}
                </p>
                <p className="text-slate-500 text-sm">
                  {hasData
                    ? 'Run analysis to calculate engineering margins'
                    : 'Upload numeric data to see engineering margins'}
                </p>
              </div>
            ) : (
              analysis.engineering_margins.map((margin, idx) => (
                <div key={idx} className="p-4 bg-slate-900/50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <h3 className="font-medium text-white">{margin.component}</h3>
                      <p className="text-sm text-slate-400">{margin.parameter}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      {getTrendIcon(margin.trend)}
                      {margin.safety_critical && (
                        <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded">
                          Safety Critical
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-400">Current: {margin.current_value.toFixed(2)}</span>
                      <span className="text-slate-400">Limit: {margin.design_limit.toFixed(2)}</span>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={clsx(
                          'h-full rounded-full',
                          margin.margin_percentage < 15 ? 'bg-red-500' :
                          margin.margin_percentage < 30 ? 'bg-yellow-500' : 'bg-green-500'
                        )}
                        style={{ width: getProgressWidth(margin.margin_percentage) }}
                      />
                    </div>
                    <p className="text-sm text-slate-400 mt-1">
                      {margin.margin_percentage.toFixed(1)}% margin remaining
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Blind Spots */}
        <div className="col-span-2 bg-slate-800 rounded-xl border border-slate-700">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-500" />
            <h2 className="text-lg font-semibold text-white">Blind Spots & Recommendations</h2>
          </div>
          <div className="p-4">
            {!analysis || analysis.blind_spots.length === 0 ? (
              <div className="text-center py-8">
                <Lightbulb className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                <p className="text-slate-300 font-medium">No Blind Spots Identified</p>
                <p className="text-slate-500 text-sm">
                  Run analysis with uploaded data to identify data gaps and improvement opportunities
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                {analysis.blind_spots.map((spot, idx) => (
                  <div key={idx} className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                    <h3 className="font-medium text-white mb-2">{spot.title}</h3>
                    <p className="text-sm text-slate-300 mb-4">{spot.description}</p>
                    {spot.recommended_sensor && (
                      <div className="p-3 bg-slate-800 rounded-lg">
                        <h4 className="text-sm font-medium text-primary-400 mb-2">Recommended Sensor</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <span className="text-slate-400">Type:</span>
                          <span className="text-white">{spot.recommended_sensor.type}</span>
                          <span className="text-slate-400">Spec:</span>
                          <span className="text-white">{spot.recommended_sensor.specification}</span>
                          <span className="text-slate-400">Cost:</span>
                          <span className="text-white">${spot.recommended_sensor.estimated_cost}</span>
                        </div>
                      </div>
                    )}
                    <div className="mt-3 text-sm">
                      <span className="text-slate-400">Coverage Gain: </span>
                      <span className="text-green-400">+{spot.diagnostic_coverage_improvement}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Field Statistics */}
        {hasData && statistics && statistics.fields.length > 0 && (
          <div className="col-span-2 bg-slate-800 rounded-xl border border-slate-700">
            <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
              <Database className="w-5 h-5 text-primary-500" />
              <h2 className="text-lg font-semibold text-white">Field Statistics</h2>
            </div>
            <div className="p-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-slate-700">
                    <th className="pb-3 pr-4">Field Name</th>
                    <th className="pb-3 pr-4">Type</th>
                    <th className="pb-3 pr-4">Unique Values</th>
                    <th className="pb-3 pr-4">Min</th>
                    <th className="pb-3 pr-4">Max</th>
                    <th className="pb-3 pr-4">Mean</th>
                    <th className="pb-3">Std Dev</th>
                  </tr>
                </thead>
                <tbody>
                  {statistics.fields.map((field, idx) => (
                    <tr key={idx} className="border-b border-slate-700/50 text-slate-300">
                      <td className="py-3 pr-4 font-mono text-white">{field.name}</td>
                      <td className="py-3 pr-4">{field.type}</td>
                      <td className="py-3 pr-4">{field.unique_count}</td>
                      <td className="py-3 pr-4">{field.min !== undefined ? field.min.toFixed(2) : '-'}</td>
                      <td className="py-3 pr-4">{field.max !== undefined ? field.max.toFixed(2) : '-'}</td>
                      <td className="py-3 pr-4">{field.mean !== undefined ? field.mean.toFixed(2) : '-'}</td>
                      <td className="py-3">{field.std !== undefined ? field.std.toFixed(2) : '-'}</td>
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
