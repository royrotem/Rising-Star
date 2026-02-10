/**
 * AnomalyExplorer — Interactive anomaly exploration page.
 *
 * Additive feature module. Provides filtering, sorting, severity
 * distribution visualization, and detailed drill-down for anomalies.
 * Removing this file and its route does not affect core functionality.
 */

import { useState, useEffect, useMemo } from 'react';
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
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';
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

export default function AnomalyExplorer() {
  const { systemId } = useParams();
  const [system, setSystem] = useState<System | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
      const result = await systemsApi.analyze(systemId);
      if (result && result.anomalies) {
        setAnomalies(result.anomalies as AnomalyItem[]);
      }
    } catch (e) {
      console.error('Failed to load anomalies:', e);
      setError('Failed to load anomaly data. Run analysis first.');
    } finally {
      setLoading(false);
    }
  };

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
          <p className="text-sm text-stone-400">Loading anomalies...</p>
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
          </p>
        </div>
        <div className="flex items-center gap-2">
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
