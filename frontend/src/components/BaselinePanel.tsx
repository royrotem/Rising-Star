import { useState, useEffect, useCallback } from 'react';
import {
  History,
  Camera,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Loader2,
  ChevronDown,
  ChevronUp,
  BarChart3,
  Minus,
} from 'lucide-react';
import clsx from 'clsx';
import { baselineApi, Snapshot, CompareResult, Deviation } from '../services/baselineApi';

interface BaselinePanelProps {
  systemId: string;
}

function DeviationBadge({ status }: { status: Deviation['status'] }) {
  switch (status) {
    case 'critical_deviation':
      return (
        <span className="px-2 py-0.5 bg-red-500/20 text-red-400 text-xs font-medium rounded-full">
          Critical
        </span>
      );
    case 'significant_deviation':
      return (
        <span className="px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs font-medium rounded-full">
          Significant
        </span>
      );
    case 'minor_deviation':
      return (
        <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs font-medium rounded-full">
          Minor
        </span>
      );
    default:
      return (
        <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs font-medium rounded-full">
          Normal
        </span>
      );
  }
}

export default function BaselinePanel({ systemId }: BaselinePanelProps) {
  const [history, setHistory] = useState<Snapshot[]>([]);
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const histRes = await baselineApi.getHistory(systemId, 20);
      setHistory(histRes.snapshots);
    } catch {
      // No history yet — that's fine
    } finally {
      setLoading(false);
    }
  }, [systemId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCapture = async () => {
    setCapturing(true);
    setError(null);
    try {
      await baselineApi.captureSnapshot(systemId);
      await loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to capture snapshot');
    } finally {
      setCapturing(false);
    }
  };

  const handleCompare = async () => {
    setComparing(true);
    setError(null);
    try {
      const result = await baselineApi.compare(systemId);
      setComparison(result);
      setExpanded(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to compare');
    } finally {
      setComparing(false);
    }
  };

  const hasHistory = history.length > 0;

  return (
    <div className="bg-stone-700 rounded-xl border border-stone-600 mb-6">
      {/* Header */}
      <div className="px-6 py-4 border-b border-stone-600 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History className="w-5 h-5 text-primary-400" />
          <h2 className="text-lg font-semibold text-white">Baseline & Historical</h2>
          {hasHistory && (
            <span className="text-xs text-stone-400 ml-2">
              {history.length} snapshot{history.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCapture}
            disabled={capturing}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-primary-500/10 hover:bg-primary-500/20 text-primary-400 text-sm rounded-lg transition-colors disabled:opacity-50"
          >
            {capturing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Camera className="w-3.5 h-3.5" />}
            Capture Snapshot
          </button>
          {hasHistory && (
            <button
              onClick={handleCompare}
              disabled={comparing}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-stone-700 hover:bg-stone-500 text-stone-300 text-sm rounded-lg transition-colors disabled:opacity-50"
            >
              {comparing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <BarChart3 className="w-3.5 h-3.5" />}
              Compare to Baseline
            </button>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="px-6 py-3 bg-red-500/10 border-b border-red-500/20">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="px-6 py-8 flex items-center justify-center">
          <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
        </div>
      )}

      {/* No history state */}
      {!loading && !hasHistory && (
        <div className="px-6 py-8 text-center">
          <History className="w-10 h-10 text-stone-400 mx-auto mb-3" />
          <p className="text-stone-400 text-sm mb-1">No snapshots captured yet</p>
          <p className="text-stone-400 text-xs">
            Capture snapshots periodically to track changes over time and detect deviations from baseline
          </p>
        </div>
      )}

      {/* Health Score Trend (mini sparkline) */}
      {hasHistory && !loading && (
        <div className="px-6 py-4">
          <div className="flex items-center gap-4 mb-3">
            <div>
              <p className="text-xs text-stone-400">Health Score Trend</p>
              <div className="flex items-center gap-2 mt-1">
                {history.slice(0, 10).reverse().map((snap, i) => {
                  const score = snap.health_score;
                  if (score == null) return <Minus key={i} className="w-3 h-3 text-stone-400" />;
                  const color = score >= 90 ? 'bg-green-500' : score >= 70 ? 'bg-yellow-500' : 'bg-red-500';
                  const height = Math.max(8, (score / 100) * 28);
                  return (
                    <div
                      key={i}
                      className={clsx('w-4 rounded-sm', color)}
                      style={{ height: `${height}px` }}
                      title={`${snap.timestamp.split('T')[0]}: ${score.toFixed(0)}%`}
                    />
                  );
                })}
              </div>
            </div>
            <div className="ml-auto text-right">
              <p className="text-xs text-stone-400">Anomaly Count Trend</p>
              <div className="flex items-center gap-2 mt-1">
                {history.slice(0, 10).reverse().map((snap, i) => {
                  const count = snap.anomaly_count;
                  const color = count === 0 ? 'bg-green-500' : count <= 3 ? 'bg-yellow-500' : 'bg-red-500';
                  const height = Math.max(8, Math.min(28, count * 6));
                  return (
                    <div
                      key={i}
                      className={clsx('w-4 rounded-sm', color)}
                      style={{ height: `${height}px` }}
                      title={`${snap.timestamp.split('T')[0]}: ${count} anomalies`}
                    />
                  );
                })}
              </div>
            </div>
          </div>

          {/* Toggle history details */}
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="flex items-center gap-1 text-xs text-stone-400 hover:text-stone-300 transition-colors"
          >
            {showHistory ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            {showHistory ? 'Hide' : 'Show'} snapshot history
          </button>

          {/* Snapshot History Table */}
          {showHistory && (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-stone-400 border-b border-stone-600">
                    <th className="pb-2 text-left">Date</th>
                    <th className="pb-2 text-right">Health</th>
                    <th className="pb-2 text-right">Records</th>
                    <th className="pb-2 text-right">Fields</th>
                    <th className="pb-2 text-right">Anomalies</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 15).map((snap) => (
                    <tr key={snap.id} className="border-b border-stone-600/50 text-stone-300">
                      <td className="py-1.5">{new Date(snap.timestamp).toLocaleString()}</td>
                      <td className="py-1.5 text-right">
                        {snap.health_score != null ? (
                          <span className={clsx(
                            snap.health_score >= 90 ? 'text-green-400' :
                            snap.health_score >= 70 ? 'text-yellow-400' : 'text-red-400'
                          )}>
                            {snap.health_score.toFixed(0)}%
                          </span>
                        ) : '—'}
                      </td>
                      <td className="py-1.5 text-right">{snap.record_count.toLocaleString()}</td>
                      <td className="py-1.5 text-right">{snap.field_count}</td>
                      <td className="py-1.5 text-right">
                        <span className={clsx(
                          snap.anomaly_count === 0 ? 'text-green-400' :
                          snap.anomaly_count <= 3 ? 'text-yellow-400' : 'text-red-400'
                        )}>
                          {snap.anomaly_count}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Comparison Results */}
      {comparison && expanded && (
        <div className="px-6 py-4 border-t border-stone-600">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-primary-400" />
              Baseline Comparison
            </h3>
            <button
              onClick={() => setExpanded(false)}
              className="text-xs text-stone-400 hover:text-stone-300"
            >
              Hide
            </button>
          </div>

          {comparison.status === 'no_baseline' || comparison.status === 'no_data' ? (
            <p className="text-sm text-stone-400">{comparison.message}</p>
          ) : (
            <>
              {/* Summary badges */}
              <div className="flex items-center gap-3 mb-4">
                {comparison.critical_deviations ? (
                  <div className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/10 border border-red-500/20 rounded-lg">
                    <AlertTriangle className="w-4 h-4 text-red-400" />
                    <span className="text-sm text-red-400 font-medium">
                      {comparison.critical_deviations} critical
                    </span>
                  </div>
                ) : null}
                {comparison.significant_deviations ? (
                  <div className="flex items-center gap-1.5 px-3 py-1.5 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                    <TrendingUp className="w-4 h-4 text-orange-400" />
                    <span className="text-sm text-orange-400 font-medium">
                      {comparison.significant_deviations} significant
                    </span>
                  </div>
                ) : null}
                {!comparison.critical_deviations && !comparison.significant_deviations && (
                  <div className="flex items-center gap-1.5 px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-lg">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-green-400 font-medium">All within baseline</span>
                  </div>
                )}
                <span className="text-xs text-stone-400">
                  Based on {comparison.snapshot_count} snapshots
                </span>
              </div>

              {/* Deviation table */}
              {comparison.deviations && comparison.deviations.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-stone-400 border-b border-stone-600">
                        <th className="pb-2 text-left">Field</th>
                        <th className="pb-2 text-right">Current</th>
                        <th className="pb-2 text-right">Baseline</th>
                        <th className="pb-2 text-right">Change</th>
                        <th className="pb-2 text-right">Z-Score</th>
                        <th className="pb-2 text-center">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.deviations.slice(0, 15).map((dev) => (
                        <tr key={dev.field} className="border-b border-stone-600/50">
                          <td className="py-1.5 text-stone-200 font-mono">{dev.field}</td>
                          <td className="py-1.5 text-right text-stone-300">{dev.current_mean}</td>
                          <td className="py-1.5 text-right text-stone-400">{dev.baseline_mean}</td>
                          <td className="py-1.5 text-right">
                            <span className={clsx(
                              'flex items-center justify-end gap-1',
                              dev.pct_change > 5 ? 'text-red-400' :
                              dev.pct_change < -5 ? 'text-blue-400' : 'text-stone-400'
                            )}>
                              {dev.pct_change > 0 ? (
                                <TrendingUp className="w-3 h-3" />
                              ) : dev.pct_change < 0 ? (
                                <TrendingDown className="w-3 h-3" />
                              ) : null}
                              {dev.pct_change > 0 ? '+' : ''}{dev.pct_change}%
                            </span>
                          </td>
                          <td className="py-1.5 text-right text-stone-300">{dev.z_deviation}</td>
                          <td className="py-1.5 text-center">
                            <DeviationBadge status={dev.status} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
