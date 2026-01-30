import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Plus,
  Activity,
  AlertTriangle,
  TrendingDown,
  Server,
  ChevronRight,
  Target,
  Loader2,
  ArrowRight,
  Sparkles,
  Shield,
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';
import type { System } from '../types';
import OnboardingGuide from '../components/OnboardingGuide';

// Mock impact radar data
const mockImpactRadar = {
  prioritized_issues: [
    {
      rank: 1,
      title: 'Motor A Current Deviation',
      impact_score: 72.5,
      affected_percentage: 34,
      severity: 'high',
    },
    {
      rank: 2,
      title: 'Battery Thermal Margin',
      impact_score: 65.0,
      affected_percentage: 28,
      severity: 'medium',
    },
    {
      rank: 3,
      title: 'Communication Latency',
      impact_score: 48.0,
      affected_percentage: 16,
      severity: 'low',
    },
  ],
};

function getSeverityColor(severity: string) {
  switch (severity) {
    case 'critical': return 'text-red-400 bg-red-500/10 border border-red-500/20';
    case 'high': return 'text-orange-400 bg-orange-500/10 border border-orange-500/20';
    case 'medium': return 'text-yellow-400 bg-yellow-500/10 border border-yellow-500/20';
    case 'low': return 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20';
    default: return 'text-slate-400 bg-slate-500/10 border border-slate-500/20';
  }
}

function getStatusColor(status: string) {
  switch (status) {
    case 'active': return 'bg-emerald-400';
    case 'anomaly_detected': return 'bg-orange-400';
    case 'maintenance': return 'bg-yellow-400';
    case 'inactive': return 'bg-slate-500';
    default: return 'bg-slate-500';
  }
}

function getHealthColor(score: number) {
  if (score >= 90) return 'text-emerald-400';
  if (score >= 70) return 'text-yellow-400';
  return 'text-red-400';
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [systems, setSystems] = useState<System[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSystems();
  }, []);

  const loadSystems = async () => {
    try {
      const data = await systemsApi.list();
      setSystems(data);
    } catch (error) {
      console.error('Failed to load systems:', error);
      setSystems([
        {
          id: '1',
          name: 'Fleet Vehicle Alpha',
          system_type: 'vehicle',
          status: 'anomaly_detected',
          health_score: 87.5,
          created_at: '2024-01-01T00:00:00Z',
        },
        {
          id: '2',
          name: 'Robot Arm Unit 7',
          system_type: 'robot',
          status: 'active',
          health_score: 94.2,
          created_at: '2024-01-02T00:00:00Z',
        },
        {
          id: '3',
          name: 'Medical Scanner MRI-3',
          system_type: 'medical_device',
          status: 'active',
          health_score: 99.1,
          created_at: '2024-01-03T00:00:00Z',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const avgHealthScore = systems.length > 0
    ? (systems.reduce((sum, s) => sum + s.health_score, 0) / systems.length).toFixed(1)
    : '0';

  const anomalyCount = systems.filter(s => s.status === 'anomaly_detected').length;

  return (
    <div className="p-8 page-enter">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Dashboard</h1>
          <p className="text-slate-400 mt-1 text-sm">
            Monitor your fleet and track critical issues
          </p>
        </div>
        <button
          onClick={() => navigate('/systems/new')}
          className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white rounded-xl font-medium transition-all duration-200 shadow-lg shadow-primary-500/20 hover:shadow-primary-500/30"
        >
          <Plus className="w-4 h-4" />
          Add System
        </button>
      </div>

      {/* Guided Onboarding */}
      {!loading && <OnboardingGuide systemCount={systems.length} />}

      {/* Stats Grid */}
      <div className="grid grid-cols-4 gap-5 mb-8">
        <div className="stat-card" style={{ '--stat-accent': '#6366f1' } as React.CSSProperties}>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-primary-500/10 rounded-xl">
              <Server className="w-5 h-5 text-primary-400" />
            </div>
            <div>
              {loading ? (
                <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
              ) : (
                <p className="text-2xl font-bold text-white">{systems.length}</p>
              )}
              <p className="text-xs text-slate-500 font-medium">Active Systems</p>
            </div>
          </div>
        </div>

        <div className="stat-card" style={{ '--stat-accent': '#f97316' } as React.CSSProperties}>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-orange-500/10 rounded-xl">
              <AlertTriangle className="w-5 h-5 text-orange-400" />
            </div>
            <div>
              {loading ? (
                <Loader2 className="w-5 h-5 text-orange-400 animate-spin" />
              ) : (
                <p className="text-2xl font-bold text-white">{anomalyCount}</p>
              )}
              <p className="text-xs text-slate-500 font-medium">Active Anomalies</p>
            </div>
          </div>
        </div>

        <div className="stat-card" style={{ '--stat-accent': '#10b981' } as React.CSSProperties}>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-emerald-500/10 rounded-xl">
              <Activity className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              {loading ? (
                <Loader2 className="w-5 h-5 text-emerald-400 animate-spin" />
              ) : (
                <p className="text-2xl font-bold text-white">{avgHealthScore}%</p>
              )}
              <p className="text-xs text-slate-500 font-medium">Avg Health Score</p>
            </div>
          </div>
        </div>

        <div className="stat-card" style={{ '--stat-accent': '#ef4444' } as React.CSSProperties}>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-red-500/10 rounded-xl">
              <TrendingDown className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">2</p>
              <p className="text-xs text-slate-500 font-medium">Margins Degrading</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* 80/20 Impact Radar */}
        <div className="col-span-2 glass-card overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-700/50">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-primary-400" />
              <h2 className="text-base font-semibold text-white">80/20 Impact Radar</h2>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Focus on the 20% of issues causing 80% of impact
            </p>
          </div>
          <div className="p-5">
            <div className="space-y-3">
              {mockImpactRadar.prioritized_issues.map((issue, idx) => (
                <Link
                  key={issue.rank}
                  to={`/systems/1`}
                  className="flex items-center gap-4 p-4 bg-slate-900/40 rounded-xl border border-slate-700/40 hover:border-primary-500/30 transition-all duration-200 group"
                  style={{ animationDelay: `${idx * 100}ms` }}
                >
                  <div className="flex items-center justify-center w-9 h-9 bg-slate-800 rounded-lg text-sm font-bold text-slate-400 group-hover:bg-primary-500/10 group-hover:text-primary-400 transition-all duration-200">
                    {issue.rank}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-sm text-white group-hover:text-primary-300 transition-colors truncate">{issue.title}</h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Affecting {issue.affected_percentage}% of fleet
                    </p>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <div className={clsx(
                      'inline-flex px-2.5 py-1 rounded-lg text-xs font-medium',
                      getSeverityColor(issue.severity)
                    )}>
                      {issue.impact_score}
                    </div>
                    <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-primary-400 transition-colors" />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Systems List */}
        <div className="glass-card overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-700/50 flex items-center justify-between">
            <h2 className="text-base font-semibold text-white">Systems</h2>
            <Link
              to="/systems"
              className="text-xs text-primary-400 hover:text-primary-300 flex items-center gap-1 font-medium transition-colors"
            >
              View all
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
          <div className="p-3">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
              </div>
            ) : systems.length === 0 ? (
              <div className="text-center py-10">
                <div className="p-3 bg-slate-800/50 rounded-xl inline-block mb-3">
                  <Server className="w-7 h-7 text-slate-600" />
                </div>
                <p className="text-slate-400 text-sm font-medium">No systems yet</p>
                <Link
                  to="/systems/new"
                  className="text-primary-400 text-xs hover:text-primary-300 mt-1 inline-block"
                >
                  Add your first system
                </Link>
              </div>
            ) : (
              <div className="space-y-0.5">
                {systems.slice(0, 5).map((system) => (
                  <Link
                    key={system.id}
                    to={`/systems/${system.id}`}
                    className="flex items-center justify-between p-3 rounded-lg hover:bg-slate-800/60 transition-all duration-200 group"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <div className={clsx(
                        'w-2 h-2 rounded-full flex-shrink-0',
                        getStatusColor(system.status)
                      )} />
                      <div className="min-w-0">
                        <p className="font-medium text-sm text-white group-hover:text-primary-300 transition-colors truncate">
                          {system.name}
                        </p>
                        <p className="text-xs text-slate-500 capitalize">{system.system_type.replace('_', ' ')}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className={clsx(
                        'text-sm font-semibold tabular-nums',
                        getHealthColor(system.health_score)
                      )}>
                        {system.health_score}%
                      </span>
                      <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-slate-400" />
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Info Banner */}
      <div className="mt-6 glass-card p-5 flex items-center gap-4">
        <div className="p-2.5 bg-primary-500/10 rounded-xl flex-shrink-0">
          <Sparkles className="w-5 h-5 text-primary-400" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-white">AI-Powered Analysis</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            5 specialized agents analyze your data across statistical, domain, pattern, root-cause, and safety dimensions
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <Shield className="w-4 h-4 text-accent-400" />
          <span className="text-xs text-accent-400 font-medium">Active</span>
        </div>
      </div>
    </div>
  );
}
