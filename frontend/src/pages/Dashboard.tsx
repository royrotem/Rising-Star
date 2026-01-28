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
  ArrowRight
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';
import type { System } from '../types';

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
    case 'critical': return 'text-red-500 bg-red-500/10';
    case 'high': return 'text-orange-500 bg-orange-500/10';
    case 'medium': return 'text-yellow-500 bg-yellow-500/10';
    case 'low': return 'text-green-500 bg-green-500/10';
    default: return 'text-slate-500 bg-slate-500/10';
  }
}

function getStatusColor(status: string) {
  switch (status) {
    case 'active': return 'text-green-500';
    case 'anomaly_detected': return 'text-orange-500';
    case 'maintenance': return 'text-yellow-500';
    case 'inactive': return 'text-slate-500';
    default: return 'text-slate-500';
  }
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
      // Use mock data if API fails
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
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-slate-400 mt-1">
            Monitor your fleet and track critical issues
          </p>
        </div>
        <button
          onClick={() => navigate('/systems')}
          className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors"
        >
          <Plus className="w-5 h-5" />
          Add System
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-primary-500/10 rounded-lg">
              <Server className="w-6 h-6 text-primary-500" />
            </div>
            <div>
              {loading ? (
                <Loader2 className="w-6 h-6 text-primary-500 animate-spin" />
              ) : (
                <p className="text-2xl font-bold text-white">{systems.length}</p>
              )}
              <p className="text-sm text-slate-400">Active Systems</p>
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-orange-500/10 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-orange-500" />
            </div>
            <div>
              {loading ? (
                <Loader2 className="w-6 h-6 text-orange-500 animate-spin" />
              ) : (
                <p className="text-2xl font-bold text-white">{anomalyCount}</p>
              )}
              <p className="text-sm text-slate-400">Active Anomalies</p>
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-green-500/10 rounded-lg">
              <Activity className="w-6 h-6 text-green-500" />
            </div>
            <div>
              {loading ? (
                <Loader2 className="w-6 h-6 text-green-500 animate-spin" />
              ) : (
                <p className="text-2xl font-bold text-white">{avgHealthScore}%</p>
              )}
              <p className="text-sm text-slate-400">Avg Health Score</p>
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-red-500/10 rounded-lg">
              <TrendingDown className="w-6 h-6 text-red-500" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">2</p>
              <p className="text-sm text-slate-400">Margins Degrading</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* 80/20 Impact Radar */}
        <div className="col-span-2 bg-slate-800 rounded-xl border border-slate-700">
          <div className="px-6 py-4 border-b border-slate-700">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-primary-500" />
              <h2 className="text-lg font-semibold text-white">80/20 Impact Radar</h2>
            </div>
            <p className="text-sm text-slate-400 mt-1">
              Focus on the 20% of issues causing 80% of impact
            </p>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {mockImpactRadar.prioritized_issues.map((issue) => (
                <Link
                  key={issue.rank}
                  to={`/systems/1`}
                  className="flex items-center gap-4 p-4 bg-slate-900/50 rounded-lg border border-slate-700 hover:border-primary-500/50 transition-colors group"
                >
                  <div className="flex items-center justify-center w-10 h-10 bg-slate-700 rounded-full font-bold text-white group-hover:bg-primary-500/20 transition-colors">
                    #{issue.rank}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium text-white group-hover:text-primary-400 transition-colors">{issue.title}</h3>
                    <p className="text-sm text-slate-400">
                      Affecting {issue.affected_percentage}% of fleet
                    </p>
                  </div>
                  <div className="text-right flex items-center gap-3">
                    <div className={clsx(
                      'inline-flex px-3 py-1 rounded-full text-sm font-medium',
                      getSeverityColor(issue.severity)
                    )}>
                      Impact: {issue.impact_score}
                    </div>
                    <ChevronRight className="w-5 h-5 text-slate-500 group-hover:text-primary-400 transition-colors" />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Systems List */}
        <div className="bg-slate-800 rounded-xl border border-slate-700">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Systems</h2>
            <Link
              to="/systems"
              className="text-sm text-primary-400 hover:text-primary-300 flex items-center gap-1"
            >
              View all
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="p-4">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 text-primary-500 animate-spin" />
              </div>
            ) : systems.length === 0 ? (
              <div className="text-center py-8">
                <Server className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                <p className="text-slate-400 text-sm">No systems yet</p>
                <Link
                  to="/systems"
                  className="text-primary-400 text-sm hover:text-primary-300"
                >
                  Add your first system
                </Link>
              </div>
            ) : (
              <div className="space-y-2">
                {systems.slice(0, 5).map((system) => (
                  <Link
                    key={system.id}
                    to={`/systems/${system.id}`}
                    className="flex items-center justify-between p-3 rounded-lg hover:bg-slate-700/50 transition-colors group"
                  >
                    <div className="flex items-center gap-3">
                      <div className={clsx(
                        'w-2 h-2 rounded-full',
                        getStatusColor(system.status).replace('text-', 'bg-')
                      )} />
                      <div>
                        <p className="font-medium text-white group-hover:text-primary-400 transition-colors">
                          {system.name}
                        </p>
                        <p className="text-sm text-slate-400 capitalize">{system.system_type.replace('_', ' ')}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={clsx(
                        'text-sm font-medium',
                        system.health_score >= 90 ? 'text-green-500' :
                        system.health_score >= 70 ? 'text-yellow-500' : 'text-red-500'
                      )}>
                        {system.health_score}%
                      </span>
                      <ChevronRight className="w-4 h-4 text-slate-500" />
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
