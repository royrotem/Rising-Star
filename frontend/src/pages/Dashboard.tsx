import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Plus,
  ChevronRight,
  Loader2,
  ArrowRight,
  Timer,
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi, schedulesApi } from '../services/api';
import type { System, Schedule } from '../types';
import OnboardingGuide from '../components/OnboardingGuide';
import { getSeverityBadgeColor, getStatusColor, getHealthColor } from '../utils/colors';

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

export default function Dashboard() {
  const navigate = useNavigate();
  const [systems, setSystems] = useState<System[]>([]);
  const [schedules, setSchedules] = useState<Record<string, Schedule>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSystems();
  }, []);

  const loadSystems = async () => {
    try {
      const data = await systemsApi.list();
      setSystems(data);
      try {
        const scheds = await schedulesApi.list();
        const map: Record<string, Schedule> = {};
        for (const s of scheds) {
          if (s.enabled) map[s.system_id] = s;
        }
        setSchedules(map);
      } catch { /* schedules not critical */ }
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

  const totalAnomalies = systems.reduce((sum, s) => sum + (s.anomaly_count || 0), 0);
  const anomalyCount = totalAnomalies || systems.filter(s => s.status === 'anomaly_detected').length;

  return (
    <div className="p-8 page-enter">
      {/* Header */}
      <div className="flex items-center justify-between mb-10">
        <div>
          <h1 className="text-xl font-semibold text-white">Dashboard</h1>
          <p className="text-stone-500 mt-1 text-sm">
            Fleet overview and critical issues
          </p>
        </div>
        <button
          onClick={() => navigate('/systems/new')}
          className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white text-sm rounded-lg font-medium transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add System
        </button>
      </div>

      {/* Guided Onboarding */}
      {!loading && <OnboardingGuide systemCount={systems.length} />}

      {/* Stats Grid */}
      <div className="grid grid-cols-4 gap-4 mb-10">
        <div className="stat-card">
          <p className="section-header mb-3">Active Systems</p>
          {loading ? (
            <Loader2 className="w-5 h-5 text-stone-600 animate-spin" />
          ) : (
            <p className="text-2xl font-semibold text-white tabular-nums">{systems.length}</p>
          )}
        </div>

        <div className="stat-card">
          <p className="section-header mb-3">Active Anomalies</p>
          {loading ? (
            <Loader2 className="w-5 h-5 text-stone-600 animate-spin" />
          ) : (
            <p className={clsx(
              'text-2xl font-semibold tabular-nums',
              anomalyCount > 0 ? 'text-orange-400' : 'text-white'
            )}>{anomalyCount}</p>
          )}
        </div>

        <div className="stat-card">
          <p className="section-header mb-3">Avg Health</p>
          {loading ? (
            <Loader2 className="w-5 h-5 text-stone-600 animate-spin" />
          ) : (
            <p className="text-2xl font-semibold text-white tabular-nums">{avgHealthScore}%</p>
          )}
        </div>

        <div className="stat-card">
          <p className="section-header mb-3">Degrading Margins</p>
          <p className={clsx(
            'text-2xl font-semibold tabular-nums',
            'text-red-400'
          )}>2</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* 80/20 Impact Radar */}
        <div className="col-span-2 glass-card">
          <div className="px-6 py-4 border-b border-stone-800">
            <h2 className="text-sm font-medium text-white">Impact Radar</h2>
            <p className="text-xs text-stone-600 mt-0.5">
              Top issues by impact score
            </p>
          </div>
          <div className="p-4">
            <div className="space-y-2">
              {mockImpactRadar.prioritized_issues.map((issue, idx) => (
                <Link
                  key={issue.rank}
                  to={`/systems/1`}
                  className="flex items-center gap-4 p-3.5 rounded-lg border border-transparent hover:border-stone-800 hover:bg-stone-800/40 transition-all duration-150 group"
                  style={{ animationDelay: `${idx * 80}ms` }}
                >
                  <span className="text-xs font-mono text-stone-600 w-5 text-center">
                    {issue.rank}
                  </span>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm text-stone-200 group-hover:text-white transition-colors truncate">{issue.title}</h3>
                    <p className="text-xs text-stone-600 mt-0.5">
                      {issue.affected_percentage}% of fleet affected
                    </p>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <span className={clsx(
                      'inline-flex px-2 py-0.5 rounded text-xs font-medium',
                      getSeverityBadgeColor(issue.severity)
                    )}>
                      {issue.impact_score}
                    </span>
                    <ChevronRight className="w-3.5 h-3.5 text-stone-700 group-hover:text-stone-500 transition-colors" />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Systems List */}
        <div className="glass-card">
          <div className="px-5 py-4 border-b border-stone-800 flex items-center justify-between">
            <h2 className="text-sm font-medium text-white">Systems</h2>
            <Link
              to="/systems"
              className="text-xs text-stone-500 hover:text-stone-300 flex items-center gap-1 transition-colors"
            >
              View all
              <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
          <div className="p-2">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-5 h-5 text-stone-600 animate-spin" />
              </div>
            ) : systems.length === 0 ? (
              <div className="text-center py-10">
                <p className="text-stone-500 text-sm">No systems yet</p>
                <Link
                  to="/systems/new"
                  className="text-primary-400 text-xs hover:text-primary-300 mt-1.5 inline-block"
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
                    className="flex items-center justify-between p-2.5 rounded-lg hover:bg-stone-800/50 transition-colors duration-150 group"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className={clsx(
                        'w-1.5 h-1.5 rounded-full flex-shrink-0',
                        getStatusColor(system.status)
                      )} />
                      <div className="min-w-0">
                        <p className="text-sm text-stone-300 group-hover:text-white transition-colors truncate">
                          {system.name}
                        </p>
                        <p className="text-[11px] text-stone-600 capitalize">{system.system_type.replace('_', ' ')}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {schedules[system.id] && (
                        <span title="Watchdog active">
                          <Timer className="w-3 h-3 text-stone-500" />
                        </span>
                      )}
                      <span className={clsx(
                        'text-xs font-medium tabular-nums',
                        getHealthColor(system.health_score)
                      )}>
                        {system.health_score}%
                      </span>
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
