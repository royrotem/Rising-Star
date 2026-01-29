import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Plus,
  Search,
  Server,
  Cpu,
  Heart,
  Car,
  Rocket,
  Factory,
  Activity,
  ChevronRight,
  Loader2,
  Eye,
  EyeOff,
  AlertCircle,
  Trash2
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';
import type { System } from '../types';

interface SystemWithDemo extends System {
  is_demo?: boolean;
}

const systemTypeIcons: Record<string, React.ElementType> = {
  vehicle: Car,
  robot: Cpu,
  medical_device: Heart,
  aerospace: Rocket,
  industrial: Factory,
  default: Server,
};

function getStatusColor(status: string) {
  switch (status) {
    case 'active': return 'bg-green-500';
    case 'anomaly_detected': return 'bg-orange-500';
    case 'maintenance': return 'bg-yellow-500';
    case 'inactive': return 'bg-slate-500';
    case 'data_ingested': return 'bg-blue-500';
    case 'configured': return 'bg-green-500';
    default: return 'bg-slate-500';
  }
}

function getHealthColor(score: number | undefined | null) {
  if (score === undefined || score === null) return 'text-slate-500';
  if (score >= 90) return 'text-green-500';
  if (score >= 70) return 'text-yellow-500';
  return 'text-red-500';
}

export default function Systems() {
  const navigate = useNavigate();
  const [systems, setSystems] = useState<SystemWithDemo[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [showDemo, setShowDemo] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSystems();
  }, [showDemo]);

  const loadSystems = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/v1/systems/?include_demo=${showDemo}`);
      if (response.ok) {
        const data = await response.json();
        setSystems(data);
      } else {
        throw new Error('Failed to fetch systems');
      }
    } catch (err) {
      console.error('Failed to load systems:', err);
      setError('Failed to load systems. Make sure the backend is running.');
      setSystems([]);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSystem = async (e: React.MouseEvent, systemId: string) => {
    e.preventDefault();
    e.stopPropagation();

    if (!confirm('Are you sure you want to delete this system?')) return;

    try {
      const response = await fetch(`/api/v1/systems/${systemId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setSystems(systems.filter(s => s.id !== systemId));
      }
    } catch (err) {
      console.error('Failed to delete system:', err);
    }
  };

  const filteredSystems = systems.filter(system =>
    system.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    system.system_type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const demoSystems = filteredSystems.filter(s => s.is_demo);
  const realSystems = filteredSystems.filter(s => !s.is_demo);

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Systems</h1>
          <p className="text-slate-400 mt-1">
            Manage and monitor your connected systems
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowDemo(!showDemo)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
              showDemo
                ? 'bg-slate-700 hover:bg-slate-600 text-white'
                : 'bg-slate-800 hover:bg-slate-700 text-slate-400'
            )}
          >
            {showDemo ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            {showDemo ? 'Hide Demo' : 'Show Demo'}
          </button>
          <button
            onClick={() => navigate('/systems/new')}
            className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors"
          >
            <Plus className="w-5 h-5" />
            Add System
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

      {/* Search */}
      <div className="relative mb-6">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
        <input
          type="text"
          placeholder="Search systems..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors"
        />
      </div>

      {/* Systems Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
        </div>
      ) : filteredSystems.length === 0 ? (
        <div className="text-center py-12">
          <Server className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400 mb-2">No systems found</p>
          <button
            onClick={() => navigate('/systems/new')}
            className="text-primary-400 hover:text-primary-300 font-medium"
          >
            Add your first system
          </button>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Real Systems */}
          {realSystems.length > 0 && (
            <div>
              <h2 className="text-lg font-semibold text-white mb-4">
                Your Systems ({realSystems.length})
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {realSystems.map((system) => {
                  const Icon = systemTypeIcons[system.system_type] || systemTypeIcons.default;
                  return (
                    <Link
                      key={system.id}
                      to={`/systems/${system.id}`}
                      className="bg-slate-800 rounded-xl border border-slate-700 p-6 hover:border-primary-500/50 transition-colors group relative"
                    >
                      <button
                        onClick={(e) => handleDeleteSystem(e, system.id)}
                        className="absolute top-4 right-4 p-2 opacity-0 group-hover:opacity-100 hover:bg-red-500/20 rounded-lg transition-all"
                        title="Delete system"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>

                      <div className="flex items-start justify-between mb-4">
                        <div className="p-3 bg-slate-700 rounded-lg group-hover:bg-primary-500/10 transition-colors">
                          <Icon className="w-6 h-6 text-slate-300 group-hover:text-primary-400 transition-colors" />
                        </div>
                        <div className="flex items-center gap-2">
                          <div className={clsx('w-2 h-2 rounded-full', getStatusColor(system.status))} />
                          <span className="text-sm text-slate-400 capitalize">
                            {system.status.replace('_', ' ')}
                          </span>
                        </div>
                      </div>

                      <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-primary-400 transition-colors">
                        {system.name}
                      </h3>
                      <p className="text-sm text-slate-400 mb-4 capitalize">
                        {system.system_type.replace('_', ' ')}
                      </p>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Activity className="w-4 h-4 text-slate-500" />
                          <span className={clsx('font-medium', getHealthColor(system.health_score))}>
                            {system.health_score !== undefined && system.health_score !== null
                              ? `${system.health_score}% Health`
                              : 'Not analyzed'}
                          </span>
                        </div>
                        <ChevronRight className="w-5 h-5 text-slate-500 group-hover:text-primary-400 transition-colors" />
                      </div>
                    </Link>
                  );
                })}
              </div>
            </div>
          )}

          {/* Demo Systems */}
          {showDemo && demoSystems.length > 0 && (
            <div>
              <h2 className="text-lg font-semibold text-slate-400 mb-4 flex items-center gap-2">
                Demo Systems ({demoSystems.length})
                <span className="text-xs px-2 py-0.5 bg-slate-700 rounded">For testing</span>
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {demoSystems.map((system) => {
                  const Icon = systemTypeIcons[system.system_type] || systemTypeIcons.default;
                  return (
                    <Link
                      key={system.id}
                      to={`/systems/${system.id}`}
                      className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-6 hover:border-slate-600 transition-colors group opacity-75 hover:opacity-100"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="p-3 bg-slate-700/50 rounded-lg group-hover:bg-slate-700 transition-colors">
                          <Icon className="w-6 h-6 text-slate-400 group-hover:text-slate-300 transition-colors" />
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs px-2 py-0.5 bg-slate-700 text-slate-400 rounded">Demo</span>
                          <div className={clsx('w-2 h-2 rounded-full', getStatusColor(system.status))} />
                        </div>
                      </div>

                      <h3 className="text-lg font-semibold text-slate-300 mb-1 group-hover:text-white transition-colors">
                        {system.name}
                      </h3>
                      <p className="text-sm text-slate-500 mb-4 capitalize">
                        {system.system_type.replace('_', ' ')}
                      </p>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Activity className="w-4 h-4 text-slate-600" />
                          <span className={clsx('font-medium', getHealthColor(system.health_score))}>
                            {system.health_score}% Health
                          </span>
                        </div>
                        <ChevronRight className="w-5 h-5 text-slate-600 group-hover:text-slate-400 transition-colors" />
                      </div>
                    </Link>
                  );
                })}
              </div>
            </div>
          )}

          {/* Empty state for real systems */}
          {realSystems.length === 0 && demoSystems.length > 0 && (
            <div className="bg-slate-800/50 rounded-xl border border-dashed border-slate-600 p-8 text-center">
              <Server className="w-12 h-12 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">No Real Systems Yet</h3>
              <p className="text-slate-400 mb-4 max-w-md mx-auto">
                The systems above are demos for testing. Add your first real system to start analyzing your own data.
              </p>
              <button
                onClick={() => navigate('/systems/new')}
                className="px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors"
              >
                Add Your First System
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
