import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Plus,
  Trash2,
  ChevronRight,
  Loader2,
  Sparkles,
  Play,
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';
import type { System } from '../types';
import { getStatusColor, getHealthColor } from '../utils/colors';

export default function Systems() {
  const navigate = useNavigate();
  const [systems, setSystems] = useState<System[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [creatingDemo, setCreatingDemo] = useState(false);

  useEffect(() => {
    loadSystems();
  }, []);

  const handleCreateDemo = async () => {
    setCreatingDemo(true);
    try {
      const response = await fetch('/api/v1/systems/demo/create', { method: 'POST' });
      if (!response.ok) throw new Error('Failed to create demo');
      const data = await response.json();
      // Navigate to the new demo system
      navigate(`/systems/${data.system_id}`);
    } catch (error) {
      console.error('Failed to create demo:', error);
      alert('Failed to create demo system. Please try again.');
    } finally {
      setCreatingDemo(false);
    }
  };

  const loadSystems = async () => {
    try {
      const data = await systemsApi.list();
      setSystems(data);
    } catch (error) {
      console.error('Failed to load systems:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, systemId: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm('Delete this system and all its data?')) return;

    setDeleting(systemId);
    try {
      await systemsApi.delete(systemId);
      setSystems((prev) => prev.filter((s) => s.id !== systemId));
    } catch (error) {
      console.error('Failed to delete system:', error);
    } finally {
      setDeleting(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-6 h-6 text-stone-600 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-8 page-enter">
      {/* Header */}
      <div className="flex items-center justify-between mb-10">
        <div>
          <h1 className="text-xl font-semibold text-white">Systems</h1>
          <p className="text-stone-400 text-sm mt-1">
            {systems.length} system{systems.length !== 1 ? 's' : ''} monitored
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleCreateDemo}
            disabled={creatingDemo}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 text-white text-sm rounded-lg font-medium transition-all shadow-lg shadow-purple-500/20"
          >
            {creatingDemo ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Creating Demo...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Try Demo
              </>
            )}
          </button>
          <button
            onClick={() => navigate('/systems/new')}
            className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white text-sm rounded-lg font-medium transition-colors"
          >
            <Plus className="w-4 h-4" />
            Add System
          </button>
        </div>
      </div>

      {systems.length === 0 ? (
        <div className="text-center py-16">
          <div className="mb-8">
            <Sparkles className="w-12 h-12 text-purple-400 mx-auto mb-4" />
            <h2 className="text-lg font-medium text-white mb-2">Welcome to UAIE</h2>
            <p className="text-stone-400 max-w-md mx-auto">
              Universal Autonomous Insight Engine — AI-powered anomaly detection for hardware systems.
              Try the demo to see 25 AI agents analyze real sensor data.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <button
              onClick={handleCreateDemo}
              disabled={creatingDemo}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 text-white rounded-lg font-medium transition-all shadow-lg shadow-purple-500/25"
            >
              {creatingDemo ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Creating Demo...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Run Interactive Demo
                </>
              )}
            </button>
            <span className="text-stone-500">or</span>
            <button
              onClick={() => navigate('/systems/new')}
              className="flex items-center gap-2 px-6 py-3 bg-stone-700 hover:bg-stone-600 text-white rounded-lg font-medium transition-colors"
            >
              <Plus className="w-5 h-5" />
              Add Your Own System
            </button>
          </div>

          <p className="mt-8 text-xs text-stone-500">
            Demo creates an HVAC system with 1000 records and embedded anomalies for all 25 AI agents
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {systems.map((system) => (
            <Link
              key={system.id}
              to={`/systems/${system.id}`}
              className="flex items-center gap-4 p-4 glass-card-hover group"
            >
              {/* Status dot */}
              <div className={clsx(
                'w-2 h-2 rounded-full flex-shrink-0',
                getStatusColor(system.status)
              )} />

              {/* Info */}
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-medium text-stone-200 group-hover:text-white transition-colors truncate">
                  {system.name}
                </h3>
                <p className="text-xs text-stone-500 capitalize mt-0.5">
                  {system.system_type.replace('_', ' ')}
                  {system.anomaly_count ? ` · ${system.anomaly_count} anomalies` : ''}
                </p>
              </div>

              {/* Health score */}
              <span className={clsx(
                'text-sm font-medium tabular-nums',
                getHealthColor(system.health_score)
              )}>
                {system.health_score}%
              </span>

              {/* Delete */}
              <button
                onClick={(e) => handleDelete(e, system.id)}
                disabled={deleting === system.id}
                className="p-1.5 text-stone-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all rounded"
              >
                {deleting === system.id ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Trash2 className="w-3.5 h-3.5" />
                )}
              </button>

              <ChevronRight className="w-4 h-4 text-stone-500 group-hover:text-stone-300 transition-colors" />
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
