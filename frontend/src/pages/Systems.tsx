import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Plus,
  Trash2,
  ChevronRight,
  Loader2,
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

  useEffect(() => {
    loadSystems();
  }, []);

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
          <p className="text-stone-500 text-sm mt-1">
            {systems.length} system{systems.length !== 1 ? 's' : ''} monitored
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

      {systems.length === 0 ? (
        <div className="text-center py-20">
          <p className="text-stone-400 mb-3">No systems configured</p>
          <button
            onClick={() => navigate('/systems/new')}
            className="text-primary-400 hover:text-primary-300 text-sm"
          >
            Add your first system
          </button>
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
                <p className="text-xs text-stone-600 capitalize mt-0.5">
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
                className="p-1.5 text-stone-700 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all rounded"
              >
                {deleting === system.id ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Trash2 className="w-3.5 h-3.5" />
                )}
              </button>

              <ChevronRight className="w-4 h-4 text-stone-700 group-hover:text-stone-500 transition-colors" />
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
