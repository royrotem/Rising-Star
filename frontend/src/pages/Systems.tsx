import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Plus,
  Search,
  Server,
  Cpu,
  Heart,
  Car,
  Activity,
  ChevronRight,
  X,
  Loader2
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';
import type { System } from '../types';

const systemTypeIcons: Record<string, React.ElementType> = {
  vehicle: Car,
  robot: Cpu,
  medical_device: Heart,
  default: Server,
};

function getStatusColor(status: string) {
  switch (status) {
    case 'active': return 'bg-green-500';
    case 'anomaly_detected': return 'bg-orange-500';
    case 'maintenance': return 'bg-yellow-500';
    case 'inactive': return 'bg-slate-500';
    default: return 'bg-slate-500';
  }
}

function getHealthColor(score: number) {
  if (score >= 90) return 'text-green-500';
  if (score >= 70) return 'text-yellow-500';
  return 'text-red-500';
}

export default function Systems() {
  const [systems, setSystems] = useState<System[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newSystem, setNewSystem] = useState({
    name: '',
    system_type: 'vehicle',
  });

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

  const handleCreateSystem = async () => {
    if (!newSystem.name.trim()) return;

    setCreating(true);
    try {
      const created = await systemsApi.create(newSystem);
      setSystems(prev => [...prev, created]);
      setShowCreateModal(false);
      setNewSystem({ name: '', system_type: 'vehicle' });
    } catch (error) {
      console.error('Failed to create system:', error);
      // Add locally for demo
      const mockSystem: System = {
        id: String(Date.now()),
        name: newSystem.name,
        system_type: newSystem.system_type,
        status: 'active',
        health_score: 100,
        created_at: new Date().toISOString(),
      };
      setSystems(prev => [...prev, mockSystem]);
      setShowCreateModal(false);
      setNewSystem({ name: '', system_type: 'vehicle' });
    } finally {
      setCreating(false);
    }
  };

  const filteredSystems = systems.filter(system =>
    system.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    system.system_type.toLowerCase().includes(searchQuery.toLowerCase())
  );

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
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors"
        >
          <Plus className="w-5 h-5" />
          Add System
        </button>
      </div>

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
          <p className="text-slate-400">No systems found</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="mt-4 text-primary-400 hover:text-primary-300"
          >
            Add your first system
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredSystems.map((system) => {
            const Icon = systemTypeIcons[system.system_type] || systemTypeIcons.default;
            return (
              <Link
                key={system.id}
                to={`/systems/${system.id}`}
                className="bg-slate-800 rounded-xl border border-slate-700 p-6 hover:border-primary-500/50 transition-colors group"
              >
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
                      {system.health_score}% Health
                    </span>
                  </div>
                  <ChevronRight className="w-5 h-5 text-slate-500 group-hover:text-primary-400 transition-colors" />
                </div>
              </Link>
            );
          })}
        </div>
      )}

      {/* Create System Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-white">Add New System</h2>
              <button
                onClick={() => setShowCreateModal(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  System Name
                </label>
                <input
                  type="text"
                  value={newSystem.name}
                  onChange={(e) => setNewSystem(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="e.g., Fleet Vehicle Beta"
                  className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  System Type
                </label>
                <select
                  value={newSystem.system_type}
                  onChange={(e) => setNewSystem(prev => ({ ...prev, system_type: e.target.value }))}
                  className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-primary-500 transition-colors"
                >
                  <option value="vehicle">Vehicle</option>
                  <option value="robot">Robot</option>
                  <option value="medical_device">Medical Device</option>
                  <option value="aerospace">Aerospace</option>
                  <option value="industrial">Industrial Equipment</option>
                </select>
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="flex-1 px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateSystem}
                  disabled={!newSystem.name.trim() || creating}
                  className="flex-1 px-4 py-3 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  {creating ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Plus className="w-5 h-5" />
                      Create System
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
