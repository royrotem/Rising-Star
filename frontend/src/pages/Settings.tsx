import { useState, useEffect } from 'react';
import {
  ArrowLeft,
  Save,
  Loader2,
  CheckCircle,
  AlertCircle,
  Key,
  Brain,
  Globe,
  Cpu,
  Eye,
  EyeOff,
  Timer,
  Power,
  Clock,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import clsx from 'clsx';
import { systemsApi, schedulesApi } from '../services/api';
import type { System, Schedule } from '../types';

interface AISettings {
  anthropic_api_key: string;
  enable_ai_agents: boolean;
  enable_web_grounding: boolean;
  extended_thinking_budget: number;
}

interface AIStatus {
  configured: boolean;
  enabled: boolean;
  ready: boolean;
  web_grounding_enabled: boolean;
  message: string;
}

export default function Settings() {
  const [aiSettings, setAiSettings] = useState<AISettings>({
    anthropic_api_key: '',
    enable_ai_agents: true,
    enable_web_grounding: true,
    extended_thinking_budget: 10000,
  });
  const [aiStatus, setAiStatus] = useState<AIStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [keyEdited, setKeyEdited] = useState(false);

  // Watchdog state
  const [systems, setSystems] = useState<System[]>([]);
  const [schedules, setSchedules] = useState<Record<string, Schedule>>({});
  const [watchdogSaving, setWatchdogSaving] = useState<string | null>(null);

  useEffect(() => {
    loadSettings();
    loadWatchdog();
  }, []);

  const loadSettings = async () => {
    setLoading(true);
    try {
      const [settingsRes, statusRes] = await Promise.all([
        fetch('/api/v1/settings/'),
        fetch('/api/v1/settings/ai/status'),
      ]);

      if (settingsRes.ok) {
        const data = await settingsRes.json();
        setAiSettings(data.ai);
      }
      if (statusRes.ok) {
        const status = await statusRes.json();
        setAiStatus(status);
      }
    } catch {
      setError('Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const loadWatchdog = async () => {
    try {
      const [sysList, schedList] = await Promise.all([
        systemsApi.list(),
        schedulesApi.list(),
      ]);
      setSystems(sysList);
      const map: Record<string, Schedule> = {};
      for (const s of schedList) map[s.system_id] = s;
      setSchedules(map);
    } catch { /* not critical */ }
  };

  const toggleWatchdog = async (systemId: string) => {
    setWatchdogSaving(systemId);
    const current = schedules[systemId];
    const newEnabled = !current?.enabled;
    try {
      const updated = await schedulesApi.set(systemId, {
        enabled: newEnabled,
        interval: current?.interval || '24h',
      });
      setSchedules((prev) => ({ ...prev, [systemId]: updated }));
    } catch { /* ignore */ }
    setWatchdogSaving(null);
  };

  const changeWatchdogInterval = async (systemId: string, interval: string) => {
    setWatchdogSaving(systemId);
    const current = schedules[systemId];
    try {
      const updated = await schedulesApi.set(systemId, {
        enabled: current?.enabled ?? true,
        interval,
      });
      setSchedules((prev) => ({ ...prev, [systemId]: updated }));
    } catch { /* ignore */ }
    setWatchdogSaving(null);
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSaved(false);

    try {
      const payload: { ai: Partial<AISettings> } = {
        ai: {
          enable_ai_agents: aiSettings.enable_ai_agents,
          enable_web_grounding: aiSettings.enable_web_grounding,
          extended_thinking_budget: aiSettings.extended_thinking_budget,
        },
      };

      // Only send the API key if it was actually edited
      if (keyEdited) {
        payload.ai.anthropic_api_key = aiSettings.anthropic_api_key;
      }

      const res = await fetch('/api/v1/settings/', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error('Failed to save settings');
      }

      setKeyEdited(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);

      // Refresh status
      const statusRes = await fetch('/api/v1/settings/ai/status');
      if (statusRes.ok) {
        setAiStatus(await statusRes.json());
      }

      // Reload settings to get masked key
      const settingsRes = await fetch('/api/v1/settings/');
      if (settingsRes.ok) {
        const data = await settingsRes.json();
        setAiSettings(data.ai);
      }
    } catch {
      setError('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <Link to="/" className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
          <ArrowLeft className="w-5 h-5 text-slate-400" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">Settings</h1>
          <p className="text-slate-400 text-sm">Configure AI agents and application behavior</p>
        </div>
      </div>

      {/* AI Status Banner */}
      {aiStatus && (
        <div className={clsx(
          'mb-6 p-4 rounded-lg border flex items-center gap-3',
          aiStatus.ready
            ? 'bg-green-500/10 border-green-500/30'
            : 'bg-yellow-500/10 border-yellow-500/30'
        )}>
          {aiStatus.ready ? (
            <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
          ) : (
            <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0" />
          )}
          <p className="text-sm text-slate-300">{aiStatus.message}</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 rounded-lg border border-red-500/30 bg-red-500/10 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <p className="text-sm text-slate-300">{error}</p>
        </div>
      )}

      {/* AI Configuration */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 mb-6">
        <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <h2 className="text-lg font-semibold text-white">AI Agent Configuration</h2>
        </div>

        <div className="p-6 space-y-6">
          {/* API Key */}
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
              <Key className="w-4 h-4 text-yellow-400" />
              Anthropic API Key
            </label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <input
                  type={showKey ? 'text' : 'password'}
                  value={aiSettings.anthropic_api_key || ''}
                  onChange={(e) => {
                    setAiSettings({ ...aiSettings, anthropic_api_key: e.target.value });
                    setKeyEdited(true);
                  }}
                  placeholder="sk-ant-..."
                  className="w-full px-4 py-2.5 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 font-mono text-sm"
                />
                <button
                  type="button"
                  onClick={() => setShowKey(!showKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                >
                  {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Required for LLM-powered multi-agent analysis. Get your key at console.anthropic.com
            </p>
          </div>

          {/* Enable AI Agents */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Cpu className="w-5 h-5 text-primary-400" />
              <div>
                <p className="text-sm font-medium text-slate-300">Enable AI Agents</p>
                <p className="text-xs text-slate-500">Run multi-agent LLM analysis alongside rule-based engine</p>
              </div>
            </div>
            <button
              onClick={() => setAiSettings({ ...aiSettings, enable_ai_agents: !aiSettings.enable_ai_agents })}
              className={clsx(
                'relative w-12 h-6 rounded-full transition-colors',
                aiSettings.enable_ai_agents ? 'bg-primary-500' : 'bg-slate-600'
              )}
            >
              <div className={clsx(
                'absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform',
                aiSettings.enable_ai_agents ? 'translate-x-6' : 'translate-x-0.5'
              )} />
            </button>
          </div>

          {/* Web Grounding */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Globe className="w-5 h-5 text-cyan-400" />
              <div>
                <p className="text-sm font-medium text-slate-300">Web Grounding</p>
                <p className="text-xs text-slate-500">Allow agents to search the web for engineering context</p>
              </div>
            </div>
            <button
              onClick={() => setAiSettings({ ...aiSettings, enable_web_grounding: !aiSettings.enable_web_grounding })}
              className={clsx(
                'relative w-12 h-6 rounded-full transition-colors',
                aiSettings.enable_web_grounding ? 'bg-primary-500' : 'bg-slate-600'
              )}
            >
              <div className={clsx(
                'absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform',
                aiSettings.enable_web_grounding ? 'translate-x-6' : 'translate-x-0.5'
              )} />
            </button>
          </div>

          {/* Extended Thinking Budget */}
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
              <Brain className="w-4 h-4 text-purple-400" />
              Extended Thinking Budget
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={1000}
                max={50000}
                step={1000}
                value={aiSettings.extended_thinking_budget}
                onChange={(e) => setAiSettings({ ...aiSettings, extended_thinking_budget: Number(e.target.value) })}
                className="flex-1 accent-primary-500"
              />
              <span className="text-sm text-slate-400 w-20 text-right font-mono">
                {aiSettings.extended_thinking_budget.toLocaleString()} tokens
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Higher budget allows the Root Cause Investigator agent to think more deeply
            </p>
          </div>
        </div>
      </div>

      {/* Watchdog (Scheduled Auto-Analysis) */}
      {systems.length > 0 && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 mb-6">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
            <Timer className="w-5 h-5 text-primary-400" />
            <h2 className="text-lg font-semibold text-white">Watchdog — Scheduled Auto-Analysis</h2>
          </div>
          <div className="p-6">
            <p className="text-xs text-slate-500 mb-4">
              Enable automatic periodic analysis per system. Each system can have its own schedule.
            </p>
            <div className="space-y-3">
              {systems.map((sys) => {
                const sched = schedules[sys.id];
                const enabled = sched?.enabled ?? false;
                const interval = sched?.interval ?? '24h';
                const isSaving = watchdogSaving === sys.id;
                return (
                  <div
                    key={sys.id}
                    className={clsx(
                      'flex items-center justify-between p-4 rounded-lg border transition-colors',
                      enabled
                        ? 'bg-primary-500/5 border-primary-500/20'
                        : 'bg-slate-900/30 border-slate-700'
                    )}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <div className={clsx(
                        'w-2 h-2 rounded-full flex-shrink-0',
                        enabled ? 'bg-primary-400' : 'bg-slate-600'
                      )} />
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-white truncate">{sys.name}</p>
                        <p className="text-xs text-slate-500 capitalize">{sys.system_type.replace('_', ' ')}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                      {/* Interval selector */}
                      <select
                        value={interval}
                        onChange={(e) => changeWatchdogInterval(sys.id, e.target.value)}
                        disabled={isSaving}
                        className="px-2 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-xs text-slate-300 focus:outline-none focus:border-primary-500"
                      >
                        <option value="1h">Every 1h</option>
                        <option value="6h">Every 6h</option>
                        <option value="12h">Every 12h</option>
                        <option value="24h">Every 24h</option>
                        <option value="7d">Every 7d</option>
                      </select>

                      {/* Last run info */}
                      {sched?.last_run_at && (
                        <span className="flex items-center gap-1 text-xs text-slate-500">
                          <Clock className="w-3 h-3" />
                          {sched.run_count} runs
                        </span>
                      )}

                      {/* Toggle */}
                      <button
                        onClick={() => toggleWatchdog(sys.id)}
                        disabled={isSaving}
                        className={clsx(
                          'relative w-10 h-5 rounded-full transition-colors flex-shrink-0',
                          enabled ? 'bg-primary-500' : 'bg-slate-600'
                        )}
                      >
                        {isSaving ? (
                          <Loader2 className="w-3 h-3 text-white animate-spin absolute top-1 left-3.5" />
                        ) : (
                          <div className={clsx(
                            'absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform',
                            enabled ? 'translate-x-5' : 'translate-x-0.5'
                          )} />
                        )}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Save Button */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 px-6 py-2.5 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white rounded-lg font-medium transition-colors"
        >
          {saving ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Save className="w-4 h-4" />
          )}
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
        {saved && (
          <span className="flex items-center gap-1 text-green-400 text-sm">
            <CheckCircle className="w-4 h-4" />
            Settings saved
          </span>
        )}
      </div>
    </div>
  );
}
