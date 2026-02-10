import { useState, useEffect } from 'react';
import {
  Loader2,
  Save,
  Timer,
  Power,
  CheckCircle2,
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi, schedulesApi } from '../services/api';
import type { System, Schedule } from '../types';

export default function Settings() {
  const [apiKey, setApiKey] = useState('');
  const [maskedKey, setMaskedKey] = useState('');
  const [keyEdited, setKeyEdited] = useState(false);
  const [enableAiAgents, setEnableAiAgents] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showKey, setShowKey] = useState(false);

  // Watchdog state
  const [systems, setSystems] = useState<System[]>([]);
  const [schedules, setSchedules] = useState<Record<string, Schedule>>({});
  const [watchdogSaving, setWatchdogSaving] = useState<string | null>(null);

  useEffect(() => {
    loadSettings();
    loadWatchdog();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await fetch('/api/v1/settings/');
      if (response.ok) {
        const data = await response.json();
        const ai = data.ai || {};
        setEnableAiAgents(ai.enable_ai_agents ?? true);
        // Backend returns masked key like "sk-ant-a..." or "***configured***"
        const key = ai.anthropic_api_key || '';
        if (key && key !== '***configured***') {
          setMaskedKey(key);
        } else if (key === '***configured***') {
          setMaskedKey(key);
        }
        // Don't set apiKey — keep it empty so user can type fresh
        setApiKey('');
        setKeyEdited(false);
      }
    } catch {
      // Settings not available
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
    } catch {
      // Non-critical
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    try {
      const aiPayload: Record<string, unknown> = {
        enable_ai_agents: enableAiAgents,
      };
      // Only send the API key if the user actually typed a new one
      if (keyEdited && apiKey.trim()) {
        aiPayload.anthropic_api_key = apiKey.trim();
      } else {
        // Send the masked key back — backend will recognize it and keep existing
        aiPayload.anthropic_api_key = maskedKey || null;
      }

      await fetch('/api/v1/settings/', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ai: aiPayload }),
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
      // Reload to get updated masked key
      await loadSettings();
    } catch {
      // Save failed
    } finally {
      setSaving(false);
    }
  };

  const handleKeyChange = (value: string) => {
    setApiKey(value);
    setKeyEdited(true);
  };

  const toggleWatchdog = async (systemId: string, currentlyEnabled: boolean) => {
    setWatchdogSaving(systemId);
    try {
      if (currentlyEnabled) {
        await schedulesApi.delete(systemId);
        setSchedules((prev) => {
          const next = { ...prev };
          delete next[systemId];
          return next;
        });
      } else {
        const sched = await schedulesApi.set(systemId, { enabled: true, interval: '24h' });
        setSchedules((prev) => ({ ...prev, [systemId]: sched }));
      }
    } catch { /* */ } finally {
      setWatchdogSaving(null);
    }
  };

  const changeWatchdogInterval = async (systemId: string, interval: string) => {
    setWatchdogSaving(systemId);
    try {
      const sched = await schedulesApi.set(systemId, { enabled: true, interval });
      setSchedules((prev) => ({ ...prev, [systemId]: sched }));
    } catch { /* */ } finally {
      setWatchdogSaving(null);
    }
  };

  const hasConfiguredKey = !!maskedKey;

  return (
    <div className="p-8 max-w-3xl page-enter">
      <div className="mb-10">
        <h1 className="text-xl font-semibold text-white">Settings</h1>
        <p className="text-stone-300 text-sm mt-1">
          Platform configuration and watchdog management
        </p>
      </div>

      {/* AI Configuration */}
      <div className="glass-card mb-8">
        <div className="px-6 py-4 border-b border-stone-600/50">
          <h2 className="text-sm font-medium text-white">AI Configuration</h2>
        </div>
        <div className="p-6 space-y-5">
          {/* API Key */}
          <div>
            <label className="block text-xs font-medium text-stone-300 mb-2">
              Anthropic API Key
            </label>
            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={keyEdited ? apiKey : ''}
                onChange={(e) => handleKeyChange(e.target.value)}
                placeholder={hasConfiguredKey ? maskedKey : 'sk-ant-api03-...'}
                className="w-full px-4 py-2.5 bg-stone-600 border border-stone-600/50 rounded-lg text-sm text-white placeholder-stone-400 focus:outline-none focus:border-stone-400 transition-colors pr-16"
              />
              <button
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-stone-300 hover:text-stone-300 text-xs transition-colors"
              >
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
            {hasConfiguredKey && !keyEdited && (
              <div className="flex items-center gap-1.5 mt-2">
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                <p className="text-xs text-emerald-400">
                  API key configured
                </p>
              </div>
            )}
            {keyEdited && apiKey.trim() && (
              <p className="text-xs text-stone-300 mt-2">
                New key will be saved when you click Save
              </p>
            )}
          </div>

          {/* AI Agents Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-white">AI Agents</p>
              <p className="text-xs text-stone-300 mt-0.5">
                Use LLM-powered analysis (requires API key)
              </p>
            </div>
            <button
              onClick={() => setEnableAiAgents(!enableAiAgents)}
              className={clsx(
                'w-10 h-5.5 rounded-full transition-colors duration-200 relative',
                enableAiAgents ? 'bg-primary-500' : 'bg-stone-500'
              )}
            >
              <div className={clsx(
                'w-4 h-4 bg-white rounded-full absolute top-0.5 transition-transform duration-200',
                enableAiAgents ? 'translate-x-5' : 'translate-x-0.5'
              )} />
            </button>
          </div>

          {/* Save */}
          <div className="pt-2">
            <button
              onClick={handleSave}
              disabled={saving}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                saved
                  ? 'bg-emerald-500/10 text-emerald-400'
                  : 'bg-primary-500 hover:bg-primary-600 text-white'
              )}
            >
              {saving ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Save className="w-3.5 h-3.5" />
              )}
              {saved ? 'Saved' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>

      {/* Watchdog Management */}
      {systems.length > 0 && (
        <div className="glass-card">
          <div className="px-6 py-4 border-b border-stone-600/50 flex items-center gap-2">
            <Timer className="w-4 h-4 text-stone-300" />
            <h2 className="text-sm font-medium text-white">Watchdog Schedules</h2>
          </div>
          <div className="divide-y divide-stone-600/40">
            {systems.map((sys) => {
              const sched = schedules[sys.id];
              const isEnabled = sched?.enabled ?? false;
              const isSaving = watchdogSaving === sys.id;

              return (
                <div key={sys.id} className="px-6 py-4 flex items-center justify-between">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-white truncate">{sys.name}</p>
                    <p className="text-xs text-stone-300 capitalize">{sys.system_type.replace('_', ' ')}</p>
                    {sched && sched.run_count > 0 && (
                      <p className="text-[11px] text-stone-300 mt-0.5 tabular-nums">
                        {sched.run_count} runs
                      </p>
                    )}
                  </div>

                  <div className="flex items-center gap-4 flex-shrink-0">
                    {/* Interval selector */}
                    {isEnabled && (
                      <select
                        value={sched?.interval || '24h'}
                        onChange={(e) => changeWatchdogInterval(sys.id, e.target.value)}
                        disabled={isSaving}
                        className="bg-stone-600 border border-stone-600/50 rounded-lg px-2.5 py-1.5 text-xs text-stone-300 focus:outline-none focus:border-stone-400 disabled:opacity-50"
                      >
                        <option value="1h">1h</option>
                        <option value="6h">6h</option>
                        <option value="12h">12h</option>
                        <option value="24h">24h</option>
                        <option value="7d">7d</option>
                      </select>
                    )}

                    {/* Toggle */}
                    <button
                      onClick={() => toggleWatchdog(sys.id, isEnabled)}
                      disabled={isSaving}
                      className={clsx(
                        'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                        isEnabled
                          ? 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15'
                          : 'bg-stone-600 text-stone-300 hover:bg-stone-500 hover:text-stone-300'
                      )}
                    >
                      {isSaving ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <Power className="w-3 h-3" />
                      )}
                      {isEnabled ? 'Active' : 'Off'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
