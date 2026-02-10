import { useState, useEffect } from 'react';
import {
  Timer,
  Power,
  Loader2,
} from 'lucide-react';
import clsx from 'clsx';
import { schedulesApi } from '../services/api';
import type { Schedule } from '../types';

interface WatchdogPanelProps {
  systemId: string;
}

const INTERVALS = [
  { value: '1h', label: 'Every hour' },
  { value: '6h', label: 'Every 6 hours' },
  { value: '12h', label: 'Every 12 hours' },
  { value: '24h', label: 'Every 24 hours' },
  { value: '7d', label: 'Weekly' },
];

export default function WatchdogPanel({ systemId }: WatchdogPanelProps) {
  const [schedule, setSchedule] = useState<Schedule | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadSchedule();
  }, [systemId]);

  const loadSchedule = async () => {
    try {
      const sched = await schedulesApi.get(systemId);
      setSchedule(sched);
    } catch {
      setSchedule(null);
    } finally {
      setLoading(false);
    }
  };

  const toggleEnabled = async () => {
    setSaving(true);
    try {
      if (schedule?.enabled) {
        await schedulesApi.delete(systemId);
        setSchedule(null);
      } else {
        const sched = await schedulesApi.set(systemId, {
          enabled: true,
          interval: schedule?.interval || '24h',
        });
        setSchedule(sched);
      }
    } catch { /* */ } finally {
      setSaving(false);
    }
  };

  const changeInterval = async (interval: string) => {
    setSaving(true);
    try {
      const sched = await schedulesApi.set(systemId, { enabled: true, interval });
      setSchedule(sched);
    } catch { /* */ } finally {
      setSaving(false);
    }
  };

  if (loading) return null;

  const isEnabled = schedule?.enabled ?? false;

  return (
    <div className="glass-card p-5 mb-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Timer className={clsx(
            'w-4 h-4',
            isEnabled ? 'text-primary-400' : 'text-stone-400'
          )} />
          <div>
            <p className="text-sm text-white">Watchdog Mode</p>
            <p className="text-xs text-stone-400 mt-0.5">
              {isEnabled
                ? `Auto-analysis ${schedule?.interval || '24h'} · ${schedule?.run_count || 0} runs`
                : 'Scheduled auto-analysis (disabled)'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {isEnabled && (
            <select
              value={schedule?.interval || '24h'}
              onChange={(e) => changeInterval(e.target.value)}
              disabled={saving}
              className="bg-stone-700 border border-stone-600 rounded-lg px-2.5 py-1.5 text-xs text-stone-300 focus:outline-none focus:border-stone-400"
            >
              {INTERVALS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          )}

          <button
            onClick={toggleEnabled}
            disabled={saving}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              isEnabled
                ? 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15'
                : 'bg-stone-600 text-stone-300 hover:bg-stone-500 hover:text-stone-200'
            )}
          >
            {saving ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <Power className="w-3 h-3" />
            )}
            {isEnabled ? 'Active' : 'Enable'}
          </button>
        </div>
      </div>
    </div>
  );
}
