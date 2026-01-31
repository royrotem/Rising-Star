import { useState, useEffect } from 'react';
import {
  Timer,
  Power,
  CheckCircle,
  AlertCircle,
  Loader2,
  Clock,
} from 'lucide-react';
import clsx from 'clsx';
import { schedulesApi } from '../services/api';
import type { Schedule } from '../types';

const INTERVAL_OPTIONS = [
  { value: '1h', label: 'Every hour' },
  { value: '6h', label: 'Every 6 hours' },
  { value: '12h', label: 'Every 12 hours' },
  { value: '24h', label: 'Every 24 hours' },
  { value: '7d', label: 'Every 7 days' },
] as const;

function formatTimeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

interface Props {
  systemId: string;
}

export default function WatchdogPanel({ systemId }: Props) {
  const [schedule, setSchedule] = useState<Schedule | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [localEnabled, setLocalEnabled] = useState(false);
  const [localInterval, setLocalInterval] = useState('24h');

  useEffect(() => {
    loadSchedule();
  }, [systemId]);

  const loadSchedule = async () => {
    setLoading(true);
    try {
      const data = await schedulesApi.get(systemId);
      setSchedule(data);
      setLocalEnabled(data.enabled);
      setLocalInterval(data.interval);
    } catch {
      // No schedule yet — defaults are fine
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = async () => {
    const newEnabled = !localEnabled;
    setLocalEnabled(newEnabled);
    await saveSchedule(newEnabled, localInterval);
  };

  const handleIntervalChange = async (interval: string) => {
    setLocalInterval(interval);
    if (localEnabled) {
      await saveSchedule(localEnabled, interval);
    }
  };

  const saveSchedule = async (enabled: boolean, interval: string) => {
    setSaving(true);
    try {
      const data = await schedulesApi.set(systemId, { enabled, interval });
      setSchedule(data);
    } catch {
      // Revert on failure
      if (schedule) {
        setLocalEnabled(schedule.enabled);
        setLocalInterval(schedule.interval);
      }
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-stone-800 rounded-xl border border-stone-700 p-6 mb-6">
        <div className="flex items-center gap-3">
          <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
          <span className="text-sm text-stone-400">Loading watchdog configuration...</span>
        </div>
      </div>
    );
  }

  const statusIcon = schedule?.last_run_status === 'success'
    ? <CheckCircle className="w-4 h-4 text-green-400" />
    : schedule?.last_run_status === 'error'
    ? <AlertCircle className="w-4 h-4 text-red-400" />
    : null;

  return (
    <div className={clsx(
      'rounded-xl border p-6 mb-6 transition-colors',
      localEnabled
        ? 'bg-primary-500/5 border-primary-500/30'
        : 'bg-stone-800 border-stone-700'
    )}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={clsx(
            'p-2.5 rounded-xl',
            localEnabled ? 'bg-primary-500/10' : 'bg-stone-700/50'
          )}>
            <Timer className={clsx('w-5 h-5', localEnabled ? 'text-primary-400' : 'text-stone-500')} />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              Watchdog Mode
              {localEnabled && (
                <span className="px-2 py-0.5 bg-primary-500/20 text-primary-400 text-xs rounded-full font-medium">
                  Active
                </span>
              )}
            </h2>
            <p className="text-xs text-stone-500">
              Automatically re-run analysis on a schedule
            </p>
          </div>
        </div>

        <button
          onClick={handleToggle}
          disabled={saving}
          className={clsx(
            'relative w-12 h-6 rounded-full transition-colors',
            localEnabled ? 'bg-primary-500' : 'bg-stone-600'
          )}
        >
          <div className={clsx(
            'absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform',
            localEnabled ? 'transtone-x-6' : 'transtone-x-0.5'
          )} />
        </button>
      </div>

      {/* Interval selector */}
      <div className="flex items-center gap-2 mb-4">
        {INTERVAL_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => handleIntervalChange(opt.value)}
            disabled={saving}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              localInterval === opt.value
                ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                : 'bg-stone-700/50 text-stone-400 border border-stone-600 hover:border-stone-500'
            )}
          >
            {opt.label}
          </button>
        ))}
        {saving && <Loader2 className="w-4 h-4 text-primary-400 animate-spin ml-2" />}
      </div>

      {/* Last run info */}
      {schedule && (schedule.last_run_at || schedule.run_count > 0) && (
        <div className="flex items-center gap-4 text-xs text-stone-500">
          {schedule.last_run_at && (
            <span className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5" />
              Last run: {formatTimeAgo(schedule.last_run_at)}
            </span>
          )}
          {statusIcon && (
            <span className="flex items-center gap-1.5">
              {statusIcon}
              {schedule.last_run_status}
            </span>
          )}
          {schedule.run_count > 0 && (
            <span className="flex items-center gap-1.5">
              <Power className="w-3.5 h-3.5" />
              {schedule.run_count} runs total
            </span>
          )}
          {schedule.last_error && (
            <span className="text-red-400 truncate max-w-xs" title={schedule.last_error}>
              {schedule.last_error}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
