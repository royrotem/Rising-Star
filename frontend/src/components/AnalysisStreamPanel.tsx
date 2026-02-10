/**
 * AnalysisStreamPanel — Live progress display for streaming analysis.
 *
 * Additive feature module. Shows real-time progress of the 6-layer
 * detection engine and AI agent swarm during analysis.
 * Removing this file does not affect core functionality.
 */

import {
  Loader2,
  CheckCircle,
  XCircle,
  Cpu,
  Layers,
  Brain,
  Zap,
} from 'lucide-react';
import clsx from 'clsx';
import type { StreamState } from '../hooks/useAnalysisStream';

interface Props {
  stream: StreamState;
}

export function AnalysisStreamPanel({ stream }: Props) {
  if (!stream.active && !stream.message) return null;

  const { progress, message, layers, models, agents, active, error } = stream;

  return (
    <div className="bg-stone-700 rounded-xl border border-stone-600 p-6 mb-6 animate-in fade-in">
      {/* Progress bar */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {active ? (
              <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
            ) : error ? (
              <XCircle className="w-4 h-4 text-red-400" />
            ) : (
              <CheckCircle className="w-4 h-4 text-green-400" />
            )}
            <span className="text-sm font-medium text-white">
              {active ? 'Analysis In Progress' : error ? 'Analysis Failed' : 'Analysis Complete'}
            </span>
          </div>
          <span className="text-xs text-stone-400">{progress}%</span>
        </div>
        <div className="h-2 bg-stone-700 rounded-full overflow-hidden">
          <div
            className={clsx(
              'h-full rounded-full transition-all duration-500 ease-out',
              error ? 'bg-red-500' : 'bg-primary-500'
            )}
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-xs text-stone-400 mt-1.5">{message}</p>
      </div>

      {/* Detection Layers */}
      {layers.length > 0 && (
        <div className="mb-4">
          <h4 className="text-xs font-medium text-stone-400 mb-2 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5" />
            Detection Layers
          </h4>
          <div className="grid grid-cols-6 gap-1.5">
            {Array.from({ length: 6 }, (_, i) => {
              const layer = layers.find((l) => l.layer_index === i + 1);
              const done = !!layer;
              const names = [
                'Statistical',
                'Threshold',
                'Trend',
                'Correlation',
                'Pattern',
                'Rate of Change',
              ];
              return (
                <div
                  key={i}
                  className={clsx(
                    'p-2 rounded-lg border text-center transition-all duration-300',
                    done
                      ? 'border-green-500/30 bg-green-500/5'
                      : active
                      ? 'border-stone-600 bg-stone-700/30 animate-pulse'
                      : 'border-stone-600 bg-stone-700/30'
                  )}
                >
                  {done ? (
                    <CheckCircle className="w-4 h-4 mx-auto text-green-400 mb-1" />
                  ) : (
                    <Zap className="w-4 h-4 mx-auto text-stone-500 mb-1" />
                  )}
                  <p className="text-[10px] text-stone-300 leading-tight">{names[i]}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ML Models */}
      {models.length > 0 && (
        <div className="mb-4">
          <h4 className="text-xs font-medium text-stone-400 mb-2 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-blue-400" />
            ML Models
          </h4>
          <div className="grid grid-cols-3 gap-1.5">
            {models.map((model, idx) => (
              <div
                key={idx}
                className={clsx(
                  'p-2 rounded-lg border text-center transition-all duration-300',
                  model.status === 'success'
                    ? 'border-blue-500/30 bg-blue-500/5'
                    : model.status === 'unavailable'
                    ? 'border-stone-600 bg-stone-700/30'
                    : 'border-red-500/30 bg-red-500/5'
                )}
              >
                <Cpu
                  className={clsx(
                    'w-4 h-4 mx-auto mb-1',
                    model.status === 'success'
                      ? 'text-blue-400'
                      : model.status === 'unavailable'
                      ? 'text-stone-500'
                      : 'text-red-400'
                  )}
                />
                <p className="text-[10px] text-white truncate">{model.model}</p>
                <p className="text-[10px] text-stone-500">
                  {model.status === 'success'
                    ? `${model.findings} findings`
                    : model.status === 'unavailable'
                    ? 'Not loaded'
                    : 'Error'}
                </p>
                {model.status === 'success' && model.elapsed_seconds > 0 && (
                  <p className="text-[10px] text-stone-600">{model.elapsed_seconds}s</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Agents */}
      {agents.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-stone-400 mb-2 flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5" />
            AI Agents
          </h4>
          <div className="grid grid-cols-4 sm:grid-cols-5 lg:grid-cols-7 gap-1.5">
            {agents.map((agent, idx) => (
              <div
                key={idx}
                className={clsx(
                  'p-2 rounded-lg border text-center transition-all duration-300',
                  agent.status === 'success'
                    ? 'border-green-500/30 bg-green-500/5'
                    : 'border-red-500/30 bg-red-500/5'
                )}
              >
                <Cpu
                  className={clsx(
                    'w-4 h-4 mx-auto mb-1',
                    agent.status === 'success' ? 'text-green-400' : 'text-red-400'
                  )}
                />
                <p className="text-[10px] text-white truncate">{agent.agent}</p>
                <p className="text-[10px] text-stone-500">
                  {agent.status === 'success'
                    ? `${agent.findings} findings`
                    : 'Error'}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}
    </div>
  );
}
