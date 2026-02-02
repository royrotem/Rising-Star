/**
 * useAnalysisStream — SSE hook for real-time analysis progress.
 *
 * Additive feature module.  Removing this file does not affect the
 * existing non-streaming analysis path.
 */

import { useCallback, useRef, useState } from 'react';

// ---------- Types ----------

export interface StreamStage {
  stage: string;
  message: string;
  progress: number;
}

export interface LayerEvent {
  layer: string;
  layer_index: number;
  total_layers: number;
  anomaly_count: number;
}

export interface AgentEvent {
  agent: string;
  status: string;
  findings: number;
  perspective?: string;
}

export interface StreamState {
  /** Whether the stream is currently active. */
  active: boolean;
  /** Current progress (0–100). */
  progress: number;
  /** Current stage message shown to the user. */
  message: string;
  /** Completed detection layers (up to 6). */
  layers: LayerEvent[];
  /** Completed AI agent results. */
  agents: AgentEvent[];
  /** Final analysis result (null until complete). */
  result: unknown | null;
  /** Error message if something went wrong. */
  error: string | null;
}

const INITIAL_STATE: StreamState = {
  active: false,
  progress: 0,
  message: '',
  layers: [],
  agents: [],
  result: null,
  error: null,
};

// ---------- Hook ----------

export function useAnalysisStream() {
  const [state, setState] = useState<StreamState>(INITIAL_STATE);
  const eventSourceRef = useRef<EventSource | null>(null);

  const start = useCallback((systemId: string, selectedAgents?: string[]) => {
    // Reset state
    setState({ ...INITIAL_STATE, active: true, message: 'Connecting...' });

    // Close any existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    let url = `/api/v1/systems/${systemId}/analyze-stream`;
    if (selectedAgents && selectedAgents.length > 0) {
      const agentsParam = encodeURIComponent(selectedAgents.join(','));
      url += `?agents=${agentsParam}`;
    }

    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.addEventListener('stage', (e: MessageEvent) => {
      const data: StreamStage = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        progress: data.progress,
        message: data.message,
      }));
    });

    es.addEventListener('layer_complete', (e: MessageEvent) => {
      const data: LayerEvent = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        layers: [...prev.layers, data],
      }));
    });

    es.addEventListener('agent_complete', (e: MessageEvent) => {
      const data: AgentEvent = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        agents: [...prev.agents, data],
      }));
    });

    es.addEventListener('result', (e: MessageEvent) => {
      const data = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        active: false,
        progress: 100,
        message: 'Analysis complete',
        result: data,
      }));
      es.close();
    });

    es.addEventListener('error', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          active: false,
          error: data.message || 'Unknown error',
        }));
      } catch {
        setState((prev) => ({
          ...prev,
          active: false,
          error: 'Connection lost',
        }));
      }
      es.close();
    });

    es.onerror = () => {
      // Don't let EventSource auto-reconnect — close immediately on error.
      // The analysis is a one-shot operation; reconnecting would start a new one
      // or hit the same timeout again.
      es.close();
      eventSourceRef.current = null;
      setState((prev) => {
        if (prev.result) return prev; // already got result, ignore
        return {
          ...prev,
          active: false,
          error: 'Connection lost — the analysis may still be running on the server. Refresh the page to check for results.',
        };
      });
    };
  }, []);

  const cancel = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setState((prev) => ({ ...prev, active: false, message: 'Cancelled' }));
  }, []);

  return { stream: state, startStream: start, cancelStream: cancel };
}
