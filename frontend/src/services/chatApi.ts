/**
 * AI Chat API Service
 *
 * Additive module — real-time conversational AI interface.
 * Removing this file does not affect core functionality.
 */

import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1/chat',
  headers: { 'Content-Type': 'application/json' },
});

export interface ChatMessageResponse {
  id: string;
  role: string;
  content: string;
  timestamp: string;
  ai_powered: boolean;
  data?: Record<string, unknown> | null;
}

export interface ChatHistory {
  system_id: string;
  messages: Array<{
    id: string;
    role: string;
    content: string;
    timestamp: string;
    data?: Record<string, unknown> | null;
  }>;
}

export const chatApi = {
  /** Send a message and get an AI response. */
  send: async (
    systemId: string,
    message: string
  ): Promise<ChatMessageResponse> => {
    const { data } = await api.post(`/systems/${systemId}`, { message });
    return data;
  },

  /** Get conversation history. */
  history: async (systemId: string, limit = 50): Promise<ChatHistory> => {
    const { data } = await api.get(`/systems/${systemId}/history`, {
      params: { limit },
    });
    return data;
  },

  /** Clear conversation history. */
  clear: async (systemId: string): Promise<void> => {
    await api.delete(`/systems/${systemId}/history`);
  },
};
