/**
 * Anomaly Feedback API Service
 *
 * Additive module — provides feedback submission and retrieval.
 * Removing this file does not affect core functionality.
 */

import axios from 'axios';
import type { AnomalyFeedback, FeedbackSummary, FeedbackType } from '../types';

const api = axios.create({
  baseURL: '/api/v1/feedback',
  headers: { 'Content-Type': 'application/json' },
});

export const feedbackApi = {
  /** Submit feedback for a specific anomaly. */
  submit: async (
    systemId: string,
    payload: {
      anomaly_id: string;
      anomaly_title: string;
      anomaly_type: string;
      severity: string;
      feedback_type: FeedbackType;
      comment?: string;
    }
  ): Promise<AnomalyFeedback> => {
    const { data } = await api.post(`/systems/${systemId}`, payload);
    return data;
  },

  /** Get all feedback entries for a system (optionally filtered). */
  list: async (
    systemId: string,
    anomalyId?: string
  ): Promise<AnomalyFeedback[]> => {
    const params: Record<string, string> = {};
    if (anomalyId) params.anomaly_id = anomalyId;
    const { data } = await api.get(`/systems/${systemId}`, { params });
    return data;
  },

  /** Get aggregated feedback summary with confidence score. */
  summary: async (systemId: string): Promise<FeedbackSummary> => {
    const { data } = await api.get(`/systems/${systemId}/summary`);
    return data;
  },

  /** Delete a feedback entry. */
  remove: async (systemId: string, feedbackId: string): Promise<void> => {
    await api.delete(`/systems/${systemId}/${feedbackId}`);
  },
};
