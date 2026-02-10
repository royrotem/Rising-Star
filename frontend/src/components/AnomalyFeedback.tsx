/**
 * Anomaly Feedback Components
 *
 * Additive feature module — provides feedback buttons and history display
 * for anomaly cards. Removing this file does not break core functionality.
 */

import { useState, useEffect } from 'react';
import {
  ThumbsUp,
  ThumbsDown,
  Clock,
  MessageSquare,
  ChevronDown,
  ChevronUp,
  X,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';
import clsx from 'clsx';
import { feedbackApi } from '../services/feedbackApi';
import type { AnomalyFeedback, FeedbackType, FeedbackSummary } from '../types';

// ---------- Feedback Buttons ----------

interface FeedbackButtonsProps {
  systemId: string;
  anomalyId: string;
  anomalyTitle: string;
  anomalyType: string;
  severity: string;
}

export function FeedbackButtons({
  systemId,
  anomalyId,
  anomalyTitle,
  anomalyType,
  severity,
}: FeedbackButtonsProps) {
  const [submitted, setSubmitted] = useState<FeedbackType | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState('');
  const [loading, setLoading] = useState(false);
  const [existingFeedback, setExistingFeedback] = useState<AnomalyFeedback[]>([]);

  useEffect(() => {
    feedbackApi.list(systemId, anomalyId).then((entries) => {
      setExistingFeedback(entries);
      if (entries.length > 0) {
        setSubmitted(entries[0].feedback_type);
      }
    }).catch(() => {
      // Silently ignore — feedback is non-critical
    });
  }, [systemId, anomalyId]);

  const handleSubmit = async (type: FeedbackType) => {
    if (loading) return;
    setLoading(true);
    try {
      await feedbackApi.submit(systemId, {
        anomaly_id: anomalyId,
        anomaly_title: anomalyTitle,
        anomaly_type: anomalyType,
        severity,
        feedback_type: type,
        comment: comment || undefined,
      });
      setSubmitted(type);
      setShowComment(false);
      setComment('');
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    } finally {
      setLoading(false);
    }
  };

  const buttons: { type: FeedbackType; label: string; icon: React.ReactNode; color: string; activeColor: string }[] = [
    {
      type: 'relevant',
      label: 'Relevant',
      icon: <ThumbsUp className="w-3.5 h-3.5" />,
      color: 'hover:bg-green-500/20 hover:text-green-400 text-stone-400',
      activeColor: 'bg-green-500/20 text-green-400 border-green-500/50',
    },
    {
      type: 'false_positive',
      label: 'False Positive',
      icon: <ThumbsDown className="w-3.5 h-3.5" />,
      color: 'hover:bg-red-500/20 hover:text-red-400 text-stone-400',
      activeColor: 'bg-red-500/20 text-red-400 border-red-500/50',
    },
    {
      type: 'already_known',
      label: 'Already Known',
      icon: <Clock className="w-3.5 h-3.5" />,
      color: 'hover:bg-yellow-500/20 hover:text-yellow-400 text-stone-400',
      activeColor: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
    },
  ];

  return (
    <div className="mt-3 pt-3 border-t border-stone-600/50">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-stone-400 mr-1">Feedback:</span>
        {buttons.map((btn) => (
          <button
            key={btn.type}
            onClick={(e) => {
              e.stopPropagation();
              handleSubmit(btn.type);
            }}
            disabled={loading}
            className={clsx(
              'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-all',
              submitted === btn.type
                ? btn.activeColor
                : `border-stone-600 ${btn.color}`,
              loading && 'opacity-50 cursor-not-allowed'
            )}
          >
            {btn.icon}
            {btn.label}
          </button>
        ))}

        {/* Comment toggle */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowComment(!showComment);
          }}
          className="flex items-center gap-1 px-2 py-1 rounded-full text-xs text-stone-400 hover:text-stone-300 transition-colors"
        >
          <MessageSquare className="w-3.5 h-3.5" />
        </button>

        {existingFeedback.length > 0 && (
          <span className="text-xs text-stone-400 ml-auto">
            {existingFeedback.length} review{existingFeedback.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Comment input */}
      {showComment && (
        <div className="mt-2 flex gap-2" onClick={(e) => e.stopPropagation()}>
          <input
            type="text"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Add a comment (optional)..."
            className="flex-1 bg-stone-700/50 border border-stone-600 rounded-lg px-3 py-1.5 text-xs text-white placeholder-stone-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && submitted) {
                handleSubmit(submitted);
              }
            }}
          />
          <button
            onClick={() => setShowComment(false)}
            className="p-1.5 text-stone-400 hover:text-stone-300"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}

// ---------- Feedback Summary Banner ----------

interface FeedbackSummaryBannerProps {
  systemId: string;
}

export function FeedbackSummaryBanner({ systemId }: FeedbackSummaryBannerProps) {
  const [summary, setSummary] = useState<FeedbackSummary | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    feedbackApi.summary(systemId).then(setSummary).catch(() => {
      // Non-critical — silently ignore
    });
  }, [systemId]);

  if (!summary || summary.total_feedback === 0) return null;

  const { by_type, false_positive_rate, confidence_score, total_feedback } = summary;

  return (
    <div className="bg-stone-700 rounded-xl border border-stone-600 p-4 mb-6">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <TrendingUp className="w-5 h-5 text-primary-400" />
          <div>
            <h3 className="text-sm font-semibold text-white">Anomaly Feedback Loop</h3>
            <p className="text-xs text-stone-400">
              {total_feedback} reviews submitted
              {confidence_score !== null && (
                <> &middot; System confidence: <span className={clsx(
                  'font-medium',
                  confidence_score >= 0.8 ? 'text-green-400' :
                  confidence_score >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                )}>{(confidence_score * 100).toFixed(0)}%</span></>
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {/* Mini stats */}
          <div className="flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1 text-green-400">
              <ThumbsUp className="w-3 h-3" /> {by_type.relevant}
            </span>
            <span className="flex items-center gap-1 text-red-400">
              <ThumbsDown className="w-3 h-3" /> {by_type.false_positive}
            </span>
            <span className="flex items-center gap-1 text-yellow-400">
              <Clock className="w-3 h-3" /> {by_type.already_known}
            </span>
          </div>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-stone-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-stone-400" />
          )}
        </div>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-stone-600 space-y-3">
          {/* False positive rate bar */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-stone-400">False Positive Rate</span>
              <span className={clsx(
                'font-medium',
                false_positive_rate <= 0.1 ? 'text-green-400' :
                false_positive_rate <= 0.3 ? 'text-yellow-400' : 'text-red-400'
              )}>
                {(false_positive_rate * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 bg-stone-700 rounded-full overflow-hidden">
              <div
                className={clsx(
                  'h-full rounded-full transition-all',
                  false_positive_rate <= 0.1 ? 'bg-green-500' :
                  false_positive_rate <= 0.3 ? 'bg-yellow-500' : 'bg-red-500'
                )}
                style={{ width: `${Math.min(false_positive_rate * 100, 100)}%` }}
              />
            </div>
          </div>

          {/* False positive patterns */}
          {summary.false_positive_patterns.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-stone-400 mb-2">Detection Accuracy by Type</h4>
              <div className="space-y-1.5">
                {summary.false_positive_patterns.slice(0, 5).map((pattern, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <span className="text-stone-300 font-mono">
                      {pattern.anomaly_type.replace(/_/g, ' ')}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-stone-400">{pattern.total} reviews</span>
                      <span className={clsx(
                        'font-medium',
                        pattern.false_positive_rate <= 0.2 ? 'text-green-400' :
                        pattern.false_positive_rate <= 0.5 ? 'text-yellow-400' : 'text-red-400'
                      )}>
                        {((1 - pattern.false_positive_rate) * 100).toFixed(0)}% accurate
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Confidence improvement hint */}
          {confidence_score !== null && confidence_score < 0.8 && (
            <div className="flex items-start gap-2 p-2 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
              <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-stone-400">
                Continue reviewing anomalies to improve detection accuracy.
                The system adapts its thresholds based on your feedback.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
