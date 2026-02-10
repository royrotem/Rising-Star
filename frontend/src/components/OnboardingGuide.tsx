import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ChevronRight,
  X,
  CheckCircle2,
} from 'lucide-react';
import clsx from 'clsx';

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  action?: () => void;
  actionLabel?: string;
  checkComplete: () => boolean;
}

interface OnboardingGuideProps {
  systemCount: number;
  hasAnalyzed?: boolean;
}

const DISMISSED_KEY = 'uaie_onboarding_dismissed';

export default function OnboardingGuide({ systemCount, hasAnalyzed }: OnboardingGuideProps) {
  const navigate = useNavigate();
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(DISMISSED_KEY);
    if (stored === 'true') setDismissed(true);
  }, []);

  const steps: OnboardingStep[] = [
    {
      id: 'create-system',
      title: 'Create a System',
      description: 'Define a system to monitor with telemetry data.',
      action: () => navigate('/systems/new'),
      actionLabel: 'Create',
      checkComplete: () => systemCount > 0,
    },
    {
      id: 'run-analysis',
      title: 'Run AI Analysis',
      description: '13 AI agents analyze your data for anomalies and patterns.',
      actionLabel: 'Go',
      action: () => { if (systemCount > 0) navigate('/systems'); },
      checkComplete: () => !!hasAnalyzed,
    },
    {
      id: 'explore',
      title: 'Explore Results',
      description: 'Drill into anomalies, chat with AI, download PDF reports.',
      actionLabel: 'Explore',
      action: () => { if (systemCount > 0) navigate('/systems'); },
      checkComplete: () => false,
    },
  ];

  const completedCount = steps.filter(s => s.checkComplete()).length;
  const allDone = completedCount === steps.length;

  if (dismissed || allDone) return null;

  const handleDismiss = () => {
    setDismissed(true);
    localStorage.setItem(DISMISSED_KEY, 'true');
  };

  return (
    <div className="mb-10 glass-card overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 flex items-center justify-between border-b border-stone-600/40">
        <div>
          <h2 className="text-sm font-medium text-white">Getting Started</h2>
          <p className="text-xs text-stone-400 mt-0.5">
            {completedCount}/{steps.length} steps completed
          </p>
        </div>
        <button
          onClick={handleDismiss}
          className="p-1.5 hover:bg-stone-700 rounded-lg transition-colors text-stone-400 hover:text-stone-300"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Steps */}
      <div className="grid grid-cols-3 divide-x divide-stone-600/40">
        {steps.map((step, idx) => {
          const done = step.checkComplete();
          return (
            <div key={step.id} className="p-5">
              <div className="flex items-center gap-2 mb-2">
                {done ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <span className="text-xs text-stone-400 font-medium">{idx + 1}</span>
                )}
                <h3 className={clsx(
                  'text-sm font-medium',
                  done ? 'text-emerald-400' : 'text-white'
                )}>
                  {step.title}
                </h3>
              </div>
              <p className="text-xs text-stone-400 leading-relaxed mb-3">
                {step.description}
              </p>
              {!done && step.action && (
                <button
                  onClick={step.action}
                  className="flex items-center gap-1 text-xs text-primary-400 hover:text-primary-300 font-medium transition-colors"
                >
                  {step.actionLabel}
                  <ChevronRight className="w-3 h-3" />
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
