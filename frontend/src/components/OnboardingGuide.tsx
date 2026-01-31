import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Rocket,
  Upload,
  Search,
  MessageSquare,
  FileText,
  CheckCircle2,
  Circle,
  ChevronRight,
  X,
  Sparkles,
  History,
} from 'lucide-react';
import clsx from 'clsx';

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
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
      title: 'Create Your First System',
      description: 'Define a system to monitor — vehicle, robot, medical device, or any engineered system.',
      icon: <Rocket className="w-5 h-5" />,
      action: () => navigate('/systems/new'),
      actionLabel: 'Create System',
      checkComplete: () => systemCount > 0,
    },
    {
      id: 'upload-data',
      title: 'Upload Telemetry Data',
      description: 'Upload CSV or JSON data. The AI will automatically discover your data schema and field meanings.',
      icon: <Upload className="w-5 h-5" />,
      action: () => navigate('/systems/new'),
      actionLabel: 'Upload Data',
      checkComplete: () => systemCount > 0, // Simplified — system creation includes data
    },
    {
      id: 'run-analysis',
      title: 'Run AI Analysis',
      description: 'The engine runs 6 detection layers + 5 AI agents to find anomalies, margins, and blind spots.',
      icon: <Search className="w-5 h-5" />,
      actionLabel: 'Go to System',
      action: () => { if (systemCount > 0) navigate('/systems'); },
      checkComplete: () => !!hasAnalyzed,
    },
    {
      id: 'chat-ai',
      title: 'Ask the AI Chat',
      description: 'Ask questions about your data in natural language — anomalies, correlations, recommendations.',
      icon: <MessageSquare className="w-5 h-5" />,
      actionLabel: 'Open Chat',
      action: () => { if (systemCount > 0) navigate('/systems'); },
      checkComplete: () => false, // User must discover this
    },
    {
      id: 'capture-baseline',
      title: 'Capture a Baseline Snapshot',
      description: 'Save periodic snapshots to track changes over time and detect deviations from normal behavior.',
      icon: <History className="w-5 h-5" />,
      actionLabel: 'View Systems',
      action: () => navigate('/systems'),
      checkComplete: () => false,
    },
    {
      id: 'download-report',
      title: 'Download a PDF Report',
      description: 'Generate a professional PDF report with health scores, anomalies, margins, and recommendations.',
      icon: <FileText className="w-5 h-5" />,
      actionLabel: 'View Systems',
      action: () => navigate('/systems'),
      checkComplete: () => false,
    },
  ];

  const completedCount = steps.filter(s => s.checkComplete()).length;
  const allDone = completedCount === steps.length;
  const progress = (completedCount / steps.length) * 100;

  if (dismissed || allDone) return null;

  const handleDismiss = () => {
    setDismissed(true);
    localStorage.setItem(DISMISSED_KEY, 'true');
  };

  return (
    <div className="mb-8 bg-gradient-to-br from-primary-500/5 via-stone-800/50 to-accent-500/5 border border-primary-500/20 rounded-2xl overflow-hidden animate-fadeIn">
      {/* Header */}
      <div className="px-6 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-primary-500/10 rounded-xl">
            <Sparkles className="w-6 h-6 text-primary-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Welcome to UAIE</h2>
            <p className="text-sm text-stone-400">
              Follow these steps to get the most out of the platform
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {/* Progress */}
          <div className="flex items-center gap-2">
            <div className="w-24 h-2 bg-stone-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary-500 to-accent-500 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="text-xs text-stone-400">{completedCount}/{steps.length}</span>
          </div>
          <button
            onClick={handleDismiss}
            className="p-1.5 hover:bg-stone-700 rounded-lg transition-colors text-stone-500 hover:text-stone-300"
            title="Dismiss guide"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Steps */}
      <div className="px-6 pb-5 grid grid-cols-3 gap-3">
        {steps.map((step, idx) => {
          const done = step.checkComplete();
          return (
            <div
              key={step.id}
              className={clsx(
                'p-4 rounded-xl border transition-all',
                done
                  ? 'bg-accent-500/5 border-accent-500/20'
                  : 'bg-stone-800/60 border-stone-700/50 hover:border-primary-500/30'
              )}
            >
              <div className="flex items-start gap-3">
                <div className={clsx(
                  'p-2 rounded-lg flex-shrink-0',
                  done ? 'bg-accent-500/10 text-accent-400' : 'bg-stone-700/50 text-stone-400'
                )}>
                  {done ? <CheckCircle2 className="w-5 h-5" /> : step.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-stone-500 font-medium">Step {idx + 1}</span>
                    {done && (
                      <span className="text-xs text-accent-400 font-medium">Done</span>
                    )}
                  </div>
                  <h3 className={clsx(
                    'text-sm font-semibold mt-0.5',
                    done ? 'text-accent-300' : 'text-white'
                  )}>
                    {step.title}
                  </h3>
                  <p className="text-xs text-stone-500 mt-1 leading-relaxed">
                    {step.description}
                  </p>
                  {!done && step.action && (
                    <button
                      onClick={step.action}
                      className="flex items-center gap-1 mt-2 text-xs text-primary-400 hover:text-primary-300 font-medium transition-colors"
                    >
                      {step.actionLabel}
                      <ChevronRight className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
