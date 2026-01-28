import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft,
  Upload,
  MessageSquare,
  Activity,
  AlertTriangle,
  TrendingDown,
  Lightbulb,
  ChevronRight,
  RefreshCw
} from 'lucide-react';
import clsx from 'clsx';

// Mock data
const mockAnalysis = {
  health_score: 87.5,
  anomalies: [
    {
      id: '1',
      type: 'behavioral_deviation',
      severity: 'medium',
      title: 'Motor current draw increased',
      description: 'Motor A is drawing 12% more current than baseline under similar load conditions.',
      natural_language_explanation: 
        'Motor A is consuming more power than expected. This started 3 days ago, ' +
        'coinciding with firmware update v2.3.1. The increased current draw is consistent ' +
        'across all operating conditions, suggesting a software-related cause rather than ' +
        'mechanical wear.',
      recommendations: [
        {
          type: 'investigation',
          priority: 'high',
          action: 'Review firmware v2.3.1 changes to motor control parameters',
        },
      ],
      impact_score: 72.5,
    },
  ],
  engineering_margins: [
    {
      component: 'Battery Pack',
      parameter: 'max_temperature',
      current_value: 38.5,
      design_limit: 45.0,
      margin_percentage: 14.4,
      trend: 'stable',
      safety_critical: true,
    },
    {
      component: 'Motor A',
      parameter: 'max_current',
      current_value: 28.5,
      design_limit: 35.0,
      margin_percentage: 18.6,
      trend: 'degrading',
      safety_critical: false,
    },
  ],
  blind_spots: [
    {
      title: 'Missing vibration data',
      description: 'We cannot fully diagnose the recurring motor anomalies because we lack high-frequency vibration data.',
      recommended_sensor: {
        type: 'Accelerometer',
        specification: '3-axis, 1kHz sampling',
        estimated_cost: 150,
      },
      diagnostic_coverage_improvement: 25,
    },
  ],
};

function getSeverityColor(severity: string) {
  switch (severity) {
    case 'critical': return 'border-red-500 bg-red-500/10';
    case 'high': return 'border-orange-500 bg-orange-500/10';
    case 'medium': return 'border-yellow-500 bg-yellow-500/10';
    case 'low': return 'border-green-500 bg-green-500/10';
    default: return 'border-slate-500 bg-slate-500/10';
  }
}

function getTrendIcon(trend: string) {
  if (trend === 'degrading') return <TrendingDown className="w-4 h-4 text-red-500" />;
  if (trend === 'improving') return <TrendingDown className="w-4 h-4 text-green-500 rotate-180" />;
  return <Activity className="w-4 h-4 text-slate-500" />;
}

export default function SystemDetail() {
  const { systemId } = useParams();
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedAnomaly, setSelectedAnomaly] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setAnalyzing(true);
    setTimeout(() => setAnalyzing(false), 2000);
  };

  const getProgressWidth = (marginPct: number) => {
    return (100 - marginPct) + '%';
  };

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link to="/" className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
          <ArrowLeft className="w-5 h-5 text-slate-400" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">Fleet Vehicle Alpha</h1>
          <p className="text-slate-400">System ID: {systemId}</p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            to={'/systems/' + systemId + '/ingest'}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
          >
            <Upload className="w-4 h-4" />
            Ingest Data
          </Link>
          <Link
            to={'/systems/' + systemId + '/chat'}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
          >
            <MessageSquare className="w-4 h-4" />
            Ask AI
          </Link>
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white rounded-lg font-medium transition-colors"
          >
            <RefreshCw className={clsx("w-4 h-4", analyzing && "animate-spin")} />
            {analyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {/* Health Score */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-white mb-1">System Health</h2>
            <p className="text-slate-400 text-sm">Overall system health score based on all parameters</p>
          </div>
          <div className="text-right">
            <div className={clsx(
              'text-5xl font-bold',
              mockAnalysis.health_score >= 90 ? 'text-green-500' :
              mockAnalysis.health_score >= 70 ? 'text-yellow-500' : 'text-red-500'
            )}>
              {mockAnalysis.health_score}%
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Anomalies */}
        <div className="bg-slate-800 rounded-xl border border-slate-700">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-white">Detected Anomalies</h2>
          </div>
          <div className="p-4 space-y-4">
            {mockAnalysis.anomalies.map((anomaly) => (
              <div
                key={anomaly.id}
                className={clsx(
                  'p-4 rounded-lg border-l-4 cursor-pointer transition-colors',
                  getSeverityColor(anomaly.severity),
                  selectedAnomaly === anomaly.id ? 'ring-2 ring-primary-500' : ''
                )}
                onClick={() => setSelectedAnomaly(
                  selectedAnomaly === anomaly.id ? null : anomaly.id
                )}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-medium text-white">{anomaly.title}</h3>
                  <span className="text-sm font-medium text-slate-400">
                    Impact: {anomaly.impact_score}
                  </span>
                </div>
                <p className="text-sm text-slate-300 mb-3">{anomaly.description}</p>
                
                {selectedAnomaly === anomaly.id && (
                  <div className="mt-4 pt-4 border-t border-slate-600">
                    <h4 className="text-sm font-medium text-primary-400 mb-2">AI Analysis</h4>
                    <p className="text-sm text-slate-300 mb-4">{anomaly.natural_language_explanation}</p>
                    <h4 className="text-sm font-medium text-primary-400 mb-2">Recommendations</h4>
                    {anomaly.recommendations.map((rec, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-sm">
                        <ChevronRight className="w-4 h-4 text-primary-500" />
                        <span className="text-slate-300">{rec.action}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Engineering Margins */}
        <div className="bg-slate-800 rounded-xl border border-slate-700">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary-500" />
            <h2 className="text-lg font-semibold text-white">Engineering Margins</h2>
          </div>
          <div className="p-4 space-y-4">
            {mockAnalysis.engineering_margins.map((margin, idx) => (
              <div key={idx} className="p-4 bg-slate-900/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h3 className="font-medium text-white">{margin.component}</h3>
                    <p className="text-sm text-slate-400">{margin.parameter}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {getTrendIcon(margin.trend)}
                    {margin.safety_critical && (
                      <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded">
                        Safety Critical
                      </span>
                    )}
                  </div>
                </div>
                <div className="mt-3">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400">Current: {margin.current_value}</span>
                    <span className="text-slate-400">Limit: {margin.design_limit}</span>
                  </div>
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={clsx(
                        'h-full rounded-full',
                        margin.margin_percentage < 15 ? 'bg-red-500' :
                        margin.margin_percentage < 30 ? 'bg-yellow-500' : 'bg-green-500'
                      )}
                      style={{ width: getProgressWidth(margin.margin_percentage) }}
                    />
                  </div>
                  <p className="text-sm text-slate-400 mt-1">
                    {margin.margin_percentage.toFixed(1)}% margin remaining
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Blind Spots */}
        <div className="col-span-2 bg-slate-800 rounded-xl border border-slate-700">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-500" />
            <h2 className="text-lg font-semibold text-white">Blind Spots & Next-Gen Recommendations</h2>
          </div>
          <div className="p-4 grid grid-cols-2 gap-4">
            {mockAnalysis.blind_spots.map((spot, idx) => (
              <div key={idx} className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                <h3 className="font-medium text-white mb-2">{spot.title}</h3>
                <p className="text-sm text-slate-300 mb-4">{spot.description}</p>
                <div className="p-3 bg-slate-800 rounded-lg">
                  <h4 className="text-sm font-medium text-primary-400 mb-2">Recommended Sensor</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <span className="text-slate-400">Type:</span>
                    <span className="text-white">{spot.recommended_sensor.type}</span>
                    <span className="text-slate-400">Spec:</span>
                    <span className="text-white">{spot.recommended_sensor.specification}</span>
                    <span className="text-slate-400">Cost:</span>
                    <span className="text-white">${spot.recommended_sensor.estimated_cost}</span>
                    <span className="text-slate-400">Coverage Gain:</span>
                    <span className="text-green-400">+{spot.diagnostic_coverage_improvement}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
