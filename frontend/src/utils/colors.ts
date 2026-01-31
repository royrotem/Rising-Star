/**
 * Shared color utility functions for UAIE frontend.
 *
 * Consolidates severity / status / health color helpers that were
 * duplicated across Dashboard, Systems, SystemDetail, and AnomalyExplorer.
 */

/** Severity → Tailwind border + bg classes (card variant). */
export function getSeverityCardColor(severity: string) {
  switch (severity) {
    case 'critical': return 'border-red-500 bg-red-500/10';
    case 'high': return 'border-orange-500 bg-orange-500/10';
    case 'medium': return 'border-yellow-500 bg-yellow-500/10';
    case 'low': return 'border-emerald-500 bg-emerald-500/10';
    default: return 'border-stone-500 bg-stone-500/10';
  }
}

/** Severity → Tailwind text + bg + border classes (badge variant used on Dashboard). */
export function getSeverityBadgeColor(severity: string) {
  switch (severity) {
    case 'critical': return 'text-red-400 bg-red-500/10 border border-red-500/20';
    case 'high': return 'text-orange-400 bg-orange-500/10 border border-orange-500/20';
    case 'medium': return 'text-yellow-400 bg-yellow-500/10 border border-yellow-500/20';
    case 'low': return 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20';
    default: return 'text-stone-400 bg-stone-500/10 border border-stone-500/20';
  }
}

/** Severity → solid dot color. */
export function getSeverityDotColor(severity: string) {
  switch (severity) {
    case 'critical': return 'bg-red-500';
    case 'high': return 'bg-orange-500';
    case 'medium': return 'bg-yellow-500';
    case 'low': return 'bg-emerald-500';
    default: return 'bg-stone-500';
  }
}

/** Severity → small badge variant (AnomalyExplorer). */
export function getSeveritySmallBadge(severity: string) {
  switch (severity) {
    case 'critical': return 'bg-red-500/15 text-red-400 border-red-500/30';
    case 'high': return 'bg-orange-500/15 text-orange-400 border-orange-500/30';
    case 'medium': return 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30';
    case 'low': return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';
    default: return 'bg-stone-500/15 text-stone-400 border-stone-500/30';
  }
}

/** System status → dot / indicator bg color. */
export function getStatusColor(status: string) {
  switch (status) {
    case 'active': return 'bg-emerald-400';
    case 'healthy': return 'bg-emerald-400';
    case 'anomaly_detected': return 'bg-orange-400';
    case 'maintenance': return 'bg-yellow-400';
    case 'inactive': return 'bg-stone-500';
    case 'data_ingested': return 'bg-blue-500';
    case 'configured': return 'bg-green-500';
    default: return 'bg-stone-500';
  }
}

/** Health score → text color class. */
export function getHealthColor(score: number | undefined | null) {
  if (score === undefined || score === null) return 'text-stone-500';
  if (score >= 90) return 'text-emerald-400';
  if (score >= 70) return 'text-yellow-400';
  return 'text-red-400';
}
