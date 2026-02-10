"""
Root Cause Reasoning Service

Provides cross-domain correlation and natural language explanations
for detected anomalies and issues.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class RootCauseAnalysis:
    """Result of root cause analysis."""
    primary_cause: str
    confidence: float
    contributing_factors: List[str]
    evidence: List[Dict]
    natural_language_explanation: str
    recommendations: List[Dict]
    correlated_events: List[Dict]


@dataclass 
class CorrelatedEvent:
    """An event that correlates with the anomaly."""
    event_type: str
    timestamp: str
    description: str
    correlation_strength: float
    time_offset_hours: float


class RootCauseService:
    """
    Analyzes anomalies to determine root causes using cross-domain
    correlation and generates natural language explanations.
    """

    def __init__(self):
        self.event_types = [
            "firmware_update",
            "configuration_change",
            "environmental_change",
            "maintenance_event",
            "software_deployment",
            "operational_change",
        ]

    async def analyze(
        self,
        anomaly: Dict,
        system_data: pd.DataFrame,
        events: List[Dict],
        historical_anomalies: List[Dict],
        system_context: Dict
    ) -> RootCauseAnalysis:
        """
        Perform comprehensive root cause analysis for an anomaly.
        
        Args:
            anomaly: The anomaly to analyze
            system_data: Recent telemetry data
            events: System events (updates, changes, maintenance)
            historical_anomalies: Past anomalies for pattern matching
            system_context: Context about the system (type, specs, etc.)
        
        Returns:
            RootCauseAnalysis with explanation and recommendations
        """
        # 1. Find correlated events
        correlated_events = await self._find_correlated_events(
            anomaly, events, system_data
        )

        # 2. Analyze data patterns
        data_patterns = await self._analyze_data_patterns(
            anomaly, system_data
        )

        # 3. Find similar historical incidents
        similar_incidents = await self._find_similar_incidents(
            anomaly, historical_anomalies
        )

        # 4. Generate hypothesis
        hypothesis = await self._generate_hypothesis(
            anomaly, correlated_events, data_patterns, similar_incidents
        )

        # 5. Generate natural language explanation
        explanation = await self._generate_explanation(
            anomaly, hypothesis, correlated_events, system_context
        )

        # 6. Generate recommendations
        recommendations = await self._generate_recommendations(
            anomaly, hypothesis, correlated_events
        )

        return RootCauseAnalysis(
            primary_cause=hypothesis["primary_cause"],
            confidence=hypothesis["confidence"],
            contributing_factors=hypothesis["contributing_factors"],
            evidence=hypothesis["evidence"],
            natural_language_explanation=explanation,
            recommendations=recommendations,
            correlated_events=[e.__dict__ for e in correlated_events],
        )

    async def _find_correlated_events(
        self,
        anomaly: Dict,
        events: List[Dict],
        system_data: pd.DataFrame
    ) -> List[CorrelatedEvent]:
        """Find events that correlate temporally with the anomaly."""
        correlated = []
        anomaly_time = datetime.fromisoformat(anomaly.get("timestamp", datetime.utcnow().isoformat()))

        for event in events:
            event_time = datetime.fromisoformat(event.get("timestamp", ""))
            time_diff = (anomaly_time - event_time).total_seconds() / 3600  # hours

            # Look for events within 72 hours before the anomaly
            if 0 < time_diff <= 72:
                # Calculate correlation strength based on time proximity
                correlation_strength = max(0, 1 - (time_diff / 72))

                correlated.append(CorrelatedEvent(
                    event_type=event.get("type", "unknown"),
                    timestamp=event.get("timestamp"),
                    description=event.get("description", ""),
                    correlation_strength=correlation_strength,
                    time_offset_hours=time_diff,
                ))

        # Sort by correlation strength
        return sorted(correlated, key=lambda x: x.correlation_strength, reverse=True)

    async def _analyze_data_patterns(
        self,
        anomaly: Dict,
        system_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze data patterns around the anomaly."""
        patterns = {
            "trend_before": None,
            "sudden_change": False,
            "cyclical_pattern": False,
            "correlation_changes": [],
        }

        affected_fields = anomaly.get("affected_fields", [])
        
        for field in affected_fields:
            if field not in system_data.columns:
                continue

            values = system_data[field].dropna()
            if len(values) < 10:
                continue

            # Detect trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            patterns["trend_before"] = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

            # Detect sudden changes
            diff = values.diff().abs()
            threshold = diff.mean() + 3 * diff.std()
            patterns["sudden_change"] = any(diff > threshold)

            # Detect cyclical patterns (simplified)
            if len(values) > 20:
                fft = np.fft.fft(values - values.mean())
                power = np.abs(fft) ** 2
                patterns["cyclical_pattern"] = any(power[1:len(power)//2] > power[0] * 0.5)

        return patterns

    async def _find_similar_incidents(
        self,
        anomaly: Dict,
        historical_anomalies: List[Dict]
    ) -> List[Dict]:
        """Find similar past anomalies for pattern matching."""
        similar = []
        current_type = anomaly.get("anomaly_type")
        current_fields = set(anomaly.get("affected_fields", []))

        for past in historical_anomalies:
            # Calculate similarity score
            similarity = 0
            
            if past.get("anomaly_type") == current_type:
                similarity += 0.4

            past_fields = set(past.get("affected_fields", []))
            field_overlap = len(current_fields & past_fields) / max(len(current_fields | past_fields), 1)
            similarity += field_overlap * 0.4

            if past.get("severity") == anomaly.get("severity"):
                similarity += 0.2

            if similarity > 0.5:
                similar.append({
                    "anomaly": past,
                    "similarity_score": similarity,
                    "resolution": past.get("resolution"),
                })

        return sorted(similar, key=lambda x: x["similarity_score"], reverse=True)[:5]

    async def _generate_hypothesis(
        self,
        anomaly: Dict,
        correlated_events: List[CorrelatedEvent],
        data_patterns: Dict,
        similar_incidents: List[Dict]
    ) -> Dict[str, Any]:
        """Generate a hypothesis for the root cause."""
        hypothesis = {
            "primary_cause": "Unknown - requires investigation",
            "confidence": 0.3,
            "contributing_factors": [],
            "evidence": [],
        }

        # Check for firmware/software correlation
        software_events = [e for e in correlated_events 
                         if e.event_type in ["firmware_update", "software_deployment", "configuration_change"]]
        
        if software_events and software_events[0].correlation_strength > 0.7:
            event = software_events[0]
            hypothesis["primary_cause"] = f"Software change: {event.description}"
            hypothesis["confidence"] = event.correlation_strength
            hypothesis["evidence"].append({
                "type": "temporal_correlation",
                "description": f"{event.event_type} occurred {event.time_offset_hours:.1f} hours before anomaly",
            })

        # Check for environmental changes
        env_events = [e for e in correlated_events if e.event_type == "environmental_change"]
        if env_events:
            hypothesis["contributing_factors"].append("Environmental conditions changed")

        # Check for maintenance correlation
        maintenance_events = [e for e in correlated_events if e.event_type == "maintenance_event"]
        if maintenance_events:
            hypothesis["contributing_factors"].append("Recent maintenance may be related")

        # Use similar incidents to boost confidence
        if similar_incidents and similar_incidents[0].get("resolution"):
            hypothesis["evidence"].append({
                "type": "historical_pattern",
                "description": f"Similar to past incident with resolution: {similar_incidents[0]['resolution']}",
            })
            hypothesis["confidence"] = min(hypothesis["confidence"] + 0.2, 0.95)

        # Use data patterns
        if data_patterns.get("sudden_change"):
            hypothesis["evidence"].append({
                "type": "data_pattern",
                "description": "Sudden change detected in affected metrics",
            })

        return hypothesis

    async def _generate_explanation(
        self,
        anomaly: Dict,
        hypothesis: Dict,
        correlated_events: List[CorrelatedEvent],
        system_context: Dict
    ) -> str:
        """
        Generate a natural language explanation.
        This is the "Why" that replaces cryptic error codes.
        """
        affected = ", ".join(anomaly.get("affected_fields", ["unknown parameters"]))
        severity = anomaly.get("severity", "unknown")
        anomaly_type = anomaly.get("anomaly_type", "anomaly")

        # Build the explanation
        explanation_parts = []

        # Opening - what was detected
        explanation_parts.append(
            f"A {severity}-severity {anomaly_type.replace('_', ' ')} was detected affecting {affected}."
        )

        # What's happening
        if anomaly.get("description"):
            explanation_parts.append(anomaly["description"])

        # The cause
        if hypothesis["confidence"] > 0.6:
            explanation_parts.append(
                f"Analysis indicates the most likely cause is: {hypothesis['primary_cause']}."
            )
        elif hypothesis["confidence"] > 0.4:
            explanation_parts.append(
                f"A possible cause is: {hypothesis['primary_cause']}, though further investigation is recommended."
            )

        # Contributing factors
        if hypothesis["contributing_factors"]:
            factors = ", ".join(hypothesis["contributing_factors"])
            explanation_parts.append(f"Contributing factors include: {factors}.")

        # Evidence
        if hypothesis["evidence"]:
            explanation_parts.append("Supporting evidence:")
            for evidence in hypothesis["evidence"]:
                explanation_parts.append(f"  - {evidence['description']}")

        # Correlated events
        if correlated_events:
            top_event = correlated_events[0]
            explanation_parts.append(
                f"Note: A {top_event.event_type.replace('_', ' ')} occurred "
                f"{top_event.time_offset_hours:.1f} hours before this anomaly was detected."
            )

        return "\n".join(explanation_parts)

    async def _generate_recommendations(
        self,
        anomaly: Dict,
        hypothesis: Dict,
        correlated_events: List[CorrelatedEvent]
    ) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on anomaly type
        anomaly_type = anomaly.get("anomaly_type", "")

        if "software" in hypothesis["primary_cause"].lower() or \
           any(e.event_type in ["firmware_update", "software_deployment"] for e in correlated_events):
            recommendations.append({
                "type": "software_rollback",
                "priority": "high",
                "action": "Consider rolling back to the previous software version",
                "rationale": "Recent software changes correlate with this anomaly",
            })

        if anomaly_type == "behavioral_deviation":
            recommendations.append({
                "type": "investigation",
                "priority": "medium",
                "action": "Review operational conditions that may have changed",
                "rationale": "System behavior has shifted from baseline",
            })

        if anomaly_type == "margin_warning":
            recommendations.append({
                "type": "hardware_review",
                "priority": "high",
                "action": "Inspect affected components for wear or degradation",
                "rationale": "Engineering margins are approaching limits",
            })

        # General recommendations
        recommendations.append({
            "type": "monitoring",
            "priority": "medium",
            "action": "Increase monitoring frequency on affected parameters",
            "rationale": "Early detection of recurrence or escalation",
        })

        if not recommendations:
            recommendations.append({
                "type": "investigation",
                "priority": "medium",
                "action": "Manual investigation required",
                "rationale": "Automated analysis could not determine definitive cause",
            })

        return recommendations

    async def explain_in_context(
        self,
        query: str,
        system_data: pd.DataFrame,
        anomalies: List[Dict],
        context: Dict
    ) -> str:
        """
        Answer an engineer's question about the data in natural language.
        This powers the "Conversational Chief Engineer" interface.
        """
        query_lower = query.lower()

        # Parse query intent
        if "why" in query_lower:
            return await self._explain_why(query, anomalies, context)
        elif "show" in query_lower or "find" in query_lower:
            return await self._query_data(query, system_data, context)
        elif "compare" in query_lower:
            return await self._compare_data(query, system_data, context)
        elif "trend" in query_lower or "predict" in query_lower:
            return await self._analyze_trend(query, system_data, context)
        else:
            return await self._general_query(query, system_data, anomalies, context)

    async def _explain_why(self, query: str, anomalies: List[Dict], context: Dict) -> str:
        """Explain why something is happening."""
        if not anomalies:
            return "No anomalies are currently detected that would explain unusual behavior."
        
        explanations = []
        for anomaly in anomalies[:3]:
            explanation = anomaly.get("natural_language_explanation", anomaly.get("description", ""))
            explanations.append(f"- {anomaly.get('title', 'Issue')}: {explanation}")
        
        return "Based on current analysis:\n" + "\n".join(explanations)

    async def _query_data(self, query: str, data: pd.DataFrame, context: Dict) -> str:
        """Query and filter data based on natural language request."""
        # This would integrate with a more sophisticated NL-to-query system
        return f"Query parsed. Found {len(data)} records matching your criteria."

    async def _compare_data(self, query: str, data: pd.DataFrame, context: Dict) -> str:
        """Compare data across time periods or cohorts."""
        return "Comparison analysis would be generated here."

    async def _analyze_trend(self, query: str, data: pd.DataFrame, context: Dict) -> str:
        """Analyze and predict trends."""
        return "Trend analysis would be generated here."

    async def _general_query(
        self, 
        query: str, 
        data: pd.DataFrame, 
        anomalies: List[Dict],
        context: Dict
    ) -> str:
        """Handle general queries about the system."""
        return f"System overview: {len(data)} data points, {len(anomalies)} active anomalies."
