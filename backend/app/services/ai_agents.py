"""
Multi-Agent AI Analysis System for UAIE

Multiple AI agents analyze data from different perspectives,
then results are unified into comprehensive anomaly reports.

Agents:
 1. Statistical Analyst        – numbers, distributions, outliers
 2. Domain Expert              – engineering domain knowledge
 3. Pattern Detective          – hidden patterns and correlations
 4. Root Cause Investigator    – deep-thinking causal reasoning
 5. Safety Auditor             – safety margins and risks
 6. Temporal Analyst           – time-series, seasonality, change-points
 7. Data Quality Inspector     – data integrity, sensor drift, corruption
 8. Predictive Forecaster      – trend extrapolation, failure prediction
 9. Operational Profiler       – operating modes, regime transitions
10. Efficiency Analyst         – energy/resource waste, optimization
11. Compliance Checker         – regulatory limits, industry standards
12. Reliability Engineer       – MTBF, degradation, wear-out patterns
13. Environmental Correlator   – cross-parameter environmental effects

Each agent can also ground its analysis with web search for
real-world engineering context.
"""

import asyncio
import json
import os
import re
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Per-agent timeout in seconds — prevents any single agent from running forever
AGENT_TIMEOUT = 90
# Global orchestrator timeout — total wall-clock limit for all agents combined
ORCHESTRATOR_TIMEOUT = 300

# All available agent names (used for selection validation)
ALL_AGENT_NAMES = [
    "Statistical Analyst",
    "Domain Expert",
    "Pattern Detective",
    "Root Cause Investigator",
    "Safety Auditor",
    "Temporal Analyst",
    "Data Quality Inspector",
    "Predictive Forecaster",
    "Operational Profiler",
    "Efficiency Analyst",
    "Compliance Checker",
    "Reliability Engineer",
    "Environmental Correlator",
]


def _get_api_key() -> str:
    """Get the Anthropic API key from app settings (with env fallback)."""
    try:
        from ..api.app_settings import get_anthropic_api_key
        return get_anthropic_api_key()
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


@dataclass
class AgentFinding:
    """A single finding from an AI agent."""
    agent_name: str
    anomaly_type: str
    severity: str  # critical, high, medium, low, info
    title: str
    description: str
    natural_language_explanation: str
    possible_causes: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, str]] = field(default_factory=list)
    affected_fields: List[str] = field(default_factory=list)
    confidence: float = 0.0
    impact_score: float = 0.0
    web_references: List[str] = field(default_factory=list)
    raw_reasoning: str = ""


@dataclass
class UnifiedAnomaly:
    """An anomaly that aggregates findings from multiple agents."""
    id: str
    type: str
    severity: str
    title: str
    description: str
    natural_language_explanation: str
    possible_causes: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, str]] = field(default_factory=list)
    affected_fields: List[str] = field(default_factory=list)
    confidence: float = 0.0
    impact_score: float = 0.0
    contributing_agents: List[str] = field(default_factory=list)
    web_references: List[str] = field(default_factory=list)
    agent_perspectives: List[Dict[str, str]] = field(default_factory=list)


# ─────────────────────── web search helper ───────────────────────

async def web_search(query: str) -> List[Dict[str, str]]:
    """Search the web for engineering context.  Returns list of {title, snippet, url}."""
    if not HAS_HTTPX:
        return []

    try:
        # Use DuckDuckGo HTML for a lightweight, key-free search
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; UAIE-Bot/1.0)"},
            )
            if resp.status_code != 200:
                return []

            text = resp.text
            results = []
            # Parse simple result blocks
            for block in re.findall(
                r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'<a class="result__snippet"[^>]*>(.*?)</a>',
                text, re.DOTALL,
            )[:5]:
                url, title, snippet = block
                title = re.sub(r"<.*?>", "", title).strip()
                snippet = re.sub(r"<.*?>", "", snippet).strip()
                if title:
                    results.append({"title": title, "snippet": snippet, "url": url})

            return results
    except Exception:
        return []


# ─────────────────────── base agent ───────────────────────

class BaseAgent:
    """Base class for all AI analysis agents."""

    name: str = "base"
    perspective: str = ""
    model: str = "claude-sonnet-4-20250514"

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize or refresh the async Anthropic client using the current API key."""
        api_key = _get_api_key()
        if HAS_ANTHROPIC and api_key:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            self.client = None

    def _build_data_summary(self, data_profile: Dict) -> str:
        """Build a concise data summary for the prompt."""
        lines = [
            f"Records: {data_profile.get('record_count', 0)}",
            f"Fields ({data_profile.get('field_count', 0)}):",
        ]
        for f in data_profile.get("fields", [])[:30]:
            stats = ""
            if f.get("mean") is not None:
                stats = f" | mean={f['mean']:.4g}  std={f.get('std', 0):.4g}  min={f.get('min', 0):.4g}  max={f.get('max', 0):.4g}"
            lines.append(
                f"  - {f['name']} ({f.get('type','?')}){stats}"
            )

        if data_profile.get("sample_rows"):
            lines.append("\nSample rows (first 5):")
            for row in data_profile["sample_rows"][:5]:
                lines.append(f"  {json.dumps(row, default=str)[:300]}")

        if data_profile.get("correlations"):
            lines.append("\nTop correlations:")
            for pair, val in list(data_profile["correlations"].items())[:10]:
                lines.append(f"  {pair}: {val:.3f}")

        return "\n".join(lines)

    async def analyze(self, system_type: str, system_name: str,
                      data_profile: Dict, metadata_context: str = "") -> List[AgentFinding]:
        """Run this agent's analysis.  Falls back to rule-based if no API key."""
        self._init_client()
        if not self.client:
            return self._fallback_analyze(system_type, data_profile)

        data_summary = self._build_data_summary(data_profile)
        prompt = self._build_prompt(system_type, system_name, data_summary, metadata_context)

        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self._system_prompt(system_type),
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=AGENT_TIMEOUT,
            )
            text = response.content[0].text
            return self._parse_response(text)
        except asyncio.TimeoutError:
            print(f"[{self.name}] Timed out after {AGENT_TIMEOUT}s — using fallback")
            return self._fallback_analyze(system_type, data_profile)
        except Exception as e:
            print(f"[{self.name}] LLM call failed: {e}")
            return self._fallback_analyze(system_type, data_profile)

    # Subclasses override these ───────────────────────────

    def _system_prompt(self, system_type: str) -> str:
        return ""

    def _build_prompt(self, system_type: str, system_name: str,
                      data_summary: str, metadata_context: str) -> str:
        return ""

    def _fallback_analyze(self, system_type: str, data_profile: Dict) -> List[AgentFinding]:
        return []

    # Shared response parser ──────────────────────────────

    def _parse_response(self, text: str) -> List[AgentFinding]:
        """Parse structured JSON findings from the LLM response."""
        findings: List[AgentFinding] = []

        # Try to extract JSON array from the response
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            try:
                items = json.loads(json_match.group(0))
                for item in items:
                    findings.append(AgentFinding(
                        agent_name=self.name,
                        anomaly_type=item.get("type", "ai_detected"),
                        severity=item.get("severity", "medium"),
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        natural_language_explanation=item.get("explanation", ""),
                        possible_causes=item.get("possible_causes", []),
                        recommendations=[
                            {"type": "ai_recommendation", "priority": r.get("priority", "medium"),
                             "action": r.get("action", r) if isinstance(r, dict) else str(r)}
                            for r in item.get("recommendations", [])
                        ],
                        affected_fields=item.get("affected_fields", []),
                        confidence=item.get("confidence", 0.7),
                        impact_score=item.get("impact_score", 50),
                    ))
                return findings
            except json.JSONDecodeError:
                pass

        # Fallback: treat entire text as a single finding
        if text.strip():
            findings.append(AgentFinding(
                agent_name=self.name,
                anomaly_type="ai_insight",
                severity="medium",
                title=f"{self.name} analysis",
                description=text[:300],
                natural_language_explanation=text,
                confidence=0.6,
                impact_score=40,
            ))
        return findings


# ─────────────────── concrete agents ─────────────────────

class StatisticalAnalyst(BaseAgent):
    name = "Statistical Analyst"
    perspective = "Looks at the data purely through numbers and distributions"

    def _system_prompt(self, system_type: str) -> str:
        return (
            "You are a senior data scientist specializing in statistical anomaly detection "
            "for {type} systems. You focus strictly on mathematical evidence: distributions, "
            "outliers, z-scores, skewness, kurtosis, and unexpected statistical properties.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{{"type":"statistical_outlier|distribution_shift|variance_anomaly",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what you found",'
            '"explanation":"detailed natural-language explanation of WHY this matters for a {type} system",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{{"priority":"high","action":"what to do"}}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}}\n'
            "Output ONLY the JSON array, nothing else."
        ).format(type=system_type)

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Analyze this {system_type} system data for statistical anomalies.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Identify all statistical anomalies you can find. Focus on:\n"
            "- Outlier distributions and their severity\n"
            "- Unexpected statistical properties (bimodal distributions, heavy tails)\n"
            "- Fields with abnormal variance\n"
            "- Data quality issues visible in the statistics\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            if f.get("std") and f.get("mean") and f["std"] > 0:
                cv = f["std"] / abs(f["mean"]) if f["mean"] != 0 else 0
                if cv > 0.5:
                    findings.append(AgentFinding(
                        agent_name=self.name,
                        anomaly_type="high_variance",
                        severity="medium",
                        title=f"High variability in {f['name']}",
                        description=f"Coefficient of variation = {cv:.2f} (>0.5 threshold)",
                        natural_language_explanation=(
                            f"The field '{f['name']}' shows a coefficient of variation of {cv:.2f}, "
                            f"meaning the standard deviation is {cv*100:.0f}% of the mean. "
                            f"This level of variability may indicate unstable operating conditions "
                            f"or mixed operating modes in the {system_type} system."
                        ),
                        possible_causes=["Mixed operating modes", "Sensor noise", "Process instability"],
                        affected_fields=[f["name"]],
                        confidence=0.7,
                        impact_score=min(100, cv * 60),
                    ))
        return findings


class DomainExpert(BaseAgent):
    name = "Domain Expert"
    perspective = "Applies deep engineering domain knowledge"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a veteran {system_type} engineer with 20+ years of experience. "
            "You understand the physics behind every sensor reading. When you see data, "
            "you think about what physical processes could produce those numbers.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"domain_anomaly|physics_violation|operational_risk",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what you found",'
            '"explanation":"detailed engineering explanation in plain language, '
            'referencing physical principles and real-world consequences",'
            '"possible_causes":["engineering cause 1","engineering cause 2","engineering cause 3"],'
            '"recommendations":[{"priority":"high|medium|low","action":"specific engineering action"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"As a {system_type} domain expert, analyze this system data.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('SYSTEM DESCRIPTION: ' + metadata_context) if metadata_context else ''}\n\n"
            "Apply your deep engineering knowledge:\n"
            "- Do the value ranges make physical sense for this type of equipment?\n"
            "- Are there any readings that violate known physics or engineering limits?\n"
            "- What operational risks do you see in these numbers?\n"
            "- What would a field engineer be worried about?\n"
            "- Consider the relationships between parameters: do they make physical sense?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            name_lower = f["name"].lower()
            if "label" in name_lower and f.get("mean") is not None:
                fault_rate = f["mean"]
                if fault_rate > 0.1:
                    findings.append(AgentFinding(
                        agent_name=self.name,
                        anomaly_type="operational_risk",
                        severity="high" if fault_rate > 0.3 else "medium",
                        title=f"Elevated fault rate detected ({fault_rate*100:.1f}%)",
                        description=f"The label field shows {fault_rate*100:.1f}% fault conditions",
                        natural_language_explanation=(
                            f"In this {system_type} system, the fault label indicates that "
                            f"{fault_rate*100:.1f}% of all readings correspond to faulty conditions. "
                            f"A healthy system typically shows less than 5% fault rate. "
                            f"This elevated rate suggests recurring issues that need investigation."
                        ),
                        possible_causes=[
                            "Systematic equipment degradation",
                            "Operating outside design parameters",
                            "Insufficient maintenance intervals",
                        ],
                        recommendations=[
                            {"type": "maintenance", "priority": "high",
                             "action": "Schedule comprehensive equipment inspection"},
                        ],
                        affected_fields=[f["name"]],
                        confidence=0.85,
                        impact_score=min(100, fault_rate * 200),
                    ))
        return findings


class PatternDetective(BaseAgent):
    name = "Pattern Detective"
    perspective = "Searches for hidden patterns and correlations"

    def _system_prompt(self, system_type: str):
        return (
            "You are an AI pattern recognition specialist. You excel at finding hidden "
            "relationships, unexpected correlations, temporal patterns, and anomalous "
            "clusters in data. You look beyond individual fields to understand how "
            "the entire system behaves as a whole.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"correlation_anomaly|hidden_pattern|cluster_anomaly|temporal_pattern",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what pattern you found",'
            '"explanation":"detailed explanation of the pattern, why it is unusual, '
            'and what it might mean for the system",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1","field2"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Search for hidden patterns in this {system_type} system data.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Look for:\n"
            "- Unexpected correlations between fields that shouldn't be related\n"
            "- Missing correlations between fields that should be related\n"
            "- Signs of distinct operating modes or clusters in the data\n"
            "- Temporal patterns (cyclic behavior, drift)\n"
            "- Multi-variate anomalies (single fields look fine but combinations are off)\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        corrs = data_profile.get("correlations", {})
        for pair, val in corrs.items():
            if abs(val) > 0.85:
                fields = pair.split(" vs ")
                findings.append(AgentFinding(
                    agent_name=self.name,
                    anomaly_type="strong_correlation",
                    severity="info",
                    title=f"Strong correlation: {pair} ({val:.2f})",
                    description=f"Fields show {val:.2f} correlation",
                    natural_language_explanation=(
                        f"A strong {'positive' if val > 0 else 'negative'} correlation of {val:.2f} "
                        f"was found between {pair}. In a {system_type} system, this may indicate "
                        f"these parameters are physically linked or one is derived from the other."
                    ),
                    affected_fields=fields,
                    confidence=0.8,
                    impact_score=30,
                ))
        return findings


class RootCauseInvestigator(BaseAgent):
    """Uses extended thinking for deep root-cause reasoning."""
    name = "Root Cause Investigator"
    perspective = "Deep thinker that reasons about fundamental causes"
    model = "claude-sonnet-4-20250514"

    async def analyze(self, system_type, system_name, data_profile,
                      metadata_context=""):
        self._init_client()
        if not self.client:
            return self._fallback_analyze(system_type, data_profile)

        data_summary = self._build_data_summary(data_profile)

        system_context = (
            f"You are a root cause investigator — a veteran engineer who reasons about "
            f"fundamental causes of anomalies in {system_type} systems. You focus on the "
            f"chain of causation, not just symptoms. You think about what physical processes "
            f"could produce the observed data.\n\n"
        )

        prompt = (
            f"{system_context}"
            f"Investigate the root causes of anomalies in this {system_type} system "
            f"called '{system_name}'.\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('SYSTEM DESCRIPTION: ' + metadata_context) if metadata_context else ''}\n\n"
            "Take your time to think deeply. Consider:\n"
            "1. What physical processes could produce this data?\n"
            "2. If there are anomalies, what chain of events could lead to them?\n"
            "3. What is the MOST LIKELY root cause vs just symptoms?\n"
            "4. What would you investigate first if you were on-site?\n"
            "5. Are there potential cascading failures?\n\n"
            "Output your findings as a JSON array. Each element:\n"
            '{"type":"root_cause|cascading_risk|systemic_issue",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what you found",'
            '"explanation":"detailed chain-of-reasoning explanation",'
            '"possible_causes":["root cause 1","root cause 2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"specific action"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

        # Get extended thinking budget from settings
        try:
            from ..api.app_settings import get_ai_settings
            ai_cfg = get_ai_settings()
            budget = ai_cfg.get("extended_thinking_budget", 10000)
        except Exception:
            budget = 10000

        try:
            # Use extended thinking for deeper reasoning (with timeout)
            # Note: extended thinking does not support the `system` parameter
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    temperature=1,  # Required for extended thinking
                    thinking={
                        "type": "enabled",
                        "budget_tokens": budget,
                    },
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=AGENT_TIMEOUT,
            )

            # Extract the text response and thinking
            text = ""
            thinking_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    text = block.text

            findings = self._parse_response(text)
            # Attach the raw reasoning to each finding
            for f in findings:
                f.raw_reasoning = thinking_text[:1000] if thinking_text else ""
            return findings

        except asyncio.TimeoutError:
            print(f"[{self.name}] Extended thinking timed out after {AGENT_TIMEOUT}s — using fallback")
            return self._fallback_analyze(system_type, data_profile)
        except Exception as e:
            print(f"[{self.name}] Extended thinking call failed: {e}")
            # Fallback to regular call without extended thinking
            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        system=system_context,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=AGENT_TIMEOUT,
                )
                text = response.content[0].text
                return self._parse_response(text)
            except (asyncio.TimeoutError, Exception) as e2:
                print(f"[{self.name}] Regular call also failed: {e2}")
                return self._fallback_analyze(system_type, data_profile)

    def _fallback_analyze(self, system_type, data_profile):
        return []


class SafetyAuditor(BaseAgent):
    name = "Safety Auditor"
    perspective = "Evaluates safety margins and risk levels"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a safety engineer auditing a {system_type} system. "
            "Your job is to identify anything that could pose a safety risk, "
            "compromise system reliability, or lead to catastrophic failure. "
            "You are conservative and err on the side of caution.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"safety_risk|reliability_concern|margin_violation",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what safety concern you found",'
            '"explanation":"detailed explanation of the safety implications",'
            '"possible_causes":["cause1"],'
            '"recommendations":[{"priority":"immediate|high|medium","action":"safety action"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Perform a safety audit on this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Evaluate:\n"
            "- Are any parameters dangerously close to limits?\n"
            "- Could any combination of values lead to a dangerous situation?\n"
            "- Are there adequate safety margins?\n"
            "- What single-point failures could occur?\n"
            "- What should be monitored most closely?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        return []


# ─────────────────── new agents (6–13) ─────────────────────


class TemporalAnalyst(BaseAgent):
    """Detects time-series anomalies: seasonality, change-points, drift."""
    name = "Temporal Analyst"
    perspective = "Analyzes time-series behaviour, periodicity, and abrupt regime shifts"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a time-series analysis specialist for {system_type} systems. "
            "You look for temporal structure: seasonality, periodicity, abrupt "
            "change-points, gradual drift, and non-stationarity.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"change_point|seasonality_break|drift|temporal_anomaly",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what you found",'
            '"explanation":"why this temporal pattern matters",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Analyze temporal patterns in this {system_type} system data.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Look for:\n"
            "- Abrupt change-points where behaviour shifts suddenly\n"
            "- Gradual drift that indicates sensor degradation or process change\n"
            "- Periodic/seasonal patterns and any breaks in those patterns\n"
            "- Non-stationarity: is the process stable over time?\n"
            "- Anomalous windows: time periods where behaviour differs markedly\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            if f.get("std") and f.get("mean") and f.get("min") is not None and f.get("max") is not None:
                value_range = f["max"] - f["min"]
                if f["mean"] != 0 and value_range > 0:
                    range_ratio = value_range / abs(f["mean"])
                    if range_ratio > 2.0:
                        findings.append(AgentFinding(
                            agent_name=self.name,
                            anomaly_type="drift",
                            severity="medium",
                            title=f"Wide operating range in {f['name']}",
                            description=(
                                f"Range ({f['min']:.4g} to {f['max']:.4g}) is {range_ratio:.1f}x the mean, "
                                f"suggesting possible regime changes or drift."
                            ),
                            natural_language_explanation=(
                                f"The field '{f['name']}' spans from {f['min']:.4g} to {f['max']:.4g}, "
                                f"a range that is {range_ratio:.1f} times the mean value. "
                                f"This wide spread may indicate the system operates in different regimes "
                                f"or has experienced drift over time."
                            ),
                            possible_causes=["Operating mode transitions", "Sensor drift", "Process change"],
                            affected_fields=[f["name"]],
                            confidence=0.65,
                            impact_score=min(100, range_ratio * 25),
                        ))
        return findings


class DataQualityInspector(BaseAgent):
    """Inspects data integrity: missing values, sensor drift, corruption."""
    name = "Data Quality Inspector"
    perspective = "Focuses on data integrity, completeness, and trustworthiness"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a data quality engineer auditing telemetry from a {system_type} system. "
            "Your job is to find data integrity issues that could compromise analysis quality: "
            "missing data patterns, sensor drift, stuck sensors, encoding errors, "
            "unit mismatches, and suspicious data artefacts.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"data_quality|missing_data|sensor_drift|stuck_sensor|encoding_error",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what data quality issue you found",'
            '"explanation":"how this affects analysis reliability",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"how to fix"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Audit the data quality of this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Look for:\n"
            "- Fields with excessive missing values or suspicious null patterns\n"
            "- Zero-variance fields that may indicate stuck/failed sensors\n"
            "- Fields where min=max or std=0 suggesting constant or stuck readings\n"
            "- Unusual data types or encodings that suggest pipeline errors\n"
            "- Value distributions that look truncated, clipped, or artificially bounded\n"
            "- Possible unit mismatches (e.g., Celsius vs Fahrenheit ranges)\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            # Stuck sensor: zero variance
            if f.get("std") is not None and f["std"] == 0 and f.get("mean") is not None:
                findings.append(AgentFinding(
                    agent_name=self.name,
                    anomaly_type="stuck_sensor",
                    severity="high",
                    title=f"Stuck/constant value in {f['name']}",
                    description=f"Field has zero variance (constant value = {f['mean']:.4g})",
                    natural_language_explanation=(
                        f"The field '{f['name']}' shows absolutely no variation — every reading "
                        f"is {f['mean']:.4g}. This strongly suggests a stuck sensor, frozen data "
                        f"pipeline, or a constant default value being reported instead of real data."
                    ),
                    possible_causes=["Sensor failure", "Frozen data pipeline", "Default value override"],
                    recommendations=[
                        {"type": "investigation", "priority": "high",
                         "action": f"Verify sensor for '{f['name']}' is operational and reporting live data"},
                    ],
                    affected_fields=[f["name"]],
                    confidence=0.9,
                    impact_score=70,
                ))
            # Low unique count (for numeric fields with high record count)
            if (f.get("unique_count") is not None and f.get("type", "").startswith(("int", "float"))
                    and f["unique_count"] <= 3 and data_profile.get("record_count", 0) > 50):
                findings.append(AgentFinding(
                    agent_name=self.name,
                    anomaly_type="data_quality",
                    severity="medium",
                    title=f"Suspiciously low cardinality in {f['name']}",
                    description=f"Only {f['unique_count']} unique values in {data_profile.get('record_count', '?')} records",
                    natural_language_explanation=(
                        f"The numeric field '{f['name']}' has only {f['unique_count']} distinct values "
                        f"across {data_profile.get('record_count', '?')} records. This may indicate "
                        f"discretization, encoding issues, or a categorical field mistyped as numeric."
                    ),
                    possible_causes=["Discretized sensor", "Encoding error", "Categorical field"],
                    affected_fields=[f["name"]],
                    confidence=0.7,
                    impact_score=40,
                ))
        return findings


class PredictiveForecaster(BaseAgent):
    """Predicts future anomalies based on trend extrapolation."""
    name = "Predictive Forecaster"
    perspective = "Extrapolates trends to predict future failures and anomalies"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a predictive maintenance specialist for {system_type} systems. "
            "You extrapolate current trends to forecast future problems. "
            "You think about degradation curves, failure timelines, and early warnings.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"predicted_failure|degradation_trend|early_warning|capacity_risk",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what future risk you predict",'
            '"explanation":"how you arrived at this prediction and what evidence supports it",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"immediate|high|medium|low","action":"preventive action"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Predict future risks for this {system_type} system based on current data.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Forecast:\n"
            "- Which parameters are trending toward dangerous levels?\n"
            "- What is the estimated time-to-failure if trends continue?\n"
            "- Are there early warning signs of impending failures?\n"
            "- What capacity limits might be reached soon?\n"
            "- What preventive maintenance should be scheduled?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            if f.get("mean") is not None and f.get("max") is not None and f.get("std") is not None:
                if f["std"] > 0 and f["max"] != 0:
                    utilization = abs(f["mean"]) / abs(f["max"]) if f["max"] != 0 else 0
                    if utilization > 0.85:
                        findings.append(AgentFinding(
                            agent_name=self.name,
                            anomaly_type="capacity_risk",
                            severity="high" if utilization > 0.95 else "medium",
                            title=f"Near-capacity operation in {f['name']}",
                            description=f"Mean ({f['mean']:.4g}) is at {utilization*100:.0f}% of observed max ({f['max']:.4g})",
                            natural_language_explanation=(
                                f"The field '{f['name']}' is operating at {utilization*100:.0f}% of its "
                                f"observed maximum. If the trend continues, this parameter may reach "
                                f"its limit, potentially causing failures or forcing shutdowns."
                            ),
                            possible_causes=["Increasing load", "Degrading capacity", "Approaching design limits"],
                            recommendations=[
                                {"type": "predictive", "priority": "high",
                                 "action": f"Plan capacity expansion or load reduction for '{f['name']}'"},
                            ],
                            affected_fields=[f["name"]],
                            confidence=0.7,
                            impact_score=min(100, utilization * 100),
                        ))
        return findings


class OperationalProfiler(BaseAgent):
    """Identifies operating modes, regime transitions, and mode anomalies."""
    name = "Operational Profiler"
    perspective = "Identifies distinct operating modes and detects abnormal transitions"

    def _system_prompt(self, system_type: str):
        return (
            f"You are an operations analyst specializing in {system_type} system behaviour. "
            "You excel at identifying distinct operating modes (startup, steady-state, "
            "peak load, shutdown, maintenance) and detecting abnormal mode transitions.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"mode_anomaly|abnormal_transition|mixed_operation|unexpected_state",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what operational anomaly you found",'
            '"explanation":"why this operating pattern is concerning",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"operational action"}],'
            '"affected_fields":["field1","field2"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Profile the operating modes of this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Analyze:\n"
            "- Can you identify distinct operating modes/regimes from the data?\n"
            "- Are there abnormal or unexpected transitions between modes?\n"
            "- Is the system spending too much time in non-optimal modes?\n"
            "- Are there fields with multimodal distributions suggesting mixed operations?\n"
            "- Do any parameter combinations indicate conflicting operating states?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            if f.get("std") and f.get("mean") and f["mean"] != 0:
                cv = f["std"] / abs(f["mean"])
                if cv > 1.0:
                    findings.append(AgentFinding(
                        agent_name=self.name,
                        anomaly_type="mixed_operation",
                        severity="medium",
                        title=f"Possible multi-mode operation in {f['name']}",
                        description=f"CV = {cv:.2f} suggests data from multiple operating regimes",
                        natural_language_explanation=(
                            f"The field '{f['name']}' has a coefficient of variation of {cv:.2f}, "
                            f"where the spread far exceeds the mean. This pattern is characteristic "
                            f"of data collected across multiple operating modes (e.g., idle vs full load). "
                            f"Analyzing each mode separately may reveal hidden anomalies."
                        ),
                        possible_causes=["Multiple operating modes", "Startup/shutdown transients", "Load cycling"],
                        affected_fields=[f["name"]],
                        confidence=0.65,
                        impact_score=min(100, cv * 30),
                    ))
        return findings


class EfficiencyAnalyst(BaseAgent):
    """Analyzes energy/resource consumption patterns for waste and optimization."""
    name = "Efficiency Analyst"
    perspective = "Identifies energy waste, resource inefficiency, and optimization opportunities"

    def _system_prompt(self, system_type: str):
        return (
            f"You are an efficiency engineer optimizing a {system_type} system. "
            "You look for energy waste, suboptimal operating points, unnecessary "
            "resource consumption, and opportunities to improve efficiency.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"efficiency_loss|energy_waste|suboptimal_operation|optimization_opportunity",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what inefficiency you found",'
            '"explanation":"how much efficiency is being lost and what to do about it",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"optimization action"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Analyze the efficiency of this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Evaluate:\n"
            "- Are there signs of energy waste or unnecessary resource consumption?\n"
            "- Is the system operating at suboptimal points that could be improved?\n"
            "- Do parameter relationships suggest mechanical losses or friction?\n"
            "- Are there idle periods consuming resources?\n"
            "- What changes could reduce waste or improve throughput?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        # Look for energy/power/current fields with high mean relative to range
        for f in data_profile.get("fields", []):
            name_lower = f["name"].lower()
            is_energy = any(kw in name_lower for kw in
                           ["power", "energy", "current", "consumption", "fuel", "watt"])
            if is_energy and f.get("mean") is not None and f.get("min") is not None:
                if f["mean"] > 0 and f["min"] >= 0:
                    base_load_ratio = f["min"] / f["mean"] if f["mean"] > 0 else 0
                    if base_load_ratio > 0.6:
                        findings.append(AgentFinding(
                            agent_name=self.name,
                            anomaly_type="efficiency_loss",
                            severity="medium",
                            title=f"High base load in {f['name']}",
                            description=(
                                f"Minimum ({f['min']:.4g}) is {base_load_ratio*100:.0f}% of mean "
                                f"({f['mean']:.4g}), indicating high idle consumption."
                            ),
                            natural_language_explanation=(
                                f"The energy-related field '{f['name']}' has a minimum value that is "
                                f"{base_load_ratio*100:.0f}% of the average. This high baseline suggests "
                                f"significant energy is consumed even at low load, pointing to standby "
                                f"losses, mechanical friction, or inefficient idle operation."
                            ),
                            possible_causes=["Standby power losses", "Mechanical friction", "Inefficient idle mode"],
                            recommendations=[
                                {"type": "optimization", "priority": "medium",
                                 "action": f"Investigate base load reduction for '{f['name']}'"},
                            ],
                            affected_fields=[f["name"]],
                            confidence=0.7,
                            impact_score=min(100, base_load_ratio * 80),
                        ))
        return findings


class ComplianceChecker(BaseAgent):
    """Checks against industry standards, regulatory limits, and best practices."""
    name = "Compliance Checker"
    perspective = "Evaluates data against regulatory limits and industry standards"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a compliance and regulatory specialist for {system_type} systems. "
            "You know industry standards (ISO, IEC, OSHA, SAE, FDA, etc.) and "
            "regulatory requirements. You check whether operating parameters "
            "comply with relevant standards and flag potential violations.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"compliance_violation|standard_deviation|regulatory_risk|best_practice_gap",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what compliance issue you found",'
            '"explanation":"which standard/regulation is at risk and what the consequences are",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"immediate|high|medium","action":"compliance action"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Check this {system_type} system for compliance issues.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Evaluate against relevant industry standards:\n"
            "- Do any parameters exceed known regulatory limits for this system type?\n"
            "- Are there operating conditions that violate industry best practices?\n"
            "- What standards (ISO, IEC, OSHA, SAE, FDA) are most relevant?\n"
            "- Are monitoring and recording practices adequate for compliance?\n"
            "- What documentation gaps exist?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        # Compliance requires domain knowledge; minimal fallback
        return []


class ReliabilityEngineer(BaseAgent):
    """Analyzes degradation patterns, MTBF indicators, and wear-out trends."""
    name = "Reliability Engineer"
    perspective = "Focuses on degradation, wear-out, and long-term reliability"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a reliability engineer analyzing a {system_type} system. "
            "You think about bathtub curves, wear-out mechanisms, MTBF, "
            "degradation trajectories, and remaining useful life. "
            "You look for early signs of component aging.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"degradation|wear_indicator|reliability_risk|aging_sign",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what reliability concern you found",'
            '"explanation":"what degradation mechanism is at play and the implications",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"immediate|high|medium|low","action":"reliability action"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Assess the reliability of this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Analyze for reliability:\n"
            "- Are there signs of component degradation or wear?\n"
            "- Do any parameters show monotonic drift suggesting aging?\n"
            "- Is variance increasing over time (wear-out signature)?\n"
            "- What is the estimated remaining useful life based on trends?\n"
            "- Are maintenance intervals adequate based on degradation rates?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            name_lower = f["name"].lower()
            is_wear = any(kw in name_lower for kw in
                          ["vibrat", "noise", "wear", "friction", "resistan", "impedanc", "degrad"])
            if is_wear and f.get("std") is not None and f.get("mean") is not None and f["std"] > 0:
                cv = f["std"] / abs(f["mean"]) if f["mean"] != 0 else 0
                if cv > 0.3:
                    findings.append(AgentFinding(
                        agent_name=self.name,
                        anomaly_type="wear_indicator",
                        severity="medium",
                        title=f"Elevated variability in wear-related field {f['name']}",
                        description=f"CV = {cv:.2f} in a wear/degradation indicator",
                        natural_language_explanation=(
                            f"The wear-related field '{f['name']}' shows CV = {cv:.2f}. "
                            f"High variability in such fields often correlates with advancing "
                            f"component degradation or inconsistent mechanical behaviour."
                        ),
                        possible_causes=["Component wear", "Bearing degradation", "Increasing friction"],
                        recommendations=[
                            {"type": "maintenance", "priority": "high",
                             "action": f"Schedule inspection of components related to '{f['name']}'"},
                        ],
                        affected_fields=[f["name"]],
                        confidence=0.7,
                        impact_score=min(100, cv * 80),
                    ))
        return findings


class EnvironmentalCorrelator(BaseAgent):
    """Finds cross-parameter environmental effects and external influences."""
    name = "Environmental Correlator"
    perspective = "Identifies environmental factors and external influences on system behaviour"

    def _system_prompt(self, system_type: str):
        return (
            f"You are an environmental impact analyst for a {system_type} system. "
            "You look for how ambient conditions (temperature, humidity, altitude, "
            "load, time-of-day) affect system performance. You find hidden "
            "environmental dependencies that operators might miss.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"environmental_impact|external_dependency|ambient_effect|load_sensitivity",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what environmental effect you found",'
            '"explanation":"how the environment is affecting system performance",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"mitigation action"}],'
            '"affected_fields":["field1","field2"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Analyze environmental influences on this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Investigate:\n"
            "- Do temperature or environmental fields correlate with performance metrics?\n"
            "- Are there parameters that are unexpectedly sensitive to external conditions?\n"
            "- Do any cross-field correlations suggest hidden environmental dependencies?\n"
            "- Are there operating conditions where environmental effects become critical?\n"
            "- What environmental mitigations should be considered?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        corrs = data_profile.get("correlations", {})
        env_keywords = ["temp", "humid", "ambient", "pressure", "altitude", "weather"]
        for pair, val in corrs.items():
            pair_lower = pair.lower()
            has_env = any(kw in pair_lower for kw in env_keywords)
            if has_env and abs(val) > 0.6:
                fields = pair.split(" vs ")
                env_field = [f for f in fields if any(kw in f.lower() for kw in env_keywords)]
                perf_field = [f for f in fields if f not in env_field]
                findings.append(AgentFinding(
                    agent_name=self.name,
                    anomaly_type="environmental_impact",
                    severity="medium" if abs(val) > 0.8 else "low",
                    title=f"Environmental dependency: {pair} (r={val:.2f})",
                    description=f"{'Strong' if abs(val) > 0.8 else 'Moderate'} correlation between environmental and performance fields",
                    natural_language_explanation=(
                        f"A {'strong' if abs(val) > 0.8 else 'moderate'} correlation of {val:.2f} "
                        f"was found between {pair}. This suggests that "
                        f"{'the environmental parameter ' + env_field[0] if env_field else 'an ambient condition'} "
                        f"significantly influences "
                        f"{'the performance metric ' + perf_field[0] if perf_field else 'system behaviour'}. "
                        f"This dependency should be accounted for in operating procedures."
                    ),
                    possible_causes=["Thermal sensitivity", "Environmental dependency", "Ambient condition effects"],
                    affected_fields=fields,
                    confidence=0.75,
                    impact_score=min(100, abs(val) * 70),
                ))
        return findings


# ─────── web-grounding enrichment ────────

async def enrich_with_web(finding: AgentFinding, system_type: str) -> AgentFinding:
    """Search the web for additional context about a finding."""
    if not finding.title:
        return finding

    query = f"{system_type} {finding.title} engineering cause solution"
    results = await web_search(query)

    if results:
        finding.web_references = [r["url"] for r in results[:3]]
        # Add relevant snippets to the explanation
        snippets = [r["snippet"] for r in results[:2] if r.get("snippet")]
        if snippets:
            web_context = " | ".join(snippets)
            finding.natural_language_explanation += (
                f"\n\nAdditional engineering context from web research: {web_context}"
            )
    return finding


# ─────── orchestrator ────────────────────

class AgentOrchestrator:
    """
    Runs all agents in parallel, then merges and de-duplicates their
    findings into a unified set of anomalies.
    """

    def __init__(self):
        self.agents: List[BaseAgent] = [
            StatisticalAnalyst(),
            DomainExpert(),
            PatternDetective(),
            RootCauseInvestigator(),
            SafetyAuditor(),
            TemporalAnalyst(),
            DataQualityInspector(),
            PredictiveForecaster(),
            OperationalProfiler(),
            EfficiencyAnalyst(),
            ComplianceChecker(),
            ReliabilityEngineer(),
            EnvironmentalCorrelator(),
        ]

    async def run_analysis(
        self,
        system_id: str,
        system_type: str,
        system_name: str,
        data_profile: Dict,
        metadata_context: str = "",
        enable_web_grounding: bool = True,
        selected_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run agents in parallel and unify results.

        Args:
            selected_agents: If provided, only run agents whose names are in this list.
                             If None or empty, all agents are run.

        Has a global timeout of ORCHESTRATOR_TIMEOUT seconds.  Individual
        agents also have their own AGENT_TIMEOUT.
        """

        # Filter agents if selection provided
        if selected_agents:
            selected_set = set(selected_agents)
            active_agents = [a for a in self.agents if a.name in selected_set]
            if not active_agents:
                active_agents = list(self.agents)  # fallback to all if none matched
        else:
            active_agents = list(self.agents)

        print(f"[Orchestrator] Running {len(active_agents)} agents: {[a.name for a in active_agents]}")

        # Create proper asyncio Tasks so we can cancel them on global timeout
        tasks = [
            asyncio.create_task(
                agent.analyze(system_type, system_name, data_profile, metadata_context),
                name=agent.name,
            )
            for agent in active_agents
        ]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=ORCHESTRATOR_TIMEOUT,
            )
        except asyncio.TimeoutError:
            print(f"[Orchestrator] Global timeout ({ORCHESTRATOR_TIMEOUT}s) hit — collecting partial results")
            # Cancel remaining tasks and collect whatever finished
            results = []
            for task in tasks:
                if task.done() and not task.cancelled():
                    try:
                        results.append(task.result())
                    except Exception as exc:
                        results.append(exc)
                else:
                    task.cancel()
                    results.append(TimeoutError("Orchestrator global timeout"))

        # Collect all findings
        all_findings: List[AgentFinding] = []
        agent_statuses = []

        for agent, result in zip(active_agents, results):
            if isinstance(result, Exception):
                print(f"[Orchestrator] Agent {agent.name} failed: {result}")
                agent_statuses.append({
                    "agent": agent.name,
                    "status": "error",
                    "findings": 0,
                    "error": str(result),
                })
            else:
                all_findings.extend(result)
                agent_statuses.append({
                    "agent": agent.name,
                    "status": "success",
                    "findings": len(result),
                    "perspective": agent.perspective,
                })

        # Web-grounding enrichment for top findings (parallel)
        if enable_web_grounding:
            top_findings = sorted(all_findings, key=lambda f: f.impact_score, reverse=True)[:5]
            grounding_tasks = [enrich_with_web(f, system_type) for f in top_findings]
            await asyncio.gather(*grounding_tasks, return_exceptions=True)

        # Merge and deduplicate
        unified = self._merge_findings(all_findings, system_id)

        return {
            "anomalies": [self._anomaly_to_dict(a) for a in unified],
            "agent_statuses": agent_statuses,
            "total_findings_raw": len(all_findings),
            "total_anomalies_unified": len(unified),
            "agents_used": [a.name for a in active_agents],
            "ai_powered": HAS_ANTHROPIC and bool(_get_api_key()),
        }

    def _merge_findings(self, findings: List[AgentFinding], system_id: str) -> List[UnifiedAnomaly]:
        """Merge findings from multiple agents, grouping similar ones."""
        if not findings:
            return []

        # Group by affected fields + type similarity
        groups: Dict[str, List[AgentFinding]] = {}

        for finding in findings:
            # Create a grouping key
            fields_key = ",".join(sorted(finding.affected_fields)) if finding.affected_fields else "general"
            group_key = f"{fields_key}|{finding.anomaly_type}"

            # Check if we should merge with an existing group
            merged = False
            for existing_key, group in groups.items():
                existing_fields = existing_key.split("|")[0]
                if fields_key == existing_fields or self._titles_similar(finding.title, group[0].title):
                    groups[existing_key].append(finding)
                    merged = True
                    break

            if not merged:
                groups[group_key] = [finding]

        # Create unified anomalies from groups
        unified = []
        for group_key, group in groups.items():
            anomaly = self._create_unified_anomaly(group, system_id)
            unified.append(anomaly)

        # Sort by impact score
        unified.sort(key=lambda a: a.impact_score, reverse=True)
        return unified[:15]  # Top 15 anomalies

    def _titles_similar(self, a: str, b: str) -> bool:
        """Check if two titles refer to the same issue."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return False
        overlap = len(a_words & b_words)
        return overlap / min(len(a_words), len(b_words)) > 0.5

    def _create_unified_anomaly(self, findings: List[AgentFinding], system_id: str) -> UnifiedAnomaly:
        """Create a single unified anomaly from a group of findings."""
        # Use the highest-severity finding as the primary
        primary = max(findings, key=lambda f: self._severity_score(f.severity))

        # Collect all unique causes and recommendations
        all_causes = []
        all_recs = []
        all_fields = set()
        all_refs = []
        perspectives = []

        for f in findings:
            all_causes.extend(f.possible_causes)
            all_recs.extend(f.recommendations)
            all_fields.update(f.affected_fields)
            all_refs.extend(f.web_references)
            if f.natural_language_explanation:
                perspectives.append({
                    "agent": f.agent_name,
                    "perspective": f.natural_language_explanation[:500],
                })

        # Deduplicate causes
        seen_causes = set()
        unique_causes = []
        for cause in all_causes:
            if cause.lower() not in seen_causes:
                seen_causes.add(cause.lower())
                unique_causes.append(cause)

        # Build a unified explanation
        explanation = primary.natural_language_explanation
        if len(findings) > 1:
            other_agents = [f.agent_name for f in findings if f != primary]
            explanation += (
                f"\n\nThis finding was corroborated by {len(findings)} AI agents: "
                f"{', '.join(set(a.agent_name for a in findings))}. "
                f"Multiple independent analysis perspectives confirm this issue."
            )

        # Create ID from content
        id_str = f"{system_id}_{primary.title}_{datetime.utcnow().timestamp()}"
        anomaly_id = hashlib.md5(id_str.encode()).hexdigest()[:12]

        return UnifiedAnomaly(
            id=anomaly_id,
            type=primary.anomaly_type,
            severity=primary.severity,
            title=primary.title,
            description=primary.description,
            natural_language_explanation=explanation,
            possible_causes=unique_causes[:5],
            recommendations=all_recs[:5],
            affected_fields=list(all_fields),
            confidence=max(f.confidence for f in findings),
            impact_score=max(f.impact_score for f in findings),
            contributing_agents=list(set(f.agent_name for f in findings)),
            web_references=list(set(all_refs))[:5],
            agent_perspectives=perspectives[:5],
        )

    def _severity_score(self, severity: str) -> int:
        return {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}.get(severity, 0)

    def _anomaly_to_dict(self, anomaly: UnifiedAnomaly) -> Dict:
        return {
            "id": anomaly.id,
            "type": anomaly.type,
            "severity": anomaly.severity,
            "title": anomaly.title,
            "description": anomaly.description,
            "affected_fields": anomaly.affected_fields,
            "natural_language_explanation": anomaly.natural_language_explanation,
            "possible_causes": anomaly.possible_causes,
            "recommendations": anomaly.recommendations,
            "confidence": anomaly.confidence,
            "impact_score": anomaly.impact_score,
            "contributing_agents": anomaly.contributing_agents,
            "web_references": anomaly.web_references,
            "agent_perspectives": anomaly.agent_perspectives,
        }


# Global instance
orchestrator = AgentOrchestrator()
