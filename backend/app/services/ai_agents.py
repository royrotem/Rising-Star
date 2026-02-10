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
14. Stagnation Sentinel        – zero-variance / frozen-sensor detection
15. Noise Floor Auditor        – missing white noise in physical sensors
16. Micro-Drift Tracker        – tiny monotonic trends (hardware wear)
17. Cross-Sensor Sync          – physics-based cross-sensor consistency
18. Vibration Ghost            – vibration parameter gaps, mechanical imbalance
19. Harmonic Distortion        – electrical quality / current noise analysis
20. Quantization Critic        – ADC resolution & sampling artefacts
21. Cyber-Injection Hunter     – data manipulation / telemetry spoofing
22. Metadata Integrity         – device-ID / unit-of-measure consistency
23. Hydraulic/Pressure Expert  – pressure leaks, filter clogs, closed-loop integrity
24. Human-Context Filter       – time-of-day / occupancy logic correlation
25. Logic State Conflict       – metadata-vs-telemetry contradiction detection

Each agent can also ground its analysis with web search for
real-world engineering context.
"""

import asyncio
import json
import logging
import os
import re
import hashlib
import time
import traceback as tb_module
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger("uaie.ai_agents")

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
# Batch size for running agents — prevents rate limit issues (8K tokens/min limit)
AGENT_BATCH_SIZE = 5
# Delay between batches in seconds — allows rate limit to reset
BATCH_DELAY_SECONDS = 12

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
    "Stagnation Sentinel",
    "Noise Floor Auditor",
    "Micro-Drift Tracker",
    "Cross-Sensor Sync",
    "Vibration Ghost",
    "Harmonic Distortion",
    "Quantization Critic",
    "Cyber-Injection Hunter",
    "Metadata Integrity",
    "Hydraulic/Pressure Expert",
    "Human-Context Filter",
    "Logic State Conflict",
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
            logger.warning("[%s] No API client (no key?) — using fallback", self.name)
            return self._fallback_analyze(system_type, data_profile)

        data_summary = self._build_data_summary(data_profile)
        prompt = self._build_prompt(system_type, system_name, data_summary, metadata_context)
        logger.info("[%s] Sending LLM request (model=%s, prompt=%d chars)...", self.name, self.model, len(prompt))

        t_start = time.time()
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
            t_elapsed = round(time.time() - t_start, 2)
            usage = {"input": response.usage.input_tokens, "output": response.usage.output_tokens} if response.usage else "?"
            logger.info("[%s] LLM response in %.2fs: stop=%s, usage=%s",
                        self.name, t_elapsed, response.stop_reason, usage)

            text = response.content[0].text
            findings = self._parse_response(text)
            logger.info("[%s] Parsed %d findings", self.name, len(findings))
            return findings
        except asyncio.TimeoutError:
            t_elapsed = round(time.time() - t_start, 2)
            logger.error("[%s] TIMEOUT after %.2fs (limit=%ds) — using fallback", self.name, t_elapsed, AGENT_TIMEOUT)
            return self._fallback_analyze(system_type, data_profile)
        except Exception as e:
            t_elapsed = round(time.time() - t_start, 2)
            logger.error("[%s] LLM call FAILED after %.2fs: %s: %s", self.name, t_elapsed, type(e).__name__, e)
            logger.error(tb_module.format_exc())
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


# ─────────────────── agents 14–25: blind-spot specialists ─────────────────────


class StagnationSentinel(BaseAgent):
    """Detects quiet anomalies: zero-variance windows where a sensor is frozen
    on a perfectly valid but suspiciously constant value."""
    name = "Stagnation Sentinel"
    perspective = "Hunts for zero-variance windows — sensors frozen on a valid but constant value"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a stagnation-detection specialist for {system_type} systems. "
            "Your mission is to find QUIET anomalies — sensors that report perfectly "
            "valid readings (e.g. 21.1°C) but never change. A real physical sensor "
            "always has micro-fluctuations; zero variance over a window of 10+ "
            "samples is almost always a frozen sensor, stuck ADC, or stale cache.\n\n"
            "METHOD:\n"
            "- Sliding window of 10 consecutive samples: if std = 0 → flag.\n"
            "- Even if the VALUE is normal, constant = suspicious.\n"
            "- Check if multiple sensors freeze simultaneously (common-mode failure).\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"frozen_sensor|stagnant_value|zero_variance|common_mode_freeze",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what you found",'
            '"explanation":"why zero variance matters even when the value looks normal",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Hunt for stagnation anomalies in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Focus on:\n"
            "- Fields with std=0 or near-zero std relative to their scale\n"
            "- Fields where min=max (perfectly constant)\n"
            "- Fields with suspiciously low unique value counts\n"
            "- Multiple fields freezing at the same time (common-mode failure)\n"
            "- Values that look normal but are TOO stable for a physical sensor\n"
            "  (e.g. temperature exactly 21.1111 with zero fluctuation)\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        frozen_fields = []
        for f in data_profile.get("fields", []):
            if f.get("std") is not None and f["std"] == 0 and f.get("mean") is not None:
                frozen_fields.append(f["name"])
                findings.append(AgentFinding(
                    agent_name=self.name,
                    anomaly_type="frozen_sensor",
                    severity="high",
                    title=f"Stagnant value in {f['name']} (frozen at {f['mean']:.4g})",
                    description=(
                        f"Zero variance detected — every sample equals {f['mean']:.4g}. "
                        f"A physical sensor should show micro-fluctuations."
                    ),
                    natural_language_explanation=(
                        f"The field '{f['name']}' reads exactly {f['mean']:.4g} with zero variation. "
                        f"While {f['mean']:.4g} may be a perfectly normal value, real physical sensors "
                        f"always exhibit tiny fluctuations from thermal noise, vibration, or ADC "
                        f"quantisation. Zero variance strongly suggests a frozen sensor, stuck cache, "
                        f"or stale data pipeline — the value you see may be minutes or hours old."
                    ),
                    possible_causes=[
                        "Sensor hardware freeze", "Data pipeline caching stale value",
                        "ADC stuck on last conversion", "Communication bus failure (last-known-good)",
                    ],
                    recommendations=[
                        {"type": "investigation", "priority": "high",
                         "action": f"Power-cycle sensor for '{f['name']}' and verify live readings"},
                    ],
                    affected_fields=[f["name"]],
                    confidence=0.92,
                    impact_score=75,
                ))
        # Common-mode freeze
        if len(frozen_fields) >= 2:
            findings.append(AgentFinding(
                agent_name=self.name,
                anomaly_type="common_mode_freeze",
                severity="critical",
                title=f"Common-mode freeze: {len(frozen_fields)} sensors frozen simultaneously",
                description=f"Fields {', '.join(frozen_fields)} all show zero variance",
                natural_language_explanation=(
                    f"Multiple sensors ({', '.join(frozen_fields)}) are frozen simultaneously. "
                    f"This pattern typically indicates a shared failure upstream — a data concentrator, "
                    f"communication bus, or PLC that stopped polling and is serving cached values."
                ),
                possible_causes=["Data concentrator failure", "Communication bus hang", "PLC polling stopped"],
                affected_fields=frozen_fields,
                confidence=0.95,
                impact_score=90,
            ))
        return findings


class NoiseFloorAuditor(BaseAgent):
    """Checks whether physical sensors exhibit the expected white noise floor.
    Absence of noise in a physical measurement is itself an anomaly."""
    name = "Noise Floor Auditor"
    perspective = "Verifies that physical sensors exhibit expected white noise — absence of noise is an anomaly"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a signal-integrity specialist for {system_type} sensor systems. "
            "Every real physical sensor (temperature, pressure, current, vibration) has "
            "an inherent noise floor from thermal noise, quantisation, and environmental "
            "micro-disturbances. When a sensor's noise floor DISAPPEARS, it usually means "
            "the digitisation chain is broken — you're seeing cached, interpolated, or "
            "synthetic data, not live measurements.\n\n"
            "METHOD:\n"
            "- For each numeric field, compute std / mean (CV). Physical sensors typically "
            "  show CV > 0.001 even in steady-state.\n"
            "- Compare the noise profile across sensors: if most sensors have normal noise "
            "  but one is perfectly smooth, flag it.\n"
            "- Check if sample values show quantisation artefacts (all integers, "
            "  or only 1–2 decimal places when more are expected).\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"noise_absent|noise_suppressed|synthetic_signal|interpolated_data",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what you found",'
            '"explanation":"why missing noise matters for data trustworthiness",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Audit the noise floor of sensors in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Evaluate:\n"
            "- Do numeric fields have a healthy noise floor (CV > 0.001)?\n"
            "- Are there fields that are suspiciously smooth compared to peers?\n"
            "- Do sample values show expected decimal precision?\n"
            "- Could any field be serving interpolated or synthetic data?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        # Collect CVs for all numeric fields to compare
        cvs = {}
        for f in data_profile.get("fields", []):
            if f.get("std") is not None and f.get("mean") is not None and abs(f["mean"]) > 1e-9:
                cvs[f["name"]] = f["std"] / abs(f["mean"])

        if not cvs:
            return findings

        median_cv = sorted(cvs.values())[len(cvs) // 2] if cvs else 0

        for name, cv in cvs.items():
            if cv < 0.001 and median_cv > 0.005:
                findings.append(AgentFinding(
                    agent_name=self.name,
                    anomaly_type="noise_absent",
                    severity="high",
                    title=f"Missing noise floor in {name}",
                    description=(
                        f"CV = {cv:.6f} while peer sensors show median CV = {median_cv:.4f}. "
                        f"This sensor is suspiciously quiet."
                    ),
                    natural_language_explanation=(
                        f"The sensor '{name}' has a coefficient of variation of {cv:.6f}, "
                        f"far below the peer median of {median_cv:.4f}. Physical sensors always "
                        f"exhibit micro-noise from thermal effects and ADC quantisation. The absence "
                        f"of this noise floor suggests the data may be cached, interpolated, or synthetic."
                    ),
                    possible_causes=[
                        "Digitisation chain broken", "Data interpolation masking real noise",
                        "Cached/stale value being served", "Over-aggressive smoothing filter",
                    ],
                    affected_fields=[name],
                    confidence=0.8,
                    impact_score=65,
                ))
        return findings


class MicroDriftTracker(BaseAgent):
    """Detects tiny monotonic trends that accumulate over weeks — the signature
    of hardware wear before it becomes a visible outlier."""
    name = "Micro-Drift Tracker"
    perspective = "Tracks tiny monotonic trends (0.01°C/sample) that signal hardware wear"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a micro-drift detection specialist for {system_type} systems. "
            "You hunt for the quietest, most dangerous anomaly: a parameter that "
            "creeps upward (or downward) by tiny amounts — too small for Z-score "
            "detection — but relentlessly, without ever reversing. Over weeks this "
            "drift compounds into a real failure.\n\n"
            "METHOD:\n"
            "- Check the rate-of-change (derivative) of each parameter.\n"
            "- A monotonic sequence of 50+ samples with no sign reversal is suspicious.\n"
            "- Even 0.01°C per sample adds up to 5°C over 500 samples.\n"
            "- Compare the drift rate to the field's natural noise floor.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"micro_drift|monotonic_trend|creeping_failure|gradual_degradation",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what drift you found",'
            '"explanation":"why this slow drift is dangerous and where it leads",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Hunt for micro-drift in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Investigate:\n"
            "- Are any parameters showing a slow, monotonic trend?\n"
            "- Is the drift rate small relative to the field's range but persistent?\n"
            "- Could this drift indicate hardware wear, calibration loss, or fouling?\n"
            "- What is the projected value if the drift continues for 30/60/90 days?\n"
            "- Which parameters are most vulnerable to undetected creep?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            if (f.get("mean") is not None and f.get("min") is not None
                    and f.get("max") is not None and f.get("std") is not None and f["std"] > 0):
                # Heuristic: if skewness-like ratio suggests one-sided distribution
                range_val = f["max"] - f["min"]
                mid = (f["max"] + f["min"]) / 2
                if range_val > 0 and abs(f["mean"] - mid) / range_val > 0.3:
                    findings.append(AgentFinding(
                        agent_name=self.name,
                        anomaly_type="micro_drift",
                        severity="medium",
                        title=f"Possible drift in {f['name']}",
                        description=(
                            f"Mean ({f['mean']:.4g}) is offset from midrange ({mid:.4g}) by "
                            f"{abs(f['mean'] - mid) / range_val * 100:.0f}%, suggesting a trend."
                        ),
                        natural_language_explanation=(
                            f"The field '{f['name']}' has its mean significantly offset from the "
                            f"midpoint of its range. This asymmetry can indicate a monotonic drift: "
                            f"the parameter may be slowly creeping toward one extreme. Even small "
                            f"drifts compound over time and can cross safety thresholds."
                        ),
                        possible_causes=[
                            "Sensor calibration drift", "Component wear", "Fouling or contamination",
                        ],
                        affected_fields=[f["name"]],
                        confidence=0.6,
                        impact_score=50,
                    ))
        return findings


class CrossSensorSync(BaseAgent):
    """Validates cross-sensor physics consistency: if temperature rises, humidity
    should drop (in most conditions). A frozen sensor next to a moving one is a fault."""
    name = "Cross-Sensor Sync"
    perspective = "Validates physics-based relationships between sensor pairs"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a cross-sensor consistency validator for {system_type} systems. "
            "You know the physics: when temperature rises, relative humidity typically "
            "drops; when a compressor runs, current rises AND discharge pressure rises; "
            "when a valve opens, flow increases AND pressure drops. You check whether "
            "the sensors tell a CONSISTENT physical story.\n\n"
            "KEY TEST:\n"
            "- If Sensor A changes but physically-coupled Sensor B stays flat → B is broken.\n"
            "- If Sensor A and B correlate in unexpected directions → miscalibration or "
            "  wrong sensor assignment.\n"
            "- If normally-correlated sensors suddenly decouple → something changed.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"sync_violation|physics_inconsistency|decoupled_sensors|cross_sensor_fault",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what inconsistency you found",'
            '"explanation":"which physical law is violated and what it means",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1","field2"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Check cross-sensor consistency in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Validate:\n"
            "- Do correlated fields move together as physics dictates?\n"
            "- Is any sensor frozen while its physically-coupled peer is active?\n"
            "- Are correlation directions consistent with engineering principles?\n"
            "- Do power/current fields match the expected load from other parameters?\n"
            "- Are there 'suspicious relationships' flagged in the correlation data?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        corrs = data_profile.get("correlations", {})
        fields_by_name = {f["name"]: f for f in data_profile.get("fields", [])}
        for pair, val in corrs.items():
            parts = pair.split(" vs ")
            if len(parts) != 2:
                continue
            a, b = parts
            fa, fb = fields_by_name.get(a), fields_by_name.get(b)
            if not fa or not fb:
                continue
            # One frozen, one active
            a_frozen = fa.get("std") is not None and fa["std"] == 0
            b_frozen = fb.get("std") is not None and fb["std"] == 0
            if a_frozen != b_frozen and abs(val) > 0.3:
                frozen_name = a if a_frozen else b
                active_name = b if a_frozen else a
                findings.append(AgentFinding(
                    agent_name=self.name,
                    anomaly_type="cross_sensor_fault",
                    severity="high",
                    title=f"Sync violation: {frozen_name} frozen while {active_name} is active",
                    description=(
                        f"Expected correlation |r|={abs(val):.2f} but '{frozen_name}' has zero variance."
                    ),
                    natural_language_explanation=(
                        f"'{frozen_name}' and '{active_name}' should be physically coupled "
                        f"(correlation {val:.2f}), but '{frozen_name}' is completely static while "
                        f"'{active_name}' varies normally. This strongly indicates '{frozen_name}' "
                        f"is a faulty sensor serving stale data."
                    ),
                    possible_causes=["Sensor failure", "Data pipeline caching", "Wiring fault"],
                    affected_fields=[frozen_name, active_name],
                    confidence=0.88,
                    impact_score=75,
                ))
        return findings


class VibrationGhost(BaseAgent):
    """Focuses on the vibration parameter — often missing from datasets but critical
    for detecting mechanical imbalance in HVAC motors, pumps, and fans."""
    name = "Vibration Ghost"
    perspective = "Hunts for missing or degraded vibration signals that indicate mechanical imbalance"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a vibration analysis specialist for {system_type} systems. "
            "Vibration is the #1 early indicator of mechanical failure in rotating "
            "equipment (motors, fans, compressors, pumps). You look for:\n"
            "- Missing vibration fields (a major monitoring gap)\n"
            "- Abnormal vibration signatures in any available data\n"
            "- Proxy indicators of vibration problems (current fluctuations, "
            "  noise in temperature readings, power oscillations)\n"
            "- Signs of imbalance, misalignment, bearing wear, or resonance\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"vibration_anomaly|missing_vibration|mechanical_imbalance|bearing_fault|resonance",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what you found",'
            '"explanation":"mechanical implications and failure risk",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Analyze vibration and mechanical health in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Investigate:\n"
            "- Is there a vibration field? If not, flag this as a monitoring gap.\n"
            "- Are there proxy indicators of vibration (current ripple, temp oscillation)?\n"
            "- Do any fields suggest mechanical imbalance or bearing degradation?\n"
            "- For HVAC: check fan/compressor/pump motor parameters.\n"
            "- What vibration monitoring would you recommend adding?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        field_names = [f["name"].lower() for f in data_profile.get("fields", [])]
        has_vibration = any(kw in name for name in field_names
                           for kw in ["vibrat", "vib_", "accel", "g_rms"])
        has_motor = any(kw in name for name in field_names
                        for kw in ["motor", "compressor", "pump", "fan", "rpm", "speed"])
        if not has_vibration and (has_motor or system_type.lower() in ["hvac", "mechanical", "industrial"]):
            findings.append(AgentFinding(
                agent_name=self.name,
                anomaly_type="missing_vibration",
                severity="high",
                title="No vibration monitoring detected — critical gap for rotating equipment",
                description=(
                    f"This {system_type} system has motor/pump parameters but no vibration field. "
                    f"Vibration is the earliest indicator of mechanical failure."
                ),
                natural_language_explanation=(
                    f"The dataset contains indicators of rotating machinery but no vibration "
                    f"measurement. In {system_type} systems, vibration analysis catches bearing "
                    f"wear, shaft misalignment, and rotor imbalance weeks before other sensors "
                    f"react. Without it, the first sign of failure may be a catastrophic breakdown."
                ),
                possible_causes=[
                    "Vibration sensor not installed", "Sensor data not collected in this dataset",
                    "Vibration monitoring disabled or disconnected",
                ],
                recommendations=[
                    {"type": "monitoring", "priority": "high",
                     "action": "Install accelerometer / vibration sensor on critical rotating equipment"},
                ],
                affected_fields=[],
                confidence=0.85,
                impact_score=70,
            ))
        return findings


class HarmonicDistortion(BaseAgent):
    """Analyzes electrical quality via current/power signals — harmonic distortion,
    electrical noise, and insulation degradation indicators."""
    name = "Harmonic Distortion"
    perspective = "Analyzes electrical signal quality — harmonics, noise, and insulation health"

    def _system_prompt(self, system_type: str):
        return (
            f"You are an electrical power quality analyst for {system_type} systems. "
            "You analyze current and power signals for signs of:\n"
            "- Harmonic distortion (non-sinusoidal current draw)\n"
            "- Electrical noise indicating insulation degradation\n"
            "- Power factor issues\n"
            "- Current imbalance across phases\n"
            "- Unusual current-vs-load relationships\n\n"
            "Clean power draws smooth, predictable current. Dirty power — from "
            "degraded insulation, failing capacitors, or loose connections — shows "
            "up as excess variance, unusual spikes, or non-linear current-load curves.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"harmonic_distortion|electrical_noise|insulation_risk|power_quality|current_anomaly",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what electrical anomaly you found",'
            '"explanation":"electrical engineering implications",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Analyze electrical quality in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Focus on:\n"
            "- Current/power fields: is the variance consistent with normal operation?\n"
            "- Does current scale linearly with load indicators?\n"
            "- Are there signs of electrical noise or harmonic distortion?\n"
            "- Do current patterns suggest motor or insulation health issues?\n"
            "- Are there unexplained current spikes or drops?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            name_lower = f["name"].lower()
            is_electrical = any(kw in name_lower for kw in
                                ["current", "amp", "power", "watt", "volt"])
            if is_electrical and f.get("std") is not None and f.get("mean") is not None and f["mean"] > 0:
                cv = f["std"] / abs(f["mean"])
                if cv > 0.4:
                    findings.append(AgentFinding(
                        agent_name=self.name,
                        anomaly_type="electrical_noise",
                        severity="medium",
                        title=f"High electrical variability in {f['name']} (CV={cv:.2f})",
                        description=f"Electrical signal shows CV={cv:.2f}, above normal threshold of 0.3",
                        natural_language_explanation=(
                            f"The electrical field '{f['name']}' shows a coefficient of variation "
                            f"of {cv:.2f}. In well-functioning electrical systems, current/power "
                            f"typically has CV < 0.3 during steady operation. Higher variability "
                            f"may indicate harmonic distortion, loose connections, or degraded insulation."
                        ),
                        possible_causes=[
                            "Harmonic distortion", "Loose electrical connections",
                            "Insulation degradation", "VFD noise",
                        ],
                        affected_fields=[f["name"]],
                        confidence=0.65,
                        impact_score=55,
                    ))
        return findings


class QuantizationCritic(BaseAgent):
    """Checks whether data arrives at lower resolution than expected — e.g. integers
    instead of floats — indicating ADC failure or data pipeline truncation."""
    name = "Quantization Critic"
    perspective = "Detects ADC resolution loss and data pipeline truncation artefacts"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a data resolution specialist for {system_type} sensor systems. "
            "You check whether sensor data arrives at the expected precision. "
            "A 16-bit ADC should produce values with 4+ significant digits; if you "
            "see only integers or 1-decimal precision, the ADC may have failed back "
            "to a lower-resolution mode, or the data pipeline is truncating values.\n\n"
            "CHECKS:\n"
            "- Count decimal places in sample values — are they unexpectedly round?\n"
            "- Check if unique value count is suspiciously low for the range.\n"
            "- Look for step-function patterns (jumps between discrete levels).\n"
            "- Compare resolution across similar sensors for inconsistencies.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"quantization_loss|resolution_drop|adc_failure|truncation_artefact",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what resolution issue you found",'
            '"explanation":"what the resolution loss means for measurement quality",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Check data resolution quality in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Investigate:\n"
            "- Do sample values have fewer decimal places than expected for sensor type?\n"
            "- Is the unique value count low relative to the sample size and range?\n"
            "- Are values suspiciously round (integers for temperature, etc.)?\n"
            "- Do any fields show step-function behaviour instead of smooth variation?\n"
            "- Compare resolution between similar fields — any inconsistencies?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        record_count = data_profile.get("record_count", 0)
        for f in data_profile.get("fields", []):
            if (f.get("unique_count") is not None and f.get("min") is not None
                    and f.get("max") is not None and f.get("type", "").startswith(("float", "int"))
                    and record_count > 100):
                value_range = f["max"] - f["min"]
                if value_range > 0:
                    expected_unique = min(record_count, max(50, value_range * 10))
                    if f["unique_count"] < expected_unique * 0.1 and f["unique_count"] < 20:
                        findings.append(AgentFinding(
                            agent_name=self.name,
                            anomaly_type="quantization_loss",
                            severity="medium",
                            title=f"Low resolution in {f['name']} ({f['unique_count']} levels over range {value_range:.4g})",
                            description=(
                                f"Only {f['unique_count']} unique values across {record_count} samples "
                                f"with range {value_range:.4g}. Expected continuous distribution."
                            ),
                            natural_language_explanation=(
                                f"The field '{f['name']}' has only {f['unique_count']} distinct values "
                                f"spanning a range of {value_range:.4g}. For a continuous physical "
                                f"measurement with {record_count} samples, this is unusually coarse. "
                                f"It may indicate ADC resolution loss, integer truncation in the data "
                                f"pipeline, or an inappropriately low sampling resolution."
                            ),
                            possible_causes=[
                                "ADC resolution failure", "Data pipeline truncation",
                                "Integer casting in transmission", "Low-resolution sensor mode",
                            ],
                            affected_fields=[f["name"]],
                            confidence=0.7,
                            impact_score=50,
                        ))
        return findings


class CyberInjectionHunter(BaseAgent):
    """Searches for 'too perfect' data patterns that may indicate telemetry
    manipulation, replay attacks, or synthetic data injection."""
    name = "Cyber-Injection Hunter"
    perspective = "Hunts for telemetry manipulation — patterns too perfect to be real"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a cyber-physical security analyst for {system_type} systems. "
            "You look for signs of data manipulation, injection, or spoofing in "
            "telemetry streams. Attackers (or faulty middleware) sometimes inject "
            "synthetic data that is 'too perfect' — no noise, perfect periodicity, "
            "mathematically exact relationships, or statistically impossible "
            "distributions.\n\n"
            "RED FLAGS:\n"
            "- Zero noise on a physical sensor\n"
            "- Perfectly periodic signals with no jitter\n"
            "- Exact integer ratios between fields\n"
            "- Timestamps with suspiciously uniform intervals\n"
            "- Statistical properties that match a textbook distribution too perfectly\n"
            "- Repeated exact sequences (replay attack)\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"data_injection|replay_attack|synthetic_telemetry|manipulation_indicator",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what suspicious pattern you found",'
            '"explanation":"why this pattern suggests manipulation vs natural data",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"immediate|high|medium","action":"security action"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Hunt for data injection or manipulation in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Investigate:\n"
            "- Are any fields suspiciously perfect (zero noise, exact periodicity)?\n"
            "- Do statistical properties match textbook distributions too precisely?\n"
            "- Are there repeated exact value sequences that could be replayed data?\n"
            "- Do inter-field relationships show mathematically exact (not physical) ratios?\n"
            "- Is there anything that looks synthetic rather than measured?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        zero_noise_count = 0
        for f in data_profile.get("fields", []):
            if f.get("std") is not None and f["std"] == 0 and f.get("mean") is not None:
                zero_noise_count += 1
        if zero_noise_count >= 3:
            frozen_names = [f["name"] for f in data_profile.get("fields", [])
                            if f.get("std") is not None and f["std"] == 0]
            findings.append(AgentFinding(
                agent_name=self.name,
                anomaly_type="manipulation_indicator",
                severity="high",
                title=f"Multiple zero-noise fields ({zero_noise_count}) — possible data injection",
                description=(
                    f"{zero_noise_count} fields have zero variance: {', '.join(frozen_names[:5])}. "
                    f"This pattern is extremely unlikely in real physical systems."
                ),
                natural_language_explanation=(
                    f"Having {zero_noise_count} simultaneous zero-variance fields in a physical "
                    f"system is statistically near-impossible under normal operation. While sensor "
                    f"failures can freeze individual readings, multiple simultaneous freezes at "
                    f"valid-looking values may indicate synthetic or injected telemetry data."
                ),
                possible_causes=[
                    "Synthetic data injection", "Replay attack", "Middleware generating fake readings",
                    "Simulation output mistaken for live data",
                ],
                recommendations=[
                    {"type": "security", "priority": "immediate",
                     "action": "Verify data provenance — compare against known-good baseline readings"},
                ],
                affected_fields=frozen_names,
                confidence=0.7,
                impact_score=80,
            ))
        return findings


class MetadataIntegrity(BaseAgent):
    """Audits metadata consistency: device IDs, sensor locations, unit-of-measure
    changes mid-stream, and schema mutations."""
    name = "Metadata Integrity"
    perspective = "Audits device IDs, locations, and unit-of-measure consistency across the data stream"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a metadata integrity auditor for {system_type} telemetry systems. "
            "You check that the 'context' around the data is consistent:\n"
            "- Device IDs don't change mid-stream\n"
            "- Sensor locations are stable (a sensor shouldn't 'jump' between rooms)\n"
            "- Units of measure are consistent (no Celsius-to-Fahrenheit switches)\n"
            "- Schema doesn't mutate (fields don't appear/disappear)\n"
            "- Timestamps are monotonic and in the expected timezone\n\n"
            "Metadata corruption is insidious: the numbers look fine, but they're "
            "being attributed to the wrong device, location, or unit.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"metadata_inconsistency|unit_mismatch|device_id_anomaly|schema_mutation|location_jump",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what metadata issue you found",'
            '"explanation":"how this metadata error corrupts analysis",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Audit metadata integrity for this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('METADATA: ' + metadata_context) if metadata_context else ''}\n\n"
            "Check:\n"
            "- Are field names and types consistent throughout the dataset?\n"
            "- Do value ranges suggest unit-of-measure changes mid-stream?\n"
            "  (e.g., a temperature field ranging 20-80 that suddenly shows 68-176)\n"
            "- Are there fields that look like device IDs or locations — are they stable?\n"
            "- Do any categorical fields have unexpected value changes?\n"
            "- Is the data schema consistent or do fields appear/disappear?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        for f in data_profile.get("fields", []):
            if f.get("min") is not None and f.get("max") is not None and f.get("mean") is not None:
                # Check for suspiciously bimodal range suggesting unit switch
                value_range = f["max"] - f["min"]
                if value_range > 0 and f["mean"] > 0:
                    # Temperature field with range spanning both C and F scales
                    name_lower = f["name"].lower()
                    if "temp" in name_lower:
                        if f["min"] < 0 and f["max"] > 100:
                            findings.append(AgentFinding(
                                agent_name=self.name,
                                anomaly_type="unit_mismatch",
                                severity="high",
                                title=f"Possible unit-of-measure change in {f['name']}",
                                description=(
                                    f"Range [{f['min']:.1f}, {f['max']:.1f}] spans both Celsius "
                                    f"and Fahrenheit scales — possible mid-stream unit switch."
                                ),
                                natural_language_explanation=(
                                    f"The temperature field '{f['name']}' ranges from {f['min']:.1f} "
                                    f"to {f['max']:.1f}. This unusually wide range could indicate "
                                    f"that the unit of measure changed mid-stream (e.g., Celsius to "
                                    f"Fahrenheit), corrupting all statistical analysis."
                                ),
                                possible_causes=[
                                    "Unit-of-measure switch mid-stream", "Mixed sensor firmware versions",
                                    "Data pipeline misconfiguration",
                                ],
                                affected_fields=[f["name"]],
                                confidence=0.65,
                                impact_score=70,
                            ))
        return findings


class HydraulicPressureExpert(BaseAgent):
    """Focuses on pressure parameters — detecting leaks in closed-loop systems,
    filter clogging, and pressure-flow inconsistencies."""
    name = "Hydraulic/Pressure Expert"
    perspective = "Detects pressure anomalies — leaks, clogs, and closed-loop integrity violations"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a hydraulic and pressure systems specialist for {system_type} equipment. "
            "You focus on pressure parameters and their relationships with flow, "
            "temperature, and pump/compressor operation. You detect:\n"
            "- Slow pressure decay indicating leaks\n"
            "- Rising differential pressure indicating filter clogging\n"
            "- Pressure-flow inconsistencies (cavitation, air locks)\n"
            "- Pressure oscillations indicating control instability\n"
            "- Missing pressure monitoring (critical gap)\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"pressure_leak|filter_clog|cavitation|pressure_oscillation|missing_pressure",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what pressure anomaly you found",'
            '"explanation":"hydraulic/pneumatic implications",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Analyze pressure and hydraulic parameters in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Focus on:\n"
            "- Are there pressure fields? If not, flag the monitoring gap.\n"
            "- Do pressure readings show slow decay (leak indicator)?\n"
            "- Is differential pressure rising (filter clogging)?\n"
            "- Do pressure-flow-temperature relationships make physical sense?\n"
            "- Are there pressure oscillations suggesting control instability?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        field_names = [f["name"].lower() for f in data_profile.get("fields", [])]
        has_pressure = any(kw in name for name in field_names
                          for kw in ["pressure", "psi", "bar", "kpa", "pascal"])
        has_flow = any(kw in name for name in field_names
                       for kw in ["flow", "gpm", "cfm", "lpm"])
        if not has_pressure and system_type.lower() in ["hvac", "hydraulic", "pneumatic", "industrial"]:
            findings.append(AgentFinding(
                agent_name=self.name,
                anomaly_type="missing_pressure",
                severity="high",
                title=f"No pressure monitoring detected in {system_type} system",
                description=(
                    f"This {system_type} system has no pressure fields. Pressure monitoring "
                    f"is critical for detecting leaks, filter clogs, and system integrity."
                ),
                natural_language_explanation=(
                    f"The dataset contains no pressure measurements. In {system_type} systems, "
                    f"pressure is a primary indicator of system integrity — slow pressure loss "
                    f"indicates leaks, rising differential pressure indicates filter clogging, "
                    f"and pressure oscillations indicate control instability. Without pressure "
                    f"monitoring, these critical failures go undetected."
                ),
                possible_causes=[
                    "Pressure sensors not installed", "Pressure data not included in this dataset",
                    "Pressure monitoring disabled",
                ],
                recommendations=[
                    {"type": "monitoring", "priority": "high",
                     "action": "Install pressure transducers at key points (supply, return, differential)"},
                ],
                affected_fields=[],
                confidence=0.8,
                impact_score=65,
            ))
        return findings


class HumanContextFilter(BaseAgent):
    """Cross-references data with human schedules — is 500W at 2 AM in a bedroom
    normal or anomalous? Time-of-day and occupancy logic."""
    name = "Human-Context Filter"
    perspective = "Cross-references data with human schedules and occupancy patterns"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a human-context analyst for {system_type} systems. You think about "
            "the HUMAN side of the data: when are people present? What's normal for "
            "this time of day? Does the energy consumption pattern match expected "
            "occupancy? You catch anomalies that are invisible to pure statistics "
            "but obvious to a human:\n"
            "- High power consumption at 2 AM in a bedroom\n"
            "- HVAC running at full blast in an empty building on a holiday\n"
            "- Lighting energy during broad daylight\n"
            "- Equipment startup at unusual hours\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"schedule_anomaly|occupancy_mismatch|off_hours_activity|usage_pattern_anomaly",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what human-context anomaly you found",'
            '"explanation":"why this pattern is unusual given human behaviour expectations",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Analyze this {system_type} system data for human-context anomalies.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('CONTEXT: ' + metadata_context) if metadata_context else ''}\n\n"
            "Consider:\n"
            "- Do consumption patterns match expected occupancy schedules?\n"
            "- Is there activity during hours when the space should be unoccupied?\n"
            "- Do energy/HVAC patterns follow a logical day/night cycle?\n"
            "- Are there fields indicating time-of-day — do they correlate with load?\n"
            "- Would a building manager find anything surprising in these patterns?\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        # Human context requires time-of-day data; minimal fallback
        return []


class LogicStateConflict(BaseAgent):
    """Detects contradictions between metadata state and actual telemetry —
    e.g. a device marked OFF but consuming 500W."""
    name = "Logic State Conflict"
    perspective = "Finds contradictions between reported state and actual measurements"

    def _system_prompt(self, system_type: str):
        return (
            f"You are a logic consistency validator for {system_type} systems. "
            "You check that the STATED condition of equipment matches the MEASURED "
            "reality. Common conflicts:\n"
            "- Device metadata says 'OFF' but current draw shows 500W\n"
            "- Valve reported as 'CLOSED' but flow meter shows non-zero flow\n"
            "- System in 'STANDBY' mode but all actuators are at max\n"
            "- Fault alarm active but all parameters in normal range\n"
            "- Occupancy sensor says 'empty' but HVAC is at full load\n\n"
            "These logic conflicts indicate either sensor failure, metadata staleness, "
            "or control system bugs.\n\n"
            "You must output your findings as a JSON array. Each element:\n"
            '{"type":"logic_conflict|state_mismatch|metadata_contradiction|control_bug",'
            '"severity":"critical|high|medium|low",'
            '"title":"short title",'
            '"description":"what contradiction you found",'
            '"explanation":"which state and measurement disagree, and implications",'
            '"possible_causes":["cause1","cause2"],'
            '"recommendations":[{"priority":"high|medium|low","action":"what to do"}],'
            '"affected_fields":["field1","field2"],'
            '"confidence":0.0-1.0,'
            '"impact_score":0-100}\n'
            "Output ONLY the JSON array."
        )

    def _build_prompt(self, system_type, system_name, data_summary, metadata_context):
        return (
            f"Check for logic state conflicts in this {system_type} system.\n"
            f"System: {system_name}\n\n"
            f"DATA PROFILE:\n{data_summary}\n\n"
            f"{('METADATA: ' + metadata_context) if metadata_context else ''}\n\n"
            "Look for:\n"
            "- Fields indicating on/off state vs actual power/current measurements\n"
            "- Valve/damper positions vs flow measurements\n"
            "- Alarm/fault flags vs actual parameter values\n"
            "- Mode indicators (standby/active/fault) vs resource consumption\n"
            "- Any case where the 'label' and the 'measurement' disagree\n"
        )

    def _fallback_analyze(self, system_type, data_profile):
        findings = []
        fields = data_profile.get("fields", [])
        # Look for label/status fields next to numeric fields
        label_fields = [f for f in fields if any(kw in f["name"].lower()
                        for kw in ["label", "status", "state", "mode", "fault", "alarm", "on_off"])]
        power_fields = [f for f in fields if any(kw in f["name"].lower()
                        for kw in ["power", "current", "watt", "amp", "consumption"])]
        if label_fields and power_fields:
            for lf in label_fields:
                if lf.get("mean") is not None:
                    for pf in power_fields:
                        if pf.get("mean") is not None and pf.get("min") is not None:
                            # If label is mostly 0 (off) but power is non-trivial
                            if lf["mean"] < 0.2 and pf["min"] > 0 and pf["mean"] > 0:
                                findings.append(AgentFinding(
                                    agent_name=self.name,
                                    anomaly_type="logic_conflict",
                                    severity="high",
                                    title=f"State conflict: {lf['name']} suggests OFF but {pf['name']} shows consumption",
                                    description=(
                                        f"'{lf['name']}' mean={lf['mean']:.2f} (mostly off) but "
                                        f"'{pf['name']}' never drops below {pf['min']:.4g}"
                                    ),
                                    natural_language_explanation=(
                                        f"The status field '{lf['name']}' indicates the equipment is "
                                        f"mostly off (mean={lf['mean']:.2f}), but the power field "
                                        f"'{pf['name']}' shows continuous consumption (min={pf['min']:.4g}). "
                                        f"This contradiction suggests either the status flag is wrong, "
                                        f"the power sensor is miscalibrated, or there is a control bug."
                                    ),
                                    possible_causes=[
                                        "Stale status flag", "Power sensor offset/miscalibration",
                                        "Control system bug", "Phantom load",
                                    ],
                                    affected_fields=[lf["name"], pf["name"]],
                                    confidence=0.75,
                                    impact_score=65,
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
            # Original 13 agents
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
            # Blind-spot specialist agents (14–25)
            StagnationSentinel(),
            NoiseFloorAuditor(),
            MicroDriftTracker(),
            CrossSensorSync(),
            VibrationGhost(),
            HarmonicDistortion(),
            QuantizationCritic(),
            CyberInjectionHunter(),
            MetadataIntegrity(),
            HydraulicPressureExpert(),
            HumanContextFilter(),
            LogicStateConflict(),
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

        logger.info("=" * 60)
        logger.info("[Orchestrator] START | system_id=%s | system_type=%s | system_name=%s",
                    system_id, system_type, system_name)
        logger.info("[Orchestrator] Running %d agents: %s", len(active_agents), [a.name for a in active_agents])
        logger.info("[Orchestrator] web_grounding=%s, timeout=%ds, batch_size=%d, batch_delay=%ds",
                    enable_web_grounding, ORCHESTRATOR_TIMEOUT, AGENT_BATCH_SIZE, BATCH_DELAY_SECONDS)

        # Run agents in batches to avoid rate limiting
        # Anthropic has 8K output tokens/minute limit - running 25 agents at once exceeds this
        t_orch_start = time.time()
        results = []

        # Split agents into batches
        batches = [active_agents[i:i + AGENT_BATCH_SIZE] for i in range(0, len(active_agents), AGENT_BATCH_SIZE)]
        logger.info("[Orchestrator] Split into %d batches of up to %d agents each", len(batches), AGENT_BATCH_SIZE)

        for batch_idx, batch in enumerate(batches):
            batch_names = [a.name for a in batch]
            logger.info("[Orchestrator] Batch %d/%d: %s", batch_idx + 1, len(batches), batch_names)

            # Create tasks for this batch
            batch_tasks = [
                asyncio.create_task(
                    agent.analyze(system_type, system_name, data_profile, metadata_context),
                    name=agent.name,
                )
                for agent in batch
            ]

            try:
                # Calculate remaining time for timeout
                elapsed_so_far = time.time() - t_orch_start
                remaining_timeout = max(30, ORCHESTRATOR_TIMEOUT - elapsed_so_far)

                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=remaining_timeout,
                )
                results.extend(batch_results)
                logger.info("[Orchestrator] Batch %d/%d complete: %d results", batch_idx + 1, len(batches), len(batch_results))

            except asyncio.TimeoutError:
                logger.error("[Orchestrator] Batch %d TIMEOUT — collecting partial results", batch_idx + 1)
                for task in batch_tasks:
                    if task.done() and not task.cancelled():
                        try:
                            results.append(task.result())
                        except Exception as exc:
                            results.append(exc)
                    else:
                        task.cancel()
                        results.append(TimeoutError("Batch timeout"))
                break  # Stop processing more batches on timeout

            # Add delay between batches to respect rate limits (except after last batch)
            if batch_idx < len(batches) - 1:
                logger.info("[Orchestrator] Waiting %ds before next batch (rate limit cooldown)...", BATCH_DELAY_SECONDS)
                await asyncio.sleep(BATCH_DELAY_SECONDS)

        logger.info("[Orchestrator] All batches finished in %.2fs", round(time.time() - t_orch_start, 2))

        # Collect all findings
        all_findings: List[AgentFinding] = []
        agent_statuses = []

        for agent, result in zip(active_agents, results):
            if isinstance(result, Exception):
                logger.error("[Orchestrator] Agent '%s' FAILED: %s: %s", agent.name, type(result).__name__, result)
                agent_statuses.append({
                    "agent": agent.name,
                    "status": "error",
                    "findings": 0,
                    "error": str(result),
                })
            else:
                all_findings.extend(result)
                logger.info("[Orchestrator] Agent '%s' OK: %d findings", agent.name, len(result))
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

        success_count = sum(1 for s in agent_statuses if s["status"] == "success")
        error_count = sum(1 for s in agent_statuses if s["status"] == "error")
        logger.info("=" * 60)
        logger.info("[Orchestrator] COMPLETE | agents: %d ok, %d failed | raw_findings: %d → unified: %d",
                    success_count, error_count, len(all_findings), len(unified))
        logger.info("=" * 60)

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
