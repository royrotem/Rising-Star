"""
Multi-Agent AI Analysis System for UAIE — base module.

Shared imports, constants, utility functions, dataclasses, and BaseAgent class.
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
    # Agentic detectors (tool-using, multi-turn)
    "Autonomous Explorer",
    "Statistical Investigator",
    "Correlation Hunter",
    "Drift & Shift Detector",
    "Physics Constraint Agent",
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
