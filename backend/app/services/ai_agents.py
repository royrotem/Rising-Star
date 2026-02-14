"""
Multi-Agent AI Analysis System for UAIE — re-export hub.

Agents are split into sub-modules for maintainability:
  - ai_agents_base.py        — constants, dataclasses, BaseAgent
  - ai_agents_core.py        — 13 core agents (1–13)
  - ai_agents_specialized.py — 12 blind-spot specialists (14–25)

This file keeps the AgentOrchestrator (which wires all agents together)
and re-exports every public symbol so that existing imports such as
    from .ai_agents import AgentFinding, BaseAgent, orchestrator
continue to work unchanged.
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# ── Re-export shared infrastructure from base module ─────────────────
from .ai_agents_base import (                     # noqa: F401
    AgentFinding,
    UnifiedAnomaly,
    BaseAgent,
    _get_api_key,
    web_search,
    logger,
    AGENT_TIMEOUT,
    ORCHESTRATOR_TIMEOUT,
    AGENT_BATCH_SIZE,
    BATCH_DELAY_SECONDS,
    ALL_AGENT_NAMES,
    HAS_ANTHROPIC,
    HAS_HTTPX,
)

# ── Re-export core agents ────────────────────────────────────────────
from .ai_agents_core import (                     # noqa: F401
    StatisticalAnalyst,
    DomainExpert,
    PatternDetective,
    RootCauseInvestigator,
    SafetyAuditor,
    TemporalAnalyst,
    DataQualityInspector,
    PredictiveForecaster,
    OperationalProfiler,
    EfficiencyAnalyst,
    ComplianceChecker,
    ReliabilityEngineer,
    EnvironmentalCorrelator,
)

# ── Re-export specialized blind-spot agents ──────────────────────────
from .ai_agents_specialized import (              # noqa: F401
    StagnationSentinel,
    NoiseFloorAuditor,
    MicroDriftTracker,
    CrossSensorSync,
    VibrationGhost,
    HarmonicDistortion,
    QuantizationCritic,
    CyberInjectionHunter,
    MetadataIntegrity,
    HydraulicPressureExpert,
    HumanContextFilter,
    LogicStateConflict,
)


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
            # Original 13 prompt-based agents
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
        self._agentic_added = False

    async def run_analysis(
        self,
        system_id: str,
        system_type: str,
        system_name: str,
        data_profile: Dict,
        metadata_context: str = "",
        enable_web_grounding: bool = True,
        selected_agents: Optional[List[str]] = None,
        on_batch_complete: Optional[Any] = None,
        raw_records: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Run agents in parallel and unify results.

        Args:
            selected_agents: If provided, only run agents whose names are in this list.
                             If None or empty, all agents are run.
            raw_records: Raw data records — passed to agentic detectors so they
                         can explore data via tool calls.  Prompt-based agents
                         still receive only the data_profile summary.

        Has a global timeout of ORCHESTRATOR_TIMEOUT seconds.  Individual
        agents also have their own AGENT_TIMEOUT.
        """
        # Lazily add agentic detectors (26–30) on first call with raw data
        if raw_records and not self._agentic_added:
            from .agentic_detectors import (
                AgenticDetector, DataToolkit,
                AutonomousExplorer, StatisticalInvestigator,
                CorrelationHunter, DriftShiftDetector, PhysicsConstraintAgent,
            )
            self.agents.extend([
                AutonomousExplorer(),
                StatisticalInvestigator(),
                CorrelationHunter(),
                DriftShiftDetector(),
                PhysicsConstraintAgent(),
            ])
            self._agentic_added = True

        # Set up toolkit for agentic detectors if raw data is available
        if raw_records:
            from .agentic_detectors import AgenticDetector, DataToolkit, get_available_providers
            import pandas as _pd
            toolkit = DataToolkit(_pd.DataFrame(raw_records))
            agentic_agents = [a for a in self.agents if isinstance(a, AgenticDetector)]

            # Distribute agents across all available providers (round-robin)
            available = get_available_providers()
            if len(available) > 1:
                for idx, agent in enumerate(agentic_agents):
                    if agent.llm_provider is None:  # only override auto
                        agent.llm_provider = available[idx % len(available)]
                logger.info(
                    "[Orchestrator] Distributed %d agentic detectors across %d providers: %s",
                    len(agentic_agents), len(available),
                    {a.name: a.llm_provider for a in agentic_agents},
                )

            for agent in agentic_agents:
                agent.set_toolkit(toolkit)

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

                if on_batch_complete:
                    await on_batch_complete(batch_idx + 1, len(batches), batch, batch_results)

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

    def get_agent_status(self) -> Dict[str, Any]:
        """Return status info about all registered agents."""
        return {
            "total_agents": len(self.agents),
            "agents": [{"name": a.name, "perspective": a.perspective} for a in self.agents],
            "ai_available": HAS_ANTHROPIC and bool(_get_api_key()),
        }

    async def stop(self):
        """Graceful shutdown."""
        logger.info("[Orchestrator] Stopped.")


# Global instance
orchestrator = AgentOrchestrator()
