"""
AI Chat Service for UAIE

Provides conversational AI capabilities backed by Claude, with full
system data context awareness. Persists conversation history to disk.
This is an additive feature module — removing it does not affect core
analysis or query endpoints.
"""

import json
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils import sanitize_for_json

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def _get_api_key() -> str:
    try:
        from ..api.app_settings import get_anthropic_api_key
        return get_anthropic_api_key()
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


# ── Conversation persistence ────────────────────────────────────────

class ConversationStore:
    """File-based conversation history storage, thread-safe."""

    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR", "/app/data")
            if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
                data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self.conv_dir = Path(data_dir) / "conversations"
        self.conv_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, system_id: str) -> Path:
        return self.conv_dir / f"{system_id}.json"

    def get_messages(self, system_id: str, limit: int = 50) -> List[Dict]:
        with self._lock:
            p = self._path(system_id)
            if not p.exists():
                return []
            try:
                msgs = json.loads(p.read_text())
            except Exception:
                return []
            return msgs[-limit:]

    def add_message(self, system_id: str, role: str, content: str,
                    data: Dict | None = None) -> Dict:
        msg = {
            "id": f"msg-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        with self._lock:
            p = self._path(system_id)
            msgs: list = []
            if p.exists():
                try:
                    msgs = json.loads(p.read_text())
                except Exception:
                    msgs = []
            msgs.append(msg)
            # Keep last 200 messages per system
            if len(msgs) > 200:
                msgs = msgs[-200:]
            p.write_text(json.dumps(msgs, indent=2, default=str))
        return msg

    def clear(self, system_id: str) -> None:
        with self._lock:
            p = self._path(system_id)
            if p.exists():
                p.unlink()


conversation_store = ConversationStore()


# ── Chat service ────────────────────────────────────────────────────

class ChatService:
    """
    AI-powered conversational interface for system data.

    Uses Claude with full context: system metadata, field statistics,
    recent analysis results, and conversation history.
    """

    MODEL = "claude-sonnet-4-20250514"

    def _get_client(self):
        api_key = _get_api_key()
        if HAS_ANTHROPIC and api_key:
            return anthropic.Anthropic(api_key=api_key)
        return None

    # ── Context builders ────────────────────────────────────────────

    def _build_system_context(self, system: Dict, records: List[Dict],
                              schema: list,
                              analysis: Optional[Dict] = None) -> str:
        """Build a concise context string about the system & its data."""
        lines = [
            f"System: {system.get('name', 'Unknown')}",
            f"Type: {system.get('system_type', 'unknown')}",
            f"Status: {system.get('status', 'active')}",
            f"Health Score: {system.get('health_score', 'N/A')}",
        ]

        meta = system.get("metadata", {})
        if meta.get("description"):
            lines.append(f"Description: {meta['description']}")

        if not records:
            lines.append("\nNo data ingested yet.")
            return "\n".join(lines)

        df = pd.DataFrame(records)
        lines.append(f"\nData: {len(df)} records, {len(df.columns)} fields")

        # Field stats
        lines.append("\nFields:")
        for col in df.columns[:30]:
            dtype = str(df[col].dtype)
            if df[col].dtype in ["int64", "float64"]:
                try:
                    lines.append(
                        f"  - {col} ({dtype}): "
                        f"min={df[col].min():.4g}, max={df[col].max():.4g}, "
                        f"mean={df[col].mean():.4g}, std={df[col].std():.4g}"
                    )
                except Exception:
                    lines.append(f"  - {col} ({dtype})")
            else:
                nunique = df[col].nunique()
                lines.append(f"  - {col} ({dtype}): {nunique} unique values")

        # Sample rows
        lines.append("\nSample rows (first 3):")
        for row in df.head(3).to_dict("records"):
            lines.append(f"  {json.dumps(sanitize_for_json(row), default=str)[:400]}")

        # Schema field info
        if schema:
            confirmed = [f for f in schema if isinstance(f, dict) and f.get("confirmed")]
            if confirmed:
                lines.append("\nConfirmed field info:")
                for f in confirmed[:15]:
                    unit = f.get("physical_unit") or f.get("confirmed_unit", "")
                    meaning = f.get("inferred_meaning") or f.get("confirmed_meaning", "")
                    if unit or meaning:
                        lines.append(f"  - {f.get('name', '?')}: {meaning} ({unit})")

        # ── Saved analysis results (from last analysis run) ──────────
        if analysis:
            lines.append("\n--- LAST ANALYSIS RESULTS ---")
            lines.append(f"Analysis timestamp: {analysis.get('timestamp', 'unknown')}")
            if analysis.get("health_score") is not None:
                lines.append(f"Computed health score: {analysis['health_score']}")

            # Anomalies
            anomalies = analysis.get("anomalies", [])
            if anomalies:
                lines.append(f"\nDetected anomalies ({len(anomalies)} total):")
                for i, a in enumerate(anomalies[:15], 1):
                    sev = a.get("severity", "unknown")
                    title = a.get("title", a.get("description", "Unnamed"))
                    impact = a.get("impact_score", "N/A")
                    fields = ", ".join(a.get("affected_fields", [])[:3])
                    explanation = a.get("natural_language_explanation", "")
                    lines.append(f"  {i}. [{sev.upper()}] {title}")
                    if fields:
                        lines.append(f"     Affected fields: {fields}")
                    lines.append(f"     Impact: {impact}")
                    if explanation:
                        lines.append(f"     Explanation: {explanation[:300]}")
                    causes = a.get("possible_causes", [])
                    if causes:
                        lines.append(f"     Possible causes: {'; '.join(causes[:3])}")
                    recs = a.get("recommendations", [])
                    if recs:
                        lines.append(f"     Recommendations: {'; '.join(recs[:3])}")
            else:
                lines.append("\nNo anomalies detected in last analysis.")

            # Engineering margins
            margins = analysis.get("engineering_margins", [])
            if margins:
                lines.append(f"\nEngineering margins ({len(margins)} tracked):")
                for m in margins[:10]:
                    comp = m.get("component", m.get("field", "unknown"))
                    param = m.get("parameter", "")
                    pct = m.get("margin_percentage", "N/A")
                    trend = m.get("trend", "stable")
                    breach = m.get("projected_breach_date")
                    safety = m.get("safety_critical", False)
                    label = f"{comp}" + (f" / {param}" if param else "")
                    lines.append(f"  - {label}: margin={pct}%, trend={trend}"
                                 + (f", projected breach: {breach}" if breach else "")
                                 + (" [SAFETY CRITICAL]" if safety else ""))

            # Blind spots
            blind_spots = analysis.get("blind_spots", [])
            if blind_spots:
                lines.append(f"\nBlind spots / data gaps ({len(blind_spots)}):")
                for bs in blind_spots[:5]:
                    title = bs.get("title", bs.get("description", "Unknown"))
                    sensor = bs.get("recommended_sensor", "")
                    improvement = bs.get("diagnostic_coverage_improvement", "")
                    lines.append(f"  - {title}")
                    if sensor:
                        lines.append(f"    Recommended sensor: {sensor}")
                    if improvement:
                        lines.append(f"    Diagnostic improvement: {improvement}%")

            # Insights & recommendations from analysis
            insights = analysis.get("insights", [])
            if isinstance(insights, list) and insights:
                lines.append(f"\nInsights ({len(insights)}):")
                for ins in insights[:5]:
                    if isinstance(ins, dict):
                        lines.append(f"  - {ins.get('title', ins.get('description', str(ins)))}")
                    else:
                        lines.append(f"  - {str(ins)[:200]}")

            recs = analysis.get("recommendations", [])
            if isinstance(recs, list) and recs:
                lines.append(f"\nRecommendations ({len(recs)}):")
                for r in recs[:5]:
                    if isinstance(r, dict):
                        lines.append(f"  - {r.get('title', r.get('description', str(r)))}")
                    else:
                        lines.append(f"  - {str(r)[:200]}")

            # AI agent analysis summary
            ai_analysis = analysis.get("ai_analysis", {})
            if isinstance(ai_analysis, dict):
                agent_count = ai_analysis.get("agents_completed", 0)
                if agent_count:
                    lines.append(f"\nAI agents that ran: {agent_count}")
                summary = ai_analysis.get("summary", "")
                if summary:
                    lines.append(f"AI summary: {str(summary)[:500]}")
        else:
            lines.append("\n--- NO ANALYSIS RUN YET ---")
            lines.append("Analysis has not been run on this system yet. "
                         "The user may ask you to help interpret raw data or suggest running analysis.")

        return "\n".join(lines)

    def _system_prompt(self) -> str:
        return (
            "You are the UAIE Conversational Chief Engineer — an AI assistant "
            "embedded in the Universal Autonomous Insight Engine platform. "
            "You help engineers understand their system's telemetry data, "
            "investigate anomalies, and make data-driven decisions.\n\n"
            "You have access to:\n"
            "- System metadata (name, type, status, health score)\n"
            "- Raw ingested data with field statistics (min, max, mean, std)\n"
            "- Schema information with physical units and inferred meanings\n"
            "- Saved analysis results (if analysis has been run), including:\n"
            "  - Detected anomalies with severity, impact, root causes, and recommendations\n"
            "  - Engineering margins with trend and projected breach dates\n"
            "  - Blind spots / data gaps with recommended sensors\n"
            "  - Insights and recommendations from the analysis engine\n"
            "- Conversation history for continuity\n\n"
            "Guidelines:\n"
            "- Be concise but thorough. Engineers appreciate precision.\n"
            "- Reference specific field names, values, and statistics from the data.\n"
            "- When explaining anomalies, reference the detected anomalies from analysis results.\n"
            "- Provide possible root causes and actionable next steps.\n"
            "- If analysis hasn't been run yet, suggest the user run analysis first.\n"
            "- If you don't have enough data to answer, say so clearly.\n"
            "- Format important values with **bold** and use bullet points for lists.\n"
            "- When asked about statistics, compute or reference the provided data.\n"
            "- Reference engineering margins and blind spots when relevant.\n"
            "- You can reference field correlations and trends if available.\n"
        )

    # ── Main chat method ────────────────────────────────────────────

    async def chat(
        self,
        system_id: str,
        user_message: str,
        system: Dict,
        records: List[Dict],
        schema: list,
        analysis: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message and return an AI response.

        Always runs the local analysis engine first so that pre-computed
        statistics are available.  When an API key is present, Claude
        receives both the raw data context AND the engine's computed
        answer so it can reference real numbers while adding deeper
        insight.  Without an API key the engine answer is returned
        directly.
        """
        # Persist user message
        conversation_store.add_message(system_id, "user", user_message)

        # Always compute the local analysis engine answer first
        engine_result = self._fallback_response(user_message, system, records, analysis)

        # Build context (includes analysis results if available)
        system_context = self._build_system_context(system, records, schema, analysis)
        history = conversation_store.get_messages(system_id, limit=20)

        # Try AI-powered response (with engine answer as extra context)
        client = self._get_client()
        if client:
            response = await self._ai_response(
                client, user_message, system_context, history,
                engine_result["content"],
            )
        else:
            response = engine_result

        # Persist assistant response
        conversation_store.add_message(
            system_id, "assistant", response["content"], response.get("data")
        )

        return response

    async def _ai_response(
        self,
        client,
        user_message: str,
        system_context: str,
        history: List[Dict],
        engine_analysis: str = "",
    ) -> Dict[str, Any]:
        """Generate response using Claude, enriched with pre-computed analysis."""
        # Build message history for context (last 10 turns)
        messages = []
        for msg in history[-10:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Make sure the last message is the current user message
        if not messages or messages[-1]["content"] != user_message:
            messages.append({"role": "user", "content": user_message})

        # Ensure messages alternate and start with user
        clean_messages = []
        for msg in messages:
            if clean_messages and clean_messages[-1]["role"] == msg["role"]:
                # Merge consecutive same-role messages
                clean_messages[-1]["content"] += "\n" + msg["content"]
            else:
                clean_messages.append(msg)
        if clean_messages and clean_messages[0]["role"] != "user":
            clean_messages = clean_messages[1:]

        if not clean_messages:
            clean_messages = [{"role": "user", "content": user_message}]

        # Build system prompt with all available context
        full_system = self._system_prompt()
        full_system += f"\n\n--- SYSTEM DATA CONTEXT ---\n{system_context}"
        if engine_analysis:
            full_system += (
                "\n\n--- PRE-COMPUTED ANALYSIS (from local engine) ---\n"
                "The following analysis was computed by the local statistical engine "
                "for the user's current question.  Use these real numbers and findings "
                "as a foundation — reference them, expand on them, add deeper "
                "engineering insight, root-cause hypotheses, and actionable "
                "recommendations.  Do NOT simply repeat the engine output verbatim; "
                "add value on top of it.\n\n"
                f"{engine_analysis}"
            )

        try:
            resp = client.messages.create(
                model=self.MODEL,
                max_tokens=2048,
                system=full_system,
                messages=clean_messages,
            )
            content = resp.content[0].text if resp.content else "I couldn't generate a response."

            return {
                "content": content,
                "ai_powered": True,
                "model": self.MODEL,
                "data": {"type": "ai_response"},
            }
        except Exception as e:
            print(f"[ChatService] Claude API error: {e}")
            # Fall back to the engine result on API error
            return {
                "content": (
                    f"AI service unavailable ({e}), showing local analysis instead:\n\n"
                    f"{engine_analysis}"
                ),
                "ai_powered": False,
                "data": {"type": "error_with_fallback", "error": str(e)},
            }

    # ── Comprehensive fallback engine (no API key needed) ───────────

    def _fallback_response(
        self,
        user_message: str,
        system: Dict,
        records: List[Dict],
        analysis: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive data-aware response engine — works entirely without
        an API key.  Uses pandas/numpy to compute real statistics, detect
        outliers, summarise health, find correlations, and more.
        When analysis results are available, uses them for richer answers.
        """
        query = user_message.lower()

        if not records:
            return {
                "content": (
                    "No data has been ingested for this system yet. "
                    "Please upload telemetry data first, then I can help "
                    "you analyze it."
                ),
                "ai_powered": False,
                "data": {"type": "no_data"},
            }

        df = pd.DataFrame(records)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # ── Check if asking about a specific field ────────────────
        target_field = self._detect_field_mention(query, df.columns.tolist())

        # ── Route to the best handler ─────────────────────────────
        if any(w in query for w in ["anomal", "problem", "issue", "wrong", "unusual", "strange", "weird", "outlier"]):
            return self._fb_anomalies(df, numeric_cols, system, analysis)

        if any(w in query for w in ["margin", "safety", "limit", "breach", "critical"]):
            return self._fb_margins(system, analysis)

        if any(w in query for w in ["blind spot", "gap", "missing", "sensor"]):
            return self._fb_blind_spots(analysis)

        if any(w in query for w in ["health", "status", "overview", "summar", "how is", "how's"]):
            return self._fb_health_summary(df, numeric_cols, system)

        if any(w in query for w in ["correlat", "relation", "connect", "between"]):
            return self._fb_correlations(df, numeric_cols)

        if any(w in query for w in ["recommend", "suggest", "what should", "action", "improve", "fix"]):
            return self._fb_recommendations(df, numeric_cols, system)

        if any(w in query for w in ["variance", "variab", "volatile", "stable", "stability", "deviation"]):
            return self._fb_variance(df, numeric_cols)

        if any(w in query for w in ["trend", "increas", "decreas", "direction", "going up", "going down", "change"]):
            return self._fb_trends(df, numeric_cols)

        if any(w in query for w in ["distribut", "histogram", "spread", "range"]):
            return self._fb_distribution(df, numeric_cols, target_field)

        if any(w in query for w in ["average", "mean"]):
            return self._fb_stat(df, numeric_cols, "mean", target_field)

        if any(w in query for w in ["max", "maximum", "highest", "peak"]):
            return self._fb_stat(df, numeric_cols, "max", target_field)

        if any(w in query for w in ["min", "minimum", "lowest"]):
            return self._fb_stat(df, numeric_cols, "min", target_field)

        if any(w in query for w in ["std", "standard dev"]):
            return self._fb_stat(df, numeric_cols, "std", target_field)

        if any(w in query for w in ["median"]):
            return self._fb_stat(df, numeric_cols, "median", target_field)

        if target_field:
            return self._fb_field_detail(df, target_field, numeric_cols)

        if any(w in query for w in ["field", "column", "show", "find", "list", "get", "data", "what do you know", "what can"]):
            return self._fb_data_overview(df, numeric_cols, system)

        # Generic fallback — still useful
        return self._fb_data_overview(df, numeric_cols, system)

    # ── Fallback handler: anomalies ──────────────────────────────

    def _fb_anomalies(self, df: pd.DataFrame, numeric_cols: list, system: Dict,
                      analysis: Optional[Dict] = None) -> Dict:
        lines = []
        findings = []

        # If we have saved analysis results, show those first (they're richer)
        saved_anomalies = (analysis or {}).get("anomalies", [])
        if saved_anomalies:
            lines.append(f"**{len(saved_anomalies)} anomalies** detected from the last analysis:\n")
            for i, a in enumerate(saved_anomalies[:10], 1):
                sev = a.get("severity", "unknown").upper()
                title = a.get("title", a.get("description", "Unnamed"))
                impact = a.get("impact_score", 0)
                explanation = a.get("natural_language_explanation", "")
                lines.append(f"**{i}. [{sev}] {title}** (impact: {impact})")
                if explanation:
                    lines.append(f"   {explanation[:250]}")
                causes = a.get("possible_causes", [])
                if causes:
                    lines.append(f"   Possible causes: {', '.join(causes[:3])}")
                recs = a.get("recommendations", [])
                if recs:
                    lines.append(f"   Recommendations: {', '.join(recs[:2])}")
                lines.append("")
                findings.append({
                    "title": title, "severity": sev,
                    "impact": impact, "source": "analysis_engine"
                })

            lines.append(f"System health score: **{system.get('health_score', 'N/A')}**")
            return {
                "content": "\n".join(lines),
                "ai_powered": False,
                "data": {"type": "anomaly_scan", "findings": sanitize_for_json(findings),
                         "source": "saved_analysis"},
            }

        # Fallback: compute from raw data
        lines.append("Here are the **anomalies** I detected by analyzing the raw data:\n")

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            mean, std = float(series.mean()), float(series.std())
            if std == 0:
                continue

            # Z-score outliers (|z| > 2.5)
            z_scores = ((series - mean) / std).abs()
            outlier_count = int((z_scores > 2.5).sum())
            if outlier_count > 0:
                worst = float(series[z_scores.idxmax()])
                z_val = float(z_scores.max())
                sev = "critical" if z_val > 4 else "high" if z_val > 3 else "medium"
                lines.append(
                    f"**{col}** — {outlier_count} outlier{'s' if outlier_count > 1 else ''} detected "
                    f"(worst value: **{worst:.4g}**, z-score: {z_val:.1f}, severity: {sev})"
                )
                findings.append({"field": col, "outliers": outlier_count, "severity": sev, "max_z": round(z_val, 2)})

            # Coefficient of variation
            cv = std / abs(mean) if mean != 0 else 0
            if cv > 0.5 and outlier_count == 0:
                lines.append(
                    f"**{col}** — High variability (CV = {cv:.2f}), may indicate "
                    f"mixed operating modes or instability"
                )
                findings.append({"field": col, "type": "high_variance", "cv": round(cv, 2)})

        if not findings:
            lines.append("No significant statistical anomalies detected — the data looks healthy.")

        lines.append(f"\nSystem health score: **{system.get('health_score', 'N/A')}**")
        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "anomaly_scan", "findings": sanitize_for_json(findings)},
        }

    # ── Fallback handler: engineering margins ─────────────────────

    def _fb_margins(self, system: Dict, analysis: Optional[Dict] = None) -> Dict:
        margins = (analysis or {}).get("engineering_margins", [])
        if not margins:
            return {
                "content": (
                    "No engineering margin data available yet. "
                    "Run a full analysis first to compute safety margins for each parameter."
                ),
                "ai_powered": False,
                "data": {"type": "margins"},
            }

        lines = [f"**Engineering Margins** ({len(margins)} tracked parameters):\n"]
        critical = []
        for m in margins:
            comp = m.get("component", m.get("field", "unknown"))
            param = m.get("parameter", "")
            pct = m.get("margin_percentage", "N/A")
            trend = m.get("trend", "stable")
            breach = m.get("projected_breach_date")
            safety = m.get("safety_critical", False)
            label = f"{comp}" + (f" / {param}" if param else "")

            marker = ""
            if isinstance(pct, (int, float)) and pct < 10:
                marker = " **[LOW]**"
            if safety:
                marker += " **[SAFETY CRITICAL]**"
                critical.append(label)

            lines.append(
                f"- **{label}**: margin = **{pct}%**, trend = {trend}"
                + (f", projected breach: {breach}" if breach else "")
                + marker
            )

        if critical:
            lines.append(f"\n**Warning**: {len(critical)} safety-critical parameter(s): {', '.join(critical)}")

        lines.append(f"\nSystem health score: **{system.get('health_score', 'N/A')}**")
        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "margins", "count": len(margins)},
        }

    # ── Fallback handler: blind spots / data gaps ──────────────

    def _fb_blind_spots(self, analysis: Optional[Dict] = None) -> Dict:
        blind_spots = (analysis or {}).get("blind_spots", [])
        if not blind_spots:
            return {
                "content": (
                    "No blind spots identified yet. Run a full analysis to detect "
                    "data gaps and missing sensor coverage."
                ),
                "ai_powered": False,
                "data": {"type": "blind_spots"},
            }

        lines = [f"**Blind Spots & Data Gaps** ({len(blind_spots)} identified):\n"]
        for i, bs in enumerate(blind_spots, 1):
            title = bs.get("title", bs.get("description", "Unknown"))
            sensor = bs.get("recommended_sensor", "")
            improvement = bs.get("diagnostic_coverage_improvement", "")
            lines.append(f"**{i}. {title}**")
            if sensor:
                lines.append(f"   Recommended sensor: {sensor}")
            if improvement:
                lines.append(f"   Diagnostic coverage improvement: {improvement}%")
            lines.append("")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "blind_spots", "count": len(blind_spots)},
        }

    # ── Fallback handler: health summary ─────────────────────────

    def _fb_health_summary(self, df: pd.DataFrame, numeric_cols: list, system: Dict) -> Dict:
        lines = [f"**System Health Summary — {system.get('name', 'Unknown')}**\n"]
        lines.append(f"- **Status**: {system.get('status', 'active')}")
        lines.append(f"- **Health Score**: {system.get('health_score', 'N/A')}")
        lines.append(f"- **Records**: {len(df):,}")
        lines.append(f"- **Fields**: {len(df.columns)} ({len(numeric_cols)} numeric)\n")

        # Quick stats for each numeric field
        if numeric_cols:
            lines.append("**Numeric field summary**:\n")
            lines.append("| Field | Mean | Std | Min | Max |")
            lines.append("|-------|------|-----|-----|-----|")
            for col in numeric_cols[:12]:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                lines.append(
                    f"| {col} | {float(s.mean()):.4g} | {float(s.std()):.4g} "
                    f"| {float(s.min()):.4g} | {float(s.max()):.4g} |"
                )

        # Warnings
        warnings = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 2:
                continue
            cv = float(s.std()) / abs(float(s.mean())) if float(s.mean()) != 0 else 0
            if cv > 0.5:
                warnings.append(f"**{col}** has high variability (CV={cv:.2f})")
            null_pct = df[col].isna().sum() / len(df) * 100
            if null_pct > 10:
                warnings.append(f"**{col}** has {null_pct:.0f}% missing values")

        if warnings:
            lines.append("\n**Warnings**:")
            for w in warnings[:8]:
                lines.append(f"- {w}")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "health_summary"},
        }

    # ── Fallback handler: correlations ───────────────────────────

    def _fb_correlations(self, df: pd.DataFrame, numeric_cols: list) -> Dict:
        if len(numeric_cols) < 2:
            return {
                "content": "Need at least 2 numeric fields to compute correlations.",
                "ai_powered": False,
                "data": {"type": "correlation"},
            }

        corr = df[numeric_cols].corr()
        lines = ["**Field Correlations** (|r| > 0.5):\n"]
        pairs = []
        seen = set()
        for i, c1 in enumerate(numeric_cols):
            for j, c2 in enumerate(numeric_cols):
                if i >= j:
                    continue
                r = float(corr.loc[c1, c2])
                if math.isnan(r):
                    continue
                key = tuple(sorted([c1, c2]))
                if key in seen:
                    continue
                seen.add(key)
                if abs(r) > 0.5:
                    direction = "positive" if r > 0 else "negative"
                    strength = "strong" if abs(r) > 0.8 else "moderate"
                    lines.append(f"- **{c1}** ↔ **{c2}**: r = {r:.3f} ({strength} {direction})")
                    pairs.append({"field1": c1, "field2": c2, "r": round(r, 3), "strength": strength})

        if not pairs:
            lines.append("No strong correlations found (all |r| < 0.5).")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "correlation", "pairs": sanitize_for_json(pairs)},
        }

    # ── Fallback handler: recommendations ────────────────────────

    def _fb_recommendations(self, df: pd.DataFrame, numeric_cols: list, system: Dict) -> Dict:
        recs = []

        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            mean, std = float(s.mean()), float(s.std())
            if std == 0:
                continue

            cv = std / abs(mean) if mean != 0 else 0
            outlier_count = int(((s - mean).abs() / std > 2.5).sum())

            if outlier_count > len(s) * 0.05:
                recs.append(
                    f"**Investigate {col}**: {outlier_count} outliers detected "
                    f"({outlier_count/len(s)*100:.1f}% of records). Check sensor calibration or process limits."
                )
            if cv > 1.0:
                recs.append(
                    f"**Stabilize {col}**: Very high variability (CV={cv:.2f}). "
                    f"Consider tighter process controls or identifying distinct operating modes."
                )
            null_pct = df[col].isna().sum() / len(df) * 100
            if null_pct > 20:
                recs.append(
                    f"**Fix data gaps in {col}**: {null_pct:.0f}% missing values. "
                    f"Check data pipeline and sensor connectivity."
                )

        if not recs:
            recs.append("No urgent recommendations — the data looks healthy overall.")

        health = system.get("health_score")
        if health and health < 70:
            recs.insert(0, f"**Priority**: Health score is **{health}** — run a full analysis to identify root causes.")

        lines = ["**Recommendations for this system**:\n"]
        for i, r in enumerate(recs[:10], 1):
            lines.append(f"{i}. {r}")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "recommendations"},
        }

    # ── Fallback handler: variance / stability ───────────────────

    def _fb_variance(self, df: pd.DataFrame, numeric_cols: list) -> Dict:
        lines = ["**Field Variability Analysis**:\n"]
        items = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 2 or float(s.mean()) == 0:
                continue
            cv = float(s.std()) / abs(float(s.mean()))
            stability = "stable" if cv < 0.1 else "moderate" if cv < 0.3 else "variable" if cv < 0.5 else "highly variable"
            items.append((col, cv, stability))

        items.sort(key=lambda x: x[1], reverse=True)
        for col, cv, stability in items[:15]:
            marker = "" if cv < 0.5 else " ⚠"
            lines.append(f"- **{col}**: CV = {cv:.3f} ({stability}){marker}")

        if not items:
            lines.append("No numeric fields with computable variance.")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "variance", "items": sanitize_for_json([{"field": c, "cv": round(v, 3)} for c, v, _ in items])},
        }

    # ── Fallback handler: trends ─────────────────────────────────

    def _fb_trends(self, df: pd.DataFrame, numeric_cols: list) -> Dict:
        lines = ["**Trend Analysis** (comparing first half vs second half of data):\n"]
        trends = []
        n = len(df)
        if n < 6:
            return {
                "content": "Not enough records to compute trends (need at least 6).",
                "ai_powered": False,
                "data": {"type": "trends"},
            }

        half = n // 2
        for col in numeric_cols:
            first_mean = float(df[col].iloc[:half].mean())
            second_mean = float(df[col].iloc[half:].mean())
            if first_mean == 0:
                continue
            pct_change = (second_mean - first_mean) / abs(first_mean) * 100
            if abs(pct_change) < 1:
                direction = "stable"
            elif pct_change > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
            trends.append((col, pct_change, direction))

        trends.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, pct, direction in trends[:12]:
            arrow = "↗" if pct > 1 else "↘" if pct < -1 else "→"
            marker = " ⚠" if abs(pct) > 20 else ""
            lines.append(f"- {arrow} **{col}**: {direction} ({pct:+.1f}%){marker}")

        if not trends:
            lines.append("No numeric fields to analyze trends.")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "trends", "trends": sanitize_for_json([{"field": c, "change_pct": round(p, 2), "direction": d} for c, p, d in trends])},
        }

    # ── Fallback handler: distribution ───────────────────────────

    def _fb_distribution(self, df: pd.DataFrame, numeric_cols: list, target_field: Optional[str]) -> Dict:
        cols = [target_field] if target_field and target_field in numeric_cols else numeric_cols[:6]
        lines = ["**Distribution Summary**:\n"]

        for col in cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            q25, q50, q75 = float(s.quantile(0.25)), float(s.quantile(0.5)), float(s.quantile(0.75))
            iqr = q75 - q25
            skew = float(s.skew()) if len(s) > 3 else 0
            skew_desc = "symmetric" if abs(skew) < 0.5 else "right-skewed" if skew > 0 else "left-skewed"
            lines.append(f"**{col}**:")
            lines.append(f"  - Range: {float(s.min()):.4g} — {float(s.max()):.4g}")
            lines.append(f"  - Quartiles: Q1={q25:.4g}, Median={q50:.4g}, Q3={q75:.4g}")
            lines.append(f"  - IQR: {iqr:.4g}, Skewness: {skew:.2f} ({skew_desc})")
            lines.append("")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "distribution"},
        }

    # ── Fallback handler: single statistic ───────────────────────

    def _fb_stat(self, df: pd.DataFrame, numeric_cols: list, stat: str, target_field: Optional[str]) -> Dict:
        cols = [target_field] if target_field and target_field in numeric_cols else numeric_cols
        label = {"mean": "average", "max": "maximum", "min": "minimum",
                 "std": "standard deviation", "median": "median"}.get(stat, stat)

        stats = {}
        for col in cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            if stat == "mean":
                stats[col] = round(float(s.mean()), 4)
            elif stat == "max":
                stats[col] = round(float(s.max()), 4)
            elif stat == "min":
                stats[col] = round(float(s.min()), 4)
            elif stat == "std":
                stats[col] = round(float(s.std()), 4)
            elif stat == "median":
                stats[col] = round(float(s.median()), 4)

        lines = [f"Here are the **{label} values** for {'**' + target_field + '**' if target_field else 'numeric fields'}:\n"]
        for col, val in stats.items():
            lines.append(f"- **{col}**: {val}")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "statistics", "statistics": sanitize_for_json(stats)},
        }

    # ── Fallback handler: specific field detail ──────────────────

    def _fb_field_detail(self, df: pd.DataFrame, field: str, numeric_cols: list) -> Dict:
        lines = [f"**Detailed analysis of `{field}`**:\n"]
        s = df[field].dropna()
        lines.append(f"- **Records**: {len(s):,} ({len(df) - len(s)} missing)")

        if field in numeric_cols and len(s) > 0:
            lines.append(f"- **Mean**: {float(s.mean()):.4g}")
            lines.append(f"- **Std Dev**: {float(s.std()):.4g}")
            lines.append(f"- **Min**: {float(s.min()):.4g}")
            lines.append(f"- **Max**: {float(s.max()):.4g}")
            lines.append(f"- **Median**: {float(s.median()):.4g}")
            if len(s) > 3:
                q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
                lines.append(f"- **Q1 / Q3**: {q25:.4g} / {q75:.4g}")
                skew = float(s.skew())
                lines.append(f"- **Skewness**: {skew:.2f}")
                z = ((s - s.mean()) / s.std()).abs()
                outliers = int((z > 2.5).sum())
                if outliers:
                    lines.append(f"- **Outliers (|z|>2.5)**: {outliers}")
        else:
            nunique = int(df[field].nunique())
            lines.append(f"- **Unique values**: {nunique}")
            top = df[field].value_counts().head(5)
            if len(top) > 0:
                lines.append("- **Top values**:")
                for val, count in top.items():
                    lines.append(f"  - `{val}`: {count} records")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {"type": "field_detail", "field": field},
        }

    # ── Fallback handler: data overview ──────────────────────────

    def _fb_data_overview(self, df: pd.DataFrame, numeric_cols: list, system: Dict) -> Dict:
        cat_cols = [c for c in df.columns if c not in numeric_cols]
        lines = [
            f"**System: {system.get('name', 'Unknown')}** | "
            f"Health: **{system.get('health_score', 'N/A')}**\n",
            f"**{len(df):,} records** across **{len(df.columns)} fields**:\n",
        ]

        if numeric_cols:
            lines.append(f"**Numeric** ({len(numeric_cols)}): {', '.join(numeric_cols[:15])}")
        if cat_cols:
            lines.append(f"**Categorical** ({len(cat_cols)}): {', '.join(cat_cols[:10])}")

        lines.append("\nYou can ask me:")
        lines.append("- \"What anomalies do you see?\"")
        lines.append("- \"Summarize the system health\"")
        lines.append("- \"Show me correlations\"")
        lines.append("- \"What are the trends?\"")
        lines.append("- \"Analyze field <name>\"")
        lines.append("- \"What recommendations do you have?\"")
        lines.append("- \"Show distribution of <field>\"")
        lines.append("- \"What's the average / max / min?\"")

        return {
            "content": "\n".join(lines),
            "ai_powered": False,
            "data": {
                "type": "data_overview",
                "record_count": len(df),
                "fields": list(df.columns),
            },
        }

    # ── Helper: detect field name in user query ──────────────────

    def _detect_field_mention(self, query: str, columns: list) -> Optional[str]:
        """Check if the user mentioned a specific field name."""
        q = query.lower()
        # Exact match first
        for col in sorted(columns, key=len, reverse=True):
            if col.lower() in q:
                return col
        # Underscore / space variants
        for col in columns:
            normalized = col.lower().replace("_", " ")
            if normalized in q:
                return col
        return None


chat_service = ChatService()
