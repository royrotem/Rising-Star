"""
PDF Report Generator for UAIE

Generates comprehensive system analysis reports in PDF format using fpdf2.
This is an additive feature module — removing it does not affect core
analysis or other endpoints.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from fpdf import FPDF


def _safe(val: Any, max_len: int = 12) -> str:
    """Convert value to a safe string for PDF rendering, truncated to max_len."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return "N/A"
        s = f"{val:.4g}"
    else:
        s = str(val)
    return s[:max_len] if len(s) > max_len else s


class UAIEReport(FPDF):
    """Custom PDF with UAIE branding."""

    def __init__(self, system_name: str, system_type: str):
        super().__init__()
        self.system_name = system_name
        self.system_type = system_type
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "UAIE - Universal Autonomous Insight Engine", align="L")
        self.cell(0, 8, f"{self.system_name}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(99, 102, 241)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 60)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(99, 102, 241)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 80)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def _safe_multi_cell(self, w, h, text):
        """multi_cell with x reset to left margin to prevent overflow errors."""
        self.set_x(self.l_margin)
        self.multi_cell(w, h, text)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self._safe_multi_cell(0, 5.5, text)
        self.ln(2)

    def key_value(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 80)
        self.cell(50, 6, f"{key}:", align="L")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 6, str(value)[:120], new_x="LMARGIN", new_y="NEXT")

    def severity_badge(self, severity: str) -> str:
        return severity.upper()

    def _health_color(self, score: float):
        if score >= 90:
            return (16, 185, 129)  # green
        if score >= 70:
            return (234, 179, 8)   # yellow
        return (239, 68, 68)       # red

    def _severity_color(self, severity: str):
        mapping = {
            "critical": (220, 38, 38),
            "high": (234, 88, 12),
            "medium": (234, 179, 8),
            "low": (34, 197, 94),
            "info": (100, 116, 139),
        }
        return mapping.get(severity, (100, 116, 139))


def generate_report(
    system: Dict[str, Any],
    analysis: Optional[Dict[str, Any]],
    statistics: Optional[Dict[str, Any]],
) -> bytes:
    """
    Generate a PDF report for a system.

    Parameters:
        system: System metadata dict
        analysis: Analysis result from the analyze endpoint (can be None)
        statistics: Data statistics dict (can be None)

    Returns:
        PDF file content as bytes
    """
    name = system.get("name", "Unknown System")
    stype = system.get("system_type", "unknown")

    pdf = UAIEReport(name, stype)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Cover / Title ─────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 14, "System Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(99, 102, 241)
    pdf.cell(0, 10, name, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Type: {stype.replace('_', ' ').title()} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # ── System Overview ───────────────────────────────────────────
    pdf.section_title("System Overview")
    pdf.key_value("Name", name)
    pdf.key_value("Type", stype.replace("_", " ").title())
    pdf.key_value("Status", system.get("status", "active").title())
    pdf.key_value("Health Score", _safe(system.get("health_score")))
    pdf.key_value("System ID", system.get("id", "N/A"))

    meta = system.get("metadata", {})
    if isinstance(meta, dict) and meta.get("description"):
        pdf.key_value("Description", meta["description"][:200])

    # ── Health Score Visual ───────────────────────────────────────
    health = system.get("health_score")
    if health is not None:
        pdf.ln(4)
        pdf.sub_title("Health Score")
        color = pdf._health_color(health)

        # Draw bar
        bar_x, bar_y = 10, pdf.get_y()
        bar_w, bar_h = 120, 10
        pdf.set_fill_color(220, 220, 230)
        pdf.rect(bar_x, bar_y, bar_w, bar_h, style="F")
        pdf.set_fill_color(*color)
        pdf.rect(bar_x, bar_y, bar_w * (health / 100), bar_h, style="F")

        pdf.set_xy(bar_x + bar_w + 4, bar_y)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*color)
        pdf.cell(30, 10, f"{health:.0f}%")
        pdf.set_x(pdf.l_margin)
        pdf.ln(14)

    # ── Data Statistics ───────────────────────────────────────────
    if statistics:
        pdf.section_title("Data Statistics")
        pdf.key_value("Total Records", f"{statistics.get('total_records', 0):,}")
        pdf.key_value("Data Sources", str(statistics.get("total_sources", 0)))
        pdf.key_value("Fields", str(statistics.get("field_count", 0)))

        fields = statistics.get("fields", [])
        if fields:
            pdf.ln(3)
            pdf.sub_title("Field Summary")

            # Table header — total width must stay under 190mm (A4 minus margins)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(240, 240, 250)
            pdf.set_text_color(50, 50, 50)
            col_widths = [38, 18, 20, 26, 26, 26, 26]  # total = 180
            headers = ["Field", "Type", "Unique", "Min", "Max", "Mean", "Std Dev"]
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
            pdf.ln()

            # Table rows
            pdf.set_font("Helvetica", "", 7)
            for f in fields[:20]:
                pdf.set_x(pdf.l_margin)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(col_widths[0], 6, str(f.get("name", ""))[:20], border=1)
                pdf.cell(col_widths[1], 6, str(f.get("type", ""))[:10], border=1, align="C")
                pdf.cell(col_widths[2], 6, str(f.get("unique_count", "")), border=1, align="C")
                pdf.cell(col_widths[3], 6, _safe(f.get("min")), border=1, align="R")
                pdf.cell(col_widths[4], 6, _safe(f.get("max")), border=1, align="R")
                pdf.cell(col_widths[5], 6, _safe(f.get("mean")), border=1, align="R")
                pdf.cell(col_widths[6], 6, _safe(f.get("std")), border=1, align="R")
                pdf.ln()

    if not analysis:
        pdf.ln(10)
        pdf.body_text("No analysis has been run yet. Run analysis to populate this report with anomaly detection, engineering margins, and AI insights.")
        return pdf.output()

    # ── Analysis Summary ──────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Analysis Summary")

    data_info = analysis.get("data_analyzed", {})
    if data_info:
        pdf.key_value("Records Analyzed", f"{data_info.get('record_count', 0):,}")
        pdf.key_value("Sources", str(data_info.get("source_count", 0)))
        pdf.key_value("Fields", str(data_info.get("field_count", 0)))

    ai_info = analysis.get("ai_analysis")
    if ai_info:
        pdf.key_value("AI Powered", "Yes" if ai_info.get("ai_powered") else "No (rule-based)")
        agents = ai_info.get("agents_used", [])
        if agents:
            pdf.key_value("AI Agents", ", ".join(agents))
        pdf.key_value("Raw Findings", str(ai_info.get("total_findings_raw", 0)))
        pdf.key_value("Unified Anomalies", str(ai_info.get("total_anomalies_unified", 0)))

    summary = analysis.get("insights_summary")
    if summary:
        pdf.ln(3)
        pdf.sub_title("Executive Summary")
        pdf.body_text(summary)

    # Key Insights
    insights = analysis.get("insights", [])
    if insights:
        pdf.sub_title("Key Insights")
        for idx, insight in enumerate(insights[:10], 1):
            pdf.body_text(f"{idx}. {insight}")

    # ── Anomalies ─────────────────────────────────────────────────
    anomalies: List[Dict] = analysis.get("anomalies", [])
    if anomalies:
        pdf.add_page()
        pdf.section_title(f"Detected Anomalies ({len(anomalies)})")

        for i, anom in enumerate(anomalies[:15], 1):
            severity = anom.get("severity", "info")
            color = pdf._severity_color(severity)

            # Anomaly header
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*color)
            title = f"{i}. [{severity.upper()}] {anom.get('title', 'Anomaly')}"
            pdf.cell(0, 8, title[:90], new_x="LMARGIN", new_y="NEXT")

            # Meta line
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(120, 120, 120)
            meta_parts = []
            if anom.get("type"):
                meta_parts.append(f"Type: {anom['type']}")
            if anom.get("impact_score") is not None:
                meta_parts.append(f"Impact: {_safe(anom['impact_score'])}")
            if anom.get("confidence") is not None:
                meta_parts.append(f"Confidence: {float(anom['confidence'])*100:.0f}%")
            affected = anom.get("affected_fields", [])
            if affected:
                meta_parts.append(f"Fields: {', '.join(affected[:5])}")
            if meta_parts:
                pdf.cell(0, 5, " | ".join(meta_parts)[:120], new_x="LMARGIN", new_y="NEXT")

            # Description
            desc = anom.get("description", "")
            if desc:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(50, 50, 50)
                pdf._safe_multi_cell(0, 5, desc[:500])

            # AI Explanation
            explanation = anom.get("natural_language_explanation", "")
            if explanation and explanation != desc:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(80, 80, 120)
                pdf._safe_multi_cell(0, 5, f"AI Analysis: {explanation[:400]}")

            # Possible Causes
            causes = anom.get("possible_causes", [])
            if causes:
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(70, 70, 70)
                pdf.cell(0, 5, "Possible causes:", new_x="LMARGIN", new_y="NEXT")
                for cause in causes[:4]:
                    pdf.cell(5)
                    pdf.cell(0, 5, f"- {cause[:100]}", new_x="LMARGIN", new_y="NEXT")

            # Recommendations
            recs = anom.get("recommendations", [])
            if recs:
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(16, 120, 80)
                pdf.cell(0, 5, "Recommendations:", new_x="LMARGIN", new_y="NEXT")
                for rec in recs[:3]:
                    action = rec.get("action", str(rec)) if isinstance(rec, dict) else str(rec)
                    priority = rec.get("priority", "") if isinstance(rec, dict) else ""
                    prefix = f"[{priority}] " if priority else ""
                    pdf.cell(5)
                    pdf.cell(0, 5, f"- {prefix}{action[:120]}", new_x="LMARGIN", new_y="NEXT")

            pdf.ln(4)

            # Page break check
            if pdf.get_y() > 250:
                pdf.add_page()

    # ── Engineering Margins ───────────────────────────────────────
    margins: List[Dict] = analysis.get("engineering_margins", [])
    if margins:
        pdf.add_page()
        pdf.section_title(f"Engineering Margins ({len(margins)})")

        # Table header — total width must stay under 190mm
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(240, 240, 250)
        pdf.set_text_color(50, 50, 50)
        mcols = [34, 28, 24, 24, 24, 22, 24]  # total = 180
        mheaders = ["Component", "Parameter", "Current", "Limit", "Margin %", "Trend", "Safety"]
        for j, h in enumerate(mheaders):
            pdf.cell(mcols[j], 7, h, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 7)
        for m in margins[:20]:
            pdf.set_x(pdf.l_margin)
            margin_pct = m.get("margin_percentage", 0)
            if margin_pct < 15:
                pdf.set_text_color(220, 38, 38)
            elif margin_pct < 30:
                pdf.set_text_color(180, 130, 0)
            else:
                pdf.set_text_color(50, 50, 50)

            pdf.cell(mcols[0], 6, str(m.get("component", ""))[:18], border=1)
            pdf.cell(mcols[1], 6, str(m.get("parameter", ""))[:14], border=1)
            pdf.cell(mcols[2], 6, _safe(m.get("current_value")), border=1, align="R")
            pdf.cell(mcols[3], 6, _safe(m.get("design_limit")), border=1, align="R")
            pdf.cell(mcols[4], 6, f"{margin_pct:.1f}%", border=1, align="R")
            pdf.cell(mcols[5], 6, str(m.get("trend", ""))[:10], border=1, align="C")
            pdf.cell(mcols[6], 6, "YES" if m.get("safety_critical") else "no", border=1, align="C")
            pdf.ln()

    # ── Blind Spots ───────────────────────────────────────────────
    blind_spots: List[Dict] = analysis.get("blind_spots", [])
    if blind_spots:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.section_title(f"Blind Spots & Coverage Gaps ({len(blind_spots)})")

        for idx, spot in enumerate(blind_spots[:10], 1):
            pdf.sub_title(f"{idx}. {spot.get('title', 'Blind Spot')}")
            desc = spot.get("description", "")
            if desc:
                pdf.body_text(desc[:400])

            sensor = spot.get("recommended_sensor")
            if sensor and isinstance(sensor, dict):
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(70, 70, 130)
                rec_text = f"Recommended: {sensor.get('type', '')} -- {sensor.get('specification', '')} (est. ${sensor.get('estimated_cost', 'N/A')})"
                pdf._safe_multi_cell(0, 5, rec_text[:200])

            improvement = spot.get("diagnostic_coverage_improvement", 0)
            if improvement:
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(16, 185, 129)
                pdf.cell(0, 5, f"Coverage improvement: +{improvement}%", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # ── Recommendations Summary ───────────────────────────────────
    recs_list: List[Dict] = analysis.get("recommendations", [])
    if recs_list:
        if pdf.get_y() > 220:
            pdf.add_page()
        pdf.section_title("Recommendations")
        for idx, rec in enumerate(recs_list[:12], 1):
            action = rec.get("action", str(rec)) if isinstance(rec, dict) else str(rec)
            priority = rec.get("priority", "") if isinstance(rec, dict) else ""
            pdf.set_font("Helvetica", "B" if priority in ("high", "immediate") else "", 9)
            pdf.set_text_color(50, 50, 50)
            prefix = f"[{priority.upper()}] " if priority else ""
            pdf.body_text(f"{idx}. {prefix}{action[:250]}")

    return pdf.output()
