"""
Multi-Agent AI Analysis System for UAIE — specialized blind-spot agents module.

Contains the 12 specialized blind-spot agent classes (agents 14–25).
"""

from typing import Dict, List

from .ai_agents_base import BaseAgent, AgentFinding


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
