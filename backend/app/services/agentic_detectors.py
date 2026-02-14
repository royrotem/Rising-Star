"""
Agentic Anomaly Detectors — true tool-using AI agents.

Unlike the prompt-based agents in ai_agents.py (which receive a data summary
and return findings in one shot), these agents use Claude's tool_use API to
*interact* with the data over multiple turns.  Each agent has a toolkit of
data-exploration functions it can call autonomously:

  - query_column          → get a column's values (or a slice)
  - compute_statistics    → descriptive stats for one or more columns
  - run_statistical_test  → normality, stationarity, Grubbs, etc.
  - correlate_fields      → Pearson/Spearman between two columns
  - slice_window          → get rows in an index range
  - detect_outliers_in    → run IQR / z-score / isolation forest on a column
  - compute_rolling       → rolling mean, std, or diff for a column
  - compare_segments      → split data and compare distributions

The agent decides what to investigate, calls tools, sees results, and
iterates until it has found anomalies or exhausted its budget.

Agents:
  1. Autonomous Explorer      — open-ended data exploration
  2. Statistical Investigator — deep statistical testing
  3. Correlation Hunter       — finds unexpected relationships
  4. Drift & Shift Detector   — looks for regime changes
  5. Physics Constraint Agent — validates physical consistency
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .ai_agents import AgentFinding, BaseAgent, _get_api_key, AGENT_TIMEOUT

logger = logging.getLogger("uaie.agentic_detectors")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Maximum tool-use turns per agent (prevents runaway loops)
MAX_TOOL_TURNS = 12

# Supported LLM providers for agentic tool-use
LLM_PROVIDER_ANTHROPIC = "anthropic"
LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_GEMINI = "gemini"


# ═══════════════════════════════════════════════════════════════════
# Data Toolkit — functions the agent can call via tool_use
# ═══════════════════════════════════════════════════════════════════

class DataToolkit:
    """Provides sandboxed data-exploration tools for agentic detectors.

    Instantiated once per analysis run with the full DataFrame.
    Each tool returns a JSON-serialisable dict.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.numeric_cols = list(df.select_dtypes(include=["number"]).columns)

    # ── Tool definitions (Claude tool_use schema) ─────────────────

    @staticmethod
    def tool_definitions() -> List[Dict[str, Any]]:
        """Return the Claude tool_use schema for all available tools."""
        return [
            {
                "name": "query_column",
                "description": (
                    "Get the values of a single column. Returns basic stats plus "
                    "the actual values (up to 200). Use this to inspect a column "
                    "before deciding what tests to run."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "Column name"},
                        "head": {"type": "integer", "description": "Return only the first N values (default: all, max 200)"},
                    },
                    "required": ["column"],
                },
            },
            {
                "name": "compute_statistics",
                "description": (
                    "Compute detailed descriptive statistics for one or more columns: "
                    "mean, std, median, min, max, skewness, kurtosis, percentiles, "
                    "number of zeros, number of NaN, coefficient of variation."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of column names (max 10)",
                        },
                    },
                    "required": ["columns"],
                },
            },
            {
                "name": "run_statistical_test",
                "description": (
                    "Run a named statistical test on a column. Available tests: "
                    "'shapiro' (normality), 'anderson' (normality), "
                    "'adfuller' (stationarity), 'levene' (variance equality, "
                    "compares first and second half), 'mannwhitney' (distribution "
                    "shift, compares first and second half), 'grubbs' (outlier test)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "Column name"},
                        "test": {
                            "type": "string",
                            "enum": ["shapiro", "anderson", "adfuller", "levene", "mannwhitney", "grubbs"],
                            "description": "Which statistical test to run",
                        },
                    },
                    "required": ["column", "test"],
                },
            },
            {
                "name": "correlate_fields",
                "description": (
                    "Compute Pearson and Spearman correlation between two columns. "
                    "Also computes rolling correlation (window=50) to detect "
                    "correlation breakdown over time."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "column_a": {"type": "string"},
                        "column_b": {"type": "string"},
                    },
                    "required": ["column_a", "column_b"],
                },
            },
            {
                "name": "slice_window",
                "description": (
                    "Get a slice of the data by row index range. Returns all "
                    "numeric columns for the specified rows. Use this to zoom "
                    "in on suspicious regions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer", "description": "Start row index (inclusive)"},
                        "end": {"type": "integer", "description": "End row index (exclusive)"},
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to include (default: all numeric)",
                        },
                    },
                    "required": ["start", "end"],
                },
            },
            {
                "name": "detect_outliers_in",
                "description": (
                    "Run outlier detection on a single column using a chosen method. "
                    "Returns the row indices and values of detected outliers. "
                    "Methods: 'zscore' (threshold=3), 'iqr' (k=1.5), "
                    "'isolation_forest', 'grubbs'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "method": {
                            "type": "string",
                            "enum": ["zscore", "iqr", "isolation_forest", "grubbs"],
                            "description": "Outlier detection method",
                        },
                    },
                    "required": ["column", "method"],
                },
            },
            {
                "name": "compute_rolling",
                "description": (
                    "Compute a rolling statistic for a column. Returns the rolling "
                    "values plus indices where the rolling stat exceeds a threshold. "
                    "Operations: 'mean', 'std', 'diff' (first difference), "
                    "'rate_of_change' (percentage change)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "operation": {
                            "type": "string",
                            "enum": ["mean", "std", "diff", "rate_of_change"],
                        },
                        "window": {"type": "integer", "description": "Window size (default: 20)"},
                    },
                    "required": ["column", "operation"],
                },
            },
            {
                "name": "compare_segments",
                "description": (
                    "Split the data into two segments (first half vs second half, "
                    "or by a custom split point) and compare distributions. Returns "
                    "mean shift, variance ratio, KS test, and Mann-Whitney test."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "split_at": {
                            "type": "integer",
                            "description": "Row index to split at (default: midpoint)",
                        },
                    },
                    "required": ["column"],
                },
            },
            {
                "name": "list_columns",
                "description": "List all available columns with their types and basic info.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    # ── Tool implementations ──────────────────────────────────────

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call to the appropriate method."""
        handlers = {
            "query_column": self._query_column,
            "compute_statistics": self._compute_statistics,
            "run_statistical_test": self._run_statistical_test,
            "correlate_fields": self._correlate_fields,
            "slice_window": self._slice_window,
            "detect_outliers_in": self._detect_outliers_in,
            "compute_rolling": self._compute_rolling,
            "compare_segments": self._compare_segments,
            "list_columns": self._list_columns,
        }
        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}
        try:
            return handler(**args)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _list_columns(self, **_kw: Any) -> Dict[str, Any]:
        cols = []
        for col in self.df.columns:
            info: Dict[str, Any] = {"name": col, "dtype": str(self.df[col].dtype)}
            if col in self.numeric_cols:
                info["numeric"] = True
                info["min"] = _safe_float(self.df[col].min())
                info["max"] = _safe_float(self.df[col].max())
            else:
                info["numeric"] = False
                info["unique"] = int(self.df[col].nunique())
            cols.append(info)
        return {"total_rows": len(self.df), "columns": cols}

    def _query_column(self, column: str, head: int = 200, **_kw: Any) -> Dict[str, Any]:
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found. Available: {list(self.df.columns)}"}
        series = self.df[column].dropna()
        head = min(head, 200)
        values = series.head(head).tolist()
        result: Dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "count": len(series),
            "null_count": int(self.df[column].isna().sum()),
            "values": [_safe_float(v) for v in values],
        }
        if column in self.numeric_cols:
            result.update({
                "mean": _safe_float(series.mean()),
                "std": _safe_float(series.std()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
            })
        return result

    def _compute_statistics(self, columns: List[str], **_kw: Any) -> Dict[str, Any]:
        columns = columns[:10]
        results = {}
        for col in columns:
            if col not in self.df.columns:
                results[col] = {"error": f"Column '{col}' not found"}
                continue
            series = self.df[col].dropna()
            if col not in self.numeric_cols or len(series) < 2:
                results[col] = {"count": len(series), "type": str(series.dtype)}
                continue
            results[col] = {
                "count": len(series),
                "mean": _safe_float(series.mean()),
                "std": _safe_float(series.std()),
                "median": _safe_float(series.median()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "skewness": _safe_float(series.skew()),
                "kurtosis": _safe_float(series.kurtosis()),
                "q25": _safe_float(series.quantile(0.25)),
                "q75": _safe_float(series.quantile(0.75)),
                "cv": _safe_float(series.std() / abs(series.mean())) if series.mean() != 0 else None,
                "zeros": int((series == 0).sum()),
                "nan_count": int(self.df[col].isna().sum()),
            }
        return {"statistics": results}

    def _run_statistical_test(self, column: str, test: str, **_kw: Any) -> Dict[str, Any]:
        if column not in self.numeric_cols:
            return {"error": f"Column '{column}' is not numeric or not found"}
        series = self.df[column].dropna()
        if len(series) < 10:
            return {"error": "Need at least 10 data points"}

        if test == "shapiro":
            sample = series.sample(min(len(series), 5000), random_state=42)
            stat, p_value = scipy_stats.shapiro(sample)
            return {"test": "Shapiro-Wilk (normality)", "statistic": _safe_float(stat),
                    "p_value": _safe_float(p_value),
                    "interpretation": "Normal" if p_value > 0.05 else "Non-normal (reject H0)"}

        elif test == "anderson":
            result = scipy_stats.anderson(series.values)
            return {"test": "Anderson-Darling (normality)", "statistic": _safe_float(result.statistic),
                    "critical_values": {f"{s}%": _safe_float(c) for s, c in zip(result.significance_level, result.critical_values)},
                    "interpretation": "Normal" if result.statistic < result.critical_values[2] else "Non-normal"}

        elif test == "adfuller":
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.values[:5000], maxlag=min(20, len(series) // 5))
            return {"test": "Augmented Dickey-Fuller (stationarity)",
                    "statistic": _safe_float(result[0]), "p_value": _safe_float(result[1]),
                    "critical_values": {k: _safe_float(v) for k, v in result[4].items()},
                    "interpretation": "Stationary" if result[1] < 0.05 else "Non-stationary (has trend/drift)"}

        elif test == "levene":
            mid = len(series) // 2
            stat, p_value = scipy_stats.levene(series.iloc[:mid], series.iloc[mid:])
            return {"test": "Levene (variance equality, 1st half vs 2nd half)",
                    "statistic": _safe_float(stat), "p_value": _safe_float(p_value),
                    "interpretation": "Equal variance" if p_value > 0.05 else "Variance changed between halves"}

        elif test == "mannwhitney":
            mid = len(series) // 2
            stat, p_value = scipy_stats.mannwhitneyu(series.iloc[:mid], series.iloc[mid:], alternative="two-sided")
            return {"test": "Mann-Whitney U (distribution shift, 1st vs 2nd half)",
                    "statistic": _safe_float(stat), "p_value": _safe_float(p_value),
                    "interpretation": "Same distribution" if p_value > 0.05 else "Distribution shifted between halves"}

        elif test == "grubbs":
            n = len(series)
            mean, std = series.mean(), series.std()
            if std == 0:
                return {"test": "Grubbs", "result": "No variance — cannot test"}
            max_dev_idx = (series - mean).abs().idxmax()
            g_stat = float(abs(series[max_dev_idx] - mean) / std)
            t_val = scipy_stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
            g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_val ** 2 / (n - 2 + t_val ** 2))
            return {"test": "Grubbs (most extreme value)",
                    "g_statistic": round(g_stat, 4), "g_critical": round(float(g_crit), 4),
                    "most_extreme_value": _safe_float(series[max_dev_idx]),
                    "most_extreme_index": int(max_dev_idx),
                    "is_outlier": g_stat > g_crit}

        return {"error": f"Unknown test: {test}"}

    def _correlate_fields(self, column_a: str, column_b: str, **_kw: Any) -> Dict[str, Any]:
        if column_a not in self.numeric_cols or column_b not in self.numeric_cols:
            return {"error": "Both columns must be numeric"}
        a = self.df[column_a].dropna()
        b = self.df[column_b].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 10:
            return {"error": "Not enough common non-null rows"}
        a, b = a.loc[common], b.loc[common]

        pearson_r, pearson_p = scipy_stats.pearsonr(a, b)
        spearman_r, spearman_p = scipy_stats.spearmanr(a, b)

        # Rolling correlation
        window = min(50, len(common) // 4)
        rolling_corr = a.rolling(window).corr(b).dropna() if window >= 5 else pd.Series(dtype=float)
        corr_min = _safe_float(rolling_corr.min()) if len(rolling_corr) > 0 else None
        corr_max = _safe_float(rolling_corr.max()) if len(rolling_corr) > 0 else None
        corr_breakdown = bool(rolling_corr.min() < -0.3 and pearson_r > 0.3) if len(rolling_corr) > 0 else False

        return {
            "column_a": column_a, "column_b": column_b,
            "pearson_r": round(float(pearson_r), 4), "pearson_p": _safe_float(pearson_p),
            "spearman_r": round(float(spearman_r), 4), "spearman_p": _safe_float(spearman_p),
            "rolling_corr_min": corr_min, "rolling_corr_max": corr_max,
            "correlation_breakdown_detected": corr_breakdown,
            "n_samples": len(common),
        }

    def _slice_window(self, start: int, end: int, columns: Optional[List[str]] = None, **_kw: Any) -> Dict[str, Any]:
        end = min(end, len(self.df))
        start = max(start, 0)
        if end - start > 200:
            end = start + 200
        cols = columns or self.numeric_cols
        cols = [c for c in cols if c in self.df.columns]
        subset = self.df.iloc[start:end][cols]
        return {
            "start": start, "end": end, "columns": cols,
            "rows": [{col: _safe_float(row[col]) for col in cols} for _, row in subset.iterrows()],
        }

    def _detect_outliers_in(self, column: str, method: str = "zscore", **_kw: Any) -> Dict[str, Any]:
        if column not in self.numeric_cols:
            return {"error": f"Column '{column}' is not numeric"}
        series = self.df[column].dropna()
        if len(series) < 10:
            return {"error": "Need at least 10 data points"}

        outliers: List[Dict] = []

        if method == "zscore":
            mean, std = series.mean(), series.std()
            if std == 0:
                return {"method": "zscore", "outliers": [], "message": "Zero variance"}
            z = ((series - mean) / std).abs()
            mask = z > 3
            for idx in series[mask].index:
                outliers.append({"index": int(idx), "value": _safe_float(series[idx]),
                                 "z_score": round(float(z[idx]), 3)})

        elif method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return {"method": "iqr", "outliers": [], "message": "Zero IQR"}
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
            for idx in series[mask].index:
                outliers.append({"index": int(idx), "value": _safe_float(series[idx]),
                                 "fence": f"[{_safe_float(lower)}, {_safe_float(upper)}]"})

        elif method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest as IF
                X = series.values.reshape(-1, 1)
                iso = IF(contamination=0.05, random_state=42, n_estimators=100)
                labels = iso.fit_predict(X)
                scores = -iso.score_samples(X)
                for i in np.where(labels == -1)[0]:
                    outliers.append({"index": int(series.index[i]),
                                     "value": _safe_float(series.iloc[i]),
                                     "anomaly_score": round(float(scores[i]), 4)})
            except ImportError:
                return {"error": "sklearn not available"}

        elif method == "grubbs":
            return self._run_statistical_test(column, "grubbs")

        # Limit output
        outliers = sorted(outliers, key=lambda o: abs(o.get("z_score", o.get("anomaly_score", 0))), reverse=True)[:30]
        return {"method": method, "column": column, "total_outliers": len(outliers), "outliers": outliers}

    def _compute_rolling(self, column: str, operation: str, window: int = 20, **_kw: Any) -> Dict[str, Any]:
        if column not in self.numeric_cols:
            return {"error": f"Column '{column}' is not numeric"}
        series = self.df[column].dropna()
        window = min(window, len(series) // 3)
        if window < 3:
            return {"error": "Not enough data for rolling computation"}

        if operation == "mean":
            rolled = series.rolling(window).mean().dropna()
        elif operation == "std":
            rolled = series.rolling(window).std().dropna()
        elif operation == "diff":
            rolled = series.diff().dropna()
        elif operation == "rate_of_change":
            rolled = series.pct_change().dropna()
        else:
            return {"error": f"Unknown operation: {operation}"}

        # Find extreme points
        if len(rolled) > 0 and rolled.std() > 0:
            z = ((rolled - rolled.mean()) / rolled.std()).abs()
            extremes = z[z > 3].head(20)
            extreme_points = [{"index": int(idx), "value": _safe_float(rolled[idx]),
                               "z_score": round(float(z[idx]), 3)} for idx in extremes.index]
        else:
            extreme_points = []

        return {
            "column": column, "operation": operation, "window": window,
            "mean": _safe_float(rolled.mean()),
            "std": _safe_float(rolled.std()),
            "min": _safe_float(rolled.min()),
            "max": _safe_float(rolled.max()),
            "extreme_points": extreme_points,
            "n_extreme": len(extreme_points),
        }

    def _compare_segments(self, column: str, split_at: Optional[int] = None, **_kw: Any) -> Dict[str, Any]:
        if column not in self.numeric_cols:
            return {"error": f"Column '{column}' is not numeric"}
        series = self.df[column].dropna()
        if len(series) < 20:
            return {"error": "Need at least 20 data points"}

        mid = split_at or len(series) // 2
        seg1, seg2 = series.iloc[:mid], series.iloc[mid:]

        ks_stat, ks_p = scipy_stats.ks_2samp(seg1, seg2)
        mw_stat, mw_p = scipy_stats.mannwhitneyu(seg1, seg2, alternative="two-sided")

        return {
            "column": column, "split_at": mid,
            "segment_1": {"n": len(seg1), "mean": _safe_float(seg1.mean()),
                          "std": _safe_float(seg1.std()), "median": _safe_float(seg1.median())},
            "segment_2": {"n": len(seg2), "mean": _safe_float(seg2.mean()),
                          "std": _safe_float(seg2.std()), "median": _safe_float(seg2.median())},
            "mean_shift_pct": _safe_float((seg2.mean() - seg1.mean()) / abs(seg1.mean()) * 100) if seg1.mean() != 0 else None,
            "variance_ratio": _safe_float(seg2.std() / seg1.std()) if seg1.std() > 0 else None,
            "ks_test": {"statistic": _safe_float(ks_stat), "p_value": _safe_float(ks_p),
                        "interpretation": "Same distribution" if ks_p > 0.05 else "Different distributions"},
            "mannwhitney_test": {"statistic": _safe_float(mw_stat), "p_value": _safe_float(mw_p),
                                 "interpretation": "Same location" if mw_p > 0.05 else "Location shift detected"},
        }


def _safe_float(v: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-safe Python floats."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, np.float64)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 6)
    if isinstance(v, float):
        if v != v or v == float("inf") or v == float("-inf"):  # NaN/inf
            return None
        return round(v, 6)
    return v


# ═══════════════════════════════════════════════════════════════════
# Provider helpers — convert tool schemas between formats
# ═══════════════════════════════════════════════════════════════════

def _tools_to_openai_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert our tool definitions (Anthropic format) to OpenAI function-calling format."""
    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def _tools_to_gemini_declarations(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert our tool definitions to Gemini function-declaration format.

    Gemini uses ``genai.protos.FunctionDeclaration`` but also accepts plain
    dicts when passed through ``genai.protos.Tool``.  We build the dicts
    manually so we don't need an import at module level.
    """
    declarations = []
    for t in tools:
        schema = t.get("input_schema", {"type": "object", "properties": {}})
        # Gemini doesn't accept 'additionalProperties' or empty required arrays gracefully
        props = {}
        for pname, pdef in schema.get("properties", {}).items():
            gprop: Dict[str, Any] = {"type_": pdef.get("type", "STRING").upper()}
            if "description" in pdef:
                gprop["description"] = pdef["description"]
            if "enum" in pdef:
                gprop["enum"] = pdef["enum"]
            if pdef.get("type") == "array" and "items" in pdef:
                gprop["type_"] = "ARRAY"
                gprop["items"] = {"type_": pdef["items"].get("type", "STRING").upper()}
            props[pname] = gprop
        declarations.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": {
                "type_": "OBJECT",
                "properties": props,
                "required": schema.get("required", []),
            },
        })
    return declarations


def _get_openai_api_key() -> str:
    """Get OpenAI API key from settings or environment."""
    try:
        from ..api.app_settings import settings_store
        ai = settings_store.get("ai", {})
        key = ai.get("openai_api_key", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


def _get_gemini_api_key() -> str:
    """Get Gemini API key from settings or environment."""
    try:
        from ..api.app_settings import settings_store
        ai = settings_store.get("ai", {})
        key = ai.get("gemini_api_key", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


def _detect_best_provider() -> str:
    """Auto-detect the best available provider based on configured keys."""
    if _get_api_key():  # Anthropic
        return LLM_PROVIDER_ANTHROPIC
    if _get_openai_api_key():
        return LLM_PROVIDER_OPENAI
    if _get_gemini_api_key():
        return LLM_PROVIDER_GEMINI
    return LLM_PROVIDER_ANTHROPIC  # default


def get_available_providers() -> List[str]:
    """Return a list of all providers that have a valid API key configured.

    Used by the orchestrator to distribute agents across multiple providers
    when the user has configured more than one API key.
    """
    available: List[str] = []
    if _get_api_key():
        available.append(LLM_PROVIDER_ANTHROPIC)
    if HAS_OPENAI and _get_openai_api_key():
        available.append(LLM_PROVIDER_OPENAI)
    if HAS_GEMINI and _get_gemini_api_key():
        available.append(LLM_PROVIDER_GEMINI)
    return available


# ═══════════════════════════════════════════════════════════════════
# Agentic Base Class — multi-turn tool-use loop (multi-provider)
# ═══════════════════════════════════════════════════════════════════

class AgenticDetector(BaseAgent):
    """Base class for tool-using agentic detectors.

    Supports multiple LLM providers for the agentic tool-use loop:
      - **anthropic** (default): Claude via Anthropic SDK tool_use
      - **openai**: GPT-4o via OpenAI SDK function calling
      - **gemini**: Gemini via Google GenAI SDK function calling

    The DataToolkit (9 tools) is shared across all providers — only the
    LLM "brain" and its tool-call protocol differ.

    Subclasses define:
      - name, perspective
      - _system_prompt(system_type) → str
      - _initial_user_message(system_type, system_name, data_summary, metadata_context) → str
      - _fallback_analyze(system_type, data_profile) → List[AgentFinding]
    """

    # Agentic detectors get slightly more time because they do multiple turns
    agent_timeout: int = 120

    # Override per-agent or set globally.  None = auto-detect.
    llm_provider: Optional[str] = None

    # Model overrides per provider (subclass or set at runtime)
    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-2.0-flash"

    def __init__(self) -> None:
        super().__init__()
        self._toolkit: Optional[DataToolkit] = None

    def set_toolkit(self, toolkit: DataToolkit) -> None:
        """Attach the data toolkit (called by the orchestrator before analyze)."""
        self._toolkit = toolkit

    def _resolve_provider(self) -> str:
        """Return the effective provider string."""
        if self.llm_provider:
            return self.llm_provider
        # Check app settings for a global override
        try:
            from ..api.app_settings import settings_store
            prov = settings_store.get("ai", {}).get("agentic_llm_provider", "")
            if prov in (LLM_PROVIDER_ANTHROPIC, LLM_PROVIDER_OPENAI, LLM_PROVIDER_GEMINI):
                return prov
        except Exception:
            pass
        return _detect_best_provider()

    # ── Main entry ────────────────────────────────────────────────

    async def analyze(self, system_type: str, system_name: str,
                      data_profile: Dict, metadata_context: str = "") -> List[AgentFinding]:
        """Run multi-turn tool-use analysis loop.

        Tries the configured provider first.  If that provider has no API key,
        cascades through the other providers before falling back to the
        heuristic (non-LLM) fallback.  This means: if the user hasn't
        entered *any* key, nothing crashes — the agent simply runs its
        rule-based fallback and logs a warning.
        """
        if not self._toolkit:
            logger.warning("[%s] No toolkit — using fallback", self.name)
            return self._fallback_analyze(system_type, data_profile)

        preferred = self._resolve_provider()

        # Build a priority-ordered list: preferred first, then the others
        all_providers = [LLM_PROVIDER_ANTHROPIC, LLM_PROVIDER_OPENAI, LLM_PROVIDER_GEMINI]
        order = [preferred] + [p for p in all_providers if p != preferred]

        for provider in order:
            has_key = (
                (provider == LLM_PROVIDER_ANTHROPIC and _get_api_key()) or
                (provider == LLM_PROVIDER_OPENAI and HAS_OPENAI and _get_openai_api_key()) or
                (provider == LLM_PROVIDER_GEMINI and HAS_GEMINI and _get_gemini_api_key())
            )
            if not has_key:
                continue

            logger.info("[%s] Using LLM provider: %s", self.name, provider)
            if provider == LLM_PROVIDER_OPENAI:
                return await self._run_openai_loop(system_type, system_name, data_profile, metadata_context)
            elif provider == LLM_PROVIDER_GEMINI:
                return await self._run_gemini_loop(system_type, system_name, data_profile, metadata_context)
            else:
                return await self._run_anthropic_loop(system_type, system_name, data_profile, metadata_context)

        # No provider has a key — skip gracefully
        logger.warning("[%s] No LLM API key configured — running heuristic fallback only", self.name)
        return self._fallback_analyze(system_type, data_profile)

    # ── Anthropic tool-use loop (original) ────────────────────────

    async def _run_anthropic_loop(self, system_type: str, system_name: str,
                                  data_profile: Dict, metadata_context: str) -> List[AgentFinding]:
        self._init_client()
        if not self.client:
            logger.warning("[%s] No Anthropic client — using fallback", self.name)
            return self._fallback_analyze(system_type, data_profile)

        data_summary = self._build_data_summary(data_profile)
        initial_msg = self._initial_user_message(system_type, system_name, data_summary, metadata_context)

        messages = [{"role": "user", "content": initial_msg}]
        tools = DataToolkit.tool_definitions()
        system_prompt = self._system_prompt(system_type)

        logger.info("[%s] Starting Anthropic agentic loop (max %d turns)...", self.name, MAX_TOOL_TURNS)
        t_start = time.time()

        for turn in range(MAX_TOOL_TURNS):
            elapsed = time.time() - t_start
            if elapsed > self.agent_timeout:
                logger.warning("[%s] Agent timeout after %.1fs, %d turns", self.name, elapsed, turn)
                break

            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        system=system_prompt,
                        tools=tools,
                        messages=messages,
                    ),
                    timeout=max(30, self.agent_timeout - elapsed),
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error("[%s] Turn %d failed: %s", self.name, turn + 1, exc)
                break

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [b for b in assistant_content if b.type == "tool_use"]

            if not tool_uses:
                text_blocks = [b.text for b in assistant_content if b.type == "text"]
                full_text = "\n".join(text_blocks)
                logger.info("[%s] Completed in %d turns, %.1fs", self.name, turn + 1, time.time() - t_start)
                findings = self._parse_response(full_text)
                logger.info("[%s] Parsed %d findings", self.name, len(findings))
                return findings

            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input
                logger.debug("[%s] Turn %d: calling %s(%s)", self.name, turn + 1, tool_name,
                             json.dumps(tool_input, default=str)[:200])

                result = self._toolkit.execute_tool(tool_name, tool_input)
                result_json = json.dumps(result, default=str)
                if len(result_json) > 8000:
                    result_json = result_json[:8000] + "...(truncated)"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_json,
                })

            messages.append({"role": "user", "content": tool_results})

        logger.warning("[%s] Exhausted %d turns — using fallback", self.name, MAX_TOOL_TURNS)
        return self._fallback_analyze(system_type, data_profile)

    # ── OpenAI function-calling loop ──────────────────────────────

    async def _run_openai_loop(self, system_type: str, system_name: str,
                               data_profile: Dict, metadata_context: str) -> List[AgentFinding]:
        api_key = _get_openai_api_key()
        if not HAS_OPENAI or not api_key:
            logger.warning("[%s] OpenAI not available — using fallback", self.name)
            return self._fallback_analyze(system_type, data_profile)

        client = AsyncOpenAI(api_key=api_key)

        data_summary = self._build_data_summary(data_profile)
        initial_msg = self._initial_user_message(system_type, system_name, data_summary, metadata_context)
        system_prompt = self._system_prompt(system_type)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_msg},
        ]
        tools = _tools_to_openai_format(DataToolkit.tool_definitions())

        logger.info("[%s] Starting OpenAI agentic loop (model=%s, max %d turns)...",
                    self.name, self.openai_model, MAX_TOOL_TURNS)
        t_start = time.time()

        for turn in range(MAX_TOOL_TURNS):
            elapsed = time.time() - t_start
            if elapsed > self.agent_timeout:
                logger.warning("[%s] Agent timeout after %.1fs, %d turns", self.name, elapsed, turn)
                break

            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        tools=tools,
                        max_tokens=4096,
                    ),
                    timeout=max(30, self.agent_timeout - elapsed),
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error("[%s] OpenAI turn %d failed: %s", self.name, turn + 1, exc)
                break

            choice = response.choices[0]
            assistant_msg = choice.message
            messages.append(assistant_msg)

            # Check for tool calls
            if not assistant_msg.tool_calls:
                # Model finished — extract text
                full_text = assistant_msg.content or ""
                logger.info("[%s] OpenAI completed in %d turns, %.1fs", self.name, turn + 1, time.time() - t_start)
                findings = self._parse_response(full_text)
                logger.info("[%s] Parsed %d findings", self.name, len(findings))
                return findings

            # Execute tool calls
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                logger.debug("[%s] OpenAI turn %d: calling %s(%s)", self.name, turn + 1,
                             tool_name, json.dumps(tool_input, default=str)[:200])

                result = self._toolkit.execute_tool(tool_name, tool_input)
                result_json = json.dumps(result, default=str)
                if len(result_json) > 8000:
                    result_json = result_json[:8000] + "...(truncated)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })

        logger.warning("[%s] OpenAI exhausted %d turns — using fallback", self.name, MAX_TOOL_TURNS)
        return self._fallback_analyze(system_type, data_profile)

    # ── Gemini function-calling loop ──────────────────────────────

    async def _run_gemini_loop(self, system_type: str, system_name: str,
                               data_profile: Dict, metadata_context: str) -> List[AgentFinding]:
        api_key = _get_gemini_api_key()
        if not HAS_GEMINI or not api_key:
            logger.warning("[%s] Gemini not available — using fallback", self.name)
            return self._fallback_analyze(system_type, data_profile)

        genai.configure(api_key=api_key)

        data_summary = self._build_data_summary(data_profile)
        initial_msg = self._initial_user_message(system_type, system_name, data_summary, metadata_context)
        system_prompt = self._system_prompt(system_type)

        # Build Gemini tool declarations
        declarations = _tools_to_gemini_declarations(DataToolkit.tool_definitions())
        gemini_tools = genai.protos.Tool(function_declarations=declarations)

        model = genai.GenerativeModel(
            model_name=self.gemini_model,
            system_instruction=system_prompt,
            tools=[gemini_tools],
        )

        logger.info("[%s] Starting Gemini agentic loop (model=%s, max %d turns)...",
                    self.name, self.gemini_model, MAX_TOOL_TURNS)
        t_start = time.time()

        # Gemini uses a chat session for multi-turn
        loop = asyncio.get_event_loop()
        chat = model.start_chat()

        # Send initial message
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: chat.send_message(initial_msg)),
                timeout=max(30, self.agent_timeout),
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.error("[%s] Gemini initial message failed: %s", self.name, exc)
            return self._fallback_analyze(system_type, data_profile)

        for turn in range(MAX_TOOL_TURNS):
            elapsed = time.time() - t_start
            if elapsed > self.agent_timeout:
                logger.warning("[%s] Gemini timeout after %.1fs, %d turns", self.name, elapsed, turn)
                break

            # Check for function calls in the response
            fn_calls = []
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    fn_calls.append(part.function_call)

            if not fn_calls:
                # Model finished — extract text
                full_text = response.text or ""
                logger.info("[%s] Gemini completed in %d turns, %.1fs", self.name, turn + 1, time.time() - t_start)
                findings = self._parse_response(full_text)
                logger.info("[%s] Parsed %d findings", self.name, len(findings))
                return findings

            # Execute function calls and send results back
            fn_responses = []
            for fc in fn_calls:
                tool_name = fc.name
                tool_input = dict(fc.args) if fc.args else {}
                logger.debug("[%s] Gemini turn %d: calling %s(%s)", self.name, turn + 1,
                             tool_name, json.dumps(tool_input, default=str)[:200])

                result = self._toolkit.execute_tool(tool_name, tool_input)
                result_json = json.dumps(result, default=str)
                if len(result_json) > 8000:
                    result_json = result_json[:8000] + "...(truncated)"

                fn_responses.append(
                    genai.protos.Part(function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": result_json},
                    ))
                )

            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: chat.send_message(fn_responses)),
                    timeout=max(30, self.agent_timeout - elapsed),
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error("[%s] Gemini turn %d failed: %s", self.name, turn + 1, exc)
                break

        logger.warning("[%s] Gemini exhausted %d turns — using fallback", self.name, MAX_TOOL_TURNS)
        return self._fallback_analyze(system_type, data_profile)

    # Subclasses must override:
    def _initial_user_message(self, system_type: str, system_name: str,
                              data_summary: str, metadata_context: str) -> str:
        return ""


# ═══════════════════════════════════════════════════════════════════
# 1. Autonomous Explorer
# ═══════════════════════════════════════════════════════════════════

class AutonomousExplorer(AgenticDetector):
    """An open-ended data exploration agent.  It decides what to look at,
    runs tests, follows leads, and reports what it finds.  No predefined
    analysis script — it's entirely autonomous."""

    name = "Autonomous Explorer"
    perspective = "open-ended data exploration"

    def _system_prompt(self, system_type: str) -> str:
        return (
            f"You are an autonomous data exploration agent analysing telemetry from a {system_type} system. "
            "You have tools to query columns, run statistical tests, detect outliers, compute correlations, "
            "and slice data windows.  Your job is to explore the data like a curious engineer:\n\n"
            "1. Start by listing columns and understanding the data shape.\n"
            "2. Pick interesting columns — look at statistics, check distributions.\n"
            "3. Follow leads: if something looks odd, dig deeper with more specific tools.\n"
            "4. Check correlations between related parameters.\n"
            "5. Look for drift, regime changes, outliers, frozen sensors.\n\n"
            "You have up to 12 tool calls. Use them wisely — prioritise the most suspicious signals.\n\n"
            "When you're done exploring, output your findings as a JSON array with the standard format:\n"
            '[{"type":"...", "severity":"critical|high|medium|low", "title":"...", '
            '"description":"...", "explanation":"...", "possible_causes":["..."], '
            '"recommendations":[{"priority":"...","action":"..."}], '
            '"affected_fields":["..."], "confidence":0.0-1.0, "impact_score":0-100}]\n\n'
            "Output ONLY the JSON array in your final message (after all tool calls)."
        )

    def _initial_user_message(self, system_type: str, system_name: str,
                              data_summary: str, metadata_context: str) -> str:
        msg = (
            f"Explore this {system_type} system's data and find anomalies.\n"
            f"System: {system_name}\n\n"
            f"DATA OVERVIEW:\n{data_summary}\n\n"
        )
        if metadata_context:
            msg += f"CONTEXT: {metadata_context}\n\n"
        msg += (
            "Start by listing the columns, then investigate whichever ones look "
            "most interesting or suspicious.  Use multiple tools to build a complete picture."
        )
        return msg

    def _fallback_analyze(self, system_type: str, data_profile: Dict) -> List[AgentFinding]:
        # Simple heuristic fallback
        findings = []
        for f in data_profile.get("fields", []):
            if f.get("std") and f.get("mean") and abs(f["mean"]) > 0:
                cv = f["std"] / abs(f["mean"])
                if cv > 1.0:
                    findings.append(AgentFinding(
                        agent_name=self.name,
                        anomaly_type="high_variability",
                        severity="medium",
                        title=f"High variability in {f['name']}",
                        description=f"Coefficient of variation = {cv:.2f}",
                        natural_language_explanation=f"The field {f['name']} has unusually high variability.",
                        affected_fields=[f["name"]],
                        confidence=0.6, impact_score=40,
                    ))
        return findings


# ═══════════════════════════════════════════════════════════════════
# 2. Statistical Investigator
# ═══════════════════════════════════════════════════════════════════

class StatisticalInvestigator(AgenticDetector):
    """Focuses on rigorous statistical testing.  Runs normality tests,
    stationarity checks, distribution comparisons, and formal outlier tests."""

    name = "Statistical Investigator"
    perspective = "formal statistical testing"

    def _system_prompt(self, system_type: str) -> str:
        return (
            f"You are a statistician analysing telemetry from a {system_type} system. "
            "You focus on FORMAL statistical tests — not just eyeballing numbers.\n\n"
            "Your approach:\n"
            "1. Check each numeric column for normality (Shapiro-Wilk or Anderson-Darling).\n"
            "2. Test for stationarity (ADF test) — non-stationary signals may indicate drift.\n"
            "3. Compare first-half vs second-half distributions (Mann-Whitney, KS test).\n"
            "4. Run Grubbs' test for formal outlier identification.\n"
            "5. Check variance stability (Levene's test).\n\n"
            "Report findings with p-values and effect sizes.  Be rigorous.\n\n"
            "When you're done, output your findings as a JSON array:\n"
            '[{"type":"...", "severity":"...", "title":"...", "description":"...", '
            '"explanation":"...", "possible_causes":["..."], '
            '"recommendations":[{"priority":"...","action":"..."}], '
            '"affected_fields":["..."], "confidence":0.0-1.0, "impact_score":0-100}]\n\n'
            "Output ONLY the JSON array in your final message."
        )

    def _initial_user_message(self, system_type: str, system_name: str,
                              data_summary: str, metadata_context: str) -> str:
        return (
            f"Run formal statistical tests on this {system_type} data.\n"
            f"System: {system_name}\n\n"
            f"DATA OVERVIEW:\n{data_summary}\n\n"
            "Start with list_columns, then systematically test each numeric column. "
            "Focus on the columns most likely to reveal real anomalies (not noise)."
        )

    def _fallback_analyze(self, system_type: str, data_profile: Dict) -> List[AgentFinding]:
        return []


# ═══════════════════════════════════════════════════════════════════
# 3. Correlation Hunter
# ═══════════════════════════════════════════════════════════════════

class CorrelationHunter(AgenticDetector):
    """Specialises in finding unexpected correlations and correlation
    breakdowns between parameters."""

    name = "Correlation Hunter"
    perspective = "inter-parameter relationship analysis"

    def _system_prompt(self, system_type: str) -> str:
        return (
            f"You are a correlation analysis specialist for {system_type} systems. "
            "Your mission is to find:\n"
            "1. Unexpected strong correlations (may indicate common-mode failure).\n"
            "2. Expected correlations that are MISSING (may indicate sensor fault).\n"
            "3. Correlations that BREAK DOWN over time (rolling correlation changes sign).\n\n"
            "Use the correlate_fields tool to check pairs of columns.  Start with the "
            "data overview to identify which pairs to check, then investigate systematically.\n\n"
            f"For a {system_type} system, think about which parameters SHOULD be correlated "
            "(e.g., temperature-current, vibration-acoustic) and test those specifically.\n\n"
            "When you're done, output your findings as a JSON array:\n"
            '[{"type":"...", "severity":"...", "title":"...", "description":"...", '
            '"explanation":"...", "possible_causes":["..."], '
            '"recommendations":[{"priority":"...","action":"..."}], '
            '"affected_fields":["..."], "confidence":0.0-1.0, "impact_score":0-100}]\n\n'
            "Output ONLY the JSON array in your final message."
        )

    def _initial_user_message(self, system_type: str, system_name: str,
                              data_summary: str, metadata_context: str) -> str:
        return (
            f"Investigate correlations in this {system_type} data.\n"
            f"System: {system_name}\n\n"
            f"DATA OVERVIEW:\n{data_summary}\n\n"
            "Start by listing columns, then check correlations between pairs that "
            "should (or shouldn't) be related for this system type."
        )

    def _fallback_analyze(self, system_type: str, data_profile: Dict) -> List[AgentFinding]:
        return []


# ═══════════════════════════════════════════════════════════════════
# 4. Drift & Shift Detector
# ═══════════════════════════════════════════════════════════════════

class DriftShiftDetector(AgenticDetector):
    """Focuses on detecting regime changes, gradual drifts, and
    concept shifts in the data over time."""

    name = "Drift & Shift Detector"
    perspective = "temporal drift and regime change detection"

    def _system_prompt(self, system_type: str) -> str:
        return (
            f"You are a drift and regime-change detection specialist for {system_type} systems. "
            "Your mission:\n"
            "1. Compare early data vs late data using compare_segments.\n"
            "2. Use compute_rolling to find where rolling mean/std changes.\n"
            "3. Check stationarity with run_statistical_test (adfuller).\n"
            "4. Look for step changes, gradual drift, and volatility shifts.\n\n"
            "Drift is dangerous because it's invisible day-to-day but accumulates into failures.\n\n"
            "When you're done, output your findings as a JSON array:\n"
            '[{"type":"...", "severity":"...", "title":"...", "description":"...", '
            '"explanation":"...", "possible_causes":["..."], '
            '"recommendations":[{"priority":"...","action":"..."}], '
            '"affected_fields":["..."], "confidence":0.0-1.0, "impact_score":0-100}]\n\n'
            "Output ONLY the JSON array in your final message."
        )

    def _initial_user_message(self, system_type: str, system_name: str,
                              data_summary: str, metadata_context: str) -> str:
        return (
            f"Search for drift and regime changes in this {system_type} data.\n"
            f"System: {system_name}\n\n"
            f"DATA OVERVIEW:\n{data_summary}\n\n"
            "Start by listing columns, then use compare_segments and compute_rolling "
            "to find temporal patterns. Focus on critical parameters first."
        )

    def _fallback_analyze(self, system_type: str, data_profile: Dict) -> List[AgentFinding]:
        return []


# ═══════════════════════════════════════════════════════════════════
# 5. Physics Constraint Agent
# ═══════════════════════════════════════════════════════════════════

class PhysicsConstraintAgent(AgenticDetector):
    """Validates physical consistency: energy conservation, monotonicity
    where expected, sensor cross-validation, impossible value combinations."""

    name = "Physics Constraint Agent"
    perspective = "physical law validation and cross-sensor consistency"

    def _system_prompt(self, system_type: str) -> str:
        return (
            f"You are a physics and engineering constraint validation agent for {system_type} systems. "
            "You check whether the data is physically consistent:\n\n"
            "1. Are values within physically possible ranges? (e.g., negative pressure in a gauge sensor)\n"
            "2. Do correlated parameters behave consistently? (e.g., more current → more heat)\n"
            "3. Are there impossible combinations? (e.g., high RPM with zero vibration)\n"
            "4. Do conservation laws hold? (e.g., input power ≈ output power + losses)\n"
            "5. Are there frozen sensors reporting constant values while physics says they should vary?\n\n"
            "Use the tools to query specific columns, check correlations, and slice windows "
            "where violations occur.\n\n"
            "When you're done, output your findings as a JSON array:\n"
            '[{"type":"...", "severity":"...", "title":"...", "description":"...", '
            '"explanation":"...", "possible_causes":["..."], '
            '"recommendations":[{"priority":"...","action":"..."}], '
            '"affected_fields":["..."], "confidence":0.0-1.0, "impact_score":0-100}]\n\n'
            "Output ONLY the JSON array in your final message."
        )

    def _initial_user_message(self, system_type: str, system_name: str,
                              data_summary: str, metadata_context: str) -> str:
        return (
            f"Validate physical constraints in this {system_type} data.\n"
            f"System: {system_name}\n\n"
            f"DATA OVERVIEW:\n{data_summary}\n\n"
            "Start by listing columns to understand what physical quantities are measured, "
            "then check cross-parameter consistency and physical plausibility."
        )

    def _fallback_analyze(self, system_type: str, data_profile: Dict) -> List[AgentFinding]:
        return []


# ═══════════════════════════════════════════════════════════════════
# All agentic detectors list
# ═══════════════════════════════════════════════════════════════════

ALL_AGENTIC_DETECTORS = [
    AutonomousExplorer,
    StatisticalInvestigator,
    CorrelationHunter,
    DriftShiftDetector,
    PhysicsConstraintAgent,
]

AGENTIC_AGENT_NAMES = [cls.name for cls in [
    AutonomousExplorer, StatisticalInvestigator, CorrelationHunter,
    DriftShiftDetector, PhysicsConstraintAgent,
]]
