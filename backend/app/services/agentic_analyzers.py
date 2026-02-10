"""
External AI Agent Integrations for UAIE

Integrates with real AI agent APIs that have autonomous capabilities:
1. OpenAI Assistants API - Code Interpreter (executes Python)
2. Google Gemini API - Code Execution capability

These are REAL agents that can explore data autonomously, not just LLM prompts.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

logger = logging.getLogger("uaie.agentic_analyzers")

# ============================================================================
# OpenAI Assistants API Integration (Code Interpreter)
# ============================================================================

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI SDK not installed - OpenAI agents disabled")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("Google Generative AI SDK not installed - Gemini agents disabled")


@dataclass
class AgentFinding:
    """A finding from an external AI agent."""
    title: str
    description: str
    severity: str
    affected_fields: List[str]
    evidence: Dict[str, Any]
    recommendation: str
    agent_name: str
    agent_type: str  # "openai_assistant" or "gemini"


class OpenAIAssistantAgent:
    """
    Uses OpenAI Assistants API with Code Interpreter.

    The Code Interpreter can:
    - Execute Python code in a sandbox
    - Analyze uploaded files (CSV, etc.)
    - Create visualizations
    - Perform statistical analysis
    """

    name = "OpenAI Code Interpreter"
    agent_type = "openai_assistant"

    def __init__(self):
        self.client = None
        self.assistant_id = None
        api_key = os.environ.get("OPENAI_API_KEY", "")

        if HAS_OPENAI and api_key:
            self.client = AsyncOpenAI(api_key=api_key)
            logger.info("[OpenAI Agent] Initialized with API key")
        else:
            logger.warning("[OpenAI Agent] No API key or SDK not installed")

    async def analyze(
        self,
        df: pd.DataFrame,
        system_type: str,
        system_name: str,
        schema_context: str = ""
    ) -> List[AgentFinding]:
        """
        Run analysis using OpenAI Assistant with Code Interpreter.

        1. Create a temporary CSV file
        2. Upload it to OpenAI
        3. Create an Assistant with Code Interpreter
        4. Run analysis thread
        5. Parse findings from response
        """
        if not self.client:
            logger.warning("[OpenAI Agent] No client available")
            return []

        findings = []
        file_id = None
        assistant_id = None

        try:
            # Step 1: Save DataFrame to temp CSV
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f, index=False)
                temp_path = f.name

            logger.info("[OpenAI Agent] Created temp CSV: %s (%d rows, %d cols)",
                       temp_path, len(df), len(df.columns))

            # Step 2: Upload file to OpenAI
            with open(temp_path, 'rb') as f:
                file_response = await self.client.files.create(
                    file=f,
                    purpose="assistants"
                )
            file_id = file_response.id
            logger.info("[OpenAI Agent] Uploaded file: %s", file_id)

            # Step 3: Create Assistant with Code Interpreter
            assistant = await self.client.beta.assistants.create(
                name="Data Anomaly Investigator",
                instructions=f"""You are an expert data analyst investigating anomalies in industrial system data.

System Type: {system_type}
System Name: {system_name}
Schema Info: {schema_context}

Your task:
1. Load and explore the uploaded CSV data
2. Perform statistical analysis on all numeric columns
3. Detect outliers using multiple methods (Z-score, IQR)
4. Find correlations between variables
5. Identify any anomalies, patterns, or data quality issues
6. Look for time-series patterns if timestamp data exists

For each finding, output a JSON block like this:
```json
{{
    "title": "Brief title",
    "description": "Detailed explanation",
    "severity": "critical|high|medium|low|info",
    "affected_fields": ["field1", "field2"],
    "evidence": {{"stat": "value"}},
    "recommendation": "What to do about it"
}}
```

Be thorough and report ALL significant findings.""",
                model="gpt-4o",
                tools=[{"type": "code_interpreter"}],
            )
            assistant_id = assistant.id
            logger.info("[OpenAI Agent] Created assistant: %s", assistant_id)

            # Step 4: Create thread and run
            thread = await self.client.beta.threads.create()

            await self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content="Please analyze the uploaded data file for anomalies and patterns. Use Python code to explore the data thoroughly.",
                attachments=[{"file_id": file_id, "tools": [{"type": "code_interpreter"}]}]
            )

            # Run the assistant
            run = await self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
            )

            # Wait for completion (with timeout)
            start_time = time.time()
            timeout = 300  # 5 minutes

            while run.status in ["queued", "in_progress"]:
                if time.time() - start_time > timeout:
                    logger.error("[OpenAI Agent] Timeout waiting for run")
                    break

                await asyncio.sleep(2)
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                logger.info("[OpenAI Agent] Run status: %s", run.status)

            if run.status == "completed":
                # Get messages
                messages = await self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )

                # Parse findings from assistant messages
                for msg in messages.data:
                    if msg.role == "assistant":
                        for content in msg.content:
                            if content.type == "text":
                                findings.extend(
                                    self._parse_findings(content.text.value)
                                )
            else:
                logger.error("[OpenAI Agent] Run failed with status: %s", run.status)

            # Cleanup
            os.unlink(temp_path)

        except Exception as e:
            logger.error("[OpenAI Agent] Error: %s", e)
            import traceback
            logger.error(traceback.format_exc())

        finally:
            # Cleanup OpenAI resources
            try:
                if file_id:
                    await self.client.files.delete(file_id)
                if assistant_id:
                    await self.client.beta.assistants.delete(assistant_id)
            except Exception as e:
                logger.warning("[OpenAI Agent] Cleanup error: %s", e)

        logger.info("[OpenAI Agent] Found %d findings", len(findings))
        return findings

    def _parse_findings(self, text: str) -> List[AgentFinding]:
        """Parse JSON findings from assistant response."""
        findings = []

        # Find JSON blocks in the response
        import re
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                findings.append(AgentFinding(
                    title=data.get("title", "Untitled"),
                    description=data.get("description", ""),
                    severity=data.get("severity", "medium"),
                    affected_fields=data.get("affected_fields", []),
                    evidence=data.get("evidence", {}),
                    recommendation=data.get("recommendation", ""),
                    agent_name=self.name,
                    agent_type=self.agent_type,
                ))
            except json.JSONDecodeError:
                continue

        return findings


class GeminiCodeAgent:
    """
    Uses Google Gemini API with code execution capability.

    Gemini can execute Python code directly and analyze data.
    """

    name = "Gemini Code Executor"
    agent_type = "gemini"

    def __init__(self):
        self.model = None
        api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

        if HAS_GEMINI and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name='gemini-2.0-flash-exp',
                tools='code_execution'
            )
            logger.info("[Gemini Agent] Initialized with API key")
        else:
            logger.warning("[Gemini Agent] No API key or SDK not installed")

    async def analyze(
        self,
        df: pd.DataFrame,
        system_type: str,
        system_name: str,
        schema_context: str = ""
    ) -> List[AgentFinding]:
        """
        Run analysis using Gemini with code execution.
        """
        if not self.model:
            logger.warning("[Gemini Agent] No model available")
            return []

        findings = []

        try:
            # Prepare data summary for Gemini
            data_summary = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "shape": df.shape,
                "sample": df.head(20).to_dict(orient="records"),
                "numeric_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            }

            prompt = f"""You are an expert data analyst. Analyze this industrial system data for anomalies.

System Type: {system_type}
System Name: {system_name}
Schema: {schema_context}

Data Summary:
- Columns: {data_summary['columns']}
- Shape: {data_summary['shape']}
- Types: {data_summary['dtypes']}

Sample Data (first 20 rows):
{json.dumps(data_summary['sample'], indent=2, default=str)}

Numeric Statistics:
{json.dumps(data_summary['numeric_stats'], indent=2, default=str)}

Please use Python code execution to:
1. Analyze the statistical properties
2. Detect outliers using Z-score and IQR methods
3. Find correlations between variables
4. Identify patterns and anomalies

For each finding, output a JSON block:
```json
{{
    "title": "Brief title",
    "description": "Detailed explanation",
    "severity": "critical|high|medium|low|info",
    "affected_fields": ["field1", "field2"],
    "evidence": {{"stat": "value"}},
    "recommendation": "What to do"
}}
```"""

            # Run Gemini with code execution
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )

            if response and response.text:
                findings = self._parse_findings(response.text)

        except Exception as e:
            logger.error("[Gemini Agent] Error: %s", e)
            import traceback
            logger.error(traceback.format_exc())

        logger.info("[Gemini Agent] Found %d findings", len(findings))
        return findings

    def _parse_findings(self, text: str) -> List[AgentFinding]:
        """Parse JSON findings from Gemini response."""
        findings = []

        import re
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                findings.append(AgentFinding(
                    title=data.get("title", "Untitled"),
                    description=data.get("description", ""),
                    severity=data.get("severity", "medium"),
                    affected_fields=data.get("affected_fields", []),
                    evidence=data.get("evidence", {}),
                    recommendation=data.get("recommendation", ""),
                    agent_name=self.name,
                    agent_type=self.agent_type,
                ))
            except json.JSONDecodeError:
                continue

        return findings


# ============================================================================
# Orchestrator for External AI Agents
# ============================================================================

class ExternalAgentOrchestrator:
    """Orchestrates external AI agent APIs."""

    def __init__(self):
        self.agents = []

        # Add available agents
        openai_agent = OpenAIAssistantAgent()
        if openai_agent.client:
            self.agents.append(openai_agent)

        gemini_agent = GeminiCodeAgent()
        if gemini_agent.model:
            self.agents.append(gemini_agent)

        logger.info("[ExternalAgentOrchestrator] Initialized with %d agents: %s",
                   len(self.agents), [a.name for a in self.agents])

    async def run_analysis(
        self,
        records: List[Dict],
        system_type: str,
        system_name: str,
        schema_context: str = "",
        selected_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run analysis using external AI agents."""

        if not self.agents:
            logger.warning("[ExternalAgentOrchestrator] No agents available")
            return {
                "findings": [],
                "agent_statuses": [],
                "total_findings": 0,
                "agents_used": [],
                "analysis_type": "external_agents",
                "error": "No external AI agents configured. Set OPENAI_API_KEY or GOOGLE_API_KEY."
            }

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Filter agents if specified
        if selected_agents:
            active_agents = [a for a in self.agents if a.name in selected_agents]
        else:
            active_agents = self.agents

        logger.info("=" * 60)
        logger.info("[ExternalAgentOrchestrator] Starting with %d agents: %s",
                   len(active_agents), [a.name for a in active_agents])

        all_findings = []
        agent_statuses = []

        # Run agents sequentially (to respect rate limits)
        for agent in active_agents:
            try:
                logger.info("[ExternalAgentOrchestrator] Running %s...", agent.name)
                t_start = time.time()

                findings = await agent.analyze(df, system_type, system_name, schema_context)

                t_elapsed = round(time.time() - t_start, 2)
                all_findings.extend(findings)

                agent_statuses.append({
                    "agent": agent.name,
                    "type": agent.agent_type,
                    "status": "success",
                    "findings": len(findings),
                    "duration_seconds": t_elapsed
                })
                logger.info("[ExternalAgentOrchestrator] %s completed in %.2fs with %d findings",
                           agent.name, t_elapsed, len(findings))

                # Delay between agents
                await asyncio.sleep(2)

            except Exception as e:
                logger.error("[ExternalAgentOrchestrator] %s failed: %s", agent.name, e)
                agent_statuses.append({
                    "agent": agent.name,
                    "type": agent.agent_type,
                    "status": "error",
                    "error": str(e)
                })

        logger.info("=" * 60)
        logger.info("[ExternalAgentOrchestrator] Complete: %d findings from %d agents",
                   len(all_findings), len(active_agents))

        return {
            "findings": [self._finding_to_dict(f) for f in all_findings],
            "agent_statuses": agent_statuses,
            "total_findings": len(all_findings),
            "agents_used": [a.name for a in active_agents],
            "analysis_type": "external_agents"
        }

    def _finding_to_dict(self, f: AgentFinding) -> Dict:
        return {
            "title": f.title,
            "description": f.description,
            "severity": f.severity,
            "affected_fields": f.affected_fields,
            "evidence": f.evidence,
            "recommendation": f.recommendation,
            "agent": f.agent_name,
            "agent_type": f.agent_type,
        }


# Global orchestrator instance
agentic_orchestrator = ExternalAgentOrchestrator()
