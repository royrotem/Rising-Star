"""
Agent Orchestrator

Coordinates multiple AI agents to perform complex analytical tasks.
Manages task distribution, agent communication, and result aggregation.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from .base import (
    BaseAgent,
    AgentMessage,
    AgentTask,
    SchemaDiscoveryAgent,
    AnomalyDetectionAgent,
    RootCauseAgent,
    BlindSpotAgent,
)


class AgentOrchestrator:
    """
    Orchestrates the workforce of AI agents.
    
    The orchestrator is responsible for:
    - Managing agent lifecycle
    - Distributing tasks to appropriate agents
    - Coordinating multi-agent workflows
    - Aggregating results from multiple agents
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, Any] = {}
        self.running = False
        
        # Initialize default agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize the default set of agents."""
        agents = [
            SchemaDiscoveryAgent(),
            AnomalyDetectionAgent(),
            RootCauseAgent(),
            BlindSpotAgent(),
        ]
        
        for agent in agents:
            self.register_agent(agent)

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]

    async def start(self):
        """Start the orchestrator and all agents."""
        self.running = True
        
        # Start all agents
        agent_tasks = []
        for agent in self.agents.values():
            agent_tasks.append(asyncio.create_task(agent.run()))
        
        # Start task processor
        await self._process_tasks()

    async def stop(self):
        """Stop the orchestrator and all agents."""
        self.running = False

    async def submit_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        target_agent: Optional[str] = None
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_type: Type of task to execute
            parameters: Task parameters
            target_agent: Specific agent to target (optional)
        
        Returns:
            Task ID
        """
        task = AgentTask(
            task_type=task_type,
            parameters=parameters,
        )
        
        await self.task_queue.put((task, target_agent))
        return task.id

    async def get_result(self, task_id: str, timeout: float = 30.0) -> Optional[Dict]:
        """Get result of a completed task."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if task_id in self.results:
                return self.results.pop(task_id)
            await asyncio.sleep(0.1)
        
        return None

    async def _process_tasks(self):
        """Process tasks from the queue."""
        while self.running:
            try:
                task, target_agent = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                # Find appropriate agent
                agent = self._find_agent(task.task_type, target_agent)
                
                if agent:
                    # Send task to agent
                    message = AgentMessage(
                        sender="orchestrator",
                        recipient=agent.agent_id,
                        message_type="task",
                        content=task.__dict__,
                    )
                    await agent.receive_message(message)
                    
                    # Wait for completion and store result
                    while agent.current_task and agent.current_task.id == task.id:
                        await asyncio.sleep(0.1)
                    
                    self.results[task.id] = task.result
                else:
                    self.results[task.id] = {"error": "No suitable agent found"}
                    
            except asyncio.TimeoutError:
                continue

    def _find_agent(
        self, 
        task_type: str, 
        target_agent: Optional[str]
    ) -> Optional[BaseAgent]:
        """Find an agent capable of handling the task."""
        if target_agent and target_agent in self.agents:
            return self.agents[target_agent]
        
        for agent in self.agents.values():
            if task_type in agent.get_capabilities():
                return agent
        
        return None

    async def analyze_system_full(
        self,
        system_id: str,
        data: List[Dict],
        events: List[Dict],
        design_specs: Dict,
    ) -> Dict[str, Any]:
        """
        Perform full system analysis using multiple agents.
        This orchestrates a complete analysis workflow.
        """
        results = {
            "system_id": system_id,
            "timestamp": datetime.utcnow().isoformat(),
            "schema_discovery": None,
            "anomalies": None,
            "root_causes": None,
            "blind_spots": None,
            "recommendations": [],
        }

        # Step 1: Schema Discovery
        schema_task_id = await self.submit_task(
            "discover_schema",
            {"data": data},
            "schema_discovery_agent"
        )
        results["schema_discovery"] = await self.get_result(schema_task_id)

        # Step 2: Anomaly Detection
        anomaly_task_id = await self.submit_task(
            "detect_anomalies",
            {"data": data, "baseline": None},
            "anomaly_detection_agent"
        )
        results["anomalies"] = await self.get_result(anomaly_task_id)

        # Step 3: Root Cause Analysis (for each anomaly)
        if results["anomalies"] and results["anomalies"].get("anomalies"):
            root_causes = []
            for anomaly in results["anomalies"]["anomalies"][:5]:  # Top 5
                rca_task_id = await self.submit_task(
                    "analyze_root_cause",
                    {"anomaly": anomaly, "context": {"events": events}},
                    "root_cause_agent"
                )
                rca_result = await self.get_result(rca_task_id)
                if rca_result:
                    root_causes.append(rca_result)
            results["root_causes"] = root_causes

        # Step 4: Blind Spot Detection
        blind_spot_task_id = await self.submit_task(
            "detect_blind_spots",
            {
                "anomalies": results["anomalies"].get("anomalies", []) if results["anomalies"] else [],
                "schema": results["schema_discovery"]
            },
            "blind_spot_agent"
        )
        results["blind_spots"] = await self.get_result(blind_spot_task_id)

        # Aggregate recommendations
        results["recommendations"] = self._aggregate_recommendations(results)

        return results

    def _aggregate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Aggregate recommendations from all analysis components."""
        recommendations = []

        # From root causes
        if analysis.get("root_causes"):
            for rca in analysis["root_causes"]:
                if rca and rca.get("recommendations"):
                    recommendations.extend(rca["recommendations"])

        # From blind spots
        if analysis.get("blind_spots"):
            for spot in analysis["blind_spots"]:
                recommendations.append({
                    "type": "sensor_addition",
                    "priority": "medium",
                    "action": f"Add sensor to address: {spot.get('diagnosis_gap', 'data gap')}",
                    "source": "blind_spot_detection",
                })

        # Deduplicate and prioritize
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            key = (rec.get("type"), rec.get("action"))
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)

        return sorted(
            unique_recommendations,
            key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.get("priority", "low"), 0),
            reverse=True
        )

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            agent_id: {
                "name": agent.name,
                "status": agent.status,
                "capabilities": agent.get_capabilities(),
                "current_task": agent.current_task.id if agent.current_task else None,
            }
            for agent_id, agent in self.agents.items()
        }


# Global orchestrator instance
orchestrator = AgentOrchestrator()
