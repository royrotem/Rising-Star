"""
Zero-Knowledge Ingestion & Adaptive Discovery Service

This service handles the "blind" ingestion of raw data and autonomous
schema discovery using AI agents.
"""

import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import hashlib
import struct


@dataclass
class DiscoveredField:
    """A field discovered during schema analysis."""
    name: str
    inferred_type: str  # "numeric", "categorical", "timestamp", "binary"
    physical_unit: Optional[str] = None  # "celsius", "volts", "psi", etc.
    inferred_meaning: Optional[str] = None  # AI-inferred semantic meaning
    confidence: float = 0.0
    sample_values: List[Any] = None
    statistics: Dict[str, float] = None
    correlations: Dict[str, float] = None  # Correlation with other fields


@dataclass 
class FieldRelationship:
    """Discovered relationship between fields."""
    field_a: str
    field_b: str
    relationship_type: str  # "correlation", "causation", "derived", "inverse"
    strength: float  # 0-1
    description: str
    confidence: float


class IngestionService:
    """
    Handles data ingestion and autonomous schema discovery.
    Implements the "Zero-Knowledge" approach where the system learns
    the data structure without prior configuration.
    """

    def __init__(self):
        self.supported_formats = {
            'csv': self._parse_csv,
            'json': self._parse_json,
            'jsonl': self._parse_jsonl,
            'parquet': self._parse_parquet,
            'can': self._parse_can_bus,
            'bin': self._parse_binary,
        }
        self.physical_unit_patterns = self._load_unit_patterns()

    def _load_unit_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for inferring physical units from field names."""
        return {
            "temperature": ["temp", "celsius", "fahrenheit", "thermal", "heat"],
            "voltage": ["volt", "voltage", "vbat", "vcc", "vdd"],
            "current": ["current", "amp", "amps", "ibat"],
            "pressure": ["pressure", "psi", "bar", "pascal", "kpa"],
            "speed": ["speed", "velocity", "rpm", "rps"],
            "acceleration": ["accel", "acceleration", "g_force"],
            "position": ["position", "pos", "lat", "lon", "altitude", "alt"],
            "angle": ["angle", "yaw", "pitch", "roll", "heading"],
            "distance": ["distance", "range", "dist", "odometer"],
            "time": ["time", "timestamp", "epoch", "duration"],
            "frequency": ["freq", "frequency", "hz"],
            "power": ["power", "watt", "watts"],
            "energy": ["energy", "kwh", "joule"],
            "percentage": ["percent", "pct", "soc", "soh", "level"],
        }

    async def ingest_file(
        self,
        file_content: BinaryIO,
        filename: str,
        system_id: str,
        source_name: str
    ) -> Dict[str, Any]:
        """
        Ingest a file and perform autonomous schema discovery.
        
        Returns:
            Dictionary containing discovered schema and sample data
        """
        file_extension = filename.split('.')[-1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        parser = self.supported_formats[file_extension]
        records, raw_schema = await parser(file_content)

        # Perform autonomous schema discovery
        discovered_fields = await self._discover_schema(records, raw_schema)
        
        # Find relationships between fields
        relationships = await self._discover_relationships(records, discovered_fields)

        # Generate human-in-the-loop confirmation requests
        confirmation_requests = self._generate_confirmation_requests(
            discovered_fields, relationships
        )

        return {
            "system_id": system_id,
            "source_name": source_name,
            "record_count": len(records),
            "discovered_fields": [f.__dict__ for f in discovered_fields],
            "relationships": [r.__dict__ for r in relationships],
            "confirmation_requests": confirmation_requests,
            "sample_records": records,  # Return all records for storage
            "ingestion_timestamp": datetime.utcnow().isoformat(),
        }

    async def _parse_csv(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse CSV file and extract records."""
        content = file_content.read().decode('utf-8')
        df = pd.read_csv(pd.io.common.StringIO(content))
        records = df.to_dict('records')
        schema = {col: str(df[col].dtype) for col in df.columns}
        return records, schema

    async def _parse_json(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse JSON file."""
        content = file_content.read().decode('utf-8')
        data = json.loads(content)
        if isinstance(data, list):
            records = [self._flatten_dict(r) if isinstance(r, dict) else {"value": r} for r in data]
        else:
            records = [self._flatten_dict(data) if isinstance(data, dict) else {"value": data}]
        schema = self._infer_json_schema(records[0]) if records else {}
        return records, schema

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation for simplicity
                if v and isinstance(v[0], dict):
                    # If list of dicts, just store length and first item summary
                    items.append((f"{new_key}_count", len(v)))
                    if v:
                        first_item = self._flatten_dict(v[0], f"{new_key}_0", sep=sep)
                        items.extend(first_item.items())
                else:
                    items.append((new_key, str(v) if len(str(v)) < 200 else f"[{len(v)} items]"))
            else:
                items.append((new_key, v))
        return dict(items)

    async def _parse_jsonl(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse JSON Lines file."""
        content = file_content.read().decode('utf-8')
        records = [json.loads(line) for line in content.strip().split('\n') if line]
        schema = self._infer_json_schema(records[0]) if records else {}
        return records, schema

    async def _parse_parquet(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse Parquet file."""
        df = pd.read_parquet(file_content)
        records = df.to_dict('records')
        schema = {col: str(df[col].dtype) for col in df.columns}
        return records, schema

    async def _parse_can_bus(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse CAN bus log file."""
        content = file_content.read()
        records = []
        
        # Standard CAN frame: ID (4 bytes) + DLC (1 byte) + Data (0-8 bytes)
        offset = 0
        while offset < len(content) - 5:
            try:
                can_id = struct.unpack('<I', content[offset:offset+4])[0]
                dlc = content[offset+4]
                data = content[offset+5:offset+5+dlc]
                
                records.append({
                    'can_id': hex(can_id),
                    'dlc': dlc,
                    'data': data.hex(),
                    'timestamp': offset  # Placeholder
                })
                offset += 5 + dlc
            except:
                offset += 1

        schema = {'can_id': 'hex', 'dlc': 'int', 'data': 'bytes'}
        return records, schema

    async def _parse_binary(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse generic binary file with heuristic analysis."""
        content = file_content.read()
        
        # Attempt to detect structure through pattern analysis
        records = [{
            'raw_data': content[:1000].hex(),
            'size_bytes': len(content),
            'hash': hashlib.md5(content).hexdigest(),
        }]
        
        schema = {'binary': True, 'requires_format_specification': True}
        return records, schema

    def _infer_json_schema(self, record: Dict) -> Dict:
        """Infer schema from a JSON record."""
        schema = {}
        for key, value in record.items():
            if isinstance(value, bool):
                schema[key] = 'boolean'
            elif isinstance(value, int):
                schema[key] = 'integer'
            elif isinstance(value, float):
                schema[key] = 'float'
            elif isinstance(value, str):
                schema[key] = 'string'
            elif isinstance(value, list):
                schema[key] = 'array'
            elif isinstance(value, dict):
                schema[key] = 'object'
            else:
                schema[key] = 'unknown'
        return schema

    async def _discover_schema(
        self,
        records: List[Dict],
        raw_schema: Dict
    ) -> List[DiscoveredField]:
        """
        Autonomously discover field meanings and types.
        This is where the "AI Agent" learns the system's DNA.
        """
        if not records:
            return []

        # Filter out columns that contain unhashable types (dicts, lists)
        clean_records = []
        for record in records:
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, (dict, list)):
                    clean_record[k] = str(v)[:200] if v else None
                else:
                    clean_record[k] = v
            clean_records.append(clean_record)

        df = pd.DataFrame(clean_records)
        discovered = []

        for column in df.columns:
            try:
                sample_values = df[column].dropna().head(5).tolist()
            except Exception:
                sample_values = []

            field = DiscoveredField(
                name=column,
                inferred_type=self._infer_field_type(df[column]),
                sample_values=sample_values
            )

            # Infer physical unit from field name
            field.physical_unit = self._infer_physical_unit(column)
            
            # Infer semantic meaning
            field.inferred_meaning = self._infer_meaning(column, df[column])

            # Calculate statistics for numeric fields
            if field.inferred_type == 'numeric':
                field.statistics = {
                    'min': float(df[column].min()) if not df[column].isna().all() else None,
                    'max': float(df[column].max()) if not df[column].isna().all() else None,
                    'mean': float(df[column].mean()) if not df[column].isna().all() else None,
                    'std': float(df[column].std()) if not df[column].isna().all() else None,
                    'null_percentage': float(df[column].isna().sum() / len(df) * 100),
                }

            # Set confidence based on how certain we are about the inference
            field.confidence = self._calculate_confidence(field)
            
            discovered.append(field)

        return discovered

    def _infer_field_type(self, series: pd.Series) -> str:
        """Infer the type of a field."""
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'timestamp'
        elif pd.api.types.is_bool_dtype(series):
            return 'boolean'
        elif series.nunique() < len(series) * 0.1:  # Low cardinality
            return 'categorical'
        else:
            return 'string'

    def _infer_physical_unit(self, field_name: str) -> Optional[str]:
        """Infer physical unit from field name patterns."""
        field_lower = field_name.lower()
        for unit, patterns in self.physical_unit_patterns.items():
            if any(pattern in field_lower for pattern in patterns):
                return unit
        return None

    def _infer_meaning(self, field_name: str, series: pd.Series) -> str:
        """Infer semantic meaning of a field."""
        field_lower = field_name.lower()
        
        # Common patterns
        if any(p in field_lower for p in ['temp', 'thermal']):
            return "Temperature measurement"
        elif any(p in field_lower for p in ['battery', 'batt', 'soc']):
            return "Battery-related metric"
        elif any(p in field_lower for p in ['motor', 'engine']):
            return "Motor/engine parameter"
        elif any(p in field_lower for p in ['error', 'fault', 'warn']):
            return "Error or warning indicator"
        elif any(p in field_lower for p in ['gps', 'lat', 'lon', 'position']):
            return "Location/positioning data"
        elif any(p in field_lower for p in ['time', 'stamp', 'date']):
            return "Timestamp or duration"
        else:
            return f"Unknown - requires engineer confirmation"

    def _calculate_confidence(self, field: DiscoveredField) -> float:
        """Calculate confidence score for field inference."""
        confidence = 0.5  # Base confidence
        
        if field.physical_unit:
            confidence += 0.2
        if field.inferred_meaning and "Unknown" not in field.inferred_meaning:
            confidence += 0.2
        if field.statistics and field.statistics.get('null_percentage', 100) < 10:
            confidence += 0.1
            
        return min(confidence, 1.0)

    async def _discover_relationships(
        self,
        records: List[Dict],
        fields: List[DiscoveredField]
    ) -> List[FieldRelationship]:
        """Discover relationships and correlations between fields."""
        if not records or len(records) < 10:
            return []

        df = pd.DataFrame(records)
        numeric_cols = [f.name for f in fields if f.inferred_type == 'numeric']
        
        if len(numeric_cols) < 2:
            return []

        relationships = []
        correlation_matrix = df[numeric_cols].corr()

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i+1:]:
                corr = correlation_matrix.loc[col_a, col_b]
                
                if abs(corr) > 0.7:  # Strong correlation threshold
                    relationship_type = "positive_correlation" if corr > 0 else "negative_correlation"
                    
                    relationships.append(FieldRelationship(
                        field_a=col_a,
                        field_b=col_b,
                        relationship_type=relationship_type,
                        strength=abs(corr),
                        description=f"{col_a} and {col_b} show {'strong positive' if corr > 0 else 'strong negative'} correlation ({corr:.2f})",
                        confidence=min(abs(corr), 0.95)
                    ))

        return relationships

    def _generate_confirmation_requests(
        self,
        fields: List[DiscoveredField],
        relationships: List[FieldRelationship]
    ) -> List[Dict]:
        """
        Generate Human-in-the-Loop confirmation requests.
        These are presented to engineers to verify AI inferences.
        """
        requests = []

        for field in fields:
            if field.confidence < 0.8:
                requests.append({
                    "type": "field_confirmation",
                    "field_name": field.name,
                    "question": f"I believe '{field.name}' is a {field.inferred_meaning}. Is this correct?",
                    "inferred_unit": field.physical_unit,
                    "inferred_type": field.inferred_type,
                    "sample_values": field.sample_values,
                    "options": ["Confirm", "Correct", "Skip"],
                })

        for rel in relationships:
            if rel.confidence < 0.9:
                requests.append({
                    "type": "relationship_confirmation",
                    "question": f"I detected that {rel.field_a} and {rel.field_b} are {rel.relationship_type}. Does this make physical sense?",
                    "strength": rel.strength,
                    "options": ["Confirm", "Reject", "Needs Investigation"],
                })

        return requests
