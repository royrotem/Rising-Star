"""
Zero-Knowledge Ingestion & Adaptive Discovery Service

This service handles the "blind" ingestion of raw data and autonomous
schema discovery using AI agents.
"""

import asyncio
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Any, BinaryIO, Dict, List, Optional

import numpy as np
import pandas as pd


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
            'tsv': self._parse_tsv,
            'txt': self._parse_text,
            'dat': self._parse_text,
            'log': self._parse_log,
            'json': self._parse_json,
            'jsonl': self._parse_jsonl,
            'ndjson': self._parse_jsonl,
            'xml': self._parse_xml,
            'yaml': self._parse_yaml,
            'yml': self._parse_yaml,
            'parquet': self._parse_parquet,
            'feather': self._parse_feather,
            'xlsx': self._parse_excel,
            'xls': self._parse_excel,
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
        discovered_fields, metadata_info = await self._discover_schema(records, raw_schema)

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
            "metadata_info": metadata_info,  # Include extracted metadata
        }

    async def _parse_csv(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse CSV file and extract records.

        Handles encoding detection (UTF-8/BOM/latin-1), delimiter sniffing
        (comma/semicolon/tab/pipe), and empty-row cleanup.
        """
        import csv as _csv
        import io as _io

        raw = file_content.read()

        # ── Detect encoding ──────────────────────────────────────
        for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                content = raw.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            content = raw.decode("utf-8", errors="replace")

        # ── Detect delimiter ─────────────────────────────────────
        sample = content[:8192]
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except _csv.Error:
            sep = ","

        df = pd.read_csv(_io.StringIO(content), sep=sep)

        # Drop fully empty rows / columns
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

        if df.empty:
            return [], {}

        records = df.to_dict("records")
        schema = {col: str(df[col].dtype) for col in df.columns}
        return records, schema

    async def _parse_tsv(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse TSV (tab-separated values) file."""
        import io as _io

        raw = file_content.read()
        for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                content = raw.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            content = raw.decode("utf-8", errors="replace")

        df = pd.read_csv(_io.StringIO(content), sep="\t")
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        if df.empty:
            return [], {}
        return df.to_dict("records"), {col: str(df[col].dtype) for col in df.columns}

    async def _parse_text(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse plain text / .dat files — tries CSV auto-detection, falls back to line-based."""
        import csv as _csv
        import io as _io

        raw = file_content.read()
        for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                content = raw.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            content = raw.decode("utf-8", errors="replace")

        # Try to detect if this is tabular data
        sample = content[:8192]
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            has_header = _csv.Sniffer().has_header(sample)
            sep = dialect.delimiter
        except _csv.Error:
            sep = None
            has_header = False

        if sep and has_header:
            # Looks like tabular data — parse as CSV
            df = pd.read_csv(_io.StringIO(content), sep=sep)
            df.dropna(how="all", inplace=True)
            df.dropna(axis=1, how="all", inplace=True)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
            if not df.empty:
                return df.to_dict("records"), {col: str(df[col].dtype) for col in df.columns}

        # Fall back to line-based parsing
        lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
        if not lines:
            return [], {}

        records = [{"line_number": i + 1, "content": line} for i, line in enumerate(lines)]
        return records, {"line_number": "integer", "content": "string"}

    async def _parse_log(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse log files — extracts timestamp, level, and message from common log formats."""
        raw = file_content.read()
        for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                content = raw.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            content = raw.decode("utf-8", errors="replace")

        lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
        if not lines:
            return [], {}

        # Common log patterns: "2024-01-01 12:00:00 INFO message" or "[timestamp] [LEVEL] message"
        log_pattern = re.compile(
            r'^[\[(\s]*'
            r'(\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)'
            r'[\])\s]*'
            r'[\[(\s]*'
            r'(DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL|TRACE)?'
            r'[\])\s]*'
            r'(.*)',
            re.IGNORECASE
        )

        records = []
        for i, line in enumerate(lines):
            match = log_pattern.match(line)
            if match:
                records.append({
                    "line_number": i + 1,
                    "timestamp": match.group(1),
                    "level": (match.group(2) or "UNKNOWN").upper(),
                    "message": match.group(3).strip(),
                })
            else:
                records.append({
                    "line_number": i + 1,
                    "timestamp": None,
                    "level": "UNKNOWN",
                    "message": line,
                })

        schema = {"line_number": "integer", "timestamp": "string", "level": "string", "message": "string"}
        return records, schema

    async def _parse_xml(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse XML files into flat records."""
        import xml.etree.ElementTree as ET
        import io as _io

        raw = file_content.read()
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            # Try with encoding declaration stripped
            content = raw.decode("utf-8", errors="replace")
            content = re.sub(r'<\?xml[^?]*\?>', '', content)
            root = ET.fromstring(content)

        records = []

        # If root has repeating child elements, treat each as a record
        children = list(root)
        if children:
            tag_counts: Dict[str, int] = {}
            for child in children:
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Find the most common child tag — those are likely the data rows
            most_common_tag = max(tag_counts, key=tag_counts.get)

            for child in children:
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag == most_common_tag:
                    record = self._xml_element_to_dict(child)
                    records.append(record)

        if not records:
            # Single-element XML — flatten the whole thing
            records = [self._xml_element_to_dict(root)]

        schema = self._infer_json_schema(records[0]) if records else {}
        return records, schema

    def _xml_element_to_dict(self, element) -> Dict:
        """Convert an XML element to a flat dictionary."""
        result = {}
        # Include attributes
        for attr, val in element.attrib.items():
            result[attr] = val
        # Include child elements
        for child in element:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if len(child) > 0:
                # Nested element — flatten with prefix
                nested = self._xml_element_to_dict(child)
                for k, v in nested.items():
                    result[f"{tag}_{k}"] = v
            else:
                result[tag] = child.text.strip() if child.text else None
        # If element has text directly and children
        if element.text and element.text.strip() and not list(element):
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            result[tag] = element.text.strip()
        return result

    async def _parse_yaml(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse YAML/YML files."""
        try:
            import yaml
        except ImportError:
            raise ValueError("YAML support requires PyYAML. Install it with: pip install pyyaml")

        content = file_content.read().decode("utf-8", errors="replace")
        data = yaml.safe_load(content)

        if isinstance(data, list):
            records = [self._flatten_dict(r) if isinstance(r, dict) else {"value": r} for r in data]
        elif isinstance(data, dict):
            # If it looks like a list of records under a key, extract that
            for key, val in data.items():
                if isinstance(val, list) and len(val) > 1 and isinstance(val[0], dict):
                    records = [self._flatten_dict(r) for r in val]
                    break
            else:
                records = [self._flatten_dict(data)]
        else:
            records = [{"value": data}]

        schema = self._infer_json_schema(records[0]) if records else {}
        return records, schema

    async def _parse_feather(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse Apache Arrow Feather files."""
        import io as _io
        raw = file_content.read()
        df = pd.read_feather(_io.BytesIO(raw))
        records = df.to_dict('records')
        schema = {col: str(df[col].dtype) for col in df.columns}
        return records, schema

    async def _parse_excel(self, file_content: BinaryIO) -> tuple[List[Dict], Dict]:
        """Parse Excel (.xlsx / .xls) file and extract records.

        Handles common edge cases:
        - Multi-sheet workbooks: reads the first sheet by default, then
          appends rows from additional sheets that share the same columns.
        - Empty leading rows: pandas skiprows is left at default (None) so
          that read_excel auto-detects the header row.
        - Merged cells: openpyxl forward-fills merged regions automatically.
        """
        import io as _io

        raw = file_content.read()
        buf = _io.BytesIO(raw)

        # Discover sheet names
        xl = pd.ExcelFile(buf, engine="openpyxl")
        sheet_names = xl.sheet_names

        # Read the first sheet
        df = xl.parse(sheet_names[0])

        # Drop fully empty rows / columns
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        # If multiple sheets share the same column set, concatenate them
        if len(sheet_names) > 1:
            base_cols = set(df.columns)
            for sheet in sheet_names[1:]:
                try:
                    extra = xl.parse(sheet)
                    extra.dropna(how="all", inplace=True)
                    extra.dropna(axis=1, how="all", inplace=True)
                    if set(extra.columns) == base_cols:
                        df = pd.concat([df, extra], ignore_index=True)
                except Exception:
                    continue

        # Normalize column names: strip whitespace
        df.columns = [str(c).strip() for c in df.columns]

        # Drop unnamed columns that pandas generates for blank Excel columns
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

        if df.empty:
            return [], {}

        records = df.to_dict("records")
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
            except (struct.error, IndexError):
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
    ) -> tuple[List[DiscoveredField], Dict[str, Any]]:
        """
        Autonomously discover field meanings and types.
        This is where the "AI Agent" learns the system's DNA.

        Returns:
            Tuple of (discovered_fields, metadata_info)
        """
        if not records:
            return [], {}

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

        # First pass: detect metadata/description fields that explain the dataset
        metadata_info = self._detect_and_extract_metadata(df)
        extracted_descriptions = metadata_info.get('field_descriptions', {})
        dataset_description = metadata_info.get('dataset_description', '')
        dataset_purpose = metadata_info.get('dataset_purpose', '')

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

            # Infer semantic meaning - first check if metadata provided a description
            if column in extracted_descriptions:
                field.inferred_meaning = extracted_descriptions[column]
            else:
                field.inferred_meaning = self._infer_meaning(column, df[column])

            # Calculate statistics for numeric fields
            if field.inferred_type == 'numeric':
                field.statistics = {
                    'min': self._safe_float(df[column].min()) if not df[column].isna().all() else None,
                    'max': self._safe_float(df[column].max()) if not df[column].isna().all() else None,
                    'mean': self._safe_float(df[column].mean()) if not df[column].isna().all() else None,
                    'std': self._safe_float(df[column].std()) if not df[column].isna().all() else None,
                    'null_percentage': self._safe_float(df[column].isna().sum() / len(df) * 100),
                }

            # Set confidence based on how certain we are about the inference
            field.confidence = self._calculate_confidence(field)

            discovered.append(field)

        return discovered, metadata_info

    def _detect_and_extract_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Deep scan all content to find metadata/description fields.
        Scans ALL values in ALL columns to find dataset documentation.
        """

        result = {
            'field_descriptions': {},
            'dataset_description': '',
            'dataset_purpose': '',
            'metadata_field': None,
            'context_texts': []  # All found contextual texts
        }

        metadata_indicators = [
            'dataset', 'this data', 'provides', 'contains', 'includes',
            'designed to', 'used for', 'features', 'field', 'column',
            'key features', 'sensor', 'measurement', 'monitoring'
        ]

        found_descriptions = []

        # Deep scan: check ALL values in ALL columns for metadata content
        for column in df.columns:
            try:
                # Get all unique non-null values (limit to prevent memory issues)
                unique_values = df[column].dropna().unique()[:100]

                for value in unique_values:
                    text = str(value)

                    # Skip short values or numeric-looking values
                    if len(text) < 100:
                        continue

                    text_lower = text.lower()

                    # Check if this looks like documentation/metadata
                    indicator_count = sum(1 for ind in metadata_indicators if ind in text_lower)

                    # If it has multiple indicators and is long enough, it's likely metadata
                    if indicator_count >= 2 and len(text) > 150:
                        found_descriptions.append({
                            'column': column,
                            'text': text,
                            'indicator_count': indicator_count,
                            'length': len(text)
                        })

            except Exception:
                continue

        # Sort by indicator count and length to get the most relevant description
        found_descriptions.sort(key=lambda x: (x['indicator_count'], x['length']), reverse=True)

        if found_descriptions:
            best_match = found_descriptions[0]
            result['metadata_field'] = best_match['column']
            result['dataset_description'] = best_match['text']
            result['context_texts'] = [d['text'] for d in found_descriptions[:3]]

            # Extract field descriptions from ALL found metadata texts
            all_columns = df.columns.tolist()
            combined_text = ' '.join(d['text'] for d in found_descriptions)
            result['field_descriptions'] = self._extract_field_descriptions_deep(combined_text, all_columns)

            # Extract purpose
            result['dataset_purpose'] = self._extract_purpose(combined_text)

        return result

    def _extract_field_descriptions_deep(self, text: str, columns: List[str]) -> Dict[str, str]:
        """
        Deep extraction of field descriptions from metadata text.
        Uses multiple strategies to find field meanings.
        """
        descriptions = {}

        # Normalize text for matching
        text_normalized = re.sub(r'\s+', ' ', text)

        for column in columns:
            if len(column) < 2:
                continue

            # Strategy 1: Direct "FieldName: description" pattern
            patterns = [
                # "Timestamp: Precise time of each sensor reading"
                rf'\b{re.escape(column)}\s*[:]\s*([^.!?\n⭐]+[.!?]?)',
                # "Timestamp - Precise time..."
                rf'\b{re.escape(column)}\s*[-–—]\s*([^.!?\n⭐]+[.!?]?)',
                # "The Timestamp field represents..."
                rf'(?:the\s+)?{re.escape(column)}(?:\s+field)?\s+(?:is|represents?|measures?|captures?|records?|indicates?|shows?|provides?)\s+([^.!?\n]+[.!?]?)',
            ]

            for pattern in patterns:
                match = re.search(pattern, text_normalized, re.IGNORECASE)
                if match:
                    desc = match.group(1).strip()
                    desc = re.sub(r'\s+', ' ', desc)
                    # Clean trailing punctuation duplicates
                    desc = re.sub(r'[.!?]+$', '.', desc)
                    if 10 < len(desc) < 300:
                        descriptions[column] = desc
                        break

            # Strategy 2: Look for column name followed by explanatory text
            if column not in descriptions:
                # Match "Column (explanation)" or "Column = explanation"
                alt_patterns = [
                    rf'\b{re.escape(column)}\s*\(([^)]+)\)',
                    rf'\b{re.escape(column)}\s*=\s*([^,.\n]+)',
                ]
                for pattern in alt_patterns:
                    match = re.search(pattern, text_normalized, re.IGNORECASE)
                    if match:
                        desc = match.group(1).strip()
                        if 5 < len(desc) < 200:
                            descriptions[column] = desc
                            break

        # Strategy 3: Look for numbered or bulleted lists
        list_patterns = [
            r'[•\-\*]\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:–-]\s*([^•\-\*\n]+)',
            r'\d+[.)]\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:–-]\s*([^0-9\n]+)',
        ]

        for pattern in list_patterns:
            for match in re.finditer(pattern, text):
                field_name = match.group(1).strip()
                desc = match.group(2).strip()

                # Find matching column (case-insensitive, underscore-insensitive)
                for column in columns:
                    col_normalized = column.lower().replace('_', '')
                    field_normalized = field_name.lower().replace('_', '')
                    if col_normalized == field_normalized and column not in descriptions:
                        if 5 < len(desc) < 300:
                            descriptions[column] = desc
                        break

        return descriptions

    def _extract_purpose(self, text: str) -> str:
        """Extract the purpose/use case from dataset description."""

        purpose_patterns = [
            r'(?:designed to|used for|intended for|suitable for|supports?)\s+([^.]+\.)',
            r'(?:use cases?|applications?|purposes?)[:]\s*([^.]+\.)',
            r'(?:predictive maintenance|fault detection|anomaly detection|monitoring)[^.]*\.',
        ]

        for pattern in purpose_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        return ''

    def _safe_float(self, value) -> Optional[float]:
        """Convert a value to float, returning None for NaN/inf values."""
        try:
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

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

    # ============ File Classification ============

    def classify_file_role(
        self,
        filename: str,
        records: List[Dict],
        discovered_fields: List[DiscoveredField],
        other_files_fields: List[str],
    ) -> str:
        """
        Classify a file's role: 'data' (measurement/telemetry records) or
        'description' (documentation that explains fields in other files).

        A file is classified as 'description' when:
        - It has very few columns (typically line_number + content from text parser)
        - Its content references field names found in other uploaded files
        - It contains description-like patterns (colon-separated field:desc, bullets, etc.)
        """
        if not records:
            return "data"

        columns = list(records[0].keys()) if records else []

        # Text files parsed as line_number + content are strong candidates
        is_text_parsed = set(columns) == {"line_number", "content"}

        if not is_text_parsed:
            # Files with many numeric columns are data files
            numeric_count = sum(1 for f in discovered_fields if f.inferred_type == "numeric")
            if numeric_count >= 3 or len(columns) >= 5:
                return "data"

        # Check if content references field names from other files
        if is_text_parsed and other_files_fields:
            combined_text = " ".join(
                str(r.get("content", "")) for r in records
            ).lower()

            # Count how many other-file field names appear in this file's text
            field_mentions = sum(
                1 for f in other_files_fields
                if len(f) > 2 and f.lower() in combined_text
            )

            # If this file mentions multiple field names from other files, it's a description
            if field_mentions >= 2:
                return "description"

            # Check for description patterns: "FieldName: description" or "FieldName - description"
            desc_pattern_count = 0
            for r in records:
                line = str(r.get("content", ""))
                if re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*[:–\-]\s+\S', line):
                    desc_pattern_count += 1

            if desc_pattern_count >= 3:
                return "description"

        # Small text files with few records that contain long text are likely descriptions
        if is_text_parsed and len(records) < 50:
            avg_content_len = sum(len(str(r.get("content", ""))) for r in records) / max(len(records), 1)
            if avg_content_len > 30:
                # Content is descriptive text, not short data entries
                desc_indicators = [
                    "field", "column", "sensor", "measurement", "data",
                    "represents", "contains", "monitoring", "parameter",
                    "description", "meaning", "purpose", "unit",
                ]
                combined = " ".join(str(r.get("content", "")) for r in records).lower()
                indicator_hits = sum(1 for ind in desc_indicators if ind in combined)
                if indicator_hits >= 3:
                    return "description"

        return "data"

    def extract_descriptions_from_file(
        self,
        records: List[Dict],
        data_field_names: List[str],
    ) -> Dict[str, str]:
        """
        Extract field name → description mappings from a description/documentation file.

        Parses common patterns:
        - "FieldName: description text"
        - "FieldName - description text"
        - "• FieldName: description"
        - Numbered lists: "1. FieldName: description"
        """
        descriptions: Dict[str, str] = {}

        # Combine all content lines
        lines = [str(r.get("content", "")) for r in records if r.get("content")]
        combined_text = "\n".join(lines)

        # Normalize field names for fuzzy matching
        field_name_map = {}
        for f in data_field_names:
            field_name_map[f.lower()] = f
            field_name_map[f.lower().replace("_", "")] = f
            field_name_map[f.lower().replace("_", " ")] = f

        # Strategy 1: "FieldName: description" or "FieldName - description"
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove bullet/number prefixes
            clean_line = re.sub(r'^[\s•\-\*\d.)\]]+\s*', '', line)

            # Match "Name: description" or "Name - description"
            match = re.match(
                r'^([A-Za-z_][A-Za-z0-9_\s]*?)\s*[:–\-]\s+(.+)$',
                clean_line,
            )
            if match:
                field_part = match.group(1).strip()
                desc_part = match.group(2).strip()

                if len(desc_part) < 5:
                    continue

                # Try to match to a known data field
                field_lower = field_part.lower()
                field_no_space = field_lower.replace(" ", "").replace("_", "")

                matched_field = None
                if field_lower in field_name_map:
                    matched_field = field_name_map[field_lower]
                elif field_no_space in {k.replace(" ", "").replace("_", ""): k for k in field_name_map}:
                    # Fuzzy match ignoring underscores/spaces
                    for key, original in field_name_map.items():
                        if key.replace(" ", "").replace("_", "") == field_no_space:
                            matched_field = field_name_map[key]
                            break

                if matched_field:
                    descriptions[matched_field] = desc_part
                else:
                    # Store with the name as-is (might match after normalization)
                    descriptions[field_part] = desc_part

        # Strategy 2: Use the deep extraction from _extract_field_descriptions_deep
        if len(descriptions) < len(data_field_names) // 2:
            deep_desc = self._extract_field_descriptions_deep(
                combined_text, data_field_names
            )
            for field, desc in deep_desc.items():
                if field not in descriptions:
                    descriptions[field] = desc

        return descriptions

    def get_description_file_context(self, records: List[Dict]) -> str:
        """Get the full text content of a description file for context enrichment."""
        return "\n".join(str(r.get("content", "")) for r in records if r.get("content"))

    # ============ Field Relevance Scoring ============

    def classify_field_relevance(self, field: Dict) -> str:
        """
        Classify a field's role in the dataset:
        - 'content': Core measurement/sensor data useful for analysis
        - 'temporal': Timestamp or time-related fields
        - 'identifier': IDs, serial numbers, names — useful for grouping but not analysis
        - 'auxiliary': Metadata, line numbers, or low-value fields
        """
        name = field.get("name", "").lower()
        ftype = field.get("inferred_type", "")
        meaning = field.get("inferred_meaning", "").lower()

        # Temporal fields
        if ftype == "timestamp" or any(
            p in name for p in ["time", "timestamp", "date", "epoch", "datetime"]
        ):
            return "temporal"

        # Identifier fields
        id_patterns = [
            "id", "_id", "uuid", "serial", "name", "label", "tag",
            "index", "row", "record_num", "line_number",
        ]
        if any(name == p or name.endswith(f"_{p}") or name.startswith(f"{p}_") for p in id_patterns):
            return "identifier"
        if name in ("line_number", "content"):
            return "auxiliary"

        # Auxiliary: fields with very low cardinality relative to records (constant values)
        stats = field.get("statistics", {})
        if stats:
            std = stats.get("std")
            if std is not None and std == 0:
                return "auxiliary"

        # Categorical fields with very few unique values might be identifiers
        if ftype == "categorical":
            unique = field.get("unique_count", 0)
            if unique <= 2:
                return "auxiliary"

        # String fields that aren't recognized as anything specific
        if ftype == "string" and "unknown" in meaning:
            return "auxiliary"

        # Everything else is content (measurement/sensor data)
        return "content"
