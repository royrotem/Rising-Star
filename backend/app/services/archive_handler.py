"""
Archive Handler — ZIP and Compressed File Support

Handles extraction and processing of compressed archives:
- ZIP files with multiple data files
- Nested archives
- Zip bomb protection
- Progress reporting during extraction

Designed for archives up to 50GB extracted size.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..core.config import settings

logger = logging.getLogger("uaie.archive")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Safety limits (from config or defaults)
MAX_ARCHIVE_SIZE_MB = getattr(settings, 'MAX_ARCHIVE_SIZE_MB', 5000)  # 5GB
MAX_EXTRACTED_SIZE_GB = getattr(settings, 'MAX_EXTRACTED_SIZE_GB', 50)  # 50GB
MAX_FILES_PER_ARCHIVE = getattr(settings, 'MAX_FILES_PER_ARCHIVE', 10000)
MAX_NESTED_DEPTH = 3  # Maximum nesting level for archives within archives

# Supported data file extensions
SUPPORTED_DATA_EXTENSIONS = {
    '.csv', '.tsv', '.txt', '.dat', '.log',
    '.json', '.jsonl', '.ndjson',
    '.parquet', '.feather',
    '.xlsx', '.xls',
    '.xml', '.yaml', '.yml',
    '.can', '.bin',
}

# Archive extensions
ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2'}


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ExtractedFile:
    """Represents an extracted file from an archive."""
    original_name: str  # Name within the archive
    extracted_path: Path  # Temporary path on disk
    size_bytes: int
    extension: str
    is_data_file: bool
    parent_archive: Optional[str] = None  # If nested


@dataclass
class ExtractionResult:
    """Result of archive extraction."""
    success: bool
    archive_name: str
    total_files: int
    data_files: List[ExtractedFile] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_extracted_bytes: int = 0
    temp_dir: Optional[Path] = None


@dataclass
class ExtractionProgress:
    """Progress information during extraction."""
    phase: str  # 'scanning', 'extracting', 'processing'
    current_file: str
    files_processed: int
    total_files: int
    bytes_extracted: int
    total_bytes: int


# ═══════════════════════════════════════════════════════════════════════════
# Archive Handler
# ═══════════════════════════════════════════════════════════════════════════


class ArchiveHandler:
    """
    Handles extraction and validation of compressed archives.

    Features:
    - Zip bomb protection (ratio and size limits)
    - Nested archive extraction (up to MAX_NESTED_DEPTH)
    - Progress callbacks for UI updates
    - Automatic cleanup of temporary files
    """

    def __init__(
        self,
        max_archive_size_mb: int = MAX_ARCHIVE_SIZE_MB,
        max_extracted_size_gb: int = MAX_EXTRACTED_SIZE_GB,
        max_files: int = MAX_FILES_PER_ARCHIVE,
    ):
        self.max_archive_size_bytes = max_archive_size_mb * 1024 * 1024
        self.max_extracted_size_bytes = max_extracted_size_gb * 1024 * 1024 * 1024
        self.max_files = max_files
        self._temp_dirs: List[Path] = []

    async def extract_archive(
        self,
        archive_path: Path,
        progress_callback: Optional[Callable[[ExtractionProgress], None]] = None,
    ) -> ExtractionResult:
        """
        Extract an archive and return information about its contents.

        Args:
            archive_path: Path to the archive file
            progress_callback: Optional callback for progress updates

        Returns:
            ExtractionResult with extracted files and metadata
        """
        archive_name = archive_path.name
        logger.info("Starting extraction of archive: %s", archive_name)

        # Validate archive size
        archive_size = archive_path.stat().st_size
        if archive_size > self.max_archive_size_bytes:
            return ExtractionResult(
                success=False,
                archive_name=archive_name,
                total_files=0,
                errors=[f"Archive too large: {archive_size / (1024**3):.2f} GB exceeds limit of {MAX_ARCHIVE_SIZE_MB / 1024:.1f} GB"]
            )

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="uaie_archive_"))
        self._temp_dirs.append(temp_dir)

        try:
            # Determine archive type and extract
            ext = archive_path.suffix.lower()
            if ext == '.zip':
                result = await self._extract_zip(
                    archive_path, temp_dir, archive_name, progress_callback
                )
            else:
                result = ExtractionResult(
                    success=False,
                    archive_name=archive_name,
                    total_files=0,
                    errors=[f"Unsupported archive format: {ext}"]
                )

            result.temp_dir = temp_dir
            return result

        except Exception as e:
            logger.error("Archive extraction failed: %s", str(e))
            return ExtractionResult(
                success=False,
                archive_name=archive_name,
                total_files=0,
                errors=[f"Extraction failed: {str(e)}"],
                temp_dir=temp_dir
            )

    async def _extract_zip(
        self,
        zip_path: Path,
        dest_dir: Path,
        archive_name: str,
        progress_callback: Optional[Callable[[ExtractionProgress], None]] = None,
        depth: int = 0,
    ) -> ExtractionResult:
        """Extract a ZIP file with safety checks."""

        result = ExtractionResult(
            success=True,
            archive_name=archive_name,
            total_files=0,
        )

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Get file list and validate
                file_list = zf.namelist()
                total_files = len(file_list)

                if total_files > self.max_files:
                    result.success = False
                    result.errors.append(
                        f"Too many files in archive: {total_files} exceeds limit of {self.max_files}"
                    )
                    return result

                # Calculate total uncompressed size (zip bomb check)
                total_uncompressed = sum(info.file_size for info in zf.infolist())
                compressed_size = zip_path.stat().st_size

                # Zip bomb detection: compression ratio > 100:1 is suspicious
                if compressed_size > 0:
                    compression_ratio = total_uncompressed / compressed_size
                    if compression_ratio > 100:
                        logger.warning(
                            "Suspicious compression ratio %.1f:1 for %s",
                            compression_ratio, archive_name
                        )
                        # Still allow if total size is reasonable
                        if total_uncompressed > self.max_extracted_size_bytes:
                            result.success = False
                            result.errors.append(
                                f"Potential zip bomb detected: compression ratio {compression_ratio:.0f}:1"
                            )
                            return result

                if total_uncompressed > self.max_extracted_size_bytes:
                    result.success = False
                    result.errors.append(
                        f"Extracted size too large: {total_uncompressed / (1024**3):.2f} GB exceeds limit"
                    )
                    return result

                # Report scanning phase
                if progress_callback:
                    progress_callback(ExtractionProgress(
                        phase='scanning',
                        current_file='',
                        files_processed=0,
                        total_files=total_files,
                        bytes_extracted=0,
                        total_bytes=total_uncompressed,
                    ))

                # Extract files
                bytes_extracted = 0
                files_processed = 0

                for info in zf.infolist():
                    # Skip directories
                    if info.is_dir():
                        continue

                    filename = info.filename
                    file_ext = Path(filename).suffix.lower()

                    # Report progress
                    if progress_callback:
                        progress_callback(ExtractionProgress(
                            phase='extracting',
                            current_file=filename,
                            files_processed=files_processed,
                            total_files=total_files,
                            bytes_extracted=bytes_extracted,
                            total_bytes=total_uncompressed,
                        ))

                    # Security: prevent path traversal
                    safe_filename = self._sanitize_filename(filename)
                    if safe_filename is None:
                        result.skipped_files.append(f"{filename} (unsafe path)")
                        continue

                    dest_path = dest_dir / safe_filename

                    # Create parent directories
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Extract file
                    try:
                        with zf.open(info) as src, open(dest_path, 'wb') as dst:
                            shutil.copyfileobj(src, dst)

                        bytes_extracted += info.file_size
                        files_processed += 1
                        result.total_extracted_bytes += info.file_size

                        # Check if it's a data file we can process
                        is_data_file = file_ext in SUPPORTED_DATA_EXTENSIONS

                        # Check for nested archive
                        if file_ext in ARCHIVE_EXTENSIONS and depth < MAX_NESTED_DEPTH:
                            logger.info("Found nested archive: %s (depth %d)", filename, depth + 1)
                            nested_result = await self._extract_zip(
                                dest_path, dest_dir / f"_nested_{files_processed}",
                                filename, progress_callback, depth + 1
                            )
                            result.data_files.extend(nested_result.data_files)
                            result.skipped_files.extend(nested_result.skipped_files)
                            result.errors.extend(nested_result.errors)
                        elif is_data_file:
                            result.data_files.append(ExtractedFile(
                                original_name=filename,
                                extracted_path=dest_path,
                                size_bytes=info.file_size,
                                extension=file_ext,
                                is_data_file=True,
                                parent_archive=archive_name if depth > 0 else None,
                            ))
                        else:
                            result.skipped_files.append(f"{filename} (unsupported type)")

                    except Exception as e:
                        result.errors.append(f"Failed to extract {filename}: {str(e)}")

                result.total_files = files_processed

                # Final progress
                if progress_callback:
                    progress_callback(ExtractionProgress(
                        phase='complete',
                        current_file='',
                        files_processed=files_processed,
                        total_files=total_files,
                        bytes_extracted=bytes_extracted,
                        total_bytes=total_uncompressed,
                    ))

        except zipfile.BadZipFile as e:
            result.success = False
            result.errors.append(f"Invalid ZIP file: {str(e)}")
        except Exception as e:
            result.success = False
            result.errors.append(f"Extraction error: {str(e)}")

        return result

    def _sanitize_filename(self, filename: str) -> Optional[str]:
        """
        Sanitize filename to prevent path traversal attacks.

        Returns None if the filename is unsafe.
        """
        # Normalize path separators
        normalized = filename.replace('\\', '/')

        # Check for path traversal
        if '..' in normalized or normalized.startswith('/'):
            logger.warning("Blocked unsafe path: %s", filename)
            return None

        # Remove leading slashes and normalize
        safe = normalized.lstrip('/')

        # Additional checks
        if not safe or safe.startswith('.'):
            return None

        return safe

    def cleanup(self) -> None:
        """Clean up all temporary directories created during extraction."""
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.debug("Cleaned up temp dir: %s", temp_dir)
            except Exception as e:
                logger.warning("Failed to cleanup %s: %s", temp_dir, e)
        self._temp_dirs.clear()

    def cleanup_result(self, result: ExtractionResult) -> None:
        """Clean up temporary files from a specific extraction result."""
        if result.temp_dir and result.temp_dir.exists():
            try:
                shutil.rmtree(result.temp_dir)
                logger.debug("Cleaned up extraction temp dir: %s", result.temp_dir)
            except Exception as e:
                logger.warning("Failed to cleanup %s: %s", result.temp_dir, e)

    def __del__(self):
        """Cleanup on garbage collection."""
        self.cleanup()


# ═══════════════════════════════════════════════════════════════════════════
# Archive Ingestion Service
# ═══════════════════════════════════════════════════════════════════════════


class ArchiveIngestionService:
    """
    High-level service for ingesting data from archives.

    Combines archive extraction with the streaming ingestion service
    to process multiple files from a single archive upload.
    """

    def __init__(self):
        self.handler = ArchiveHandler()

    async def ingest_archive(
        self,
        archive_content,
        archive_filename: str,
        system_id: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest all data files from an archive.

        Args:
            archive_content: File-like object containing the archive
            archive_filename: Original filename of the archive
            system_id: System to associate the data with
            progress_callback: Optional callback for progress updates

        Returns:
            Summary of ingested files and any errors
        """
        from .streaming_ingestion import StreamingIngestionService
        from .statistical_profiler import IncrementalFieldProfiler

        logger.info("Starting archive ingestion: %s for system %s", archive_filename, system_id)

        # Save archive to temp file
        temp_archive = Path(tempfile.mktemp(suffix=Path(archive_filename).suffix))
        try:
            with open(temp_archive, 'wb') as f:
                shutil.copyfileobj(archive_content, f)

            # Extract archive
            def extraction_progress(p: ExtractionProgress):
                if progress_callback:
                    progress_callback({
                        "phase": "extraction",
                        "status": p.phase,
                        "current_file": p.current_file,
                        "files_processed": p.files_processed,
                        "total_files": p.total_files,
                        "percent": int((p.bytes_extracted / max(p.total_bytes, 1)) * 50),  # 0-50%
                    })

            extraction_result = await self.handler.extract_archive(
                temp_archive, extraction_progress
            )

            if not extraction_result.success:
                return {
                    "status": "error",
                    "archive_name": archive_filename,
                    "errors": extraction_result.errors,
                }

            if not extraction_result.data_files:
                return {
                    "status": "error",
                    "archive_name": archive_filename,
                    "errors": ["No supported data files found in archive"],
                    "skipped_files": extraction_result.skipped_files,
                }

            # Process each data file
            streaming_service = StreamingIngestionService()
            profiler = IncrementalFieldProfiler()

            ingested_files = []
            all_records = []
            total_record_count = 0
            file_errors = []

            for idx, data_file in enumerate(extraction_result.data_files):
                if progress_callback:
                    progress_callback({
                        "phase": "ingestion",
                        "status": "processing",
                        "current_file": data_file.original_name,
                        "files_processed": idx,
                        "total_files": len(extraction_result.data_files),
                        "percent": 50 + int((idx / len(extraction_result.data_files)) * 45),  # 50-95%
                    })

                try:
                    with open(data_file.extracted_path, 'rb') as f:
                        result = await streaming_service.ingest_large_file(
                            file_content=f,
                            filename=data_file.original_name,
                            system_id=system_id,
                            source_id=f"archive_{idx}",
                            progress_callback=None,  # Skip per-file progress
                        )

                        ingested_files.append({
                            "filename": data_file.original_name,
                            "record_count": result.get("record_count", 0),
                            "field_count": len(result.get("discovered_fields", [])),
                            "size_bytes": data_file.size_bytes,
                        })

                        total_record_count += result.get("record_count", 0)

                        # Add sample records
                        sample = result.get("sample_records", [])[:100]
                        all_records.extend(sample)

                except Exception as e:
                    logger.error("Failed to ingest %s: %s", data_file.original_name, e)
                    file_errors.append({
                        "filename": data_file.original_name,
                        "error": str(e),
                    })

            # Build combined field profiles
            if progress_callback:
                progress_callback({
                    "phase": "profiling",
                    "status": "building_profiles",
                    "percent": 95,
                })

            # Final result
            result = {
                "status": "success",
                "archive_name": archive_filename,
                "total_files_extracted": extraction_result.total_files,
                "data_files_processed": len(ingested_files),
                "total_record_count": total_record_count,
                "total_extracted_bytes": extraction_result.total_extracted_bytes,
                "ingested_files": ingested_files,
                "skipped_files": extraction_result.skipped_files,
                "file_errors": file_errors,
                "sample_records": all_records[:100],
            }

            if progress_callback:
                progress_callback({
                    "phase": "complete",
                    "status": "success",
                    "percent": 100,
                })

            return result

        finally:
            # Cleanup
            if temp_archive.exists():
                temp_archive.unlink()
            self.handler.cleanup()


# Global instance
archive_handler = ArchiveHandler()
archive_ingestion_service = ArchiveIngestionService()
