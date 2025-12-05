"""
File Type Detection and Categorization Service.

This service provides intelligent file type detection using python-magic
and categorizes files for appropriate processing pipelines.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False


class FileCategory(str, Enum):
    """High-level file categories for processing pipeline selection."""
    CAD = "cad"                    # CAD/BIM files (DWG, DXF, IFC, etc.)
    DOCUMENT = "document"          # Text documents (PDF, DOCX, TXT, etc.)
    SPREADSHEET = "spreadsheet"    # Spreadsheets (XLSX, CSV, etc.)
    PRESENTATION = "presentation"  # Presentations (PPTX, etc.)
    IMAGE = "image"                # Images (PNG, JPG, TIFF, etc.)
    ARCHIVE = "archive"            # Archives (ZIP, TAR, etc.)
    EMAIL = "email"                # Email files (EML, MSG)
    UNKNOWN = "unknown"            # Unknown file types


class FileTypeInfo(BaseModel):
    """Information about a detected file type."""
    extension: str = Field(description="File extension (lowercase, with dot)")
    mime_type: str = Field(description="MIME type of the file")
    category: FileCategory = Field(description="High-level category")
    description: str = Field(description="Human-readable description")
    requires_ocr: bool = Field(default=False, description="Whether OCR is needed")
    requires_specialized_parser: bool = Field(default=False, description="Whether a specialized parser is needed")
    supported: bool = Field(default=True, description="Whether the file type is currently supported")


# Extension to category mapping
EXTENSION_CATEGORIES: Dict[str, Tuple[FileCategory, str, bool, bool]] = {
    # CAD/BIM files - require specialized parsers
    ".dwg": (FileCategory.CAD, "AutoCAD Drawing", False, True),
    ".dxf": (FileCategory.CAD, "AutoCAD DXF", False, True),
    ".ifc": (FileCategory.CAD, "Industry Foundation Classes (BIM)", False, True),
    ".rvt": (FileCategory.CAD, "Revit Project", False, True),
    ".rfa": (FileCategory.CAD, "Revit Family", False, True),
    ".nwd": (FileCategory.CAD, "Navisworks Document", False, True),
    ".skp": (FileCategory.CAD, "SketchUp Model", False, True),
    ".step": (FileCategory.CAD, "STEP CAD File", False, True),
    ".stp": (FileCategory.CAD, "STEP CAD File", False, True),
    ".iges": (FileCategory.CAD, "IGES CAD File", False, True),
    ".igs": (FileCategory.CAD, "IGES CAD File", False, True),
    
    # Documents
    ".pdf": (FileCategory.DOCUMENT, "PDF Document", False, False),  # OCR only if scanned
    ".docx": (FileCategory.DOCUMENT, "Microsoft Word Document", False, False),
    ".doc": (FileCategory.DOCUMENT, "Microsoft Word Document (Legacy)", False, False),
    ".txt": (FileCategory.DOCUMENT, "Plain Text", False, False),
    ".md": (FileCategory.DOCUMENT, "Markdown Document", False, False),
    ".rtf": (FileCategory.DOCUMENT, "Rich Text Format", False, False),
    ".odt": (FileCategory.DOCUMENT, "OpenDocument Text", False, False),
    ".html": (FileCategory.DOCUMENT, "HTML Document", False, False),
    ".htm": (FileCategory.DOCUMENT, "HTML Document", False, False),
    
    # Spreadsheets
    ".xlsx": (FileCategory.SPREADSHEET, "Microsoft Excel Spreadsheet", False, False),
    ".xls": (FileCategory.SPREADSHEET, "Microsoft Excel Spreadsheet (Legacy)", False, False),
    ".csv": (FileCategory.SPREADSHEET, "Comma-Separated Values", False, False),
    ".ods": (FileCategory.SPREADSHEET, "OpenDocument Spreadsheet", False, False),
    
    # Presentations
    ".pptx": (FileCategory.PRESENTATION, "Microsoft PowerPoint Presentation", False, False),
    ".ppt": (FileCategory.PRESENTATION, "Microsoft PowerPoint Presentation (Legacy)", False, False),
    ".odp": (FileCategory.PRESENTATION, "OpenDocument Presentation", False, False),
    
    # Images - may require OCR
    ".png": (FileCategory.IMAGE, "PNG Image", True, False),
    ".jpg": (FileCategory.IMAGE, "JPEG Image", True, False),
    ".jpeg": (FileCategory.IMAGE, "JPEG Image", True, False),
    ".tiff": (FileCategory.IMAGE, "TIFF Image", True, False),
    ".tif": (FileCategory.IMAGE, "TIFF Image", True, False),
    ".bmp": (FileCategory.IMAGE, "Bitmap Image", True, False),
    ".gif": (FileCategory.IMAGE, "GIF Image", True, False),
    ".webp": (FileCategory.IMAGE, "WebP Image", True, False),
    
    # Archives
    ".zip": (FileCategory.ARCHIVE, "ZIP Archive", False, True),
    ".tar": (FileCategory.ARCHIVE, "TAR Archive", False, True),
    ".gz": (FileCategory.ARCHIVE, "GZIP Archive", False, True),
    ".rar": (FileCategory.ARCHIVE, "RAR Archive", False, True),
    ".7z": (FileCategory.ARCHIVE, "7-Zip Archive", False, True),
    
    # Email
    ".eml": (FileCategory.EMAIL, "Email Message", False, True),
    ".msg": (FileCategory.EMAIL, "Outlook Email Message", False, True),
}


# MIME type to category fallback
MIME_CATEGORIES: Dict[str, FileCategory] = {
    "application/pdf": FileCategory.DOCUMENT,
    "application/msword": FileCategory.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileCategory.DOCUMENT,
    "application/vnd.ms-excel": FileCategory.SPREADSHEET,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileCategory.SPREADSHEET,
    "application/vnd.ms-powerpoint": FileCategory.PRESENTATION,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": FileCategory.PRESENTATION,
    "text/plain": FileCategory.DOCUMENT,
    "text/html": FileCategory.DOCUMENT,
    "text/csv": FileCategory.SPREADSHEET,
    "image/png": FileCategory.IMAGE,
    "image/jpeg": FileCategory.IMAGE,
    "image/tiff": FileCategory.IMAGE,
    "image/gif": FileCategory.IMAGE,
    "application/zip": FileCategory.ARCHIVE,
    "application/x-tar": FileCategory.ARCHIVE,
    "message/rfc822": FileCategory.EMAIL,
}


def detect_mime_type(file_path: str = None, file_content: bytes = None) -> str:
    """
    Detect the MIME type of a file.
    
    Args:
        file_path: Path to the file (optional if content provided)
        file_content: Raw file content (optional if path provided)
    
    Returns:
        MIME type string
    """
    if not MAGIC_AVAILABLE:
        # Fallback to extension-based detection
        if file_path:
            ext = Path(file_path).suffix.lower()
            return _extension_to_mime(ext)
        return "application/octet-stream"
    
    try:
        if file_content:
            mime = magic.from_buffer(file_content, mime=True)
        elif file_path:
            mime = magic.from_file(file_path, mime=True)
        else:
            return "application/octet-stream"
        return mime
    except Exception:
        return "application/octet-stream"


def _extension_to_mime(ext: str) -> str:
    """Fallback extension to MIME type mapping."""
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".ppt": "application/vnd.ms-powerpoint",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".zip": "application/zip",
    }
    return mime_map.get(ext, "application/octet-stream")


def detect_file_type(
    filename: str,
    file_content: bytes = None
) -> FileTypeInfo:
    """
    Detect and categorize a file type.
    
    Args:
        filename: Name of the file (used for extension detection)
        file_content: Optional raw file content for MIME detection
    
    Returns:
        FileTypeInfo with complete file type information
    """
    extension = Path(filename).suffix.lower()
    
    # Detect MIME type
    if file_content:
        mime_type = detect_mime_type(file_content=file_content)
    else:
        mime_type = _extension_to_mime(extension)
    
    # Look up extension in our mapping
    if extension in EXTENSION_CATEGORIES:
        category, description, requires_ocr, requires_specialized = EXTENSION_CATEGORIES[extension]
        
        # CAD files are not yet fully supported
        supported = category != FileCategory.CAD
        
        return FileTypeInfo(
            extension=extension,
            mime_type=mime_type,
            category=category,
            description=description,
            requires_ocr=requires_ocr,
            requires_specialized_parser=requires_specialized,
            supported=supported
        )
    
    # Fallback to MIME-based category
    category = MIME_CATEGORIES.get(mime_type, FileCategory.UNKNOWN)
    
    return FileTypeInfo(
        extension=extension,
        mime_type=mime_type,
        category=category,
        description=f"File type: {extension}",
        requires_ocr=False,
        requires_specialized_parser=False,
        supported=category != FileCategory.UNKNOWN
    )


def get_supported_extensions() -> List[str]:
    """Get list of all supported file extensions."""
    return [ext for ext, (cat, _, _, _) in EXTENSION_CATEGORIES.items() 
            if cat != FileCategory.CAD]  # CAD not yet supported


def get_extensions_by_category(category: FileCategory) -> List[str]:
    """Get all extensions for a specific category."""
    return [ext for ext, (cat, _, _, _) in EXTENSION_CATEGORIES.items() if cat == category]


def is_text_extractable(file_info: FileTypeInfo) -> bool:
    """Check if text can be directly extracted from the file."""
    return (
        file_info.category in [FileCategory.DOCUMENT, FileCategory.SPREADSHEET, FileCategory.PRESENTATION]
        and not file_info.requires_specialized_parser
        and file_info.supported
    )


def needs_content_analysis(file_info: FileTypeInfo) -> bool:
    """
    Check if the file needs content analysis for routing.
    
    CAD files can be routed by extension, but PDFs and DOCXs
    need content analysis to determine the appropriate department.
    """
    # These extensions have clear department mappings
    direct_routing_extensions = {".dwg", ".dxf", ".ifc", ".rvt", ".xlsx", ".pptx"}
    
    return file_info.extension not in direct_routing_extensions


# LangChain tool wrapper for file type detection
from langchain_core.tools import tool


@tool
def detect_file_type_tool(filename: str) -> str:
    """
    Detect the type and category of a file based on its filename.
    
    Args:
        filename: The name of the file to analyze
    
    Returns:
        A JSON string with file type information including category,
        MIME type, and whether specialized processing is needed.
    """
    file_info = detect_file_type(filename)
    return file_info.model_dump_json()
