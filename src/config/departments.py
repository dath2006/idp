"""
Department Configuration for Role-Based Document Routing.

This module defines the organizational structure, departments, and routing rules
for intelligent document dissemination in infrastructure operations.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field


class Department(str, Enum):
    """Available departments in the organization."""
    ENGINEERING = "engineering"
    FINANCE = "finance"
    PROCUREMENT = "procurement"
    HR = "hr"
    OPERATIONS = "operations"
    SAFETY = "safety"
    COMPLIANCE = "compliance"
    LEGAL = "legal"
    MANAGEMENT = "management"


class DepartmentConfig(BaseModel):
    """Configuration for a department."""
    id: Department
    name: str
    description: str
    email: str = Field(description="Department email for notifications")
    keywords: List[str] = Field(default_factory=list, description="Keywords that indicate relevance to this department")
    file_extensions: List[str] = Field(default_factory=list, description="File extensions primarily handled by this department")
    priority: int = Field(default=5, description="Priority level for routing (1=highest, 10=lowest)")


# Department definitions with routing metadata
DEPARTMENTS: Dict[Department, DepartmentConfig] = {
    Department.ENGINEERING: DepartmentConfig(
        id=Department.ENGINEERING,
        name="Engineering",
        description="Handles technical documents, CAD files, BIM models, and engineering specifications",
        email="engineering@company.com",
        keywords=[
            "engineering", "technical", "design", "CAD", "BIM", "specification",
            "drawing", "blueprint", "schematic", "mechanical", "electrical",
            "structural", "civil", "architecture", "construction", "RFI",
            "submittal", "shop drawing", "as-built"
        ],
        file_extensions=[".dwg", ".dxf", ".ifc", ".rvt", ".rfa", ".nwd", ".skp", ".step", ".stp", ".iges"],
        priority=2
    ),
    
    Department.FINANCE: DepartmentConfig(
        id=Department.FINANCE,
        name="Finance",
        description="Handles invoices, budgets, financial reports, and payment documents",
        email="finance@company.com",
        keywords=[
            "invoice", "payment", "budget", "cost", "expense", "financial",
            "billing", "receipt", "purchase order", "PO", "quote", "estimate",
            "pricing", "revenue", "profit", "loss", "tax", "audit"
        ],
        file_extensions=[".xlsx", ".xls", ".csv"],
        priority=2
    ),
    
    Department.PROCUREMENT: DepartmentConfig(
        id=Department.PROCUREMENT,
        name="Procurement",
        description="Handles RFPs, vendor documents, contracts, and supply chain materials",
        email="procurement@company.com",
        keywords=[
            "procurement", "vendor", "supplier", "RFP", "RFQ", "bid",
            "contract", "agreement", "purchase", "supply", "material",
            "inventory", "order", "delivery", "logistics"
        ],
        file_extensions=[],
        priority=3
    ),
    
    Department.HR: DepartmentConfig(
        id=Department.HR,
        name="Human Resources",
        description="Handles employee documents, policies, training materials, and personnel records",
        email="hr@company.com",
        keywords=[
            "employee", "HR", "human resources", "personnel", "hiring",
            "recruitment", "training", "policy", "handbook", "benefits",
            "payroll", "leave", "performance", "onboarding", "termination"
        ],
        file_extensions=[],
        priority=4
    ),
    
    Department.OPERATIONS: DepartmentConfig(
        id=Department.OPERATIONS,
        name="Operations",
        description="Handles operational procedures, maintenance logs, and field reports",
        email="operations@company.com",
        keywords=[
            "operations", "maintenance", "field", "site", "inspection",
            "schedule", "work order", "task", "procedure", "SOP",
            "standard operating", "daily report", "progress", "status"
        ],
        file_extensions=[],
        priority=3
    ),
    
    Department.SAFETY: DepartmentConfig(
        id=Department.SAFETY,
        name="Safety",
        description="Handles safety documents, incident reports, and compliance materials",
        email="safety@company.com",
        keywords=[
            "safety", "incident", "accident", "hazard", "risk", "OSHA",
            "PPE", "emergency", "evacuation", "first aid", "injury",
            "near miss", "JSA", "job safety analysis", "toolbox talk"
        ],
        file_extensions=[],
        priority=1  # High priority for safety-related documents
    ),
    
    Department.COMPLIANCE: DepartmentConfig(
        id=Department.COMPLIANCE,
        name="Compliance",
        description="Handles regulatory documents, permits, certifications, and audit materials",
        email="compliance@company.com",
        keywords=[
            "compliance", "regulation", "permit", "license", "certification",
            "audit", "inspection", "standard", "code", "requirement",
            "EPA", "environmental", "regulatory", "violation", "corrective"
        ],
        file_extensions=[],
        priority=1  # High priority for compliance documents
    ),
    
    Department.LEGAL: DepartmentConfig(
        id=Department.LEGAL,
        name="Legal",
        description="Handles legal documents, contracts, and dispute-related materials",
        email="legal@company.com",
        keywords=[
            "legal", "law", "attorney", "contract", "agreement", "lawsuit",
            "litigation", "dispute", "claim", "liability", "indemnity",
            "terms", "conditions", "NDA", "confidential"
        ],
        file_extensions=[],
        priority=2
    ),
    
    Department.MANAGEMENT: DepartmentConfig(
        id=Department.MANAGEMENT,
        name="Management",
        description="Handles executive reports, strategic documents, and high-level summaries",
        email="management@company.com",
        keywords=[
            "executive", "management", "strategy", "summary", "report",
            "dashboard", "KPI", "metrics", "performance", "quarterly",
            "annual", "board", "stakeholder", "decision"
        ],
        file_extensions=[".pptx", ".ppt"],
        priority=5
    ),
}


# File type to department routing rules
FILE_TYPE_ROUTING_RULES: Dict[str, List[Department]] = {
    # CAD/BIM files -> Engineering
    ".dwg": [Department.ENGINEERING],
    ".dxf": [Department.ENGINEERING],
    ".ifc": [Department.ENGINEERING],
    ".rvt": [Department.ENGINEERING],
    ".rfa": [Department.ENGINEERING],
    ".nwd": [Department.ENGINEERING],
    ".skp": [Department.ENGINEERING],
    ".step": [Department.ENGINEERING],
    ".stp": [Department.ENGINEERING],
    ".iges": [Department.ENGINEERING],
    
    # Spreadsheets -> Finance (primary), Operations (secondary)
    ".xlsx": [Department.FINANCE, Department.OPERATIONS],
    ".xls": [Department.FINANCE, Department.OPERATIONS],
    ".csv": [Department.FINANCE, Department.OPERATIONS],
    
    # Presentations -> Management
    ".pptx": [Department.MANAGEMENT],
    ".ppt": [Department.MANAGEMENT],
    
    # PDFs and Word docs require content analysis (no default routing)
    ".pdf": [],  # Content-based routing required
    ".docx": [],  # Content-based routing required
    ".doc": [],  # Content-based routing required
    ".txt": [],  # Content-based routing required
    
    # Images may be for any department
    ".png": [],
    ".jpg": [],
    ".jpeg": [],
    ".tiff": [],
    ".bmp": [],
}


def get_department_by_id(dept_id: Department) -> Optional[DepartmentConfig]:
    """Get department configuration by ID."""
    return DEPARTMENTS.get(dept_id)


def get_department_for_file_type(extension: str) -> List[Department]:
    """
    Get departments that should receive a file based on its extension.
    
    Returns empty list if content-based routing is required.
    """
    ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    return FILE_TYPE_ROUTING_RULES.get(ext, [])


def get_all_departments() -> List[DepartmentConfig]:
    """Get all department configurations."""
    return list(DEPARTMENTS.values())


def get_department_keywords() -> Dict[Department, Set[str]]:
    """Get a mapping of departments to their keywords for text analysis."""
    return {dept_id: set(config.keywords) for dept_id, config in DEPARTMENTS.items()}
