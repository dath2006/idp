"""Configuration module for IDP system."""

from .departments import (
    Department,
    DepartmentConfig,
    DEPARTMENTS,
    FILE_TYPE_ROUTING_RULES,
    get_department_by_id,
    get_department_for_file_type,
    get_all_departments,
)

__all__ = [
    "Department",
    "DepartmentConfig",
    "DEPARTMENTS",
    "FILE_TYPE_ROUTING_RULES",
    "get_department_by_id",
    "get_department_for_file_type",
    "get_all_departments",
]
