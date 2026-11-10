"""Pydantic schemas for authentication, authorization, and persistent metadata in EconoCausal."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """Role-based access control tiers."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


# ---------------------------------------------------------
# Authentication & User Management Schemas
# ---------------------------------------------------------

class UserBase(BaseModel):
    email: str = Field(..., description="Unique email address")
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = Field(default=UserRole.ANALYST, description="RBAC role assigned")


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="Plaintext password to be hashed")


class UserLogin(BaseModel):
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., description="Plaintext password")


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None


class UserInDB(UserResponse):
    hashed_password: str


class UserRoleUpdate(BaseModel):
    role: UserRole


class UserStatusUpdate(BaseModel):
    is_active: bool


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenPayload(BaseModel):
    sub: str
    username: str
    role: str
    exp: int


# ---------------------------------------------------------
# Persistent Metadata Schemas
# ---------------------------------------------------------

class DatasetMetadata(BaseModel):
    id: Optional[str] = None
    name: str = Field(..., description="Dataset display or identifier name")
    description: Optional[str] = ""
    file_path: str = Field(..., description="Local filesystem path where dataset resides")
    row_count: int
    column_count: int
    columns: List[str]
    file_size_bytes: int
    source_type: str = "synthetic"  # synthetic, real, upload
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: Optional[str] = "system"


class TrainingRunMetadata(BaseModel):
    id: Optional[str] = None
    run_id: str
    model_name: str
    model_type: str
    parameters: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    artifact_path: Optional[str] = None
    dataset_id: Optional[str] = None
    status: str = "completed"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: Optional[str] = "system"


class ModelRegistryEntry(BaseModel):
    id: Optional[str] = None
    model_name: str
    version: str
    model_type: str
    artifact_path: str
    metrics: Dict[str, Any] = {}
    is_active: bool = True
    description: Optional[str] = ""
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    registered_by: Optional[str] = "system"


class PricingRecommendationRecord(BaseModel):
    id: Optional[str] = None
    scenario_name: str
    strategy: str
    parameters: Dict[str, Any] = {}
    summary_metrics: Dict[str, Any] = {}
    recommendations_count: int = 0
    sample_recommendations: List[Dict[str, Any]] = []
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: Optional[str] = "system"


class PromotionCampaignRecord(BaseModel):
    id: Optional[str] = None
    campaign_id: str
    campaign_name: str
    budget: float
    status: str = "active"  # draft, active, completed, archived
    target_segment: Optional[str] = "all"
    objectives: Dict[str, Any] = {}
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: Optional[str] = "system"


class OptimizationResultRecord(BaseModel):
    id: Optional[str] = None
    campaign_id: str
    total_budget: float
    allocated_budget: float
    expected_uplift: float
    customer_count: int
    status: str = "optimal"
    summary: Dict[str, Any] = {}
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: Optional[str] = "system"


class AuditLogEntry(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    action: str
    entity_type: str
    entity_id: Optional[str] = None
    details: Dict[str, Any] = {}
    ip_address: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------
# Phase 9: Unified API & Pipeline Orchestration Schemas
# ---------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    details: Any


class APIResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Any] = None
    error: Optional[ErrorDetail] = None


class PipelineStageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStageResult(BaseModel):
    stage: str
    status: PipelineStageStatus
    duration_seconds: float = 0.0
    details: Dict[str, Any] = {}
    error: Optional[str] = None


class PipelineRunSummary(BaseModel):
    pipeline_run_id: str
    status: PipelineRunStatus
    created_at: str
    completed_at: Optional[str] = None
    stages: List[PipelineStageResult] = []
    summary: Dict[str, Any] = {}
    error: Optional[str] = None


class PipelineRunRequest(BaseModel):
    dataset_id: Optional[str] = None
    stages: Optional[List[str]] = None
    budget: Optional[float] = 2500.0
    timeout_seconds: Optional[float] = 300.0


class TrainingStartRequest(BaseModel):
    dataset_id: Optional[str] = None
    model_type: str = "all"
    version: str = "v1.0.0"
    parameters: Dict[str, Any] = {}


class CausalAnalyzeRequest(BaseModel):
    dataset_id: Optional[str] = None
    treatment_col: str = "treatment"
    outcome_col: str = "purchase"
    confounders: Optional[List[str]] = None


class UpliftAnalyzeRequest(BaseModel):
    dataset_id: Optional[str] = None
    treatment_col: str = "treatment"
    outcome_col: str = "purchase"
    features: Optional[List[str]] = None


class PricingRecommendCandidate(BaseModel):
    customer_id: str
    product_id: Optional[str] = "P001"
    category: str = "Electronics"
    current_price: float = Field(..., gt=0)
    customer_history: Optional[float] = 100.0
    seasonality: Optional[float] = 1.0
    treatment: Optional[int] = 0
    unit_cost: Optional[float] = None


class PricingRecommendBatchRequest(BaseModel):
    candidates: List[PricingRecommendCandidate]
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_price_change_pct: Optional[float] = None


class PromotionCandidate(BaseModel):
    customer_id: str = Field(min_length=1, max_length=100)
    baseline_price: float = Field(gt=0)
    predicted_ite: float
    category: Optional[str] = "Electronics"
    segment: Optional[str] = "Persuadables"


class PromotionOptimizeRequest(BaseModel):
    candidates: List[PromotionCandidate]
    budget: float = Field(ge=0)
    discount_pct: Optional[float] = Field(default=10.0, gt=0, le=100)
    allowed_discounts: Optional[List[float]] = None
    objective: Optional[str] = "incremental_conversions"
