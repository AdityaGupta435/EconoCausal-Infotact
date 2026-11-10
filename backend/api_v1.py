"""FastAPI API router for EconoCausal: Unified API Integration & End-to-End Orchestration (Phases 8 & 9)."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status

from backend.auth import (
    create_access_token,
    get_current_active_user,
    get_current_user,
    require_role,
)
from backend.database import check_db_health, get_db
from backend.schemas import (
    APIResponse,
    AuditLogEntry,
    CausalAnalyzeRequest,
    ChangePasswordRequest,
    DatasetMetadata,
    ModelRegistryEntry,
    OptimizationResultRecord,
    PricingRecommendBatchRequest,
    PricingRecommendCandidate,
    PricingRecommendationRecord,
    PipelineRunRequest,
    PromotionCampaignRecord,
    PromotionOptimizeRequest,
    TokenResponse,
    TrainingRunMetadata,
    TrainingStartRequest,
    UpliftAnalyzeRequest,
    UserCreate,
    UserLogin,
    UserResponse,
    UserRole,
    UserRoleUpdate,
    UserStatusUpdate,
)
from backend.services import (
    authenticate_user,
    cancel_pipeline_run_service,
    change_user_password,
    create_user,
    delete_dataset,
    execute_causal_analysis_service,
    execute_uplift_analysis_service,
    get_audit_logs,
    get_budget_analysis_service,
    get_causal_diagnostics_service,
    get_causal_history_service,
    get_causal_results_service,
    get_dashboard_data_summary,
    get_dashboard_metrics,
    get_dashboard_model_performance,
    get_dashboard_overview,
    get_dashboard_recent_activity,
    get_dataset_by_id,
    get_dataset_preview,
    get_dataset_summary_stats,
    get_datasets,
    get_models,
    get_optimization_results,
    get_pipeline_history_service,
    get_pipeline_status_service,
    get_pricing_elasticity_service,
    get_pricing_performance_service,
    get_pricing_recommendations,
    get_promotion_campaign_by_id,
    get_promotion_campaigns,
    get_training_run_logs,
    get_training_run_metrics,
    get_training_run_status,
    get_training_runs,
    get_uplift_customer_segments_service,
    get_uplift_history_service,
    get_uplift_performance_service,
    get_uplift_results_service,
    get_user_by_id,
    list_users,
    optimize_promotions_service,
    recommend_batch_prices_service,
    recommend_single_price_service,
    record_dataset,
    record_model,
    record_optimization_result,
    record_pricing_recommendation,
    record_promotion_campaign,
    record_training_run,
    save_uploaded_dataset,
    start_pipeline_run_service,
    start_training_run_service,
    sync_local_artifacts_to_db,
    update_user_role,
    update_user_status,
)

logger = logging.getLogger("econocausal.api_v1")

router = APIRouter()


def api_envelope(data: Any = None, message: str = "Operation completed successfully", success: bool = True) -> Dict[str, Any]:
    """Helper to format standard Phase 9 API responses."""
    return {
        "success": success,
        "message": message,
        "data": data,
        "error": None,
    }


# =========================================================
# 1. Database Health & System Synchronization
# =========================================================

@router.get("/db/health", tags=["Database"])
def database_health():
    """Check live MongoDB Atlas connection status safely without leaking credentials."""
    return check_db_health()


@router.post("/db/sync", tags=["Database"])
def sync_artifacts(
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Scan and synchronize local Phase 1–7 artifacts/metadata into MongoDB collections."""
    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB is not connected. Ensure MONGODB_URI is configured.",
        )
    result = sync_local_artifacts_to_db(db, actor=current_user)
    return api_envelope(result, message="Local artifacts synchronized to database.")


# =========================================================
# 2. Authentication Endpoints
# =========================================================

@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Authentication"])
def register(user_data: UserCreate, request: Request):
    """Register a new user account with hashed password and RBAC role."""
    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )
    try:
        user = create_user(db, user_data)
        return user
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
def login(credentials: UserLogin, request: Request):
    """Authenticate with username or email and password; receive JWT bearer token."""
    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )

    client_ip = request.client.host if request.client else None
    user = authenticate_user(
        db,
        credentials.username_or_email,
        credentials.password,
        ip_address=client_ip,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_payload = {
        "sub": user["id"],
        "username": user["username"],
        "role": user["role"],
    }
    access_token = create_access_token(token_payload)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 1800,  # 30 minutes
        "user": user,
    }


@router.post("/auth/logout", tags=["Authentication"])
def logout():
    """Stateless JWT logout endpoint."""
    return api_envelope(message="Logged out successfully.")


@router.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
def get_me(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Return profile details for currently authenticated user."""
    return current_user


@router.post("/auth/change-password", tags=["Authentication"])
def user_change_password(
    req: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Allow an authenticated user to change their account password."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    success = change_user_password(
        db,
        user_id=current_user["id"],
        current_password=req.current_password,
        new_password=req.new_password,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Current password verification failed.")
    return api_envelope(message="Password changed successfully.")


# =========================================================
# 3. User Administration (Admin Only)
# =========================================================

@router.get("/users", response_model=List[UserResponse], tags=["User Management"])
def get_all_users(
    skip: int = 0,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN])),
):
    """List all registered users (Admin only)."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    return list_users(db, skip=skip, limit=limit)


@router.get("/users/{user_id}", response_model=UserResponse, tags=["User Management"])
def get_user_details(
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Retrieve details for a specific user (Admin or self)."""
    if current_user.get("role") != UserRole.ADMIN and current_user.get("id") != user_id:
        raise HTTPException(status_code=403, detail="Cannot access other user accounts.")
    db = get_db()
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


@router.patch("/users/{user_id}/role", response_model=UserResponse, tags=["User Management"])
def set_user_role(
    user_id: str,
    payload: UserRoleUpdate,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN])),
):
    """Update assigned role of a user (Admin only)."""
    db = get_db()
    updated = update_user_role(db, user_id, payload.role, actor=current_user)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return updated


@router.patch("/users/{user_id}/status", response_model=UserResponse, tags=["User Management"])
def set_user_status(
    user_id: str,
    payload: UserStatusUpdate,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN])),
):
    """Enable or disable a user account (Admin only)."""
    db = get_db()
    updated = update_user_status(db, user_id, payload.is_active, actor=current_user)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return updated


# =========================================================
# 4. Dashboard API
# =========================================================

@router.get("/dashboard/overview", tags=["Dashboard"])
def dashboard_overview(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve high-level project metrics across all modules."""
    db = get_db()
    data = get_dashboard_overview(db)
    return api_envelope(data, message="Dashboard overview retrieved.")


@router.get("/dashboard/metrics", tags=["Dashboard"])
def dashboard_metrics(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve numerical statistics and model evaluation results."""
    db = get_db()
    data = get_dashboard_metrics(db)
    return api_envelope(data, message="Dashboard metrics retrieved.")


@router.get("/dashboard/model-performance", tags=["Dashboard"])
def dashboard_model_performance(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve detailed validation and test performance of trained ML models."""
    db = get_db()
    data = get_dashboard_model_performance(db)
    return api_envelope(data, message="Model performance retrieved.")


@router.get("/dashboard/recent-activity", tags=["Dashboard"])
def dashboard_recent_activity(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve recent platform audit activities."""
    db = get_db()
    data = get_dashboard_recent_activity(db, limit=10)
    return api_envelope(data, message="Recent activity logs retrieved.")


@router.get("/dashboard/data-summary", tags=["Dashboard"])
def dashboard_data_summary(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve dataset summary counts and source distributions."""
    db = get_db()
    data = get_dashboard_data_summary(db)
    return api_envelope(data, message="Data summary retrieved.")


# =========================================================
# 5. Dataset Management API
# =========================================================

@router.get("/datasets", tags=["Datasets"])
def list_all_datasets(
    skip: int = 0,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """List persisted dataset metadata records."""
    db = get_db()
    data = get_datasets(db, skip=skip, limit=limit)
    return data  # Preserves backwards compatibility with Phase 8 tests


@router.post("/datasets", tags=["Datasets"], status_code=status.HTTP_201_CREATED)
def create_dataset_record_metadata(
    metadata: DatasetMetadata,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Record dataset metadata in MongoDB (large CSV files remain on disk)."""
    db = get_db()
    saved = record_dataset(db, metadata.dict(exclude={"id"}), actor=current_user)
    return saved


@router.post("/datasets/upload", tags=["Datasets"], status_code=status.HTTP_201_CREATED)
async def upload_dataset_file(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Upload CSV dataset, validate content, save securely, and record in MongoDB."""
    content = await file.read()
    db = get_db()
    try:
        saved = save_uploaded_dataset(db, file.filename, content, actor=current_user)
        return api_envelope(saved, message="Dataset uploaded and registered successfully.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}", tags=["Datasets"])
def get_dataset_by_identifier(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve metadata of a specific dataset."""
    db = get_db()
    ds = get_dataset_by_id(db, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return api_envelope(ds, message="Dataset metadata retrieved.")


@router.get("/datasets/{dataset_id}/preview", tags=["Datasets"])
def dataset_preview(
    dataset_id: str,
    n_rows: int = 5,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Inspect top N rows of the dataset without reading the full file into memory."""
    db = get_db()
    try:
        preview = get_dataset_preview(db, dataset_id, n_rows=n_rows)
        return api_envelope(preview, message="Dataset preview generated.")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/summary", tags=["Datasets"])
def dataset_summary(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve summary statistics and column distributions for a dataset."""
    db = get_db()
    try:
        summary = get_dataset_summary_stats(db, dataset_id)
        return api_envelope(summary, message="Dataset summary statistics retrieved.")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/datasets/{dataset_id}", tags=["Datasets"])
def delete_dataset_record(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Safely delete uploaded dataset (Core system datasets are strictly protected)."""
    db = get_db()
    try:
        deleted = delete_dataset(db, dataset_id, actor=current_user)
        return api_envelope({"deleted": deleted}, message="Dataset deleted successfully.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# =========================================================
# 6. ML Training API
# =========================================================

@router.post("/training/start", tags=["ML Training"], status_code=status.HTTP_202_ACCEPTED)
def start_training(
    req: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Trigger asynchronous model training pipeline without blocking API server."""
    db = get_db()
    job = start_training_run_service(db, req.dict(), actor=current_user, background_tasks=background_tasks)
    return api_envelope(job, message="Training run queued successfully.")


@router.get("/training/status/{training_run_id}", tags=["ML Training"])
def training_status(
    training_run_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Poll current execution status and progress of a training run."""
    db = get_db()
    status_doc = get_training_run_status(db, training_run_id)
    if not status_doc:
        raise HTTPException(status_code=404, detail=f"Training run '{training_run_id}' not found.")
    return api_envelope(status_doc, message="Training status retrieved.")


@router.get("/training/history", tags=["ML Training"])
def training_history(
    skip: int = 0,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """List historical training runs."""
    db = get_db()
    runs = get_training_runs(db, skip=skip, limit=limit)
    return api_envelope(runs, message="Training history retrieved.")


@router.get("/training/{training_run_id}/metrics", tags=["ML Training"])
def training_metrics(
    training_run_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve detailed evaluation metrics for a specific training run."""
    db = get_db()
    metrics = get_training_run_metrics(db, training_run_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Training run '{training_run_id}' not found.")
    return api_envelope(metrics, message="Training metrics retrieved.")


@router.get("/training/{training_run_id}/logs", tags=["ML Training"])
def training_logs(
    training_run_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve execution log stream for a specific training run."""
    db = get_db()
    logs = get_training_run_logs(db, training_run_id)
    return api_envelope(logs, message="Training logs retrieved.")


# Backward compatibility for Phase 8 endpoint
@router.get("/training-runs", tags=["Training Runs"])
def list_training_runs(
    skip: int = 0,
    limit: int = 50,
    model_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    db = get_db()
    return get_training_runs(db, skip=skip, limit=limit, model_type=model_type)


# =========================================================
# 7. Causal Inference API
# =========================================================

@router.post("/causal/analyze", tags=["Causal Inference"])
def analyze_causal_effects(
    req: CausalAnalyzeRequest,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Execute causal inference analysis: ATE, CATE, DML, propensity scores, and refutations."""
    db = get_db()
    result = execute_causal_analysis_service(db, req.dict(), actor=current_user)
    return api_envelope(result, message="Causal inference analysis completed.")


@router.get("/causal/results/{analysis_id}", tags=["Causal Inference"])
def get_causal_results(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve causal estimation results and treatment effects for a specific run."""
    db = get_db()
    results = get_causal_results_service(db, analysis_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"Causal analysis '{analysis_id}' not found.")
    return api_envelope(results, message="Causal analysis results retrieved.")


@router.get("/causal/history", tags=["Causal Inference"])
def get_causal_history(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """List historical causal inference analyses."""
    db = get_db()
    history = get_causal_history_service(db)
    return api_envelope(history, message="Causal history retrieved.")


@router.get("/causal/diagnostics/{analysis_id}", tags=["Causal Inference"])
def get_causal_diagnostics(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve propensity score overlap, common support, and covariate balance diagnostics."""
    db = get_db()
    diag = get_causal_diagnostics_service(db, analysis_id)
    return api_envelope(diag, message="Causal diagnostics retrieved.")


# =========================================================
# 8. Uplift Modeling API
# =========================================================

@router.post("/uplift/analyze", tags=["Uplift Modeling"])
def analyze_uplift(
    req: UpliftAnalyzeRequest,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Execute uplift modeling analysis: meta-learners, Qini curves, AUUC, and 4-quadrant segmentation."""
    db = get_db()
    res = execute_uplift_analysis_service(db, req.dict(), actor=current_user)
    return api_envelope(res, message="Uplift modeling analysis completed.")


@router.get("/uplift/results/{analysis_id}", tags=["Uplift Modeling"])
def get_uplift_results(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve uplift scores and evaluation metrics for a specific run."""
    db = get_db()
    res = get_uplift_results_service(db, analysis_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Uplift analysis '{analysis_id}' not found.")
    return api_envelope(res, message="Uplift results retrieved.")


@router.get("/uplift/history", tags=["Uplift Modeling"])
def get_uplift_history(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """List historical uplift modeling runs."""
    db = get_db()
    history = get_uplift_history_service(db)
    return api_envelope(history, message="Uplift history retrieved.")


@router.get("/uplift/customer-segments", tags=["Uplift Modeling"])
def get_uplift_customer_segments(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve customer distribution across Persuadables, Sure Things, Lost Causes, and Sleeping Dogs."""
    db = get_db()
    segments = get_uplift_customer_segments_service(db)
    return api_envelope(segments, message="Customer segments retrieved.")


@router.get("/uplift/performance", tags=["Uplift Modeling"])
def get_uplift_performance(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve Qini, AUUC, and model comparison metrics against random targeting."""
    db = get_db()
    perf = get_uplift_performance_service(db)
    return api_envelope(perf, message="Uplift performance metrics retrieved.")


# =========================================================
# 9. Dynamic Pricing API
# =========================================================

@router.post("/pricing/recommend", tags=["Dynamic Pricing"])
def recommend_price(
    candidate: PricingRecommendCandidate,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Generate optimal dynamic price recommendation subject to elasticity and business bounds."""
    try:
        rec = recommend_single_price_service(candidate.dict())
        return api_envelope(rec, message="Dynamic price recommendation generated.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pricing/batch-recommend", tags=["Dynamic Pricing"])
def batch_recommend_prices(
    req: PricingRecommendBatchRequest,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Generate dynamic price recommendations for a batch of opportunities in a vectorized call."""
    db = get_db()
    try:
        res = recommend_batch_prices_service(db, [c.dict() for c in req.candidates], actor=current_user)
        return api_envelope(res, message="Batch price recommendations generated.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/pricing/recommendations", tags=["Dynamic Pricing"])
def list_pricing_recommendations(
    skip: int = 0,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """List persisted pricing recommendations."""
    db = get_db()
    return get_pricing_recommendations(db, skip=skip, limit=limit)


@router.get("/pricing/recommendations/{recommendation_id}", tags=["Dynamic Pricing"])
def get_pricing_recommendation_by_id(
    recommendation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve a specific persisted pricing recommendation."""
    db = get_db()
    recs = get_pricing_recommendations(db)
    found = next((r for r in recs if r.get("id") == recommendation_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Pricing recommendation record not found.")
    return api_envelope(found, message="Pricing recommendation retrieved.")


@router.get("/pricing/elasticity", tags=["Dynamic Pricing"])
def get_pricing_elasticity(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve overall and category-level price elasticity of demand."""
    elasticity = get_pricing_elasticity_service()
    return api_envelope(elasticity, message="Price elasticity data retrieved.")


@router.get("/pricing/performance", tags=["Dynamic Pricing"])
def get_pricing_performance(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Retrieve pricing demand model metrics and expected revenue benchmarks."""
    perf = get_pricing_performance_service()
    return api_envelope(perf, message="Pricing performance metrics retrieved.")


# =========================================================
# 10. Promotion Optimization API
# =========================================================

@router.post("/promotions/optimize", tags=["Promotion Optimization"])
def optimize_promotions_endpoint(
    req: PromotionOptimizeRequest,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Allocate marketing budget across candidates to maximize incremental purchases via SciPy MILP."""
    db = get_db()
    try:
        res = optimize_promotions_service(db, req.dict(), actor=current_user)
        return api_envelope(res, message="Promotion optimization solved successfully.")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/promotions/campaigns", tags=["Promotion Optimization"])
def list_promotion_campaigns(
    skip: int = 0,
    limit: int = 50,
    campaign_status: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """List marketing promotion campaigns."""
    db = get_db()
    return get_promotion_campaigns(db, skip=skip, limit=limit, status=campaign_status)


@router.get("/promotions/campaigns/{campaign_id}", tags=["Promotion Optimization"])
def get_campaign_by_id(
    campaign_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve a specific promotion campaign."""
    db = get_db()
    c = get_promotion_campaign_by_id(db, campaign_id)
    if not c:
        raise HTTPException(status_code=404, detail="Campaign not found.")
    return api_envelope(c, message="Campaign retrieved.")


@router.get("/promotions/results", tags=["Promotion Optimization"])
def list_optimization_results(
    skip: int = 0,
    limit: int = 50,
    campaign_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """List historical optimization run allocations."""
    db = get_db()
    results = get_optimization_results(db, skip=skip, limit=limit, campaign_id=campaign_id)
    return results


@router.get("/promotions/results/{optimization_id}", tags=["Promotion Optimization"])
def get_optimization_result_by_id(
    optimization_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve allocation results for a specific optimization run."""
    db = get_db()
    results = get_optimization_results(db)
    found = next((r for r in results if r.get("id") == optimization_id or r.get("campaign_id") == optimization_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Optimization result record not found.")
    return api_envelope(found, message="Optimization result retrieved.")


@router.get("/promotions/budget-analysis", tags=["Promotion Optimization"])
def get_promotions_budget_analysis(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Compare multi-scenario budget allocations across different spending limits."""
    db = get_db()
    analysis = get_budget_analysis_service(db)
    return api_envelope(analysis, message="Budget scenario analysis retrieved.")


# =========================================================
# 11. End-to-End Pipeline Orchestration API
# =========================================================

@router.post("/pipeline/run", tags=["Pipeline Orchestration"], status_code=status.HTTP_202_ACCEPTED)
def run_pipeline(
    req: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Execute end-to-end orchestration pipeline sequentially across all engines."""
    db = get_db()
    pipeline_job = start_pipeline_run_service(db, req.dict(), actor=current_user, background_tasks=background_tasks)
    return api_envelope(pipeline_job, message="Pipeline run initiated.")


@router.get("/pipeline/status/{pipeline_run_id}", tags=["Pipeline Orchestration"])
def get_pipeline_status(
    pipeline_run_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Poll execution status and stage progress of an orchestrated pipeline run."""
    db = get_db()
    status_doc = get_pipeline_status_service(db, pipeline_run_id)
    if not status_doc:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{pipeline_run_id}' not found.")
    return api_envelope(status_doc, message="Pipeline status retrieved.")


@router.get("/pipeline/history", tags=["Pipeline Orchestration"])
def get_pipeline_history(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """List historical pipeline execution runs."""
    db = get_db()
    history = get_pipeline_history_service(db)
    return api_envelope(history, message="Pipeline history retrieved.")


@router.get("/pipeline/results/{pipeline_run_id}", tags=["Pipeline Orchestration"])
def get_pipeline_results(
    pipeline_run_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve final results and stage outputs for a completed pipeline run."""
    db = get_db()
    status_doc = get_pipeline_status_service(db, pipeline_run_id)
    if not status_doc:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{pipeline_run_id}' not found.")
    return api_envelope(status_doc, message="Pipeline results retrieved.")


@router.post("/pipeline/cancel/{pipeline_run_id}", tags=["Pipeline Orchestration"])
def cancel_pipeline(
    pipeline_run_id: str,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Cancel a queued or active pipeline run."""
    db = get_db()
    cancelled = cancel_pipeline_run_service(db, pipeline_run_id, actor=current_user)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Pipeline run cannot be cancelled (already completed or not found).")
    return api_envelope({"cancelled": True}, message="Pipeline run cancelled.")


# =========================================================
# 12. Model Registry & Audit Logs (Phase 8 Compatibility)
# =========================================================

@router.get("/models", tags=["Model Registry"])
def list_models(
    skip: int = 0,
    limit: int = 50,
    is_active: Optional[bool] = None,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Retrieve registered models."""
    db = get_db()
    return get_models(db, skip=skip, limit=limit, is_active=is_active)


@router.post("/models", tags=["Model Registry"], status_code=status.HTTP_201_CREATED)
def register_model_entry(
    model_data: ModelRegistryEntry,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    """Register a trained model artifact metadata."""
    db = get_db()
    saved = record_model(db, model_data.dict(exclude={"id"}), actor=current_user)
    return saved


@router.get("/audit-logs", tags=["Audit Logs"])
def list_audit_trail(
    skip: int = 0,
    limit: int = 50,
    action: Optional[str] = None,
    user_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN])),
):
    """Retrieve system audit logs for administrative monitoring (Admin only)."""
    db = get_db()
    return get_audit_logs(db, skip=skip, limit=limit, action=action, user_id=user_id)
