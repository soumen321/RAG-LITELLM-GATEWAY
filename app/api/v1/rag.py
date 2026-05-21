from fastapi import APIRouter, Depends
from app.models.request import QueryRequest
from app.models.response import QueryResponse, StatsResponse, GatewayInfoResponse,HealthReport
from app.api.deps import verify_key
from app.rag.pipeline import get_pipeline
from app.gateway.router import get_available_aliases
from app.gateway.health import get_health_report
from app.gateway.fallback import FALLBACK_CHAINS
from app.core.config import get_settings

router   = APIRouter()
settings = get_settings()


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    _: str = Depends(verify_key),
):
    result = get_pipeline().query(
        question=req.question,
        top_k=req.top_k,
        model_alias=req.model_alias,        # ← Phase 2: pass alias
        metadata_filter=req.metadata_filter,
    )
    return QueryResponse(**result)


@router.get("/stats", response_model=StatsResponse)
async def stats(_: str = Depends(verify_key)):
    return StatsResponse(**get_pipeline().stats())


# ── NEW: show what providers/models are available ──────────────────
@router.get("/gateway/info", response_model=GatewayInfoResponse)
async def gateway_info(_: str = Depends(verify_key)):
    aliases = get_available_aliases()
    # Derive which providers are loaded from alias definitions
    from app.gateway.router import get_router
    models    = get_router().model_list
    providers = sorted({
        m["litellm_params"]["model"].split("/")[0]
        if "/" in m["litellm_params"]["model"]
        else "openai"
        for m in models
    })
    return GatewayInfoResponse(
        available_aliases=aliases,
        default_alias=settings.default_model_alias,
        providers_loaded=providers,
    )
    
# ── NEW in Phase 3 ──────────────────────────────────────────────────────
@router.get("/gateway/health", response_model=HealthReport)
async def gateway_health(_: str = Depends(verify_key)):
    return HealthReport(
        providers=get_health_report(),
        fallback_chains=FALLBACK_CHAINS,
    )    