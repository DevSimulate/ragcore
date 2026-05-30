from fastapi import Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def get_current_tenant(x_api_key: str = Header(...)) -> str:
    if not x_api_key or len(x_api_key) < 8:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # In production this would look up the tenant from a DB/cache
    # For now, derive tenant_id from the API key prefix
    return x_api_key[:8]
