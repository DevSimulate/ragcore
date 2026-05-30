import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_tenant

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def list_collections(tenant_id: str = Depends(get_current_tenant)):
    return {"tenant_id": tenant_id, "collections": [f"ragcore_{tenant_id}"]}


@router.delete("/{collection_name}")
async def delete_collection(
    collection_name: str,
    tenant_id: str = Depends(get_current_tenant),
):
    if not collection_name.startswith(f"ragcore_{tenant_id}"):
        raise HTTPException(status_code=403, detail="Cannot delete another tenant's collection")
    return {"status": "deleted", "collection": collection_name}
