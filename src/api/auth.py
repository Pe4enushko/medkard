"""
api/auth.py — Bearer-token auth dependency for the pull API.

A single unified key authenticates the integrating app, but is scoped to a
specific set of organizations (see storage/api_keys_storage.py). Since
authorization depends on which org a request is for, this dependency takes
the ?org= query param itself rather than being a router-wide dependency —
it resolves the org id, checks the key against api_key_organizations, and
returns the resolved (organization_id, organization_name) for the route to
use directly.

The key is hashed and looked up against api_keys.key_hash — comparison
happens via an indexed DB lookup on the hash, not a raw-secret comparison
in application code, so timing-safe comparison isn't a separate concern here.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from storage.api_keys_storage import ApiKeysStorage
from storage.organizations_storage import OrganizationsStorage

_bearer_scheme = HTTPBearer(auto_error=True)


async def require_org_access(
    org: str = Query(...),
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> tuple[str, str]:
    """Resolve ?org= (case-insensitively) and authorize the bearer key against it.

    Returns (organization_id, canonical_organization_name) on success.
    """
    async with OrganizationsStorage() as organizations:
        try:
            org_id = await organizations.get_id_by_name_ci(org)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown organization: {org}")
        org_name = await organizations.get_name_by_id(org_id)

    async with ApiKeysStorage() as api_keys:
        authorized = await api_keys.is_key_authorized_for_org(credentials.credentials, org_id)
        if not authorized:
            valid_key = await api_keys.is_valid_key(credentials.credentials)

    if not authorized:
        if valid_key:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Key is not authorized for org: {org}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")

    return org_id, org_name
