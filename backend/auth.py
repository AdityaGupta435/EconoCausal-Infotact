"""Authentication, password security, JWT management, and RBAC middleware."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend import config
from backend.database import get_db
from backend.schemas import UserRole

logger = logging.getLogger("econocausal.auth")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ---------------------------------------------------------
# Password Hashing & Verification
# ---------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt with salt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception as exc:
        logger.warning("Password verification error: %s", str(exc))
        return False


# ---------------------------------------------------------
# JWT Token Generation & Validation
# ---------------------------------------------------------

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generate a signed JWT access token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=config.JWT_ACCESS_TOKEN_EXPIRES_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": now,
    })

    encoded_jwt = jwt.encode(
        to_encode,
        config.JWT_SECRET_KEY,
        algorithm=config.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            config.JWT_SECRET_KEY,
            algorithms=[config.JWT_ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("JWT token has expired.")
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug("Invalid JWT token: %s", str(exc))
        return None
    except Exception as exc:
        logger.warning("Unexpected error decoding JWT: %s", str(exc))
        return None


# ---------------------------------------------------------
# FastAPI Dependencies for User Authentication & RBAC
# ---------------------------------------------------------

def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Validate Bearer token and retrieve the authenticated user from the database."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise credentials_exception

    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable for user authentication.",
        )

    from backend.services import get_user_by_id
    user = get_user_by_id(db, user_id)
    if user is None:
        raise credentials_exception

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive or disabled.",
        )

    return user


def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency verifying that the current authenticated user is active."""
    return current_user


def require_role(allowed_roles: Union[List[UserRole], List[str]]):
    """Enforce Role-Based Access Control (RBAC).

    Usage:
        @app.get("/admin-only", dependencies=[Depends(require_role(["admin"]))])
        or
        def admin_view(user: dict = Depends(require_role([UserRole.ADMIN]))):
    """
    normalized_roles = [
        r.value if isinstance(r, UserRole) else str(r).lower()
        for r in allowed_roles
    ]

    def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> Dict[str, Any]:
        user_role = str(current_user.get("role", "")).lower()
        if user_role not in normalized_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Requires one of [{', '.join(normalized_roles)}] role(s). Current role: '{user_role}'.",
            )
        return current_user

    return role_checker
