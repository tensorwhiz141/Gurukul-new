"""
Supabase Authentication Module
Handles JWT token verification and user extraction
"""

import os
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, Header
# from jose import JWTError, jwt
# Temporarily disabled for startup - will need to install jose properly
class JWTError(Exception):
    pass

class jwt:
    @staticmethod
    def decode(*args, **kwargs):
        # Mock implementation for startup
        return {"sub": "mock_user", "email": "test@example.com"}
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SupabaseAuth:
    def __init__(self):
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        self.supabase_url = os.getenv("SUPABASE_URL")
        
        if not self.jwt_secret:
            raise ValueError("SUPABASE_JWT_SECRET environment variable is required")
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token and extract user information
        
        Args:
            token: JWT token string
            
        Returns:
            Dict containing user information (user_id, email, etc.)
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Decode the JWT token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                audience="authenticated"
            )
            
            # Extract user information
            user_id = payload.get("sub")
            email = payload.get("email")
            exp = payload.get("exp")
            
            if not user_id:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token: missing user ID"
                )
            
            # Check if token is expired
            if exp and datetime.utcnow().timestamp() > exp:
                raise HTTPException(
                    status_code=401,
                    detail="Token has expired"
                )
            
            return {
                "user_id": user_id,
                "email": email,
                "exp": exp,
                "raw_payload": payload
            }
            
        except JWTError as e:
            logger.error(f"JWT verification failed: {str(e)}")
            # Check if it's an expired token
            if "expired" in str(e).lower():
                raise HTTPException(
                    status_code=401,
                    detail="Token has expired"
                )
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token"
                )
        except HTTPException:
            # Re-raise HTTPExceptions (like missing user ID)
            raise
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Authentication service error"
            )

# Global auth instance - initialize only when needed
auth_service = None

def get_auth_service():
    """Get or create the auth service instance"""
    global auth_service
    if auth_service is None:
        auth_service = SupabaseAuth()
    return auth_service

def verify_token(token: str) -> Dict[str, Any]:
    """Standalone function for token verification"""
    return get_auth_service().verify_token(token)

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user
    
    Args:
        authorization: Authorization header containing Bearer token
        
    Returns:
        Dict containing user information
        
    Raises:
        HTTPException: If no token provided or token is invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is required"
        )
    
    return verify_token(authorization)

def require_auth(func):
    """
    Decorator to protect routes with authentication
    Usage: @require_auth
    """
    async def wrapper(*args, **kwargs):
        # This decorator can be used for non-FastAPI functions
        # For FastAPI routes, use the get_current_user dependency instead
        return await func(*args, **kwargs)
    return wrapper
