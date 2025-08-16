"""
BHIV API Gateway
Comprehensive API gateway for the BHIV ecosystem

Features:
- Authentication and authorization
- Rate limiting and throttling
- Request/response logging
- API key management
- External API integration
- Load balancing
- Security middleware
- Monitoring and analytics
"""

import os
import json
import logging
import asyncio
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from functools import wraps
import time
from collections import defaultdict, deque

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.base import BaseHTTPMiddleware
import redis

# Local imports
from bhiv_core_schema import UserInfo, UserPersona

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AuthenticationManager:
    """Handles authentication and authorization"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "bhiv-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        self.api_keys = self._load_api_keys()
        
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment or database"""
        # In production, this would load from a secure database
        return {
            "bhiv-core-key": {
                "name": "BHIV Core Service",
                "permissions": ["read", "write", "admin"],
                "rate_limit": 1000,  # requests per hour
                "expires": None
            },
            "frontend-key": {
                "name": "Frontend Application",
                "permissions": ["read", "write"],
                "rate_limit": 500,
                "expires": None
            },
            "mobile-key": {
                "name": "Mobile Application",
                "permissions": ["read", "write"],
                "rate_limit": 300,
                "expires": None
            }
        }
    
    def create_access_token(self, user_info: UserInfo) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "user_id": user_info.user_id,
            "persona": user_info.persona.value if user_info.persona else "student",
            "permissions": user_info.permissions,
            "lang": user_info.lang,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "bhiv-core"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key"""
        if api_key in self.api_keys:
            key_info = self.api_keys[api_key]
            
            # Check if key has expired
            if key_info.get("expires"):
                if datetime.now() > datetime.fromisoformat(key_info["expires"]):
                    logger.warning(f"API key {api_key} has expired")
                    return None
            
            return key_info
        
        logger.warning(f"Invalid API key: {api_key}")
        return None
    
    def check_permissions(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions or "admin" in user_permissions


class RateLimiter:
    """Rate limiting functionality"""
    
    def __init__(self):
        self.redis_client = self._init_redis()
        self.memory_store = defaultdict(deque)  # Fallback if Redis unavailable
        
    def _init_redis(self):
        """Initialize Redis client"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(redis_url)
            client.ping()  # Test connection
            logger.info("Redis connected for rate limiting")
            return client
        except Exception as e:
            logger.warning(f"Redis not available, using memory store: {e}")
            return None
    
    async def is_rate_limited(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int = 3600
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited"""
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        if self.redis_client:
            return await self._redis_rate_limit(identifier, limit, window_seconds, current_time)
        else:
            return self._memory_rate_limit(identifier, limit, window_start, current_time)
    
    async def _redis_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int, 
        current_time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Redis-based rate limiting"""
        try:
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(identifier, 0, current_time - window_seconds)
            
            # Count current requests
            pipe.zcard(identifier)
            
            # Add current request
            pipe.zadd(identifier, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(identifier, window_seconds)
            
            results = pipe.execute()
            current_count = results[1]
            
            rate_limit_info = {
                "limit": limit,
                "remaining": max(0, limit - current_count - 1),
                "reset_time": current_time + window_seconds,
                "current_count": current_count + 1
            }
            
            return current_count >= limit, rate_limit_info
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fallback to memory store
            return self._memory_rate_limit(identifier, limit, current_time - window_seconds, current_time)
    
    def _memory_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window_start: float, 
        current_time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Memory-based rate limiting"""
        
        # Clean old entries
        while self.memory_store[identifier] and self.memory_store[identifier][0] < window_start:
            self.memory_store[identifier].popleft()
        
        current_count = len(self.memory_store[identifier])
        
        # Add current request
        self.memory_store[identifier].append(current_time)
        
        rate_limit_info = {
            "limit": limit,
            "remaining": max(0, limit - current_count - 1),
            "reset_time": current_time + 3600,  # 1 hour window
            "current_count": current_count + 1
        }
        
        return current_count >= limit, rate_limit_info


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for API gateway"""
    
    async def dispatch(self, request: Request, call_next):
        # Add security headers
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware"""
    
    def __init__(self, app, log_requests: bool = True):
        super().__init__(app)
        self.log_requests = log_requests
        
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            logger.info(f"Request: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        if self.log_requests:
            logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class BHIVAPIGateway:
    """Main BHIV API Gateway"""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager()
        self.rate_limiter = RateLimiter()
        self.security = HTTPBearer()
        self.external_apis = self._init_external_apis()
        
    def _init_external_apis(self) -> Dict[str, Dict[str, Any]]:
        """Initialize external API configurations"""
        return {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "rate_limit": 100,
                "timeout": 30
            },
            "elevenlabs": {
                "base_url": "https://api.elevenlabs.io/v1",
                "api_key": os.getenv("ELEVENLABS_API_KEY"),
                "rate_limit": 50,
                "timeout": 60
            },
            "gemini": {
                "base_url": "https://generativelanguage.googleapis.com/v1",
                "api_key": os.getenv("GEMINI_API_KEY"),
                "rate_limit": 200,
                "timeout": 30
            }
        }
    
    async def authenticate_request(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        """Authenticate incoming request"""
        
        token = credentials.credentials
        
        # Try JWT token first
        jwt_payload = self.auth_manager.verify_token(token)
        if jwt_payload:
            return {
                "type": "jwt",
                "user_id": jwt_payload["user_id"],
                "permissions": jwt_payload["permissions"],
                "persona": jwt_payload.get("persona", "student"),
                "lang": jwt_payload.get("lang", "en")
            }
        
        # Try API key
        api_key_info = self.auth_manager.verify_api_key(token)
        if api_key_info:
            return {
                "type": "api_key",
                "name": api_key_info["name"],
                "permissions": api_key_info["permissions"],
                "rate_limit": api_key_info["rate_limit"]
            }
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    async def check_rate_limit(
        self, 
        request: Request, 
        auth_info: Dict[str, Any]
    ):
        """Check rate limiting"""
        
        # Determine rate limit
        if auth_info["type"] == "api_key":
            limit = auth_info.get("rate_limit", 100)
        else:
            limit = 500  # Default for JWT users
        
        # Create identifier
        identifier = f"{auth_info.get('user_id', auth_info.get('name'))}:{request.client.host}"
        
        # Check rate limit
        is_limited, rate_info = await self.rate_limiter.is_rate_limited(identifier, limit)
        
        if is_limited:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(int(rate_info["reset_time"]))
                }
            )
        
        return rate_info
    
    def require_permission(self, required_permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get auth info from kwargs (injected by dependency)
                auth_info = kwargs.get("auth_info")
                if not auth_info:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # Check permission
                user_permissions = auth_info.get("permissions", [])
                if not self.auth_manager.check_permissions(user_permissions, required_permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission '{required_permission}' required"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    async def proxy_external_api(
        self, 
        api_name: str, 
        endpoint: str, 
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Proxy request to external API"""
        
        if api_name not in self.external_apis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"External API '{api_name}' not configured"
            )
        
        api_config = self.external_apis[api_name]
        
        # Check rate limit for external API
        identifier = f"external:{api_name}"
        is_limited, _ = await self.rate_limiter.is_rate_limited(
            identifier, api_config["rate_limit"]
        )
        
        if is_limited:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {api_name} API"
            )
        
        # Prepare request
        url = f"{api_config['base_url']}/{endpoint.lstrip('/')}"
        request_headers = headers or {}
        
        # Add API key if available
        if api_config.get("api_key"):
            if api_name == "openai":
                request_headers["Authorization"] = f"Bearer {api_config['api_key']}"
            elif api_name == "elevenlabs":
                request_headers["xi-api-key"] = api_config["api_key"]
            elif api_name == "gemini":
                request_headers["Authorization"] = f"Bearer {api_config['api_key']}"
        
        # Make request (this would use aiohttp in practice)
        try:
            # Placeholder for actual HTTP request
            logger.info(f"Proxying {method} request to {api_name}: {url}")
            
            # Return mock response for now
            return {
                "status": "success",
                "api": api_name,
                "endpoint": endpoint,
                "method": method,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"External API request failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External API '{api_name}' request failed"
            )
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get API gateway statistics"""
        return {
            "external_apis": list(self.external_apis.keys()),
            "auth_methods": ["jwt", "api_key"],
            "rate_limiting": "enabled",
            "security_headers": "enabled",
            "timestamp": datetime.now().isoformat()
        }
