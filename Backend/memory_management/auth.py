"""
Authentication and authorization for the Memory Management API.

This module handles API key validation, rate limiting, and user authentication
for secure access to memory management endpoints.
"""

import os
import time
import hashlib
import logging
from typing import Dict, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configure logging
logger = logging.getLogger(__name__)

# Security configuration
API_KEYS = {
    # Default API keys - should be configured via environment variables
    "memory_api_key_dev": "development_user",
    "memory_api_key_prod": "production_user",
    "memory_api_key_test": "test_user"
}

# Load API keys from environment
env_api_keys = os.getenv("MEMORY_API_KEYS", "")
if env_api_keys:
    try:
        # Format: "key1:user1,key2:user2"
        for pair in env_api_keys.split(","):
            key, user = pair.strip().split(":")
            API_KEYS[key] = user
    except Exception as e:
        logger.error(f"Failed to parse MEMORY_API_KEYS: {e}")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("MEMORY_RATE_LIMIT_REQUESTS", "1000"))
RATE_LIMIT_WINDOW = int(os.getenv("MEMORY_RATE_LIMIT_WINDOW", "3600"))  # 1 hour
RATE_LIMIT_BURST = int(os.getenv("MEMORY_RATE_LIMIT_BURST", "100"))  # Burst limit

# Security instance
security = HTTPBearer()

# Rate limiting storage
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.burst_requests: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            identifier: Unique identifier (API key hash)
            
        Returns:
            bool: True if request is allowed
        """
        now = time.time()
        
        # Clean old requests
        self._cleanup_old_requests(identifier, now)
        
        # Check burst limit (last minute)
        burst_window = now - 60  # 1 minute
        burst_count = sum(1 for req_time in self.burst_requests[identifier] if req_time > burst_window)
        
        if burst_count >= RATE_LIMIT_BURST:
            logger.warning(f"Burst rate limit exceeded for {identifier}")
            return False
        
        # Check hourly limit
        window_start = now - RATE_LIMIT_WINDOW
        hourly_count = sum(1 for req_time in self.requests[identifier] if req_time > window_start)
        
        if hourly_count >= RATE_LIMIT_REQUESTS:
            logger.warning(f"Hourly rate limit exceeded for {identifier}")
            return False
        
        # Record this request
        self.requests[identifier].append(now)
        self.burst_requests[identifier].append(now)
        
        return True
    
    def _cleanup_old_requests(self, identifier: str, now: float):
        """Remove old requests outside the rate limit window."""
        # Clean hourly requests
        window_start = now - RATE_LIMIT_WINDOW
        while (self.requests[identifier] and 
               self.requests[identifier][0] <= window_start):
            self.requests[identifier].popleft()
        
        # Clean burst requests
        burst_window = now - 60
        while (self.burst_requests[identifier] and 
               self.burst_requests[identifier][0] <= burst_window):
            self.burst_requests[identifier].popleft()
    
    def get_remaining_requests(self, identifier: str) -> Dict[str, int]:
        """Get remaining requests for an identifier."""
        now = time.time()
        self._cleanup_old_requests(identifier, now)
        
        window_start = now - RATE_LIMIT_WINDOW
        hourly_count = sum(1 for req_time in self.requests[identifier] if req_time > window_start)
        
        burst_window = now - 60
        burst_count = sum(1 for req_time in self.burst_requests[identifier] if req_time > burst_window)
        
        return {
            "hourly_remaining": max(0, RATE_LIMIT_REQUESTS - hourly_count),
            "burst_remaining": max(0, RATE_LIMIT_BURST - burst_count),
            "reset_time": int(now + RATE_LIMIT_WINDOW)
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


def hash_api_key(api_key: str) -> str:
    """Create a hash of the API key for rate limiting."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify API key and return associated user.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        str: User identifier associated with the API key
        
    Raises:
        HTTPException: If API key is invalid or rate limited
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = credentials.credentials
    
    # Validate API key
    if api_key not in API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check rate limits
    key_hash = hash_api_key(api_key)
    if not rate_limiter.is_allowed(key_hash):
        remaining = rate_limiter.get_remaining_requests(key_hash)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Remaining": str(remaining["hourly_remaining"]),
                "X-RateLimit-Reset": str(remaining["reset_time"]),
                "Retry-After": "3600"
            }
        )
    
    user = API_KEYS[api_key]
    logger.info(f"Authenticated user: {user}")
    
    return user


def get_current_user(user: str = Depends(verify_api_key)) -> str:
    """
    Get the current authenticated user.
    
    Args:
        user: User from API key verification
        
    Returns:
        str: Current user identifier
    """
    return user


def get_rate_limit_info(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, int]:
    """
    Get rate limit information for the current API key.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Dict with rate limit information
    """
    if not credentials or credentials.credentials not in API_KEYS:
        return {
            "hourly_remaining": 0,
            "burst_remaining": 0,
            "reset_time": int(time.time())
        }
    
    key_hash = hash_api_key(credentials.credentials)
    return rate_limiter.get_remaining_requests(key_hash)


class SecurityHeaders:
    """Security headers middleware."""
    
    @staticmethod
    def add_security_headers(response):
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response


# IP-based rate limiting for additional security
class IPRateLimiter:
    """IP-based rate limiter for additional protection."""
    
    def __init__(self):
        self.ip_requests: Dict[str, deque] = defaultdict(deque)
        self.blocked_ips: Set[str] = set()
        self.block_duration = 3600  # 1 hour
        self.ip_blocks: Dict[str, float] = {}
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        now = time.time()
        
        # Check if IP is currently blocked
        if ip_address in self.ip_blocks:
            if now - self.ip_blocks[ip_address] < self.block_duration:
                return False
            else:
                # Unblock IP
                del self.ip_blocks[ip_address]
                self.blocked_ips.discard(ip_address)
        
        # Clean old requests
        window_start = now - 300  # 5 minutes
        while (self.ip_requests[ip_address] and 
               self.ip_requests[ip_address][0] <= window_start):
            self.ip_requests[ip_address].popleft()
        
        # Check if too many requests from this IP
        if len(self.ip_requests[ip_address]) >= 200:  # 200 requests per 5 minutes
            logger.warning(f"Blocking IP {ip_address} for excessive requests")
            self.blocked_ips.add(ip_address)
            self.ip_blocks[ip_address] = now
            return False
        
        # Record this request
        self.ip_requests[ip_address].append(now)
        return True


# Global IP rate limiter
ip_rate_limiter = IPRateLimiter()


def check_ip_rate_limit(ip_address: str):
    """Check IP-based rate limit."""
    if not ip_rate_limiter.is_ip_allowed(ip_address):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="IP rate limit exceeded",
            headers={"Retry-After": "3600"}
        )
