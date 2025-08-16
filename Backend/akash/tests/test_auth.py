"""
Tests for Supabase authentication module
"""

import pytest
from fastapi import HTTPException
from auth.supabase_auth import SupabaseAuth, verify_token

class TestSupabaseAuth:
    
    def test_init_with_valid_env(self):
        """Test SupabaseAuth initialization with valid environment variables"""
        auth = SupabaseAuth()
        assert auth.jwt_secret == "test-secret-key"
        assert auth.supabase_url == "https://test.supabase.co"
    
    def test_verify_valid_token(self, valid_jwt_token):
        """Test token verification with valid token"""
        auth = SupabaseAuth()
        result = auth.verify_token(valid_jwt_token)
        
        assert result["user_id"] == "test-user-123"
        assert result["email"] == "test@example.com"
        assert "exp" in result
        assert "raw_payload" in result
    
    def test_verify_token_with_bearer_prefix(self, valid_jwt_token):
        """Test token verification with Bearer prefix"""
        auth = SupabaseAuth()
        bearer_token = f"Bearer {valid_jwt_token}"
        result = auth.verify_token(bearer_token)
        
        assert result["user_id"] == "test-user-123"
    
    def test_verify_expired_token(self, expired_jwt_token):
        """Test token verification with expired token"""
        auth = SupabaseAuth()
        
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token(expired_jwt_token)
        
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()
    
    def test_verify_invalid_token(self, invalid_jwt_token):
        """Test token verification with invalid token"""
        auth = SupabaseAuth()
        
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token(invalid_jwt_token)
        
        assert exc_info.value.status_code == 401
        assert "invalid token" in exc_info.value.detail.lower()
    
    def test_verify_token_missing_user_id(self):
        """Test token verification with missing user ID"""
        from jose import jwt
        from datetime import datetime, timedelta
        
        payload = {
            "email": "test@example.com",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "aud": "authenticated"
            # Missing "sub" field
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        auth = SupabaseAuth()
        
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token(token)
        
        assert exc_info.value.status_code == 401
        assert "missing user ID" in exc_info.value.detail

class TestAuthDependencies:
    
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, valid_jwt_token):
        """Test get_current_user dependency with valid token"""
        from auth.supabase_auth import get_current_user
        
        authorization = f"Bearer {valid_jwt_token}"
        result = await get_current_user(authorization)
        
        assert result["user_id"] == "test-user-123"
        assert result["email"] == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_get_current_user_no_header(self):
        """Test get_current_user dependency without authorization header"""
        from auth.supabase_auth import get_current_user
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None)
        
        assert exc_info.value.status_code == 401
        assert "authorization header is required" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, invalid_jwt_token):
        """Test get_current_user dependency with invalid token"""
        from auth.supabase_auth import get_current_user
        
        authorization = f"Bearer {invalid_jwt_token}"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(authorization)
        
        assert exc_info.value.status_code == 401
