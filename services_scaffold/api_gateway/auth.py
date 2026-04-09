from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from api_gateway import schemas, config
import httpx

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

async def authenticate_user(email: str, password: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{config.USER_PROFILE_SERVICE_URL}/auth/verify",
                json={"email": email, "password": password}
            )
            response.raise_for_status()
            user_data = response.json()
            return user_data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return None
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"User service error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to user service: {e}")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if email is None or user_id is None:
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
    except JWTError:
        raise credentials_exception

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{config.USER_PROFILE_SERVICE_URL}/users/{user_id}")
            response.raise_for_status()
            user_data = response.json()
            
            # SEC-003: Check if user is active
            if not user_data.get('is_active'):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is deactivated")

            return schemas.UserResponse(**user_data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise credentials_exception
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"User service error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to user service: {e}")
