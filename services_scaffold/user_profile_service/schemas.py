from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional
from user_profile_service.models import GenderEnum

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "user_profile_service"

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserInDB(UserBase):
    id: int
    hashed_password: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserLoginInternal(BaseModel):
    email: EmailStr
    password: str

class UserAuthResponseInternal(BaseModel):
    id: int
    email: EmailStr
    is_active: bool

    class Config:
        from_attributes = True

class UserProfileBase(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[GenderEnum] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    primary_sport: Optional[str] = None
    # New fields for injury history/status
    injury_status: Optional[str] = None
    injury_date: Optional[datetime] = None
    recovery_date: Optional[datetime] = None

class UserProfileCreate(UserProfileBase):
    pass

class UserProfileUpdate(UserProfileBase):
    pass

class UserProfileResponse(UserProfileBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
