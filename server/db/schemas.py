from pydantic import BaseModel


class CreateUser(BaseModel):
    """创建用户"""
    name: str
    password: str


class CreateUserWithImage(CreateUser):
    """创建带有照片的用户"""
    image: str  # base64


class UserInDB(CreateUser):
    """数据库中的用户"""
    id: int

    class Config:
        orm_mode = True


class ReadUser(BaseModel):
    """读取用户"""
    id: int
    name: str
    image: str  # base64

    class Config:
        orm_mode = True


class CreateUserData(BaseModel):
    user_id: int
    image: bytes
    bbox: str
    encoding: str


class UserDataInDB(CreateUserData):
    """用户相关的数据"""
    id: int

    class Config:
        orm_mode = True


class Token(BaseModel):
    """用户Token模型"""
    access_token: str
    token_type: str


class UpdatePasswordRequest(BaseModel):
    """Request model for updating password"""
    old_password: str
    new_password: str


class UpdateImageRequest(BaseModel):
    image_base64: str
