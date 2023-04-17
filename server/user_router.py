import json
from datetime import timedelta, datetime
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from .db.database import SessionLocal
from .db import schemas, crud, models
from . import utils
from global_var import face_recognizer

user_router = APIRouter()

SECRET_KEY = "a4bbe5d51e83bc2f3607fa789e2724d2e833bc73ec37690459bd50257042fc0c"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@user_router.post('/register', response_model=schemas.ReadUser)
def register(user_info: schemas.CreateUserWithImage, db: Session = Depends(get_db)):
    """用户注册"""
    # 检验用户是否已存在
    db_user = crud.get_user_by_name(db=db, name=user_info.name)
    if db_user:
        raise HTTPException(status_code=400, detail="用户名已存在。")
    user_info.password = get_password_hash(user_info.password)

    # 检测图片是否有人脸
    image_bytes = utils.base64_to_bytes(user_info.image)
    image_ndarray = utils.bytes_to_numpy(image_bytes)
    res = face_recognizer.get_face_encoding(image_ndarray)
    if not res['success']:
        raise HTTPException(status_code=400, detail=res['error_info'])

    bbox = utils.list_to_dict(res['bbox'])
    encoding = utils.list_to_dict(res['encoding'])

    user_to_create = schemas.CreateUser(name=user_info.name, password=user_info.password)
    user_created = crud.create_user(db=db, user=user_to_create)

    user_data_to_create = schemas.CreateUserData(
        user_id=user_created.id,
        image=image_bytes,
        bbox=json.dumps(bbox),
        encoding=json.dumps(encoding)
    )

    user_data_created = crud.create_user_data(db=db, user_data=user_data_to_create)

    return user_created


def authenticate_user(username: str, password: str, db: Session = Depends(get_db)):
    """用户认证"""
    user = crud.get_user_by_name(name=username, db=db)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = crud.get_user_by_name(db=db, name=username)
    if user is None:
        raise credentials_exception
    return user


@user_router.post('/login', response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.name}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "Bearer"}


@user_router.get("/me", response_model=schemas.ReadUser)
def read_user_me(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_data = crud.get_user_data(db=db, user_id=current_user.id)
    user_image_base64 = utils.bytes_to_base64(user_data.image)
    me = schemas.ReadUser(
        id=current_user.id,
        name=current_user.name,
        image=user_image_base64,
    )
    return me


@user_router.put("/update_password")
def update_password(request: schemas.UpdatePasswordRequest, current_user: models.User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    if not verify_password(request.old_password, current_user.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid old password.")

    # Update user's password
    current_user.password = get_password_hash(request.new_password)
    db.commit()

    return {"message": 'Password updated successfully'}


@user_router.put('/update_image')
def update_image(request: schemas.UpdateImageRequest, current_user: models.User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    user_data = crud.get_user_data(db, current_user.id)
    user_data.image = utils.base64_to_bytes(request.image_base64)
    db.commit()

    return {"message": "Image updated successfully"}

