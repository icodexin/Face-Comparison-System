from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse
from starlette import status
from typing import List

from global_var import face_recognizer
from sqlalchemy.orm import Session
from .user_router import get_current_user, get_db
from .utils import bytes_to_numpy, dict_to_list
from .db import models, crud
import base64

ai_router = APIRouter()


@ai_router.post('/detect_image')
async def detect_image(file1: UploadFile = File(), file2: UploadFile = File()):
    if not file1.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File 1 isn't an image.")
    if not file2.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File 2 isn't an image.")

    f1 = await file1.read()
    f2 = await file2.read()
    image1 = bytes_to_numpy(f1)
    image2 = bytes_to_numpy(f2)
    ret = face_recognizer.detect_image(image1, image2)
    if ret['success'] == False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ret, headers={"X-Error": "Invalid Image"})
    else:
        return ret
    
@ai_router.post('/detect_image_me')
async def detect_image_me(file: UploadFile = File(), current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File isn't an image.")
    
    f = await file.read()
    image = bytes_to_numpy(f)
    user_data = crud.get_user_data(db=db, user_id=current_user.id)
    user_encoding = dict_to_list(eval(user_data.encoding))
    ret = face_recognizer.detect_image_encoding(image, user_encoding)
    if ret['success'] == False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ret, headers={"X-Error": "Invalid Image"})
    else:
        return ret