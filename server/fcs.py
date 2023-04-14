import logging
import logging.handlers
from typing import List

import cv2
import numpy as np
import uvicorn
from PIL.Image import Image
from fastapi import FastAPI, UploadFile, File, Body, HTTPException, status
from pydantic import BaseModel
from config import fastapi_config, fastapi_log_fmt, log_dirpath, server_config
from global_var import face_recognizer

# 实例化web server app
app = FastAPI(
    title=fastapi_config['title'],
    description=fastapi_config['description'],
    version=fastapi_config['version'],
)

# 设置FastAPI的日志格式
log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = fastapi_log_fmt
log_config["formatters"]["default"]["fmt"] = fastapi_log_fmt


@app.on_event("startup")
async def startup_event() -> None:
    """记录FastAPI日志到文件"""
    loggers = [
        logging.getLogger('uvicorn.error'),
        logging.getLogger('uvicorn.access'),
        logging.getLogger('uvicorn.asgi')
    ]
    for logger in loggers:
        handler = logging.handlers.TimedRotatingFileHandler(
            f"{log_dirpath}/fcs.log",
            when='midnight',
            backupCount=7,
            encoding='utf-8'
        )
        handler.suffix = '%Y-%m-%d.log'
        handler.setFormatter(logging.Formatter(fastapi_log_fmt))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)


@app.get('/')
def app_root():
    return {
        "api_doc_url": fastapi_config['docs_url']
    }


def bytes_to_numpy(image_bytes):
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


@app.post('/detect_image')
async def detect_image(file1: UploadFile = File(), file2: UploadFile = File()):
    if not file1.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File 1 is't an image.")
    if not file2.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File 1 is't an image.")
    
    f1 = await file1.read()
    f2 = await file2.read()
    image1 = bytes_to_numpy(f1)
    image2 = bytes_to_numpy(f2)
    ret = face_recognizer.detect_image(image1, image2)
    if ret['success'] == False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ret, headers={"X-Error": "Invalid Image"})
    else:
        return ret
