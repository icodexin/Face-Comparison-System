import logging
import logging.handlers

import uvicorn
from fastapi import FastAPI, UploadFile, File, Body, HTTPException, status
from fastapi.responses import HTMLResponse
from config import fastapi_config, fastapi_log_fmt, log_dirpath, server_config
from .ai_router import ai_router
from .user_router import user_router

# 实例化web server app
app = FastAPI(
    title=fastapi_config['title'],
    description=fastapi_config['description'],
    version=fastapi_config['version'],
)

app.include_router(user_router, prefix='/user', tags=['用户管理'])
app.include_router(ai_router, prefix='/ai', tags=['AI模块'])

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


@app.get('/user_agreement')
async def user_agreement():
    with open("server/static/user_agreement.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)
