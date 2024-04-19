# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:59:16 2023

"""

from fastapi import FastAPI, Response, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import io
from app.supcodes import s3_access as sa
from app.supcodes import model_op as mo
import logging

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging to write logs to a file
logging.basicConfig(
    filename='app.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log information about the request
    log_message = f"{request.client.host} - \"{request.method} {request.url}\""

    try:
        response = await call_next(request)
        log_message += f" {response.status_code} {response.headers['content-type']}"
        return response
    except HTTPException as e:
        log_message += f" {e.status_code} {e.detail}"
        raise e
    except Exception as e:
        log_message += f" 500 Internal Server Error"
        logging.exception("Internal Server Error")
        raise e
    finally:
        logging.info(log_message)
        # sa.upload_to_s3('logs', open('app.log', "rb").read(), 'app.log')


@app.get("/")
async def root():
    return {"message": "API for Open-Vino"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads an image file.

    Parameters:
    - file: UploadFile - The uploaded file. Acccept jpg/png file only

    Returns:
    - dict: A response indicating the filename, content type and file_key.
    """
    # try:
    contents = await file.read()
    if ((file.filename).split('.')[-1]).lower() in ['jpg', 'jpeg', 'png']:
        file_url, error_msg = sa.upload_to_s3('uploads',
                                              contents,
                                              file.filename)
        if file_url:
            return {"filename": file.filename,
                    "content_type": file.content_type,
                    "file_key": file_url}
        else:
            raise HTTPException(status_code=500,
                                detail=error_msg)
    else:
        raise HTTPException(status_code=500,
                            detail="Error: Only jpg and png files acceptable")


@app.get("/getresults/")
async def get_results(file_key: str):
    try:
        file_bytes, error_msg = sa.download_from_s3(file_key)
        ext = file_key.split('.')[-1]
        file_name = '/'.join(file_key.split('/')[1:])
        file_contents = mo.model_output(file_bytes, ext)
        file_url, error_msg = sa.upload_to_s3('results',
                                              file_contents,
                                              file_name)
        file_bytes, error_msg = sa.download_from_s3(file_url)
        if file_bytes:
            return StreamingResponse(io.BytesIO(file_bytes),
                                     media_type="image/jpeg")
        else:
            raise HTTPException(status_code=500,
                                detail="Failed to retrieve file from S3")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=error_msg)


@app.get("/downloadfile/")
async def download_from_s3(file_key: str):
    file_bytes, error_msg = sa.download_from_s3(file_key)
    ftype = file_key.split('.')[-1]

    if ftype.lower() in ['jpg', 'jpeg']:
        media_type = 'image/jpeg'
    elif ftype.lower() == 'png':
        media_type = 'image/png'
    else:
        error_msg = "Invalid image format. Supported formats: jpg, png"
        raise HTTPException(status_code=400,
                            detail=error_msg)

    if file_bytes:
        response = Response(content=file_bytes, media_type=media_type)
        fname = file_key.split('/')[-1]
        response.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
        return response
    else:
        raise HTTPException(
            status_code=500,
            detail=error_msg)


@app.get("/downloadlogs/")
async def download_logs(file_key: str):
    file_bytes, error_msg = sa.download_from_s3(file_key)

    if file_bytes:
        response = Response(content=file_bytes, media_type='text/plain')
        fname = file_key.split('/')[-1]
        response.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
        return response
    else:
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="192.168.1.10", port=9876, reload=True)
