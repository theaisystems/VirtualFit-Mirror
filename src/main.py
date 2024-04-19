# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:59:16 2023

@author: adeelseraj
"""

from fastapi import FastAPI, Response, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import io
import finalstable_VITON as fv



app = FastAPI(debug=True)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Backend for Tryon"}

@app.post("/uploadimage/")
async def upload_image(cloth: str, file: UploadFile = File(...)):
    # try:
    contents = await file.read()
    print(cloth)
    if ((file.filename).split('.')[-1]).lower() in ['jpg', 'jpeg']:
        try:
            file_path = f'E:\AsadHussain\tryon\{file.filename}'
            result = fv.main_viton(contents, cloth)
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            return JSONResponse(content={"result": result})
        except Exception as e:
            return JSONResponse(content={"message": f"Error: {str(e)}"}, status_code=500)
    else:
        raise HTTPException(status_code=500,
                            detail="Error: Only jpg files acceptable")
if __name__ == "__main__":
    uvicorn.run("main:app", host="192.168.1.10", port=1140, reload=True)
