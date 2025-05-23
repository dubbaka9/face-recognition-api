from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
import shutil
import os
import uuid

import faceRecognition as fr  # import with alias
import compareFaces as cf  # import with alias

app = FastAPI(title="Face Recognition API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.post("/recognise-faces/")
async def handle_recognise_faces(image: UploadFile = File(...)):
    """
    Recognize faces in an uploaded image
    """
    # Validate file type
    if not image.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400, 
            content={"error": "File must be an image"}
        )
    
    # Generate unique filename to avoid conflicts
    file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
    temp_path = f"temp_recognize_{uuid.uuid4()}.{file_extension}"
    
    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        
        # Process the image
        result = fr.recognise_faces(temp_path)
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/compare-faces/")
async def handle_compare_faces(
    known: UploadFile = File(..., description="Known face image"),
    unknown: UploadFile = File(..., description="Unknown face image to compare")
):
    """
    Compare two face images to determine if they match
    """
    # Validate file types
    if not known.content_type.startswith('image/') or not unknown.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "Both files must be images"}
        )
    
    # Generate unique filenames
    known_ext = known.filename.split('.')[-1] if '.' in known.filename else 'jpg'
    unknown_ext = unknown.filename.split('.')[-1] if '.' in unknown.filename else 'jpg'
    
    known_path = f"temp_known_{uuid.uuid4()}.{known_ext}"
    unknown_path = f"temp_unknown_{uuid.uuid4()}.{unknown_ext}"
    
    try:
        # Save uploaded files temporarily
        with open(known_path, "wb") as f:
            shutil.copyfileobj(known.file, f)
        with open(unknown_path, "wb") as f:
            shutil.copyfileobj(unknown.file, f)
        
        # Compare faces
        result = cf.compare_faces(known_path, unknown_path)
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Comparison failed: {str(e)}"}
        )
    finally:
        # Clean up temp files
        for path in [known_path, unknown_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)