from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import shutil
import json
from pathlib import Path

from biomedcat.pipeline import process_file
from biomedcat.sse_broker import sse_broker
from biomedcat.config import settings

app = FastAPI(title="BiomedCAT API")

class ProcessRequest(BaseModel):
    file_path: str
    model_id: Optional[str] = "gemma4:12b-it-qat"

@app.post("/process", status_code=202)
async def start_processing(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Launches the processing pipeline for a given file in the background.
    Returns 202 (Accepted) immediately.
    """
    try:
        # Add the task to FastAPI's background task manager. Since process_file is a synchronous function, 
        # FastAPI will execute it in a separate thread from the main event loop.
        background_tasks.add_task(process_file, request.file_path, model_id=request.model_id)
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Processing started",
                "file_path": request.file_path,
                "model_id": request.model_id
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule task: {str(e)}")

@app.post("/upload", status_code=202)
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks, model_id: Optional[str] = "gemma4:12b-it-qat"):
    """
    Receives a file, saves it to the dataset directory, and triggers the pipeline.
    """
    try:
        # Use only the filename to prevent path traversal attacks
        filename = Path(file.filename).name
        dest_path = Path(settings.dataset_path) / filename
        
        # Ensure the destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the uploaded file to disk
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Trigger the pipeline in the background
        background_tasks.add_task(process_file, str(dest_path), model_id=model_id)

        return JSONResponse(
            status_code=202,
            content={
                "message": "File uploaded and processing started",
                "file_path": str(dest_path),
                "model_id": model_id
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.get("/runs")
async def list_runs():
    """
    Lists all completed pipeline runs found in the output directory.
    Returns a list of objects containing the filename and its access URL.
    """
    output_dir = Path(settings.output_path)
    if not output_dir.exists():
        return []

    runs = []
    # Iterate through all .json files in the output directory
    for file_path in output:
        runs.append({
            "filename": file_path.name,
            "url": f"/results/{file_path.name}"
        })
    
    return runs

@app.get("/results/{filename}")
async def get_result(filename: str):
    """
    Retrieves the full JSON content of a specific pipeline run.
    """
    # Use .name to prevent path traversal attacks
    file_path = Path(settings.output_path) / Path(filename).name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found.")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result file: {str(e)}")

@app.get("/events")
async def sse_endpoint():
    """
    SSE endpoint to broadcast pipeline events (OCR, NER, Norm) 
    to the frontend in real-time.
    """
    return StreamingResponse(
        sse_broker.event_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvancorn.run(app, host="0.0.0.0", port=8000)
