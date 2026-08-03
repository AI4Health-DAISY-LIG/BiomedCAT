from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

from biomedcat.pipeline import process_file
from biomedcat.sse_broker import sse_broker

app = FastAPI(title="BiomedCAT API")

class ProcessRequest(BaseModel):
    file_path: str
    model_id: Optional[str] = "gemma4:12b-it-qat"

@app.post("/process", status_code=202)
async def start_processing(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Lance le pipeline de traitement pour un fichier donné en arrière-plan.
    Retourne immédiatement un code 202 (Accepted).
    """
    try:
        # On ajoute la tâche au gestionnaire de tâches en arrière-plan de FastAPI.
        # Comme process_file est une fonction synchrone, FastAPI l'exécutera 
        # dans un thread séparé du loop d'événements principal.
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

@app.get("/events")
async def sse_endpoint():
    """
    Endpoint SSE pour diffuser les événements du pipeline (OCR, NER, Norm) 
    vers le frontend en temps réel.
    """
    return StreamingResponse(
        sse_broker.event_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
