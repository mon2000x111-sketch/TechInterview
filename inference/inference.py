from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_URL = os.getenv("MODEL_URL", "http://model:9000")

app = FastAPI()

@app.get("/")
def root():
    try:
        logger.info(f"Making request to {MODEL_URL}/infer")
        r = requests.get(f"{MODEL_URL}/infer", timeout=5)
        r.raise_for_status()  # Raise exception for 4xx/5xx status codes
        return r.json()
    
    except requests.exceptions.Timeout:
        error_msg = f"Request to model service timed out after 5 second"
        logger.error(error_msg)
        raise HTTPException(status_code=504, detail=error_msg)
    
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Could not connect to model service at {MODEL_URL}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)
    
    except requests.exceptions.HTTPError as e:
        error_msg = f"Model service returned error: {e.response.status_code} - {e.response.text}"
        logger.error(error_msg)
        raise HTTPException(status_code=e.response.status_code, detail=error_msg)
    
    except requests.exceptions.JSONDecodeError:
        error_msg = "Model service returned invalid JSON"
        logger.error(error_msg)
        raise HTTPException(status_code=502, detail=error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        r = requests.get(f"{MODEL_URL}/infer", timeout=5)
        r.raise_for_status()
        return {"status": "healthy", "model_service": "reachable"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )