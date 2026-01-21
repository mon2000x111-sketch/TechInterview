from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import time
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Inference Service", version="1.0.0")

# Request/Response models
class InferenceRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64 encoded image or image URL")
    model_version: Optional[str] = Field("v1", description="Model version to use")

class InferenceResponse(BaseModel):
    result: str
    confidence: float
    processing_time: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    uptime: float
    model_loaded: bool

# Global state
START_TIME = time.time()
MODEL_LOADED = True

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unexpected error processing {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path)
        }
    )

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint with detailed status"""
    try:
        uptime = time.time() - START_TIME
        return {
            "status": "healthy",
            "uptime": round(uptime, 2),
            "model_loaded": MODEL_LOADED
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/infer", response_model=InferenceResponse)
def infer_get(model_version: str = "v1"):
    """GET endpoint for simple inference (for compatibility)"""
    return _perform_inference(model_version)

@app.post("/infer", response_model=InferenceResponse)
def infer_post(request: InferenceRequest):
    """POST endpoint for inference with request body"""
    return _perform_inference(request.model_version)

def _perform_inference(model_version: str = "v1") -> dict:
    """Internal function to perform inference with error handling"""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if not MODEL_LOADED:
            logger.error("Model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate model version
        valid_versions = ["v1", "v2"]
        if model_version not in valid_versions:
            logger.warning(f"Invalid model version requested: {model_version}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model version. Valid versions: {valid_versions}"
            )
        
        logger.info(f"Starting inference with model version: {model_version}")
        
        # Simulate inference processing
        time.sleep(2)
        
        # Simulate occasional errors (10% chance)
        if random.random() < 0.1:
            logger.error("Simulated inference failure")
            raise HTTPException(status_code=500, detail="Inference processing failed")
        
        # Simulate different results based on model version
        results = {
            "v1": ["cat", "dog", "bird"],
            "v2": ["cat", "dog", "bird", "fish", "horse"]
        }
        result = random.choice(results[model_version])
        confidence = round(random.uniform(0.75, 0.99), 2)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Inference completed: {result} (confidence: {confidence}, time: {processing_time:.2f}s)")
        
        return {
            "result": result,
            "confidence": confidence,
            "processing_time": round(processing_time, 3),
            "model_version": model_version
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "Model Inference API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "infer_get": "/infer?model_version=v1",
            "infer_post": "/infer",
            "docs": "/docs"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Model service starting up...")
    # Simulate model loading
    global MODEL_LOADED
    MODEL_LOADED = True
    logger.info("Model loaded successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Model service shutting down...")
    global MODEL_LOADED
    MODEL_LOADED = False