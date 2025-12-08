"""
FastAPI Backend for Unified AI System
Exposes chat and hardware identification via REST API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_setup import (
    setup_gemini, setup_chat_model, setup_vision_model,
    classify_hardware, unified_chat_response, HARDWARE_CLASSES
)
import torch
import os
from PIL import Image
import io

app = FastAPI(title="AI Hardware Assistant API", version="1.0")

# Enable CORS for Vue frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {
    "gemini": None,
    "phi2_model": None,
    "phi2_tokenizer": None,
    "vision_model": None,
    "vision_processor": None,
    "vision_device": None,
    "conversation_histories": {}  # Store per-session history
}

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    source: str  # 'gemini', 'phi2', or 'predefined'
    session_id: str

class HardwareResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_valid: bool
    top3: list
    all_probabilities: dict

class StatusResponse(BaseModel):
    gemini_available: bool
    phi2_loaded: bool
    vision_model_loaded: bool
    vision_accuracy: float
    gpu_name: str
    device: str

@app.on_event("startup")
async def load_models():
    """Load all AI models on startup"""
    print("=" * 80)
    print("🚀 Starting AI Hardware Assistant API")
    print("=" * 80)
    print("📚 Loading AI models...")
    
    # Load Gemini
    print("⏳ Loading Gemini 2.5 Flash...")
    models["gemini"] = setup_gemini()
    if models["gemini"]:
        print("✅ Gemini AI ready!")
    else:
        print("⚠️ Gemini unavailable (check API key)")
    
    # Load Phi-2
    print("⏳ Loading Phi-2 (2.7B params)...")
    models["phi2_model"], models["phi2_tokenizer"] = setup_chat_model()
    print("✅ Phi-2 model ready!")
    
    # Load Vision Model
    print("⏳ Loading Vision Transformer...")
    vision_model, vision_processor, vision_device = setup_vision_model()
    models["vision_model"] = vision_model
    models["vision_processor"] = vision_processor
    models["vision_device"] = vision_device
    
    if vision_model and os.path.exists("models/best_vit_model.pth"):
        print("✅ Vision model loaded! (82.50% accuracy)")
    else:
        print("⚠️ Vision model not found (train first)")
    
    print("=" * 80)
    print("✅ All models loaded successfully!")
    print("📡 API ready at http://localhost:8000")
    print("📖 Documentation at http://localhost:8000/docs")
    print("=" * 80)

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "message": "AI Hardware Assistant API - Ready to serve!",
        "version": "1.0",
        "endpoints": {
            "chat": "POST /chat",
            "identify": "POST /identify",
            "status": "GET /status",
            "clear_history": "POST /clear/{session_id}",
            "docs": "/docs"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with automatic Gemini/Phi-2 routing"""
    try:
        # Get conversation history for this session
        history = models["conversation_histories"].get(request.session_id, "")
        
        # Generate response
        response, source = unified_chat_response(
            models["gemini"],
            models["phi2_model"],
            models["phi2_tokenizer"],
            request.message,
            history
        )
        
        # Update conversation history
        history += f"\nUser: {request.message}\nAssistant: {response}"
        
        # Keep only last 6 exchanges (12 messages total)
        history_parts = history.split("\nUser:")
        if len(history_parts) > 6:
            history = "\nUser:".join(history_parts[-6:])
        
        models["conversation_histories"][request.session_id] = history
        
        return ChatResponse(
            response=response,
            source=source,
            session_id=request.session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/identify", response_model=HardwareResponse)
async def identify(file: UploadFile = File(...)):
    """Hardware identification endpoint"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save temporarily
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        
        # Classify
        predicted, confidence, is_valid, all_probs = classify_hardware(
            temp_path,
            models["vision_model"],
            models["vision_processor"],
            models["vision_device"]
        )
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if predicted is None:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Get top 3
        top_3_indices = all_probs.argsort()[-3:][::-1]
        top3 = [
            {
                "class": HARDWARE_CLASSES[idx],
                "confidence": float(all_probs[idx] * 100)
            }
            for idx in top_3_indices
        ]
        
        # All probabilities
        all_probabilities = {
            HARDWARE_CLASSES[i]: float(prob * 100)
            for i, prob in enumerate(all_probs)
        }
        
        return HardwareResponse(
            predicted_class=predicted,
            confidence=float(confidence * 100),
            is_valid=is_valid,
            top3=top3,
            all_probabilities=all_probabilities
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification error: {str(e)}")

@app.post("/clear/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in models["conversation_histories"]:
        del models["conversation_histories"][session_id]
        return {"message": "History cleared", "session_id": session_id}
    return {"message": "No history found", "session_id": session_id}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    vision_trained = os.path.exists("models/best_vit_model.pth")
    accuracy = 82.50 if vision_trained else 0.0
    
    return StatusResponse(
        gemini_available=models["gemini"] is not None,
        phi2_loaded=models["phi2_model"] is not None,
        vision_model_loaded=vision_trained,
        vision_accuracy=accuracy,
        gpu_name=gpu_name,
        device=device
    )

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting AI Hardware Assistant API Server...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
