#!/usr/bin/env python3
"""LLaVA Inference Service.

Example commands:
    # Run with default checkpoint
    uv run python -m vlm.inference.inference_service

    # Run with custom checkpoint
    uv run python -m vlm.inference.inference_service \\
        --checkpoint checkpoints/custom.pt

    # Run on specific device
    uv run python -m vlm.inference.inference_service --device cuda --port 8080

    # Test inference API
    curl -X POST "http://localhost:8000/infer" -H "Content-Type: application/json" -d '{ "text": "What is in this image?"}'

    # Health check
    curl http://localhost:8000/health
"""
import argparse
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model_loader import load_model_from_checkpoint
from .inference import generate_response
from ..models.llava import LLaVAModel


class InferenceRequest(BaseModel):
    """Request model for inference API."""
    image_path: Optional[str] = None
    text: str = ""


class InferenceResponse(BaseModel):
    """Response model for inference API."""
    response: str


def create_app(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> FastAPI:
    """Create FastAPI app with loaded model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file (not used yet)
        device: Device to run inference on
        
    Returns:
        FastAPI app instance
    """
    model: Optional[LLaVAModel] = None
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for model loading."""
        nonlocal model
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, device=device)
        yield
        # Cleanup if needed
    
    app = FastAPI(title="LLaVA Inference API", lifespan=lifespan)
    
    @app.post("/infer", response_model=InferenceResponse)
    async def infer(request: InferenceRequest) -> InferenceResponse:
        """Run inference on image and text.
        
        Args:
            request: Inference request with image_path and text
            
        Returns:
            Generated response
        """
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate image path if provided
        if request.image_path and not Path(request.image_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Image not found: {request.image_path}"
            )
        
        try:
            response = generate_response(
                model=model,
                image_path=request.image_path,
                text=request.text,
                device=device,
            )
            return InferenceResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "model_loaded": model is not None}
    
    return app


def main():
    """Run the API server."""
    # Default checkpoint path (relative to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    default_checkpoint = project_root / "checkpoints" / "checkpoint_final.pt"
    
    parser = argparse.ArgumentParser(description="LLaVA Inference API")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_checkpoint),
        help=f"Path to model checkpoint (default: {default_checkpoint})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to run inference on (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Please provide a valid checkpoint path with --checkpoint")
        return 1
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    print(f"Loading model from {checkpoint_path} on {device}")
    app = create_app(str(checkpoint_path), device=device)
    
    print(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

