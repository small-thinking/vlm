#!/usr/bin/env python3
"""LLaVA Inference Service.

Example commands:
    # Run with default checkpoint
    python -m vlm.inference.inference_service

    # Run with custom checkpoint
    python -m vlm.inference.inference_service --checkpoint ~/models/llava/checkpoint_phase1.pt

    # Run on specific device
    python -m vlm.inference.inference_service --device cuda --port 8080

    # Test inference API (with image - path in ~/dataset)
    curl -X POST "http://localhost:8000/infer" -H "Content-Type: application/json" -d '{"text": "What is in this image?", "image_path": "~/dataset/llava-pretrain/00000/000000012.jpg"}'
    
    # Test inference API (with image - absolute path)
    curl -X POST "http://localhost:8000/infer" -H "Content-Type: application/json" -d '{"text": "What is in this image?", "image_path": "/absolute/path/to/image.jpg"}'
    
    # Test inference API (text only)
    curl -X POST "http://localhost:8000/infer" -H "Content-Type: application/json" -d '{"text": "Hello, how are you?"}'
    
    # Test inference API (with max_tokens)
    curl -X POST "http://localhost:8000/infer" -H "Content-Type: application/json" -d '{"text": "What is in this image?", "image_path": "~/dataset/llava-pretrain/00000/000000012.jpg", "max_tokens": 256}'
    
    # Test streaming inference API (Server-Sent Events)
    curl -N -X POST "http://localhost:8000/infer/stream" -H "Content-Type: application/json" -d '{"text": "What is in this image?", "image_path": "~/dataset/llava-pretrain/00000/000000012.jpg", "max_tokens": 32}'

    # Health check
    curl http://localhost:8000/health
"""
import argparse
import json
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, AsyncIterator
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .model_loader import load_model_from_checkpoint
from .inference import generate_response, generate_response_stream
from ..models.llava import LLaVAModel


class InferenceRequest(BaseModel):
    """Request model for inference API.

    image_path can be:
    - Absolute path: /absolute/path/to/image.jpg
    - Path with ~ expansion: ~/dataset/llava-pretrain/00000/image.jpg
    """
    image_path: Optional[str] = None
    text: str = ""
    temperature: Optional[float] = None
    seed: Optional[int] = None
    use_greedy: Optional[bool] = None
    max_tokens: Optional[int] = 50


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
    # Resolve project root for relative path resolution
    project_root = Path(__file__).parent.parent.parent.parent
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for model loading."""
        nonlocal model
        # Expand ~ if present in checkpoint path
        expanded_path = Path(checkpoint_path).expanduser()
        if not expanded_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {expanded_path}")
        model = load_model_from_checkpoint(str(expanded_path), device=device)
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
        # Log request received with timestamp
        request_time = datetime.now()
        print(f"[{request_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Received inference request")
        print(f"  - Text: {request.text[:100]}{'...' if len(request.text) > 100 else ''}")
        print(f"  - Image path: {request.image_path or 'None'}")
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Resolve image path: accepts absolute paths, paths with ~ expansion, or relative paths to repo root
        image_path = None
        if request.image_path:
            path_input = Path(request.image_path).expanduser()  # Expand ~ to home directory
            if path_input.is_absolute():
                # Use absolute path as-is (including expanded ~ paths)
                image_path = path_input
            else:
                # Resolve relative path from project root
                image_path = project_root / request.image_path
            
            if not image_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Image not found: {image_path}"
                )
            image_path = str(image_path)
        
        try:
            # Use request parameters or defaults
            kwargs = {
                "model": model,
                "image_path": image_path,
                "text": request.text,
                "device": device,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.seed is not None:
                kwargs["seed"] = request.seed
            if request.use_greedy is not None:
                kwargs["use_greedy"] = request.use_greedy
            if request.max_tokens is not None:
                kwargs["max_new_tokens"] = request.max_tokens

            # Generate response and log timing
            inference_start = datetime.now()
            print(f"[{inference_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting inference...")
            
            response = generate_response(**kwargs)
            
            inference_end = datetime.now()
            elapsed = (inference_end - inference_start).total_seconds()
            print(f"[{inference_end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Inference completed in {elapsed:.2f}s")
            print(f"  - Response length: {len(response)} characters")
            
            return InferenceResponse(response=response)
        except Exception as e:
            error_time = datetime.now()
            print(f"[{error_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Error during inference: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/infer/stream")
    async def infer_stream(request: InferenceRequest) -> StreamingResponse:
        """Run streaming inference on image and text.
        
        Returns a Server-Sent Events (SSE) stream of tokens as they are generated.
        
        Args:
            request: Inference request with image_path and text
            
        Returns:
            Streaming response with generated tokens
        """
        # Log request received with timestamp
        request_time = datetime.now()
        print(f"[{request_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Received streaming inference request")
        print(f"  - Text: {request.text[:100]}{'...' if len(request.text) > 100 else ''}")
        print(f"  - Image path: {request.image_path or 'None'}")
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Resolve image path: accepts absolute paths, paths with ~ expansion, or relative paths to repo root
        image_path = None
        if request.image_path:
            path_input = Path(request.image_path).expanduser()  # Expand ~ to home directory
            if path_input.is_absolute():
                # Use absolute path as-is (including expanded ~ paths)
                image_path = path_input
            else:
                # Resolve relative path from project root
                image_path = project_root / request.image_path
            
            if not image_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Image not found: {image_path}"
                )
            image_path = str(image_path)
        
        # Prepare generation parameters
        kwargs = {
            "model": model,
            "image_path": image_path,
            "text": request.text,
            "device": device,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.use_greedy is not None:
            kwargs["use_greedy"] = request.use_greedy
        if request.max_tokens is not None:
            kwargs["max_new_tokens"] = request.max_tokens
        
        async def generate_stream() -> AsyncIterator[str]:
            """Async generator that yields SSE-formatted tokens."""
            inference_start = datetime.now()
            print(f"[{inference_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting streaming inference...")
            
            # Use a queue to communicate between threads
            token_queue: queue.Queue = queue.Queue()
            exception_holder = [None]
            
            def run_generator():
                """Run the synchronous generator in a background thread."""
                try:
                    for token in generate_response_stream(**kwargs):
                        token_queue.put(("token", token))
                    token_queue.put(("done", None))
                except Exception as e:
                    exception_holder[0] = e
                    token_queue.put(("error", str(e)))
            
            # Start the generator in a background thread
            thread = threading.Thread(target=run_generator, daemon=True)
            thread.start()
            
            try:
                while True:
                    # Wait for next token with timeout to allow checking thread status
                    try:
                        item_type, item_data = token_queue.get(timeout=0.1)
                    except queue.Empty:
                        # Check if thread is still alive
                        if not thread.is_alive() and exception_holder[0] is None:
                            # Thread finished but no done signal - might be an issue
                            break
                        continue
                    
                    if item_type == "token":
                        # Send just the new token - client will append it
                        data = json.dumps({"token": item_data, "done": False})
                        yield f"data: {data}\n\n"
                    elif item_type == "done":
                        # Send completion signal
                        completion_data = json.dumps({"token": "", "done": True})
                        yield f"data: {completion_data}\n\n"
                        break
                    elif item_type == "error":
                        raise Exception(item_data)
                
                inference_end = datetime.now()
                elapsed = (inference_end - inference_start).total_seconds()
                print(f"[{inference_end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Streaming inference completed in {elapsed:.2f}s")
            except Exception as e:
                error_time = datetime.now()
                error_msg = str(exception_holder[0]) if exception_holder[0] else str(e)
                print(f"[{error_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Error during streaming inference: {error_msg}")
                error_data = json.dumps({"error": error_msg, "done": True})
                yield f"data: {error_data}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "model_loaded": model is not None}
    
    return app


def main():
    """Run the API server."""
    # Default checkpoint path in ~/models/llava
    default_checkpoint = Path.home() / "models" / "llava" / "checkpoint_phase1.pt"
    
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
    
    # Validate checkpoint exists (expand ~ if present)
    checkpoint_path = Path(args.checkpoint).expanduser()
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

