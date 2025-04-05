from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx
import weaviate
from weaviate import WeaviateClient
import os
from dotenv import load_dotenv
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Get environment variables with validation
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")

if not OLLAMA_BASE_URL or not WEAVIATE_URL:
    raise ValueError("""
    Missing required environment variables:
    - OLLAMA_BASE_URL must be set
    - WEAVIATE_URL must be set
    """)

logger.info(f"Connecting to Ollama at {OLLAMA_BASE_URL}")
logger.info(f"Connecting to Weaviate at {WEAVIATE_URL}")

# Initialize Weaviate client (v4 syntax)
try:
    weaviate_client = WeaviateClient(
        connection_params=weaviate.connect.ConnectionParams.from_url(WEAVIATE_URL)
    )
    client = weaviate_client.connect()
    logger.info("Successfully connected to Weaviate")
except Exception as e:
    logger.error(f"Weaviate connection failed: {str(e)}")
    raise

@app.on_event("startup")
async def startup():
    if not weaviate_client.is_ready():
        raise RuntimeError("Weaviate connection not ready")
    
    # Test Ollama connection
    try:
        async with httpx.AsyncClient() as test_client:
            response = await test_client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
        logger.info("Successfully connected to Ollama")
    except Exception as e:
        logger.error(f"Ollama connection test failed: {str(e)}")
        raise

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    try:
        if path == "api/generate":
            return await handle_rag_request(request)
        return await forward_request(request, path)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_rag_request(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "")
    
    try:
        results = client.collections.get("Document").query.near_text(
            query=user_prompt,
            limit=3
        )
        context = "\n".join([obj.properties["content"] for obj in results.objects])
        enhanced_prompt = f"Context:\n{context}\n\nQuestion: {user_prompt}"
        
        async with httpx.AsyncClient() as http_client:
            ollama_response = await http_client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={**data, "prompt": enhanced_prompt},
                timeout=60.0
            )
            ollama_response.raise_for_status()
            
            async def generate():
                async for chunk in ollama_response.aiter_bytes():
                    yield chunk
                    
            return StreamingResponse(generate())
    except Exception as e:
        logger.error(f"RAG processing failed: {str(e)}")
        raise

async def forward_request(request: Request, path: str):
    try:
        async with httpx.AsyncClient() as http_client:
            url = f"{OLLAMA_BASE_URL}/{path}"
            headers = dict(request.headers)
            headers.pop("host", None)
            
            if request.method == "GET":
                response = await http_client.get(url, headers=headers)
            elif request.method == "POST":
                response = await http_client.post(url, headers=headers, data=await request.body())
            else:
                raise HTTPException(405, "Method not allowed")
            
            response.raise_for_status()
            
            return StreamingResponse(response.aiter_bytes(), headers=dict(response.headers))
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama API error: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Forward request failed: {str(e)}")
        raise

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "weaviate": weaviate_client.is_ready(),
            "ollama": await check_ollama_health()
        }
    }

async def check_ollama_health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            return response.status_code == 200
    except:
        return False