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
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "50051")  # Default gRPC port
logger.info(f"Connecting to Weaviate gRPC Port {WEAVIATE_GRPC_PORT}")

try:
    weaviate_client = WeaviateClient(
        connection_params=weaviate.connect.ConnectionParams.from_url(
            url=WEAVIATE_URL,
            grpc_port=WEAVIATE_GRPC_PORT  # Required in v4
        )
    )
    client = weaviate_client.connect()
    logger.info(f"Successfully connected to Weaviate at {WEAVIATE_URL} (gRPC: {WEAVIATE_GRPC_PORT})")
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
    logger.info(f"[api_route]path: {path}")

    try:
        if path == "api/chat":
            return await handle_rag_request(request)
        return await forward_request(request, path)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def insert_rag_context(messages, rag_context):
    # Find the index of the last user message (which is the last question)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] == 'user':
            # Insert RAG context just before the last user question
            messages.insert(i, {'role': 'user', 'content': rag_context})
            break
    return messages

async def handle_rag_request(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "")
    logger.info(f"[handle_rag_request]data: {data}")
    logger.info(f"[handle_rag_request]user_prompt: {user_prompt}")
    try:
        # results = client.collections.get("Document").query.near_text(
        #     query=user_prompt,
        #     limit=3
        # )
        # context = "\n".join([obj.properties["content"] for obj in results.objects])
        rag_context = f"rag context"
        insert_rag_context(data['messages'], rag_context)
        logger.info(f"[handle_rag_request]enhanced request:\n{data}")
        async with httpx.AsyncClient() as http_client:
            ollama_response = await http_client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={**data},
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
            
            logger.info(f"[forward_request]url: {url}")
            logger.info(f"[forward_request]headers: {headers}")
            logger.info(f"[forward_request]request: {request.method} {request.body()}")
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