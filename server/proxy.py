from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx
import weaviate
from weaviate import WeaviateClient
import os
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import traceback

logger = None
ollama_url = None
encoder = None
# Load environment variables
load_dotenv()

weaviate_client = None
def weaviate_connect() -> WeaviateClient:
    """
    Ensures that the connection is established only once.
    """    
    try:
        logger.info(f"Connecting to Weaviate")
        client = None
        if client is None:
            # Initialize the Weaviate client (this is just an example URL)
            client = weaviate.connect_to_custom(
                http_host=os.getenv("WEAVIATE_HTTP_HOST", "localhost"),
                http_port=os.getenv("WEAVIATE_HTTP_PORT", "8080"),
                http_secure=False,  # Use True for HTTPS
                grpc_host=os.getenv("WEAVIATE_GRPC_HOST", "localhost"),
                grpc_port=os.getenv("WEAVIATE_GRPC_PORT", "50051"),
                grpc_secure=False  # Use True for secure gRPC
            )
            # Check if connection is ready
            if client.is_ready():
                logger.info(f"Successfully connected to Weaviate {client.get_meta()})")
            else:
                raise Exception("Weaviate connection is not ready.")
    
    except Exception as e:
        # Let all other exceptions propagate
        logger.error(f"An error occurred: {str(e)}")
        raise

    return client

async def startup():
    global logger
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    global ollama_url
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    global weaviate_client
    weaviate_client = weaviate_connect()

    # Test Ollama connection
    logger.info(f"Connecting to Ollama at {ollama_url}")
    try:
        async with httpx.AsyncClient() as test_client:
            response = await test_client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
        logger.info("Successfully connected to Ollama")
    except Exception as e:
        logger.error(f"Ollama connection test failed: {str(e)}")
        raise
    
    global encoder
    encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device='cuda')


def shutown():
    if weaviate_client:
        weaviate_client.close()
    
# Define a custom lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    shutown()

app = FastAPI(lifespan=lifespan)

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


async def handle_rag_request(request: Request):
    data = await request.json()
    messages = data['messages']
    if not messages or 'content' not in messages[-1]:
        raise HTTPException(status_code=400, detail="Invalid request format")

    url = f"{ollama_url}/api/chat"
    headers = dict(request.headers)
    headers.pop("host", None)
            
    prompt = messages[-1]['content']
    logger.info(f"[handle_rag_request]headers: {headers}")
    logger.info(f"[handle_rag_request]data: {data}")
    logger.info(f"[handle_rag_request]user_prompt: {prompt}")
    try:
        prompt_vector = encoder.encode(prompt)
        results = weaviate_client.collections.get("TextChunk").query.near_vector(
            near_vector=prompt_vector.tolist(),
            limit=3
        )

        if not results.objects:
            context = "无相关文章内容。"
        else:
            context = "\n".join([obj.properties["content"] for obj in results.objects])

        logger.info(f"[handle_rag_request]original message:\n{messages}")
        new_messages = [*messages[:-1], {"role": "user", "content": f"模拟下面文章人物回答问题，只要答案，不要分析过程：\n{context}\n问题：{prompt}"}]
        logger.info(f"[handle_rag_request]new message:\n{new_messages}")
        data['messages'] = new_messages
        logger.info(f"[handle_rag_request]new data: {data}")

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http_client:
            ollama_response = await http_client.post(url, json=data)
            ollama_response.raise_for_status()
            
            async def chat_response():
                async for chunk in ollama_response.aiter_bytes():
                    yield chunk
                    
            return StreamingResponse(chat_response())
    except Exception as e:
        logger.error(f"RAG processing failed: {type(e)} {str(e)}")
        logger.error(traceback.format_exc())
        logger.error(f"Response body: {await e.response.text()}")  # Log the response body for more details
        raise

async def forward_request(request: Request, path: str):
    try:
        async with httpx.AsyncClient() as http_client:
            url = f"{ollama_url}/{path}"
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
            response = await client.get(f"{ollama_url}/api/tags", timeout=5.0)
            return response.status_code == 200
    except:
        return False