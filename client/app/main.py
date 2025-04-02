from fastapi import FastAPI
import weaviate
import httpx

app = FastAPI()
weaviate_client = weaviate.Client("http://weaviate:8080")
LLM_URL = "http://llm:5000"

@app.get("/ask")
async def ask(question: str):
    # Search
    vector = encoder.encode(question)
    contexts = weaviate_client.query.get(
        "NovelChunk",
        ["content", "novel"]
    ).with_near_vector({
        "vector": vector
    }).with_limit(3).do()

    # Generate
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{LLM_URL}/generate",
            json={
                "prompt": f"根据以下内容回答：{contexts}\n\n问题：{question}",
                "max_tokens": 300
            }
        )
    
    return {"answer": response.json()["text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)