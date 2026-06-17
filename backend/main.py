
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from rag import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection("plants")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


class SearchRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "PlantWise API Running"
    }


@app.post("/search")
def search(data: SearchRequest):

    query_embedding = model.encode(data.query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10
    )

    unique_results = []
    seen = set()

    for metadata, document in zip(
        results["metadatas"][0],
        results["documents"][0]
    ):

        title = metadata["title"]

        if title in seen:
            continue

        seen.add(title)

        unique_results.append({
            "title": title,
            "category": metadata["category"],
            "content": document[:1000]
        })

        if len(unique_results) == 5:
            break

    context = "\n\n".join(
        [result["content"] for result in unique_results]
    )

    answer = generate_answer(
        data.query,
        context
    )

    return {
        "query": data.query,
        "answer": answer,
        "sources": [
            {
                "title": result["title"],
                "category": result["category"]
            }
            for result in unique_results
        ]
    }
