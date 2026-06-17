
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="../chroma_db"
)

collection = client.get_collection("plants")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

query = input("Enter query: ")

query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=20
)

print("\n" + "=" * 50)
print("SEARCH RESULTS")
print("=" * 50)

seen = set()
count = 0

for metadata, document in zip(
    results["metadatas"][0],
    results["documents"][0]
):

    title = metadata["title"]

    if title in seen:
        continue

    seen.add(title)
    count += 1

    print(f"\nResult {count}")
    print(f"Title: {title}")
    print(f"Category: {metadata['category']}")

    print("\nPreview:")
    print(document[:500])

    print("\n" + "-" * 50)

    if count == 5:
        break

