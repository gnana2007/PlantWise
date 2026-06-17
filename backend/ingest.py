
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

pfaf = pd.read_csv("../data/pfaf.csv").fillna("")
crop = pd.read_csv("../data/crop.csv").fillna("")
indoor = pd.read_csv("../data/indoor.csv").fillna("")

documents = []

for _, row in pfaf.iterrows():
    text = f"""
    Plant Name: {row['Common Name']}
    Scientific Name: {row['Scientific Name']}
    Family: {row['Family']}

    Summary:
    {row['Summary']}

    Medicinal Properties:
    {row['Medicinal Properties']}

    Medicinal Rating:
    {row['Medicinal Rating']}

    Edible Uses:
    {row['Edible Uses']}

    Care Requirements:
    {row['Care Requirements']}

    Cultivation Details:
    {row['Cultivation Details']}

    Native Range:
    {row['Native Range']}

    Known Hazards:
    {row['Known Hazards']}
    """

    documents.append({
        "title": str(row["Common Name"]),
        "category": "medicinal",
        "content": text
    })

for _, row in crop.iterrows():
    text = f"""
    Crop: {row['label']}

    Temperature:
    {row['temperature']}

    Humidity:
    {row['humidity']}

    Rainfall:
    {row['rainfall']}

    Nitrogen:
    {row['N']}

    Phosphorus:
    {row['P']}

    Potassium:
    {row['K']}

    Soil pH:
    {row['ph']}
    """

    documents.append({
        "title": str(row["label"]),
        "category": "crop",
        "content": text
    })

for _, row in indoor.iterrows():
    text = f"""
    Indoor Plant Record

    Health Notes:
    {row['Health_Notes']}

    Watering Amount:
    {row['Watering_Amount_ml']} ml

    Watering Frequency:
    {row['Watering_Frequency_days']} days

    Sunlight Exposure:
    {row['Sunlight_Exposure']}

    Humidity:
    {row['Humidity_%']}

    Soil Type:
    {row['Soil_Type']}

    Health Score:
    {row['Health_Score']}
    """

    documents.append({
        "title": f"Indoor-{row['Plant_ID']}",
        "category": "indoor",
        "content": text
    })

print("Total Documents:", len(documents))

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc["content"] for doc in documents]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

try:
    client.delete_collection("plants")
except:
    pass

collection = client.get_or_create_collection(
    "plants"
)

batch_size = 5000

for start in range(0, len(documents), batch_size):

    end = min(start + batch_size, len(documents))

    collection.add(
        ids=[str(i) for i in range(start, end)],
        documents=texts[start:end],
        embeddings=embeddings[start:end].tolist(),
        metadatas=[
            {
                "title": doc["title"],
                "category": doc["category"]
            }
            for doc in documents[start:end]
        ]
    )

    print(f"Added documents {start} to {end}")

print("Documents in ChromaDB:", collection.count())

