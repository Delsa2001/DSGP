from sentence_transformers import SentenceTransformer

# Load SBERT from the online model hub and save it locally
print("Downloading SBERT model... Please wait.")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("./sbert_model")  # Saves the model in a folder named 'sbert_model'
print("SBERT model downloaded and saved in 'sbert_model/' folder.")
