import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Load SBERT from local storage
sbert_model = SentenceTransformer("./sbert_model")  # Uses locally stored model

# Load test data
df = pd.read_csv("chatbot_test_data.csv")

# Define intent categories
intent_templates = {
    "ask_planting_techniques": [
        "How do I grow a plant?",
        "What soil type is required?",
        "How much water does this plant need?",
        "What is the best soil for Tomatoes?",
        "Which soil is best for vegetables?",
        "Can I grow Basil indoors?",
        "What’s the best way to water spinach?",
        "Do tomato plants need daily watering?",
        "How much sunlight does a cucumber need?",
        "Can I use compost for my vegetable garden?",
        "What is the ideal pH level for carrots?",
        "What’s the best way to fertilize pepper plants?",
        "Is hydroponics better for growing lettuce?",
        "Can I grow rosemary in sandy soil?",
        "Should I water strawberries in the morning or evening?",
        "How much fertilizer do leafy greens need?",
        "What’s the ideal temperature for growing eggplants?",
        "Do herbs grow better in shade or sunlight?"
    ],
    "ask_plant_disease": [
        "What are the symptoms of a disease?",
        "How do I treat a plant disease?",
        "How can I prevent this disease?",
        "What is Rust?",
        "How do I stop Powdery Mildew?",
        "What does Leaf Spot look like?",
        "What is the best organic treatment for plant infections?",
        "How does Blight affect plants?",
        "Can fungal infections kill my crops?",
        "Which plants are resistant to Downy Mildew?",
        "How do I identify early signs of Anthracnose?",
        "What’s the best way to prevent Root Rot?",
        "Can overwatering cause plant diseases?",
        "What are common bacterial diseases in vegetables?",
        "How do I remove fungal infections naturally?",
        "Which crops are most vulnerable to Downy Mildew?",
        "What’s the best fungicide for home gardens?",
        "Can I use neem oil to treat plant diseases?"
    ]
}

# Encode templates using SBERT
intent_embeddings = {intent: [sbert_model.encode(example) for example in examples] for intent, examples in intent_templates.items()}

# Function to match query to closest intent
def predict_intent(query):
    query_embedding = sbert_model.encode(query)
    best_match = None
    highest_score = 0

    for intent, embeddings in intent_embeddings.items():
        for emb in embeddings:
            similarity = util.pytorch_cos_sim(query_embedding, emb).item()
            if similarity > highest_score:
                highest_score = similarity
                best_match = intent

    return best_match if highest_score > 0.5 else "unknown"

# Evaluate SBERT predictions
expected_intents = []
predicted_intents = []

for _, row in df.iterrows():
    query = row["query"]
    expected_intent = row["intent"]
    predicted_intent = predict_intent(query)

    expected_intents.append(expected_intent)
    predicted_intents.append(predicted_intent)

# Calculate accuracy
accuracy = accuracy_score(expected_intents, predicted_intents)
print(f"SBERT Intent Matching Accuracy: {accuracy:.2%}")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(expected_intents, predicted_intents, labels=df["intent"].unique())

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df["intent"].unique(), yticklabels=df["intent"].unique())
plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.title("SBERT Intent Matching Confusion Matrix")
plt.show()

# Print Classification Report
print("Classification Report:\n", classification_report(expected_intents, predicted_intents))
