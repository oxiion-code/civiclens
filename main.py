from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# Enable CORS (allow frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with ["http://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Complaint Categories
CATEGORIES = [
    "road",
    "street_light",
    "garbage",
    "water",
    "medical",
    "electricity",        # power outages or issues
    "traffic_signal",     # broken traffic lights
    "potholes",           # road damage
    "drainage",           # clogged drains or sewage
    "parking",            # illegal or inadequate parking
    "pollution",          # air or noise pollution
    "animal_control",     # stray animals or related hazards
    "public_transport",   # buses, metro, etc. issues
    "parks",              # maintenance of public parks
    "fire_safety",        # fire hazards or incidents
    "construction",       # illegal construction or unsafe sites
    "education",          # school/college related complaints
    "sanitation",         # cleanliness in public areas
    "waterlogging",       # flooding during rains
    "other"               # anything not listed above
]

# Pydantic model for request body
class Complaint(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "CitiVoice Backend with Gemini API is running!"}


@app.post("/classify_text/")
def classify_text(complaint: Complaint):
    """
    Classify complaint text into predefined categories using Gemini API.
    """
    text = complaint.text

    # Prompt for Gemini
    prompt = f"""
    Classify the following citizen complaint into one of these categories:
    {CATEGORIES}
    Complaint: "{text}"
    Respond only with the category name.
    """

    # Gemini API request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 50
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract Gemini’s output
        predicted_label = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        return {
            "text": text,
            "predicted_class": predicted_label
        }

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {
            "text": text,
            "predicted_class": "Error: Failed to connect to Gemini API"
        }
