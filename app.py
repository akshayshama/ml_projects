# app.py - Final FastAPI Backend with Double-Hybrid Model and CSV Download

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
import warnings
import csv
import io
import json

# Suppress Hugging Face warnings during model loading for a cleaner console
warnings.filterwarnings("ignore")

app = FastAPI(title="AI Plagiarism Detector Core Engine (RoBERTa Semantic)")

# --- 1. CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- 2. GLOBAL MODEL SETUP ---
AI_MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta" 
SEMANTIC_MODEL_NAME = 'BAAI/bge-large-en-v1.5'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AI_PLAGIARISM_THRESHOLD = 0.60    
SEMANTIC_PLAGIARISM_THRESHOLD = 0.75 

AI_TOKENIZER, AI_MODEL, SEMANTIC_MODEL = None, None, None

@app.on_event("startup")
def load_models():
    """Load large models only once when the application starts."""
    global AI_TOKENIZER, AI_MODEL, SEMANTIC_MODEL
    try:
        print(f"Loading models on device: {DEVICE}")
        
        # 1. Load AI Detection Model (RoBERTa For Classification)
        AI_TOKENIZER = AutoTokenizer.from_pretrained(AI_MODEL_NAME)
        AI_MODEL = AutoModelForSequenceClassification.from_pretrained(AI_MODEL_NAME).to(DEVICE)
        
        # 2. Load Semantic Similarity Model (BGE/SentenceTransformer)
        SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME).to(DEVICE)
        print("All models loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load required models. Analysis will be disabled. Error: {e}")
        AI_TOKENIZER, AI_MODEL, SEMANTIC_MODEL = None, None, None

# --- 3. Core Analysis Functions ---

def predict_ai_probability(text: str) -> float:
    """Predicts the probability that the text is AI-generated (0.0 to 1.0)."""
    if AI_MODEL is None: return 0.5 
    
    inputs = AI_TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = AI_MODEL(**inputs)
        
    probabilities = torch.softmax(outputs.logits, dim=1)
    return probabilities[0][1].item()

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculates semantic similarity between two texts using the BGE model."""
    if SEMANTIC_MODEL is None: return 0.0
    
    with torch.no_grad():
        emb1 = SEMANTIC_MODEL.encode(text1, convert_to_tensor=True).to(DEVICE)
        emb2 = SEMANTIC_MODEL.encode(text2, convert_to_tensor=True).to(DEVICE)
    
    similarity = util.cos_sim(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return similarity

def generate_csv_report(results: List[Dict[str, Any]]) -> io.StringIO:
    """Converts the list of result dictionaries into an in-memory CSV file stream."""
    
    fieldnames = [
        "filename",
        "verdict",
        "ai_probability",
        "semantic_score",
        "is_ai_plagiarism"
    ]
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for row in results:
        writer.writerow({
            "filename": row["filename"],
            "verdict": row["verdict"],
            "ai_probability": row["ai_probability"],
            "semantic_score": row["semantic_score"],
            "is_ai_plagiarism": row["is_ai_plagiarism"]
        })
        
    output.seek(0)
    return output

# --- 4. ENDPOINTS ---

@app.get("/ping")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "service": "AI Detector v2 (RoBERTa Semantic)", 
        "ai_model_loaded": AI_MODEL is not None,
        "semantic_model_loaded": SEMANTIC_MODEL is not None
    }

@app.post("/api/analyze_submissions")
async def analyze_submissions(files: list[UploadFile] = File(...), format: str = Form("json")):
    """
    Analyzes exactly two uploaded files for AI generation probability and semantic similarity,
    returning results as JSON or a downloadable CSV file.
    """
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Must upload exactly 2 files for comparison.")
    
    if AI_MODEL is None or SEMANTIC_MODEL is None:
        raise HTTPException(status_code=503, detail="AI and Semantic models are not loaded. Check server logs.")

    # 1. Read Contents
    contents = []
    filenames = []
    for f in files:
        try:
            content = await f.read()
            contents.append(content.decode('utf-8'))
            filenames.append(f.filename)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Could not read/decode file: {f.filename}. Ensure it's a plain text/code file.")

    # 2. Run Analysis
    results = []
    semantic_score = calculate_semantic_similarity(contents[0], contents[1])
    
    for i in range(2):
        text_content = contents[i]
        
        # AI Detection
        ai_probability = predict_ai_probability(text_content)
        is_ai_plagiarism = ai_probability > AI_PLAGIARISM_THRESHOLD
        
        # Determine Verdict
        verdict = "ORIGINAL"
        if is_ai_plagiarism:
            verdict = "PLAGIARISM (High AI Prob)"
        elif semantic_score > SEMANTIC_PLAGIARISM_THRESHOLD:
            verdict = "PLAGIARISM (High Semantic Match)"
            
        results.append({
            "filename": filenames[i],
            "ai_probability": round(ai_probability, 4),
            "is_ai_plagiarism": is_ai_plagiarism,
            "semantic_score": round(semantic_score, 4), 
            "verdict": verdict,
        })
    
    # 3. Determine Overall Verdict
    overall_verdict = "ORIGINAL (All Checks Passed)"
    if results[0]['is_ai_plagiarism'] or results[1]['is_ai_plagiarism']:
        overall_verdict = "PLAGIARISM DETECTED (One or both files are AI-generated)"
    elif semantic_score > SEMANTIC_PLAGIARISM_THRESHOLD:
        overall_verdict = "PLAGIARISM DETECTED (High semantic match between the two files)"
        
    # --- 4. Return Response ---
    
    if format.lower() == "csv":
        csv_stream = generate_csv_report(results)
        headers = {
            'Content-Disposition': 'attachment; filename="plagiarism_report.csv"'
        }
        return StreamingResponse(
            csv_stream,
            headers=headers,
            media_type="text/csv"
        )
    
    # Default: Return JSON response
    final_response = {
        "overall_verdict": overall_verdict,
        "file_results": results,
        "semantic_similarity_score_A_B": round(semantic_score, 4),
    }
    return JSONResponse(content=final_response)