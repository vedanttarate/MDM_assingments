"""
LSTM Text Prediction — FastAPI Deployment
==========================================
Lab Assignment 5: LSTM-Based Sequence Prediction System

Run:
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

Test:
    Open http://localhost:8000/docs  (Swagger UI)
    OR: python test_api.py  (in a second terminal)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import re
import json
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ── Global Variables ──────────────────────────────────────────────────────────
lstm_model   = None
tokenizer    = None
model_config = {}


# ── Helper: Text Cleaning ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Preprocess input text identically to training:
      - Lowercase
      - Remove non-alphabetic characters
      - Collapse extra whitespace
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Lifespan: Load Model on Server Startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup.
    Loads config → tokenizer (JSON) → rebuilds model → loads weights.
    """
    global lstm_model, tokenizer, model_config

    # Step 1: Load config
    config_path = "model_config.json"
    if not os.path.exists(config_path):
        print("⚠️  model_config.json not found — using defaults.")
        model_config = {
            "seq_len"   : 10,
            "vocab_size": 2576,
            "embed_dim" : 100,
            "model_file": "lstm_weights.weights.h5",
            "tok_file"  : "tokenizer.json"
        }
    else:
        with open(config_path, "r") as f:
            model_config = json.load(f)
    print(f"✅ Config loaded: {model_config}")

    seq_len    = model_config.get("seq_len",    10)
    vocab_size = model_config.get("vocab_size", 2576)
    embed_dim  = model_config.get("embed_dim",  100)

    # Step 2: Load tokenizer from JSON (version-independent)
    tok_path = model_config.get("tok_file", "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"⚠️  Tokenizer '{tok_path}' not found.")
    else:
        with open(tok_path, "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
        print(f"✅ Tokenizer loaded. Vocabulary size: {len(tokenizer.word_index)}")

    # Step 3: Rebuild model architecture
    # (avoids Keras version incompatibility with full .h5 loading)
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            input_length=seq_len,
            name='Embedding'
        ),
        tf.keras.layers.LSTM(
            128, return_sequences=True,
            dropout=0.2, recurrent_dropout=0.2,
            name='LSTM_1'
        ),
        tf.keras.layers.Dropout(0.3, name='Dropout_1'),
        tf.keras.layers.LSTM(
            64, dropout=0.2, recurrent_dropout=0.2,
            name='LSTM_2'
        ),
        tf.keras.layers.Dropout(0.3, name='Dropout_2'),
        tf.keras.layers.Dense(64, activation='relu', name='Dense_Hidden'),
        tf.keras.layers.Dense(vocab_size, activation='softmax', name='Output')
    ], name='LSTM_TextPredictor')

    lstm_model.build(input_shape=(None, seq_len))
    print(f"✅ Model architecture built. Parameters: {lstm_model.count_params():,}")

    # Step 4: Load weights from numpy arrays (version-independent format)
    weights_path = model_config.get("model_file", "lstm_weights.npy")
    if not os.path.exists(weights_path):
        print(f"⚠️  Weights file '{weights_path}' not found.")
    else:
        weights = np.load(weights_path, allow_pickle=True)
        lstm_model.set_weights(weights)
        print(f"✅ Weights loaded from '{weights_path}'.")
        print(f"   Total weight arrays: {len(weights)}")

    print("\n🚀 Server ready! Open http://localhost:8000/docs\n")
    yield
    print("🛑 Shutting down.")


# ── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="LSTM Text Prediction API",
    description="""
## 🤖 LSTM-Based Next Word Prediction

**Lab Assignment 5** — LSTM Sequence Prediction with FastAPI Deployment

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`        | Health check |
| `POST` | `/predict` | Predict next word |
| `POST` | `/generate`| Generate N words |

**Dataset:** Alice's Adventures in Wonderland (Project Gutenberg)
**Model:** Embedding → LSTM(128) → LSTM(64) → Dense(Softmax)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
# NOTE: Pydantic v2 — use model_config = ConfigDict(...) ONLY.
#       Do NOT also define class Config — that causes PydanticUserError.

class PredictRequest(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "text"       : "alice was beginning to get very tired of sitting by",
                "top_k"      : 5,
                "temperature": 1.0
            }
        }
    )

    text: str = Field(
        ...,
        example="alice was beginning to get very tired of sitting by",
        description="Input seed text. Model uses the last SEQ_LEN words."
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top candidates to return (1–20)."
    )
    temperature: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Temperature: <1=conservative, 1=normal, >1=creative."
    )


class WordCandidate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    word       : str
    probability: float
    percentage : str


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    input_text    : str
    cleaned_input : str
    predicted_word: str
    top_candidates: List[WordCandidate]
    model_info    : dict


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status           : str
    model_loaded     : bool
    tokenizer_loaded : bool
    vocab_size       : int
    seq_length       : int
    total_parameters : str
    message          : str


# ── GET / — Health Check ──────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, summary="Health Check", tags=["System"])
async def health_check():
    """
    Returns API status. Verify model and tokenizer are loaded before predicting.
    """
    params = lstm_model.count_params() if lstm_model is not None else 0
    return HealthResponse(
        status           = "running" if lstm_model is not None else "model_not_loaded",
        model_loaded     = lstm_model is not None,
        tokenizer_loaded = tokenizer is not None,
        vocab_size       = len(tokenizer.word_index) if tokenizer else 0,
        seq_length       = model_config.get("seq_len", 0),
        total_parameters = f"{params:,}",
        message          = "✅ LSTM Text Prediction API is live! Go to /docs for Swagger UI."
    )


# ── POST /predict — Next Word Prediction ─────────────────────────────────────

@app.post("/predict", response_model=PredictResponse,
          summary="Predict Next Word", tags=["Prediction"])
async def predict_next_word(request: PredictRequest):
    """
    ## Predict the Next Word

    Steps:
    1. Clean input (lowercase, remove punctuation)
    2. Tokenize using trained vocabulary
    3. Pad/truncate to SEQ_LEN=10
    4. LSTM outputs probability over all vocabulary words
    5. Temperature scaling applied (optional)
    6. Top-K candidates returned

    **Try these seeds:**
    - `"alice was beginning to get very tired of sitting by"`
    - `"the queen shouted off with"`
    - `"curiouser and curiouser cried"`
    - `"the white rabbit looked at its watch and"`
    """
    if lstm_model is None or tokenizer is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Check server startup logs.")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        seq_len    = model_config.get("seq_len", 10)
        cleaned    = clean_text(request.text)
        token_list = tokenizer.texts_to_sequences([cleaned])[0]

        if len(token_list) == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No words found in model vocabulary. "
                    "Model trained on Alice in Wonderland. "
                    "Try: alice, rabbit, queen, cat, mad, tea, king, wonder"
                )
            )

        token_array = pad_sequences(
            [token_list], maxlen=seq_len,
            padding='pre', truncating='pre'
        )

        probs = lstm_model.predict(token_array, verbose=0)[0]

        if request.temperature != 1.0:
            probs = np.log(probs + 1e-10) / request.temperature
            probs = np.exp(probs)
            probs = probs / np.sum(probs)

        top_k_idx = np.argsort(probs)[-request.top_k:][::-1]
        best_word = tokenizer.index_word.get(int(top_k_idx[0]), "<UNK>")

        top_candidates = [
            WordCandidate(
                word        = tokenizer.index_word.get(int(i), "<UNK>"),
                probability = round(float(probs[i]), 6),
                percentage  = f"{float(probs[i]) * 100:.2f}%"
            )
            for i in top_k_idx
        ]

        return PredictResponse(
            input_text     = request.text,
            cleaned_input  = cleaned,
            predicted_word = best_word,
            top_candidates = top_candidates,
            model_info     = {
                "model_type"  : "LSTM",
                "architecture": "Embedding(100) → LSTM(128) → LSTM(64) → Dense(Softmax)",
                "seq_length"  : seq_len,
                "vocab_size"  : len(tokenizer.word_index),
                "temperature" : request.temperature,
                "dataset"     : "Alice in Wonderland — Project Gutenberg"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ── POST /generate — Text Generation ─────────────────────────────────────────

@app.post("/generate", summary="Generate Text Sequence", tags=["Prediction"])
async def generate_text(
    text       : str   = "alice was very curious about",
    n_words    : int   = 20,
    temperature: float = 1.0
):
    """
    ## Generate a Text Sequence

    Generates `n_words` new words from a seed text using the LSTM model.
    Each predicted word feeds back as input for the next step (autoregressive).

    - `temperature=0.5` → conservative / repetitive but coherent
    - `temperature=1.0` → balanced (default)
    - `temperature=1.5` → creative / varied
    """
    if lstm_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    n_words      = max(1, min(n_words, 50))
    seq_len      = model_config.get("seq_len", 10)
    output_text  = text
    current_seed = text

    for _ in range(n_words):
        cleaned   = clean_text(current_seed)
        tok_list  = tokenizer.texts_to_sequences([cleaned])[0]
        tok_array = pad_sequences(
            [tok_list], maxlen=seq_len, padding='pre', truncating='pre'
        )
        probs     = lstm_model.predict(tok_array, verbose=0)[0]
        probs     = np.log(probs + 1e-10) / temperature
        probs     = np.exp(probs)
        probs     = probs / np.sum(probs)
        next_idx  = int(np.random.choice(len(probs), p=probs))
        next_word = tokenizer.index_word.get(next_idx, '')
        output_text  += ' ' + next_word
        current_seed  = ' '.join(output_text.split()[-seq_len:])

    return {
        "seed_text"     : text,
        "generated_text": output_text,
        "words_added"   : n_words,
        "temperature"   : temperature
    }


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)