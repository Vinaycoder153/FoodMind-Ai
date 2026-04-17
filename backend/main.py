"""
FoodMind AI – FastAPI Backend
"""

import csv
import io
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nlp_engine import analyse_review, analyse_review_precise, aggregate_insights

app = FastAPI(
    title="FoodMind AI API",
    description="AI-Powered Food Review Intelligence + Customisation Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ReviewsRequest(BaseModel):
    reviews: List[str]


class AnalyseResponse(BaseModel):
    individual: List[dict]
    insights: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {"status": "ok", "service": "FoodMind AI"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyse")
def analyse(request: ReviewsRequest) -> AnalyseResponse:
    """
    Main analysis endpoint.
    Accepts a list of raw review strings.
    Returns per-review analysis + aggregated insights.
    """
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")

    # Filter out empty strings
    raw_reviews = [r.strip() for r in request.reviews if r.strip()]
    if not raw_reviews:
        raise HTTPException(status_code=400, detail="All reviews are empty")

    individual = [analyse_review(text) for text in raw_reviews]
    insights = aggregate_insights(individual)

    return {"individual": individual, "insights": insights}


@app.post("/analyse-precise")
def analyse_precise(request: ReviewsRequest):
    """
    Precision endpoint that returns strict per-review JSON schema.
    """
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")

    raw_reviews = [r.strip() for r in request.reviews if r.strip()]
    if not raw_reviews:
        raise HTTPException(status_code=400, detail="All reviews are empty")

    individual = [analyse_review_precise(text) for text in raw_reviews]
    return JSONResponse(content={"individual": individual})


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept a CSV file with a 'review' column.
    Returns the same analysis as /analyse.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    try:
        decoded = content.decode("utf-8-sig")  # handle BOM
    except UnicodeDecodeError:
        decoded = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(decoded))

    # Try common column names
    possible_cols = ["review", "Review", "REVIEW", "text", "Text", "comment", "Comment"]
    reviews: List[str] = []

    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # Detect column
    col_name: Optional[str] = None
    for pc in possible_cols:
        if pc in rows[0]:
            col_name = pc
            break

    if col_name is None:
        # Use first column
        col_name = list(rows[0].keys())[0]

    for row in rows:
        val = row.get(col_name, "").strip()
        if val:
            reviews.append(val)

    if not reviews:
        raise HTTPException(status_code=400, detail="No review text found in CSV")

    individual = [analyse_review(text) for text in reviews]
    insights = aggregate_insights(individual)

    return JSONResponse(content={"individual": individual, "insights": insights})
