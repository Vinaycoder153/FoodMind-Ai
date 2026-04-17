# FoodMind-Ai

**AI-Powered Food Review Intelligence + Customisation Engine**

> An AI system that not only *understands* customer feedback but continuously improves food quality through measurable, data-driven decisions.

![FoodMind AI Hero](https://github.com/user-attachments/assets/8fae96f0-225c-4f48-be66-b52453d6ee18)

---

## 🚀 Features

| Module | What it does |
|--------|-------------|
| **Input Module** | Paste raw reviews (English / Hindi / Kannada) or upload a CSV |
| **NLP Processing** | Extracts oil / spice / quantity / taste features + sentiment classification |
| **Insight Engine** | Complaint % per feature, top complaint, trend detection, alert banners |
| **Action Engine** | 2–3 AI recommendations with confidence % and reasoning |
| **Impact Simulator** | Predicts rating improvement after implementing recommendations |
| **Smart Customisation** | Generates new menu options (Light Oil, Custom Spice, Extra Protein) |
| **Feedback Loop Dashboard** | Before vs After complaint comparison with bar visualisations |

---

## 📸 Screenshots

### Overview Dashboard
![Overview](https://github.com/user-attachments/assets/5853f011-7b4b-4264-a9d7-547abee0dc02)

### AI Recommendations
![Recommendations](https://github.com/user-attachments/assets/e1eadfc4-877c-48e7-a6c5-c3b4533c4d34)

---

## 🏗️ Architecture

```
frontend/          ← HTML + CSS + JS dashboard (zero build step)
  index.html       ← Single-page app
  styles.css       ← Dark-themed dashboard styles
  app.js           ← API calls, DOM rendering, chart logic
  charts.js        ← Lightweight canvas-based chart library (no CDN deps)

backend/           ← FastAPI Python service
  main.py          ← REST endpoints (/analyse, /upload-csv)
  nlp_engine.py    ← Keyword-based NLP: feature extraction + sentiment
  requirements.txt ← fastapi, uvicorn, python-multipart
```

**Data flow:**
```
Reviews (text / CSV)
  → POST /analyse
    → nlp_engine.analyse_review()      per-review tags + sentiment
    → nlp_engine.aggregate_insights()  complaint %, recommendations, impact
  → JSON response
    → frontend renders charts + cards
```

---

## ⚡ Quick Start

### 1. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Start the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the helper script:

```bash
bash start_backend.sh
```

### 3. Open the frontend

Serve the `frontend/` directory with any static server:

```bash
cd frontend
python3 -m http.server 3000
# open http://localhost:3000
```

---

## 🧠 NLP Engine

The engine uses a **keyword + negation** approach — fast, explainable, zero ML dependencies:

- **Feature detection** — keyword sets for `oil`, `spice`, `quantity`, `taste`
- **Multilingual mapping** — Hindi (`तेल`, `masala`, `teekha`) and Kannada (`ಎಣ್ಣೆ`, `ಖಾರ`, `ರುಚಿ`) tokens are normalised to English before processing
- **Sentiment** — positive/negative word lists with a negation window (`not good` → negative)
- **Feature-level sentiment** — each feature gets its own positive/negative/neutral label

---

## 📡 API Reference

### `POST /analyse`

```json
{
  "reviews": ["The food was too oily!", "Amazing biryani, loved it!"]
}
```

Returns `individual` (per-review analysis) + `insights` (aggregated):
- `sentiment_overview` — positive/negative/neutral %
- `feature_complaints` — complaint % per feature
- `top_complaint` — worst feature
- `alerts` — critical issues (threshold ≥ 25%)
- `recommendations` — 2–3 actions with confidence + reasoning
- `impact` — simulated before/after ratings
- `customisations` — new menu options
- `trend` — complaint direction over time

### `POST /upload-csv`

Upload a CSV file with a `review` column. Returns the same response as `/analyse`.

---

## 🛠️ Tech Stack

- **Frontend:** HTML5 / CSS3 / Vanilla JS — no framework, no build step
- **Charts:** Custom canvas-based chart library (doughnut + bar)
- **Backend:** FastAPI (Python 3.9+)
- **NLP:** Pure Python keyword engine — no external ML libraries
