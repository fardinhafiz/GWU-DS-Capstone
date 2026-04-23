# AI Fashion Curator Upgrade

This package upgrades the original Streamlit prototype into a React + FastAPI project with:

- semantic NLP search using Sentence-BERT
- fashion synonym normalization such as hoodie -> sweatshirt and trainers -> sneakers
- multimodal style profiling with uploaded inspiration images using CLIP
- persistent user profiles and interactions in SQLite
- model comparison and selection across multiple classifiers
- a more polished React frontend

## Project structure

```text
fashion_curator_upgrade/
├── backend/
│   ├── app/
│   ├── data/
│   │   └── styles.csv
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Backend setup

From the `backend` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API runs at:

```text
http://localhost:8000
```

## Frontend setup

From the `frontend` folder:

```bash
npm install
npm run dev
```

The React app runs at:

```text
http://localhost:5173
```

## Notes

Place your catalog product images inside:

```text
backend/images/
```

and name them by item id, for example:

```text
1163.jpg
1164.png
```

The uploaded style images are stored in:

```text
backend/data/user_uploads/
```

## Model training

The app compares these models once you have enough feedback data:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier
- MLPClassifier

Use the UI button or call:

```text
POST /train-models
```

If there is not enough data yet, the app falls back to the hybrid semantic scoring system.

## What changed from the original prototype

- TF-IDF search was replaced with semantic sentence embeddings
- synonym normalization was added for fashion vernacular
- a trainable model-selection layer was added
- image upload is now part of the user profile
- recommendations use text, profile, image, and interaction signals together
- the frontend is now React instead of Streamlit
