# Sentiment Analysis MVP — 2-Hour Workshop

Fast, working Streamlit demo that analyzes customer reviews using a pre-trained DistilBERT model.

## What you get
- Single review sentiment analysis with a polished badge UI
- Bulk CSV upload with progress bar, interactive Plotly charts, and CSV download
- Uses Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` (no training required)

## Files
- `app.py` — Streamlit app (single-file)
- `requirements.txt` — Exact dependency versions
- `sample_reviews.csv` — Example CSV to test bulk upload

## Quick setup (2–3 minutes)
1. Create and activate a virtual environment (recommended)
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. On first run the model will download — this may take a minute.

## How to demo
- Type a positive review → see green badge and high confidence
- Type a negative review → see red badge and lower confidence
- Upload `sample_reviews.csv` → press "Analyze All Reviews" → watch progress bar → view charts → download results

## Notes & troubleshooting
- The app uses internet to download the model on first run.
- If the model fails to load, Streamlit shows an error and stops.
- CSV must include a `review` or `text` column (case-insensitive checks included).

## Next improvements (post-workshop)
- Train on custom dataset
- Add aspect-based sentiment
- Add caching for CSV results
- Deploy to Streamlit Cloud or similar
