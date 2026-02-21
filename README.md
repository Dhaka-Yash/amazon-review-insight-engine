# Amazon Review Insight Engine

A Python project for extracting actionable insights from Amazon product reviews using semantic embeddings, unsupervised clustering, vector search, and trend analysis.

## What It Does

- Cleans and samples raw review data from `data/Reviews.csv`
- Builds sentence embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Reduces embedding dimensions with UMAP and clusters reviews with HDBSCAN
- Calculates cluster-level impact using average rating (`Score`)
- Tracks issue/topic trends over time (`YearMonth`)
- Powers a Streamlit dashboard with semantic search + cluster analytics

## Project Structure

```text
.
├── data/
│   ├── Reviews.csv                # raw input dataset
│   ├── clean_reviews.csv          # generated after preprocessing
│   ├── embeddings.npy             # generated embeddings cache
│   └── clustered_reviews.csv      # generated clustered output
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── notebooks/                     # step-by-step exploration notebooks
├── src/
│   ├── config.py                  # paths + model/cluster settings
│   ├── data_processing.py         # cleaning + feature prep
│   ├── embeddings.py              # embedding pipeline
│   ├── clusterings.py             # UMAP + HDBSCAN + silhouette score
│   ├── insight_engine.py          # cluster impact helpers
│   ├── trend_analysis.py          # monthly trend aggregation
│   └── vector_search.py           # FAISS semantic search
├── main.py                        # end-to-end pipeline entry point
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

## Run the Pipeline

```powershell
python main.py
```

Pipeline steps in `main.py`:

1. Load and clean reviews
2. Generate (or reuse cached) embeddings
3. Reduce dimensions and cluster reviews
4. Save clustered results
5. Print cluster impact summary
6. Print trend snapshot

## Launch the Dashboard

After running the pipeline:

```powershell
streamlit run dashboard/app.py
```

Dashboard includes:

- Semantic review search (FAISS)
- Top recurring issue clusters
- Average rating per cluster
- Cluster trends over time

## Configuration

Tune settings in `src/config.py`:

- `SAMPLE_SIZE` (default: `50000`)
- `EMBEDDING_MODEL` (default: `all-MiniLM-L6-v2`)
- `MIN_CLUSTER_SIZE` (default: `100`)
- Input/output file paths

## Data Requirements

`data/Reviews.csv` should contain at least:

- `Text`
- `Summary`
- `Score`
- `Time`

`Time` supports Unix epoch values or parseable datetime strings.

## Notes

- Embeddings are cached in `data/embeddings.npy` and reused when row counts match.
- For first run, model download can take time.
- If GPU is available, embedding generation uses CUDA/MPS automatically.
