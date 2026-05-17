# Movie Recommendation System

A content-based movie recommendation engine built with Python, using the Netflix titles dataset. Given a movie title, it returns the most similar titles based on description similarity.

## How It Works

Each movie description is vectorized using TF-IDF, then cosine similarity is computed pairwise across the dataset. The top-N closest matches by similarity score are returned as recommendations.

## Run

```bash
pip install pandas scikit-learn
python movie_recommender.py
```

Make sure `netflix_titles.csv` is in the project directory before running.

## Implementations

| File | Approach |
|------|----------|
| `movie_recommender.py` | TF-IDF + Cosine Similarity (primary) |
| `movie-recommendation(pandas).py` | Pandas-only exploration |
| `movie-recommendation(TensorFlow keras).py` | Neural embedding approach |

## Tech

- Python
- Pandas — data loading and preprocessing
- Scikit-learn — TF-IDF vectorization and cosine similarity

## License

MIT
