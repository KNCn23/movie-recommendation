"""Content-based movie recommender.

Given a movie title, returns the most similar titles using TF-IDF over each
film's plot overview plus its (genre-weighted) tags, scored by cosine
similarity. The similarity neighbours are precomputed once and reused, so the
same engine powers both the CLI here and the static site data export
(``build_data.py``).

Dataset: Pablinho/movies-dataset (TMDB, CC0). Columns:
    Release_Date, Title, Overview, Popularity, Vote_Count,
    Vote_Average, Original_Language, Genre, Poster_Url

Usage:
    python engine/recommender.py "The Matrix"
    python engine/recommender.py "Inception" --n 15
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "movies.csv"

# Similarity = W_OVERVIEW · cosine(plot TF-IDF)
#            + W_GENRE    · cosine(genre multi-hot)
#            + W_QUALITY  · (candidate rating / 10)
#
# Genres are kept as a separate multi-hot signal rather than folded into the
# text: as plain tokens their high document-frequency makes TF-IDF ignore them,
# yet "same genres" is exactly what makes two films feel alike. The small
# quality term is a tie-breaker so that, among equally-similar matches, the
# better-rated film surfaces first (and avoids obscure low-quality look-alikes).
W_OVERVIEW = 1.0
W_GENRE = 0.9
W_QUALITY = 0.05
TOP_N = 12


@dataclass
class Catalog:
    """Cleaned movie table plus its precomputed similarity neighbours."""

    df: pd.DataFrame
    neighbours: np.ndarray  # (n_movies, TOP_N) int32 indices, nearest first
    scores: np.ndarray      # (n_movies, TOP_N) float32 cosine scores

    def __len__(self) -> int:
        return len(self.df)


def _genre_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [g.strip() for g in str(raw).split(",") if g.strip()]


def load_movies(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    """Load and clean the raw dataset into a stable, reset-indexed table."""
    df = pd.read_csv(csv_path)
    df["Title"] = df["Title"].fillna("").astype(str).str.strip()
    df = df[df["Title"] != ""]

    df["Overview"] = df["Overview"].fillna("").astype(str)
    df["Genre"] = df["Genre"].fillna("").astype(str)
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce").fillna(0.0)
    df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce").fillna(0.0)
    df["Original_Language"] = df["Original_Language"].fillna("").astype(str)

    # Year from the release date (e.g. "2021-12-15" -> 2021).
    df["Year"] = pd.to_datetime(df["Release_Date"], errors="coerce").dt.year

    # Smaller, faster poster variant from the original-size TMDB URL.
    df["Poster"] = (
        df["Poster_Url"].fillna("").astype(str).str.replace("/original/", "/w500/")
    )

    df["GenreList"] = df["Genre"].map(_genre_list)

    # De-duplicate by title, keeping the most popular entry.
    df = (
        df.sort_values("Popularity", ascending=False)
        .drop_duplicates(subset="Title", keep="first")
        .sort_values("Popularity", ascending=False)
        .reset_index(drop=True)
    )
    return df


def _build_matrices(df: pd.DataFrame):
    """Return (overview, genre) L2-normalised sparse matrices + rating vector."""
    overview = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=60_000,
    ).fit_transform(df["Overview"])

    genre = MultiLabelBinarizer(sparse_output=True).fit_transform(df["GenreList"])
    genre = normalize(genre.astype(np.float32), norm="l2")  # cosine = dot

    rating = (df["Vote_Average"].to_numpy(dtype=np.float32) / 10.0).clip(0, 1)
    return overview, genre, rating


def _top_neighbours(overview, genre, rating, top_n: int, batch: int = 512):
    """Weighted cosine top-N per row without materialising the full N×N matrix.

    Each L2-normalised block contributes a cosine (= dot product) term; the
    catalog is scored one batch of rows at a time to keep memory flat.
    """
    n = overview.shape[0]
    nbr = np.empty((n, top_n), dtype=np.int32)
    scr = np.empty((n, top_n), dtype=np.float32)
    overview_t = overview.T.tocsr()
    genre_t = genre.T.tocsr()
    quality = W_QUALITY * rating  # added per candidate column

    for start in range(0, n, batch):
        end = min(start + batch, n)
        sims = W_OVERVIEW * (overview[start:end] @ overview_t).toarray()
        sims += W_GENRE * (genre[start:end] @ genre_t).toarray()
        sims += quality  # broadcast over rows
        # Exclude self-matches.
        for local, row in enumerate(range(start, end)):
            sims[local, row] = -1.0
        # argpartition for the top_n, then sort those by score desc.
        part = np.argpartition(-sims, top_n, axis=1)[:, :top_n]
        rows = np.arange(end - start)[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        idx = part[rows, order]
        nbr[start:end] = idx
        scr[start:end] = sims[rows, idx]
    return nbr, scr


def build_catalog(csv_path: Path = DATA_CSV, top_n: int = TOP_N) -> Catalog:
    df = load_movies(csv_path)
    overview, genre, rating = _build_matrices(df)
    neighbours, scores = _top_neighbours(overview, genre, rating, top_n)
    return Catalog(df=df, neighbours=neighbours, scores=scores)


def find_index(title: str, df: pd.DataFrame) -> int | None:
    """Resolve a title to a row index (exact, then case-insensitive)."""
    title = title.strip()
    exact = df.index[df["Title"] == title]
    if len(exact):
        return int(exact[0])
    ci = df.index[df["Title"].str.lower() == title.lower()]
    return int(ci[0]) if len(ci) else None


def recommend(title: str, catalog: Catalog, n: int = TOP_N) -> pd.DataFrame:
    idx = find_index(title, catalog.df)
    if idx is None:
        raise KeyError(f"Movie '{title}' not found in the catalog.")
    nbr = catalog.neighbours[idx][:n]
    out = catalog.df.iloc[nbr][["Title", "Year", "Genre", "Vote_Average"]].copy()
    out["Similarity"] = catalog.scores[idx][:n].round(3)
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Content-based movie recommender")
    parser.add_argument("title", help="movie title to get recommendations for")
    parser.add_argument("--n", type=int, default=TOP_N, help="how many to return")
    args = parser.parse_args(argv)

    print("🎬  Building catalog (TF-IDF + cosine similarity)…", file=sys.stderr)
    catalog = build_catalog(top_n=max(args.n, TOP_N))
    print(f"    {len(catalog):,} movies indexed.\n", file=sys.stderr)

    try:
        recs = recommend(args.title, catalog, n=args.n)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Because you liked “{args.title}”, you might enjoy:\n")
    for i, row in recs.iterrows():
        year = "" if pd.isna(row["Year"]) else f" ({int(row['Year'])})"
        print(f"  {i + 1:>2}. {row['Title']}{year}  —  {row['Genre']}"
              f"  ·  ★{row['Vote_Average']:.1f}  ·  sim {row['Similarity']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
