"""Export the recommender output as a single static JSON for the web app.

Runs the content-based engine once and writes ``docs/data/movies.json`` —
every movie's metadata plus its precomputed list of similar-movie indices.
The site then needs no server: all recommendations are a lookup in this file.

    python engine/build_data.py
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

from recommender import ROOT, TOP_N, build_catalog

OUT_PATH = ROOT / "docs" / "data" / "movies.json"


def _year(value) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def main() -> int:
    print("Building catalog…")
    catalog = build_catalog(top_n=TOP_N)
    df = catalog.df
    print(f"  {len(df):,} movies indexed.")

    movies = []
    genre_set: set[str] = set()
    for i, row in enumerate(df.itertuples(index=False)):
        genres = list(row.GenreList)
        genre_set.update(genres)
        movies.append(
            {
                "title": row.Title,
                "year": _year(row.Year),
                "genres": genres,
                "rating": round(float(row.Vote_Average), 1),
                "poster": row.Poster or None,
                "overview": row.Overview,
                "lang": row.Original_Language,
                "rec": catalog.neighbours[i].tolist(),
            }
        )

    payload = {
        "generated": date.today().isoformat(),
        "count": len(movies),
        "genres": sorted(genre_set),
        "movies": movies,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"  wrote {OUT_PATH.relative_to(ROOT)}  ({size_mb:.1f} MB, "
          f"{len(payload['genres'])} genres)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
