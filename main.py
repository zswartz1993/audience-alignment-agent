import os
import argparse
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from clients import (
    llm_request_to_filter,
    discover_movies,
    find_rt_url,
    get_rt_movie_id,
    collect_reviews_fast,
    summarize_reviews_with_claude,
)
from scoring import (
    embed_texts,
    batch_sentiment_scores,
    compute_alignment,
)

async def get_movie_ids_concurrent(movies, concurrency=4):
    sem = asyncio.Semaphore(concurrency)

    async def one(movie):
        title = movie["title"]
        title_q = title.replace("&", "and")
        rt_url = find_rt_url(title_q)
        if not rt_url:
            return title, None
        async with sem:
            try:
                mid = await get_rt_movie_id(rt_url)
            except Exception:
                mid = None
            return title, mid

    pairs = await asyncio.gather(*[one(m) for m in movies])
    return dict(pairs)

async def collect_many(movie_ids, max_reviews=100, concurrency=4):
    sem = asyncio.Semaphore(concurrency)

    async def one(mid):
        async with sem:
            try:
                reviews = await collect_reviews_fast(mid, max_reviews=max_reviews, verified=False)
            except Exception:
                reviews = []
            return mid, reviews

    tasks = [asyncio.create_task(one(mid)) for mid in movie_ids]
    return dict(await asyncio.gather(*tasks))

async def run_pipeline(input_text: str, topn: int = 3):
    # Models (local)
    sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True,
        device=-1,
    )

    # 1) LLM: classify + query_review + tmdb_filter + 10 LLM recs
    out = llm_request_to_filter(input_text)
    query_review = out["query_review"]
    tmdb_filter = out["tmdb_filter"]
    llm_recs = out["recommendations"]

    print(f'Input is {out["input_type"]}')
    print()

    if out["input_type"] == 'movie_title':
        print(f'Normalized title: {out["normalized_title"]}')
        print(f'Audience review summary: {out["query_review"]}')
        print()

    print('Claude initial recommendations:')
    for rec in llm_recs:
        print(f'    * {rec['title']} ({rec['year']})')
    print()

    print('TMDB API filter:')
    print(tmdb_filter)
    print()

    # 2) TMDB candidates
    tmdb_movies = discover_movies(tmdb_filter)
    print('TMDB candidates:')
    for i, m in enumerate(tmdb_movies, 1):
        print(f"    * {m['title']} ({m['release_date']})  ★{m['vote_average']}  votes={m['vote_count']}  pop={m['popularity']:.1f}")
    print()

    # 3) Union candidates
    titles = set([d["title"] for d in tmdb_movies] + [d["title"] for d in llm_recs])
    movies = [{"title": t} for t in titles if t != out["normalized_title"]]

    # 4) RT movie ids
    print('Fetching RT movie IDs...')
    movie_ids = await get_movie_ids_concurrent(movies, concurrency=4)
    movie_ids = {title: mid for title, mid in movie_ids.items() if mid}
    id_to_title = {mid: title for title, mid in movie_ids.items()}

    if not movie_ids:
        print("No RT movie IDs resolved. (RT URL lookup / Playwright capture likely failed.)")
        return

    # 5) Fetch reviews
    print('Fetching RT audience reviews...')
    print()
    audience_reviews = await collect_many(list(movie_ids.values()), max_reviews=100, concurrency=4)

    # 6) Compute review embeddings + sentiment
    print('Computing review embeddings and sentiment...')
    print()
    movie_data = {}
    for mid, reviews in audience_reviews.items():
        reviews = list(set(reviews))
        if len(reviews) < 10:
            continue

        embs = embed_texts(reviews, sentence_model, batch_size=64)
        sents = batch_sentiment_scores(reviews, sentiment_model, batch_size=64)
        movie_data[mid] = {"reviews": reviews, "embeddings": embs, "sentiment": sents}

    if not movie_data:
        print("No movies had >=10 reviews after fetching.")
        return

    # 7) Query embedding + sentiment
    print('Computing query embedding and sentiment...')
    print()
    q_emb = embed_texts([query_review], sentence_model, batch_size=1)[0]
    q_sent = float(batch_sentiment_scores([query_review], sentiment_model, batch_size=1)[0])

    # 8) Compute alignment per movie
    print('Computing query-review alignment')
    print()
    alignments = {}
    for mid, d in movie_data.items():
        a = compute_alignment(q_emb, d["embeddings"], q_sent, d["sentiment"])
        alignments[mid] = a

    # 9) Sort best
    print('Ranking titles by alignment...')
    print()
    ranked = sorted(alignments.items(), key=lambda kv: kv[1]["alignment_score"], reverse=True)

    # 10) Present
    print("-" * 60)
    for rank_idx, (mid, a) in enumerate(ranked[:topn], start=1):
        title = id_to_title.get(mid, mid)
        reviews = movie_data[mid]["reviews"]

        top5 = [reviews[i] for i in a["top_reviews_k"][:5]]
        summary = summarize_reviews_with_claude(top5)

        print(f"\nRecommendation #{rank_idx}: {title}")
        print(f"Alignment score: {a['alignment_score']:.4f} (mean_sem={a['mean_sem']:.4f}, mean_sent={a['mean_sent']:.4f}, mean_comp={a['mean_comp']:.4f}, mean_comp_k={a['mean_comp_k']:.4f}, pr_comp={a['pr_comp']:.3f})\n")
        print(f"Overall, many users say: {summary['one_liner']}")
        print()
        print("Themes:")
        for kp in summary["key_points"]:
            print(f"  * {kp}")

        print("\nMost aligned audience reviews:")
        for r in top5:
            print(f"  - {r}\n")

        print("-" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_text", type=str, help="Movie title or review-style description")
    parser.add_argument("--topn", type=int, default=3)
    args = parser.parse_args()

    # Env var sanity (fail loud)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Missing ANTHROPIC_API_KEY")
    if not os.environ.get("TMDB_API_KEY"):
        raise RuntimeError("Missing TMDB_API_KEY")

    asyncio.run(run_pipeline(args.input_text, topn=args.topn))

if __name__ == "__main__":
    main()