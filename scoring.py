import numpy as np

def embed_texts(texts, model, batch_size=64):
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)

def batch_sentiment_scores(texts, sentiment_model, batch_size=64):
    scores = np.zeros(len(texts), dtype=np.float32)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = sentiment_model(
            batch,
            truncation=True,
            max_length=512,
            padding=True,
        )
        for j, out in enumerate(outputs):
            probs = {d["label"].lower(): d["score"] for d in out}
            scores[i + j] = probs.get("positive", 0.0) - probs.get("negative", 0.0)

    return scores

def similarity_semantic(query_embedding, review_embeddings):
    # embeddings normalized => dot = cosine
    return review_embeddings @ query_embedding

def similarity_sentiment(query_score, review_scores):
    # convert difference to [-1, 1]
    return 1.0 - np.abs(query_score - review_scores)

def similarity_composite(sim_sem, sim_sent):
    return sim_sem * (0.7 + 0.3 * sim_sent)

def compute_alignment(query_emb, review_embs, query_sent, review_sents, alpha=0.7, tau=0.3, k=10):
    sims_sem = similarity_semantic(query_emb, review_embs)
    sims_sent = similarity_sentiment(query_sent, review_sents)
    sims_comp = similarity_composite(sims_sem, sims_sent)

    order = np.argsort(-sims_comp)  # descending
    top_k_idx = order[:k].tolist()

    mean_comp_k = float(np.mean(sims_comp[top_k_idx])) if len(top_k_idx) else float(np.mean(sims_comp))
    above_tau_idx = np.where(sims_comp > tau)[0].tolist()
    pr_comp = float(len(above_tau_idx) / len(sims_comp)) if len(sims_comp) else 0.0

    alignment_score = alpha * mean_comp_k + (1 - alpha) * pr_comp

    return {
        "size": int(len(review_embs)),
        "mean_sem": float(np.mean(sims_sem)),
        "mean_sent": float(np.mean(sims_sent)),
        "mean_comp": float(np.mean(sims_comp)),
        "mean_comp_k": float(mean_comp_k),
        "pr_comp": float(pr_comp),
        "alignment_score": float(alignment_score),
        "top_reviews_k": top_k_idx,
        "top_reviews_tau": above_tau_idx,
    }
