# Audience Alignment Agent

## Overview

This project implements a lightweight expert agent that recommends movies based on how closely Rotten Tomatoes audience reviews align with user input. User input can take on two forms:
1. **A movie description**
2. **A movie title**

Rather than relying solely on an LLM to “pick” a movie, the system:
1. Uses an LLM to **interpret user intent**
3. Retrieves **candidate movies** from an LLM
4. Retrieves **candidate movies** from TMDB
5. Collects **real audience reviews** from Rotten Tomatoes
6. Measures **semantic + sentiment alignment with user intent**
7. Ranks movies using **explainable alignment metrics**
8. Summarizes the most aligned reviews for interpretability

The result is a recommendation pipeline that is:
- data-driven
- reproducible
- interpretable

---

## High-Level Architecture

**Input (movie title or review-style description)**  
↓  
**LLM (Claude)**  
• Classifies input (title vs review)  
• Produces a normalized “query review”  
• Generates a conservative TMDB filter  
• Suggests 10 additional candidate movies  
↓  
**TMDB Candidate Retrieval**  
↓  
**Rotten Tomatoes Audience Review Collection**  
↓  
**Embedding + Sentiment Analysis (local models)**  
↓  
**Alignment Scoring & Ranking**  
↓  
**LLM Summary of Most Aligned Reviews**

---

## Why This Design

### LLMs are used only where needed
- Interpreting ambiguous natural language
- Normalizing user intent
- Generating candidates
- Summarizing human feedback

They are **not** used for scoring, ranking, or evaluation.

### Decisions are made using measurable signals
Movie ranking is driven by:
- semantic similarity between user intent and audience reviews
- sentimental similarity between user intent and audience reviews
- distributional metrics over real review data

This approach minimizes halucinated and non-deterministic outputs, reduces the Claude API footprint, and ensures reccomendations are data driven. 

---

## Alignment Metric

For each movie, we compute:

### 1. Semantic similarity
Cosine similarity between:
- embedding(query_review)
- embeddings(audience_reviews)

### 2. Sentiment similarity
Distance between:
- sentiment(query_review)
- sentiment(audience_reviews)

### 3. Composite similarity (per review)

composite = semantic_similarity * (0.7 + 0.3 * sentiment_similarity)

### 4. Movie-level alignment
We combine:
- mean composite similarity of top-K reviews
- proportion of reviews above a threshold τ

This balances **strength of alignment** with **consistency across reviewers**.

---

## Evaluation Strategy

The system exposes multiple interpretable metrics per movie:
- mean semantic similarity
- mean sentiment similarity
- mean composite similarity
- top-K composite mean
- proportion of reviews exceeding τ
- final alignment score

These metrics make it easier to compare movies and understand failure modes

---

## Interpretability

For each recommended movie, the system returns:
- a one-line “audience consensus” description
- a short Claude-generated summary of key themes
- the most aligned audience reviews

This allows users to better understand why a movie was recommended.

---

## Running the Project

### Setup

python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  
python -m playwright install chromium  

Set environment variables:

export ANTHROPIC_API_KEY="..."  
export TMDB_API_KEY="..."  

### Run

python main.py "Realistic cybersecurity movie"  

Or using a title:

python main.py "Zero Days (2016)"

Set number of reccomendations (default = 3):

python main.py "bleak post-apocalyptic thriller" --topn 2  

---

## Project Structure

.
├── main.py        # Orchestration + CLI  
├── clients.py     # External systems (Claude, TMDB, Rotten Tomatoes)  
├── scoring.py     # Embeddings, sentiment, alignment metrics  
├── requirements.txt  
└── README.md  