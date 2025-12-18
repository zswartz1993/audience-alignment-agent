import os
import re
import json
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from anthropic import Anthropic
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from playwright.async_api import async_playwright
import httpx

TMDB_BASE = "https://api.themoviedb.org/3"
RTCF_RE = re.compile(r"/napi/rtcf/v1/movies/([^/]+)/reviews", re.IGNORECASE)


# ---------------------------
# Schemas + Prompts
# ---------------------------

FILTER_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "include_genres": {"type": "array", "items": {"type": "string"}},
    "exclude_genres": {"type": "array", "items": {"type": "string"}},
    "tone_tags":      {"type": "array", "items": {"type": "string"}},
    "keywords":       {"type": "array", "items": {"type": "string"}},
    "year_min": {"type": ["integer", "null"]},
    "year_max": {"type": ["integer", "null"]},
    "production_level": {
      "type": "string",
      "enum": ["micro_indie", "indie", "mid_budget", "blockbuster", "unknown"]
    },
    "min_critic_score":   {"type": ["integer", "null"]},
    "min_audience_score": {"type": ["integer", "null"]},
    "max_results": {"type": "integer"}
  },
  "required": [
    "include_genres",
    "exclude_genres",
    "tone_tags",
    "keywords",
    "production_level",
    "max_results"
  ]
}

RECS_SCHEMA = {
  "type": "array",
  "minItems": 10,
  "maxItems": 10,
  "items": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "title": {"type": "string"},
      "year": {"type": ["integer", "null"]}
    },
    "required": ["title", "year"]
  }
}

OUTPUT_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "input_type": {"type": "string", "enum": ["movie_title", "review", "ambiguous"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "normalized_title": {"type": ["string", "null"]},
    "normalized_year": {"type": ["integer", "null"]},
    "query_review": {"type": "string"},
    "tmdb_filter": FILTER_SCHEMA,
    "recommendations": RECS_SCHEMA
  },
  "required": [
    "input_type", "confidence", "normalized_title", "normalized_year",
    "query_review", "tmdb_filter", "recommendations"
  ]
}

SYSTEM_PROMPT = f"""
You convert user input into:
(1) a query_review (text we will embed for similarity search)
(2) a tmdb_filter JSON object for candidate retrieval
(3) a list of 10 movie recommendations (title + year) that match the query_review

CRITICAL RULES:
- Output ONLY valid JSON.
- Do NOT include explanations, markdown, or comments.
- The JSON MUST conform exactly to this schema:
{json.dumps(OUTPUT_SCHEMA, indent=2)}

Classification:
- input_type="movie_title" if the input is primarily a film title (optionally with a year like "(2016)") and NOT a description.
- input_type="review" if it contains descriptive/opinion language about what they want.
- input_type="ambiguous" if it could plausibly be either (single-word titles like "Heat", "Crash", etc.).

Behavior:
- If input_type="movie_title":
  - normalized_title = the title (strip year if present)
  - normalized_year = parsed year if present else null
  - query_review = write a realistic Rotten Tomatoes *audience* review blurb (2–4 sentences)
    focusing ONLY on tone/pacing/humor/violence/production feel. NO plot details or character names.
  - IMPORTANT: Do NOT euphemize or sanitize. The review must sound like genuine audience reception:
    * If the movie is widely disliked, say so plainly ("boring", "waste of time", "cringey", etc.).
    * If the movie is polarizing, reflect that ("some will love it, others will hate it") without softening.
    * Use strong language if it is typical of audience reviews, but avoid slurs/hate speech.
    * Do not add diplomatic qualifiers like "might not be for everyone" unless that is the core point.
    * Do not default to positivity—match typical audience bluntness.
- If input_type="review":
  - normalized_title = null, normalized_year = null
  - query_review = lightly cleaned version of the user text (preserve meaning)
- If input_type="ambiguous":
  - normalized_title/year = null
  - query_review = treat as review (do not invent a title)
  - confidence should be low.

Filter guidelines:
- Be conservative: do not over-filter.
- If years or scores are not specified, use null.
- Map phrases:
  - "mid level production", "mid budget" → production_level="mid_budget"
  - "blockbuster", "big budget" → "blockbuster"
  - "indie", "low budget" → "indie"
  - "micro indie", "micro-budget" → "micro_indie"
- tone_tags = vibes (funny, gritty, heartfelt, etc.)
- keywords = concepts (buddy-cop, heist, revenge, space, etc.)
- max_results: default 20 if unspecified.

Recommendations guidelines:
- recommendations must contain EXACTLY 10 items.
- Each recommendation must be a MOVIE (not a TV show).
- Include year if you are confident; otherwise use null.
- Avoid extremely obscure picks unless the query_review implies niche/indie.
- Do not include the same movie twice.

Brevity:
- Keep the entire JSON compact.
- query_review must be <= 60 words.
""".strip()


SUMMARY_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "key_points": {
      "type": "array",
      "minItems": 3,
      "maxItems": 6,
      "items": {"type": "string"}
    },
    "one_liner": {"type": "string"}
  },
  "required": ["key_points", "one_liner"]
}

SUMMARY_SYSTEM = f"""
You summarize a small set of Rotten Tomatoes audience reviews into key points.

CRITICAL RULES:
- Output ONLY valid JSON.
- Do NOT include markdown or commentary.
- The JSON MUST conform exactly to this schema:
{json.dumps(SUMMARY_SCHEMA, indent=2)}

Guidelines:
- Be faithful to what is stated. Do not add plot facts.
- No euphemizing: reflect praise/complaints bluntly if present.
- key_points should be short, distinct bullets (max ~12 words each).
- one_liner should be <= 20 words.
""".strip()


# ---------------------------
# Small utilities
# ---------------------------

def extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    return text

def _looks_truncated(t: str) -> bool:
    return (t.count("{") > t.count("}")) or (t.count("[") > t.count("]"))


# ---------------------------
# Claude calls
# ---------------------------

def llm_request_to_filter(user_input: str) -> dict:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    last_err = None
    for attempt in range(2):
        max_tokens = 900 if attempt == 0 else 1600
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_input}],
        )

        raw = resp.content[0].text.strip()
        if _looks_truncated(raw):
            last_err = RuntimeError("Claude output appears truncated; retrying with higher max_tokens.")
            continue

        text = extract_json(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            last_err = RuntimeError(f"LLM did not return valid JSON:\n{raw}")
            continue

        try:
            validate(instance=data, schema=OUTPUT_SCHEMA)
        except ValidationError:
            last_err = RuntimeError(f"JSON does not match schema:\n{text}")
            continue

        return data

    raise last_err if last_err else RuntimeError("Claude failed to return valid JSON.")


def summarize_reviews_with_claude(top_reviews: list[str]) -> dict:
    if not top_reviews:
        return {"key_points": [], "one_liner": ""}

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    payload = {"reviews": top_reviews[:5]}

    last_err = None
    for attempt in range(2):
        max_tokens = 500 if attempt == 0 else 800
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            temperature=0,
            system=SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )

        raw = resp.content[0].text.strip()
        if _looks_truncated(raw):
            last_err = RuntimeError("Claude summary output appears truncated; retrying.")
            continue

        text = extract_json(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            last_err = RuntimeError(f"LLM did not return valid JSON:\n{raw}")
            continue

        try:
            validate(instance=data, schema=SUMMARY_SCHEMA)
        except ValidationError:
            last_err = RuntimeError(f"JSON does not match schema:\n{text}")
            continue

        return data

    raise last_err if last_err else RuntimeError("Claude failed to summarize reviews.")


# ---------------------------
# TMDB
# ---------------------------

def tmdb_get(path: str, params=None) -> dict:
    params = params or {}
    params["api_key"] = os.environ["TMDB_API_KEY"]
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_genre_map() -> dict:
    data = tmdb_get("/genre/movie/list", params={"language": "en-US"})
    return {g["name"].strip().lower(): g["id"] for g in data.get("genres", [])}

def coerce_date_bounds(year_min, year_max):
    gte = f"{year_min}-01-01" if year_min else None
    lte = f"{year_max}-12-31" if year_max else None
    return gte, lte

def apply_production_level(params: dict, production_level: str):
    if production_level == "micro_indie":
        params["vote_count.gte"] = 20
        params["popularity.lte"] = 20
    elif production_level == "indie":
        params["vote_count.gte"] = 200
        params["popularity.lte"] = 40
    elif production_level == "mid_budget":
        params["vote_count.gte"] = 800
        params["popularity.lte"] = 120
    elif production_level == "blockbuster":
        params["vote_count.gte"] = 4000
        params["popularity.gte"] = 80

def build_discover_params(movie_filter: dict, genre_map: dict) -> dict:
    params = {
        "language": "en-US",
        "include_adult": "false",
        "sort_by": "popularity.desc",
        "page": 1,
    }

    include_ids, exclude_ids = [], []
    for g in movie_filter.get("include_genres", []):
        gid = genre_map.get(g.strip().lower())
        if gid: include_ids.append(str(gid))
    for g in movie_filter.get("exclude_genres", []):
        gid = genre_map.get(g.strip().lower())
        if gid: exclude_ids.append(str(gid))

    if include_ids: params["with_genres"] = ",".join(include_ids)
    if exclude_ids: params["without_genres"] = ",".join(exclude_ids)

    gte, lte = coerce_date_bounds(movie_filter.get("year_min"), movie_filter.get("year_max"))
    if gte: params["primary_release_date.gte"] = gte
    if lte: params["primary_release_date.lte"] = lte

    min_critic = movie_filter.get("min_critic_score")
    min_aud = movie_filter.get("min_audience_score")
    mins = [x for x in [min_critic, min_aud] if x is not None]
    if mins:
        params["vote_average.gte"] = round(min(mins) / 10.0, 1)

    params["vote_count.gte"] = 1000

    apply_production_level(params, movie_filter.get("production_level", "unknown"))
    return params

def discover_movies(movie_filter: dict) -> list[dict]:
    genre_map = get_genre_map()
    params = build_discover_params(movie_filter, genre_map)

    max_results = int(movie_filter.get("max_results", 20))
    out = []
    page = 1

    while len(out) < max_results:
        params["page"] = page
        data = tmdb_get("/discover/movie", params=params)
        results = data.get("results", [])
        if not results:
            break
        out.extend(results)

        total_pages = data.get("total_pages", page)
        if page >= total_pages:
            break
        page += 1

    out = out[:max_results]
    return [{
        "id": m.get("id"),
        "title": m.get("title"),
        "release_date": m.get("release_date"),
        "overview": m.get("overview"),
        "vote_average": m.get("vote_average"),
        "vote_count": m.get("vote_count"),
        "popularity": m.get("popularity"),
    } for m in out]


# ---------------------------
# Rotten Tomatoes
# ---------------------------

def slugify_rt(title: str) -> str:
    s = title.lower()
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def url_ok(url: str) -> bool:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code == 200
    except requests.RequestException:
        return False

def find_rt_url(title: str, year: int | None = None) -> str | None:
    slug = slugify_rt(title)
    candidates = [
        f"https://www.rottentomatoes.com/m/{slug}",
        f"https://www.rottentomatoes.com/m/{slug.replace('_','-')}",
    ]
    for u in candidates:
        if url_ok(u):
            return u

    q = f'site:rottentomatoes.com/m "{title}"'
    if year:
        q += f" {year}"

    with DDGS() as ddgs:
        for r in ddgs.text(q, max_results=5):
            href = r.get("href") or r.get("url")
            if not href:
                continue
            if re.search(r"^https?://(www\.)?rottentomatoes\.com/m/", href):
                return href.split("#")[0].split("?")[0]
    return None

async def get_rt_movie_id(rt_movie_url: str, kind: str = "all-audience") -> str | None:
    reviews_url = rt_movie_url.rstrip("/") + f"/reviews/{kind}"
    movie_id = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120 Safari/537.36"
        )
        page = await context.new_page()

        def on_response(resp):
            nonlocal movie_id
            if movie_id:
                return
            m = RTCF_RE.search(resp.url)
            if m:
                movie_id = m.group(1)

        page.on("response", on_response)

        await page.goto(reviews_url, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(2500)
        await page.mouse.wheel(0, 2500)
        await page.wait_for_timeout(1500)

        await browser.close()

    return movie_id

async def get_movie_ids_concurrent_simple(movies: list[dict], concurrency: int = 4) -> dict:
    sem = httpx.AsyncSemaphore(concurrency) if hasattr(httpx, "AsyncSemaphore") else None
    raise RuntimeError("Call get_movie_ids_concurrent_simple in main.py where asyncio.Semaphore is available.")

async def fetch_rtcf_page(client: httpx.AsyncClient, movie_id: str, after: str = "", page_count: int = 10, verified: bool = False):
    url = f"https://www.rottentomatoes.com/napi/rtcf/v1/movies/{movie_id}/reviews"
    params = {
        "after": after, "before": "", "pageCount": str(page_count),
        "topOnly": "false", "type": "audience",
        "verified": "true" if verified else "false",
    }
    r = await client.get(url, params=params)
    r.raise_for_status()
    return r.json()

def extract_texts(payload: dict) -> list[str]:
    reviews = payload.get("reviews") or []
    out = []
    for rv in reviews:
        t = rv.get("review") or rv.get("text") or rv.get("quote") or rv.get("comment") or rv.get("body")
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out

def next_after(payload: dict) -> str | None:
    pi = payload.get("pageInfo") or {}
    return pi.get("endCursor") or pi.get("after")

def has_next(payload: dict) -> bool:
    pi = payload.get("pageInfo") or {}
    return bool(pi.get("hasNextPage"))

async def collect_reviews_fast(movie_id: str, max_reviews: int = 60, page_count: int = 10, verified: bool = False) -> list[str]:
    texts, after = [], ""
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        while len(texts) < max_reviews:
            payload = await fetch_rtcf_page(client, movie_id, after=after, page_count=page_count, verified=verified)
            batch = extract_texts(payload)
            if not batch:
                break
            texts.extend(batch)
            if not has_next(payload):
                break
            na = next_after(payload)
            if not na or na == after:
                break
            after = na
    return texts[:max_reviews]
