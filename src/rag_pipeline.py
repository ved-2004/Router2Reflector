"""
RAG Retrieval Pipeline for R2R
==============================
Cascading query extraction, Wikipedia opensearch API,
and embedding-based similarity filtering.

Usage:
    from rag_pipeline import RAGRetriever, SimilarityFilter

    retriever = RAGRetriever()
    sim_filter = SimilarityFilter(embed_model)

    passages = retriever.retrieve(question)
    filtered, sims = sim_filter.filter_passages(question, passages)
"""

import re
import sys
import time
import requests
import numpy as np

# Wikipedia rejects the default python-requests User-Agent with HTTP 403, which the
# original code swallowed silently (so no passages were ever retrieved). Always send one.
USER_AGENT = "R2R-NLP-CSCI544/1.0 (research project)"


class RAGRetriever:
    """Retrieves passages from Wikipedia using cascading query extraction."""

    API = "https://en.wikipedia.org/w/api.php"

    def __init__(self, top_k=3, max_retries=8, min_interval=0.25, search_mode="opensearch"):
        """search_mode: "opensearch" = original title-prefix search; "fulltext" = MediaWiki full-text
        search (list=search), which matches article bodies, not just titles."""
        if search_mode not in ("opensearch", "fulltext"):
            raise ValueError(search_mode)
        self.search_mode = search_mode
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.max_retries = max_retries
        self.min_interval = min_interval   # seconds between requests (serial, polite crawling)
        self._last_request = 0.0
        self.n_errors = 0
        self.n_throttled = 0               # 429/5xx responses that were waited out and retried
        self.last_failed = False           # True if any request in the last retrieve() call failed
        self.query_delay = 0.1             # pause between cascade queries (live API only)
        self.top_k = top_k
        self.stopwords = {
            "what","which","who","whom","where","when","why","how",
            "is","are","was","were","the","a","an","of","in","to",
            "for","and","or","that","this","with","from","by","on",
            "at","does","did","do","has","have","had","be","been",
            "being","it","its","not","no","can","could","would",
            "should","will","shall","may","might","about","most",
            "following","true","false","correct","incorrect","many",
            "much","some","any","all","each","every","both","few",
            "more","other","than","then","also","just","only",
            "very","well","even","still","already","often","never",
            "always","sometimes","usually","really","actually",
            "between","during","before","after","above","below",
            "under","over","into","through","against","among",
            "these","those","such","like","used","known","called",
        }

    def extract_queries(self, question):
        """Extract search queries using cascading strategy:
        1. Named entities (capitalized phrases)
        2. Content keywords (stopword-filtered)
        3. Cleaned question fallback
        """
        queries = []

        # Strategy 1: Named entities
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        entities = re.findall(cap_pattern, question[:500])
        words = question[:500].split()
        starters = set()
        for i, w in enumerate(words):
            if i == 0 or (i > 0 and words[i-1].endswith(('?', '.', '!'))):
                starters.add(w.strip("?,.'\""))
        entities = [e for e in entities if e.split()[0] not in starters]
        if entities:
            queries.append(" ".join(entities[:3]))
            for e in entities[:2]:
                if len(e) > 2:
                    queries.append(e)

        # Strategy 2: Content keywords
        all_words = re.findall(r'\b[A-Za-z]+\b', question[:500])
        keywords = [w for w in all_words if w.lower() not in self.stopwords and len(w) > 2]
        if keywords:
            queries.append(" ".join(keywords[:5]))

        # Strategy 3: Cleaned question
        clean_q = re.sub(
            r'^(what|which|who|where|when|how|why|is|are|was|were|does|did|do)\s+',
            '', question[:200].lower(), flags=re.IGNORECASE
        ).strip('?. ')
        if clean_q and len(clean_q) > 5:
            queries.append(clean_q[:80])

        # Deduplicate
        seen = set()
        unique = []
        for q in queries:
            ql = q.lower().strip()
            if ql not in seen and len(ql) > 2:
                seen.add(ql)
                unique.append(q)
        return unique[:4]

    def _get(self, params):
        """GET the MediaWiki API with pacing, Retry-After-aware backoff on 429/5xx, and error logging.
        Returns parsed JSON, or None (and sets last_failed) after max_retries."""
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()
            try:
                resp = self.session.get(self.API, params=params, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    self.n_throttled += 1
                    retry_after = resp.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else min(60, 2 ** attempt)
                    time.sleep(delay)
                    continue
                print(f"[rag] HTTP {resp.status_code} for {params.get('search') or params.get('titles')!r}",
                      file=sys.stderr)
                break
            except Exception as exc:  # network / JSON errors: back off and retry
                print(f"[rag] request error ({exc}); retrying", file=sys.stderr)
                time.sleep(min(60, 2 ** attempt))
        self.n_errors += 1
        self.last_failed = True
        return None

    def search_wikipedia(self, query):
        """Search Wikipedia and return the intro ("summary", first 600 chars) of the top 3 pages.

        ONE request per query: a search generator combined with ``prop=extracts&exintro`` (the same
        lead-section text that ``wikipediaapi``'s ``page.summary`` returns).
          - "opensearch" mode uses ``generator=prefixsearch`` — the title-prefix backend that the
            original ``action=opensearch`` call uses.
          - "fulltext" mode uses ``generator=search`` (CirrusSearch over article text).
        Redirects are resolved, so the returned title is the target article's title.
        """
        if self.search_mode == "opensearch":
            gen = {"generator": "prefixsearch", "gpssearch": query, "gpslimit": 3, "gpsnamespace": 0}
        else:
            gen = {"generator": "search", "gsrsearch": query, "gsrlimit": 3, "gsrnamespace": 0}
        data = self._get({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                          "exlimit": 3, "redirects": 1, "format": "json", **gen})
        pages = list(((data or {}).get("query") or {}).get("pages", {}).values())
        pages.sort(key=lambda pg: pg.get("index", 99))
        results = []
        for pg in pages[:3]:
            summary = (pg.get("extract") or "").strip()
            if summary and "missing" not in pg:
                results.append((pg["title"], summary[:600]))
        return results

    def retrieve(self, question):
        """Retrieve passages for a question.
        Returns list of (title, text) tuples.
        """
        self.last_failed = False
        queries = self.extract_queries(question)
        passages, seen = [], set()
        for query in queries:
            for title, text in self.search_wikipedia(query):
                if title.lower() not in seen:
                    seen.add(title.lower())
                    passages.append((title, text))
            if len(passages) >= self.top_k:
                break
            if self.query_delay:
                time.sleep(self.query_delay)
        return passages[:self.top_k]


class OfflineWikiRetriever(RAGRetriever):
    """Same interface as RAGRetriever, backed by a local Wikipedia snapshot (no network, no rate limits).

    Built by ``modal_app/r2r_modal.py::build_wiki_index`` from HF ``wikimedia/wikipedia`` (20231101.en).
    A passage is the article's first 600 characters (the lead section, as ``page.summary[:600]``).

    search_mode="opensearch": title-prefix search (case-insensitive), the offline analogue of the
        original ``action=opensearch``; ranked exact-title first, then by article length (a popularity
        proxy, since the live endpoint ranks by popularity). Uses the original cascading queries.
    search_mode="fulltext": BM25 (bm25s, English stemming + stopwords) over title + lead text,
        queried with the whole question (+ passage for ReClor), the standard BM25-RAG setup.
    """

    def __init__(self, index_dir, top_k=3, search_mode="opensearch"):
        import pickle
        super().__init__(top_k=top_k, search_mode=search_mode)
        self.query_delay = 0.0
        index_dir = str(index_dir)
        with open(f"{index_dir}/docs.pkl", "rb") as f:
            d = pickle.load(f)
        self.titles, self.leads, self.lengths = d["titles"], d["leads"], d["lengths"]
        self.sorted_lower, self.order = d["sorted_lower"], d["order"]
        self.bm25 = None
        if search_mode == "fulltext":
            import bm25s
            import Stemmer
            self.bm25 = bm25s.BM25.load(f"{index_dir}/bm25", mmap=True)
            self.stemmer = Stemmer.Stemmer("english")

    def _prefix(self, query):
        import bisect
        ql = query.lower().strip()
        if not ql:
            return []
        lo = bisect.bisect_left(self.sorted_lower, ql)
        mid = bisect.bisect_right(self.sorted_lower, ql)            # [lo, mid) = exact title matches
        hi = bisect.bisect_left(self.sorted_lower, ql + "\uffff")   # [mid, hi) = proper prefix matches
        exact = self.order[lo:mid]
        rest = self.order[mid:hi]
        if len(rest) > 3:
            top = np.argpartition(-self.lengths[rest], 2)[:3]
            rest = rest[top[np.argsort(-self.lengths[rest][top], kind="stable")]]
        return [int(i) for i in np.concatenate([exact, rest])[:3]]

    def _bm25(self, query, k=3):
        import bm25s
        # token strings (not ids) so bm25s maps them onto the INDEX vocabulary
        toks = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=False,
                              show_progress=False)
        toks = [[t for t in toks[0] if t in self.bm25.vocab_dict]]
        if not toks[0]:
            return []
        docs, scores = self.bm25.retrieve(toks, k=k, show_progress=False)
        return [int(i) for i, sc in zip(docs[0], scores[0]) if sc > 0]

    def search_wikipedia(self, query):
        idx = self._prefix(query) if self.search_mode == "opensearch" else self._bm25(query)
        return [(self.titles[i], self.leads[i]) for i in idx if self.leads[i].strip()]

    def retrieve(self, question):
        self.last_failed = False
        if self.search_mode == "fulltext":
            return self.search_wikipedia(question[:2000])[: self.top_k]
        return super().retrieve(question)


class SimilarityFilter:
    """Filters retrieved passages using embedding-based cosine similarity."""

    def __init__(self, embed_model, threshold=0.3):
        self.model = embed_model
        self.threshold = threshold

    def filter_passages(self, question, passages):
        """Filter passages by cosine similarity to the question.
        Returns (filtered_passages, similarities).
        """
        if not passages:
            return [], []
        q_emb = self.model.encode(question[:512], normalize_embeddings=True)
        filtered, sims = [], []
        for title, text in passages:
            p_emb = self.model.encode(text[:300], normalize_embeddings=True)
            sim = float(np.dot(q_emb, p_emb))
            sims.append(sim)
            if sim >= self.threshold:
                filtered.append((title, text))
        return filtered, sims


def get_rag_context(question, retriever, sim_filter, max_chars=1500):
    """Convenience function: retrieve and filter in one call.
    Returns context string (empty if nothing passes filter).
    """
    passages = retriever.retrieve(question)
    filtered, _ = sim_filter.filter_passages(question, passages)
    if filtered:
        parts = [f"[{title}]\n{text}" for title, text in filtered]
        context = "\n\n".join(parts)
        return context[:max_chars] if len(context) > max_chars else context
    return ""
