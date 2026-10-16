"""Chinese/Latin BM25 and reciprocal rank fusion, independent of model SDKs."""

from collections import Counter, defaultdict
from dataclasses import replace
import math
import re
import unicodedata

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk


def tokenize(text: str) -> list[str]:
    """CJK characters + bigrams preserve Chinese terms; retain complete SKU tokens."""
    parts = re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+(?:[-_.][a-z0-9]+)*",
                       unicodedata.normalize("NFKC", text).lower())
    tokens = []
    for part in parts:
        if "\u3400" <= part[0] <= "\u9fff":
            tokens.extend(part)
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = defaultdict(list)
        for index, chunk in enumerate(chunks):
            terms = Counter(tokenize(f"{chunk.doc} {chunk.section} {chunk.text}"))
            self.lengths.append(sum(terms.values()))
            for term, count in terms.items():
                self.postings[term].append((index, count))
        self.average = sum(self.lengths) / max(len(chunks), 1) or 1.0

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (len(self.chunks) - len(postings) + .5) / (len(postings) + .5))
            for index, frequency in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = sorted(scores, key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        return [RetrievedChunk(self.chunks[i], scores[i], score_type="bm25",
                               bm25_score=scores[i]) for i in ranked[:max(top_k, 0)]]


def reciprocal_rank_fusion(dense: list[RetrievedChunk], sparse: list[RetrievedChunk],
                           k: int = 60) -> list[RetrievedChunk]:
    if k <= 0:
        raise ValueError("RRF k 必须大于 0")
    merged = {}
    for channel, hits in (("dense", dense), ("bm25", sparse)):
        seen = set()
        for hit in hits:
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            if key not in merged:
                merged[key] = replace(hit, score=0.0, score_type="rrf",
                                      dense_score=None, bm25_score=None, rrf_score=0.0)
            result = merged[key]
            result.score += 1.0 / (k + len(seen))
            result.rrf_score = result.score
            setattr(result, f"{channel}_score", hit.score)
    return sorted(merged.values(), key=lambda h: (-h.score, h.chunk.chunk_id))
