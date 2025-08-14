from typing import List
from rank_bm25 import BM25Okapi

def build_bm25(corpus: List[str]) -> BM25Okapi:
    tokenized_corpus = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)

def search(bm25: BM25Okapi, query: str, k: int = 5):
    return bm25.get_top_n(query.split(), bm25.corpus, n=k)
