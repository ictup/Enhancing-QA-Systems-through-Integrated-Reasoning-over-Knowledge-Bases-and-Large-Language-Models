def test_imports():
    import src
    import src.retrieval.bm25 as bm25
    assert callable(bm25.build_bm25)
