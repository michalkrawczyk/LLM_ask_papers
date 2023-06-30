import pytest

from datasets import PaperDatasetLC

def test_add_documents():
    pass

def test_search():
    dataset = PaperDatasetLC()
    doc_id = dataset.add_text_file("tests/sample_documents/2302.00386.pdf")
    print(doc_id)
    docs = dataset.similarity_search("Results", n_results=2)
    assert len(docs) == 2
    print(docs[0].metadata)
