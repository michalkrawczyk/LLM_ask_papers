import pytest
import os
from pathlib import Path
# TODO: test all add functions
# TODO: test llm search?

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from datasets import PaperDatasetLC
ROOT_PATH = Path(__file__).resolve().parents[2]

# def test_sample_files_exist():
#     """ Check if needed samples for dir exists"""
#     assert os.path.isdir(ROOT_PATH / "sample_documents"), "No Sample Dir"
#     assert os.path.isfile(ROOT_PATH / "sample_documents/2302.00386.pdf")


def test_add_documents():
    embeddings = HuggingFaceEmbeddings()
    dataset = PaperDatasetLC(db=Chroma(embedding_function=embeddings))
    dataset.add_pdf_file(str(ROOT_PATH / "sample_documents/2302.00386.pdf"))

    sample_text = ["Lorem impsum something something", "Some Text"]
    sample_metas = [{"source": "sth"}, {"other": "sth"}]
    dataset.add_texts(sample_text, sample_metas, skip_invalid=True)

    stored_docs = dataset.list_papers_by_uuid()
    assert len(stored_docs) == 6    # 5 from pdf + one text
    assert any(name == "sth" for _, name in stored_docs)


#TODO:
# def test_search():
#     dataset = PaperDatasetLC(db=Chroma(embedding_function=HuggingFaceEmbeddings()))
#     doc1 = dataset.add_papers_by_id(["2301.05586"])
#     assert len(doc1) > 0
#     doc2 = dataset.add_documents_by_query(query="yolov7", max_docs=1)
#
#     assert len(doc2) > 0
#     docs = dataset.similarity_search("Results", n_results=2)
#
#     assert len(docs) == 2


