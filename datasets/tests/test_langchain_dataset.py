import pytest
import os
from shutil import rmtree
from pathlib import Path
# TODO: test all add functions
# TODO: test similarity search
# TODO: test llm search?

from datasets import PaperDatasetLC
ROOT_PATH = Path(__file__).resolve().parents[2]

# def test_sample_files_exist():
#     """ Check if needed samples for dir exists"""
#     assert os.path.isdir(ROOT_PATH / "sample_documents"), "No Sample Dir"
#     assert os.path.isfile(ROOT_PATH / "sample_documents/2302.00386.pdf")


def test_add_documents():
    # test_chroma_path = ROOT_PATH / "datasets/tests/.chroma"
    # if test_chroma_path.exists():
    #     rmtree(str(test_chroma_path))

    dataset = PaperDatasetLC()
    dataset.add_pdf_file(str(ROOT_PATH / "sample_documents/2302.00386.pdf"))

    sample_text = ["Lorem impsum something something", "Some Text"]
    sample_metas = [{"source": "sth"}, {"other": "sth"}]
    dataset.add_texts(sample_text, sample_metas, skip_invalid=True)

    stored_docs = dataset.list_papers_by_uuid()
    assert len(stored_docs) == 6    # 5 from pdf + one text
    assert any(name == "sth" for _, name in stored_docs)

# def test_search():
#     dataset = PaperDatasetLC()
#     doc_id = dataset.add_text_file("tests/sample_documents/2302.00386.pdf")
#     print(doc_id)
#     docs = dataset.similarity_search("Results", n_results=2)
#     assert len(docs) == 2
#     print(docs[0].metadata)