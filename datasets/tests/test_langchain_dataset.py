import pytest
import os
from pathlib import Path
# TODO: source pdf for testing add functions
# TODO: test all add functions
# TODO: test similarity search
# TODO: test llm search?

from datasets import PaperDatasetLC
# import openai
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
ROOT_PATH = Path(__file__).resolve().parent.parent.parent

# def test_sample_files_exist():
#     """ Check if needed samples for dir exists"""
#     assert os.path.isdir(ROOT_PATH / "sample_documents"), "No Sample Dir"
#     assert os.path.isfile(ROOT_PATH / "sample_documents/2302.00386.pdf")


def test_add_documents():
    dataset = PaperDatasetLC()
    doc_id = dataset.add_pdf_file(str(ROOT_PATH / "sample_documents/2302.00386.pdf"))
    sample_text = ["Lorem impsum something something", "Some Text"]
    sample_metas = [{"source": "sth"}, {"other": "sth"}]
    other_ids = dataset.add_texts(sample_text, sample_metas, skip_invalid=True)

    dataset.
    # print(doc_id)
    print(other_ids)
    print(dataset.similarity_search("Lorem"))

    print(dataset.list_of_papers())

# def test_search():
#     dataset = PaperDatasetLC()
#     doc_id = dataset.add_text_file("tests/sample_documents/2302.00386.pdf")
#     print(doc_id)
#     docs = dataset.similarity_search("Results", n_results=2)
#     assert len(docs) == 2
#     print(docs[0].metadata)