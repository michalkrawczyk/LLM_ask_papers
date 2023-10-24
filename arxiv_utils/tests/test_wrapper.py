import pytest
from arxiv_utils import ArxivAPIWrapper2

# TODO: Ignore for Arxiv HTTP exception?


def test_download():
    max_docs = 1
    wrapper = ArxivAPIWrapper2(max_docs=max_docs, save_pdf=False)
    docs = wrapper.load("Deep Learning")
    assert len(docs) <= max_docs, "Download Error - Exceeded maximum number of docs"

    for doc in docs:
        assert doc.metadata["title"], "Download Error - Paper downloaded without title"

        # if wrapper.save_pdf:
        #     assert doc.metadata["file_path"], "Download Error - File Path unspecified"
