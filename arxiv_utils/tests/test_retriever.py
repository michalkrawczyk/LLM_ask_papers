import pytest
from arxiv_utils import ExtendedArxivRetriever


@pytest.mark.parametrize("download_type", ["id", "query", "filter_query"])
def test_download(download_type):
    paper_id = ["1812.01187", "2207.02696"]
    query = "Yolov7"

    retriever = ExtendedArxivRetriever(max_docs=2, save_pdf=False)

    if download_type == "id":
        docs = retriever.get_documents_by_id(paper_id[:1])
        assert len(docs) == 1
    elif download_type == "query":
        docs = retriever.get_relevant_documents(query)
        assert len(docs) == 2
    else:
        docs = retriever.get_filtered_documents_by_query(paper_id, query)
        assert len(docs) == 1
