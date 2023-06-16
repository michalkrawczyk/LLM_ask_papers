from typing import List

from langchain.schema import BaseRetriever, Document
from .wrapper import ArxivAPIWrapper2


class ExtendedArxivRetriever(BaseRetriever, ArxivAPIWrapper2):
    """
    Wrapper for ArxivAPIWrapper2 (Extended ArxivAPIWrapper. Used without changes).
    Wraps load() for obtaining documents with
    get_relevant_documents(), get_documents_by_id() and get_filtered_documents_by_query()
    """

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents by search quarry"""
        # TODO: Should be changed self.sort_docs_by?
        return self.load(query=query)

    def get_documents_by_id(self, id_list: List[str]) -> List[Document]:
        """Get relevant documents by document ids"""
        return self.load(id_list=id_list)

    def get_filtered_documents_by_query(
        self, id_list: List[str], query: str
    ) -> List[Document]:
        """Get relevant documents by document ids matching query"""
        return self.load(query=query, id_list=id_list)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
