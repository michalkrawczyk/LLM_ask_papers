from typing import List

from langchain.schema import BaseRetriever, Document
from .wrapper import ArxivAPIWrapper2


class ExtendedArxivRetriever(BaseRetriever, ArxivAPIWrapper2):
    def get_relevant_documents(self, query: str) -> List[Document]:
        """ Get relevant documents by search quarry"""
        # TODO
        pass

    def get_documents_by_id(self, id_list: List[int]) -> List[Document]:
        """ Get relevant documents by document ids"""
        # TODO
        pass

    def filter_documents_by_query(self, id_list: List[str], query: str) -> List[Document]:
        """ Get relevant documents by document ids matching query"""
        # TODO
        pass

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
