from arxiv_utils import (
    # download_paper_from_arxiv,
    # download_recent_papers_by_querry,
    ExtendedArxivRetriever,
    # PaperData,
)
# from gpt_core import get_description_json

from langchain.base_language import BaseLanguageModel
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.schema import Document
from langchain.vectorstores import Chroma, VectorStore
import openai

# from tqdm import tqdm

from enum import Enum
import logging
import os
import re
from typing import (
    Any, Dict, List,
    # Union,
    Tuple,  Optional, Iterable)


logger = logging.getLogger(__name__)


def _get_document_name(document: Document):
    """Get name of document (Title or source)"""
    if document.metadata:
        title = document.metadata.get("title")
        name = None

        if not title or title == "Unknown Text":
            name = document.metadata.get("source") \
                   or document.metadata.get("file_path")

        if name or title:
            return name or title

    return "Unknown"


class SearchType(Enum):
    """ Search Type Enum for VectorStore.search()"""
    # TODO: Delete if restrict only Chroma in _db
    MMR = "mmr"
    SIMILARITY = "similarity"


class PaperDatasetLC:
    _db: Optional[VectorStore] = None
    _papers: Dict = dict()  # For listing included documents, retrieve whole documents
    _default_llm: Optional[BaseLanguageModel] = None

    # Also because of limits made from langchain on get() function

    def __init__(self, db: Optional[VectorStore] = None,
                 llm: Optional[BaseLanguageModel] = None):
        if not db:
            if llm:
                raise ValueError("PaperDatasetLC cannot be instantiated "
                                 "if only llm model is given,"
                                 " without proper embeddings")

            logger.info("Dataset with not specified db"
                        " - using Chroma with OpenAI embeddings")
            self._db = Chroma(embedding_function=OpenAIEmbeddings(
                openai_api_key=openai.api_key))
            self._default_llm = OpenAI(temperature=0, openai_api_key=openai.api_key)

        else:
            self._db = db
            self._default_llm = llm

            if self._default_llm:
                logger.warning("LLM model not provided -"
                               " search functions requiring it will not be available")

    def add_document(
            self, document: Document, metadata: Optional[Dict] = None
    ) -> List[str]:
        """Add Document object to vector database

        Parameters
        ----------
        document:
            Document Object
        metadata: Optional Dict
            Document additional metadata. Used to fill missing fields if necessary.

        Returns
        -------
        doc_uuids: List[str]
            Uuid in list for added document in vector database

        """
        if metadata:
            for key in metadata:
                if key not in document.metadata or not document.metadata["key"]:
                    # Add missing data
                    document.metadata[key] = metadata["key"]

        try:
            doc_uuids = self._db.add_documents([document])
            self._papers[doc_uuids[0]] = document

            return doc_uuids

        except Exception as err:
            logger.info(
                f"Failed to add Document {document.metadata.get('title', 'Unnamed Document')} \n"
                f"  {err}"
            )
            return []  # No Object added - return empty list

    def add_pdf_file(self, filepath: str, metadata: Optional[Dict] = None) -> List[str]:
        """Add pdf file to vector database

        Parameters
        ----------
        filepath: str
            Path to text
        metadata: Optional[Dict]
            Document additional metadata.

        Returns
        -------
        doc_uuids: List[str]
            Uuid in list for added document with text in vector database

        """
        if not filepath.endswith(".pdf"):
            f"Currently not supported format {os.path.splitext(filepath)[-1]}"
            f" for file: {filepath}"

        try:
            data = PyMuPDFLoader(filepath).load()

            if metadata:
                for page in data:
                    for key in metadata.keys():
                        if key not in page.metadata or not page.metadata[key]:
                            # Add missing data
                             page.metadata[key] = metadata["key"]

            doc_uuids = self._db.add_documents(data)
            self._papers.update({uid: doc for uid, doc in zip(doc_uuids, data)})

            return doc_uuids

        except Exception as err:
            logger.info(f"Failed to add Document {filepath} \n" f"  {err}")
            return []  # No Object added - return empty list

    def add_texts(
            self, texts: Iterable[str], metadatas: List[Dict],
            skip_invalid: bool = False
    ) -> List[str]:
        """Add multiple texts as separate documents to vector database

        Parameters
        ----------
        texts: Iterable[str]
            Iterable container with texts to add
        metadatas: Iterable[Dict]
            Iterable container with metadata for each text to add
        skip_invalid: bool
            If True - Only texts with errors will be skipped during add to database operation.
            Otherwise -  None of text will be added if error occurs in one of them.

        Returns
        -------
        doc_uuids: List[str]
            Uuids in list for added text documents in vector database

        """
        try:
            if len(texts) != len(metadatas):
                raise ValueError("Number of metadata object must match number of texts")
        except ValueError as err:
            logger.info(f"Failed to add texts to dataset: {err}")
            return []  # No Object added - return empty list

        valid_records = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            try:
                if not meta.get("source"):
                    # Empty or None source metadata
                    raise ValueError(
                        "Paper Dataset not accept texts without specified source in metadata \n"
                        f"Index of problematic record: '{i}'"
                    )

                if not meta.get("title"):
                    meta["title"] = "Unknown Text"

                valid_records.append(Document(page_content=text, metadata=meta))

            except ValueError as err:
                logger.info(err)

                if not skip_invalid:
                    logger.info(
                        "Adding texts to datasets skipped due to error and 'skip_invalid' flag value"
                    )
                    return []  # No Object added - return empty list

            try:
                doc_uuids = self._db.add_documents(valid_records)
                self._papers.update(
                    {uid: doc for uid, doc in zip(doc_uuids, valid_records)}
                )

                return doc_uuids

            except Exception as err:
                logger.info(f"Failed to add texts \n" f"  {err}")
                return []  # No Object added - return empty list

    def add_papers_by_id(self, id_list: Iterable[str], **kwargs: Any) -> List[str]:
        """Search on arxiv papers by ID and add them to vector database

        Parameters
        ----------
        id_list:
            Iterable container of paper ids to find
        kwargs:
            Additional kwargs for arxiv_utils.ExtendedArxivRetriever object

        Returns
        -------
        doc_uuids: List[str]
            Uuids in list for added documents in vector database


        """
        retriever = ExtendedArxivRetriever(**kwargs)
        try:
            docs = retriever.get_documents_by_id(id_list)

            doc_uuids = self.add_documents(docs)
            self._papers.update({uid: doc for uid, doc in zip(doc_uuids, docs)})

            return doc_uuids

        except Exception as err:
            logger.info(f"Failed to add papers: {err}")

        return []  # No Object added - return empty list

    def add_documents_by_query(self, query: str, **kwargs: Any) -> List[str]:
        """Search on arxiv papers by search query and add them to vector database

        Parameters
        ----------
        query: str
           search_query for Arxiv
        kwargs:
            Additional kwargs for arxiv_utils.ExtendedArxivRetriever object

        Returns
        -------
        doc_uuids: List[str]
            Uuids in list for added documents in vector database

        """

        retriever = ExtendedArxivRetriever(**kwargs)
        try:
            docs = retriever.get_relevant_documents(query)
            doc_uuids = self.add_documents(docs)

            self._papers.update({uid: doc for uid, doc in zip(doc_uuids, docs)})
            return doc_uuids

        except Exception as err:
            logger.info(f"Failed to add documents: {err}")

        return []  # No Object added - return empty list

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add Multiple documents to vector database"""
        try:
            doc_uuids = self._db.add_documents(documents)
            self._papers.update({uid: doc for uid, doc in zip(doc_uuids, documents)})
            return doc_uuids

        except Exception as err:
            logger.info(f"Failed to add documents: {err}")

        return []  # No Object added - return empty list

    # def add_page(self, url: str):
    #     #TODO
    #     pass

    # def add_youtube_video(self, url: str) -> List[str]:
    #     #TODO
    #     raise NotImplementedError()

    def list_of_papers(self, regex_filter: Optional[str] = None) -> List[str]:
        """List of papers stored in vector database"""
        if regex_filter:
            # TODO: Change to funcion type instead of regex?
            list(
                set(
                    _get_document_name(doc)
                    for doc in filter(
                        lambda txt: re.match(regex_filter, txt), self._papers.values()
                    )
                )
            )

        return list(set(_get_document_name(doc) for doc in self._papers.values()))

    def list_papers_by_uuid(self) -> List[Tuple[str, str]]:
        """Shortened List of papers stored in vector database by uuid"""
        # TODO: Filters?
        papers = []
        for uid, doc in self._papers.items():
            title = _get_document_name(doc)

            if doc.metadata.get("page"):
                title += f" page {doc.metadata['page']}"

            papers.append((uid, title))

        return papers

    def similarity_search(
            self, query: str, n_results: int = 3,
            search_type: SearchType = SearchType.MMR,
            filter: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """Search by similarity of embeddings

        Parameters
        ----------
        query: str
            Search Query
        n_results: int
            Limit results to 'n' documents
        search_type: SearchType
            Type of search:
            - SearchType.MMR for max relevance search
            - SearchType.SIMILARITY for similarity search
        filter: Optional[Dict[str, str]]
            Optional filters for database (If Chroma)

        Returns
        -------
        List[Document]
            List of relevant Documents

        """
        if filter and not isinstance(self._db, Chroma):
            logger.warning(
                "Filter option was not tested for other vector storages"
                " and may not work with other databases than Chroma"
            )

        return self._db.search(query=query, search_type=search_type.value, k=n_results, filter=filter)

    def similiraty_search_with_scores(
            self, query: str, n_results: int = 3,
            score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """

        Parameters
        ----------
        query: str
        n_results: int
        score_threshold: float

        Returns
        -------
        List[Tuple[Document, float]]
            List of tuples (Relevant document, score)

        """
        if filter and not isinstance(self._db, Chroma):
            logger.warning(
                "Filter option was not tested for other vector storages and may not work with other databases than Chroma"
            )

        return self._db.similarity_search_with_relevance_scores(
            query=query, k=n_results, filter=filter, score_threshold=score_threshold
        )

    def llm_search(
            self, query: str, llm: Optional[BaseLanguageModel] = None, **kwargs: Any
    ) -> str:
        """LLM search engine for searching content in documents

        Parameters
        ----------
        query: str
            Search Query
        llm: Optional[BaseLanguageModel]
            LLM model to use. If not provided OpenAI default model is used
        kwargs: Any
            Keyword arguments for 'RetrievalQA.from_chain_type' method

        Returns
        -------
        str
            LLM response

        """
        llm = llm or self._default_llm
        if not llm:
            raise RuntimeError("LLM not provided")

        chain = RetrievalQA.from_chain_type(
            llm, retriever=self._db.as_retriever(), **kwargs
        )
        return chain.run(query)

    def llm_search_with_sources(
            self, query: str, llm: Optional[BaseLanguageModel] = None, **kwargs: Any
    ) -> dict:
        """LLM search engine for searching content in documents with sources

        Parameters
        ----------
        query
        llm
        kwargs

        Returns
        -------
        dict
            Dictionary containing response and relevant source documents

        """
        llm = llm or self._default_llm
        if not llm:
            raise RuntimeError("LLM not provided")

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, retriever=self._db.as_retriever(), **kwargs
        )
        return chain({chain.question_key: query})

    def list_containing_field(self, field_name: str, include: Optional[List[str]] = None):
        # TODO: Docs
        return self.get(where={field_name: {"$ne": ""}},
                        include=include)
    @property
    def list_new_features(self):
        # TODO: Should be added here identify_features with reload param?
        return self.get(where={"new_features": {"$ne": ""}})

    # def filter_by_categories(self, category: Union[Tuple[str], str]):
    # TODO:
    #     pass
    def get(self, **kwargs):
        if isinstance(self._db, Chroma):
            return self._db.get(**kwargs)

    def identify_features(self, llm: Optional[BaseLanguageModel] = None, limit_pages: Optional[int] = None, ):
        # TODO: loop over docs and add features and other keys identified by LM
        # TODO: limit_pages - analyze only 'n' first pages and add keys to whole
        # TODO: skip already added in metadata
        # TODO: Think about extracting 'Abstract' and "Results" to analyze
        # TODO: Maybe scan n_pages after "Abstract" and after "Results"?

        # TODO: 17.08
        # TODO: Idea: Keywords can be reduced without llm - just string/set filter
        pass

    @staticmethod
    def get_document_info(document: Document):
        """Get document metadata (info)"""
        return document.metadata

    def summarize_paper(self, paper_name):
        #TODO: Summarize with LLMChain and map reduce
        #TODO: Idea: Keywords can be reduced without llm - just string/set filter
        pass


# class PaperDataset:
#     _papers = dict()
#
#     def add_papers_by_id(self, id_list: List[str], output_dir: str = "."):
#         """Download and add to dataset papers from Arxiv by ID
#
#         Parameters
#         ----------
#         id_list: List[str]
#             List with ids for each paper to download
#
#         output_dir: str
#             Directory where documents will be saved
#
#         Returns
#         -------
#         self
#             self obejct for chaining
#
#         """
#         downloaded_papers = set(download_paper_from_arxiv(id_list, output_dir))
#         self._add_papers_by_filepath(downloaded_papers)
#
#         return self
#
#     def search_and_add_papers(
#             self, search_query: str, limit: float = 10.0, output_dir: str = "."
#     ):
#         """
#         Download and add to dataset recent papers from Arxiv by results from search quarry
#         Parameters
#         ----------
#         search_query: str
#             Search query (e.g. 'Deep Learning')
#
#         limit: float
#             Maximum number of papers to download.
#
#         output_dir: str
#             Directory where documents will be saved
#
#         Returns
#         -------
#         self
#             self obejct for chaining
#
#         """
#
#         downloaded_papers = set(
#             download_recent_papers_by_querry(search_query, limit, output_dir)
#         )
#         self._add_papers_by_filepath(downloaded_papers)
#
#         return self
#
#     def add_paper(self, filepath: str, reload_if_exist: bool = False):
#         """Add single paper with summary to dataset
#
#         Parameters
#         ----------
#         filepath: str
#             Path to file
#
#         reload_if_exist: bool
#             If True - overwrites paper data in dataset if exist
#
#         Returns
#         -------
#         self
#             self obejct for chaining
#
#         """
#         if os.path.basename(filepath) in self._papers and not reload_if_exist:
#             # Don't add already existing file if not required
#             return
#
#         try:
#             paper = PaperData(filepath)
#             summary = get_description_json(paper)
#             summary["filepath"] = filepath
#             # summary["New Features"] = summary["New Features"].split(',')
#
#             self._papers[os.path.basename(filepath)] = summary
#
#         except Exception as err:
#             logging.error(f"Failed to update: {filepath} - {err}")
#
#         return self
#
#     def _add_papers_by_filepath(self, files: set):
#         """Update dataset dictionary with new files and their summaries.
#
#         Parameters
#         ----------
#         files: set
#             files to add
#
#         """
#         files_to_add = set(os.path.basename(f) for f in files).difference(
#             set(self._papers.keys())
#         )
#         files_to_add = set(f for f in files if os.path.basename(f) in files_to_add)
#
#         for f_path in tqdm(files_to_add, desc="Updating list of papers"):
#             self.add_paper(f_path)
#
#     def refresh_summary(self):
#         """Reload file summaries in dataset dictionary, overwriting existing data"""
#         for filename, data in self._papers.items():
#             f_path = data.get("filepath", "")
#
#             try:
#                 paper = PaperData(f_path)
#                 summary = get_description_json(paper)
#
#                 for key, val in summary.items():
#                     data[key] = val
#
#             except Exception as err:
#                 logging.error(f"Failed to update: {filename} - {err}")
#
#     @property
#     def list_of_papers(self):
#         return self._papers.keys()
#
#     @property
#     def list_data_fields(self):
#         data_fields = set()
#
#         for data in self._papers.values():
#             data_fields.update(data.keys())
#
#         return list(data_fields)
#
#     def list_values_by_field(self, search_key: str):
#         values = set()
#
#         for data in self._papers.values():
#             value = data.get(search_key, "")
#
#             if value:
#                 values.add(value)
#
#         return values
#
#     @property
#     def list_new_features(self):
#         return self.list_values_by_field("New Features")
#
#     def search_by_field_value(self, field: str, value: str, regex_search: bool = True):
#         """Search papers with specific value in given field
#
#         Parameters
#         ----------
#         field: str
#             Search Field in papers (e.g. "New Features")
#         value: str
#             Searched Value
#         regex_search: str
#             If true - value is treaten as regex and may be not exact match
#             Else - value must be exact match
#
#         Returns
#         -------
#         found: dict
#             Dictionary containing all papers matching searched value
#
#         """
#         if regex_search:
#             found = {
#                 paper: data
#                 for paper, data in self._papers.items()
#                 if re.search(value, data.get(field, ""))
#             }
#         else:
#             found = {
#                 paper: data
#                 for paper, data in self._papers.items()
#                 if data.get(field, "") == value
#             }
#
#         if not found:
#             logging.warning(
#                 f"Values for field: '{field}' not found "
#                 f"- probably field not exist in dataset"
#             )
#
#         return found
#
#     def get_paper_by_filename(self, filename):
#         """Search Paper Data by filename"""
#         return self._papers.get(filename, None)
#
#     def filter_by_categories(self, category: Union[Tuple[str], str]):
#         """Filter Papers by Category (e.g. Object Detection)
#
#         Parameters
#         ----------
#         category: Union[Tuple[str], str]
#             One String or tuple with strings with categories to search
#
#         Returns
#         -------
#         found: dict
#             Dictionary containing all papers matching searched categories
#
#         """
#         if isinstance(category, str):
#             categories = tuple(category.lower())
#         else:
#             categories = tuple(c.lower() for c in category)
#
#         found = {
#             paper: data
#             for paper, data in self._papers.items()
#             if data.get("Category", "").lower() in categories
#         }
#
#         return found
