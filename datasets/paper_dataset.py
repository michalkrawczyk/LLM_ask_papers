from enum import Enum
import logging
import os
import re
from typing import (
    Any, Dict, List,
    Union,
    Tuple, Optional, Iterable)

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document, BasePromptTemplate
from langchain.vectorstores import Chroma, VectorStore
from tqdm import tqdm

from arxiv_utils import (
    ExtendedArxivRetriever,
)
from templates import ShortInfoSummary, DEFAULT_PROMPT_REGISTER, PromptHolder

logger = logging.getLogger(__name__)

""" TODO:

test search functions
summarize_paper - test + prompt
add_page (305 line)
add_youtube_summary (310 line)
list_available_fields - test

- For single paper dataset should be used only single embedding and llm
- Nougat reader (example on Hugging Face)
- 56 line
- 
 - add mising docs
- TensorRT acceleration
- method to load dataset with variables
- normally identify summary for page and one refined for whole paper?
"""


def _get_document_name(document: Union[Document, Dict]) -> str:
    """Get name of document (Title or source)"""
    if isinstance(document, Document):
        meta = document.metadata
    else:
        meta = document

    if meta:
        title = meta.get("title")
        name = None

        if not title or title == "Unknown Text":
            name = meta.get("source") \
                   or meta.get("file_path")
            name = os.path.basename(name)

        if name or title:
            return f"{name or title} - page: {meta.get('page', 0)}"

    return "Unknown"


class SearchType(Enum):
    """ Search Type Enum for VectorStore.search()"""
    # TODO: Delete if restrict only Chroma in _db
    MMR = "mmr"
    SIMILARITY = "similarity"


class PaperDatasetLC:
    _db: Optional[VectorStore] = None
    _papers: Dict = dict()  # For listing included documents, retrieve whole documents - TODO: remove
    _default_llm: Optional[BaseLanguageModel] = None
    _prompts: PromptHolder = DEFAULT_PROMPT_REGISTER

    _feature_list: Optional[List[str]] = None   # meta private
    max_num_words = 1500    # TODO: method to take from llm and setters

    def __init__(self, db: Optional[VectorStore] = None,
                 llm: Optional[BaseLanguageModel] = None):
        if not db:
            import openai
            from langchain.llms.openai import OpenAI

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

        for prompt_name in ["identify_feature", "summarize_paper", "summarize_paper_refine"]:
            if prompt_name not in self._prompts and prompt_name in DEFAULT_PROMPT_REGISTER:
                self._prompts.load_defined_prompt(name=prompt_name, prompt=DEFAULT_PROMPT_REGISTER[prompt_name])
                logger.info(f"Prompt {prompt_name} loaded from default register")

            elif prompt_name not in self._prompts:
                logger.warning(f"Prompt {prompt_name} not defined -"
                               f" Some functions requiring it will not be available")

    def _split_document_by_length(self, document: Document) -> List[Document]:
        """Split document by max_num_words"""
        regex = r"\n+[A-Z0-9.]{1,5}[.:][\t ]?[A-Z][\w -]{4,}\n+"

        word_count = len(document.page_content.split(' '))

        if word_count > self.max_num_words:
            documents = []
            # Search for sections
            regex_matches = re.findall(regex, document.page_content)

            if (len(regex_matches) + 1) >= word_count / self.max_num_words:
                # Split by found sections
                for match, text in zip(regex_matches, re.split(regex, document.page_content)):
                    documents.append(Document(page_content=match + ' ' + text, metadata=document.metadata))
            else:
                words = document.page_content.split(' ')
                last_index = 0
                for i in range(1, len(words), self.max_num_words):
                    documents.append(
                        Document(page_content=' '.join(words[last_index:i * self.max_num_words]),
                                 metadata=document.metadata))
                    last_index = i
        else:
            documents = [document]

        return documents

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

        documents = self._split_document_by_length(document)

        try:
            doc_uuids = self._db.add_documents(documents)
            self._papers.update({uid: doc for uid, doc in zip(doc_uuids, documents)})

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
            data = []

            for doc in tqdm(PyMuPDFLoader(filepath).load(), desc="Loading pdf pages"):
                split_docs = self._split_document_by_length(doc)
                data.extend(split_docs)

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
            if len(texts) != len(metadatas):  # type: ignore
                raise ValueError("Number of metadata object must match number of texts")
        except ValueError as err:
            logger.info(f"Failed to add texts to dataset: {err}")
            return []  # No Object added - return empty list
        # TODO: split texts by max_num_words

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

                valid_records.extend(
                    self._split_document_by_length(
                        Document(page_content=text, metadata=meta)))

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
            docs = retriever.get_documents_by_id(id_list)  # type: ignore

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
            splited_documents = []
            for doc in tqdm(documents, "Splitting documents"):
                splited_documents.extend(self._split_document_by_length(doc))

            doc_uuids = self._db.add_documents(splited_documents)
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

    def unique_list_of_documents(self, regex_filter: Optional[str] = None) -> List[Tuple[str, ...]]:
        """ List of papers stored in vector database to tuple of (title, source, file_path).
        List is shortened to not contain next pages of same document.


        Parameters
        ----------
        regex_filter: Optional[str]
            Optional Regex pattern to filter documents by title, source or file_path

        Returns
        -------
        List[Tuple[str, ...]]
            List of tuples (title, source, file_path) of unique documents

        """
        document_set = set()

        for meta in tqdm(self.get(include=["metadatas"])["metadatas"], desc="Listing documents"):
            fields = [meta.get(field, "") for field in ["title", "source", "file_path"]]

            if not regex_filter or any(re.match(regex_filter, field) for field in fields):
                document_set.add(tuple(fields))

        return list(document_set)

    def list_documents_by_id(self) -> List[Tuple[str, str]]:
        """Shortened List of papers stored in vector database by id"""
        # TODO: Filters?
        documents = self.get(include=["metadatas"])

        return [(doc_id, doc) for doc_id, doc in zip(documents["ids"], map(_get_document_name, documents["metadatas"]))]

    def list_new_features(self, store_result: bool = True) -> List[str]:
        """  List of all new features detected in documents

        Parameters
        ----------
        store_result: bool
            If True - Store result in class variable for faster access

        Returns
        -------
        List[str]
            List of all new features detected in documents

        """
        if self._feature_list:
            return self._feature_list

        feature_set = set()

        for metadata in tqdm(self.get(where={"new_features": {"$ne": ""}}, include=["metadatas"])["metadatas"],
                             desc="Listing features"):
            features = metadata["new_features"].lower().split(", ")
            feature_set.update(features)

        feature_list = list(feature_set)

        if store_result:
            self._feature_list = feature_list  # type: ignore

        return feature_list

    def list_available_fields(self) -> List[str]:
        """ List of all available metadata fields in documents """
        features = set()
        for metadata in tqdm(self.get(include=["metadatas"])["metadatas"], desc="Listing available fields"):
            features.update(metadata.keys())

        return list(features)

    def get(self, **kwargs) -> Dict[str, Any]:
        """

        Parameters
        ----------
        kwargs: Any
         Possible arguments:
            ids: The ids of the embeddings to get. Optional.
            where: A Where type dict used to filter results by.
                   E.g. `{"color" : "red", "price": 4.20}`. Optional.
            limit: The number of documents to return. Optional.
            offset: The offset to start returning results from.
                    Useful for paging results with limit. Optional.
            where_document: A WhereDocument type dict used to filter by the documents.
                            E.g. `{$contains: {"text": "hello"}}`. Optional.
            include: A list of what to include in the results.
                     Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
                     Ids are always included.
                     Defaults to `["metadatas", "documents"]`. Optional.

        Returns
        -------
        Dict[str, Any]
            Dictionary of documents with included keys

        """
        if isinstance(self._db, Chroma):
            return self._db.get(**kwargs)

    def get_by_id(self, ids: Union[str, Iterable[str]], include: Optional[List[str]] = None) -> Dict[str, Any]:
        """ Get documents by id

        Parameters
        ----------
        ids: Union[str, Iterable[str]]
            Ids of documents to get
        include : Optional[List[str]]
            List of fields to include in result.
            Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
            Ids are always included.
            Defaults to `["metadatas", "documents"]`

        Returns
        -------
        Dict[str, Any]
            Dictionary of documents with included keys

        """
        return self.get(ids=ids, include=include)

    def get_containing_field(self, field_name: str, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """ Get documents with non-empty given field

        Parameters
        ----------
        field_name: str
            Field name to search
        include: Optional[List[str]]
            List of fields to include in result.
            Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
            Ids are always included.
            Defaults to `["metadatas", "documents"]`

        Returns
        -------
        Dict[str, Any]
            Dictionary of documents with included keys

        """
        return self.get(where={field_name: {"$ne": ""}},
                        include=include)

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

    def similarity_search_with_scores(
            self, query: str, n_results: int = 3,
            score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """

        Parameters
        ----------
        query: str
            Search Query
        n_results: int
            Limit results to 'n' documents
        score_threshold: float
            Optional score threshold for filtering results

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
        query: str
            Search Query
        llm: Optional[BaseLanguageModel]
            LLM model to use. If not provided default model is used
        kwargs: Any
            Keyword arguments for 'RetrievalQAWithSourcesChain.from_chain_type' method

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


    # def filter_by_categories(self, category: Union[Tuple[str], str]):
    # TODO:
    #     pass

    # def get_as_documents(self, **kwargs):

    def update_document_features(self, document_ids: Union[str, Iterable[str]] = None,
                                 llm: Optional[BaseLanguageModel] = None, force_reload: bool = False):
        """ Update document with features detected by LLM model

        Parameters
        ----------
        document_ids: Union[str, Iterable[str]]
            Document ids to update
        llm: Optional[BaseLanguageModel]
            LLM model to use. If not provided, default model is used
        force_reload: bool
            If True - Documents with already identified features will be also updated

        """
        if isinstance(document_ids, str):
            document_ids = [document_ids]

        docs = self.get(ids=document_ids, include=["metadatas", "documents"])

        if not docs:
            logger.warning(f"Documents {document_ids} not found")
            return

        # TODO: move output parser to prompt (with assert that prompt has parser)
        parser = PydanticOutputParser(pydantic_object=ShortInfoSummary)

        llm: BaseLanguageModel = llm or self._default_llm
        base_prompt: BasePromptTemplate = self._prompts.get("identify_features")

        if self._prompts is None:
            logger.warning("Prompt 'identify_features' is not defined")
            return

        llm_chain = LLMChain(llm=llm, prompt=base_prompt)

        for doc_id, doc_text, metadata in tqdm(zip(docs["ids"], docs["documents"], docs["metadatas"]), "Updating features"):

            if metadata.get("new_features") and not force_reload:
                # Skip already identified documents
                continue

            if not doc_text:
                # Skip empty documents
                logger.warning(f"Document {doc_id} is empty")
                continue

            response = llm_chain.predict(text=doc_text, format_instructions=parser.get_format_instructions())

            data = parser.parse(response).dict()
            # Note: Lists are not accepted in metadata
            # TODO: correct metadata by function?
            data.update({k: ", ".join(v) for k, v in data.items() if isinstance(v, list)})
            metadata.update(data)

            # Update document in database
            self._db.update_document(doc_id, Document(page_content=doc_text, metadata=metadata))

    # @staticmethod
    # def get_document_info(document: Document):
    #     """Get document metadata (info)"""
    #     #TODO: Needed?
    #     return document.metadata

    def search_by_field(self, field_name: str, search_value: str, regex_match: bool = False,
                        include: Optional[List[str]] = None) -> Dict[str, List[Any]]:
        """ Search documents by value in given field

        Parameters
        ----------
        field_name: str
            Field name to search
        search_value: str
            Value to search
        regex_match:
            If True - Search as regex: 'name' is treated as regex pattern.
        include: Optional[List[str]]
            List of fields to include in result.
            Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
            Ids are always included.
            Defaults to `["metadatas", "documents"]`

        Returns
        -------
        Dict[str, Any]
            Dictionary of documents with included keys

        """

        if not regex_match:
            return self.get(where={field_name: {"$eq": search_value}}, include=include)

        # Double get from db due to Chroma not supporting regex search
        # and lower memory usage with filtering only by metadata first
        documents = self.get_containing_field(field_name=field_name, include=["metadatas"])
        found_ids = []

        for i, doc_meta in enumerate(tqdm(documents["metadatas"], desc="Searching documents")):

            if search_value in doc_meta[field_name]:
                found_ids.append(documents["ids"][i])

        return self.get(ids=found_ids, include=include)

    def search_by_name(self, search_value: str, regex_match: bool = False,
                       include: Optional[List[str]] = None) -> Dict[str, List[Any]]:
        """ Search documents by name (title or source)

        Parameters
        ----------
        search_value: str
            Name to search (title or source)
        regex_match:
            If True - Search as regex: 'name' is treated as regex pattern.
        include: Optional[List[str]]
            List of fields to include in result.
            Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
            Ids are always included.
            Defaults to `["metadatas", "documents"]`

        Returns
        -------
        Dict[str, Any]
            Dictionary of documents with included keys

        """
        if not regex_match:
            query = {
                "$or": [
                    {
                        "title": {
                            "$eq": search_value
                        }
                    },
                    {
                        "source": {
                            "$eq": search_value
                        }
                    }
                ]
            }
            return self.get(where=query, include=include)

        # Double get from db due to Chroma not supporting regex search
        # and lower memory usage with filtering only by metadata first

        documents = self.get(include=['metadatas'])
        found_ids = []

        for i, doc_meta in enumerate(tqdm(documents["metadatas"], desc="Searching documents")):
            if re.match(search_value, doc_meta["title"]) or re.match(search_value, doc_meta["source"]):
                found_ids.append(documents["ids"][i])

        return self.get(ids=found_ids, include=include)

    def summarize_paper(self, paper_name):
        docs = self.get(where={"title": paper_name})["documents"]

        if not docs:
            logger.warning(f"Paper {paper_name} not found")
            return

        for prompt_name in ["summarize_paper", "summarize_paper_refine"]:
            if prompt_name not in self._prompts:
                logger.warning(f"Prompt {prompt_name} not defined -"
                               f" Summary function will not be available")
                return

        llm_chain = load_summarize_chain(llm=self._default_llm,
                                         chain_type="refine",
                                         question_prompt=self._prompts["summarize_paper"],
                                         refine_prompt=self._prompts["summarize_paper_refine"],
                                         document_variable_name="text",
                                         input_key="documents",
                                         output_key="summary")

        result = llm_chain({"documents": docs})

        return result["summary"]

    # def identify_features(self, documents: Union[str, Iterable[str]],
    #                       llm: Optional[BaseLanguageModel] = None):
    #
    #     # Note: Each document should identify separately, even from same paper
    #
    #     if isinstance(documents, str):
    #         documents = [documents]
    #
    #     parser = PydanticOutputParser(pydantic_object=ShortInfoSummary)
    #
    #     llm: BaseLanguageModel = llm or self._default_llm
    #     base_prompt: BasePromptTemplate = DEFAULT_PROMPT_REGISTER["identify_features"]
    #     llm_chain = LLMChain(llm=llm, prompt=base_prompt)
    #
    #     for doc in tqdm(documents, desc="Identifying features"):
    #         # if not force_reload and doc.metadata.get("new_features"):
    #         #     # Skip already identified documents
    #         #     continue
    #
    #         # response = llm_chain.predict_and_parse(text=doc)
    #         response = llm_chain.predict(text=doc, format_instructions=parser.get_format_instructions())
    #
    #         data = parser.parse(response)
    #         yield data
    #
    #         # TODO: way to update it in Chroma Database
    #
    #     # TODO: limit_pages - analyze only 'n' first pages and add keys to whole
    #     # TODO: skip already added in metadata
    #     # TODO: Think about extracting 'Abstract' and "Results" to analyze
    #     # TODO: Maybe scan n_pages after "Abstract" and after "Results"?
    #
    #     # TODO: Idea: Keywords can be reduced without llm - just string/set filter