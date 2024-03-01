from enum import Enum
import logging
import os
import re
from typing import Any, Dict, List, Union, Tuple, Optional, Iterable

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.language_models.chat_models import BaseChatModel

from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document, BaseOutputParser
from langchain_community.vectorstores import Chroma, VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain.prompts import PromptTemplate

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from arxiv_utils import (
    ExtendedArxivRetriever,
)
from templates import ShortInfoSummary, DEFAULT_PROMPT_REGISTER, PromptHolder
from utils import check_same_doc, get_document_name, split_docs, SplitType

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Search Type Enum for VectorStore.search()"""

    MMR = "mmr"
    SIMILARITY = "similarity"


class PaperDatasetLC:
    _db: Optional[Chroma] = None
    _default_llm: Optional[Union[BaseLanguageModel, BaseChatModel]] = None
    _prompts: PromptHolder = DEFAULT_PROMPT_REGISTER
    _split_type: SplitType = SplitType.SECTION

    _feature_list: Optional[List[str]] = None  # meta private
    max_num_words = 300  # TODO: method to take from llm and setters

    def __init__(
        self,
        db: Optional[Chroma] = None,
        llm: Optional[Union[BaseLanguageModel, BaseChatModel]] = None,
        doc_split_type: SplitType = SplitType.SECTION,
        max_num_words: int = 300,
    ):
        self._split_type = doc_split_type
        self.max_num_words = max_num_words
        self._db = db
        self._default_llm = llm

        if not self._db or not isinstance(self._db, VectorStore):
            logger.warning(
                "Dataset with not specified or invalid database"
                " - using Chroma with OpenAI embeddings and LLM"
            )
            try:
                import openai
                from langchain_community.chat_models import ChatOpenAI
                from langchain_community.embeddings import OpenAIEmbeddings

                self._db = Chroma(
                    embedding_function=OpenAIEmbeddings(
                        openai_api_key=openai.api_key,
                        model="text-embedding-3-small",
                        collection_metadata={"hnsw:space": "cosine"},
                    )
                )
                self._default_llm = ChatOpenAI(
                    temperature=0,
                    openai_api_key=openai.api_key,
                    model_name="gpt-3.5-turbo",
                )

            except ImportError as err:
                logger.error(f"Failed to import OpenAI and dependencies: {err}")

        for prompt_name in [
            "summarize_doc",
            "summarize_doc_refine",
        ]:
            if (
                prompt_name not in self._prompts
                and prompt_name in DEFAULT_PROMPT_REGISTER
            ):
                self._prompts.load_defined_prompt(
                    name=prompt_name, prompt=DEFAULT_PROMPT_REGISTER[prompt_name]
                )
                logger.info(f"Prompt {prompt_name} loaded from default register")

            elif prompt_name not in self._prompts:
                logger.warning(
                    f"Prompt {prompt_name} not defined -"
                    f" Some functions using it, may not work or require changing prompt in parameters "
                )

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

        documents = split_docs(
            [document], max_words=self.max_num_words, split_type=self._split_type
        )

        try:
            doc_uuids = self._db.add_documents(documents)

            return doc_uuids

        except Exception as err:
            logger.warning(
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
            data = split_docs(
                PyMuPDFLoader(filepath).load(),
                max_words=self.max_num_words,
                split_type=self._split_type,
            )

            # for doc in tqdm(PyMuPDFLoader(filepath).load(), desc="Loading pdf pages"):
            #     split_docs = self._split_document_by_length(doc)
            #     data.extend(split_docs)

            if metadata:
                for page in data:
                    for key in metadata.keys():
                        if key not in page.metadata or not page.metadata[key]:
                            # Add missing data
                            page.metadata[key] = metadata["key"]

            doc_uuids = self._db.add_documents(data)

            return doc_uuids

        except Exception as err:
            logger.warning(f"Failed to add Document {filepath} \n" f"  {err}")
            return []  # No Object added - return empty list

    def add_texts(
        self, texts: Iterable[str], metadatas: List[Dict], skip_invalid: bool = False
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
            logger.warning(f"Failed to add texts to dataset: {err}")
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
                    split_docs(
                        [Document(page_content=text, metadata=meta)],
                        max_words=self.max_num_words,
                        split_type=self._split_type,
                    )
                )

            except ValueError as err:
                logger.warning(err)

                if not skip_invalid:
                    logger.info(
                        "Adding texts to datasets skipped due to error and 'skip_invalid' flag value"
                    )
                    return []  # No Object added - return empty list

        try:
            doc_uuids = self._db.add_documents(valid_records) if valid_records else []

            return doc_uuids

        except Exception as err:
            logger.warning(f"Failed to add texts \n" f"  {err}")
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

            return doc_uuids

        except Exception as err:
            logger.warning(f"Failed to add papers: {err}")

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

            return doc_uuids

        except Exception as err:
            logger.warning(f"Failed to add documents: {err}")

        return []  # No Object added - return empty list

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add Multiple documents to vector database"""
        grouped_docs = []

        for i, doc in enumerate(documents):
            if i == 0:
                grouped_docs.append([doc])
                continue

            # if doc.metadata.get("source") == documents[i - 1].metadata.get("source"):
            if check_same_doc([doc, documents[i - 1]]):
                # Same source will be grouped for splitting
                grouped_docs[-1].append(doc)
            else:
                grouped_docs.append([doc])

        try:
            splited_documents = []

            for doc_group in grouped_docs:
                splited_documents.extend(
                    split_docs(
                        doc_group,
                        max_words=self.max_num_words,
                        split_type=self._split_type,
                    )
                )

            doc_uuids = self._db.add_documents(splited_documents)
            return doc_uuids

        except Exception as err:
            logger.warning(f"Failed to add documents: {err}")

        return []  # No Object added - return empty list

    def unique_list_of_documents(
        self, regex_filter: Optional[str] = None
    ) -> List[Tuple[str, ...]]:
        """List of papers stored in vector database to tuple of (title, source, file_path).
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

        for meta in tqdm(
            self.get(include=["metadatas"])["metadatas"], desc="Listing documents"
        ):
            fields = [meta.get(field, "") for field in ["title", "source", "file_path"]]

            if not regex_filter or any(
                re.search(regex_filter, field) for field in fields
            ):
                document_set.add(tuple(fields))

        return list(document_set)

    def list_documents_by_id(self) -> List[Tuple[str, str]]:
        """Shortened List of papers stored in vector database by id"""
        documents = self.get(include=["metadatas"])

        return [
            (doc_id, doc)
            for doc_id, doc in zip(
                documents["ids"], map(get_document_name, documents["metadatas"])
            )
        ]

    # def list_new_features(self, store_result: bool = True) -> List[str]:
    #     """List of all new features detected in documents
    #
    #     Parameters
    #     ----------
    #     store_result: bool
    #         If True - Store result in class variable for faster access
    #
    #     Returns
    #     -------
    #     List[str]
    #         List of all new features detected in documents
    #
    #     """
    #     if self._feature_list:
    #         return self._feature_list
    #
    #     feature_set = set()
    #
    #     for metadata in tqdm(
    #         self.get(where={"new_features": {"$ne": ""}}, include=["metadatas"])[
    #             "metadatas"
    #         ],
    #         desc="Listing features",
    #     ):
    #         features = metadata["new_features"].lower().split(", ")
    #         feature_set.update(features)
    #
    #     feature_list = list(feature_set)
    #
    #     if store_result:
    #         self._feature_list = feature_list  # type: ignore
    #
    #     return feature_list

    def list_available_fields(self) -> List[str]:
        """List of all available metadata fields in documents"""
        features = set()
        for metadata in tqdm(
            self.get(include=["metadatas"])["metadatas"],
            desc="Listing available fields",
        ):
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

    def get_by_id(
        self, ids: Union[str, Iterable[str]], include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get documents by id

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

    def get_containing_field(
        self, field_name: str, include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get documents with non-empty given field

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
        return self.get(where={field_name: {"$ne": ""}}, include=include)

    def similarity_search(
        self,
        query: str,
        n_results: int = 3,
        search_type: SearchType = SearchType.MMR,
        db_filter: Optional[Dict[str, str]] = None,
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
        db_filter: Optional[Dict[str, str]]
            Optional filters for database (If Chroma)

        Returns
        -------
        List[Document]
            List of relevant Documents

        """
        if db_filter and not isinstance(self._db, Chroma):
            logger.warning(
                "Filter option was not tested for other vector storages"
                " and may not work with other databases than Chroma"
            )

        return self._db.search(
            query=query, search_type=search_type.value, k=n_results, filter=db_filter
        )

    def similarity_search_with_scores(
        self,
        query: str,
        n_results: int = 3,
        score_threshold: Optional[float] = None,
        search_type: SearchType = SearchType.MMR,
        db_filter: Optional[Dict[str, str]] = None,
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
        search_type: SearchType
            Type of search:
            - SearchType.MMR for max relevance search
            - SearchType.SIMILARITY for similarity search
        db_filter: Optional[Dict[str, str]]
            Optional filters for database (If Chroma)

        Returns
        -------
        List[Tuple[Document, float]]
            List of tuples (Relevant document, score)

        """
        if db_filter and not isinstance(self._db, Chroma):
            logger.warning(
                "Filter option was not tested for other vector storages and "
                "may not work with other databases than Chroma"
            )

        search_results = []

        if search_type == search_type.SIMILARITY:
            search_results = self._db.similarity_search_with_relevance_scores(
                query=query,
                k=n_results,
                score_threshold=score_threshold,
                filter=db_filter,
            )

        elif search_type == search_type.MMR:
            # MMR search
            embedding = self._db.embeddings.embed_query(query)

            # Unfortunately _collection.query() method is not public and there is no public function to get results other way
            filter_results = self._db._collection.query(
                query_embeddings=embedding,
                n_results=n_results * 2,
                where=db_filter,
                # where_document=where_document,
                include=["metadatas", "documents", "distances", "embeddings"],
            )

            mmr_selected = maximal_marginal_relevance(
                np.array(embedding, dtype=np.float32),
                filter_results["embeddings"][0],
                k=n_results,
                lambda_mult=score_threshold or 0.5,
            )

            search_results = [
                (
                    Document(
                        page_content=filter_results["documents"][0][i],
                        metadata=filter_results["metadatas"][0][i],
                    ),
                    filter_results["distances"][0][i],
                )
                for i in mmr_selected
            ]

        else:
            logger.error(f"Search type {search_type} not supported")

        return search_results

    def llm_search(
        self,
        query: str,
        chain_type: str = "map_reduce",
        return_source_documents: bool = False,
        llm: Optional[BaseLanguageModel] = None,
        retriever_kwargs: Optional[Dict] = None,
        chain_kwargs: Optional[Dict] = None,
    ) -> Tuple[str, Optional[List[Document]]]:
        """LLM RAG search engine for searching content in documents

        Parameters
        ----------
        query: str
            Search Query
        llm: Optional[BaseLanguageModel]
            LLM model to use. If not provided OpenAI default model is used
        chain_type: str
            Type of chain to use:
            - "map_reduce" first applies an LLM chain to each document individually (the Map step),
                treating the chain output as a new document.
                It then passes all the new documents to a separate combine documents chain to get a single output
                 (the Reduce step).
                  It can optionally first compress, or collapse,
                  the mapped documents to make sure that they fit in the combine documents chain
                  (which will often pass them to an LLM)
            - "stuff" takes a list of documents, inserts them all into a prompt and passes that prompt to an LLM.
            - "refine" constructs a response by looping over the input documents and iteratively updating its answer.
                For each document, it passes all non-document inputs,
                the current document, and the latest intermediate answer to an LLM chain to get a new answer.
            - "map_rerank"
                The map re-rank documents chain runs an initial prompt on each document,
                that not only tries to complete a task but also gives a score for how certain it is in its answer.
                The highest scoring response is returned.

        return_source_documents: bool
            If True - Return also source documents

        retriever_kwargs: Optional[Dict]
            Keyword arguments for database retriever method (VectorStoreRetriever)

        chain_kwargs: Optional[Dict]
            Keyword arguments for 'RetrievalQA.from_chain_type' method

        Returns
        -------
        Tuple[str, Optional[List[Document]]]
            Tuple of (Answer, List of source documents (if 'return_source_documents')).

        """
        llm = llm or self._default_llm
        if not llm:
            raise RuntimeError("LLM not provided")

        if not retriever_kwargs:
            retriever_kwargs = {}

        if not chain_kwargs:
            chain_kwargs = {}
        chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self._db.as_retriever(**retriever_kwargs),
            chain_type=chain_type,
            return_source_documents=return_source_documents,
            **chain_kwargs,
        )
        result = chain({"query": query})
        return result["result"], result.get("source_documents", None)

    def update_document_features(
        self,
        document_ids: Union[str, Iterable[str]] = None,
        llm: Optional[BaseLanguageModel] = None,
        force_reload: bool = False,
        pydantic_object: BaseModel = ShortInfoSummary,
    ):
        parser = PydanticOutputParser(pydantic_object=pydantic_object)
        self.llm_doc_meta_updater(
            update_key="new_features",
            prompt="identify_features",
            predefined_prompt=True,
            document_ids=document_ids,
            llm=llm,
            force_reload=force_reload,
            output_parser=parser,
        )

    def llm_doc_meta_updater(
        self,
        update_key: str,
        prompt: Union[str, PromptTemplate],
        predefined_prompt: bool = True,
        document_ids: Union[str, Iterable[str], None] = None,
        llm: Optional[BaseLanguageModel] = None,
        output_parser: Optional[BaseOutputParser] = None,
        force_reload: bool = False,
        **var_kwargs,
    ):
        """Update document metadata with answers from LLM model based on predefined prompt questions

        Parameters
        ----------
        update_key: str
            Key in metadata to update
            If output_parser is structured - this will contain generated additional keys in format:
            {update_key}_{key}

        prompt: Union[str, PromptTemplate]
            Prompt to use for updating metadata.

            .. note:: if prompt is string, it will be picked from predefined or created,
             based on `predefined_prompt` value

        predefined_prompt: bool
            If True - use predefined prompt from register
            Otherwise - create prompt from given string

        document_ids:
            Document ids to update. If None - all documents will be updated

        llm: Optional[BaseLanguageModel]
            LLM model to use. If not provided, default model is used

        output_parser: Optional[BaseOutputParser]
            Output parser to use. If not provided, parser from prompt will be used

        force_reload: bool
            If True - Documents with already identified features will be also updated

        var_kwargs: Optional[Dict]
            Additional variables to pass when invoking prompt

        Returns
        -------

        """
        llm: BaseLanguageModel = llm or self._default_llm

        if not update_key or update_key in (
            "title",
            "source",
            "file_path",
            "date",
            "date_int",
            "page",
            "total_pages",
            "author",
            "creationDate",
            "split_part",
        ):
            logger.error(f"Update key {update_key} is not allowed")
            return

        if "." in update_key:
            logger.error(
                f"Update key {update_key} containing dot is restricted to substructures"
            )
            return

        if isinstance(prompt, str):
            prompt_template = (
                self._prompts.get(prompt)
                if predefined_prompt
                else PromptTemplate(template=prompt)
            )

        else:
            # Already defined PromptTemplate
            prompt_template = prompt

        if prompt_template is None:
            logger.error(f"Failed to find prompt - probably not defined")
            return

        if "text" not in prompt_template.input_variables:
            #
            logger.error(f"Prompt {prompt_template.name} has no 'text' variable")
            return

        if output_parser:
            if "format_instructions" not in prompt_template.template:
                instructed_prompt = prompt_template.template.copy()
                instructed_prompt.template = (
                    instructed_prompt.template
                    + "\n"
                    + output_parser.get_format_instructions()
                )

            else:
                instructed_prompt = prompt_template.partial(
                    format_instructions=output_parser.get_format_instructions()
                )

            chain = instructed_prompt | llm | output_parser

        else:
            chain = prompt_template | llm

        docs = self.get(ids=document_ids, include=["metadatas", "documents"])

        if not docs:
            logger.error(f"Documents {document_ids} not found")
            return

        for doc_id, doc_text, metadata in tqdm(
            zip(docs["ids"], docs["documents"], docs["metadatas"]), "Updating metadata"
        ):
            # TODO: think about rewrite for running in parallel if llm allows
            if metadata.get(update_key) and not force_reload:
                # Skip already identified documents
                continue

            if not doc_text:
                # Skip empty documents
                logger.warning(f"Document {doc_id} is empty")
                continue

            data = chain.invoke({"text": doc_text, **var_kwargs})

            # Metadata cannot contain other values than str, bool and int
            if isinstance(data, (str, float, int, bool)):
                metadata[update_key] = data
            elif data is None:
                logger.warning(f"Failed to update document {doc_id} - no data returned")
                metadata[update_key] = "None"

            else:
                added_keys = [f"{update_key}.cls._type"]
                data_dict = data.dict()

                for key, value in data_dict.items():
                    key_name = f"{update_key}.{key}"
                    metadata[key_name] = (
                        value
                        if isinstance(value, (str, float, int, bool))
                        else str(value)
                    )
                    added_keys.append(key_name)

                metadata[f"{update_key}.cls._type"] = data.__class__.__name__
                # Assumption no key will have comma in name (like in e.g. PydenticOutputParser)
                metadata[update_key] = f"metadata keys [{', '.join(added_keys)}]"

            # Update document in database
            self._db.update_document(
                doc_id, Document(page_content=doc_text, metadata=metadata)
            )

    def search_by_field(
        self,
        field_name: str,
        search_value: str,
        regex_match: bool = False,
        include: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        """Search documents by value in given field

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
        documents = self.get_containing_field(
            field_name=field_name, include=["metadatas"]
        )
        found_ids = []

        for i, doc_meta in enumerate(
            tqdm(documents["metadatas"], desc="Searching documents")
        ):
            if re.search(search_value, doc_meta[field_name]):
                found_ids.append(documents["ids"][i])

        return self.get(ids=found_ids if found_ids else [""], include=include)

    def search_by_name(
        self,
        search_value: str,
        regex_match: bool = False,
        include: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        """Search documents by name (title or source)

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
                    {"title": {"$eq": search_value}},
                    {"source": {"$eq": search_value}},
                ]
            }
            return self.get(where=query, include=include)

        # Double get from db due to Chroma not supporting regex search
        # and lower memory usage with filtering only by metadata first

        documents = self.get(include=["metadatas"])
        found_ids = []

        for i, doc_meta in enumerate(
            tqdm(documents["metadatas"], desc="Searching documents")
        ):
            if re.search(
                search_value,
                doc_meta.get("title", "") + " " + doc_meta.get("source", ""),
            ):
                found_ids.append(documents["ids"][i])

        return self.get(ids=found_ids if found_ids else [""], include=include)

    def summarize_docs(
        self,
        document_ids: Union[str, Iterable[str]] = None,
        llm: Optional[BaseLanguageModel] = None,
        summarize_prompt: str = "summarize_doc",
        refine_prompt: str = "summarize_doc_refine",
    ):
        """Summarize documents by LLM model

        .. note: This only uses predefined prompts as it is not intended to support custom prompts

        Parameters
        ----------
        document_ids: Union[str, Iterable[str]]
            Document ids to summarize (single id or list of ids)
        llm: Optional[BaseLanguageModel]
            LLM model to use. If not provided, default model is used
        summarize_prompt: str
            Name of prompt to use for summarization
        refine_prompt: str
            Name of prompt to use for summarization refinement

        Returns
        -------


        """
        docs = self.get_by_id(ids=document_ids, include=["documents"])
        llm: BaseLanguageModel = llm or self._default_llm

        if not docs:
            logger.error(f"Documents {document_ids} not found")
            return

        for prompt_name in [summarize_prompt, refine_prompt]:
            if prompt_name not in self._prompts:
                logger.error(
                    f"Prompt {prompt_name} not defined -"
                    f" Unable to summarize documents"
                )
                return

        llm_chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=self._prompts[summarize_prompt],
            refine_prompt=self._prompts[refine_prompt],
            document_variable_name="text",
            input_key="documents",
            output_key="summary",
        )

        result = llm_chain({"documents": docs}, return_only_outputs=True)
        return result["summary"]
