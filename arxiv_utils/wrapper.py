# langchain/utilities/arxiv.py
# Indexes/Retrievers/Arxiv
import logging
from typing import Dict, List, Union, Any
import os

from arxiv import (
    Search,
    SortCriterion,
    SortOrder,
    ArxivError,
    HTTPError,
    UnexpectedEmptyPageError,
)
from langchain.schema import Document
import fitz
from pydantic import BaseModel, root_validator
from tqdm import tqdm

from .download_papers import _clean_filename_str

logger = logging.getLogger(__name__)


class ArxivAPIWrapper2(BaseModel):
    """Extended Wrapper around ArxivAPI (rewritten langchain.utilitiesArxivAPIWrapper2).

    This Extension allows to get more of Arxiv search engine (e.g. by filtering papers in query itself)
    and also allow to store downloaded files for later usage

    To use, you should have the ``arxiv`` python package installed.
    https://lukasschwab.me/arxiv.py/index.html

    This wrapper will use the Arxiv API to conduct searches and
    fetch document summaries. By default, it will return the document summaries
    of the top-k results.

    It limits the Document content by doc_content_chars_max.
    Set doc_content_chars_max=None if you don't want to limit the content size.

    Parameters:
        max_docs:
            A limit to the number of loaded documents

        sort_docs_by:
            Sort Criterion for fetched documents (used only on searching stage)
            This may be useful when limit of max_docs is set lower than number of documents found.

        sort_order:
            Sorting Order for fetched documents (Ascending, Descending)
            This may be useful when limit of max_docs is set lower than number of documents found.

        load_all_available_meta:
          if True: the `metadata` of the loaded Documents gets all available meta info
            (see https://lukasschwab.me/arxiv.py/index.html#Result),
          if False: the `metadata` gets only the most informative fields.

        doc_content_chars_max:
            The cut limit on the text from obtained paper.

        ARXIV_MAX_QUERY_LENGTH:
            The cut limit on the query used for the arxiv tool.

        save_pdf:
            If set to True: Downloaded paper with be kept at 'file_save_dir' directory
            Else: Delete pdf file

        file_save_dir:
        Directory for keeping downloaded files (even temporary)

        overwrite_existing:
            If set to True: Existing pdf files will be ignored and new ones will be downloaded
            Else: Will use existing files downloaded earlier

        separate_pages:
            If set to True: Each page of pdf file will be treated as separate document


    """

    _arxiv_exceptions: Any  # meta private

    max_docs: int = 10
    top_k_results: int = 3
    sort_docs_by: SortCriterion = SortCriterion.Relevance
    sort_order: SortOrder = SortOrder.Descending
    load_all_available_meta: bool = False

    doc_content_chars_max: Union[int, None] = None
    ARXIV_MAX_QUERY_LENGTH: Union[int, None] = 300

    save_pdf: bool = True
    file_save_dir: str = "."
    overwrite_existing: bool = False
    separate_pages: bool = True

    # verbose: bool = False

    @root_validator()
    def validate_variables(cls, values: Dict) -> Dict:
        values["_arxiv_exceptions"] = (ArxivError, UnexpectedEmptyPageError, HTTPError)

        if 0 > values["max_docs"] >= 300000:
            # 300 000 is limit of arXiv API
            raise ValueError(
                "Number of Maximum Documents to obtain "
                "should be in range [1; 300 000]"
            )

        if values["save_pdf"] and not os.path.isdir(values["file_save_dir"]):
            raise ValueError("Invalid Output directory for downloaded documents ")

        for v in ["ARXIV_MAX_QUERY_LENGTH", "doc_content_chars_max"]:
            if values[v] and values[v] < 1:
                logger.warning(f"String indexing with zero or less value: {v}")
                print(values[v])

        return values

    def _search_results(self, query: str = "", id_list: List[str] = None):
        if id_list is None:
            id_list = []

        if not id_list and not query:
            return "Empty query and id list for Arxiv Search"

        results = Search(
            query=query[: self.ARXIV_MAX_QUERY_LENGTH],
            id_list=id_list,
            max_results=self.max_docs,
            sort_by=self.sort_docs_by,
            sort_order=self.sort_order,
        ).results()

        return results

    def run(self, query: str = "", id_list: List[str] = None):
        try:
            results = self._search_results(query, id_list)

        except self._arxiv_exceptions as err:
            return f"Arxiv Error: {err}"

        docs = [
            f"Published: {result.updated.date()}\nTitle: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}"
            for result in results
        ]
        if docs:
            return "\n\n".join(docs)[: self.doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"

    def load(self, query: str = "", id_list: List[str] = None) -> List[Document]:
        try:
            results = self._search_results(query, id_list)

        except self._arxiv_exceptions as err:
            logger.warning(f"Error on arxiv: {err}")
            return []

        docs: List[Document] = []

        for result in tqdm(results, desc="Loading files..."):
            paper_id = os.path.split(result.entry_id)[-1]
            paper_title = _clean_filename_str(result.title)
            paper_filepath = f"{self.file_save_dir}{os.sep}{paper_id}.{paper_title}.pdf"

            try:
                already_exist = os.path.isfile(paper_filepath)

                if not already_exist or self.overwrite_existing:
                    file_path = result.download_pdf(
                        dirpath=self.file_save_dir,
                        filename=f"{os.path.basename(paper_filepath)}",
                    )
                else:
                    file_path = paper_filepath
                    logger.info(f"File Already Exist: {paper_filepath}")

                with fitz.open(file_path) as f:
                    texts = (
                        [page.get_text() for page in f]
                        if self.separate_pages
                        else ["".join(page.get_text() for page in f)]
                    )

            except FileNotFoundError as err:
                logger.warning(err)
                continue

            if self.load_all_available_meta:
                extra_metadata = {
                    "entry_id": result.entry_id,
                    "published_first_time": str(result.published.date()),
                    "comment": result.comment,
                    "journal_ref": result.journal_ref,
                    "doi": result.doi,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "links": [link.href for link in result.links],
                    "pdf_url": result.pdf_url,
                }
            else:
                extra_metadata = {}
            metadata = {
                "title": result.title,
                "source": f"[{result.entry_id}] {result.title}",
                "authors": ", ".join(a.name for a in result.authors),
                "summary": result.summary,
                "published": str(result.published.date()),
                "date": str(result.updated.date()),
                "file_path": paper_filepath if already_exist or self.save_pdf else "",
                **extra_metadata,
            }

            for idx, text in enumerate(texts):
                metadata["page"] = (
                    len(texts) - 1 and idx
                )  # Set page 0 if only one page, to mark whole document
                doc = Document(
                    page_content=text[: self.doc_content_chars_max],
                    metadata=metadata.copy(),  # metadata.copy() to avoid overwriting
                )
                docs.append(doc)

            if not self.save_pdf:
                os.remove(file_path)

        return docs
