from enum import Enum
import re
import os
from typing import List, Union, Dict

from langchain.schema import Document


class SplitType(Enum):
    # Enum to allow extending split operations
    PARAGRAPH = 0
    SECTION = 1


def get_document_name(document: Union[Document, Dict]) -> str:
    """Get name of document (Title or source)"""
    if isinstance(document, Document):
        meta = document.metadata
    else:
        meta = document

    if meta:
        title = meta.get("title")
        name = None

        if not title or title == "Unknown Text":
            name = meta.get("source") or meta.get("file_path")
            name = os.path.basename(name)

        if name or title:
            return f"{name or title} - page: {meta.get('page', 0)}"

    return "Unknown"


def check_same_doc(documents: List[Document]):
    """ Check if all documents come from the same source (title or source)"""
    doc_names = [doc.metadata.get("source", doc.metadata.get("title")) for doc in documents]
    return len(set(doc_names)) == 1


def locate_metadata(documents: List[Document], pages_end_idx: List[int], search_idx: int):
    """ Locate metadata for given text index

    Parameters
    ----------
    documents: List[Document]
        List of documents
    pages_end_idx: List[int]
        List of text indexes where each document ends
    search_idx: int
        Text index to locate metadata for

    Returns
    -------

    """
    last_idx = 0
    while search_idx < pages_end_idx[last_idx]:
        last_idx += 1

    return documents[last_idx].metadata


def split_by_sections(text: str):
    """ Split text into sections

    Parameters
    ----------
    text: str
        Text to split

    Returns
    -------
    List[int]
        List of indexes where text should be split

    """
    regex = r"\n+[A-Z0-9.]{1,5}[.:][\t ]?[A-Z][\w -]{4,}\n+"
    match_patterns = re.compile(regex)
    separate_txt_idx = [0] + [match.end() for match in match_patterns.finditer(text)]

    return separate_txt_idx


def split_by_paragraphs(text: str):
    """ Split text into paragraphs

        Parameters
        ----------
        text: str
            Text to split

        Returns
        -------
        List[int]
            List of indexes where text should be split

        """
    regex = r"\w+[.]\n+"    # split by dot and new line
    match_patterns = re.compile(regex)
    separate_txt_idx = [0] + [match.end() for match in match_patterns.finditer(text)]

    return separate_txt_idx


def split_docs(documents: List[Document], max_words: int = 300, split_type: SplitType = SplitType.PARAGRAPH):
    """ Split documents into paragraphs or chunks of max_words size if paragraph is too long

    Parameters
    ----------
    documents: List[Document]
        List of documents to split

    max_words: int
        Max number of words in chunk
    split_type: SplitType(Enum)
        Type of split operation to perform

    Returns
    -------

    """

    split_op = {
        SplitType.PARAGRAPH: split_by_paragraphs,
        SplitType.SECTION: split_by_sections
    }

    if not check_same_doc(documents):
        raise ValueError("All documents must come from the same source (title or source)")

    split_documents = []
    merged_text = " ".join([doc.page_content for doc in documents])
    pages_length = [len(doc.page_content) for doc in documents]
    pages_end_idx = [sum(pages_length[:i]) for i in range(len(pages_length))]

    separate_txt_idx = split_op.get(split_type, split_by_sections)(merged_text)
    # TODO: max_words doesn't work properly

    for i, txt_idx in enumerate(separate_txt_idx[1:]):

        if txt_idx - separate_txt_idx[i - 1] <= max_words:
            # if concatenated text is smaller than max_words, add it to list
            # merged_text.append(text[separate_txt_idx[i - 1]: txt_idx])
            doc_metadata = locate_metadata(documents, pages_end_idx, txt_idx)
            split_documents.append(Document(page_content=merged_text[separate_txt_idx[i - 1]: txt_idx],
                                            metadata=doc_metadata))

        else:
            # split chunk into smaller batches with size of max_words
            # Note: This may cut sentences,
            chunk_to_split = merged_text[separate_txt_idx[i - 1]: txt_idx].split(" ")
            text_batches = [chunk_to_split[j: j + max_words] for j in range(0, len(chunk_to_split), max_words)]
            # TODO: Check if need locating for each batch
            doc_metadata = locate_metadata(documents, pages_end_idx, txt_idx)

            split_documents.extend(([Document(page_content=" ".join(batch), metadata=doc_metadata)
                                     for batch in text_batches]))

    return split_documents


