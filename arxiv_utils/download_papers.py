import arxiv
from tqdm import tqdm

import logging
import os
import re
import string
from typing import List


def _clean_filename_str(f_str: str):
    """Clean filename string from punctuations and whitespaces"""
    regex = re.compile("[%s]" % re.escape(string.punctuation + string.whitespace))
    return regex.sub("_", f_str).lstrip("_")


def download_paper_from_arxiv(
    id_list: List[str], output_dir: str = ".", update_existed: bool = False
) -> List[str]:
    """Download research papers from Arxiv by paper id

    Parameters
    ----------
    id_list: List[str]
        List with ids for each paper to download

    output_dir: str
        Directory where documents will be saved

    update_existed: bool
        If true - replace existing file with downloaded


    Returns
    -------
    papers_downloaded: List[str]
        List of paths for each downloaded file

    """
    papers_downloaded = []
    search = arxiv.Search(id_list=id_list)

    for paper in tqdm(search.results(), desc="Downloading files..."):
        paper_id = os.path.split(paper.entry_id)[-1]
        paper_title = _clean_filename_str(paper.title)
        paper_filepath = f"{output_dir}{os.sep}{paper_id}.{paper_title}.pdf"

        if not os.path.isfile(paper_filepath) or update_existed:
            file_path = paper.download_pdf(
                dirpath=output_dir, filename=f"{os.path.basename(paper_filepath)}"
            )
            papers_downloaded.append(file_path)
        else:
            logging.info(f"File Already Exist: {paper_filepath}")
            papers_downloaded.append(paper_filepath)

    return papers_downloaded


def download_recent_papers_by_querry(
    querry: str,
    limit: float = 10.0,
    output_dir: str = ".",
    update_existed: bool = False,
) -> List[str]:
    """Download research papers from Arxiv by result of search querry

    Parameters
    ----------
    querry: str
        Search querry (e.g. 'Deep Learning')

    limit: float
        Maximum number of papers to download.

    output_dir: str
        Directory where documents will be saved

    update_existed: bool
        If true - replace existing file with downloaded

    Returns
    -------
    papers_downloaded: List[str]
        List of paths for each downloaded file

    """
    papers_downloaded = []
    search = arxiv.Search(
        query=querry, max_results=limit, sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for paper in tqdm(search.results(), desc="Downloading files..."):
        paper_id = os.path.split(paper.entry_id)[-1]
        paper_title = _clean_filename_str(paper.title)
        paper_filepath = f"{output_dir}{os.sep}{paper_id}.{paper_title}.pdf"

        if not os.path.isfile(paper_filepath) or update_existed:
            file_path = paper.download_pdf(
                dirpath=output_dir, filename=f"{os.path.basename(paper_filepath)}"
            )
            papers_downloaded.append(file_path)
        else:
            logging.info(f"File Already Exist: {paper_filepath}")
            papers_downloaded.append(paper_filepath)

    return papers_downloaded
