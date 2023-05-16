from arxiv_utils import download_paper_from_arxiv, download_recent_papers_by_querry, \
    PaperData
from gpt_core import get_description_json

from tqdm import tqdm

import logging
import os
import re
from typing import List, Union, Tuple


class PaperDataset:
    _papers = dict()

    def add_papers_by_id(self, id_list: List[str], output_dir: str = "."):
        """ Download and add to dataset papers from Arxiv by ID

        Parameters
        ----------
        id_list: List[str]
            List with ids for each paper to download

        output_dir: str
            Directory where documents will be saved

        Returns
        -------
        self
            self obejct for chaining

        """
        downloaded_papers = set(download_paper_from_arxiv(id_list, output_dir))
        self._add_papers_by_filepath(downloaded_papers)

        return self

    def search_and_add_papers(self, search_query: str, limit: float = 10.0,
                              output_dir: str = "."):
        """
        Download and add to dataset recent papers from Arxiv by results from search quarry
        Parameters
        ----------
        search_query: str
            Search query (e.g. 'Deep Learning')

        limit: float
            Maximum number of papers to download.

        output_dir: str
            Directory where documents will be saved

        Returns
        -------
        self
            self obejct for chaining

        """

        downloaded_papers = set(download_recent_papers_by_querry(
            search_query, limit, output_dir))
        self._add_papers_by_filepath(downloaded_papers)

        return self

    def add_paper(self, filepath: str, reload_if_exist: bool = False):
        """ Add single paper with summary to dataset

        Parameters
        ----------
        filepath: str
            Path to file

        reload_if_exist: bool
            If True - overwrites paper data in dataset if exist

        Returns
        -------
        self
            self obejct for chaining

        """
        if os.path.basename(filepath) in self._papers and not reload_if_exist:
            # Don't add already existing file if not required
            return

        try:
            paper = PaperData(filepath)
            summary = get_description_json(paper)
            summary["filepath"] = filepath
            # summary["New Features"] = summary["New Features"].split(',')

            self._papers[os.path.basename(filepath)] = summary

        except Exception as err:
            logging.error(f"Failed to update: {filepath} - {err}")

        return self

    def _add_papers_by_filepath(self, files: set):
        """ Update dataset dictionary with new files and their summaries.

        Parameters
        ----------
        files: set
            files to add

        """
        files_to_add = set(os.path.basename(f) for f in files).difference(set(self._papers.keys()))
        files_to_add = set(f for f in files if os.path.basename(f) in files_to_add)

        for f_path in tqdm(files_to_add, desc="Updating list of papers"):
            self.add_paper(f_path)

    def refresh_summary(self):
        """ Reload file summaries in dataset dictionary, overwriting existing data"""
        for filename, data in self._papers.items():
            f_path = data.get("filepath", "")

            try:
                paper = PaperData(f_path)
                summary = get_description_json(paper)

                for key, val in summary.items():
                    data[key] = val

            except Exception as err:
                logging.error(f"Failed to update: {filename} - {err}")

    @property
    def list_of_papers(self):
        return self._papers.keys()

    @property
    def list_data_fields(self):
        data_fields = set()

        for data in self._papers.values():
            data_fields.update(data.keys())

        return list(data_fields)

    def list_values_by_field(self, search_key: str):
        values = set()

        for data in self._papers.values():
            value = data.get(search_key, "")

            if value:
                values.add(value)

        return values

    @property
    def list_new_features(self):
        return self.list_values_by_field("New Features")

    def search_by_field_value(self, field: str, value: str, regex_search: bool = True):
        """ Search papers with specific value in given field

        Parameters
        ----------
        field: str
            Search Field in papers (e.g. "New Features")
        value: str
            Searched Value
        regex_search: str
            If true - value is treaten as regex and may be not exact match
            Else - value must be exact match

        Returns
        -------
        found: dict
            Dictionary containing all papers matching searched value

        """
        if regex_search:
            found = {paper: data for paper, data in self._papers.items()
                     if re.search(value, data.get(field, ""))}
        else:
            found = {paper: data for paper, data in self._papers.items()
                     if data.get(field, "") == value}

        if not found:
            logging.warning(f"Values for field: '{field}' not found "
                            f"- probably field not exist in dataset")

        return found

    def get_paper_by_filename(self, filename):
        """ Search Paper Data by filename"""
        return self._papers.get(filename, None)

    def filter_by_categories(self, category: Union[Tuple[str], str]):
        """ Filter Papers by Category (e.g. Object Detection)

        Parameters
        ----------
        category: Union[Tuple[str], str]
            One String or tuple with strings with categories to search

        Returns
        -------
        found: dict
            Dictionary containing all papers matching searched categories

        """
        if isinstance(category, str):
            categories = tuple(category.lower())
        else:
            categories = tuple(c.lower() for c in category)

        found = {paper: data for paper, data in self._papers.items()
                 if data.get("Category", "").lower() in categories}

        return found
