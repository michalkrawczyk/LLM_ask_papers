from arxiv_utils import download_paper_from_arxiv, download_recent_papers_by_querry, \
    PaperData
from gpt_core import get_description_json

from tqdm import tqdm

import logging
import os
import re
from typing import List


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

        """
        downloaded_papers = set(download_paper_from_arxiv(id_list, output_dir))
        self._add_papers_by_filepath(downloaded_papers)

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

        """

        downloaded_papers = set(download_recent_papers_by_querry(
            search_query, limit, output_dir))
        self._add_papers_by_filepath(downloaded_papers)

    def add_paper(self, filepath: str, reload_if_exist: bool = False):
        """ Add single paper with summary to dataset

        Parameters
        ----------
        filepath: str
            Path to file

        reload_if_exist: bool
            If True - overwrites paper data in dataset if exist

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

    def get_values_by_key(self, search_key: str):
        values = set()

        for data in self._papers.values():
            value = data.get(search_key, "")

            if value:
                values.add(value)

        return values

    @property
    def list_new_features(self):
        return self.get_values_by_key("New Features")

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
        found: list
            List containing all papers matching searched value

        """
        if regex_search:
            found = [data for data in self._papers.values()
                     if re.search(value, data.get(field, ""))]
        else:
            found = [data for data in self._papers.values()
                     if data.get(field, "") == value]

        if not found:
            logging.warning(f"Values for field: '{field}' not found "
                            f"- probably field not exist in dataset")

        return found

    def get_paper_by_filename(self, filename):
        """ Search Paper Data by filename"""
        return self._papers.get(filename, None)
