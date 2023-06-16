from PyPDF2 import PdfReader

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PageData:
    """Container for page(text) data

    Parameters
    ----------
    text : str
        Text of the page

    Attributes
    ----------
    words_count : int
        Number of words in 'text' parameter.
        .. note:: Used as aprroximate number of tokens for chatGPT

    """

    text: str

    def __add__(self, other_page):
        # Create concatenated page with other page or text
        if isinstance(other_page, str):
            return self.__class__(self.text + " \n" + other_page)

        return self.__class__(self.text + " \n" + other_page.text)

    def __repr__(self) -> str:
        return f"PageData(number of words: {self.words_count})"

    @property
    def words_count(self) -> int:
        return len(self.text.split())


@dataclass
class PaperData:
    """Container for research paper's data (pages with text and filepath to paper)

    Parameters
    ----------
    filepath : str
        Path to PDF file with research paper

    """

    filepath: str
    _pages: List[PageData] = field(init=False)

    def __post_init__(self):
        reader = PdfReader(self.filepath)
        self._pages = [PageData(page.extract_text()) for page in reader.pages]

    def __getitem__(self, i):
        # Access each page by index
        return self._pages[i]

    def __repr__(self) -> str:
        return (
            f"PaperData(file: '{self.filepath}',"
            f" number of pages: {len(self._pages)})"
        )

    def join_pages_by_length(self, max_words: int = 1100) -> List[PageData]:
        """Concatenate pages by number of maximum words to contain.

        .. note:: Used as token limiter for prompts filling

        Parameters
        ----------
        max_words: int
            Maximum number of words allowed per new page, obtained from join

        Returns
        -------
        joined_pages: List[PageData]
            List of pages, after concatenation


        """
        joined_pages = []
        last_join_page = None

        for i, page in enumerate(self._pages):
            if last_join_page is not None:
                if last_join_page.words_count + page.words_count <= max_words:
                    # Create joined version of pages
                    last_join_page = last_join_page + page

                else:
                    # Last candidate is too long to add more pages - add it to list
                    # and go further
                    joined_pages.append(last_join_page)
                    last_join_page = page

            else:
                # last_join_page not exist yet - use current page
                last_join_page = page

            if i == (len(self._pages) - 1):
                # Fill list with last element
                joined_pages.append(last_join_page)

        return joined_pages
