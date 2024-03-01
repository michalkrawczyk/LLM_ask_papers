import pytest
import yaml
from pathlib import Path

from langchain_community.vectorstores import Chroma

# TODO: embeddings and dataset same for all test units?

from datasets import PaperDatasetLC
from utils.doc_operations import SplitType

ROOT_PATH = Path(__file__).resolve().parents[2]

try:
    import openai
    from langchain_community.embeddings import OpenAIEmbeddings

    with open(ROOT_PATH / "openai_key.yaml", "r") as f:
        openai.api_key = yaml.safe_load(f)["openai_api_key"]
        OPENAI_AVAILABLE = True

except Exception as err:
    OPENAI_AVAILABLE = False
    print(err)

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import sentence_transformers

    SENTENCE_TRANSFORMERS_AVAILABLE = True

except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(
    not (SENTENCE_TRANSFORMERS_AVAILABLE or OPENAI_AVAILABLE),
    reason="Not found any llm module",
)
@pytest.mark.filterwarnings("ignore:.* was deprecated*")
def test_document_storage():
    embeddings = (
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
        if SENTENCE_TRANSFORMERS_AVAILABLE
        else (
            OpenAIEmbeddings(
                openai_api_key=openai.api_key, model="text-embedding-3-small"
            )
        )
    )

    dataset = PaperDatasetLC(
        db=Chroma(
            embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"}
        ),
        doc_split_type=SplitType.SECTION,
    )
    # add documents
    dataset.add_pdf_file(str(ROOT_PATH / "sample_documents/2302.00386.pdf"))
    sample_text = ["Lorem impsum something something", "Some Other Text"]
    sample_metas = [{"source": "sth", "v": "test"}, {"other": "sth"}]
    # ^ second one should be ignored due to not specified source
    assert (
        len(dataset.add_texts(sample_text, sample_metas, skip_invalid=True)) == 1
    )  # only one text should be added

    # Listing documents
    stored_docs = dataset.list_documents_by_id()

    assert (
        len(stored_docs) == 17
    ), "Invalid length from 'list_documents_by_id'"  # 16 from pdf + one text
    # print(dataset.unique_list_of_documents())

    assert (
        len(dataset.unique_list_of_documents()) == 2
    ), "'unique_list_of_documents' should be limited to 2 (pdf + text)"

    assert dataset.get_by_id(stored_docs[0][0], include=["metadatas"])[
        "metadatas"
    ], "get_by_id failed"

    print(
        "Found not empty", dataset.get(where={"source": "sth"}, include=["documents"])
    )

    assert dataset.get_containing_field("file_path", include=["metadatas"])[
        "metadatas"
    ], "file_path should be in metadata for pdf files"

    # Searching
    assert dataset.search_by_name(
        "2302.00386.pdf", regex_match=True, include=["metadatas"]
    )["metadatas"], "search_by_name should contain found data"

    assert (
        len(dataset.similarity_search("new feature", n_results=2)) == 2
    ), "similarity_search should return 2 results"

    assert (
        len(dataset.similarity_search_with_scores("new feature", n_results=3)) == 3
    ), "similarity_search_with_scores should return 3 results"

    assert (
        len(
            dataset.search_by_field(
                "source", "00386", regex_match=True, include=["metadatas"]
            )["metadatas"]
        )
        == 16
    ), "search_by_field should return 14 results (from 2302.00386.pdf)"


@pytest.mark.skipif(
    not OPENAI_AVAILABLE,
    reason="OpenAI not available (not installed or invalid key)",
)
@pytest.mark.filterwarnings("ignore:.* was deprecated*")
def test_llm():
    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOpenAI
    from templates import DEFAULT_PROMPT_REGISTER, create_and_register_prompt

    text = """Create short, specific summary for research paper. Identify the following items for given text:
  - Model Name
  - Model category(e.g Object Detection, NLP or image generation)
  - SOTA: if Model is State-of-the-Art
  - New Features: Introduced model components, layers or other features, as keywords
  - New Strategies: New introduced learning strategies
  - Year: Year of publishing

  text: {text}

  {format_instructions}
  """
    create_and_register_prompt(
        name="identify_features",
        template=text,
        input_variables=["text", "format_instructions"],
    )

    prompt = PromptTemplate(
        template=text, input_variables=["text", "format_instructions"]
    )

    DEFAULT_PROMPT_REGISTER.load_defined_prompt(name="identify_features", prompt=prompt)

    embeddings = OpenAIEmbeddings(
        openai_api_key=openai.api_key, model="text-embedding-3-small"
    )
    model = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-3.5-turbo")
    # chain = LLMChain(llm=model, prompt=prompt)

    dataset = PaperDatasetLC(
        db=Chroma(
            embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"}
        ),
        llm=model,
    )

    docs_ids = dataset.add_pdf_file(str(ROOT_PATH / "sample_documents/2302.00386.pdf"))

    print("llm search")
    # print(dataset.llm_search_with_sources("identify_features"))
    dataset.update_document_features(docs_ids[0])
    print(dataset.get_containing_field("new_features", include=["metadatas"]))

    result, docs = dataset.llm_search(
        "what is the architecture of yolov6",
        chain_type="map_reduce",
        return_source_documents=True,
    )
    print("llm search result", result)
