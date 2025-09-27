import os
import re
import fitz
import torch
import textwrap
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from markdown_it.rules_block import paragraph
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from tqdm.auto import tqdm
from spacy.lang.en import English
from transformers.utils import logging
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer, util

logging.set_verbosity_info()
logging.set_verbosity_debug()

BATCH_SIZE = 32
PAGE_OFFSET = 41 # book main content starts from page 41 of pdf
CHAR_TO_TOKEN = 4
MIN_TOKEN_LENGTH = 30
ACCEPTABLE_ANSWERS = 3
SENTENCE_CHUNK_SIZE = 10

QUERY = "good foods for protein"
file_name = "human-nutrition-text.pdf"
EMBEDDED_TEXT_FILE_PATH = "embedded.csv"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
file_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

nlp = English()
nlp.add_pipe("sentencizer")
embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=DEVICE)
pages_and_chunks = list()

# TODO: There should be a new feature to fix probable typo or semantic mistakes in the query
def pdf_exists(file_path: str, file_link: str):
    """
    Check if pdf file exists. If file does not exist, download it.

    Params:
         file_path(str): path of pdf file.
         file_link(str): link to download pdf file.
    """

    if not os.path.exists(file_path):
        print("File doesn't exist, downloading...")
        response = requests.get(file_link)

        if response.status_code == 200:
            with open(file_name, "wb") as file:
                file.write(response.content)
                print(f"PDF file saved in {file_name}")

        else:
            print(f"PDF file could not be downloaded due to error: {response.status_code}")
    else:
        print("File exists.")

def punctuation_formatting(text: str) -> str:
    """
    Find all statements that end with ! or ? (including combinations) and replace them with periods.

    Params:
        text(str): input text to format.

    Return:
        str: text with ! and ? replaced by periods.
    """

    pattern = r'[!?]+(?=\s|$)'
    formatted_text = re.sub(pattern, '.', text)
    return formatted_text

def text_formatter(text: str) -> str:
    """
    Replace all \n with space and delete all extra spaces from start and end of the text.

    Params:
         text(str): input text to format.

    Return:
        str: formatted text.
    """

    formatted_text = punctuation_formatting(text)
    formatted_text = formatted_text.replace("\n", " ").strip()
    return formatted_text

def read_pdf(file_path: str) -> list[dict]:
    """
    Read pdf file page by page and extracts statistics.

    Params:
        file_path(str): path of the pdf file.

    Returns:
        list[dict]: extracted statistics containing page number, number of characters, number of words, number of
        sentences, number of tokens and the whole formatted text.
     """

    result = list()
    pdf = fitz.open(file_path)

    for page_number, page in tqdm(enumerate(pdf)):
        text = page.get_text()
        text = text_formatter(text)
        temp = {"page_number": page_number - PAGE_OFFSET,
                "characters_count": len(text),
                "words_count": len(text.split(" ")),
                "sentences_count": len(text.split(". ")),
                "token_count": len(text) / CHAR_TO_TOKEN,
                "text": text
                }
        result.append(temp)
    return result

def split_page(page_sentences: list, chunk_size: int) -> list[list[str]]:
    """
    slices input list to another sub_lists with size chunk_size. last sub_list may have size less than chunk size.

    Params:
        page_sentences(list): contains all sentences of a page.
        chunk_size(int): size of sub_lists
    Returns:
         list[list[str]]: a list contains sub_lists of sentences.
    """

    page_sentences = [str(sentence) for sentence in page_sentences]
    return [page_sentences[i: i + chunk_size] for i in range(0, len(page_sentences), chunk_size)]

def print_wrapped(text: str, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def retrieve_relevant_resources(query: str, embedded_book: torch.tensor, model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5, print_time: bool=True) -> tuple:
    embedded_query = model.encode(query, convert_to_tensor=True)

    start_time = timer()
    dot_score = util.dot_score(a=embedded_query, b=embedded_book)[0]
    end_time = timer()

    if print_time:
        print(f"Take {end_time - start_time:.5f} for array with length {len(embeddings)}")

    scores, indices = torch.topk(dot_score, k=n_resources_to_return)
    return scores, indices

def print_top_results_and_scores(query: str, embedded_book: torch.tensor, pages_and_chunks_embedded: list[dict]):

    scores, indices = retrieve_relevant_resources(query, embedded_book)

    print(f"Query: {QUERY}")
    print("Results:")

    for score, index in zip(scores, indices):
        print(f"score: {score:.4f}")
        print("Text:")
        print_wrapped(pages_and_chunks_embedded[index]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks_embedded[index]["page_number"]}")

    doc = fitz.open(file_name)
    page = doc.load_page(int(scores[0] + PAGE_OFFSET))
    img = page.get_pixmap(dpi=300)

    doc.close()

    img_array = np.frombuffer(img.samples_mv, dtype=np.uint8).reshape((img.h, img.w, img.n))
    plt.figure(figsize=(13, 10))
    plt.imshow(img_array)
    plt.title(f"Query: '{QUERY}' | Most relevant page:")
    plt.axis('off')
    plt.show()


pages_and_text = read_pdf(file_name)

for item in tqdm(pages_and_text):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences_count_spacy"] = len(item["sentences"])

for item in tqdm(pages_and_text):
    item["sentence_chunks"] = split_page(item["sentences"], SENTENCE_CHUNK_SIZE)
    item["chunk_count"] = len(item["sentence_chunks"])

for item in tqdm(pages_and_text):
    for sentence_chunk in item["sentence_chunks"]:
        chunk = dict()
        chunk["page_number"] = item["page_number"]

        paragraph = "".join(sentence_chunk).replace("  ", " ").strip()
        paragraph = re.sub(r'\.([A-Z])', r'. \1', paragraph)

        chunk["paragraph"] = paragraph
        chunk["char_count"] = len(paragraph)
        chunk["word_count"] = [word for word in paragraph.split(" ")]
        chunk["token_count"] = len(paragraph) / CHAR_TO_TOKEN

        pages_and_chunks.append(chunk)

data = pd.DataFrame(pages_and_chunks)
pages_and_chunks_main = data[data["token_count"] > MIN_TOKEN_LENGTH].to_dict(orient="records")
text_chunks = [item["paragraph"] for item in pages_and_chunks_main]

# embedded_text = embedding_model.encode(text_chunks, batch_size=BATCH_SIZE, convert_to_tensor=True)
# embedded_text_df = pd.DataFrame(embedded_text).to_csv(EMBEDDED_TEXT_FILE_PATH)
embedded_text = pd.read_csv("text_chunks_and_embeddings_df.csv")
embedded_text["embedding"] = embedded_text["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
embedded_pages_and_chunks = embedded_text.to_dict(orient="records")
embeddings = torch.tensor(np.array(embedded_text["embedding"].to_list()), dtype=torch.float32).to(DEVICE)

# embedded_query = embedding_model.encode(QUERY, convert_to_tensor=True)
# start_time = timer()
# dot_score = util.dot_score(a=embedded_query, b=embeddings)[0]
# end_time = timer()
# print(f"Take {end_time - start_time:.5f} for array with length {len(embeddings)}")

# top_results_dot_score = torch.topk(dot_score, k=ACCEPTABLE_ANSWERS)

# print(f"Query: {QUERY}")
# print("Results:")

# for score, index in zip(top_results_dot_score[0], top_results_dot_score[1]):
#      print(f"score: {score:.4f}")
#      print("Text:")
#      print_wrapped(embedded_pages_and_chunks[index]["sentence_chunk"])
#      print(f"Page number: {embedded_pages_and_chunks[index]["page_number"]}")

# doc = fitz.open(file_name)
# page = doc.load_page(int(top_results_dot_score[1][0] + PAGE_OFFSET))
# img = page.get_pixmap(dpi=300)
#
# doc.close()
#
# img_array = np.frombuffer(img.samples_mv, dtype=np.uint8).reshape((img.h, img.w, img.n))
# plt.figure(figsize=(13, 10))
# plt.imshow(img_array)
# plt.title(f"Query: '{QUERY}' | Most relevant page:")
# plt.axis('off')
# plt.show()

print_top_results_and_scores(QUERY, embeddings, embedded_pages_and_chunks)