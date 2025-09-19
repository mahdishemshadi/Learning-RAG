import os
import re
import fitz
import random
import requests
import pandas as pd

from tqdm.auto import tqdm
from spacy.lang.en import English

PAGE_OFFSET = 41 # book main content starts from page 41 of pdf
CHAR_TO_TOKEN = 4
SENTEMCE_CHUNK_SIZE = 10
file_name = "human-nutrition-text.pdf"
file_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

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

    return [page_sentences[i: i+1] for i in range(0, len(page_sentences), chunk_size)]


pages_and_text = read_pdf(file_name)

nlp = English()
nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_text):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences_count_spacy"] = len(item["sentences"])

for item in tqdm(pages_and_text):
    item["sentence_chunks"] = split_page(item["sentences"], SENTEMCE_CHUNK_SIZE)
    item["chunk_count"] = len(item["sentence_chunks"])

random_pages = random.sample(pages_and_text, k=1)
data = pd.DataFrame(pages_and_text)
print(data.describe().round(2))
