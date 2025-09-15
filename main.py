import os
import fitz
import tqdm
import requests

file_name = "human-nutrition-text.pdf"
file_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

def pdf_exists(file_path: str, file_link: str):
    """
    Check if pdf file exists. If file does not exist, download it.

    Args:
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

def text_formatter(text: str) -> str:
    """
    Replace all \n with space and delete all extra spaces from start and end of the text.

    Args:
         text(str): input text to format.

    Return:
        str: formatted text.
    """

    formatted_text = text.replace("\n", " ").strip()
    return formatted_text

