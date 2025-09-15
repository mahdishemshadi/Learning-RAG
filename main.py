import os
import requests

file_name = "human-nutrition-text.pdf"
file_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

if not os.path.exists(file_name):
    print("File doesn't exist, downloading...")
    response = requests.get(file_url)

    if response.status_code == 200:
        with open(file_name, "wb") as file:
            file.write(response.content)
            print(f"PDF file saved in {file_name}")

    else:
        print(f"PDF file could not be downloaded due to error: {response.status_code}")
else:
    print("File exists.")



