import os
import re
from docx import Document
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation

def extract_words(file_path: str) -> str:
    doc = Document(file_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return ' '.join(fullText)

def main() -> None:
    unique_words_doc = set()
    total_words = 0
    count = 0
    lemmatizer = WordNetLemmatizer()
    for foldername, subfolders, filenames in os.walk('/Users/tylercranmer/Documents'):
        for filename in filenames:
            if filename.endswith('.docx'):
                count += 1
                try:
                    words = extract_words(os.path.join(foldername, filename))
                    w = re.sub(r'[^-\w\s]+', '', words)
                    tokens = word_tokenize(w)
                    lower_tokens = [token.lower() for token in tokens]
                    filtered_tokens = [token for token in lower_tokens if token not in punctuation]
                    lem_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
                    total_words += len(lem_tokens)
                    unique_words_doc.update(lem_tokens)
                except:
                    print("error reading: ", filename)

    print("number of unique words: ", len(unique_words_doc))
    print("number of documents read: ", count)
    print("total amount of words: ", total_words)
    print("average number of words per document: ", total_words / count)


if __name__ == '__main__':
    main()