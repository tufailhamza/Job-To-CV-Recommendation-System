# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 07:26:08 2024

@author: hp
"""

import os
import re
from collections import defaultdict

def read_files_from_folder(folder):
    file_contents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                file_contents.append(file.read().lower())
    return file_contents

def tokenize(text):
    # Simple tokenizer to split on non-alphanumeric characters
    return re.findall(r'\b\w+\b', text)

def build_inverted_index(documents):
    inverted_index = defaultdict(set)
    for doc_id, document in enumerate(documents):
        tokens = tokenize(document)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            inverted_index[token].add(doc_id)
    return inverted_index

def calculate_document_frequencies(inverted_index, total_documents):
    doc_frequencies = {term: len(doc_ids) for term, doc_ids in inverted_index.items()}
    return doc_frequencies

def write_stopwords_to_file(doc_frequencies, total_documents, threshold=0.75):
    with open('stopwords.txt', 'w', encoding='utf-8') as stopwords_file:
        for term, freq in doc_frequencies.items():
            if freq / total_documents >= threshold:
                stopwords_file.write(term + '\n')

def main():
    cv_folder = 'CV'
    jd_folder = 'JD'

    # Read files from both folders
    cv_documents = read_files_from_folder(cv_folder)
    jd_documents = read_files_from_folder(jd_folder)

    # Combine all documents
    all_documents = cv_documents + jd_documents
    total_documents = len(all_documents)

    # Build inverted index
    inverted_index = build_inverted_index(all_documents)

    # Calculate document frequencies
    doc_frequencies = calculate_document_frequencies(inverted_index, total_documents)

    # Write stopwords to file
    write_stopwords_to_file(doc_frequencies, total_documents)

if __name__ == "__main__":
    main()
