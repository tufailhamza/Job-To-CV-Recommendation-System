import re
import os
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import string
import time
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()

folder_paths = {'CV': 'CV', 'JD': 'JD'}
inverted_index_file = 'inverted_index.json'
document_texts_file = 'documents_text.json'

unique_id_base = 0

lookup_table = {}
document_texts = []
document_texts_CV = []
document_texts_JD = []

def generate_unique_id(num):
    return f'{num:010d}'


def read_stopwords(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='latin-1') as f:
            stopwords = set(f.read().splitlines())
        return stopwords
    except Exception as e:
        print(f"Error reading stopwords file: {e}")
        return set()

def read_and_process_file(file_path, unique_id, stopwords_path):
    try:
        stopwords = read_stopwords(stopwords_path)
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
            tokens = word_tokenize(content)
            lowercase_tokens = [token.lower() for token in tokens]
            table = str.maketrans('', '', string.punctuation)
            stripped_tokens = [token.translate(table) for token in lowercase_tokens]

            stemmer = PorterStemmer()
            filtered_tokens = []
            for token in stripped_tokens:
                if token.isalpha() and len(token) >= 2 and len(token) <= 17:
                    if token not in stopwords:
                        stemmed_token = stemmer.stem(token)
                        stemmed_token = stemmed_token.encode('utf-8').decode('unicode-escape')
                        filtered_tokens.append(stemmed_token)

            return ' '.join(filtered_tokens)
    except UnicodeDecodeError:
        print(f"Failed to decode {file_path}")
        return ''
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ''


if os.path.exists(inverted_index_file):
    with open(inverted_index_file, 'r') as f:
        inverted_index_data = json.load(f)
        inverted_index = inverted_index_data['inverted_index']
        lookup_table = inverted_index_data['lookup_table']
        unique_id_base = inverted_index_data['unique_id_base']
    
    if os.path.exists(document_texts_file):
        with open(document_texts_file, 'r') as f:
            document_texts = json.load(f)
    else:
        document_texts = []
        for file, unique_id in lookup_table.items():
            folder_prefix = 'CV' if file.startswith('CV_') else 'JD'
            file_path = os.path.join(folder_paths[folder_prefix], file[len(folder_prefix) + 1:])
            document_texts.append(read_and_process_file(file_path, unique_id, 'stopwords.txt'))
else:
    inverted_index = {}
    lookup_table = {}
    unique_id_base = 0

    for folder_prefix, folder_path in folder_paths.items():
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                unique_id = generate_unique_id(unique_id_base)
                unique_id_base += 1
                lookup_table[f'{folder_prefix}_{file}'] = unique_id
                processed_text = read_and_process_file(file_path, unique_id, 'stopwords.txt')
                document_texts.append(processed_text)

                if folder_prefix == 'CV':
                    document_texts_CV.append(processed_text)
                elif folder_prefix == 'JD':
                    document_texts_JD.append(processed_text)

                for term in set(processed_text.split()):
                    if term not in inverted_index:
                        inverted_index[term] = []
                    inverted_index[term].append(unique_id)

    with open(document_texts_file, 'w') as f:
        json.dump(document_texts, f, indent=4)

    with open(inverted_index_file, 'w') as f:
        json.dump({'inverted_index': inverted_index, 'lookup_table': lookup_table, 'unique_id_base': unique_id_base}, f, indent=4)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(document_texts)
terms = tfidf_vectorizer.get_feature_names_out()

indices_CV = [i for i, file_name in enumerate(lookup_table.keys()) if file_name.startswith('CV_')]
indices_JD = [i for i, file_name in enumerate(lookup_table.keys()) if file_name.startswith('JD_')]

tfidf_matrix_CV = tfidf_matrix[indices_CV, :]
tfidf_matrix_JD = tfidf_matrix[indices_JD, :]

# Elbow method to find the optimal number of clusters
def find_optimal_clusters(data):
    distortions = []
    K = range(1, 15)
    for k in K:
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(data)
        distortions.append(kmean_model.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    return distortions.index(min(distortions[1:])) + 1

find_optimal_clusters(tfidf_matrix_CV)
find_optimal_clusters(tfidf_matrix_JD)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

tfidf_matrix_CV_pca = pca.fit_transform(tfidf_matrix_CV.toarray())

tfidf_matrix_JD_pca = pca.fit_transform(tfidf_matrix_JD.toarray())

print("Shape of tfidf_matrix_CV:", tfidf_matrix_CV.shape)
print("Shape of tfidf_matrix_JD:", tfidf_matrix_JD.shape)

num_clusters = 9
kmeans_CV = KMeans(n_clusters=num_clusters, random_state=0).fit(tfidf_matrix_CV)
kmeans_JD = KMeans(n_clusters=num_clusters, random_state=0).fit(tfidf_matrix_JD)

labels_CV = kmeans_CV.labels_
labels_JD = kmeans_JD.labels_

reverse_lookup_table = {v: k for k, v in lookup_table.items()}

# print("\nCV Clusters:")
for cluster_num in range(num_clusters):
    cluster_docs = [indices_CV[i] for i in range(len(labels_CV)) if labels_CV[i] == cluster_num]
    cluster_doc_ids = [reverse_lookup_table[f'{i:010d}'] for i in cluster_docs]
#    print(f"Cluster {cluster_num + 1}: {', '.join(cluster_doc_ids)}")

# print("\nJD Clusters:")
for cluster_num in range(num_clusters):
    cluster_docs = [indices_JD[i] for i in range(len(labels_JD)) if labels_JD[i] == cluster_num]
    cluster_doc_ids = [reverse_lookup_table[f'{i:010d}'] for i in cluster_docs]
#    print(f"Cluster {cluster_num + 1}: {', '.join(cluster_doc_ids)}")


# print(reverse_lookup_table)


plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    cluster_indices = np.where(labels_CV == i)[0]
    plt.scatter(tfidf_matrix_CV_pca[cluster_indices, 0], tfidf_matrix_CV_pca[cluster_indices, 1], label=f'Cluster {i+1}')
plt.title('PCA Visualization for CV')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    cluster_indices = np.where(labels_JD == i)[0]
    plt.scatter(tfidf_matrix_JD_pca[cluster_indices, 0], tfidf_matrix_JD_pca[cluster_indices, 1], label=f'Cluster {i+1}')
plt.title('PCA Visualization for JD')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

chosen_folder = input("Enter 'CV' or 'JD' to select the folder: ")

if chosen_folder == 'CV':
    print("\nDocument names in CV:")
    for file_name in os.listdir(folder_paths['CV']):
        print(file_name)
elif chosen_folder == 'JD':
    print("\nDocument names in JD:")
    for file_name in os.listdir(folder_paths['JD']):
        print(file_name)
else:
    print("Invalid input. Please enter 'CV' or 'JD'.")

chosen_document = input("Enter the document name (e.g., 12): ")
if chosen_folder == 'CV':
    chosen_document = "CV_" + chosen_document + ".txt"
elif chosen_folder == 'JD':
    chosen_document = "JD_row_" + chosen_document + ".txt"

chosen_document_index = [i for i, file_name in enumerate(lookup_table.keys()) if file_name == chosen_document][0]
chosen_document_tfidf = tfidf_matrix[chosen_document_index, :]

if chosen_folder == 'CV':
    cluster_centroids_CV = kmeans_JD.cluster_centers_
elif chosen_folder == 'JD':
    cluster_centroids_CV = kmeans_CV.cluster_centers_

chosen_document_tfidf_transpose = chosen_document_tfidf.T
chosen_document_tfidf_dense = chosen_document_tfidf_transpose.toarray().reshape(-1)

dot_products = []
for centroid in cluster_centroids_CV[0:num_clusters]:
    dot_product = np.dot(chosen_document_tfidf_dense, centroid)
    dot_products.append(dot_product)

for i, dot_product in enumerate(dot_products, start=1):
    print(f"DOT between chosen_document_tfidf and cluster_centroids_CV[{i}]: {dot_product}")

highest_similarity_cluster_index = np.argmax(dot_products)
print(f"The cluster with the highest similarity is: Cluster {highest_similarity_cluster_index + 1}")

if chosen_folder == 'CV':
    cluster_docs_indices = [indices_JD[i] for i in range(len(labels_JD)) if labels_JD[i] == highest_similarity_cluster_index]
elif chosen_folder == 'JD':
    cluster_docs_indices = [indices_CV[i] for i in range(len(labels_CV)) if labels_CV[i] == highest_similarity_cluster_index]

cosine_similarities = []
for doc_index in cluster_docs_indices:
    cluster_doc_tfidf = tfidf_matrix[doc_index, :]
    cosine_similarity = np.dot(chosen_document_tfidf_dense, cluster_doc_tfidf.toarray().reshape(-1))
    cosine_similarities.append((reverse_lookup_table[f'{doc_index:010d}'], cosine_similarity))

cosine_similarities.sort(key=lambda x: x[1], reverse=True)

if len(cosine_similarities) > 5:
    cosine_similarities = cosine_similarities[:5]
else:
    cosine_similarities = cosine_similarities

print(f"\nDocuments in Cluster {highest_similarity_cluster_index + 1} by descending similarity:")
for doc_id, similarity in cosine_similarities:
    print(f"Document ID: {doc_id}, Similarity: {similarity}")

from wordcloud import WordCloud

cluster_documents = [document_texts[index] for index in cluster_docs_indices]
cluster_text = ' '.join(cluster_documents)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Cluster')
plt.axis('off')
plt.show()

user_feedback = {}
relevant_doc_ids = []
irrelevant_doc_ids = []

for doc_id, _ in cosine_similarities:
    feedback = input(f"Is document {doc_id} relevant? (yes/no): ")
    user_feedback[doc_id] = feedback.lower() == 'yes'

relevant_docs = [doc_id for doc_id, relevant in user_feedback.items() if relevant]
irrelevant_docs = [doc_id for doc_id, relevant in user_feedback.items() if not relevant]


relevant_doc_ids = []
irrelevant_doc_ids = []

for file_name in relevant_docs:
    for key, value in reverse_lookup_table.items():
        if value == file_name:
            relevant_doc_ids.append(key)
            break
        
for file_name in relevant_docs:
    for key, value in reverse_lookup_table.items():
        if value == file_name:
            irrelevant_doc_ids.append(key)
            break


import numpy as np
from scipy.sparse import csr_matrix

def rocchio_feedback(query_vector, relevant_doc_ids, irrelevant_doc_ids, tfidf_matrix, alpha=1, beta=0.75, gamma=0.15):
    relevant_doc_ids = [int(doc_id) for doc_id in relevant_doc_ids]
    irrelevant_doc_ids = [int(doc_id) for doc_id in irrelevant_doc_ids]

    relevant_vectors = tfidf_matrix[relevant_doc_ids, :]
    irrelevant_vectors = tfidf_matrix[irrelevant_doc_ids, :]

    relevant_sum = np.sum(relevant_vectors, axis=0)
    irrelevant_sum = np.sum(irrelevant_vectors, axis=0)

    num_relevant = relevant_vectors.shape[0]
    num_irrelevant = irrelevant_vectors.shape[0]

    updated_query_vector = alpha * query_vector + beta * (relevant_sum / num_relevant) - gamma * (irrelevant_sum / num_irrelevant)

    if isinstance(updated_query_vector, csr_matrix):
        updated_query_vector = updated_query_vector.toarray().reshape(-1)
    else:
        updated_query_vector = np.array(updated_query_vector).reshape(-1)
    
    return updated_query_vector

updated_query_vector = rocchio_feedback(chosen_document_tfidf_dense, 
                                        relevant_doc_ids,
                                        irrelevant_doc_ids, tfidf_matrix)

if chosen_folder == 'CV':
    other_space_tfidf_matrix = tfidf_matrix_JD
    other_space_indices = indices_JD
elif chosen_folder == 'JD':
    other_space_tfidf_matrix = tfidf_matrix_CV
    other_space_indices = indices_CV

dot_products_updated = []

for i in range(other_space_tfidf_matrix.shape[0]):
    doc_tfidf_vector = other_space_tfidf_matrix[i, :].toarray().reshape(-1)
    dot_product = np.dot(updated_query_vector, doc_tfidf_vector)
    dot_products_updated.append((i, dot_product))

dot_products_updated.sort(key=lambda x: x[1], reverse=True)

N = 5
most_similar_documents = dot_products_updated[:N]


print("\nMost similar documents after relevance feedback:")
user_rankings = []

for rank, (doc_index, similarity) in enumerate(most_similar_documents, start=1):

    doc_id = reverse_lookup_table[f'{other_space_indices[doc_index]:010d}']
    print(f"Document ID: {doc_id}, Similarity: {similarity}")
    
    user_rank = int(input(f"Please rank the relevance of Document ID {doc_id} (1 is high, 5 is low): "))
    user_rankings.append((doc_id, user_rank, rank))

user_ranks = [ranking[1] for ranking in user_rankings]
system_ranks = [ranking[2] for ranking in user_rankings]

absolute_errors = [abs(user_rank - system_rank) for user_rank, system_rank in zip(user_ranks, system_ranks)]
mae = sum(absolute_errors) / len(absolute_errors)
print(f"\nMean Absolute Error (MAE): {mae:.2f}")

print("\nRanking comparison:")
for doc_id, user_rank, system_rank in user_rankings:
    print(f"Document ID: {doc_id}, User Rank: {user_rank}, System Rank: {system_rank}")


relevant_docs_count = len(relevant_docs)
total_retrieved_docs = len(cosine_similarities)
true_positives = sum(1 for doc_id, _ in cosine_similarities[:10] if doc_id in relevant_docs)

precision = true_positives / total_retrieved_docs
recall = true_positives / relevant_docs_count
f1_score = 2 * (precision * recall) / (precision + recall)

print("\nEvaluation Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")

end_time = time.time()
elapsed_time = end_time - start_time
print("\nElapsed time:", elapsed_time, "seconds")
