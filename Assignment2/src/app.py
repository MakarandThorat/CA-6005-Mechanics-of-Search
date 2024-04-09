from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import xml.etree.ElementTree as ET
import re
import nltk
import numpy as np
import operator
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import os
from flask import Flask, render_template, request

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

#PreProcessing 

def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence)

def get_tokenized_list(doc_text):
    tokens = nltk.word_tokenize(doc_text)
    return tokens

#removing stop words
def remove_stopwords(doc_text):
    cleaned_text = []
    for words in doc_text:
        if words not in stop_words:
            cleaned_text.append(words)
    return cleaned_text


def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

def stem_lem(tokens):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    processed_tokens = []
    for token in tokens:
        lemmatized_token = lemmatizer.lemmatize(token)
        stemmed_token = stemmer.stem(lemmatized_token)
        processed_tokens.append(stemmed_token)
    return processed_tokens



#preprocessing our doc

def preprocess(element,flag):

    cleaned_corpus = [ ]

    element = remove_punctuation(element)
    #print("Punctuation removal :",element)
    type(element)
    tokens = get_tokenized_list(element)
    #print("token creation",tokens)
    doc_text = remove_stopwords(tokens)
    #print("stop word removal",doc_text)

    for w in lemmatize_words(doc_text):
        cleaned_corpus.append(w)
#     for w in word_stemmer(doc_text):
#         cleaned_corpus.append(w)
    #cleaned_corpus = ''.join(cleaned_corpus)
    #print("cleaned_corpus",cleaned_corpus) 
    if(flag==1):
        answer=''
        for x in cleaned_corpus:
            answer += " "  + x
        return answer.lower()
    else:
        return cleaned_corpus

app = Flask(__name__)

def read_surrogates(folder_path):
    surrogates = []
    filenames = os.listdir(folder_path)
    surrogate_filenames = [filename for filename in filenames if filename.endswith('_surrogate.txt')]
    surrogate_indices = [int(re.search(r'\d+', filename).group()) for filename in surrogate_filenames]
    sorted_surrogate_indices = sorted(surrogate_indices)
    for surrogate_index in sorted_surrogate_indices:
        filename = f'image_{surrogate_index}_surrogate.txt'
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            surrogate = file.read().strip()
            surrogates.append(surrogate)
    return surrogates

def rank_images(query, surrogates):
    # Preprocess the query and surrogates
    preprocessed_surrogates = [preprocess(surrogate,1) for surrogate in surrogates]
    preprocessed_query = preprocess(query,1)

    # Vectorize the surrogates
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(preprocessed_surrogates)

    # Transform the query to the same vector space as the surrogates
    query_vector = vectorizer.transform([preprocessed_query])

    # Compute the cosine similarity between the query vector and the surrogate vectors
    cosine_similarities = linear_kernel(query_vector, doc_vectors).flatten()

    # Pair surrogate indices with their scores
    scored_surrogates = list(enumerate(cosine_similarities))

    # Sort the surrogates by score in descending order
    sorted_surrogates = sorted(scored_surrogates, key=operator.itemgetter(1), reverse=True)

    # Get the top 5 surrogates
    top_surrogates = sorted_surrogates[:5]
    

    # Prepare the results
    results = []
    for surrogate_id, score in top_surrogates:
        image_path = f'images/image_{surrogate_id}.jpg'
        results.append({'surrogate': surrogates[surrogate_id], 'image_path': image_path, 'score': round(score, 2)})
    
    return results


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    surrogates = read_surrogates('static/images')
    results = rank_images(query, surrogates)
    filtered_results = [result for result in results if result['score'] > 0]
    return render_template('search.html', query=query, results=filtered_results)

if __name__ == '__main__':
    app.run(debug=True)
    