import re
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")


data = pd.read_csv("Novel.csv")
data['text'] = data['Title'] + ' ' + data['Deskripsi']
data['text'] = data['text'].str.lower()
data['Genre_low'] = data['Genre'].str.lower()
data['Author_low'] = data['Author'].str.lower()


def preprocess_text(text):
    # remove numbers
    text = re.sub(r"\d+", "", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove whitespace leading & trailing
    text = text.strip()
    # remove multiple whitespace into single whitespace
    text = re.sub('\s+', ' ', text)
    # remove single char
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    return text


for col in ['text', 'Genre_low', 'Author_low']:
    data[col] = data[col].apply(preprocess_text)

stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))


def stopwords_removal(words):
    return list(set(words) - stop_words)


data['text'] = data['text'].apply(stopwords_removal)
data['text'] = data['text'].agg(lambda x: ' '.join(map(str, x)))
data['text'] = data['text'] + ' ' + \
    data['Author_low'] + ' ' + data['Genre_low']

# Tag documents
tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)])
               for i, doc in enumerate(data['text'])]

# Train Doc2Vec models
pvdm_model = Doc2Vec(tagged_data, vector_size=100,
                     window=3, min_count=2, workers=6, dm=1)
pvdbow_model = Doc2Vec(tagged_data, vector_size=100,
                       window=3, min_count=2, workers=6, dm=0)

# Compute PVDM and PVDBOW vectors for each document
data['pvdm_vector'] = data['text'].apply(
    lambda x: pvdm_model.infer_vector(x.split()))
data['pvdbow_vector'] = data['text'].apply(
    lambda x: pvdbow_model.infer_vector(x.split()))

# Compute similarity scores
pvdm = data['pvdm_similarity'] = data['pvdm_vector'].apply(
    lambda x: cosine_similarity([x], list(data['pvdm_vector'])).flatten())
pvdbow = data['pvdbow_similarity'] = data['pvdbow_vector'].apply(
    lambda x: cosine_similarity([x], list(data['pvdbow_vector'])).flatten())

# reset index dataframe
data = data.reset_index()
titles = data['Title']

# membuat map dari index dan judul buku
indices = pd.Series(data.index, index=data['Title']).drop_duplicates()


def rec_pvdbow(Title, pvdbow=pvdbow):
    recommendation = pd.DataFrame(columns=['Book Idx', 'Title', 'Score'])
    count = 0

    idx = indices[Title]
    sim_scores = list(enumerate(pvdbow[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]

    for i in book_indices:
        recommendation.at[count, 'Book Idx'] = book_indices[count]
        recommendation.at[count, 'Title'] = titles.iloc[book_indices[count]]
        recommendation.at[count, 'Score'] = sim_scores[count][1]
        count += 1

    # Convert DataFrame to HTML table
    table_html = recommendation.to_html(index=False)

    return table_html
