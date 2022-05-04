#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:55:36 2021
@author: svizciano
"""
import pandas as pd
import numpy  as np
from AdvancedAnalytics.Text import text_analysis, text_plot

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition           import LatentDirichletAllocation
from sklearn.decomposition           import TruncatedSVD
from sklearn.decomposition           import NMF
 
#from collections import Counter
#from PIL         import Image
 
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
 
df   = pd.read_excel("CaliforniaChardonnay.xlsx")
text = df['description']
 
ta   = text_analysis(synonyms=None, stop_words=['drink', 'chardonnay', 
                                                'like', 'one', 'wine'],
                     pos=True, stem=True)
cv   = CountVectorizer(max_df=0.95, min_df=0.05, max_features=None,
                       binary=False, analyzer=ta.analyzer)
tf    = cv.fit_transform(text)
terms = cv.get_feature_names()

# Constants
m_features      = None  # default is None, numerical values.
n_topics        =  9   # number of topics
max_iter        = 10   # maximum number of iterations
max_df          = 1.0  # max proportion of docs/reviews allowed for a term
learning_offset = 10.      # default is 10
learning_method = 'online' # alternative is 'batch' for large files
tfidf = True  # Use TF-IDF Weighting if True
svd   = False # Use SVD topic factorization if True

def sortSecond(e):
    return e[1]
# Show Word Cloud based on TFIDF weighting
if tfidf == True:
    # Construct the TF/IDF matrix from the data
    print("\nConducting Term/Frequency Matrix using TF-IDF")
    # Default for norm is 'l2', use norm=None to supress
    tfidf_vect = TfidfTransformer(norm=None, use_idf=True)
    # tf matrix is (n_reviews)x(m_features)
    tf = tfidf_vect.fit_transform(tf) 
    
    term_idf_sums = tf.sum(axis=0)
    term_idf_scores = []
    for i in range(len(terms)):
        term_idf_scores.append([terms[i], term_idf_sums[0,i]])
    print("The Term/Frequency matrix has", tf.shape[0],\
          " rows, and", tf.shape[1], " columns.")
    print("The Term list has", len(terms), " terms.")
    term_idf_scores.sort(key=sortSecond, reverse=True)
    print("\nTerms with Highest TF-IDF Scores:")
    term_cloud= {}
    n_terms = len(terms)
    for i in range(n_terms):
        term_cloud[term_idf_scores[i][0]] = term_idf_scores[i][1]
        if i < 10:
            print('{:<15s}{:>8.2f}'.format(term_idf_scores[i][0], 
              term_idf_scores[i][1]))
            
dcomp='lda'
if dcomp == 'lda':
    # LDA Analysis
    uv = LatentDirichletAllocation(n_components=n_topics, 
                                   max_iter=max_iter,
                            learning_method=learning_method, 
                            learning_offset=learning_offset, 
                            random_state=12345)
if dcomp == 'svd':
    # In sklearn, SVD is synonymous with LSA (Latent Semantic Analysis)
    uv = TruncatedSVD(n_components=n_topics, algorithm='arpack',
                                    tol=0, random_state=12345)
   
if dcomp == 'nmf':
    uv = NMF(n_components=n_topics, random_state=12345, alpha=0.1, 
             l1_ratio=0.5)

if dcomp == 'kld':
    uv = NMF(n_components=n_topics, random_state=12345, alpha=0.1,    
             l1_ratio=0.5, beta_loss='kullback-leibler', solver='mu', 
             max_iter=1000)
    
U = uv.fit_transform(tf)
# Display the topic selections
print("\n********** GENERATED TOPICS **********")
text_analysis.display_topics(uv, terms, n_terms=20, word_cloud=True)

# Store predicted topic and its probability in array <topics>
n_reviews = df.shape[0]
# Initialize <topics> to all zeros
topics    = np.array([0]*n_reviews, dtype=float)
# Assign topics to reviews based on largest row value in U
df_topics = text_analysis.score_topics(U, display=True)
df        = df.join(df_topics)

# Prepare and Print Avg Points and Price by Topic
df1       = df.groupby('topic')['description'].count()
df_topics = df.groupby('topic')[['points', 'price']].mean()
df_topics = df_topics.join(df1)
df_topics['percent'] = \
                  100*df_topics['description']/df_topics['description'].sum()
print("\nTopic   Points      Price    Description  Percent")
print("-------------------------------------------")
for i in range(n_topics):
    print("{:>3d}{:>11.1f}{:>9.2f}{:>9d}{:>9.1f}%".format(i, 
                                            df_topics['points'].loc[i],
                                            df_topics['price'].loc[i],
                                            df_topics['description'].loc[i],
                                            df_topics['percent'].loc[i]))
print("-------------------------------------------\n")

