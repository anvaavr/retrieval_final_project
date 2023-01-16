### retrieval_final_project
#### general information
 1) final_proj_IR contains all related code for building the inverted_index and all dictionaries we used in our implementation
 2) inverted_index_gcp is the file given to us in homework 3 and 4. we changes it a little bit to be fitted to our implementation and usage of our google bucket
 3) my_searching_code contains all function we used inorder to complete the search_frontend file
 4) search_fronted is the main search file
 
#### explenation about the functions

This script is a search engine implementation that uses TF-IDF and BM25 to retrieve relevant documents for a given query. The script includes several functions that perform different tasks in the search process:


##### Functionality
**generate_query_tfidf_dict** takes a query and a dictionary of words and their idf values, and returns a dictionary of the query's words and their tf-idf values.

**get_candidate_documents_and_score**s takes a query, a dictionary of words, an index, an idf dictionary, a dictionary of document lengths, and a blob name and returns a dictionary of candidate documents and their scores.

**get_candidate_binary** takes a query, a dictionary of words, an index, and a blob name, and returns a dictionary of candidate documents and the number of times each word appears in them.

**cosine_similarity** takes two dictionaries, D and Q, representing the tf-idf vectors of the candidate documents and the query, respectively, and a dictionary of document norm values. It returns a dictionary of document ids and their cosine similarity scores.

**get_top_n** takes a dictionary of similarity scores and returns the top N documents with the highest scores.

**query_tokenize** takes a query and returns a list of the query's words after tokenizing, lowercasing, and removing stopwords.

**get_titles** takes a list of the top documents and a dictionary of document titles, and returns a list of the titles of the top documents.

**The title_similarity** function takes a dictionary D of candidate documents and their similarity scores and a dictionary of document titles, and returns a list of tuples where the first element is the document id and the second element is the title of the document.

The **merge** function takes a list of candidate dictionaries and returns a single dictionary that is the result of merging all of the input dictionaries. Each input dictionary represents a different feature (e.g. term frequency, title similarity) that contributes to the final score of each candidate document. This function gives different weights to the different features, by multiplying the score of each feature with a different weighting factor.

The **get_candidate_documents_and_scores_bm25** function takes a query, a dictionary of words, an index, an idf dictionary, a dictionary of document lengths, a blob name, the average length of documents, and two parameters b and k1 for the BM25 ranking function. It returns a dictionary of candidate documents and their scores. The function calculates the BM25 score for each term in the query for each document in the index, and then sums up the scores for each document to get the final score for each candidate document. BM25 is a ranking function that is commonly used in information retrieval systems to rank the relevance of documents to a given query.




