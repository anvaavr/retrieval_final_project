import pickle
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from google.cloud import storage
import re
from my_searching_code import *


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

#### load indices

bucket_name = 'abody'

client = storage.Client()
bucket = client.get_bucket(bucket_name)

blob = bucket.blob('text_index.pkl')
pkl = blob.download_as_string()
idx_text = pickle.loads(pkl)

blob = bucket.blob('titles.pkl')
pkl = blob.download_as_string()
idx_titles = pickle.loads(pkl)

blob = bucket.blob('anchor_text.pkl')
pkl = blob.download_as_string()
idx_anchor = pickle.loads(pkl)

### load dictionaries

blob = bucket.blob('doc_norma.pkl')
pkl = blob.download_as_string()
doc_norma = pickle.loads(pkl)

blob = bucket.blob('anchor_len_dict.pkl')
pkl = blob.download_as_string()
anchor_len = pickle.loads(pkl)

blob = bucket.blob('len_docs_body.pkl')
pkl = blob.download_as_string()
body_len = pickle.loads(pkl)

blob = bucket.blob('titles_len_dict.pkl')
pkl = blob.download_as_string()
titles_len = pickle.loads(pkl)

blob = bucket.blob('body_term_idf.pkl')
pkl = blob.download_as_string()
body_term_idf = pickle.loads(pkl)

blob = bucket.blob('anchor_term_idf.pkl')
pkl = blob.download_as_string()
anchor_term_idf = pickle.loads(pkl)

blob = bucket.blob('title_term_idf.pkl')
pkl = blob.download_as_string()
title_term_idf = pickle.loads(pkl)

blob = bucket.blob('doc_title.pkl')
pkl = blob.download_as_string()
doc_titles = pickle.loads(pkl)

blob = bucket.blob('anchor_idf_bm25.pkl')
pkl = blob.download_as_string()
anchor_idf_bm25 = pickle.loads(pkl)

blob = bucket.blob('body_idf_bm25.pkl')
pkl = blob.download_as_string()
body_idf_bm25 = pickle.loads(pkl)

blob = bucket.blob('title_idf_bm25.pkl')
pkl = blob.download_as_string()
title_idf_bm25 = pickle.loads(pkl)

blob = bucket.blob('avgdl_dict.pkl')
pkl = blob.download_as_string()
avgdl_dict = pickle.loads(pkl)

blob = bucket.blob('page_rank_dict.pkl')
pkl = blob.download_as_string()
page_rank_dict = pickle.loads(pkl)

blob = bucket.blob('page_view_dict.pkl')
pkl = blob.download_as_string()
page_view_dict = pickle.loads(pkl)


@app.route("/search")
def search():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    words_body = list(idx_text.df.keys())
    words_title = list(idx_titles.df.keys())
    words_anchor = list(idx_anchor.df.keys())
    query = query_tokenize(query)

    if len(query) == 1:
        candidate_body = get_candidate_documents_and_scores_bm25(query, words_body, idx_text, body_idf_bm25, body_len,
                                                                 'body_index', avgdl_dict['body'])
        candidate_title = get_candidate_documents_and_scores_bm25(query, words_title, idx_titles, title_idf_bm25,
                                                                  titles_len, 'title_index', avgdl_dict['title'])
        candidate_anchor = get_candidate_documents_and_scores_bm25(query, words_anchor, idx_anchor, anchor_idf_bm25,
                                                                   anchor_len, 'anchor_index',
                                                                   avgdl_dict['anchor_text'])

        merged_dict = merge([candidate_body, candidate_title, candidate_anchor])
        top_n = get_top_n(merged_dict)
        res = get_titles(top_n, doc_titles)
    else:
        candidate_body = get_candidate_binary(query, words_body, idx_text, 'body_index')
        candidate_title = get_candidate_binary(query, words_title, idx_titles, 'title_index')
        candidate_anchor = get_candidate_binary(query, words_anchor, idx_anchor, 'anchor_index')

        merged_dict = merge([candidate_body, candidate_title, candidate_anchor])
        top_n = get_top_n(merged_dict)
        res = get_titles(top_n, doc_titles)

    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    words_body = idx_text.df.keys()
    query = query_tokenize(query)
    Q = generate_query_tfidf_dict(query, words_body, body_term_idf)
    D = get_candidate_documents_and_scores(query, words_body, idx_text, body_term_idf,
                                           body_len,
                                           'body_index')
    cs_dict = cosine_similarity(D, Q, doc_norma)
    top_n = get_top_n(cs_dict)
    res = get_titles(top_n, doc_titles)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    words_title = idx_titles.df.keys()

    query = query_tokenize(query)
    D = get_candidate_documents_and_scores(query, words_title, idx_titles,
                                                                             title_term_idf,
                                                                             titles_len,
                                                                             'title_index')
    res = title_similarity(D, doc_titles)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    words_anchor = idx_anchor.df.keys()

    query = query_tokenize(query)
    D = get_candidate_documents_and_scores(query, words_anchor, idx_anchor,
                                                                             anchor_term_idf,
                                                                             anchor_len,
                                                                             'anchor_index')
    res = title_similarity(D, doc_titles)  # same calculate as title search
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for doc_id in wiki_ids:
        res.append(page_rank_dict[doc_id])

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for doc_id in wiki_ids:
        res.append(page_view_dict[doc_id])

    # END SOLUTION
    return jsonify(res)




if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
