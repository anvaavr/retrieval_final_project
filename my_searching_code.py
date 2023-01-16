import re
from collections import Counter, defaultdict

import numpy as np
from nltk.corpus import stopwords


def generate_query_tfidf_dict(query_to_search, words, body_term_idf):
    Q = {}

    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in words:  # avoid terms that do not appear in the index.
            tf = np.divide(counter[token], len(query_to_search))  # term frequency divded by the length of the query
            idf = body_term_idf[token]
            Q[token] = np.multiply(tf, idf)

    return Q


def get_candidate_documents_and_scores(query_to_search, words, index, idf_dict, docs_len, blob_name):
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = index.read_posting_list(term, blob_name)
            for tup in list_of_doc:
                doc_id = tup[0]
                tf = np.divide(tup[1], docs_len[doc_id])
                idf = idf_dict[term]
                candidates[(doc_id, term)] = np.multiply(tf, idf)

    return candidates


def get_candidate_binary(query_to_search, words, index, blob_name):
    candidates = defaultdict(int)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = index.read_posting_list(term, blob_name)
            for tup in list_of_doc:
                doc_id = tup[0]
                candidates[doc_id] += 1

    return candidates


def cosine_similarity(D, Q, doc_norma):
    cs_dict = {}
    q_norm = np.linalg.norm(np.array([float(x) for x in list(Q.values())]))

    for k in D.keys():
        if k[0] in cs_dict.keys():
            cs_dict[k[0]] += np.multiply(D[k], Q[k[1]])
        else:
            cs_dict[k[0]] = np.multiply(D[k], Q[k[1]])

    for key in cs_dict.keys():
        cs_dict[key] = np.divide(cs_dict[key], np.multiply(doc_norma[key], q_norm))

    return cs_dict


def get_top_n(sim_dict, N=100):
    ret = []
    for key in sim_dict.keys():
        ret.append((key, float(sim_dict[key])))
    ret = sorted(ret, key=lambda x: x[1], reverse=True)
    return ret[:N]


def query_tokenize(query):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    query = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = [token for token in query if token not in all_stopwords]
    return query


def get_titles(top_n_list, doc_titles):
    ret = []
    for tup in top_n_list:
        if tup[0] in doc_titles:
            ret.append((tup[0], doc_titles[tup[0]]))
        else:
            ret.append((tup[0], ''))

    return ret


def title_similarity(D, doc_titles):
    tuples = D.keys()
    all_docs = [x[0] for x in tuples]
    counter = Counter(all_docs)
    sorted_docs = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)

    ret = []
    for doc in sorted_docs:
        if doc in doc_titles:
            ret.append((doc, doc_titles[doc]))
        else:
            ret.append((doc, ''))
    return ret


def merge(c_lst):
    ret_dict = defaultdict(float)
    for c in range(len(c_lst)):
        for doc_id, score in c_lst[c].items():
            if c == 0:
                weighted_score = score * 1
            elif c == 1:
                weighted_score = score * 0.5
            else:
                weighted_score = score * 0.05
            ret_dict[doc_id] += weighted_score
    return ret_dict


def get_candidate_documents_and_scores_bm25(query_to_search, words, index, idf_dict, docs_len, blob_name, avgdl, b=0.75,
                                            k1=1.5):
    candidates = defaultdict(float)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = index.read_posting_list(term, blob_name)
            for tup in list_of_doc:
                doc_id = tup[0]
                tf = tup[1]
                idf = idf_dict[term]
                numerator = np.multiply(idf, np.multiply(tf, (k1 + 1)))
                denominator = tf + np.multiply(k1, (1 - b + np.multiply(b, np.divide(docs_len[doc_id], avgdl))))
                candidates[doc_id] += np.divide(numerator, denominator)

    return candidates
