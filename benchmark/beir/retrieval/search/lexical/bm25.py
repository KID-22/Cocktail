from .. import BaseSearch
from typing import List, Dict
import numpy as np
import math
from rank_bm25 import BM25Okapi
import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm.autonotebook import trange
import logging
import tqdm
# import nltk
# nltk.download('stopwords')
# import nltk
# nltk.download('punkt')


class BM25(BaseSearch):
    def __init__(self, version: int=2, k: float=2, b: float=0.75):
        self.version = version
        self.k = k
        self.b = b
        # load stopwords from stopwords.txt, each line is a stopword
        # pwd_path = os.path.abspath(os.path.dirname(__file__))
        # with open(pwd_path + "/stopwords.txt", 'r') as f:
        #     words = f.readlines()
        # self.stopwords = [i.strip() for i in words]
        self.stopwords = set(stopwords.words('english'))
        self.results = {}

    def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        # remove stopwords for corpus and queries
        # and calculate df and idf
        df = {}
        avg_doc_len = 0
        for doc_id, doc in corpus.items():
            title_tokens = [word for word in word_tokenize(doc['title']) if word not in self.stopwords]
            text_tokens = [word for word in word_tokenize(doc['text']) if word not in self.stopwords]
            corpus[doc_id]['title'] = ' '.join(title_tokens)
            corpus[doc_id]['text'] = ' '.join(text_tokens)

            doc_content = title_tokens + text_tokens
            avg_doc_len += len(doc_content)
            for word in set(doc_content):
                df[word] = df.get(word, 0) + 1

        for query_id, query in queries.items():
            queries[query_id] = ' '.join([word for word in word_tokenize(query) if word not in self.stopwords])

        avg_doc_len /= len(corpus)
        idf = {word: math.log(len(corpus) / df[word]) for word in df}
        logging.info("*****num of words***** %d", len(df))
        logging.info("*****avg_doc_len***** %.4f", avg_doc_len)

        if self.version == 2:
            # Preprocess the corpus for BM25
            corpus_tokens = [[word for word in (doc['title'] + ' ' + doc['text']).split()] for doc in corpus.values()]
            corpus_ids = [doc_id for doc_id in corpus.keys()]
            bm25_model = BM25Okapi(corpus_tokens)

            # get topK docs according to bm25 score
            doc_ids = list(corpus.keys())
            query_ids = list(queries.keys())
            documents = [corpus[doc_id] for doc_id in doc_ids]
            for start_idx in trange(0, len(query_ids), desc='query'):
                qid = query_ids[start_idx]
                query_tokens = queries[qid].split()
                scores = bm25_model.get_scores(query_tokens)
                top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
                self.results[qid] = {doc_ids[idx]: scores[idx] for idx in top_k_idx}

        else:
            bm25 = {}
            for doc_id, doc in corpus.items():
                bm25[doc_id] = {}
                doc_content = (doc['title'] + ' ' + doc['text']).split()
                # count bm25 for each word in doc
                for word in set(doc_content):
                    tf = doc_content.count(word)
                    if self.version == 0:
                        bm25[doc_id][word] = (tf * (self.k + 1)) / (tf + self.k * (1 - self.b + self.b * len(doc_content) / avg_doc_len)) * idf[word]
                    elif self.version == 1:
                        bm25[doc_id][word] = (tf * (self.k + 1)) / (tf + self.k * (1 - self.b + self.b * len(doc_content) / avg_doc_len)) * math.log(1 + (len(corpus) - df[word] + 0.5) / (df[word] + 0.5), math.e)

            logging.info("*****begin search*****")
            # get topK docs according to bm25 score
            doc_ids = list(corpus.keys())
            query_ids = list(queries.keys())
            documents = [corpus[doc_id] for doc_id in doc_ids]
            for start_idx in trange(0, len(query_ids), desc='query'):
                qid = query_ids[start_idx]
                # calculate scores for each doc
                scores = []
                for doc_id in doc_ids:
                    score = 0
                    for word in queries[qid].split():
                        score += bm25[doc_id].get(word, 0)
                    scores.append(score)
                # get top k results with np.argpartition
                top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
                self.results[qid] = {doc_ids[idx]: scores[idx] for idx in top_k_idx}

            logging.info("*****end search*****")

        return self.results
