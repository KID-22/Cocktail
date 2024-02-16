import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from benchmark.beir import util, LoggingHandler
from benchmark.beir.datasets.data_loader import GenericDataLoader
from benchmark.beir.retrieval.evaluation import EvaluateRetrieval
from benchmark.beir.retrieval.search.lexical import BM25 as BM25
from benchmark.beir.reranking.models import CrossEncoder
from benchmark.beir.reranking import Rerank
from benchmark.evaluate.utils import sole_evaluate, multi_mixed_target_evaluate, evaluate_human_corpus_rate

import json
import logging
import pathlib
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="retrieval model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
parser.add_argument("--dataset", type=str, help="test dataset", default="scifact")
parser.add_argument("--save_results", type=int, help="save results or not", default=0)
parser.add_argument("--corpus_list", nargs='+', type=str, help="test corpus list", default=["human", "llama-2-7b-chat-tmp0.2"])
parser.add_argument("--target_list", nargs='+', type=str, help="target source list", default="")
parser.add_argument("--k_values", nargs='+', type=int, help="k values", default=[1,3,5,10])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--top_k', type=int, help="top k first stage retrieval", default=100)
parser.add_argument('--retrieval_model', type=str, help="first stage retrieval model", default="bm25")
args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### print args
logging.info("model: {}".format(args.model))
logging.info("dataset: {}".format(args.dataset))
logging.info("save_results: {}".format(args.save_results))
logging.info("corpus_list: {}".format(args.corpus_list))
logging.info("target_list: {}".format(args.target_list))
logging.info("k_values: {}".format(args.k_values))
logging.info("batch_size: {}".format(args.batch_size))
logging.info("top_k: {}".format(args.top_k))
logging.info("first stage retrieval_model: {}".format(args.retrieval_model))

#### Download scifact.zip dataset and unzip the dataset
dataset_path = f"../dataset/{args.dataset}/"

full_corpus = {}
for corpus in args.corpus_list:
    if args.dataset[:7] == "msmarco":
        tmp_corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path, corpus_file=f"corpus/{corpus}.jsonl", corpus_source=f"-{corpus}").load(split="dev")
    else:
        tmp_corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path, corpus_file=f"corpus/{corpus}.jsonl", corpus_source=f"-{corpus}").load(split="test")  
    full_corpus.update(tmp_corpus)


###############################################
#### (1) RETRIEVE Top-1000 docs using BM25 #### 
###############################################
first_stage_scores_path = f"./result/{args.dataset}/{args.retrieval_model}/"
with open(os.path.join(first_stage_scores_path, f"{'-'.join(args.corpus_list)}.json"), 'r') as f:
    results = json.load(f)

##############################################
#### (2) RERANK Top-100 docs using MonoT5 ####
##############################################

# Document Ranking with a Pretrained Sequence-to-Sequence Model 
# https://aclanthology.org/2020.findings-emnlp.63/

#### Check below for reference parameters for different MonoT5 models 
#### Two tokens: token_false, token_true


cross_encoder_model = CrossEncoder(f"{args.model}")
reranker = Rerank(cross_encoder_model, batch_size=args.batch_size)

# Rerank top-100 results using the reranker provided
rerank_results = reranker.rerank(full_corpus, queries, results, top_k=args.top_k)

### Check if query_id is in results i.e. remove it from docs incase if it appears ####
### Quite Important for ArguAna and Quora ####
num = 0
for query_id in rerank_results:
    for target_source in args.corpus_list:
        if query_id + "-" + target_source in rerank_results[query_id]:
            rerank_results[query_id].pop(query_id, None)
            num += 1
logging.info("num of removed doc: {}".format(num))


# save rerank_results
if args.save_results:
    result_path = f"./result/{args.dataset}/{args.model}_1_{args.retrieval_model}/"
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
    for query_id in rerank_results:
        rerank_results[query_id] = dict(sorted(rerank_results[query_id].items(), key=lambda item: item[1], reverse=True))
    with open(f"{result_path}/{'-'.join(args.corpus_list)}.json", 'w') as f:
        json.dump(rerank_results, f, indent=4)


#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K
# mixed evaluation for each target source in corpus list
if args.target_list == "":
    retriever = EvaluateRetrieval(k_values=args.k_values)
    logging.info("="*60)
    logging.info("******mixed evaluation for each target source******")
    for target_source in args.corpus_list:
        logging.info("*"*40)
        logging.info("target source: {}".format(target_source))
        tmp_qrels = multi_mixed_target_evaluate(qrels, [target_source])
        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(tmp_qrels, rerank_results, retriever.k_values)
else:
    # mixed evaluation for the specific target source list
    logging.info("\n******mixed evaluation for specific target source list******")
    logging.info("target source list: {}".format(args.target_list))
    tmp_qrels = multi_mixed_target_evaluate(qrels, args.target_list)
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(tmp_qrels, rerank_results, retriever.k_values)


# human corpus rate evaluation
logging.info("="*60)
logging.info("******human corpus rate evaluation******")
evaluate_human_corpus_rate(rerank_results, args.k_values)