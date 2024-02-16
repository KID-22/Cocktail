import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from benchmark.beir import util, LoggingHandler
from benchmark.beir.datasets.data_loader import GenericDataLoader
from benchmark.beir.retrieval.evaluation import EvaluateRetrieval
from benchmark.beir.retrieval.search.lexical import BM25Search as BM25
from benchmark.evaluate.utils import sole_evaluate, multi_mixed_target_evaluate, evaluate_human_corpus_rate

import json
import logging
import pathlib
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="test dataset", default="msmarco")
parser.add_argument("--save_results", type=int, help="save results or not", default=0)
parser.add_argument("--corpus_list", nargs='+', type=str, help="test corpus list", default=["human", "llama-2-7b-chat-tmp0.2"])
parser.add_argument("--target_list", nargs='+', type=str, help="target source list", default="")
parser.add_argument("--k_values", nargs='+', type=int, help="k values", default=[1,3,5,10])
parser.add_argument('--initialize', action="store_true", default=True)
args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### print args
logging.info("dataset: {}".format(args.dataset))
logging.info("save_results: {}".format(args.save_results))
logging.info("corpus_list: {}".format(args.corpus_list))
logging.info("target_list: {}".format(args.target_list))
logging.info("k_values: {}".format(args.k_values))
logging.info("initialize: {}".format(args.initialize))

#### Download scifact.zip dataset and unzip the dataset
dataset_path = f"../dataset/{args.dataset}/"

full_corpus = {}
for corpus in args.corpus_list:
    if args.dataset[:7] == "msmarco":
        tmp_corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path, corpus_file=f"corpus/{corpus}.jsonl", corpus_source=f"-{corpus}").load(split="dev")
    else:
        tmp_corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path, corpus_file=f"corpus/{corpus}.jsonl", corpus_source=f"-{corpus}").load(split="test")  
    full_corpus.update(tmp_corpus)

#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

hostname = "localhost:9200/" #localhost

#### Intialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
index_name = f"{args.dataset}_{'-'.join(args.corpus_list)}"

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
# SciFact is a relatively small dataset! (limit shards to 1)
# number_of_shards = 1
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

# number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, initialize=args.initialize)

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model, k_values=args.k_values) 

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(full_corpus, queries) # only top-k docs are returned

### Check if query_id is in results i.e. remove it from docs incase if it appears ####
### Quite Important for ArguAna and Quora ####
num = 0
for query_id in results:
    for target_source in args.corpus_list:
        if query_id + "-" + target_source in results[query_id]:
            results[query_id].pop(query_id + "-" + target_source, None)
            num += 1
logging.info("num of removed doc: {}".format(num))


# save results
if args.save_results:
    result_path = f"./result/{args.dataset}/bm25/"
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
    for query_id in results:
        results[query_id] = dict(sorted(results[query_id].items(), key=lambda item: item[1], reverse=True))
    with open(f"{result_path}/{'-'.join(args.corpus_list)}.json", 'w') as f:
        json.dump(results, f, indent=4)


# mixed evaluation for each target source in corpus list
if args.target_list == "":
    logging.info("="*60)
    logging.info("******mixed evaluation for each target source******")
    for target_source in args.corpus_list:
        logging.info("*"*40)
        logging.info("target source: {}".format(target_source))
        tmp_qrels = multi_mixed_target_evaluate(qrels, [target_source])
        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(tmp_qrels, results, retriever.k_values)
else:
    # mixed evaluation for the specific target source list
    logging.info("******mixed evaluation for specific target source list******")
    logging.info("target source list: {}".format(args.target_list))
    tmp_qrels = multi_mixed_target_evaluate(qrels, args.target_list)
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(tmp_qrels, results, retriever.k_values)


# human corpus rate evaluation
logging.info("="*60)
logging.info("******human corpus rate evaluation******")
evaluate_human_corpus_rate(results, args.k_values)
