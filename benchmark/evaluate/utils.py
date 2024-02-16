from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

def sole_evaluate(results: Dict[str, Dict[str, float]], target_source: str):
    # input: 
    # reuslts: {'qid': {'doc-id-target_source': score}}
    # target_source: "human" or "llm-0.2" or ......
    # output: 
    # new_results: {'qid': {'doc-id': score}}
    new_results = {}
    for qid in results:
        new_results[qid] = {}
        for docid in results[qid]:
            if docid.endswith(f"-{target_source}"):
                new_results[qid][docid[:-len(target_source)-1]] = results[qid][docid]
    return new_results


def multi_mixed_target_evaluate(qrels: Dict[str, Dict[str, int]], target_source_list: List[str]):
    # input: 
    # qrels: {'qid': {'doc-id': score}}
    # target_source_list: ["human","llm-0.2",......]
    # output: 
    # new_qrels: {'qid': {'doc-id-target_source': score}}
    new_qrels = {}
    for qid in qrels:
        new_qrels[qid] = {}
        for docid in qrels[qid]:
            for target_source in target_source_list:
                new_qrels[qid][docid + "-" + target_source] = qrels[qid][docid]
    return new_qrels


def evaluate_human_corpus_rate(results: Dict[str, Dict[str, float]], k_values: List[int] = [1,3,5,10,100,1000]):
    rates = {}
    for k in k_values:
        rates[f"rate@{k}"] = 0.0
    for query_id, doc_scores in results.items():
        sorted_doc_scores = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        for k in k_values:
            topk_doc_scores = sorted_doc_scores[:k]
            topk_docids = [docid for docid, score in topk_doc_scores]
            rate = sum([1 for docid in topk_docids if docid.endswith("human")]) / k
            rates[f"rate@{k}"] += rate
    for k in k_values:
        rates[f"rate@{k}"] = round(rates[f"rate@{k}"]/len(results), 5)
    
    logger.info("\n")
    for k in k_values:
        logger.info("Rate@{}: {:.4f}".format(k, rates[f"rate@{k}"]))