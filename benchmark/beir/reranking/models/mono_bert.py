# Majority of the code has been copied from PyGaggle MonoT5 implementation
# https://github.com/castorini/pygaggle/blob/master/pygaggle/rerank/transformer.py

from transformers import (AutoTokenizer,
                          PreTrainedTokenizer,
                          AutoModelForSequenceClassification)
from typing import List, Union, Tuple, Mapping, Optional
from dataclasses import dataclass
from tqdm.autonotebook import trange
import torch

class MonoBERT:
    def __init__(self,
                 model_path: str,
                 tokenizer: PreTrainedTokenizer = None,
                 use_amp = False):
        self.model = self.get_model(model_path)
        self.tokenizer = tokenizer or self.get_tokenizer(model_path)
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/monobert-large-msmarco',
                  *args, device: str = None, **kwargs) -> AutoModelForSequenceClassification:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                  *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 'bert-large-uncased',
                      *args, **kwargs) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs)
    
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int = 32, **kwargs) -> List[float]:
        
        sentence_dict, queries, scores = {}, [], []

        # T5 model requires a batch of single query and top-k documents
        for (query, doc_text) in sentences:
            if query not in sentence_dict:
                sentence_dict[query] = []
                queries.append(query) # Preserves order of queries
            sentence_dict[query].append(doc_text) 
        
        for start_idx in trange(0, len(queries), 1): # Take one query at a time
            query, docs = queries[start_idx], sentence_dict[queries[start_idx]]
            for batch_idx in range(0, len(docs), batch_size):
                doc_texts = docs[batch_idx:batch_idx + batch_size]
                batch_input = [(query, doc) for doc in doc_texts]
                ret = self.tokenizer.batch_encode_plus(batch_input,
                                             docs,
                                             max_length=512,
                                             truncation='only_second',
                                             return_token_type_ids=True,
                                             padding=True,
                                             return_tensors='pt')
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    input_ids = ret['input_ids'].to(self.device)
                    tt_ids = ret['token_type_ids'].to(self.device)
                    output, = self.model(input_ids, token_type_ids=tt_ids, return_dict=False)
                    if output.size(1) > 1:
                        score = torch.nn.functional.log_softmax(
                            output, 1)[:, -1].tolist()
                    else:
                        score = output.tolist()
                    scores.extend(score)
        assert len(scores) == len(sentences) # Sanity check, should be equal
        return scores