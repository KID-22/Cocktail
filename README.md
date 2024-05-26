<h1 align="center">
<img style="vertical-align:middle" width="180" height="180" src="assert/cocktail.png" />
</h1>


# Cocktail: A Comprehensive Information Retrieval Benchmark with LLM-Generated Documents Integration

## Introduction
Cocktail, a comprehensive benchmark designed to evaluate Information Retrieval (IR) models amidst the evolving landscape of AI-generated content (AIGC). In an era dominated by Large Language Models (LLMs), the traditional IR corpus, previously solely composed of human-written texts, has expanded to include a significant proportion of LLM-generated content. Cocktail emerges as a valuable resource to response to this transformation, aiming to provide a robust framework for assessing the performance and bias of IR models in handling mixed corpora in this LLM era.


## Features
+ **Comprehensive Dataset Collection**: Cocktail comprises 15 existing IR datasets in a standard format, diversified across a range of text retrieval tasks and domains, each enriched with an LLM-generated corpus using Llama2.

+ **Up-to-Date Evaluation Dataset**: Introducing Natural Question Up-To-Date (NQ-UTD), a dataset featuring queries derived from the latest events, specifically designed to test the responsiveness of LLM-based IR models to new information not included in their pre-training data.

+ **Easy-to-use Evaluation Tool**:  Cocktail includes a user-friendly evaluation tool, simplifying the process of assessing various IR models on the benchmarked dataset. This tool is designed with adaptability, allowing for seamless integration of new models and datasets, thereby enabling researchers and developers to efficiently evaluate the performance and bias of their IR systems.


## File Structure
```shell
.
├── dataset  # * dataset path
│   ├── climate-fever
│   ├── cqadupstack
│   ├── ...
│   ├── trec-covid
│   └── webis-touche2020 
└── benchmark  # * evaluation benchmark
    ├── beir  # * requirements codes from beir
    ├── evaluate  # * codes for evaluation
    │   ├── rerank # * code for re-rankers
    │   ├── retrieval # * code for retreiever
    │   └── utils # * codes for different evaluation setting
    └── shell  # * script for quick evaluation
```

## Quick Start

We provide the detail scripts for all the benchmarked models in the folder ``benchmark/shell``. Using neural retrieval models as an example, you can quickly and easily reproduce our results using the following scripts:
```shell
GPU=0
batch_size=128
for dataset in "msmarco" "dl19" "dl20" "trec-covid" "nfcorpus" "nq" "hotpotqa" "fiqa" "webis-touche2020" "cqadupstack" "dbpedia-entity" "scidocs" "fever" "climate-fever" "nq-utd"
do
    for model in "bert" "roberta" "tasb" "contriever" "dragon" "cocondenser" "ance" "retromae"
    do
        mkdir -p ./log/${dataset}/${model}/

        # sole human-written corpus evaluation
        CUDA_VISIBLE_DEVICES=$GPU python evaluate/retrieval/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100\
        --corpus_list human \
        --save_results=1 \
        --dataset=${dataset} \
        --batch_size=${batch_size} \
        > ./log/${dataset}/${model}/human.log 2>&1

        # sole llm-generated corpus evaluation
        CUDA_VISIBLE_DEVICES=$GPU python evaluate/retrieval/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100\
        --corpus_list llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        --batch_size=${batch_size} \
        > ./log/${dataset}/${model}/llama2.log 2>&1

        # mix evaluation
        CUDA_VISIBLE_DEVICES=$GPU python evaluate/retrieval/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100\
        --corpus_list human llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        --batch_size=${batch_size} \
        > ./log/${dataset}/${model}/human_llama2.log 2>&1

        # mix evaluation
        CUDA_VISIBLE_DEVICES=$GPU python evaluate/retrieval/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100\
        --corpus_list human llama-2-7b-chat-tmp0.2 \
        --target_list human llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        --batch_size=${batch_size} \
        > ./log/${dataset}/${model}/human+llama2.log 2>&1
    done
done
```

Our evaluation tool is designed to support a variety of customized assessments, including the integration of corpora from different sources and the computation of metrics for specific target corpora. For personalized customization options, please refer to the code in our ``evaluate`` folder.

## Available Datasets
All the 16 benchmarked datasets in Cocktail are listed in the following table and are available [here](https://huggingface.co/IR-Cocktail) at HuggingFace. 

| Dataset       |   Raw Website  |  Cocktail Download | Cocktail-Name      | md5 for Processed Data                              | Domain      | Relevancy | # Test Query | # Corpus |
| ------------- | ------------------------------------------------------------ | ------------------ | ---------------------------------- | ----------- | --------- | ------------ | -------- |-------- |
| MS MARCO      | [Homepage](https://microsoft.github.io/msmarco/) |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/msmarco)   | `msmarco`          | `985926f3e906fadf0dc6249f23ed850f` | Misc.       | Binary    | 6,979        | 542,203  |
| DL19          | [Homepage](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019) | [Homepage](https://huggingface.co/datasets/IR-Cocktail/dl19)  |   `dl19`             | `d652af47ec0e844af43109c0acf50b74` | Misc.       | Binary    | 43           | 542,203  |
| DL20          | [Homepage](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020) | [Homepage](https://huggingface.co/datasets/IR-Cocktail/dl20)  | `dl20`             | `3afc48141dce3405ede2b6b937c65036` | Misc.       | Binary    | 54           | 542,203  |
| TREC-COVID    | [Homepage](https://ir.nist.gov/covidSubmit/index.html)    |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/trec-covid)  | `trec-covid`       | `1e1e2264b623d9cb7cb50df8141bd535` | Bio-Medical | 3-level   | 50           | 128,585  |
| NFCorpus      | [Homepage](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/nfcorpus) | `nfcorpus`         | `695327760647984c5014d64b2fee8de0` | Bio-Medical | 3-level   | 323          | 3,633    |
| NQ            | [Homepage](https://ai.google.com/research/NaturalQuestions) |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/nq)  | `nq`               | `a10bfe33efdec54aafcc974ac989c338` | Wikipedia   | Binary    | 3,446        | 104,194  |
| HotpotQA      | [Homepage](https://hotpotqa.github.io/)                   |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/hotpotqa)  | `hotpotqa`         | `74467760fff8bf8fbdadd5094bf9dd7b` | Wikipedia   | Binary    | 7,405        | 111,107  |
| FiQA-2018     | [Homepage](https://sites.google.com/view/fiqa/)           |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/fiqa)  | `fiqa`             | `4e1e688539b0622630fb6e65d39d26fa` | Finance     | Binary    | 648          | 57,450   |
| Touché-2020   | [Homepage](https://webis.de/events/touche-20/shared-task-1.html) |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/webis-touche2020)  | `webis-touche2020` | `d58ec465ccd567d8f75edb419b0faaed` | Misc.       | 3-level   | 49           | 101,922  |
| CQADupStack   | [Homepage](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/dcqadupstackl19)  | `cqadupstack`      | `d48d963bc72689c765f381f04fc26f8b` | StackEx.    | Binary    | 1,563        | 39,962   |
| DBPedia       | [Homepage](https://github.com/iai-group/DBpedia-Entity/)     |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/dbpedia-entity)  | `dbpedia-entity`   | `43292f4f1a1927e2e323a4a7fa165fc1` | Wikipedia   | 3-level   | 400          | 145,037  |
| SCIDOCS       | [Homepage](https://allenai.org/data/scidocs)                 |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/scidocs)  | `scidocs`          | `4058c0915594ab34e9b2b67f885c595f` | Scientific  | Binary    | 1,000        | 25,259   |
| FEVER         | [Homepage](http://fever.ai/)                                 |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/fever)  | `fever`            | `98b631887d8c38772463e9633c477c69` | Wikipedia   | Binary    | 6,666        | 114,529  |
| Climate-FEVER | [Homepage](http://climatefever.ai/)                          |   [Homepage](https://huggingface.co/datasets/IR-Cocktail/climate-fever) | `climate-fever`    | `5734d6ac34f24f5da496b27e04ff991a` | Wikipedia   | Binary    | 1,535        | 101,339  |
| SciFact       | [Homepage](https://github.com/allenai/scifact)               |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/scifact)  | `scifact`          | `b5b8e24ccad98c9ca959061af14bf833` | Scientific  | Binary    | 300          | 5,183    |
| NQ-UTD        | [Homepage](https://anonymous.4open.science/r/Cocktail-BA4B/) |  [Homepage](https://huggingface.co/datasets/IR-Cocktail/nq-utd)  | `nq-utd`           | `2e12e66393829cd4be715718f99d2436` | Misc.       | 3-level   | 80           | 800      |


To verify the downloaded files, you can use the command to generate an MD5 hash using Terminal: ``md5sum filename.zip``.

## Checkpoints
We also provide some checkpoints trained with ``train_msmarco_v3.py`` in [BEIR]([https://github.com/beir-cellar/beir](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3.py)). Please see the following table:

|                   Model                   |         PLM        | Pooling Strategy |                                    Download                                 |
|:-----------------------------------------:|:------------------:|:---------------:|:----------------------------------------------------------------------------:|
| bert-base-uncased-mean-v3-msmarco         | bert-base-uncased  |       mean      | [Link](https://huggingface.co/IR-Cocktail/bert-base-uncased-mean-v3-msmarco)         |
| bert-base-uncased-cls-v3-msmarco          | bert-base-uncased  |       cls       | [Link](https://huggingface.co/IR-Cocktail/bert-base-uncased-cls-v3-msmarco)          |
| bert-base-uncased-last-v3-msmarco         | bert-base-uncased  |       last      | [Link](https://huggingface.co/IR-Cocktail/bert-base-uncased-last-v3-msmarco)         |
| bert-base-uncased-max-v3-msmarco          | bert-base-uncased  |       max       | [Link](https://huggingface.co/IR-Cocktail/bert-base-uncased-max-v3-msmarco)          |
| bert-base-uncased-weightedmean-v3-msmarco | bert-base-uncased  |   weighted-mean  | [Link](https://huggingface.co/IR-Cocktail/bert-base-uncased-weightedmean-v3-msmarco) |
| bert-mini-mean-v3-msmarco                 | bert-mini          |       mean      | [Link](https://huggingface.co/IR-Cocktail/bert-mini-mean-v3-msmarco)                 |
| bert-small-mean-v3-msmarco                | bert-small         |       mean      | [Link](https://huggingface.co/IR-Cocktail/bert-small-mean-v3-msmarco)                |
| bert-large-uncased-mean-v3-msmarco        | bert-large-uncased |       mean      | [Link](https://huggingface.co/IR-Cocktail/bert-large-uncased-mean-v3-msmarco)        |
| roberta-base-mean-v3-msmarco              | roberta-base       |       mean      | [Link](https://huggingface.co/IR-Cocktail/roberta-base-mean-v3-msmarco)              |
| robreta-base-cls-v3-msmarco               | roberta-base       |       cls       | [Link](https://huggingface.co/IR-Cocktail/roberta-base-cls-v3-msmarco)               |
| robreta-base-last-v3-msmarco              | roberta-base       |       last      | [Link](https://huggingface.co/IR-Cocktail/roberta-base-last-v3-msmarco)              |
| robreta-base-max-v3-msmarco               | roberta-base       |       max       | [Link](https://huggingface.co/IR-Cocktail/roberta-base-max-v3-msmarco)               |
| robreta-base-weightedmean-v3-msmarco      | roberta-base       |   weighted-mean  | [Link](https://huggingface.co/IR-Cocktail/roberta-base-weightedmean-v3-msmarco)      |

## Reference
The Cocktail benchmark is built based on the following project:
- [BEIR](https://github.com/beir-cellar/beir)
- [Sentence Transformers](https://huggingface.co/sentence-transformers)


## Citation
If you find our benchmark or work useful for your research, please cite our work.
```
@article{dai2024cocktail,
  title={Cocktail: A Comprehensive Information Retrieval Benchmark with LLM-Generated Documents Integration},
  author={Dai, Sunhao and Liu, Weihao and Zhou, Yuqi and Pang, Liang and Ruan, Rongju and Wang, Gang and Dong, Zhenhua and Xu, Jun and Wen, Ji-Rong},
  journal={Findings of the Association for Computational Linguistics: ACL 2024},
  year={2024}
}

@article{dai2024neural,
  title={Neural Retrievers are Biased Towards LLM-Generated Content},
  author={Dai, Sunhao and Zhou, Yuqi and Pang, Liang and Liu, Weihao and Hu, Xiaolin and Liu, Yong and Zhang, Xiao and Wang, Gang and Xu, Jun},
  journal={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```


## License
The proposed NQ-UTD dataset use [MIT license](LICENSE). All data and code in this project can only be used for academic purposes.
