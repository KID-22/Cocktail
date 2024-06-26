GPU=0
batch_size=128
for dataset in "msmarco" "dl19" "dl20" "trec-covid" "nfcorpus" "nq" "hotpotqa" "fiqa" "webis-touche2020" "cqadupstack" "dbpedia-entity" "scidocs" "fever" "climate-fever" "scifact" "nq-utd"
do
    for model in "crossencoder" "monot5"
    do
        mkdir -p ./log/${dataset}/rerank/${model}/

        CUDA_VISIBLE_DEVICES=$GPU python evaluate/rerank/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100\
        --top_k=100 \
        --corpus_list human \
        --save_results=1 \
        --dataset=${dataset} \
        --batch_size=${batch_size} \
        > ./log/${dataset}/rerank/${model}/human.log 2>&1

        CUDA_VISIBLE_DEVICES=$GPU python evaluate/rerank/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100\
        --top_k=100 \
        --corpus_list llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        --batch_size=${batch_size} \
        > ./log/${dataset}/rerank/${model}/llama2.log 2>&1

        CUDA_VISIBLE_DEVICES=$GPU python evaluate/rerank/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100\
        --top_k=100 \
        --corpus_list human llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        --batch_size=${batch_size} \
        > ./log/${dataset}/rerank/${model}/human_llama2.log 2>&1

        CUDA_VISIBLE_DEVICES=$GPU python evaluate/rerank/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100\
        --top_k=100 \
        --corpus_list human llama-2-7b-chat-tmp0.2 \
        --target_list human llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        --batch_size=${batch_size} \
        > ./log/${dataset}/rerank/${model}/human+llama2.log 2>&1
    done
done
