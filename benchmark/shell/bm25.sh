for dataset in "msmarco" "dl19" "dl20" "trec-covid" "nfcorpus" "nq" "hotpotqa" "fiqa" "webis-touche2020" "cqadupstack" "dbpedia-entity" "scidocs" "fever" "climate-fever" "nq-utd"
do
    for model in "bm25"
    do
        mkdir -p ./log/${dataset}/${model}/

        python evaluate/retrieval/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100 1000\
        --corpus_list human \
        --save_results=1 \
        --dataset=${dataset} \
        > ./log/${dataset}/${model}/human.log 2>&1

        python evaluate/retrieval/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100 1000\
        --corpus_list llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        > ./log/${dataset}/${model}/llama2.log 2>&1

        python evaluate/retrieval/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100 1000\
        --corpus_list human llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        > ./log/${dataset}/${model}/human_llama2.log 2>&1

        python evaluate/retrieval/${model}.py \
        --k_values 1 2 3 4 5 6 7 8 9 10 100 1000\
        --corpus_list human llama-2-7b-chat-tmp0.2 \
        --target_list human llama-2-7b-chat-tmp0.2 \
        --save_results=1 \
        --dataset=${dataset} \
        > ./log/${dataset}/${model}/human+llama2.log 2>&1
    done
done
