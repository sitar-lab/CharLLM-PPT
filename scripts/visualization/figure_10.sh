python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_parse.py --model=gpt3-175b
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_parse.py --model=llama3-70b
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_parse.py --model=mixtral-8x7b
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_parse.py --model=mixtral-8x22b

python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_parse_fsdp.py --model=gpt3-175b
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_parse_fsdp.py --model=llama3-70b
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_parse_fsdp.py --model=mixtral-8x7b
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_parse_fsdp.py --model=mixtral-8x22b

python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_1.py
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_10_2.py

python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/concat_pdfs.py ${CHARLLM_ROOT}/CharLLM-PPT/figures/figures-2cols-fsdp/no_legend/model_parallel_kernel_times_no_legend-fsdp.pdf ${CHARLLM_ROOT}/CharLLM-PPT/figures/figures-2cols/full/model_parallel_kernel_times_full.pdf ${CHARLLM_ROOT}/CharLLM-PPT/figures/figure_10.pdf
