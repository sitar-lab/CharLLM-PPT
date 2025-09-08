echo "Parsing trace files for Figure 15..."
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_15_parse.py --model=gpt3-175b
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_15_parse_fsdp.py --model=gpt3-175b

echo "Generating subfigures for Figure 15..."
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_15_1.py
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_15_2.py
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_15_1_fsdp.py
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_15_2_fsdp.py

echo "Concatenating subfigures into Figure 15..."
python concat_pdfs.py ${CHARLLM_ROOT}/CharLLM-PPT/figures/microbatch-sweep-h200-fsdp-mbs1/no_bottom/gpt3-175b_rank_comparison_no_bottom-fsdp.pdf ${CHARLLM_ROOT}/CharLLM-PPT/figures/microbatch-sweep-h200-mbs1/no_bottom/gpt3-175b_rank_comparison_no_bottom.pdf ${CHARLLM_ROOT}/CharLLM-PPT/figures/figure_15_1.pdf
python concat_pdfs.py ${CHARLLM_ROOT}/CharLLM-PPT/figures/microbatch-sweep-h200-fsdp-mbs4/no_legend/gpt3-175b_rank_comparison_no_legend-fsdp.pdf ${CHARLLM_ROOT}/CharLLM-PPT/figures/microbatch-sweep-h200-mbs4/no_legend/gpt3-175b_rank_comparison_no_legend.pdf ${CHARLLM_ROOT}/CharLLM-PPT/figures/figure_15_2.pdf
python stack_pdfs.py ${CHARLLM_ROOT}/CharLLM-PPT/figures/figure_15_1.pdf ${CHARLLM_ROOT}/CharLLM-PPT/figures/figure_15_2.pdf ${CHARLLM_ROOT}/CharLLM-PPT/figures/figure_15.pdf

echo "Saved Figure 3 to ${CHARLLM_ROOT}/CharLLM-PPT/figures/figure_15.pdf"