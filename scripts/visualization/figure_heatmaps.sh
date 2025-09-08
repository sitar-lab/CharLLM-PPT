echo "Generating heatmaps for Figure 5 and 17..."
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_heatmaps.py
python ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/figure_heatmaps_fsdp.py

echo "Saved heatmaps to ${CHARLLM_ROOT}/CharLLM-PPT/figures/heatmaps and ${CHARLLM_ROOT}/CharLLM-PPT/figures/heatmaps-fsdp."