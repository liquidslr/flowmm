ckpt="/work/kumarg/flowmm/runs/trash/2024-12-19/16-27-45/null_params-rfm_cspnet-kgzbid3l/rfmcsp-conditional-oc_20/fh7y6kih/checkpoints/epoch=559-step=441280.ckpt"
subdir="/work/kumarg/flowmm/runs/trash/2024-12-19/16-27-45/null_params-rfm_cspnet-kgzbid3l/rfmcsp-conditional-oc_20/fh7y6kih/checkpoints/"

slope=1
neb_step=1
reaction_type=0

python scripts_model/evaluate.py reconstruct ${ckpt} --subdir ${subdir} --inference_anneal_slope ${slope} --stage test && \
python scripts_model/evaluate.py consolidate ${ckpt} --subdir ${subdir} && \
python scripts_model/evaluate.py old_eval_metrics ${ckpt} --subdir ${subdir} --stage test && \
python scripts_model/evaluate.py lattice_metrics ${ckpt} --subdir ${subdir} --stage test
