import itertools
import subprocess
import os

reaction_types = (1,)
neb_steps = range(0, 9)
guidance_weights = [-1.0, 0.0, 0.5, 1.0]
combinations = itertools.product(neb_steps, reaction_types, guidance_weights)


# ckpt="/work/kumarg/flowmm/runs/trash/2025-01-11/12-10-57/null_params-rfm_cspnet-t2l3w2p1/rfmcsp-conditional-oc_20/0wau3bq1/checkpoints/epoch=579-step=457040.ckpt"
# subdir="/work/kumarg/flowmm/runs/trash/2025-01-11/12-10-57/null_params-rfm_cspnet-t2l3w2p1/rfmcsp-conditional-oc_20/0wau3bq1/checkpoints/"

# graph level
# ckpt="/work/kumarg/flowmm/runs/trash/2024-12-19/16-27-45/null_params-rfm_cspnet-kgzbid3l/rfmcsp-conditional-oc_20/fh7y6kih/checkpoints/epoch=559-step=441280.ckpt"
# subdir="/work/kumarg/flowmm/runs/trash/2024-12-19/16-27-45/null_params-rfm_cspnet-kgzbid3l/rfmcsp-conditional-oc_20/fh7y6kih/checkpoints/"

#Updated graph model
# ckpt="/work/kumarg/flowmm/runs/trash/2025-02-07/21-52-08/null_params-rfm_cspnet-ff66r3wu/rfmcsp-conditional-oc_20/riyfewg5/checkpoints/epoch=939-step=370360.ckpt"
# subdir="/work/kumarg/flowmm/runs/trash/2025-02-07/21-52-08/null_params-rfm_cspnet-ff66r3wu/rfmcsp-conditional-oc_20/riyfewg5/checkpoints/"

#RH MOdel
ckpt="/work/kumarg/flowmm/runs/trash/2025-02-16/11-02-22/null_params-rfm_cspnet-4o7z3p5a/rfmcsp-conditional-rh/7qakpe1w/checkpoints/epoch=234-step=529220.ckpt"
subdir="/work/kumarg/flowmm/runs/trash/2025-02-16/11-02-22/null_params-rfm_cspnet-4o7z3p5a/rfmcsp-conditional-rh/7qakpe1w/checkpoints/"


# base model
# ckpt="/work/kumarg/flowmm/runs/trash/2025-01-11/12-10-57/null_params-rfm_cspnet-t2l3w2p1/rfmcsp-conditional-oc_20/0wau3bq1/checkpoints/epoch=579-step=457040.ckpt"
# subdir="/work/kumarg/flowmm/runs/trash/2025-01-11/12-10-57/null_params-rfm_cspnet-t2l3w2p1/rfmcsp-conditional-oc_20/0wau3bq1/checkpoints/"

# ckpt="/work/kumarg/flowmm/runs/trash/2024-12-19/16-27-45/null_params-rfm_cspnet-kgzbid3l/rfmcsp-conditional-oc_20/fh7y6kih/checkpoints/epoch=559-step=441280.ckpt"
# subdir="/work/kumarg/flowmm/runs/trash/2024-12-19/16-27-45/null_params-rfm_cspnet-kgzbid3l/rfmcsp-conditional-oc_20/fh7y6kih/checkpoints/"


for neb_step, reaction_type, guidance_weight in combinations:
    slope = 1
    filename = f"reconstruct_neb_{neb_step}_reaction_{reaction_type}_w_{guidance_weight}"
    subprocess.run([
        "python", "scripts_model/evaluate.py", "reconstruct", ckpt, "--subdir", subdir,
        "--inference_anneal_slope", str(slope),
        "--reaction_type", str(reaction_type),
        "--neb_step", str(neb_step),
        "--guidance_weight", str(guidance_weight),
        "--stage", "test"
    ], check=True)
    subprocess.run([
        "python", "scripts_model/evaluate.py", "consolidate", ckpt, "--subdir", subdir , "--filename", filename,  "--task_to_save" ,"reconstruct"
    ], check=True)
 
