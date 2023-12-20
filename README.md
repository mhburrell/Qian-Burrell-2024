Code for TD and ANCCR simulations in Qian et al., 2023

## Folders

photometry_data_analysis: contains the code used to analyse experimental data presented

matlab_simulation_code: contains code for generating simulated experiments, running the belief state model (modified from: https://github.com/cstarkweather/Belief-State-TD-Model/, Starkweather et al., 2018) and ANCCR simulations (modified from: https://github.com/namboodirilab/ANCCR)

python_simulation_code: code for simulating cue-context and CSC models, training and generating data from the value RNNs

value_rnn (install with pip): Forked from https://github.com/mobeets/valuernn, contains frozen version of valuernn as used

r_analysis_code: Code for finding common state space of RNNs



## Requirements
- Python 3.9

conda create --name qian-2023 python=3.9 pytorch matplotlib numpy scipy scikit-learn pandas tqdm

conda activate qian-2023

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pyarrow

