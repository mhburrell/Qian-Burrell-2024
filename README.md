Code for TD and ANCCR simulations in Qian et al., 2023

## Requirements
- Python 3.9

conda create --name qian-2023 python=3.9 pytorch matplotlib numpy scipy scikit-learn pandas tqdm
conda activate qian-2023
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyarrow
