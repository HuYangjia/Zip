conda create -n zip python=3.10 -y
conda activate zip
pip install torch torchvision torchaudio transformers datasets

# 每次开始实验前都先确认在 zip 环境
python -c "import os,torch; print('env=', os.environ.get('CONDA_DEFAULT_ENV')); print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available())"