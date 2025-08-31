cd ~
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate


conda create -n verl python==3.10
conda activate verl

cd verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install vllm==0.6.3  # this must be installed to get weight update working
pip3 install flash-attn --no-build-isolation
#git clone git@github.com:SynthLabsAI/verl.git 
#cd verl
pip3 install -e .
