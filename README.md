# evaluate attention benchmark

## envrioment setup
```shell
pip install -r requirements.txt
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
git checkout b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7 # 2.7.2 release
cd hopper
python setup.py install

git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
pip install .
```