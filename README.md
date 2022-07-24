# TransforMatcher: Match-to-Match Attention for Semantic Correspondence
This is the official pytorch implementation of the paper "TransforMatcher: Match-to-Match Attention for Semantic Correspondence" by Seungwook Kim, Juhong Min and Minsu Cho. Implemented on Python 3.7 and PyTorch 1.7.0.

![](http://cvlab.postech.ac.kr/research/TransforMatcher/images/figs/transformatcher_overview.PNG)

Check out our project [[website](http://cvlab.postech.ac.kr/research/TransforMatcher/)] and the paper on [[arXiv](https://arxiv.org/abs/2205.11634)]!

## Requirements

Conda environment settings:
```bash
conda create -n tfm python=3.7
conda activate tfm

conda install pytorch=1.7.0 torchvision cudatoolkit=10.2 -c pytorch
conda install -c anaconda requests
conda install -c anaconda scipy
conda install -c anaconda pandas
conda install -c conda-forge einops
conda install -c conda-forge albumentations
pip install tensorboardX
pip install rotary-embedding-torch
pip install -U albumentations
```

## Training	

```bash
python train.py --benchmark {spair, pfpascal}
```

## Testing
Trained models will be made available soon.
```bash
python test.py --benchmark {spair, pfpascal, pfwillow} 
               --load 'path_to_trained_model'
```

## BibTeX
If you find our code or paper to be useful for your research, please consider citing our work:
```
@inproceedings{swkim2022tfmatcher,
  title={TransforMatcher:Match-to-Match Attention for Semantic Correspondence},
  author={Kim, Seungwook and Min, Juhong and Cho, Minsu },
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

## Contact

Seungwook Kim (wookiekim@postech.ac.kr)

Feel free to reach out to me! 

