# PRGS: Patch-to-Region Graph Search for Visual Place Recognition



## Setup
Install the python environment using 

```
pip3 install -r requirements.txt
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Getting Started

This repo follows the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark). You can refer to it ([VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader)) to prepare datasets.

The dataset should be organized in a directory tree as such:

```

── datasets
    └── pitts30k
        └── images
            ├── train
            │   ├── database
            │   └── queries
            ├── val
            │   ├── database
            │   └── queries
            └── test
                ├── database
                └── queries
```

## Train in MSLS
For using MSLS dataset, you should get directory mapillary_sls from [msls](https://github.com/mapillary/mapillary_sls.git) to use MSLS datasets. And then
you should get DeiT backbone from [msls_v2_deits.pth](https://drive.google.com/file/d/1XBNjbbNUrp6NIv6REHrMLdHFw5joustx/view?usp=sharing), save to resume directory.
Lastly, you should perform pre_compute_mining to get hard negative list. 
```commandline
python pre_compute_mining.py
```

```commandline
bash train_reranking.sh
```
## Test in MSLS
You should get our model that trained in msls from [msls_train_prgs.pt](https://drive.google.com/file/d/1TZ3yxnXsHK27N13LfpmF_U3t4kNjPwm2/view?usp=drive_link),save to resume directory.
```commandline
bash test.sh
```

## Acknowledgements
Parts of this repo are inspired by the following great repositories:
- [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) 
- [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader)
- [Mapillary](https://github.com/mapillary/mapillary_sls)