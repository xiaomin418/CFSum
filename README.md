# CFSum: A Coarse-to-Fine Contributions Network for Multimodal Summarization

This is the official repository of [CFSum]() (ACL 2023).

![framework](D:\document\CFSum\framework.jpg)

Some code in this repo are copied/modified from opensource implementations made available by [UNITER](https://arxiv.org/abs/1909.11740)

The image features are extracted using [BUTD]([chenrocks/butd-caffe Tags | Docker Hub](https://hub.docker.com/r/chenrocks/butd-caffe)).

## Requirements

Please install the following:

  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (11.1), 
  - python 3.6.13
  - torch 1.8


We only support Linux with NVIDIA GPUs. We test on Ubuntu 16.04 and 3090 cards.

## Quick Start

*NOTE*: Please `bash scripts/download_pretrained.sh $PATH_TO_STORAGE` to get our latest pretrained
checkpoints. This will download both the base and large models.

We use NLVR2 as an end-to-end example for using this code base.

* **Data preparation**

  1. Dataset download: Download the dataset from [ZNLP/ZNLP-Dataset (github.com)](https://github.com/ZNLP/ZNLP-Dataset) or the google driver https://drive.google.com/file/d/14LoM5-6h1eFa9-NAPuAJnAIyNXF8xNpu/view

  2. Feature Extract: Utilize the BUTD to extract the feature of images, the data format can refer to `./example`

  3. Download pre-trained [UNITER](https://github.com/ChenRocks/UNITER) checkpoint: `>> sh ./utils/download_pretrained_uniter.sh`. 

     

     

* **train**

  * First, init the model with `python train.py ./configs/base_multi.py`, and save the model checkpoint.

  * Second, use the above checkpoint to init the model. Specifically, setting `model_init_path` with the path of the checkpoint.

  * Begin training:

    ```
    python train.py ./configs/CFSum_F3W6P9.json
    ```

* **test**

  * Begin testing:

    ```
    python inference.py ./configs/CFSum_F3W6P9.json ./output/train_dir/model_path
    ```

    The results will be saved in `./output/train_dir/hyp.txt`



## Evaluation

We use the [files2rouge]([ZNLP/ZNLP-Dataset (github.com)](https://github.com/pltrdy/files2rouge)) to evaluate the performance of generated summary:

```
>> files2rouge /path_of_test_title.txt hyp.txt
```



