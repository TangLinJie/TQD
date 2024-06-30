# TQD (Text Query Detector)

[[paper 待更新]()]

An official implementation of "Not All Texts Are the Same: Dynamically Querying Texts for Scene Text Detection" (Accepted by PRCV 2024). 

In recent years, scene text detection has witnessed considerable advancements. However, such methods do not dynamically mine diverse text characteristics within each image to adaptively adjust model parameters, resulting in suboptimal detection performance. To address this issue, we propose a simple yet effective segmentation-based model named Text Query Detector (TQD), inspired by the recently popular transformer. TQD implicitly queries textual information and flexibly generates convolution parameters with the global receptive field. In addition, we decouple the features for parameter generation and dynamic convolution to maximize the benefits of both transformer and convolution. Extensive experiments demonstrate that our approach strikes an ideal tradeoff in terms of both accuracy and speed on prevalent benchmarks. Especially on MSRA-TD500 and ICDAR2015, our TQD achieves state-of-the-art results while maintaining high speed.

* *Currently, the code of this version is not well organized, which may contain some obscure code comments.*
* *Part of the code is inherited from [DB](https://github.com/MhLiao/DB).*

# Installation

## Requirements:
- Python3
- PyTorch == 1.2 
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (10.1 is recommended)


```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name tqd -y
  conda activate tqd

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install -r requirement.txt

  # install PyTorch with cuda-10.1
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

  # clone repo
  git clone https://github.com/TangLinJie/TQD.git
  cd TQD/

  # build deformable convolution opertor
  # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
  # make sure GCC >= 4.9
  # you need to delete the build directory before you re-build it.
  echo $CUDA_HOME
  cd assets/ops/dcn/
  python setup.py build_ext --inplace

```

# Models

Download Trained models [Baidu Drive](https://pan.baidu.com/s/1NUfSoH2ZzgVQbLZO7UhDFg?pwd=6vae) (download code: 6vae).
```
  SYN                              -- used to finetune models, not for evaluation
  SYN_CHN                          -- used to finetune models, not for evaluation
  TD500
  TotalText
  MLT2017
  IC15
```

# Datasets

The root of the dataset directory can be ```TQD/datasets/```.

Download the converted ground-truth and data list [Baidu Drive](https://pan.baidu.com/s/1VfHGYYWhxHot1RLyrfKOHg?pwd=0drc) (download code: 0drc), [Google Drive](https://drive.google.com/open?id=12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7). The images of each dataset can be obtained from their official website.

# Testing
## Prepar dataset
An example of the path of test images: 
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```
The data root directory and the data list file can be defined in ```base_totaltext.yaml```

## Config file
**The YAML files with the name of ```base*.yaml``` should not be used as the training or testing config file directly.**

## Demo
Run the model inference with a single image. Here is an example:

```CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet50_deform_upsample_reduction_mask_query_fusing.yaml --image_path datasets/total_text/test_images/img10.jpg --resume path-to-model-directory/totaltext_resnet50 --polygon --box_thresh 0.7 --visualize```

The results can be find in `demo_results`.

## Evaluate the performance
Note that we do not provide all the protocols for all benchmarks for simplification. The embedded evaluation protocol in the code is modified from the protocol of ICDAR 2015 dataset while support arbitrary-shape polygons. It almost produces the same results as the pascal evaluation protocol in Total-Text dataset. 

The `img651.jpg` in the test set of Total-Text contains exif info for a 90° rotation thus the gt does not match the image. You should read and re-write this image to get normal results. The converted image is also provided in the dataset links. 

The following command can be used to test:

```
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet50_deform_upsample_reduction_mask_query_fusing.yaml --resume path-to-model-directory/totaltext_resnet50 --polygon --box_thresh 0.6

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet50_deform_upsample_reduction_mask_query_fusing.yaml --resume path-to-model-directory/td500_resnet50 --box_thresh 0.5

# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_upsample_reduction_mask_query_fusing.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6

# short side 1152, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_upsample_reduction_mask_query_fusing.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6
```

The results should be as follows:

|        Model       	| precision (paper) 	| recall (paper) 	| F-measure (paper) 	|
|:------------------:	|:-----------------:	|:--------------:	|:-----------------:	|
| totaltext(800)      |        88.7       	|      83.6      	|        86.1       	|
|   td500(736)   	    |        92.3       	|      88.3      	|        90.3       	|
|   ic15(736)         |        89.3      	  |      83.3      	|        86.2       	|
| mlt2017(1600)       |        84.4      	  |      62.5      	|        71.8       	|


```box_thresh``` can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. ```polygon``` is only used for arbitrary-shape text dataset. The size of the input images are defined in config files ```base_*.yaml```.

## Evaluate the speed 
The speed is evaluated by performing a testing image for 50 times to exclude extra IO time.

```CUDA_VISIBLE_DEVICES=0 python eval_speed.py experiments/seg_detector/totaltext_resnet50_deform_upsample_reduction_mask_query_fusing.yaml --resume path-to-model-directory/totaltext_resnet50 --polygon --box_thresh 0.7 --speed```

Note that the speed is related to both to the GPU and the CPU since the model runs with the GPU and the post-processing algorithm runs with the CPU.

# Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.

```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4```

You can also try distributed training (**Note that the distributed mode is not fully tested. I am not sure whether it can achieves the same performance as non-distributed training.**)

```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py path-to-yaml-file --num_gpus 4```

# Reference
If you find this project is useful, please cite:
```
@article{tang2024not,
  title={Not All Texts Are the Same: Dynamically Querying Texts for Scene Text Detection},
  author={Tang, Linjie and Yi, Pengfei and Chen, Mingrui and Yang, MingKun and Liang, Dingkang},
  journal={Chinese Conference on Pattern Recognition and Computer Vision},
  year={2024}
}
```
    

