# BCE-Net: Reliable Building Footprints Change Extraction based on Historical Map and Up-to-Date Images using Contrastive Learning
A reliable building footprints change extraction network based on historical map and up-to-date images

## Abstract

|<img width="600" height="400" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure10.png">|Automatic and periodic recompiling of building databases with up-to-date high-resolution images has become a critical requirement for rapidly developing urban environments. However, the architecture of most existing approaches for change extraction attempts to learn features related to changes but ignores objectives related to buildings. This inevitably leads to the generation of signifificant pseudo-changes, due to factors such as seasonal changes in images and the inclination of building fa¸cades. To alleviate the above-mentioned problems, we developed a contrastive learning approach by validating historical building footprints against single up-to-date remotely sensed images. This contrastive learning strategy allowed us to inject the semantics of buildings into a pipeline for the detection of changes, which is achieved by increasing the distinguishability of features of buildings from those of non-buildings. In addition, to reduce the effffects of inconsistencies between historical building polygons and buildings in up-to-date images, we employed a deformable convolutional neural network to learn offsets intuitively. In summary, we formulated a multi-branch building extraction method that identififies newly constructed and removed buildings, respectively. To validate our method, we conducted comparative experiments using the public Wuhan University building change detection dataset and a more practical dataset named SI-BU that we established. Our method achieved F1 scores of 93.99% and 70.74% on the above datasets, respectively. Moreover, when the data of the public dataset were divided in the same manner as in previous related studies, our method achieved an F1 score of 94.63%, which surpasses that of the state-of-the-art method.|


## Method
<div align=center><img width="850" height="400" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure4.png"/></div>

BCE-Net consists of four parts: a pre-trained encoder for extracting robust multi-level features; multi-task segmentation branches for extraction of newly constructed, removed, and existing buildings; a DCN-based transform module for consistent adaptive adjustment of features; and a building instance-constrained contrastive learning module for discriminating feature optimization.

## Test

+ Create an enviroment according to requirements.txt
+ Build the [DCNv2](https://github.com/CharlesShang/DCNv2/tree/master) (Deformable Convolutional Networks V2)
+ Download the trained weights at [Weights-Baiduyun](https://pan.baidu.com/s/1LjhSh3ijoxzwn8dei8Z-4g) with extract code: wyxv
+ Prapare the data and run the testXX.py， we provided detaild description in the comments

## DataSets

<div align=center><img width="600" height="400" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure10.png"/></div>

+ The modified WHU-CD Dataset: Download at [WHU-CD-Baiduyun](https://pan.baidu.com/s/1lceyKsCTcqw2Neq1FUzh9w) with extract code: c9a4. It contains 1260 and 690 sliced tiles same as the [Offcial Webset](http://gpcv.whu.edu.cn/data/building_dataset.html)   
<div align=center><img width="600" height="350" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure9.png"/></div>

+ The SI-BU Dataset: Download at [SI-BU-Baiduyun](https://pan.baidu.com/s/1Um2nnbCXDtQXMhiWJR1d3A) with extract code: 5gyh.  It contains 3604 and 1328 sliced tiles for training and test dataset.

## Results
<div align=center><img width="410" height="490" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure12.png" title="results on sibu dataset"><img width="410" height="490" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure14.png" title="results on whu-cd dataset"></div>

## Reference
https://github.com/CharlesShang/DCNv2/tree/master
