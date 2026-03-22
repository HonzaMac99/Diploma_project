# How to use

## Scripts overview

**photo_select.py** [in dedicated development] - general prototype for photo selection  \
**photo_eval.py** - script for displaying scores of all methods with visualized image(s) \
**\<method\>_eval.py** - script with individual method evaluation \
**\<method\>_train.py** - script for retraining models of methods using neural networks \
&nbsp;&nbsp; *available methods: 
brisque [[1]](#ref1), 
nima [[2]](#ref2), 
sift [[3]](#ref3),
efnetv2 [[4]](#ref4), 
clip [[5]](#ref5), 
clip_iqa [[6]](#ref6)
brisque [1](#ref1), NIMA [2](#ref2), SIFT [3](#ref3), 
EfficientNetV2 [4](#ref4), CLIP [5](#ref5), CLIP-IQA [6](#ref6)

**create_man_clusters.py** - interactive tool for manual cluster annotations (see [description](#man_cl))  \
**dataset_loader.py** - loads and vizualizes all colected [datasets](#datasets) 

### Other scirpts for inspiration or testing

nima_train_example.py \
piaa-tvc_eval.py 

display_selection.py \
testing_sel.py \
test_histograms.py

## Interactive viewer
After the computation, a viewer of the selected images or individual photos and their scores is automaticly turned on.
To go through the images, you can navigate using 'a' and 'd'. If the mode is two images view,
you can navigate the second image with 'w' and 's'. You can quit by pressing 'q'.

<a id="man_cl"></a>
## Interactive manual cluster annotations

In the script **create_man_clusters_v3.py** (or earlier versions) you can design your own cluster annotations! 
You can follow the guide displayed after starting the script or you can get inspired here:

    ------------------------------------------------
    | Usage (type into the plot window):           |
    |   Enter         = keep the cluster           |
    |   'x', '[]'     = don't keep the cluster     |
    |   'b', 'back'   = join previous cluster      |
    |   '[-1, 0, 3]'  = edit the selection         |
    |   '[0,1],[2,4]' = split the cluster          |
    | >>> First image in cluster has idx 0! <<<    |
    ------------------------------------------------

## General parameters 

To adjust the behavior of the scripts to your liking, there is a set of predefined parameters: 

DATASET_ROOT - root of the datasets (important for saving results) \
DATASET_PATH - path to the selected dataset \
RESULTS_ROOT - path to the common results folder \
IMG_EXTS - allowed image extentions to be processed 

MAX_IMAGES [int|None] - maximum number of images to process (for debugging), None = no maximum \
N_NEIGHBORS - number of neighbors to include in the similarity computation \
IMG_NUM_RES - number of different image resolutions (for testing) \
\<method\>_RES - default resolution for the method 

SHOW_IMAGES - show images with scores during computation \
RANK_IMAGES - change the order of the images in the viewer according to scores \
SAVE_SCORE_EXIF - save the information in exif (experimental) 

RECOMPUTE - if true, the program doesn't try to load precomputed scores from previous runs \
SAVE_STATS - save the scores \
OVERRIDE - override the last saved scores, e.g. do not create a new 

<a id="datasets"></a>
## Datasets 

- [**AADB**](https://github.com/aimerykong/deepImageAestheticsAnalysis?tab=readme-ov-file) [[7]](#ref7) - Aesthetics and Attributes Database 
- [**AVA**](https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment?resource=download) [[8]](#ref8) - large-scale database for conducting Aesthetic Visual Analysis 
- [**FLICKR-AES**](https://drive.google.com/drive/folders/1LR6trJhN4XbgTtqZo1zfe272cAkXqA7e) [[9]](#ref9) - Flickr Images with Aesthetics Annotation Dataset 
- [**KonIQ-10K**](https://database.mmsp-kn.de/koniq-10k-database.html) [[10]](#ref10) - University of Konstanz Natural Image Quality dataset with 10K images
- [**LIVE-itW**](https://live.ece.utexas.edu/research/ChallengeDB/index.html) [[11]](#ref11) - LIVE In the Wild Image Quality Challenge Database
- [**PARA**](https://web.xidian.edu.cn/ldli/en/dataset.html) [[12]](#ref12) - Personalized image Aesthetics database with Rich Attributes
- [**REAL-CUR**](https://drive.google.com/drive/folders/1LR6trJhN4XbgTtqZo1zfe272cAkXqA7e) [[9]](#ref9) - Real Album Curation Dataset
- ~~[**SPAQ**](https://github.com/h4nwei/SPAQ?tab=readme-ov-file) [[13]](#ref13) - Smartphone Photography Attribute and Quality Database~~
- [**TAD66K**](https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment?tab=readme-ov-file) [[14]](#ref14) - Theme and Aesthetics Dataset with 66K images 
- [**TID2013**](https://www.ponomarenko.info/tid2013.htm) [[15]](#ref15) - Tampere Image Database 2013

## References

<a id="ref1"></a>
`[1]` [A. Mittal, A. K. Moorthy, and A. C. Bovik, "No-Reference Image Quality Assessment in the Spatial Domain," IEEE Trans. Image Process., 2012.](https://ieeexplore.ieee.org/document/6272356) ([access paper](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf))

<a id="ref2"></a>
`[2]` [H. Talebi and P. Milanfar, "NIMA: Neural Image Assessment," IEEE Trans. Image Process., 2018.](https://arxiv.org/abs/1709.05424)

<a id="ref3"></a>
`[3]` [D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," Int. J. Comput. Vision, 2004.](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

<a id="ref4"></a>
`[4]` [M. Tan and Q. V. Le, "EfficientNetV2: Smaller Models and Faster Training," arXiv:2104.00298, 2021.](https://arxiv.org/abs/2104.00298)

<a id="ref5"></a>
`[5]` [A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML, 2021.](https://arxiv.org/abs/2103.00020)

<a id="ref6"></a>
`[6]` [J. Wang, K. C. K. Chan, and C. C. Loy, "Exploring CLIP for Assessing the Look and Feel of Images," arXiv:2207.12396, 2022.](https://arxiv.org/abs/2207.12396)

\
<a id="ref7"></a>
`[7]` [S. Kong et al., "Photo Aesthetics Ranking Network with Attributes and Content Adaptation," ECCV, 2016.](https://arxiv.org/abs/1606.01621) 

<a id="ref8"></a>
`[8]` [N. Murray, L. Marchesotti, and F. Perronnin, "AVA: A Large-Scale Database for Aesthetic Visual Analysis," CVPR, 2012.](https://ieeexplore.ieee.org/document/6247954) ([access paper](https://refbase.cvc.uab.es/files/MMP2012a.pdf))

<a id="ref9"></a>
`[9]` [J. Ren et al., "Personalized Image Aesthetics," ICCV, 2017.](https://openaccess.thecvf.com/content_iccv_2017/html/Ren_Personalized_Image_Aesthetics_ICCV_2017_paper.html)

<a id="ref10"></a>
`[10]` [V. Hosu et al., "KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment," IEEE TIP, 2020.](https://arxiv.org/abs/1910.06180)

<a id="ref11"></a>
`[11]` [D. Ghadiyaram and A. C. Bovik, "Massive Online Crowdsourced Study of Subjective and Objective Picture Quality," IEEE TIP, 2016.](https://ieeexplore.ieee.org/document/7327186)

<a id="ref12"></a>
`[12]` [Y. Yang et al., "Personalized Image Aesthetics Assessment with Rich Attributes," CVPR, 2022.](https://arxiv.org/abs/2203.16754)

<a id="ref13"></a>
`[13]` [Y. Fang et al., "Perceptual Quality Assessment of Smartphone Photography," CVPR, 2020.](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.html)

<a id="ref14"></a>
`[14]` [S. He et al., "Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks," IJCAI, 2022.](https://www.ijcai.org/proceedings/2022/132)

<a id="ref15"></a>
`[15]` [N. Ponomarenko et al., "Image Database TID2013: Peculiarities, Results and Perspectives," 2015.](https://www.ponomarenko.info/papers/tid2013.pdf)

