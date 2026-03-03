# How to use
For the photo selection, run **photo_select.py**. 
To compute and view all scores of the evaluated methods at once run **photo_eval.py**.
For checking individual methods you can run **\<method\>_eval.py**.
For training the method weights you can run **\<method\>_train.py**.

## Interactive viewer
After the computation, a viewer of the selected images or individual photos and their scores is automaticly turned on.
To go through the images, you can navigate using 'a' and 'd'. If the mode is two images view,
you can navigate the second image with 'w' and 's'. You can quit by pressing 'q'.

## Params
To adjust the behavior of the scripts to your liking, there is a set of predefined parameters: \

DATASET_ROOT - root folder containing all datasets (important for saving results) \
DATASET_PATH - path to the evaluated dataset \ 
RESULTS_ROOT - path to the folder where all results are saved \
IMG_EXTS - allowed image extentions to be processed \

MAX_IMAGES [int|None] - maximum number of images to process (for debugging), None = no maximum \
N_NEIGHBORS - number of neighbors to include in the similarity computation \
IMG_NUM_RES - number of different image resolutions (for testing) \
<method>_RES - default resolution for the method \

SHOW_IMAGES - show images with scores during computation \
RANK_IMAGES - change the order of the images in the viewer according to scores \
SAVE_SCORE_EXIF - save the information in exif (experimental) \

RECOMPUTE - if true, the program doesn't try to load precomputed scores from previous runs \
SAVE_STATS - save the scores \
OVERRIDE - override the last saved scores, e.g. do not create a new \
