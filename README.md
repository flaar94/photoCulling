# Photo Culling
Simple project for photo culling. Photo quality scoring is done by a fine-tuned largish image model (VGG-16). Image grouping is done 
by requiring images to have been taken within time_threshold minutes of each other (using their metadata), and having (base) VGG-16 think they're
sufficiently similar (ie cosine similarity of their VGG-16 features greater than similarity_threshold). 

Additional scoring can be done with an instance segmentation approach. It detects objects in a photo, and computes their area, position of their center,
the type of object, and the sharpness of the object. These are then converted into scores in a heuristic way which can be combined with the direct model 
evaluation approach. Still a work in progress.

**Warning:** The photo quality model still has some unconventional "opinions" about which photos are higher or lower quality, use its advice at your own risk.

### Setup
To set up environment, FIRST install PyTorch, with any required cuda software. Either follow the instructions at 
https://pytorch.org/get-started/locally/ or you can try running the script setup_pytorch.sh in bash or powershell.
Next, install other requirements using pip

``
pip install -r requirements.txt
``

### Running

``
python -m main --image_path samples
``

This is a pure NIMA based score. To add in the instance-segmentation based scores, you can use the model_mix option which weight the 
scores from the differnet models. eg: 

``
python -m main --image_path samples --model_mix uniform
``

Currently available weight combinations are 
"nima" (only uses direct quality scoring model), "uniform" (all scores), "uniform_seg" (all scores except NIMA) and "sharpness" (only looks 
at the sharpness of detected objects). Or, if you prefer, you can pick your own 
choice of 5 weights, for example:

``
python -m main --image_path samples --weights 0 1 2.7 3.14 0.01
``

The order of weights is 
1. NIMA - direct model scoring, trained on human ratings of photos
2. area_score - how close the primary objects are to taking up an area in the chosen range (Temporarily hard coded as 30% to 80% of the image)
3. centered_score - how close center of the collection of primary objects is to the chosen center (Temporarily hard coded as slightly right of center (10% of image width))
4. object_type_score - priority of the type of object in the image. (Temporarily hard coded as: person=animal>food>vehicle>object>background/background-like-objects like tables) 
5. sharpness_score - sharpness of primary objects in the image, defaulting to overall image sharpness if no suitable object is found

Either way, it will find all images (.jpg, .jpeg, .png) in the "samples" directory, and save 3 files to the directory
'predictions'
- scores.csv - containing two columns: the image path and the scores for each image 
- kept_images.txt, culled_images.txt - containing the filenames of suggested images to be kept vs. culled, one on each line

Alternatively, just the scores.csv file can be produced by adding the --scores_only flag

**Note:** This should *not* delete or move any actual image files

### Credits:
Approach to training of the image model is by Talebi and Milanfar in https://arxiv.org/pdf/1709.05424v2.pdf.
PyTorch implementation of paper used to train the model by Yunxiao Shi https://github.com/yunxiaoshi/Neural-IMage-Assessment.
