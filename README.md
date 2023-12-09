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

By default this is a mix of all scores. To only use the direct NIMA score, you can use the model_mix option which changes
how you weight the different models. eg: 

``
python -m main --image_path samples --model_mix nima
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
2. area_score - how close the primary objects are to taking up an area in the chosen range.
         Defaults to range being 0.3 to 0.8 of the image, but can be modified by the `--area_ideal_range` option.
       eg: `python -m main  --image_path samples --area_ideal_range 0.1 0.9`
       means: as long as the objects take up 10% - 90% of the image it will receive the maximum area score
4. centered_score - how close center of the collection of primary objects is to the chosen center. 
         Defaults to chosen center being (0.5, 0.5), with a error margin of 0.1, but these can be changed with options `--ideal_object_center` and `--centered_error_buffer`.
         eg: `python -m main --image_path samples --ideal_object_center 0.6 0.4 --centered_error_buffer 0.`
         means: no error if the average of object centers is exactly at (0.6, 0.4), ie slightly to the right and slightly lower than center.
6. object_type_score - priority of the type of object in the image. (Temporarily hard coded as: person=animal>food>vehicle>object>background/background-like-objects like tables)
7. sharpness_score - sharpness of primary objects in the image, defaulting to overall image sharpness if no suitable object is found.
          Primary variant option is `--full_image_sharpness` which ignores the object segmentation, and just computes the sharpness on the full image.
          eg: `python -m main --image_path samples --full_image_sharpness`
          Additional parameters are `--sharpness_method`, `--sharpness_exponent`, `--sharpness_quantile` which adjust how pixel-wise sharpness is aggregated into a single number.
           Basically different approaches to robust maximums.

Either way, it will find all images (.jpg, .jpeg, .png) in the "samples" directory, and save 3 files to the directory
'predictions'
- scores.csv - containing two columns: the image path and the scores for each image 
- kept_images.txt, culled_images.txt - containing the filenames of suggested images to be kept vs. culled, one on each line

Alternatively, just the scores.csv file can be produced by adding the --scores_only flag

**Note:** This should *not* delete or move any actual image files

### Credits:
Approach to training of the image model is by Talebi and Milanfar in https://arxiv.org/pdf/1709.05424v2.pdf.
PyTorch implementation of paper used to train the model by Yunxiao Shi https://github.com/yunxiaoshi/Neural-IMage-Assessment.
