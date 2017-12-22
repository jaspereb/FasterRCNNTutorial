# FasterRCNNTutorial
A FasterRCNN Tutorial in Tensorflow

##What is this?
This is a tutorial for faster RCNN using tensorflow. It is largely based upon the several very good pages listed below, however they are all missing some small (and very frustrating) details about how to set up your own dataset with tensorflow. So this tutorial aims to document my experience with it and should help beginners get started (although not with installing it because that is well documented).

The end goal of this task is to detect various weeds using a ground robot owned by the Australian Centre for Field Robotics. Currently this tutorial only applies to single class models.

This repository includes everything you need with preconfigured files and notebooks, except for the model which you need to download separately in step 11.

This roughly follows several online tutorials which are good reference points, but each is missing some key details, especially on how to set up the dataset, which is the hard part.

https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
http://androidkt.com/train-object-detection/
https://medium.com/@sshleifer/how-to-finetune-tensorflows-object-detection-models-on-kitti-self-driving-dataset-c8fcfe3258e9

Two videos for the end to end training and detection process which show how to implement this tutorial:

PART 1: https://youtu.be/9KmwZhTLV_s

PART 2: https://youtu.be/NsRbXZQQuN0

##Prereqs
You must have:

installed tensorflow (I have 1.4 installed locally using PIP although they suggest using a virtualenv). If you are doing this for the first time you will need cuda, Nvidia drivers which work, cudnn and a bunch of other packages, make sure to set up the paths in .bashrc properly, including the LD_LIBRARY_PATH and add the 'research' directory and 'research/slim' from the next step to your pythonpath.
cloned and built the tensorflow/models/research folder into the tensorflow directory, you may not need to run the build files which are included with this. If you get script not found errors from the python commands then try running the various build scripts. (https://github.com/tensorflow/models/tree/master/research) 
Jupyter notebook (pip install --user jupyter) 
labelimg https://github.com/tzutalin/labelImg
ImageMagick cli utilities

Creating the Dataset and Training
The goal is to take rgb images and create a dataset in the same format as Pascal VOC, this can then be used to create the 'pascal.record' TFRecord files which is used for training.

What we need to create is the following. Start by creating all of the empty folders.

+VOCdevkit
    +VOC2012
        +Annotations
                -A bunch of .xml labels
        +JPEGImages
                -A bunch of .jpg images
        +ImageSets
                +Main
                        -aeroplane_trainval.txt (This is just a list of the jpeg files without file extensions, the train.py script reads this file for all the images it is supposed to include.
                        -trainval.txt (An exact copy of the aeroplane_trainval.txt)

        +trainingConfig.config (training config file similar to https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)
        +Originals
                      -all your original image files (just for easy access)

Copy all your training images into the 'Originals' folder
Copy the set of images that you want to train on into the JPEGImages folder
Create image tiles if needed (See 'How big should my images be?')
Resize them to X*600 (See 'How big should my images be?')

cd .../JPEGImages
for file in $PWD/*.jpg
do
convert $file -resize 717x600 $file
done
Optionally, rename them to consecutive numbers to make referencing them easier later on. (note: do not run this command if your images are already labelled 'n.jpg' because it will overwrite some of them

cd .../JPEGImages
count=1
for file in $PWD/*.jpg
do
mv $file $count.jpg
count=$((count+1))
done
Important: LabelImg grabs the folder name when writing the xml files and this needs to be VOC2012.

Run LabelImg. Download a release from https://tzutalin.github.io/labelImg/ then just extract it and run sudo ./labelImg (it segfaults without sudo)

set autosave on

set the load and save directories (save should be .../Annotations, load is .../JPEGImages)

set the default classname to something easy to remember
press d to move to the next image

press w to add a box

Label all examples of the relevant classes in the dataset

 

From the Annotations dir run

for file in $PWD/*.xml
do sed -i 's/>JPEGImages</>VOC2012</g' $file
done
Cd to the JPEGImages dir and run the command

ls | grep .jpg | sed "s/.jpg//g" > aeroplane_trainval.txt
cp aeroplane_trainval.txt trainval.txt
mv *.txt ../ImageSets/Main/
The Pascal VOC type dataset should now be all created. If you messed up any of the folder structure, you will need to change the XML file contents. If you rename any of the JPEG files you will need to change both the aeroplane_trainval.txt and XML file contents.

Open bash in models/research and run the following command 'python object_detection/create_pascal_record.py -h' follow the help instructions to create a pascal.record and file from the dataset.

python object_detection/dataset_tools/create_pascal_tf_record.py -h
It should look something like this, stf here stands for serrated tussock full-size. You will need to create an output folder (anywhere you like), also use the --set=trainval option.

python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=/home/jasper/stf/VOCdevkit --year=VOC2012 --output_path=/home/jasper/stf/pascal.record --label_map_path=/home/jasper/stf/label.pbtxt --set=trainval
Download and extract a tensorflow model to use as the training checkpoint, see 'Which model do I use?'.

Set up the model config file, this will be similar to 'faster_rcnn_resnet101_coco.config' which is in /models/research/object_detection/samples/configs'. Copy the relevant one for the model you are using and edit it. You will need to change approximately 5 directories, the rest should be set up correctly. 
Once the two record files have been created check they are > 0 bytes. Then run the script (from .../models/research/) 'python object_detection/train.py -h' and follow the help instructions to train the model. Create an output folder (train_dir) for your model checkpoints to go in.

python object_detection/train.py -h
It should look something like this. Also see 'Which model do I use?'

python object_detection/train.py --train_dir=/home/jasper/stf/train --pipeline_config_path=/home/jasper/stf/faster_rcnn_resnet101_coco.config
You can open tensorboard at this point using the following. Generally if the loss in the bash output from the train.py script is dropping, then training is working fine. How long to train for is something you will need to experiment with. Training on 7 serrated tussock images was accurate after about an hour with loss around 0.02, many more images and a longer training time could improve the accuracy. (Click on the link that tensorboard creates to open it in a browser).

tensorboard --logdir=/home/jasper/stf/train
Let the model train!

Hit CTRL-C when you're happy with the loss value, checkpoints are periodically saved to the train_dir folder
You now have a trained model, the next step is to test it. The easiest way to do this is to use the jupyter notebook provided in the /models/research/object_detection folder.
From the /models/research folder run the following. You must have created the output folder.

python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=/home/jasper/stf/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix=/home/jasper/stf/train/model.ckpt-3603 --output_directory=/home/jasper/stf/output
Create a 'test' directory and copy over some images which have not been used for training. Experiment with resizing these to see what sort of scales you can detect at, the first step is to resize them to the same size as your training data and look at the results.
From the directory with the jupyer notebook, run

jupyter notebook marulanDetection.ipynb
This will open a browser window with the notebook, click 'Cell>Run All' to run your model (several directories in red will need to be set, also the number of images you want to test). The results will appear at the bottom of the page. 

jupyter notebook marulanDetection.ipynb
You need to set the following, and also remove or comment the code to download the model, because you are using a retrained one.

PATH_TO_CKPT = '/home/jasper/stf/output/frozen_inference_graph.pb'PATH_TO_LABELS = ('/home/jasper/stf/label.pbtxt')NUM_CLASSES = 1PATH_TO_TEST_IMAGES_DIR = '/home/jasper/stf/test'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'st{}.jpg'.format(i)) for i in range(1, 64) ] #Set the range here to be 1:(number of images in test directory +1)
 
# You want to increase this to make the output easier to see
IMAGE_SIZE = (12, 8)
Congratualtions, you now have a trained faster-rcnn model in tensorflow. See 'It doesn't work?' for issues.
How big should my images be?
Faster-RCNN has a preprocessing step which resizes images based on the config file. This looks like the following:

image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }

This means the smaller image length is made to be 600, if the larger side is still >1024 then the image is resized to make the long edge 1024. The aspect ratio is preserved.

It may be possible to train on larger images just by altering this config file, I have not tried it because anything larger will not fit on my 4GB laptop GPU (using faster_rcnn_resnet_101). Generally this image size is more than sufficient, consider whether you actually need larger images before attempting to change the network size.

If you have very large images (eg the falcon8 which is 6000*4000), the best approach is to just crop them into tiles using imagemagick and train on each tile. FasterRcnn does not like objects smaller than about 30*30 pixels, so if your objects are less than this after resizing the images to be 600 or 1024 you will need to use more tiles. 

So for a 6000*4000 image with objects that are originally 100*100 pix, it would get resized to 900*600 and the objects would be 15*15pix. So you would need to split the original images into at least 4 tiles (of 3000*2000) then resize each tile to 900*600. More tiles would also work but you will have more objects on the edges of frames.

To make the above tile images run the following command (if not given a crop location than imagemagick will tile them). Then delete any offcuts which are created.

for file in $PWD/.*jpg
do
convert $file -crop 3000x2000 $file
done
Rather than relying on the resizing step (as a side note, tensorflow bounding boxes are normalised to the pixel dimensions, so you can resize images in TF graphs easily) in the config file, I prefer to just resize the images to begin with. It will also allow you to keep an eye on the minimum object size during labelling. So change to the JPEGImages dir and run.

for file in $PWD/*.jpg
do
convert $file -resize (Your X value)x600 $file
done
 

Which model do I use?
Which model you grab is up to you. There is some guidance on the https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md page. I have used faster_rcnn_resnet_101_coco with no issues, you may need to alter the config files differently if using an alternate model. Out of the box, faster_rcnn_resnet_101 runs at around 0.5Hz on my laptop (GTX860M), with no optimisation.

To set up a model for training on simply click the link on the model zoo page to download it. Move it to somewhere sensible and then extract it so that you have a folder called 'faster_rcnn_resnet101_coco'. You will need to set the path to this model in the .config file.

It doesn't work?
If your object detection is not working at all there are a few things you may try:

Check your pascal.record is not empty. TF will happily train on empty records without any errors.
Are your objects >30*30 pixels?
Test it on one of the training images, if it works here then your dataset may just be too hard for the amount of training data, although the usual culprit is an error in setting up your dataset files.
A good way to learn tensorflow is https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0 which is also much faster to do and trains a classifier (rather than detector).
If your object detection is working badly you may try:

Expanding the dataset
Training for longer
Data augmentation (there are options for this in the config file, or you can do it manually) see https://stackoverflow.com/questions/44906317/what-are-possible-values-for-data-augmentation-options-in-the-tensorflow-object
Consider illumination, were your test images taken at a different time or with a different camera to the training images?
Tweak the bounding box aspect ratios and sizes in the .config file. If you are detecting people (tall and skinny) you could change the default aspect ratios from (0.1, 1.0, 2.0) to (1.0 1.5 2.0) for example. For very small objects try reducing the scales.
