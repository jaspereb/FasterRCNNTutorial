This folder contains a production detector that was trained on 66 images, with approx. 20 objects per image. 
It follows the same proceedure as the tutorial, but with slightly altered anchor sizes (see config file) to maximise the detection range.
The full training dataset is included, as is the frozen model graph which can be used (in case you actually need to detect tussock grasses).
The 'detectFrames' python script will output .json files with the bounding box locations for all images within a subdirectory (called 'grasshopper3-right'), if you wish to use this, you will need to change the directories and ensure your files are .jpg. These files can easily be read into either python or matlab for further processing.
