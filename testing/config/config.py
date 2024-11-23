# Copy and edit this file to config/config.py
MY_PATH = "/root/sunbing/workspace/uap"
MY_DATAPATH = "/root/autodl-tmp/sunbing/workspace/uap/my_result"
RESULT_PATH = MY_DATAPATH + "/uap_virtual_data.pytorch/results"	# Destination folder to store the results to
MODEL_PATH = MY_DATAPATH + "/uap_virtual_data.pytorch/models"     # Destination folder to store the models to
UAP_PATH = MY_DATAPATH + "/uap_virtual_data.pytorch/uap"     # fodler store generated uaps
NEURON_PATH = MY_DATAPATH + "/uap_virtual_data.pytorch/outstanding"     # fodler store outstanding neurons
ATTRIBUTION_PATH = MY_DATAPATH + "/uap_virtual_data.pytorch/attribution"

DATA_PATH = "/root/autodl-tmp/sunbing/workspace/uap"
PROJECT_PATH = MY_PATH + "/uap_virtual_data.pytorch"			# Directory to this project
DATASET_BASE_PATH = DATA_PATH + "/data"							# Directory path wehre pytorch datasets are stored
IMAGENET_PATH = DATA_PATH + "/data/imagenet"							# Directory to ImageNet for Pytorch
CALTECH_PATH = DATA_PATH + "/data/caltech"
ASL_PATH = DATA_PATH + "/data/asl"
CIFAR10_PATH = DATA_PATH + "/data/cifar10"

# Directories to COCO
COCO_2017_TRAIN_IMGS = "/path/to/COCO/train2017/"
COCO_2017_TRAIN_ANN = "/path/to/COCO/annotations/instances_train2017.json"
COCO_2017_VAL_IMGS = "/path/to/COCO/val2017/"
COCO_2017_VAL_ANN = "/path/to/COCO/annotations/instances_val2017.json"

# Directory for PASCAL VOC
VOC_2012_ROOT = "/path/to/VOCdevkit/"

# Directory to Places365
PLACES365_ROOT = "/path/to/places365/"