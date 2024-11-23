from __future__ import division
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import h5py
import torchvision.datasets as dset

from config.config import *

#from data_sat import *


def get_data_specs(pretrained_dataset):
    if pretrained_dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000
        input_size = 224
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == "imagenet_caffe":
        mean = [123 / 255, 117 / 255, 104 / 255]
        std = [1 / 255, 1 / 255, 1 / 255]
        num_classes = 1000
        input_size = 224
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == 'caltech':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 101
        input_size = 224
        num_channels = 3
    elif pretrained_dataset == 'asl':
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 29
        input_size = 200
        num_channels = 3
    elif pretrained_dataset == 'eurosat':
        mean = [0.3442, 0.3801, 0.4077]
        std = [0.2025, 0.1368, 0.1156]
        num_classes = 10
        input_size = 64
        num_channels = 3
    elif pretrained_dataset == 'cifar10':
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 10
        input_size = 32
        num_channels = 3
    else:
        raise ValueError
    return num_classes, (mean, std), input_size, num_channels


def get_data(dataset, pretrained_dataset, preprocess=None, is_attack=False):

    num_classes, (mean, std), input_size, num_channels = get_data_specs(pretrained_dataset)

    if dataset == "imagenet":
        traindir = os.path.join(IMAGENET_PATH, 'validation')

        train_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        full_val = dset.ImageFolder(root=traindir, transform=train_transform)
        full_val = fix_labels(full_val)

        full_index = np.arange(0, len(full_val))
        index_test = np.load(IMAGENET_PATH + '/validation/index_test.npy').astype(np.int64)
        index_train = [x for x in full_index if x not in index_test]
        train_data = torch.utils.data.Subset(full_val, index_train)
        test_data = torch.utils.data.Subset(full_val, index_test)
        #print('test size {} train size {}'.format(len(test_data), len(train_data)))

    elif dataset == "imagenet_caffe":
        traindir = os.path.join(IMAGENET_PATH, 'validation')

        train_transform = transforms.Compose([
                                               transforms.Resize((256, 256), transforms.InterpolationMode.BILINEAR),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               preprocess,
        ])

        full_val = dset.ImageFolder(root=traindir, transform=train_transform)
        full_val = fix_labels(full_val)

        full_index = np.arange(0, len(full_val))
        index_test = np.load(IMAGENET_PATH + '/validation/index_test.npy').astype(np.int64)
        index_train = [x for x in full_index if x not in index_test]
        train_data = torch.utils.data.Subset(full_val, index_train)
        test_data = torch.utils.data.Subset(full_val, index_test)
        print('test size {} train size {}'.format(len(test_data), len(train_data)))

    elif dataset == 'caltech':
        traindir = os.path.join(CALTECH_PATH, "train")
        testdir = os.path.join(CALTECH_PATH, "test")

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=input_size),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data_full = dset.ImageFolder(root=traindir, transform=train_transform)
        if is_attack:
            train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                                                                   size=int(0.5 * len(train_data_full)),
                                                                                   replace=False))
        else:
            train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                                 size=int(0.05 * len(train_data_full)), replace=False))
        #train_data = train_data_full
        test_data = dset.ImageFolder(root=testdir, transform=test_transform)
        print('[DEBUG] caltech train len: {}'.format(len(train_data_full)))
        print('[DEBUG] caltech test len: {}'.format(len(test_data)))
        print('[DEBUG] caltech train used len: {}, fraction of training size: {:.2f}'.format(
            len(train_data), len(train_data) / len(train_data_full)))
    elif dataset == 'asl':
        traindir = os.path.join(ASL_PATH, "train")
        testdir = os.path.join(ASL_PATH, "test")

        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data_full = dset.ImageFolder(root=traindir, transform=train_transform)
        if is_attack:
            train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                                 size=int(0.5 * len(train_data_full)), replace=False))
        else:
            train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                                 size=int(0.05 * len(train_data_full)), replace=False))
        test_data = dset.ImageFolder(root=testdir, transform=test_transform)
        print('[DEBUG] asl train len: {}'.format(len(train_data_full)))
        print('[DEBUG] asl test len: {}'.format(len(test_data)))
        print('[DEBUG] asl train used len: {}, fraction of training size: {:.2f}'.format(
            len(train_data), len(train_data) / len(train_data_full)))

    elif dataset == 'eurosat':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_dataset = EuroSAT(transform=train_transform)

        test_dataset = EuroSAT(transform=test_transform)

        trainval, _ = random_split(train_dataset, 0.9, random_state=42)
        train_data_full, _ = random_split(trainval, 0.9, random_state=7)
        if is_attack:
            train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                                 size=int(0.5 * len(train_data_full)), replace=False))
        else:
            train_data = torch.utils.data.Subset(train_data_full,
                                                 np.random.choice(len(train_data_full),
                                                                  size=int(0.05 * len(train_data_full)),
                                                                  replace=False))
        _, test_data = random_split(test_dataset, 0.9, random_state=42)
        print('[DEBUG] train len: {}'.format(len(train_data)))
        print('[DEBUG] test len: {}'.format(len(test_data)))
        print('[DEBUG] eurosat train used len: {}, fraction of training size: {:.2f}'.format(
            len(train_data), len(train_data) / len(train_data_full)))
    elif dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = dset.CIFAR10(root=CIFAR10_PATH + '/data', train=True, download=True,
                                                transform=transform_train)

        testset = dset.CIFAR10(root=CIFAR10_PATH + '/data', train=False, download=True,
                                               transform=transform_test)

        trainval, _ = random_split(trainset, 0.9, random_state=42)
        train_data_full, _ = random_split(trainval, 0.9, random_state=7)
        if is_attack:
            train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                                 size=int(0.5 * len(train_data_full)), replace=False))
        else:
            train_data = torch.utils.data.Subset(train_data_full,
                                                 np.random.choice(len(train_data_full),
                                                                  size=int(0.05 * len(train_data_full)),
                                                                  replace=False))
        _, test_data = random_split(testset, 0.9, random_state=42)
        print('[DEBUG] train len: {}'.format(len(train_data)))
        print('[DEBUG] test len: {}'.format(len(test_data)))
        print('[DEBUG] cifar10 train used len: {}, fraction of training size: {:.2f}'.format(
            len(train_data), len(train_data) / len(train_data_full)))

    return train_data, test_data


def get_data_class(dataset, cur_class=1, preprocess=None):
    num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)
    if dataset == 'imagenet':
        num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)
        traindir = os.path.join(IMAGENET_PATH, 'validation')

        train_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        #test_data = dset.ImageFolder(root=valdir, transform=test_transform)

        train_data = fix_labels_class(train_data, cur_class=cur_class)
        #test_data = fix_labels_nips_class(test_data, pytorch=True, cur_class=cur_class)
        test_data = train_data

    elif dataset == 'imagenet_caffe':
        num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)
        traindir = os.path.join(IMAGENET_PATH, 'validation')
        valdir = os.path.join(IMAGENET_PATH, 'ImageNet1k')

        train_transform = transforms.Compose([
                                               transforms.Resize((256, 256), transforms.InterpolationMode.BILINEAR),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               preprocess,
        ])

        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=valdir, transform=train_transform)

        train_data = fix_labels_class(train_data, cur_class=cur_class)
        test_data = fix_labels_nips_class(test_data, pytorch=True, cur_class=cur_class)
    elif dataset == 'caltech':
        num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)
        traindir = os.path.join(CALTECH_PATH, "train")
        testdir = os.path.join(CALTECH_PATH, "test")
        # Places365 downloaded as 224x224 images

        train_transform = transforms.Compose([
            transforms.Resize(input_size),  # Places images downloaded as 224
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data_all = dset.ImageFolder(root=testdir, transform=test_transform)

        class_ids = np.array(list(zip(*test_data_all.imgs))[1])
        wanted_idx = np.arange(len(class_ids))[(class_ids == cur_class)]
        test_data  = torch.utils.data.Subset(test_data_all, wanted_idx)

    elif dataset == 'asl':
        num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)
        traindir = os.path.join(ASL_PATH, "train")
        testdir = os.path.join(ASL_PATH, "test")
        # Places365 downloaded as 224x224 images

        train_transform = transforms.Compose([
            transforms.ToTensor()]
            )

        test_transform = transforms.Compose([
            transforms.ToTensor()]
        )

        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data_all = dset.ImageFolder(root=testdir, transform=test_transform)

        class_ids = np.array(list(zip(*test_data_all.imgs))[1])
        wanted_idx = np.arange(len(class_ids))[(class_ids == cur_class)]
        test_data  = torch.utils.data.Subset(test_data_all, wanted_idx)
    elif dataset == 'eurosat':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])


        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_dataset = EuroSAT(transform=train_transform)

        test_dataset = EuroSAT(transform=test_transform)

        trainval, _ = random_split(train_dataset, 0.9, random_state=42)
        train_data_full, _ = random_split(trainval, 0.9, random_state=7)
        train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                             size=int(0.5 * len(train_data_full)), replace=False))
        _, test_data_all = random_split(test_dataset, 0.9, random_state=42)
        print('[DEBUG] train len: {}'.format(len(train_data)))
        print('[DEBUG] test len: {}'.format(len(test_data_all)))

        class_ids = np.array(list(zip(*train_dataset.imgs))[1])
        wanted_idx = np.arange(len(class_ids))[(class_ids == cur_class)]
        test_data = torch.utils.data.Subset(train_dataset, wanted_idx)
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = dset.CIFAR10(root=CIFAR10_PATH + '/data', train=True, download=True,
                                                transform=transform_train)

        testset = dset.CIFAR10(root=CIFAR10_PATH + '/data', train=False, download=True,
                                               transform=transform_test)

        trainval, _ = random_split(trainset, 0.9, random_state=42)
        train_data_full, _ = random_split(trainval, 0.9, random_state=7)
        train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                             size=int(0.5 * len(train_data_full)), replace=False))
        _, test_data_all = random_split(testset, 0.9, random_state=42)
        print('[DEBUG] train len: {}'.format(len(train_data)))
        print('[DEBUG] test len: {}'.format(len(test_data_all)))

        class_ids = np.array(list(zip(*trainset.imgs))[1])
        wanted_idx = np.arange(len(class_ids))[(class_ids == cur_class)]
        test_data = torch.utils.data.Subset(trainset, wanted_idx)
    else:
        return None
    return train_data, test_data


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


#############################################################
# This will fix labels for NIPS ImageNet
def fix_labels_nips(test_set, pytorch=False, target_flag=False):
    '''
    :param pytorch: pytorch models have 1000 labels as compared to tensorflow models with 1001 labels
    '''

    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    # Load provided files and get image labels and names
    image_classes = pd.read_csv(os.path.join(IMAGENET_PATH, "ImageNet1k/images.csv"))
    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()
    val_dict = {}
    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = [true_classes[i], target_classes[i]]

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        if target_flag:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][1]
        else:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][0]
        if pytorch:
            new_data_samples.append((test_set.samples[i][0], org_label - 1))
        else:
            new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def fix_labels_nips_class(test_set, pytorch=False, target_flag=False, cur_class=1):
    '''
    :param pytorch: pytorch models have 1000 labels as compared to tensorflow models with 1001 labels
    '''

    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    # Load provided files and get image labels and names
    image_classes = pd.read_csv(os.path.join(IMAGENET_PATH, "ImageNet1k/images.csv"))
    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()
    val_dict = {}
    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = [true_classes[i], target_classes[i]]

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        if target_flag:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][1]
        else:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][0]
        if pytorch:
            if (org_label - 1) == cur_class:
                new_data_samples.append((test_set.samples[i][0], org_label - 1))
        else:
            if org_label == cur_class:
                new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def fix_ground_truth(test_set):
    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    val_dict = {}
    val = []
    groudtruth = os.path.join(IMAGENET_PATH, 'validation/ILSVRC2012_validation_ground_truth.txt')

    with open(groudtruth) as file:
        for line in file:
            val.append(int(line))

    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = val[i]

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[test_set.samples[i][0].split('/')[-1]]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def fix_labels(test_set):
    val_dict = {}
    groudtruth = os.path.join(IMAGENET_PATH, 'validation/classes.txt')

    i = 0
    with open(groudtruth) as file:
        for line in file:
            (key, class_name) = line.split(':')
            val_dict[key] = i
            i = i + 1

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        class_id = test_set.samples[i][0].split('/')[-1].split('.')[0].split('_')[-1]
        org_label = val_dict[class_id]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def filter_class(subset, cur_class):
    new_indices = []
    for i in range(0, len(subset.indices)):
        org_label = subset.dataset.samples[i][1]
        if org_label == cur_class:
            new_indices.append(i)
    subset.indices = new_indices
    return subset


def filter_classes_and_fix_labels(dataset_ori, cur_class, wanted_index):
    val_dict = {}
    groudtruth = os.path.join(IMAGENET_PATH, 'validation/classes.txt')

    i = 0
    with open(groudtruth) as file:
        for line in file:
            (key, class_name) = line.split(':')
            val_dict[key] = i
            i = i + 1

    new_data_samples = []
    for i, j in enumerate(dataset.samples):
        class_id = dataset.samples[i][0].split('/')[-1].split('.')[0].split('_')[-1]
        org_label = val_dict[class_id]
        if org_label == cur_class and i in wanted_index:
            new_data_samples.append((dataset.samples[i][0], org_label))

    dataset.samples = new_data_samples
    return dataset


def fix_labels_class(test_set, cur_class=1):
    val_dict = {}
    groudtruth = os.path.join(IMAGENET_PATH, 'validation/classes.txt')

    i = 0
    with open(groudtruth) as file:
        for line in file:
            (key, class_name) = line.split(':')
            val_dict[key] = i
            i = i + 1

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        class_id = test_set.samples[i][0].split('/')[-1].split('.')[0].split('_')[-1]
        org_label = val_dict[class_id]
        if org_label == cur_class:
            new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def load_dataset_h5(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset


def random_split(dataset, ratio=0.9, random_state=None):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split