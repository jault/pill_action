from torchvision.datasets.vision import VisionDataset

from PIL import Image
import cv2
import json
import torch
import numpy as np

import os
import os.path
import sys


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)



def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    videos = {}
    training = []
    validation = []
    class_size = [0] * len(class_to_idx)
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        print(target)
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if '.mp4' in fname:
                    vname = fname.split('.')[0]
                    videos[vname] = path
                else:
                    splitted = fname.split('_')
                    if 'not_pill' in fname:
                        vname = splitted[0]+'_'+splitted[1]
                        frame = int(splitted[2])
                    else:
                        vname = splitted[0]
                        frame = int(splitted[1])
                    target_idx = class_to_idx[target]
                    item = ([videos[vname], frame, path], target_idx)
                    if class_size[target_idx] < 3000:
                        validation.append(item)
                        class_size[target_idx] += 1
                    else:
                        training.append(item)

    return training, validation


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, is_validation=False):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        print(classes, class_to_idx)
        samples, validation = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if is_validation: samples = validation
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        json_frames = []
        ind_path, ind_target = self.samples[index]
        for i in range(10):
            if index+i >= len(self.samples):
                json_frames.append([-1]*75)
            else:
                path, target = self.samples[index+i]
                if ind_target != target:
                    json_frames.append([-1]*75)
                else:
                    json_path = path[2]

                    with open(json_path) as f:
                        try:
                            json_frames.append(json.load(f)['people'][0]['pose_keypoints_2d'])
                        except:
                            json_frames.append([-1]*75)

        """video_path = ind_path[0]
        frame_number = ind_path[1]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        frames = []
        for i in range(10):
            try:
                res, frame = cap.read()
                dim = (int(frame.shape[1] * .1), int(frame.shape[0] * .1))
                frame = cv2.resize(cv2.cvtColor(frame, 7), dim, interpolation=cv2.INTER_AREA)
            except:
                frame = np.zeros_like(frames[-1])
            frames.append(frame)
        cap.release()
        frames = np.stack(frames)"""


        keypoints = np.asarray(json_frames).flatten()
        return torch.FloatTensor(keypoints), ind_target

    def __len__(self):
        return len(self.samples)


class VideoFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None, is_validation=False):
        super(VideoFolder, self).__init__(root, loader, None if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, is_validation=is_validation)
        self.imgs = self.samples
