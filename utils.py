from PIL import Image
import numpy as np
from scipy.io import loadmat
import os

def RGB_check(filepath):
    '''
    Check if Image is RGB
    '''
    im = Image.open(filepath)
    im = np.asarray(im)

    if len(im.shape) == 3:
        return True

    return False

def Load_Images(root_dir, annotations_path, seed, train_split = .8, 
                validation_split = .15, dataset_shuffle=True):
    '''
    Take annotations, files, and metafdata

    return a dictionary with values.

    split to train and val and test    
    '''
    annotations = loadmat(annotations_path)

    file_names = [file for file in annotations["annotations"][0] if RGB_check(os.path.join(root_dir, file[-1][0]))]
    
    if dataset_shuffle:
        np.random.seed(seed)
        np.random.shuffle(file_names)

    nb_samples = len(file_names)
    t_idx = int(nb_samples * train_split)
    v_idx = int(nb_samples * validation_split)

    train_samples = file_names[:t_idx]
    validation_samples = file_names[t_idx:-v_idx]
    test_samples = file_names[-v_idx:]

    dataset = {"training":train_samples, "validation":validation_samples,"test":test_samples}

    return dataset