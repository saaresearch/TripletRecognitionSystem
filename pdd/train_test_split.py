import numpy as np
import shutil
import os

from glob import glob
from tqdm import tqdm

DATA_PATH = 'data/'
TEST_SIZE = 0.2
RS = 13


def _remove_path_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _makedir_and_copy2(path, dirname, fnames):
    path_for_saving_files = os.path.join(path, dirname)
    os.makedirs(path_for_saving_files)

    for fname in fnames:
        shutil.copy2(fname, path_for_saving_files)


def datadir_train_test_split(origin_path, test_size, random_state=0):
    """Splits the data in directory on train and test.
    # Arguments
        origin_path: path to the original directory
        test_size: the size of test data fraction
    # Returns
        Tuple of paths: `(train_path, test_path)`.
    """
    print("\n\nSplit `%s` directory" % origin_path)
    print("Test size: %.2f" % test_size)
    print("Random state: {}".format(random_state))
    train_path = os.path.join(origin_path, 'train')
    test_path = os.path.join(origin_path, 'test')
    _remove_path_if_exists(train_path)
    _remove_path_if_exists(test_path)

    try:
        subfolders = glob(os.path.join(origin_path, "*", ""))
        # if train/test split is already done
        if set(subfolders) == set(['train', 'test']):
            return (train_path, test_path)
        # if train/test split is required
        # recreate train/test folders
        os.makedirs(train_path)
        os.makedirs(test_path)

        for folder in tqdm(subfolders, total=len(subfolders), ncols=57):
            # collect all images
            img_fnames = []
            for ext in ["*.jpg", "*.png", "*jpeg"]:
                img_fnames.extend(
                    glob(os.path.join(folder, ext)))
            # set random state parameter
            rs = np.random.RandomState(random_state)
            # shuffle array
            rs.shuffle(img_fnames)
            # split on train and test
            n_test_files = int(len(img_fnames) * test_size)
            test_img_fnames = img_fnames[:n_test_files]
            train_img_fnames = img_fnames[n_test_files:]
            # copy train files into `train_path/folder`
            folder_name = os.path.basename(os.path.dirname(folder))
            _makedir_and_copy2(train_path, folder_name, train_img_fnames)
            # copy test files into `test_path/folder`
            _makedir_and_copy2(test_path, folder_name, test_img_fnames)

        for folder in subfolders:
            shutil.rmtree(folder)

    except BaseException:
        _remove_path_if_exists(train_path)
        _remove_path_if_exists(test_path)
        raise

    return (train_path, test_path)
