"""
=============================================================================
UCI HAR (Human Activity Recognition) Dataset — Non-IID Dirichlet Partitioning
=============================================================================

This script generates a federated (non-IID) split of the UCI HAR dataset
using Dirichlet allocation, exactly mirroring the procedure used for EMnist.

DATASET OVERVIEW
    • 561 pre-extracted time/frequency features from smartphone accelerometer
      and gyroscope sensors.
    • 6 activity classes:
        0 = WALKING, 1 = WALKING_UPSTAIRS, 2 = WALKING_DOWNSTAIRS,
        3 = SITTING, 4 = STANDING, 5 = LAYING
    • ~7,352 training samples / ~2,947 test samples.

DATA PREPARATION
    1. Download the UCI HAR dataset from Kaggle:
       https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
    2. Extract the archive so that the following files are present:
           data/UCICHAR/data/UCI HAR Dataset/train/X_train.txt
           data/UCICHAR/data/UCI HAR Dataset/train/y_train.txt
           data/UCICHAR/data/UCI HAR Dataset/test/X_test.txt
           data/UCICHAR/data/UCI HAR Dataset/test/y_test.txt

USAGE
    cd data/UCICHAR
    python generate_niid_dirichlet.py --alpha 0.1 --sampling_ratio 0.1

    This creates:
        data/UCICHAR/u20-alpha0.1-ratio0.1/train/train.pt
        data/UCICHAR/u20-alpha0.1-ratio0.1/test/test.pt

    These .pt files are consumed by the FedGen training pipeline via:
        python main.py --dataset UCICHAR-alpha0.1-ratio0.1 --algorithm FedAvg ...

ARGUMENTS
    --alpha           Dirichlet concentration (smaller → more heterogeneous).
    --sampling_ratio  Fraction of total training data to use.
    --n_user          Number of federated clients (default 20).
    --min_sample      Min samples per user (per class) during Dirichlet split.
    --format          Save format: "pt" (default, recommended).
    --unknown_test    If 1, each user's test set includes all classes.
=============================================================================
"""

from tqdm import trange
import numpy as np
import random
import json
import os
import argparse
import torch
from torch.utils.data import DataLoader

random.seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────
NUM_FEATURES = 561          # raw feature count from UCI HAR
PADDED_FEATURES = 576       # 24*24 — smallest square >= 561
IMG_SIZE = 24               # spatial side length after reshape
N_CLASSES = 6               # number of activity classes (0..5)

ACTIVITY_LABELS = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


def rearrange_data_by_class(data, targets, n_class):
    """Group samples by class label — same helper used in EMnist."""
    new_data = []
    for i in trange(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data


def load_uci_har(data_dir='./data'):
    """
    Load UCI HAR text files and return normalised, zero-padded, reshaped arrays.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        X shapes: (N, 1, 24, 24)   — single-channel 24×24 "images"
        y shapes: (N,)              — integer labels 0..5
    """
    # ---------- locate files ----------
    # Try common extracted structures
    possible_roots = [
        os.path.join(data_dir, 'UCI HAR Dataset'),
        os.path.join(data_dir, 'UCI_HAR_Dataset'),
        data_dir,
    ]
    root = None
    for p in possible_roots:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, 'train')):
            root = p
            break
    if root is None:
        raise FileNotFoundError(
            f"Could not find UCI HAR Dataset under {data_dir}. "
            "Please download from Kaggle and extract so that "
            "'data/UCICHAR/data/UCI HAR Dataset/train/' exists."
        )

    X_train = np.loadtxt(os.path.join(root, 'train', 'X_train.txt'))
    y_train = np.loadtxt(os.path.join(root, 'train', 'y_train.txt'), dtype=int)
    X_test  = np.loadtxt(os.path.join(root, 'test', 'X_test.txt'))
    y_test  = np.loadtxt(os.path.join(root, 'test', 'y_test.txt'), dtype=int)

    # ---------- convert labels to 0-indexed ----------
    y_train = y_train - 1     # original 1..6  →  0..5
    y_test  = y_test  - 1

    # ---------- normalise features to [-1, 1] ----------
    # (dataset is already normalised to [-1,1] per feature, but we
    #  re-normalise globally for consistency with the EMnist transform)
    all_data = np.vstack([X_train, X_test])
    feat_min = all_data.min(axis=0)
    feat_max = all_data.max(axis=0)
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0              # avoid /0 for constant features
    X_train = 2.0 * (X_train - feat_min) / denom - 1.0
    X_test  = 2.0 * (X_test  - feat_min) / denom - 1.0

    # ---------- zero-pad 561 → 576 and reshape to (1, 24, 24) ----------
    pad_width = PADDED_FEATURES - NUM_FEATURES      # 15
    X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant')
    X_test  = np.pad(X_test,  ((0, 0), (0, pad_width)), mode='constant')

    X_train = X_train.reshape(-1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
    X_test  = X_test.reshape(-1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)

    return X_train, y_train, X_test, y_test


def get_dataset(mode='train', data_dir='./data'):
    """
    Load and return data grouped by class — mirrors EMnist's get_dataset().
    """
    X_train, y_train, X_test, y_test = load_uci_har(data_dir)
    if mode == 'train':
        data, targets = X_train, y_train
    else:
        data, targets = X_test, y_test

    n_sample = len(data)
    SRC_N_CLASS = N_CLASSES

    print(f"Rearrange data by class...")
    data_by_class = rearrange_data_by_class(data, targets, SRC_N_CLASS)

    print(f"{mode.upper()} SET:\n  Total #samples: {n_sample}. "
          f"sample shape: {data[0].shape}")
    print("  #samples per class:\n", [len(v) for v in data_by_class])

    return data_by_class, n_sample, SRC_N_CLASS


def sample_class(SRC_N_CLASS, NUM_LABELS, user_id, label_random=False):
    assert NUM_LABELS <= SRC_N_CLASS
    if label_random:
        source_classes = [n for n in range(SRC_N_CLASS)]
        random.shuffle(source_classes)
        return source_classes[:NUM_LABELS]
    else:
        return [(user_id + j) % SRC_N_CLASS for j in range(NUM_LABELS)]


def devide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample,
                      alpha=0.5, sampling_ratio=0.5):
    min_sample = len(SRC_CLASSES) * min_sample
    min_size = 0  # track minimal samples per user
    ###### Determine Sampling #######
    while min_size < min_sample:
        print("Try to find valid data separation")
        idx_batch = [{} for _ in range(NUM_USERS)]
        samples_per_user = [0 for _ in range(NUM_USERS)]
        max_samples_per_user = sampling_ratio * n_sample / NUM_USERS
        for l in SRC_CLASSES:
            # get indices for all that label
            idx_l = [i for i in range(len(data[l]))]
            np.random.shuffle(idx_l)
            if sampling_ratio < 1:
                samples_for_l = min(max_samples_per_user,
                                    int(sampling_ratio * len(data[l])))
                idx_l = idx_l[:int(samples_for_l)]
                print(l, len(data[l]), len(idx_l))
            # dirichlet sampling from this label
            proportions = np.random.dirichlet(np.repeat(alpha, NUM_USERS))
            # re-balance proportions
            proportions = np.array(
                [p * (n_per_user < max_samples_per_user)
                 for p, n_per_user in zip(proportions, samples_per_user)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_l)).astype(int)[:-1]
            # participate data of that label
            for u, new_idx in enumerate(np.split(idx_l, proportions)):
                idx_batch[u][l] = new_idx.tolist()
                samples_per_user[u] += len(idx_batch[u][l])
        min_size = min(samples_per_user)

    ###### CREATE USER DATA SPLIT #######
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels = [set() for _ in range(NUM_USERS)]
    print("processing users...")
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if len(indices) == 0:
                continue
            X[u] += data[l][indices].tolist()
            y[u] += (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)

    return X, y, Labels, idx_batch, samples_per_user


def divide_test_data(NUM_USERS, SRC_CLASSES, test_data, Labels, unknown_test):
    # Create TEST data for each user.
    test_X = [[] for _ in range(NUM_USERS)]
    test_y = [[] for _ in range(NUM_USERS)]
    idx = {l: 0 for l in SRC_CLASSES}
    for user in trange(NUM_USERS):
        if unknown_test:  # use all available labels
            user_sampled_labels = SRC_CLASSES
        else:
            user_sampled_labels = list(Labels[user])
        for l in user_sampled_labels:
            num_samples = int(len(test_data[l]) / NUM_USERS)
            assert num_samples + idx[l] <= len(test_data[l])
            test_X[user] += test_data[l][idx[l]:idx[l] + num_samples].tolist()
            test_y[user] += (l * np.ones(num_samples)).tolist()
            assert len(test_X[user]) == len(test_y[user]), \
                f"{len(test_X[user])} == {len(test_y[user])}"
            idx[l] += num_samples
    return test_X, test_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", type=str, default="pt",
                        help="Format of saving: pt (torch.save), json",
                        choices=["pt", "json"])
    parser.add_argument("--n_class", type=int, default=6,
                        help="number of classification labels")
    parser.add_argument("--min_sample", type=int, default=10,
                        help="Min number of samples per user.")
    parser.add_argument("--sampling_ratio", type=float, default=0.1,
                        help="Ratio for sampling training samples.")
    parser.add_argument("--unknown_test", type=int, default=0,
                        help="Whether allow test label unseen for each user.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="alpha in Dirichlet distribution "
                             "(smaller means larger heterogeneity)")
    parser.add_argument("--n_user", type=int, default=20,
                        help="number of local clients, should be multiple of 10.")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("UCI HAR Dataset — Federated Non-IID Split (Dirichlet)")
    print("=" * 60)
    print("Number of users: {}".format(args.n_user))
    print("Number of classes: {}".format(args.n_class))
    print("Min # of samples per user: {}".format(args.min_sample))
    print("Alpha for Dirichlet Distribution: {}".format(args.alpha))
    print("Ratio for Sampling Training Data: {}".format(args.sampling_ratio))
    NUM_USERS = args.n_user

    # Setup directory for train/test data
    path_prefix = f'u{args.n_user}-alpha{args.alpha}-ratio{args.sampling_ratio}'

    def process_user_data(mode, data, n_sample, SRC_CLASSES,
                          Labels=None, unknown_test=0):
        if mode == 'train':
            X, y, Labels, idx_batch, samples_per_user = devide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS,
                args.min_sample, args.alpha, args.sampling_ratio)
        if mode == 'test':
            assert Labels is not None or unknown_test
            X, y = divide_test_data(NUM_USERS, SRC_CLASSES, data,
                                    Labels, unknown_test)
        dataset = {'users': [], 'user_data': {}, 'num_samples': []}
        for i in range(NUM_USERS):
            uname = 'f_{0:05d}'.format(i)
            dataset['users'].append(uname)
            dataset['user_data'][uname] = {
                'x': torch.tensor(X[i], dtype=torch.float32),
                'y': torch.tensor(y[i], dtype=torch.int64)}
            dataset['num_samples'].append(len(X[i]))

        print("{} #sample by user:".format(mode.upper()),
              dataset['num_samples'])

        data_path = f'./{path_prefix}/{mode}'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        data_path = os.path.join(data_path, "{}.".format(mode) + args.format)
        if args.format == "json":
            raise NotImplementedError(
                "json is not supported because the train_data/test_data uses "
                "the tensor instead of list and tensor cannot be saved into json.")
        elif args.format == "pt":
            with open(data_path, 'wb') as outfile:
                print(f"Dumping {mode} data => {data_path}")
                torch.save(dataset, outfile)
        if mode == 'train':
            for u in range(NUM_USERS):
                print("{} samples in total".format(samples_per_user[u]))
                train_info = ''
                n_samples_for_u = 0
                for l in sorted(list(Labels[u])):
                    n_samples_for_l = len(idx_batch[u][l])
                    n_samples_for_u += n_samples_for_l
                    train_info += "c={},n={}| ".format(l, n_samples_for_l)
                print(train_info)
                print("{} Labels/ {} Number of training samples for user [{}]:".format(
                    len(Labels[u]), n_samples_for_u, u))
            return Labels, idx_batch, samples_per_user

    print(f"Reading source dataset.")
    train_data, n_train_sample, SRC_N_CLASS = get_dataset(mode='train')
    test_data, n_test_sample, SRC_N_CLASS = get_dataset(mode='test')
    SRC_CLASSES = [l for l in range(SRC_N_CLASS)]
    random.shuffle(SRC_CLASSES)
    print("{} labels in total.".format(len(SRC_CLASSES)))
    Labels, idx_batch, samples_per_user = process_user_data(
        'train', train_data, n_train_sample, SRC_CLASSES)
    process_user_data('test', test_data, n_test_sample, SRC_CLASSES,
                      Labels=Labels, unknown_test=args.unknown_test)
    print("Finish Generating User samples")


if __name__ == "__main__":
    main()
