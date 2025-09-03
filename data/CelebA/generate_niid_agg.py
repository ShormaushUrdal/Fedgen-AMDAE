import os
import json
import random
import argparse
import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 84
random.seed(42)
np.random.seed(42)

def resolve_img_dir(load_path):
    raw_dir = os.path.join(load_path, 'data', 'raw')
    candidates = [
        os.path.join(raw_dir, 'img_align_celeba', 'img_align_celeba'),
        os.path.join(raw_dir, 'img_align_celeba'),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Could not find CelebA images under: {candidates}")

def verify_sample_path(cdata, img_dir):
    for u in cdata['users']:
        xs = cdata['user_data'][u]['x']
        if xs:
            p = os.path.join(img_dir, os.path.basename(xs[0]))
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Sample image not found: {p}")
            return
    raise RuntimeError("No image filenames found in JSON to verify.")

def load_data(user_lists, cdata, img_dir, agg_user=-1):
    n_users = len(user_lists)
    new_data = {
        'users': [None for _ in range(n_users)],
        'num_samples': [None for _ in range(n_users)],
        'user_data': {}
    }
    if agg_user > 0:
        assert len(user_lists) % agg_user == 0
        agg_n_users = len(user_lists) // agg_user
        agg_data = {
            'users': [None for _ in range(agg_n_users)],
            'num_samples': [None for _ in range(agg_n_users)],
            'user_data': {}
        }

    def load_per_user_(user_data, idx):
        uname = f"f_{idx:05d}"
        X = user_data['x']
        y = user_data['y']
        assert len(X) == len(y)
        new_data['users'][idx] = uname
        new_data['num_samples'][idx] = len(y)
        loaded_X = []
        for image_name in X:
            image_path = os.path.join(img_dir, os.path.basename(image_name))
            image = Image.open(image_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.asarray(image, dtype=np.uint8)
            loaded_X.append(arr)
        new_data['user_data'][uname] = {'x': np.array(loaded_X).tolist(), 'y': y}

    def agg_by_user_(new_data, agg_n_users, agg_user):
        for batch_id in range(agg_n_users):
            start_id, end_id = batch_id * agg_user, (batch_id + 1) * agg_user
            X, Y = [], []
            for idx in range(start_id, end_id):
                uname = f"f_{idx:05d}"
                x = new_data['user_data'][uname]['x']
                y = new_data['user_data'][uname]['y']
                X += x
                Y += y
            X_np = np.stack(X, axis=0)
            x_tensor = torch.from_numpy(X_np).permute(0, 3, 1, 2).to(torch.float32)
            y_tensor = torch.tensor(Y, dtype=torch.int64)
            batch_user_name = f"f_{batch_id:05d}"
            agg_data['users'][batch_id] = batch_user_name
            agg_data['num_samples'][batch_id] = len(Y)
            agg_data['user_data'][batch_user_name] = {'x': x_tensor, 'y': y_tensor}

    for idx, uname in enumerate(user_lists):
        user_data = cdata['user_data'][uname]
        load_per_user_(user_data, idx)

    if agg_user == -1:
        return new_data
    else:
        agg_by_user_(new_data, agg_n_users, agg_user)
        return agg_data

def process_data(args):
    load_path = os.path.join(args.load_path, 'data')
    train_dir = os.path.join(load_path, 'train')
    test_dir = os.path.join(load_path, 'test')
    train_jsons = sorted([f for f in os.listdir(train_dir) if f.endswith('.json')])
    test_jsons = sorted([f for f in os.listdir(test_dir) if f.endswith('.json')])
    assert train_jsons, f"No JSON files found in {train_dir}"
    assert test_jsons, f"No JSON files found in {test_dir}"
    img_dir = resolve_img_dir(args.load_path)

    def sample_users(cdata, ratio=0.1, excludes=set()):
        user_lists = [u for u in cdata['users']]
        if ratio <= 1:
            n_selected_users = int(len(user_lists) * ratio)
        else:
            n_selected_users = int(ratio)
        random.shuffle(user_lists)
        new_users, i = [], 0
        for u in user_lists:
            if u not in excludes:
                new_users.append(u)
                i += 1
            if i == n_selected_users:
                return new_users
        return new_users

    def process_(mode, tf, ratio=0.1, user_lists=None, agg_user=-1):
        read_path = os.path.join(load_path, mode if mode != 'proxy' else 'train', tf)
        with open(read_path, 'r') as inf:
            cdata = json.load(inf)
        n_users = len(cdata['users'])
        if ratio > 1:
            assert ratio < n_users
        else:
            assert ratio < 1
        verify_sample_path(cdata, img_dir)
        if mode == 'train' or user_lists is None:
            user_lists = sample_users(cdata, ratio)
        else:
            users_in_split = set(cdata['users'])
            filtered = [u for u in user_lists if u in users_in_split]
            if not filtered:
                filtered = sample_users(cdata, ratio)
            user_lists = filtered
        new_data = load_data(user_lists, cdata, img_dir, agg_user=agg_user)
        if ratio > 1:
            nu = int(ratio)
            if agg_user > 0:
                nu = int(nu // agg_user)
        else:
            nu = int(len(cdata['users']) * ratio)
        if agg_user > 0:
            dump_root = os.path.join(args.dump_path, f"user{nu}-agg{agg_user}")
        else:
            dump_root = os.path.join(args.dump_path, f"user{nu}")
        os.makedirs(os.path.join(dump_root, mode), exist_ok=True)
        dump_path = os.path.join(dump_root, f"{mode}/{mode}.pt")
        with open(dump_path, 'wb') as outfile:
            torch.save(new_data, outfile)
        return user_lists

    mode = 'train'
    tf = train_jsons[0]
    user_lists = process_(mode, tf, ratio=args.ratio, agg_user=args.agg_user)
    mode = 'test'
    tf = test_jsons[0]
    process_(mode, tf, ratio=args.ratio, user_lists=user_lists, agg_user=args.agg_user)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg_user", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=250)
    parser.add_argument("--load_path", type=str, default="./celeba")
    parser.add_argument("--dump_path", type=str, default="./")
    args = parser.parse_args()
    print(f"Number of FL devices: {int(args.ratio // args.agg_user)}")
    process_data(args)
