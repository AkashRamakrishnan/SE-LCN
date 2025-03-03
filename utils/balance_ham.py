import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.utils import resample
import random
from tqdm import tqdm

current_dir = '../../../data/HAM10000_inpaint/'
new_dir = '../../../data/HAM10000_balanced/'

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

def downsample(df, required_size):
    # Randomly select a subset of samples for a class
    if len(df) > required_size:
        df_downsampled = resample(df, replace=False, n_samples=required_size, random_state=123)
    else:
        df_downsampled = df
    for index, row in df_downsampled.iterrows():
        img_name = row['image_id'] + '.png'
        src_path = os.path.join(current_dir, img_name)
        dst_path = os.path.join(new_dir, img_name)
        if not os.path.exists(dst_path):
            os.link(src_path, dst_path)
    return df_downsampled

def upsample(df, required_size):
    # Use image augmentation to upsample data
    length = len(df)
    new_df = pd.DataFrame()
    # while len(df) < required_size:
    for index, row in df.iterrows():
        size = len(df)
        if len(df) >= required_size:
            break
        img_name = row['image_id'] + '.png'
        img_path = os.path.join(current_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} does not exist. Skipping...")
            continue
        img = Image.open(img_path)
        augmented_images = [img, img.transpose(Image.FLIP_LEFT_RIGHT), img.transpose(Image.FLIP_TOP_BOTTOM), img.rotate(90)]
        names = ['original', 'mirror', 'flip', 'rotate']
        for i in range(4):
        # for augmented_img in augmented_images:
            new_image_id = row['image_id'] + '_' + names[i]
            new_img_name = new_image_id + '.png'
            augmented_img_path = os.path.join(new_dir, new_img_name)
            augmented_images[i].save(augmented_img_path)
            new_row = row.copy()
            new_row['image_id'] = new_image_id
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    n_samples = min(required_size, len(new_df))
    new_df = new_df.sample(n=n_samples)
    new_df = new_df.reset_index(drop=True)
    return new_df


data_path = '../data_files/HAM10000_metadata_train.csv'
data = pd.read_csv(data_path)

counts = data['dx'].value_counts()
# min_count = counts.min()
min_count = 2000

tables = {}
for col in counts.index:
    print('Processing class', col)
    class_df = data[data['dx'] == col]
    if counts[col] > min_count:
        class_df = downsample(class_df, min_count)
    else:
        class_df = upsample(class_df, min_count)
    tables[col] = class_df

# Combine back to a single dataframe if necessary
balanced_data = pd.concat(tables.values(), ignore_index=True)

balanced_data.to_csv('../data_files/HAM10000_metadata_balanced.csv')
