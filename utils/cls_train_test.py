import os
import pandas as pd
from tqdm import tqdm

def split_data(txt_file, csv_file, outfile):
    with open(txt_file) as f:
        df = pd.read_csv(csv_file)
        file = f.readlines()
        new_df = pd.DataFrame()
        for line in tqdm(file):
            id = line.split()[0].split('/')[2].split('.')[0]
            row = df[df['image_id'] == id]
            new_df = pd.concat([new_df, row], ignore_index=True)
    new_df.to_csv(outfile, index=False)       

txt_file = '../data_files/HAM10000_seg_val.txt'
csv_file = '../data_files/HAM10000_metadata.csv'
outfile = '../data_files/HAM10000_metadata_val.csv'

split_data(txt_file, csv_file, outfile)