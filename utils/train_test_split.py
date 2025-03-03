import os
import random
import pandas as pd

# Define paths
# train_file = "../data_files/SymDerm_v2_train.csv"
# input_file = "../data_files/SymDerm_v2_labels_binary.csv"
# test_file = "../data_files/SymDerm_v2_test.csv"
# label_column  = 'labels_symmetry'

train_file = "../data_files/HAM10000_metadata_train.csv"
input_file = "../data_files/HAM10000_metadata.csv"
test_file = "../data_files/HAM10000_metadata_test.csv"
label_column  = 'dx'

# val_file = "../data_files/SymDerm_v2_val.csv"

# # Read image paths from the input file
# with open(input_file, 'r') as file:
#     image_paths = file.readlines()

# # Shuffle the image paths randomly
# random.shuffle(image_paths)

# # Calculate split sizes
# total_images = len(image_paths)
# train_size = int(0.8 * total_images)
# test_size = int(0.1 * total_images)

# # Split the data
# train_data = image_paths[:train_size]
# test_data = image_paths[train_size:train_size+test_size]
# val_data = image_paths[train_size+test_size:]

# # Write to train file
# with open(train_file, 'w') as file:
#     file.writelines(train_data)

# # Write to test file
# with open(test_file, 'w') as file:
#     file.writelines(test_data)

# # Write to validation file
# with open(val_file, 'w') as file:
#     file.writelines(val_data)

import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the CSV file is named 'data.csv' and is located in the same directory as this script.
# You can change the file path and name as per your requirements.
# file_path = 'data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Split the DataFrame into training and testing sets first (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=42)

# Further split the training set into training and validation sets (75% train, 25% val of the remaining 80%)
# train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# Save the split DataFrames to new CSV files
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)
# val_df.to_csv(val_file, index=False)




print("Train/Test/Val split created successfully.")
