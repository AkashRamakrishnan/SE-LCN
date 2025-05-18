import os

# Specify the directory containing the files
directory = "../datasets/HAM10000_inpaint/"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if the file matches the pattern
    if filename.endswith(".jpg.png"):
        # Construct old and new file paths
        old_path = os.path.join(directory, filename)
        new_filename = filename.replace(".jpg.png", ".png")
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
