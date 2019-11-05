# Plan here is to take our data sets and put them NumPy arrays

from pathlib import Path

rootdir = Path('data/10s_website1/')
file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

for f in file_list:
    print(f)


# For the numpy array, we might need to make it multi-dimensional
# We need to decide if we want to have an entire array for all websites or 
# to create on array per site and remove it after to save on storage
