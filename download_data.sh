#! /bin/bash

echo "Delete old dataset (if exists)."
rm -rf data animefacedataset.zip

echo "Downloading dataset..."
kaggle datasets download -d splcher/animefacedataset

echo "Unzipping dataset..."
unzip animefacedataset.zip
mv images data

rm animefacedataset.zip

echo "Done !"
