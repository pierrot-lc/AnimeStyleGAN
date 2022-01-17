"""Some preparation for the project.
"""
import os
import shutil
from zipfile import ZipFile

from sklearn.model_selection import train_test_split


def extract_images(archive_path: str, dir_tgt: str):
    """Extract the zip archive and move the data to a specific directory.
    """
    # Remove old data if necessary
    if os.path.exists(dir_tgt):
        shutil.rmtree(dir_tgt)

    # Extract data
    with ZipFile(archive_path, 'r') as archive:
        print('Extracting...')
        archive.extractall('archive')

    # Move data to the wanted directory
    # Only select the subdir with all the images (others are duplicates)
    print(f'Moving to {dir_tgt} directory...')
    os.makedirs(dir_tgt)
    for filename in os.listdir('archive/data/data'):
        filepath = os.path.join('archive/data/data', filename)
        n_filepath = os.path.join(dir_tgt, filename)
        shutil.move(filepath, n_filepath)

    # Remove temporary archive
    shutil.rmtree('archive')


def split_images(
        dir_src: str,
        test_size: int = 0.1,
        seed: int = 0
    ):
    """Move all images to a specific training and testing dataset.
    """
    paths = [
        os.path.join(dir_src, p)
        for p in os.listdir(dir_src)
    ]

    paths_train, paths_test = train_test_split(
        paths,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    print('Splitting to train and test images...')
    for paths, name_dir in [(paths_train, 'train'), (paths_test, 'test')]:
        dir_tgt = os.path.join(dir_src, name_dir)

        if not os.path.exists(dir_tgt):
            os.makedirs(dir_tgt)

        for file in paths:
            filename = os.path.basename(file)
            shutil.move(file, os.path.join(dir_tgt, filename))


def init_project():
    extract_images('archive.zip', 'data')
    split_images('data')

    print('Creating models directory...')
    if not os.path.exists('models'):
        os.mkdir('models')

    print('Initialisation done!')


if __name__ == '__main__':
    init_project()
