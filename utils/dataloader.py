import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

categories = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 2,
    'Infiltration': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
    'Consolidation': 8,
    'Edema': 9,
    'Emphysema': 10,
    'Fibrosis': 11,
    'Pleural_Thickening': 12,
    'Hernia': 13,
}

augmentation = {
    "train": {
        'rescale': 1./255,
    },
    "val": {
        'rescale': 1./255,
    },
    "test": {
        'rescale': 1./255,
    },
}

DATA_DIR = 'CXR8/'
SEED = 42


def get_generator(batch_size, image_size, dtype='train'):
    if dtype not in ['train', 'val', 'test']:
        raise ValueError("No such type, must be 'train', 'val', or 'test'")
    df = pd.read_csv(os.path.join(DATA_DIR, dtype + '_label.csv'))
    index_dir = os.path.join(DATA_DIR, dtype + '_label.csv')
    datagen = ImageDataGenerator(**augmentation.get(dtype))
    generator = datagen.flow_from_dataframe(
        df,
        directory=os.path.join(DATA_DIR, 'images'),
        x_col='FileName',
        y_col=categories.keys(),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True if dtype == 'train' else False,
        seed=SEED,
    )
    return generator

if __name__ == "__main__":
    generator = get_generator(32, (1024, 1024), dtype='test')
    for i in range(10):
        print(generator.next()[0].shape)
