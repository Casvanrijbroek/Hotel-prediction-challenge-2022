import os
import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm
from joblib import Parallel, delayed

PAD = True
WIDTH = 256
HEIGHT = 256

data_folder = "/kaggle/input/hotel-id-to-combat-human-trafficking-2022-fgvc9/"
train_folder = os.path.join(data_folder, 'train_images')
chain_names = os.listdir(train_folder)

print(os.listdir(data_folder))
print(len(chain_names))

train_df = pd.DataFrame(columns={'image_id', 'hotel_id'})
for hotel_id in tqdm(chain_names):
    for image_id in os.listdir(os.path.join(train_folder, hotel_id)):
        train_df = train_df.append({'image_id': image_id, 'hotel_id': hotel_id}, ignore_index=True)

def pad_image(img):
    w, h, c = np.shape(img)
    if w > h:
        pad = int((w - h) / 2)
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = int((h - w) / 2)
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        
    return img


def open_and_preprocess_image(image_folder, image_name):
    img = cv2.imread(os.path.join(image_folder, image_name))
    
    if PAD:
        img = pad_image(img)
    
    return cv2.resize(img, (WIDTH, HEIGHT))


def save_image(image_name, img):
    cv2.imwrite(image_name, img)
    
    
def process_chain(data_folder, chain_name):
    chain_folder = os.path.join(data_folder, chain_name)
    
    for image_name in os.listdir(chain_folder):
        img = open_and_preprocess_image(chain_folder, image_name)
        save_image(image_name, img)

%%time
dfs_proc = Parallel(n_jobs=4, prefer='threads')(delayed(process_chain)(train_folder, chain_names[i]) for i in range(0, len(chain_names)))

!cd /kaggle/working/ & zip -jqr images.zip .
!find . -name "*.jpg" -delete
train_df.to_csv('train.csv', index=False)
