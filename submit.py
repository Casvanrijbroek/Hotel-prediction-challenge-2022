"""
Script for submitting models for the Kaggle hotel challenge.
Credit to Michal for publishing a backbone for the challenge.
"""

import numpy as np
import pandas as pd
import random
import os
import math

from PIL import Image as pil_image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import cosine_similarity

from transformers import SwinForImageClassification, SwinConfig

import shutil
import zipfile

import albumentations as A
import albumentations.pytorch as APT
import cv2

if not os.path.exists('data'):
    os.mkdir('data')
    shutil.copy('../input/hotel-id-image-preprocessing-256x256/train.csv', 'data/train.csv')

    with zipfile.ZipFile('../input/hotel-id-image-preprocessing-256x256/images.zip', 'r') as zip_ref:
        zip_ref.extractall('data/images')

SEED = 42
IMG_SIZE = 256
N_MATCHES = 5

PROJECT_FOLDER = "../input/hotel-id-to-combat-human-trafficking-2022-fgvc9/"
TRAIN_DATA_FOLDER = "data/images"
TEST_DATA_FOLDER = PROJECT_FOLDER + "test_images/"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

base_transform = A.Compose([
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])


def pad_image(img):
    w, h, c = np.shape(img)
    if w > h:
        pad = int((w - h) / 2)
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = int((h - w) / 2)
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    return img


def open_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pad_image(img)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


class HotelImageDataset:
    def __init__(self, data, transform=None, data_folder="train_images/"):
        self.data = data
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        image_path = self.data_folder + record["image_id"]

        image = np.array(open_and_preprocess_image(image_path)).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return {
            "image": image,
        }


class EmbeddingModel(nn.Module):
    def __init__(self, n_classes=100, embedding_size=64, backbone_name="microsoft/swin-tiny-patch4-window7-224"):
        super(EmbeddingModel, self).__init__()

        config = SwinConfig.from_pretrained(backbone_name)
        config.output_hidden_states = True
        config.hidden_size = 768
        in_features = config.hidden_size

        self.backbone = SwinForImageClassification.from_pretrained(backbone_name,
                                                                   config=config)
        self.pooler = nn.AdaptiveAvgPool1d(output_size=1)

        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.classifier = nn.Linear(embedding_size, n_classes)

    def embed_and_classify(self, x):
        x = self.forward(x)
        return x, self.classifier(x)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooler(x.hidden_states[-1].transpose(-1, -2)).squeeze(-1)
        x = self.embedding(x)
        return x


class EmbeddingModelConv(nn.Module):
    def __init__(self, n_classes=100, embedding_size=64, backbone_name="efficientnet_b0"):
        super(EmbeddingModelConv, self).__init__()

        self.backbone = timm.create_model(backbone_name, num_classes=n_classes, pretrained=True)

        self.backbone.classifier = nn.Identity()

        self.embedding = nn.Linear(n_classes, embedding_size)
        self.classifier = nn.Linear(embedding_size, n_classes)

    def embed_and_classify(self, x):
        x = self.forward(x)
        return x, self.classifier(x)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x


def generate_embeddings(args, loader, model, bar_desc="Generating embeds"):
    outputs_all = []

    model.eval()
    with torch.no_grad():
        t = tqdm(loader, desc=bar_desc)
        for i, sample in enumerate(t):
            input = sample['image'].to(args.device)
            output = model(input)
            print(output)
            outputs_all.extend(output.detach().cpu().numpy())

    return outputs_all


def find_matches(query, base_embeds, base_targets, k=N_MATCHES):
    distance_df = pd.DataFrame(index=np.arange(len(base_targets)), data={"hotel_id": base_targets})
    # calculate cosine distance of query embeds to all base embeds
    distance_df["distance"] = cosine_similarity([query], list(base_embeds))[0]
    print(distance_df)
    # sort by distance and hotel_id
    distance_df = distance_df.sort_values(by=["distance", "hotel_id"], ascending=False).reset_index(drop=True)
    # return first 5 different hotel_id_codes
    return distance_df["hotel_id"].unique()[:N_MATCHES]


def predict(args, base_embeddings_df, test_loader, model):
    test_embeds = generate_embeddings(args, test_loader, model, "Generate test embeddings")

    preds = []
    for query_embeds in tqdm(test_embeds, desc="Similarity - match finding"):
        tmp = find_matches(query_embeds,
                           base_embeddings_df["embeddings"].values,
                           base_embeddings_df["hotel_id"].values)
        preds.extend([tmp])

    return preds


test_df = pd.DataFrame(data={"image_id": os.listdir(TEST_DATA_FOLDER), "hotel_id": ""}).sort_values(by="image_id")


def get_model(backbone_name, checkpoint_path, args):
    model = EmbeddingModel(args.n_classes, args.embedding_size, backbone_name)

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    model = model.to(args.device)

    return model


class args:
    batch_size = 64
    num_workers = 2
    embedding_size = 128
    device = ('cuda' if torch.cuda.is_available() else 'cpu')


seed_everything(seed=SEED)

test_dataset = HotelImageDataset(test_df, base_transform, data_folder=TEST_DATA_FOLDER)
test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

TRAINING_OUT_FOLDER = "../input/hotel-id-train-swin-256"
CHECKPOINT_FILE = "checkpoint-embedding-model-microsoft-swin-tiny-patch4-window7-224-256x256-last.pt"
PICKLE_FILE = "embedding-model-microsoft-swin-tiny-patch4-window7-224-256x256_image-embeddings-last.pkl"

base_embeddings_df = pd.read_pickle(f'{TRAINING_OUT_FOLDER}/{PICKLE_FILE}')
# display(base_embeddings_df.head()['embeddings'][0])
display(base_embeddings_df.head())

args.n_classes = base_embeddings_df["hotel_id"].nunique()

model = get_model("swin",
                  f'{TRAINING_OUT_FOLDER}/{CHECKPOINT_FILE}',
                  args)

preds = predict(args, base_embeddings_df, test_loader, model)



# transform array of hotel_ids into string
test_df["hotel_id"] = [str(list(l)).strip("[]").replace(",", "") for l in preds]

test_df.to_csv("submission.csv", index=False)
test_df.head()

shutil.rmtree('data')

