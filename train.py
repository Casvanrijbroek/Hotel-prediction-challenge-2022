"""
Script for training models for the Kaggle hotel challenge.
Credit to Michal for publishing a backbone for the challenge.
"""


import numpy as np
import pandas as pd
import random
import os
import zipfile
import shutil
from PIL import Image as pil_image
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm

from sklearn.metrics.pairwise import cosine_similarity
from transformers import SwinForImageClassification, SwinConfig

import albumentations as A
import albumentations.pytorch as APT
import cv2

"""
Loads in the preprocessed data if it doesn't exist yet.
Use 224x224 for ViT
"""
if not os.path.exists('data'):
    os.mkdir('data')
    shutil.copy('../input/hotel-id-image-preprocessing-256x256/train.csv', 'data/train.csv')

    with zipfile.ZipFile('../input/hotel-id-image-preprocessing-256x256/images.zip', 'r') as zip_ref:
        zip_ref.extractall('data/images')

IMG_SIZE = 256
SEED = 42
N_MATCHES = 5

PROJECT_FOLDER = "../input/hotel-id-to-combat-human-trafficking-2022-fgvc9/"
DATA_FOLDER = "data/"
IMAGE_FOLDER = DATA_FOLDER + "images/"
OUTPUT_FOLDER = ""

train_df = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# used for training dataset - augmentations and occlusions
train_transform = A.Compose([
    A.HorizontalFlip(p=0.75),
    A.VerticalFlip(p=0.25),
    A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.OpticalDistortion(p=0.25),
    A.Perspective(p=0.25),
    A.CoarseDropout(p=0.5, min_holes=1, max_holes=6,
                    min_height=IMG_SIZE // 16, max_height=IMG_SIZE // 4,
                    min_width=IMG_SIZE // 16, max_width=IMG_SIZE // 4),  # normal coarse dropout

    A.CoarseDropout(p=1., max_holes=1,
                    min_height=IMG_SIZE // 4, max_height=IMG_SIZE // 2,
                    min_width=IMG_SIZE // 4, max_width=IMG_SIZE // 2,
                    fill_value=(255, 0, 0)),  # simulating occlusions in test data

    A.RandomBrightnessContrast(p=0.75),
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])

# used for validation dataset - only occlusions
val_transform = A.Compose([
    A.CoarseDropout(p=1., max_holes=1,
                    min_height=IMG_SIZE // 4, max_height=IMG_SIZE // 2,
                    min_width=IMG_SIZE // 4, max_width=IMG_SIZE // 2,
                    fill_value=(255, 0, 0)),  # simulating occlusions
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])

# no augmentations
base_transform = A.Compose([
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])


class HotelTrainDataset:
    def __init__(self, data, transform=None, data_path="train_images/"):
        self.data = data
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        image_path = self.data_path + record["image_id"]
        image = np.array(pil_image.open(image_path)).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return {
            "image": image,
            "target": record['hotel_id_code'],
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


# method to iterate loader and generate embeddings of images
# returns embeddings and image class
def generate_embeddings(loader, model, bar_desc="Generating embeds"):
    targets_all = []
    outputs_all = []

    model.eval()
    with torch.no_grad():
        t = tqdm(loader, desc=bar_desc)
        for i, sample in enumerate(t):
            input = sample['image'].to(args.device)
            target = sample['target'].to(args.device)
            output = model(input)

            targets_all.extend(target.cpu().numpy())
            outputs_all.extend(output.detach().cpu().numpy())

    targets_all = np.array(targets_all).astype(np.float32)
    outputs_all = np.array(outputs_all).astype(np.float32)

    return outputs_all, targets_all


def save_checkpoint(model, scheduler, optimizer, epoch, name, loss=None, score=0, best_map5=None, last_epoch = False):
    if score > best_map5:
        checkpoint = {"epoch": epoch,
                      "model": model.state_dict(),
                      "scheduler": scheduler.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "loss": loss,
                      "score": score,
                      }

        torch.save(checkpoint, f"{OUTPUT_FOLDER}checkpoint-{name}-best.pt")
        best_map5 = score
    if last_epoch:
        checkpoint = {"epoch": epoch,
                      "model": model.state_dict(),
                      "scheduler": scheduler.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "loss": loss,
                      "score": score,
                      }

        torch.save(checkpoint, f"{OUTPUT_FOLDER}checkpoint-{name}-last.pt")
    return best_map5


def load_checkpoint(model, scheduler, optimizer, name):
    checkpoint = torch.load(f"{OUTPUT_FOLDER}checkpoint-{name}-best.pt")

    model.load_state_dict(checkpoint["model"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, scheduler, optimizer, checkpoint["epoch"]


def get_model(backbone_name, checkpoint_path, args):
    model = EmbeddingModel(args.n_classes, args.embedding_size, backbone_name)

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    model = model.to(args.device)

    return model


def train_epoch(args, model, loader, criterion, optimizer, scheduler, epoch):
    losses = []
    targets_all = []
    outputs_all = []

    model.train()
    t = tqdm(loader)

    for i, sample in enumerate(t):
        optimizer.zero_grad()

        images = sample['image'].to(args.device)
        targets = sample['target'].to(args.device)

        _, outputs = model.embed_and_classify(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        losses.append(loss.item())
        targets_all.extend(targets.cpu().numpy())
        outputs_all.extend(torch.sigmoid(outputs).detach().cpu().numpy())

        score = np.mean(targets_all == np.argmax(outputs_all, axis=1))
        desc = f"Training epoch {epoch}/{args.epochs} - loss:{loss:0.4f}, accuracy: {score:0.4f}"
        t.set_description(desc)

    return np.mean(losses), score


def test_classification(loader, model):
    targets_all = []
    outputs_all = []

    model.eval()
    t = tqdm(loader, desc="Classification")

    for i, sample in enumerate(t):
        images = sample['image'].to(args.device)
        targets = sample['target'].to(args.device)

        _, outputs = model.embed_and_classify(images)

        targets_all.extend(targets.cpu().numpy())
        outputs_all.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    # repeat targets to N_MATCHES for easy calculation of MAP@5
    y = np.repeat([targets_all], repeats=N_MATCHES, axis=0).T
    # sort predictions and get top 5
    preds = np.argsort(-np.array(outputs_all), axis=1)[:, :N_MATCHES]
    # check if any of top 5 predictions are correct and calculate mean accuracy
    acc_top_5 = (preds == y).any(axis=1).mean()
    # calculate prediction accuracy
    acc_top_1 = np.mean(targets_all == np.argmax(outputs_all, axis=1))

    print(f"Classification accuracy: {acc_top_1:0.4f}, MAP@5: {acc_top_5:0.4f}")
    return acc_top_5


# find 5 most similar images from different hotels and return their hotel_id_code
def find_matches(query, base_embeds, base_targets, k=N_MATCHES):
    distance_df = pd.DataFrame(index=np.arange(len(base_targets)), data={"hotel_id_code": base_targets})
    # calculate cosine distance of query embeds to all base embeds
    distance_df["distance"] = cosine_similarity([query], base_embeds)[0]
    # sort by distance and hotel_id
    distance_df = distance_df.sort_values(by=["distance", "hotel_id_code"], ascending=False).reset_index(drop=True)
    # return first 5 different hotel_id_codes
    return distance_df["hotel_id_code"].unique()[:N_MATCHES]


def test_similarity(args, base_loader, test_loader, model):
    base_embeds, base_targets = generate_embeddings(base_loader, model, "Generate base embeddings")
    test_embeds, test_targets = generate_embeddings(test_loader, model, "Generate test embeddings")

    preds = []
    for query_embeds in tqdm(test_embeds, desc="Similarity - match finding"):
        tmp = find_matches(query_embeds, base_embeds, base_targets)
        preds.extend([tmp])

    preds = np.array(preds)
    test_targets_N = np.repeat([test_targets], repeats=N_MATCHES, axis=0).T
    # check if any of top 5 predictions are correct and calculate mean accuracy
    acc_top_5 = (preds == test_targets_N).any(axis=1).mean()
    # calculate prediction accuracy
    acc_top_1 = np.mean(test_targets == preds[:, 0])
    print(f"Similarity accuracy: {acc_top_1:0.4f}, MAP@5: {acc_top_5:0.4f}")


data_df = pd.read_csv(DATA_FOLDER + "train.csv")
# encode hotel ids
data_df["hotel_id_code"] = data_df["hotel_id"].astype('category').cat.codes.values.astype(np.int64)

# save hotel_id encoding for later decoding
hotel_id_code_df = data_df.drop(columns=["image_id"]).drop_duplicates().reset_index(drop=True)
hotel_id_code_df.to_csv(OUTPUT_FOLDER + 'hotel_id_code_mapping.csv', index=False)
# hotel_id_code_map = hotel_id_code_df.set_index('hotel_id_code').to_dict()["hotel_id"]


def show_images(ds, title_text, n_images=5):
    fig, ax = plt.subplots(1, 5, figsize=(22, 8))

    ax[0].set_ylabel(title_text)

    for i in range(5):
        d = ds.__getitem__(i)
        ax[i].imshow(d["image"].T)


"""
With or without agumentations
"""
# train_dataset = HotelTrainDataset(data_df, base_transform, data_path=IMAGE_FOLDER)
# show_images(train_dataset, 'No augmentations')

train_dataset = HotelTrainDataset(data_df, train_transform, data_path=IMAGE_FOLDER)
show_images(train_dataset, 'Train augmentations')

test_image = np.array(pil_image.open('../input/hotel-id-to-combat-human-trafficking-2022-fgvc9/test_images/abc.jpg')).astype(np.uint8)
plt.figure(figsize=(6,6))
plt.imshow(test_image)

TRAINING_OUT_FOLDER = "../input/hotel-id-train-swin-256"
CHECKPOINT_FILE = "checkpoint-embedding-model-microsoft-swin-tiny-patch4-window7-224-256x256-last.pt"


def train_and_validate(args, data_df):
    model_name = f"embedding-model-{args.backbone_name}-{IMG_SIZE}x{IMG_SIZE}".replace('/', '-')
    print(model_name)

    seed_everything(seed=SEED)

    # split data into train and validation set
    hotel_image_count = data_df.groupby("hotel_id")["image_id"].count()
    # hotels that have more images than samples for validation
    valid_hotels = hotel_image_count[hotel_image_count > args.val_samples]
    # data that can be split into train and val set
    valid_data = data_df[data_df["hotel_id"].isin(valid_hotels.index)]
    # if hotel had less than required val_samples it will be only in the train set
    valid_df = valid_data.groupby("hotel_id").sample(args.val_samples, random_state=SEED)
    train_df = data_df[~data_df["image_id"].isin(valid_df["image_id"])]

    train_dataset = HotelTrainDataset(train_df, train_transform, data_path=IMAGE_FOLDER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    valid_dataset = HotelTrainDataset(valid_df, val_transform, data_path=IMAGE_FOLDER)
    valid_loader = DataLoader(valid_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    # base dataset for image similarity search
    base_dataset = HotelTrainDataset(train_df, base_transform, data_path=IMAGE_FOLDER)
    base_loader = DataLoader(base_dataset, num_workers=args.num_workers, batch_size=args.batch_size * 4, shuffle=False)

    # model = EmbeddingModel(args.n_classes, args.embedding_size ,args.backbone_name)
    model = get_model("swin",
                      f'{TRAINING_OUT_FOLDER}/{CHECKPOINT_FILE}',
                      args)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        div_factor=10,
        final_div_factor=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    start_epoch = 1
    best_map5 = 0
    map5 = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_score = train_epoch(args, model, train_loader, criterion, optimizer, scheduler, epoch)
        map5 = test_classification(valid_loader, model)
        best_map5 = save_checkpoint(model, scheduler, optimizer, epoch, model_name, train_loss, map5, best_map5)

    print(map5)
    print(best_map5)
    save_checkpoint(model, scheduler, optimizer, epoch, model_name, train_loss, score=map5, best_map5=best_map5,
                    last_epoch=True)
    test_similarity(args, base_loader, valid_loader, model)

    # generate embeddings for all train images and save them for inference
    base_dataset = HotelTrainDataset(data_df, base_transform, data_path=IMAGE_FOLDER)
    base_loader = DataLoader(base_dataset, num_workers=args.num_workers, batch_size=args.batch_size * 4, shuffle=False)
    base_embeds, _ = generate_embeddings(base_loader, model, "Generate embeddings for all images")
    data_df["embeddings"] = list(base_embeds)
    data_df.to_pickle(f"{OUTPUT_FOLDER}{model_name}_image-embeddings-best.pkl")

    model, scheduler, optimizer, epoch = load_checkpoint(model, scheduler, optimizer, model_name)

    # generate embeddings for all train images and save them for inference
    base_dataset = HotelTrainDataset(data_df, base_transform, data_path=IMAGE_FOLDER)
    base_loader = DataLoader(base_dataset, num_workers=args.num_workers, batch_size=args.batch_size * 4, shuffle=False)
    base_embeds, _ = generate_embeddings(base_loader, model, "Generate embeddings for all images")
    data_df["embeddings"] = list(base_embeds)
    data_df.to_pickle(f"{OUTPUT_FOLDER}{model_name}_image-embeddings-last.pkl")


class args:
    epochs = 15
    lr = 1e-3
    batch_size = 32
    num_workers = 2
    val_samples = 1
    embedding_size = 128
    backbone_name = "microsoft/swin-tiny-patch4-window7-224"
    n_classes = data_df["hotel_id_code"].nunique()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

train_and_validate(args, data_df)
