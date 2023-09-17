from PIL import Image, ExifTags
from pathlib import Path
import itertools
import datetime
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import copy
from model.model import NIMA
from collections.abc import Iterable
import argparse

MODEL = "model_weights/filtered_v2_epoch-15.pth"

SIMILARITY_THRESHOLD = 0.3
TIME_THRESHOLD_MIN = 60


class TypedNamespace(argparse.Namespace):
    image_path: str
    model: str
    predictions: str


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='path to file or folder containing images')
    parser.add_argument('--model', type=str, default="model_weights/filtered_v2_epoch-15.pth",
                        help='path to pretrained model')
    parser.add_argument('--predictions', type=str, default="predictions", help='output file to store predictions')
    return parser


def score_group(model: NIMA, group: Iterable[Path]) -> tuple[Path, dict[Path, float]]:
    """
    Takes a collection of paths, and computes the predicted aesthetic score for each one, returning the predicted best
    image along with predicted scores for all images

    Args:
        group: A collection of paths to images

    Returns: The path to the best image, and a dictionary mapping image paths to associated scores

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    scores = []
    for i, img in enumerate(group):
        mean, std = 0.0, 0.0
        im = Image.open(img)
        im = im.convert('RGB')
        imt = test_transform(im)
        imt = imt.unsqueeze(dim=0)
        imt = imt.to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            out = model(imt)
        out = out.view(10, 1)
        for j, e in enumerate(out, 1):
            mean += j * e
        for k, e in enumerate(out, 1):
            std += e * (k - mean) ** 2
        std = std ** 0.5

        mean, std = float(mean), float(std)
        msg = f'{img.stem} mean: {mean:.3f} | std: {std:.3f} \n'
        scores.append((img, mean))

    # sort from highest to lowest
    scores.sort(key=lambda x: -x[1])

    # Find highest scored images, then take one of them
    # top_score = max(scores.values())
    # top_images = [img for img, val in scores.items() if val == top_score]
    return scores[0][0], dict(scores)


def extract_features(image_paths: Iterable[Path]) -> list[tuple[Path, datetime.datetime, torch.tensor]]:
    """
    Takes paths, and extracts the time it was taken via metadata and also computes the features using the base model

    Args:
        image_paths: the paths to the desired images to look at

    Returns: A list of triples of the (original) paths, datetime taken, and base model features

    """
    feature_model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')

    seed = 42
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = feature_model.to(device)

    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Grab metadata and compute VGG features
    data = []
    for image_path in image_paths:
        img = Image.open(image_path)
        exif = {ExifTags.TAGS[k]: v for k, v in img.getexif().items() if k in ExifTags.TAGS}
        picture_time = datetime.datetime.strptime(exif["DateTime"],
                                                  "%Y:%m:%d %H:%M:%S") if "DateTime" in exif else datetime.datetime(1,
                                                                                                                    1,
                                                                                                                    1)
        im = Image.open(image_path)
        im = im.convert('RGB')
        imt = test_transform(im)
        imt = imt.unsqueeze(dim=0)
        imt = imt.to(device)
        features = model.features(imt)

        data.append((image_path, picture_time, features))
    return data


def group_by_features(data: Iterable[tuple[Path, datetime.datetime, torch.tensor]]) -> set[frozenset[Path]]:
    """
    Determine pairs of items that are similar in time and VGG features

    Args:
        data: triples of path, datetime, and (base) model features

    Returns: Set of groups of images that are similar in time and features

    """
    similar_images = []
    all_connected_images = set()
    for (image_path1, datetime1, features1), (image_path2, datetime2, features2) in itertools.combinations(data, 2):
        if abs(datetime1 - datetime2) < datetime.timedelta(minutes=30) and F.cosine_similarity(
                features1.view(1, -1), features2.view(1, -1)).cpu().data > 0.3:
            similar_images.append([image_path1, image_path2])
            all_connected_images |= {image_path1, image_path2}

    # Turning pairs of similar items into full groups. Too tired to figure out the right way of doing this
    graph = {img: {img_pair[0] if img_pair[1] == img else img_pair[1] for img_pair in similar_images if img in img_pair}
             for img in all_connected_images}
    old_graph = {}
    while graph != old_graph:
        old_graph = copy.deepcopy(graph)
        graph = {img: set.union(*[graph[img2] for img2 in connected_imgs]) for img, connected_imgs in graph.items()}

    #
    groups = {frozenset(connected_imgs) for connected_imgs in graph.values()}
    return groups


def get_evaluation_model(weight_path):
    base_model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    model = NIMA(base_model)

    try:
        model.load_state_dict(torch.load(weight_path))
        print('successfully loaded evaluation model')
    except:
        raise RuntimeError("Could not load state dictionary, are you sure your path is correct?")

    seed = 42
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.eval()

    return model


def main():
    parser = get_parser()
    args = parser.parse_args(namespace=TypedNamespace())
    print(args)

    image_dir = Path(args.image_path)
    image_paths = list(itertools.chain(image_dir.glob("*.jpg"),
                                       image_dir.glob("*.jpeg"),
                                       image_dir.glob("*.png")))

    data = extract_features(image_paths)
    groups = group_by_features(data)

    model = get_evaluation_model(weight_path=MODEL)

    for group in groups:
        top_image, scores = score_group(model, group)
        print(f"top_image={top_image.name}, {scores=}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
