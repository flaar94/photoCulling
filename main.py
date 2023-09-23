from PIL import Image, ExifTags
from pathlib import Path
import itertools
import datetime
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import copy
from model.model import NIMA
from collections.abc import Iterable
import argparse
import pandas as pd
import logging
import sys


class TypedNamespace(argparse.Namespace):
    image_path: str
    model_path: str
    predictions: str
    score_only: bool
    similarity_threshold: float
    time_threshold: int
    silence: bool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='path to file or folder containing images')
    parser.add_argument('--model_path', type=str, default="model_weights/filtered_v2_epoch-40.pth",
                        help='path to pretrained model')
    parser.add_argument('--prediction_path', type=str, default="predictions",
                        help='output directory to store predictions')
    parser.add_argument('--score_only', action='store_true',
                        help='Whether to skip the culling step, and instead only score the images')
    parser.add_argument('--similarity_threshold', type=float, default=0.3,
                        help='Threshold for how similar two items need to be in order to group them ')
    parser.add_argument("--time_threshold", type=int, default=10,
                        help='In order to group two photos, the number of minutes between them must be less than this '
                             'number')
    parser.add_argument("--silence", action='store_true',
                        help='Whether to print out the status of the algorithm as it is running')
    return parser


def score_paths(model: NIMA, image_paths: Iterable[Path]) -> dict[Path, float]:
    """
    Takes a collection of paths, and computes the predicted aesthetic score for each one, returning the predicted best
    image along with predicted scores for all images

    Args:
        model: A model which computes quality scores for images
        image_paths: A collection of paths to images

    Returns: The path to the best image, and a dictionary mapping image paths to associated scores

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    scores = []
    for i, img in enumerate(image_paths):
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
        scores.append((img, mean))

    return dict(scores)


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


def group_by_features(data: Iterable[tuple[Path, datetime.datetime, torch.tensor]], time_threshold,
                      similarity_threshold) -> set[frozenset[Path]]:
    """
    Determine pairs of items that are similar in time and VGG features

    Args:
        data: triples of path, datetime, and (base) model features
        time_threshold: how many minutes between pictures is allowed for two photos to be grouped
        similarity_threshold: the minimum cosine similarity between photos' base-model features for them to be grouped

    Returns: Set of groups of images that are similar in time and features

    """
    similar_images = []
    all_connected_images = set()
    for (image_path1, datetime1, features1), (image_path2, datetime2, features2) in itertools.combinations(data, 2):
        if abs(datetime1 - datetime2) <= datetime.timedelta(minutes=time_threshold) and F.cosine_similarity(
                features1.view(1, -1), features2.view(1, -1)).cpu().data >= similarity_threshold:
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


def get_evaluation_model(weight_path: Path | str) -> NIMA:
    """
    Sets up the model used to evaluate photo quality

    :param weight_path: path to the filed containing the weights for the pretrained model
    :return: the NIMA model with loaded weights
    """
    base_model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    model = NIMA(base_model)

    try:
        model.load_state_dict(torch.load(weight_path))
        logging.info('successfully loaded evaluation model')
    except OSError:
        raise OSError("Could not load state dictionary, are you sure your path is correct?")

    seed = 42
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.eval()

    return model


def extract_top_scored(group: Iterable[Path], scores: dict[Path, float]) -> Path:
    return max(group, key=lambda path: scores[path])


def main():
    parser = get_parser()
    args = parser.parse_args(namespace=TypedNamespace())

    logging_level = logging.WARNING if args.silence else logging.INFO
    logging.basicConfig(level=logging_level, stream=sys.stdout, format="%(message)s")
    logging.info(args)

    image_dir = Path(args.image_path)
    image_paths = list(itertools.chain(image_dir.glob("*.jpg"),
                                       image_dir.glob("*.jpeg"),
                                       image_dir.glob("*.png")))

    model = get_evaluation_model(weight_path=args.model_path)
    scores = score_paths(model, image_paths)
    prediction_path = Path(args.prediction_path)

    if not prediction_path.exists():
        prediction_path.mkdir(parents=True)
    pd.DataFrame(scores.items(), columns=["path", "score"]).to_csv(prediction_path / "scores.csv", index=False)
    logging.info(f"Image scores saved to {prediction_path / 'scores.csv'}")

    if not args.score_only:
        path_date_features = extract_features(image_paths)
        logging.info("Image features extracted, now grouping images")
        groups = group_by_features(path_date_features,
                                   time_threshold=args.time_threshold, similarity_threshold=args.similarity_threshold)
        logging.info("Images successfully grouped")
        culled_unflattened = [group - {extract_top_scored(group, scores)} for group in groups]
        culled_images = frozenset.union(*culled_unflattened)
        # We've restricted to groups of images, so we need to put back in the ungrouped images
        kept_images = set(image_paths) - culled_images

        with open(prediction_path / "kept_images.txt", "w") as f:
            f.write("\n".join([kept_image.name for kept_image in kept_images]))
        logging.info(f"Kept image list saved to {prediction_path / 'kept_images.txt'}")

        with open(prediction_path / "culled_images.txt", "w") as f:
            f.write("\n".join([culled_image.name for culled_image in culled_images]))
        logging.info(f"Culled image list saved to {prediction_path / 'culled_images.txt'}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
