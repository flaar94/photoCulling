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
from model.segment import MaskList, SegmentationScores
from collections.abc import Iterable
from collections import defaultdict
import argparse
import pandas as pd
import logging
import sys
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

IMAGE_DIMS = 224

class TypedNamespace(argparse.Namespace):
    image_path: str
    model_path: str
    predictions: str
    score_only: bool
    similarity_threshold: float
    time_threshold: int
    silence: bool
    model_mix: bool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='path to file or folder containing images')
    parser.add_argument('--model_path', type=str, default="model_weights/filtered_v4_epoch-35.pth",
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
                        help="Whether to prevent the script from printing out its updates as it's running")
    parser.add_argument("--model_mix", default="nima_only", choices=["nima_only", "uniform", "uniform_seg_only"],
                        help="Which models/scores to use to make predictions")
    parser.add_argument("--weights", default=None, required=False, nargs='+', type=float,
                        help="Explicit set of 5 space separated weights. Overrides model_mix. Eg: '0.1 1 1 1 2.5' Still"
                             " in development, so you might break something with cryptic errors if you provide an "
                             "invalid input.")
    return parser


def nima_score_paths(nima_model: NIMA, image_paths: Iterable[Path]) -> dict[Path, float]:
    """
    Takes a collection of paths, and computes the predicted aesthetic score for each one

    Args:
        nima_model: A model which computes quality scores for images
        image_paths: A collection of paths to images

    Returns: A dictionary mapping image paths to associated scores

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_DIMS, IMAGE_DIMS)),
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
            out = nima_model(imt)

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
        transforms.Resize((IMAGE_DIMS, IMAGE_DIMS)),
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
    model = NIMA(base_model, image_dims=IMAGE_DIMS)

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

def segmentation_score_paths(image_paths: Iterable[Path]) -> dict[Path, SegmentationScores]:
    segmentation_weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    segmentation_model = maskrcnn_resnet50_fpn_v2(weights=segmentation_weights)
    segmentation_transforms = segmentation_weights.transforms()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmentation_model = segmentation_model.eval().to(device)
    scores = {}
    for i, image_path in enumerate(image_paths):
        print(image_path)
        # if "atom" not in str(image_path):
        #     continue
        im = Image.open(image_path)
        im = im.convert('RGB')
        imt = segmentation_transforms(im).to(device)
        with torch.no_grad():
            output = segmentation_model(imt.unsqueeze(0))
        masks = MaskList.from_model_output(output[0]).select_top_masks()
        # print([mask for mask in masks])
        scores[image_path] = masks.scores(imt)

    return scores

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

    if args.model_mix == "nima_only":
        # TODO: improve readability of weights.
        # Weight order: NIMA, area_score, centered_score, object_type_score, sharpness_score
        weights = [1, 0, 0, 0, 0]
    elif args.model_mix == "uniform":
        # Note, nima evaluates from 1 to 10 instead of (sometimes roughly) 0 to 1, so we scale it down by a factor of 10
        weights = [1 / 50, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
    elif args.model_mix == "uniform_seg_only":
        # Note, nima evaluates from 1 to 10, so we scale it down
        weights = [0, 1 / 4, 1 / 4, 1 / 4, 1 / 4]
    else:
        raise NotImplementedError(f"{args.model_mix=} not yet implemented (how did you even get here?)")


    if weights[0]:
        nima_model = get_evaluation_model(weight_path=args.model_path)
        nima_scores = nima_score_paths(nima_model, image_paths)
    else:
        # if no relevant weights, we just grab a dictionary that returns zeros
        nima_scores = defaultdict(int)

    if any(weight > 0 for weight in weights[1:]):
        segmentation_scores = segmentation_score_paths(image_paths)
    else:
        # if no relevant weights, we just grab a dictionary that returns zeros
        segmentation_scores: dict[Path, SegmentationScores] = defaultdict(lambda: SegmentationScores(0, 0, 0, 0))

    # Could do this slightly less expanded out, but this makes the weights more explicit, so I'll just leave it alone
    # for now
    scores = {
        image_path:
        nima_scores[image_path] * weights[0] +
        segmentation_scores[image_path].area_score * weights[1] +
        segmentation_scores[image_path].centered_score * weights[2] +
        segmentation_scores[image_path].object_type_score * weights[3] +
        segmentation_scores[image_path].sharpness_score * weights[4]
     for image_path in image_paths
     }

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
