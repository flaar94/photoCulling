import time

from PIL import Image, ExifTags
from pathlib import Path
import itertools
import datetime
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import copy
from nima.model import NIMAMean
from segment import MaskList, SegmentationScores
from collections.abc import Iterable
from collections import defaultdict
import argparse
import pandas as pd
import logging
import sys
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torch.utils.data import Dataset, DataLoader

# This is the maximum number of pictures that will be fed into the VGG-based models at once. Should be small enough to
# not use 8GB of VRAM, but you can make it smaller if running into memory problems, or increase it if you want to try
# speeding it up.
# Note: doesn't apply to instance segmentation model which doesn't rescale images, so can potentially use lots of memory
BATCH_SIZE = 10

# The size of images that VGG16/NIMA expects. Images will be resized to this for those purposes
# Note: doesn't apply to the instance segmentation model
IMAGE_DIMS = 224


class TypedNamespace(argparse.Namespace):
    """Stub-like class whose purpose is just to give the 'args' Namespace intelliSense"""
    image_path: str
    model_path: str
    predictions: str
    score_only: bool
    similarity_threshold: float
    time_threshold: int
    silence: bool
    model_mix: bool
    weights: list[float]


def get_parser():
    """Sets up the CLI argument parser"""
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
    parser.add_argument("--model_mix", default="nima",
                        choices=["nima", "uniform", "uniform_seg", "sharpness"],
                        help="Which models/scores to use to make predictions")
    parser.add_argument("--weights", default=None, required=False, nargs=5, type=float,
                        help="Explicit set of 5 space separated weights. Overrides model_mix. Eg: '0.1 1 1 1 2.5'. Order"
                             "of weights NIMA, area_score, centered_score, object_type_score, sharpness_score. Still "
                             "in development.")
    return parser


class ImageDataset(Dataset):
    """
    PyTorch dataset for loading and processing images using PIL
    """
    image_paths: list[Path]
    transform: callable

    def __init__(self, image_paths: Iterable[Path], transform=None):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def extract_top_scored(group: Iterable[Path], scores: dict[Path, float]) -> Path:
    """Grabs path whose corresponding image got the highest score in the group"""
    return max(group, key=lambda path: scores[path])


def flatten_batches(batches: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
    """Takes a list or other iterable of batched model outputs, and flattens them"""
    return (item for batch in batches for item in batch)


def apply_model_to_paths(model: nn.Module,
                         image_paths: Iterable[Path],
                         transform: callable = None) -> dict[Path, torch.Tensor]:
    """
    Takes an arbitrary pytorch module, and applies it each image in the image_path list

    :param model: the PyTorch module, which will apply its __call__ function to each batch of images
    :param image_paths: the list or other iterable of paths to the images
    :param transform: the preprocessing operation expected by the PyTorch module
        (eg: converting to a tensor, normalization, etc...)
    :return: dictionary mapping each image path to the model output (eg: a numerical score for NIMA, or
        tensor features for VGG-16)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dataset = ImageDataset(image_paths, transform=transform)
    image_dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False)

    batches = []
    for images_batch in image_dataloader:
        with torch.no_grad(), torch.autocast(device_type=device):
            images_batch = images_batch.to(device)
            batch = model(images_batch)
            batches.append(batch)
    flattened_tensors = list(flatten_batches(batches))

    return {image_path: model_output for image_path, model_output in zip(image_paths, flattened_tensors)}


def extract_time_taken_metadata(image_paths: Iterable[Path]) -> dict[Path, datetime.datetime]:
    """
    Grabs the time the picture was taken from the metadata. If it isn't available, the date/time will be given a
    placeholder value (0001-01-01 00:00:00). In this way, photos without this piece of metadata can be grouped with each
    other purely using model features

    :param image_paths: the paths to each image
    :return: A dictionary mapping each image path to its metadata DateTime, hopefully the time taken.
    """
    photo_capture_times = {}
    for image_path in image_paths:
        image = Image.open(image_path)
        # convert metadata into a human readable dictionary
        image_metadata = {ExifTags.TAGS[k]: v for k, v in image.getexif().items() if k in ExifTags.TAGS}
        photo_capture_time = datetime.datetime.strptime(image_metadata["DateTime"],
                                                        "%Y:%m:%d %H:%M:%S") if "DateTime" in image_metadata \
            else datetime.datetime(1, 1, 1)
        photo_capture_times[image_path] = photo_capture_time
    return photo_capture_times


def group_by_features(image_features: dict[Path, torch.Tensor],
                      time_metadata: dict[Path, datetime.datetime],
                      similarity_threshold,
                      time_threshold,
                      ) -> set[frozenset[Path]]:
    """
    Determine pairs of items that are similar in time and VGG features

    :param image_features: dictionary mapping image paths to computed VGG-16 feature vectors
    :param time_metadata: dictionary mapping image paths to extracted image collection times
    :param similarity_threshold: how many minutes between pictures is allowed for two photos to be grouped
    :param time_threshold: the minimum cosine similarity between photos' base-model features for them to be grouped
    :return: Set of groups of images that are similar in time and features
    """

    similar_images = []
    all_connected_images = set()
    for image_path1, image_path2 in itertools.combinations(image_features.keys(), 2):
        features1, features2 = image_features[image_path1], image_features[image_path2]
        datetime1, datetime2 = time_metadata[image_path1], time_metadata[image_path2]

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


def get_evaluation_model(weight_path: Path | str) -> tuple[NIMAMean, callable]:
    """
    Sets up the model used to evaluate photo quality

    :param weight_path: path to the filed containing the weights for the pretrained model
    :return: Tuple of the  model and a transform
        model: the direction evaluation NIMA module, trained on human ratings
        transform: that preprocesses the data into something the model will understand
    """

    # These weights will get overwritten, but it's actually faster to load them, I guess due to weight initialization
    weights = models.VGG16_Weights.IMAGENET1K_V1
    base_model = models.vgg16(weights=weights)
    model = NIMAMean(base_model, image_dims=IMAGE_DIMS)
    try:
        model.load_state_dict(torch.load(weight_path))
        logging.info('successfully loaded evaluation model')
    except OSError:
        raise OSError("Could not load state dictionary, are you sure your path is correct?")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    nima_transform = transforms.Compose([
        transforms.Resize((IMAGE_DIMS, IMAGE_DIMS)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return model, nima_transform


def get_feature_model() -> tuple[nn.Module, callable]:
    """
    Function for loading and setting up a VGG-16 module, and its transforms

    :return: Tuple of a model and a transform
        model: the feature calculating VGG-16 module
        transform: callable that preprocesses the data into something the model will understand
    """
    weights = models.VGG16_Weights.IMAGENET1K_V1
    feature_model = models.vgg16(weights=weights)
    feature_model.to("cuda" if torch.cuda.is_available() else "cpu")
    feature_model.eval()

    return feature_model.features, weights.transforms()


def segmentation_score_paths(image_paths: Iterable[Path]) -> dict[Path, SegmentationScores]:
    segmentation_weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    segmentation_model = maskrcnn_resnet50_fpn_v2(weights=segmentation_weights)
    segmentation_transforms = segmentation_weights.transforms()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmentation_model = segmentation_model.eval().to(device)

    image_dataset = ImageDataset(image_paths, transform=segmentation_transforms)

    scores = {}
    for image_path, image_tensor in zip(image_paths, image_dataset):
        image_tensor = image_tensor.to(device=device)
        with torch.no_grad():
            output = segmentation_model(image_tensor.unsqueeze(0))
        mask_list = MaskList.from_model_output(output[0]).select_top_masks()
        scores[image_path] = mask_list.scores(image_tensor)
    return scores


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(args=argv, namespace=TypedNamespace())

    logging_level = logging.WARNING if args.silence else logging.INFO
    logging.basicConfig(level=logging_level, stream=sys.stdout, format="%(message)s")
    logging.info(args)

    seed = 42
    torch.manual_seed(seed)

    image_dir = Path(args.image_path)
    image_paths = list(itertools.chain(image_dir.glob("*.jpg"),
                                       image_dir.glob("*.jpeg"),
                                       image_dir.glob("*.png")))

    weight_options = {
        # Weight order: NIMA, area_score, centered_score, object_type_score, sharpness_score
        "nima": [1, 0, 0, 0, 0],
        # Note, nima is usually between 4 and 6 instead of (sometimes roughly) 0 to 1, so we scale it down by a factor of 2
        "uniform": [1 / 10, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
        "uniform_seg": [0, 1 / 4, 1 / 4, 1 / 4, 1 / 4],
        "sharpness": [0, 0, 0, 0, 1]
    }

    if args.weights is not None:
        weights = args.weights
    else:
        weights = weight_options[args.model_mix]

    logging.info(f"Using {weights=}")
    if weights[0]:
        nima_model, nima_transforms = get_evaluation_model(weight_path=args.model_path)
        nima_scores = apply_model_to_paths(nima_model, image_paths, transform=nima_transforms)
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
            float(nima_scores[image_path]) * weights[0] +
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
        feature_model, feature_transforms = get_feature_model()
        image_features = apply_model_to_paths(feature_model, image_paths, transform=feature_transforms)

        time_metadata = extract_time_taken_metadata(image_paths)
        logging.info("Image features extracted, now grouping images")
        groups = group_by_features(image_features,
                                   time_metadata,
                                   similarity_threshold=args.similarity_threshold,
                                   time_threshold=args.time_threshold,
                                   )
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
