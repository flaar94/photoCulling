from typing import Self, ClassVar, NamedTuple, Callable
from dataclasses import dataclass
import math
from operator import mul
from functools import reduce
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
import torch

from utils import weighted_quantile_mean, weighted_central_root_moment

LAPLACE_KERNEL = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32,
                              requires_grad=False).unsqueeze(0).unsqueeze(0) / 6.

animals = ("cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")
vehicles = ("bicycle", "car", "motorcycle", "airplane", "bus", "truck", "boat")
food = ("banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake")
objects = ("traffic light", "stop sign", "parking meter", "fire hydrant", "wine glass", "cup", "fork", "knife", "spoon",
           "bowl", "potted plant", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
           "toaster", "toilet", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
           "frisbee", "skis", "snowboard", "sportsball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "backpack", "umbrella", "handbag", "tie", "suitcase")
mediums = ("bed", "dining table", "sink", "refrigerator", "chair", "bench")
misc = ("__background__", "N/A")
CATEGORY_GROUPS = {"person": ["person"], "animal": animals, "vehicle": vehicles, "food": food, "object": objects,
                   "medium": mediums, "misc": misc}

PRIORITIES = {"person": 2, "animal": 2, "food": 3, "vehicle": 4, "object": 5, "medium": 6, "misc": 6}

SEGMENTATION_MODEL_CATEGORIES = tuple(MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"])


class SegmentationScores(NamedTuple):
    """Namedtuple to make the output of MaskList.scores() more readable"""
    area_score: float
    centered_score: float
    object_type_score: float
    sharpness_score: float


@dataclass
class Mask:
    """Class to hold an entity/object discovered by an instance segmentation model, and compute useful features"""
    mask: torch.Tensor
    label: str
    confidence: float

    # class vars containing model output information
    priorities: ClassVar[dict[str, int]] = PRIORITIES
    category_groups: ClassVar[dict[str, list[str]]] = CATEGORY_GROUPS

    @property
    def area(self) -> float:
        """Computes the (probabilistic mean) area of the entity/object, as a fraction of the full image"""
        return float(self.mask.sum() / (self.mask.shape[-1] * self.mask.shape[-2]))

    @property
    def center(self) -> tuple[float, float]:
        """Computes the center of the entity/object in the image, where width and height are the fraction of the images
        total width and height, and so are between 0 and 1
        Note: ((x,y) measured from bottom left of image, left to right, bottom to top)"""

        center = (
            float((self.mask.mean(-2) * torch.arange(0, self.mask.shape[-1], device=self.mask.device) /
                   self.mask.shape[-1]).sum() / self.mask.mean(-2).sum()),
            1 - float((self.mask.mean(-1) * torch.arange(0, self.mask.shape[-2], device=self.mask.device) /
             self.mask.shape[-2]).sum() / self.mask.mean(-1).sum())
        )
        return center

    @property
    def priority(self) -> float:
        """Checks if the entity's category is in the priority list, if so uses the value, otherwise gives max possible"""
        for category_name, subcategories in self.category_groups.items():
            if self.label in subcategories:
                return self.priorities[category_name]
        else:
            # if the for loop finishes without break, ie if no priority is found
            return max(self.priorities.values())

    def _effective_priority(self, area_threshold):
        return self.priority if self.area > area_threshold else self.priority + 1

    def __repr__(self):
        return f"(Mask {self.label}, area={self.area:.3f}, center=({self.center[0]:.3f}, {self.center[1]:.3f}), confidence={self.confidence:.3f})"


class MaskList(list):
    @classmethod
    def from_model_output(cls, model_output: dict, score_threshold: float = 0.5) -> Self:
        """
        Takes the raw output of a PyTorch segmentation model on a single image, and creates a MaskList

        :param model_output: Raw output of a pytorch segmentation model
        :param score_threshold: A lower bound on the score when determining which masks to keep
        :return: A MaskList containing the information from the segmentation model output
        """
        return cls((Mask(mask, SEGMENTATION_MODEL_CATEGORIES[int(label)], float(score)) for mask, label, score in
                    zip(model_output["masks"], model_output["labels"], model_output["scores"]) if
                    score > score_threshold))

    def select_top_masks(self, peripheral_threshold: float = 0.2, area_multiplier: float = 4,
                         priority_adjustment: float = 2) -> Self:
        """
        Filters the mask list so it (hopefully) only contains important objects, using heuristics for determining which
        objects in the picture are what people are actually trying to display.

        :param peripheral_threshold: how close to each side the object's center is allowed to be as a fraction of the image
                size (0, 1)
        :param area_multiplier: If an object of priority one less than the max exists, it is ignored unless its area is
                area_multiplier times as large as the higher priority objects.
                Also used to divide the largest area to determine the lower bound on the area of masks we will keep
        :param priority_adjustment: Used to adjust the lower bound on the area of masks we keep. In particular, masks of
                higher priority will be included if priority_adjustment * (their area) is larger than the unadjusted
                lower bound
        :return: A Mask sublist of the original list representing the objects which are likely important
        """

        # Highly peripheral objects are unlikely to be the goal even if they are high priority objects
        non_peripheral_masks = [mask for mask in self if all(
            peripheral_threshold <= position <= 1 - peripheral_threshold for position in mask.center)]
        if not non_peripheral_masks:
            return self.__class__()
        # If an object is very small, we allow a single point of wiggle room on priority. We don't want too much though, since things like tables or whatever are often big but almost never relevant
        highest_priority = min(mask.priority for mask in non_peripheral_masks)
        highest_priority_largest_area = max(
            mask.area for mask in non_peripheral_masks if mask.priority == highest_priority)
        second_priority_largest_area = max(
            (mask.area for mask in non_peripheral_masks if mask.priority == highest_priority + 1), default=0)
        if highest_priority_largest_area * area_multiplier > second_priority_largest_area:
            return self.__class__([mask for mask in non_peripheral_masks if
                                   mask.priority == highest_priority and mask.area > highest_priority_largest_area / area_multiplier])
        else:
            return self.__class__([mask for mask in non_peripheral_masks if (
                    mask.priority == highest_priority - 1 and mask.area > highest_priority_largest_area / area_multiplier) or
                                   (mask.priority == highest_priority and mask.area > highest_priority_largest_area / (
                                           priority_adjustment * area_multiplier))
                                   ])

    def merge(self) -> torch.Tensor:
        """
        Takes a list of masks, and combines them into a single tensor mask

        :return: tensor mask, (note: not the Mask class, since it has no single object type)
        """
        return 1 - reduce(mul, torch.prod(torch.stack([1 - mask.mask for mask in self]), dim=0))

    @staticmethod
    def pointwise_sharpness(image: torch.Tensor) -> torch.Tensor:
        """
        Computes the "sharpness" at each point, ie convolves the laplace kernel with each square of 9 pixels.

        :param image: a tensor representing an image of shape (3 x width x height)
        :return: A tensor in the shape of the original image but with 2 pixels less width and height
        """
        with torch.no_grad():
            pointwise_sharpness = torch.conv2d(image.unsqueeze(1),
                                               weight=LAPLACE_KERNEL.to(image.device)).squeeze().sum(dim=0)
        return pointwise_sharpness

    def area_error(self, ideal_range: tuple[float, float] = (0.3, 0.8), eps=0.01) -> float:
        """
        Computes how much an objects area differs from the ideal range.

        Note: our piecewise choice of metrics may look a bit strange. For areas greater than the upper bound, the raw
        area is probably fine, but for the lower bound, we use a log scale. This is because I think a discrepancy of 0.2
        isn't much if we're going from 0.5 to 0.3, but it is huge if we're going from 0.21 to 0.01.

        :param ideal_range: tuple of floats. Any total areas within this range will result in an error of 0.
        :param eps: regularization parameter. Roughly like the fraction of the image that we consider very bad,
                    and making it even smaller doesn't increase the error by much
        :return: float between 0 and 1
        """
        if not self:
            return 1.
        total_area = sum(mask.area for mask in self)
        if total_area < ideal_range[0]:
            max_unnormalized = (math.log2(ideal_range[0] + eps) - math.log2(eps))
            return (math.log2(ideal_range[0] + eps) - math.log2(total_area + eps)) / max_unnormalized
        elif total_area > ideal_range[1]:
            return total_area - ideal_range[0]
        else:
            return 0

    def mask_weighted_average(self, value_fn: Callable[[Mask], float]) -> float:
        """Computes a weighted average of some values calculated from a mask, eg: an attribute via lambda x:x.my_attr"""
        return sum([value_fn(mask) * mask.confidence * mask.area for mask in self]) / sum(
            [mask.confidence * mask.area for mask in self])

    def centered_error(self, ideal_object_center=(0.5, 0.5), error_buffer=0.1) -> float:
        """
        Computes how much the center of objects differs from the center. We use L1 distance based on the assumption that
        diagonal distance is particularly bad

        :param ideal_object_center: the ideal center of the object
        :param error_buffer: how far from the ideal_center the object is allowed to be without taking a penalty
                Note: the buffer zone is a diamond, not a flat square
        :return:
                float between 0 and 1
        """
        if not self:
            return 1.
        weighted_center = (self.mask_weighted_average(lambda mask: mask.center[0]),
                           self.mask_weighted_average(lambda mask: mask.center[1]))
        return max(abs(ideal_object_center[0] - weighted_center[0]) - error_buffer, 0) + \
            max(abs(ideal_object_center[1] - weighted_center[1]) - error_buffer, 0)

    def object_type_error(self):
        """
        Computes a weighted average of the priority of objects in the MaskList, as a fraction of max priority
        """
        if not self:
            return 1.
        weighted_priority_mean = self.mask_weighted_average(lambda mask: mask.priority)
        return weighted_priority_mean / max(Mask.priorities.values())

    def object_sharpness(self, image: torch.Tensor, method: str = "quantile_mean",
                         p: float = 4, quantile: float = 0.995, full_image_sharpness: bool = False) -> float:
        """
        Computes the sharpness of an image on the region of image determined by our mask (eg: an object)

        :param image: The image tensor
        :param method: A robust estimate of the maximum used to aggregate pointwise image sharpness
        :param p: Parameter in maximum estimate functions -- exponent used in calculation
        :param quantile: Parameter in maximum estimate functions -- the fraction of smaller values to ignore
        :param full_image_sharpness: ignore the mask and just computes (robust max) sharpness on the full image
        :return: A float estimating the sharpness of the image on that object's sharpest places
        """
        truncated_sharpness = self.pointwise_sharpness(image)
        # Ignore the boundary pixels which are affected by padding
        if full_image_sharpness or not self:
            # If the list of masks is empty, ie there are no detected objects, it defaults to full image sharpness
            truncated_mask = None
        else:
            truncated_mask = self.merge()[1:-1, 1:-1]

        if method == "central_moment":
            return weighted_central_root_moment(truncated_sharpness, truncated_mask, p=p, quantile=quantile)
        elif method == "quantile_mean":
            return weighted_quantile_mean(truncated_sharpness, truncated_mask, p=p, quantile=quantile)
        else:
            raise NotImplementedError(f"{method=} has not been implemented, please use one of ['central_moment']")

    def scores(self, image: torch.tensor,
               area_kwargs: dict[str, any] | None = None,
               centered_kwargs: dict[str, any] | None = None,
               sharpness_kwargs: dict[str, any] | None = None
               ) -> SegmentationScores:
        """
        Thin wrapper for the 4 primary scores coming from this class

        :param image: the original image
        :param area_kwargs: arguments to pass into the area_error method
        :param centered_kwargs: arguments to pass into the centered_error method
        :param sharpness_kwargs: arguments to pass into the object_sharpness method
        :return: A named tuple containing the different scores
        """
        if area_kwargs is None:
            area_kwargs = {}
        if centered_kwargs is None:
            centered_kwargs = {}
        if sharpness_kwargs is None:
            sharpness_kwargs = {}
        return SegmentationScores(area_score=1 - self.area_error(**area_kwargs),
                                  centered_score=1 - self.centered_error(**centered_kwargs),
                                  object_type_score=1 - self.object_type_error(),
                                  sharpness_score=self.object_sharpness(image, **sharpness_kwargs)
                                  )
