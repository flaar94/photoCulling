"""
file - model.py
Implements the aesthetic model and emd loss used in paper.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import torch.nn as nn
import torch
import math


class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""

    def __init__(self, base_model, num_classes=10, image_dims=224):
        super().__init__()
        self.features = base_model.features

        # Check how many output features the model will give us
        with torch.no_grad():
            output_shape = self.features(torch.zeros(1, 3, image_dims, image_dims)).shape
        in_features = math.prod(output_shape)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=in_features, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out
