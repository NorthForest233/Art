import torch
import torch.nn as nn
from torchvision import models, transforms

class ContentLoss(nn.Module):
    def __init__(self, content_feature):
        super(ContentLoss, self).__init__()
        self.content_feature = content_feature.detach()
        self.criterion = nn.MSELoss()

    def forward(self, combination):
        return self.criterion(combination, self.content_feature)


class GramMatrix(nn.Module):
    def forward(self, x):
        # b, n, h, w = x.size()
        # features = x.view(b * n, h * w)
        # G = torch.mm(features, features.t())
        # return G.div(b * n * h * w)
        b, c, h, w = x.shape
        features = x.view(b, c, h * w)
        gram_matrix = torch.einsum('ijl,ikl->ijk', features, features)
        return gram_matrix / (h * w)


class StyleLoss(nn.Module):
    def __init__(self, style_feature, mask=None):
        super(StyleLoss, self).__init__()
        self.style_feature = style_feature.detach()
        self.mask = mask
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # output = combination
        if self.mask is None:
            style_feature = self.gram(self.style_feature)
            combination_features = self.gram(x)
        else:
            style_feature = self.gram(self.style_feature * self.mask)
            combination_features = self.gram(x * self.mask)
        return self.criterion(combination_features, style_feature)

class VGG19Features(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        vgg19 = models.vgg19()
        if isinstance(weights, str):
            vgg19.load_state_dict(torch.load(weights))
        elif isinstance(weights, dict):
            vgg19.load_state_dict(weights)
        self.features = vgg19.features
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        output = []
        x = self.normalize(x)
        for layer in self.features:
            x = layer(x)
            output.append(x)
        return output