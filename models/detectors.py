import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from detectors.TRACER.model.TRACER import TRACER
from detectors.TRACER.config import getConfig
import utils

class TracerModule(nn.Module):
    def __init__(self, weights, criterion) -> None:
        super().__init__()
        print('Loading TRACER')
        args = getConfig()
        self.tracer = nn.DataParallel(TRACER(args))        
        # self.criterion = Criterion(args)
        self.criterion = utils.instantiate_from_config(criterion)
        if isinstance(weights, str):
            self.tracer.load_state_dict(torch.load(weights))
        elif isinstance(weights, dict):
            self.tracer.load_state_dict(weights)
        self.tracer.eval()
        for parameter in self.tracer.parameters():
            parameter.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, x):
        return *self.tracer(self.normalize(F.interpolate(x, size=(320, 320), mode='bilinear'))), x.shape[-2:]

    def get_loss(self, x, mask):
        tracer_mask, _, ds_map, size = x
        tracer_mask = F.interpolate(tracer_mask, size=size, mode='bilinear')
        ds_map = [F.interpolate(m, size=size, mode='bilinear') for m in ds_map]
        loss = self.criterion(tracer_mask, mask) + sum([self.criterion(m, mask) for m in ds_map])
        return loss
    
    def get_mask(self, x):
        tracer_mask, _, _, size = x
        return F.interpolate(tracer_mask, size=size, mode='bilinear')