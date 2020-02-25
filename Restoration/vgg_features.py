import torch
from torch import nn
from torchvision.models import vgg16
import numpy as np

class vgg_features(nn.Module):
    def __init__(self):
        super(vgg_features, self).__init__()
        # get vgg16 features up to conv 4_3
        self.model = nn.Sequential(*list(vgg16(pretrained=True).features)[:23])
        self.model = self.model.cuda(1)
        #self.model = self.model.cuda()
        # will not need to compute gradients
        for param in self.parameters():
            param.requires_grad=False

    def forward(self, x, renormalize=True):
        # change normaliztion form [-1,1] to VGG normalization

        if renormalize:
            x = ((x*.5+.5)-torch.cuda.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))/torch.cuda.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            #x = ((x*.5+.5)-torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1))/torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        #x = x.cpu()
        x = x.cuda(1)
        x = self.model(x)
        #x = x.cuda()
        x = x.cpu()
        x = x.cuda(0)
        return x
