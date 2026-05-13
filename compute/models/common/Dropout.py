import torch
import torch.nn as nn

class FeatureDropout(nn.Module):
    def __init__(self, 
                 feat_dropout:float=0.0
                ) -> None:

        super(FeatureDropout, self).__init__()

        self.feat_dropout = feat_dropout
        self.register_buffer("mask", None)
    
    def reset_mask(self) -> None:
        self.mask = None

    def forward(self, 
                feat_tensor:torch.Tensor
                ) -> torch.Tensor:

        if not self.training or self.feat_dropout == 0:
            return feat_tensor

        if self.mask is None or self.mask.size(0) != feat_tensor.size(0):
            mask_shape = (feat_tensor.size(0),) + (1,) + (feat_tensor.size(-1),)
            self.mask = feat_tensor.new_empty(mask_shape).bernoulli_(1 - self.feat_dropout).div_(1 - self.feat_dropout)
           
        return feat_tensor * self.mask

