import numpy as np
import torch
from torch.autograd import Variable
import ray

from segment import DRNSeg

@ray.remote(num_gpus=1)
class Segmentor:
    def __init__(self, arch, classes, pretrained):
        single_model = DRNSeg(arch, classes, pretrained_model=None,
                pretrained=False)
        single_model.load_state_dict(torch.load(pretrained))
        self.model = torch.nn.DataParallel(single_model).cuda()

    def segment_image(self, image):
        # Load image from numpy array to pytorch variable
        image = torch.from_numpy(image.transpose([2, 0, 1])).unsqueeze(0).float()
        image_var = Variable(image, requires_grad=False, volatile=True)

        # Get model prediction
        final = self.model(image_var)[0]
        _, pred = torch.max(final, 1)

        return pred.cpu().data.numpy()[0].astype(np.uint8)

