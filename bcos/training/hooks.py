import torch
from torchvision.transforms import functional as F

IMAGENET_MEAN_ADDINVERSE = (0.485, 0.456, 0.406, 0.515, 0.544, 0.594)
IMAGENET_STD_ADDINVERSE = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)

class Hook:
    # This is a hook to be used to explicitly manipulate the gradients 
    # of the B parameters in the Bcos layers.
    def __init__(self, mod, start=1, end=2):
        self.mod = mod
        self.start = start 
        self.end = end
    def __call__(self, grad):
        ## If the b value is less than the supposed start value, then it should be set to start
        if self.mod.b < self.start:
            self.mod.b.data = torch.tensor(float(self.start + 1e-6)).cuda()
        ## If b value is greater than the supposed end value, then return 0 gradient
        if self.mod.b >= self.end:
            # When the final value is reached, no updates should be done anymore
            return torch.zeros_like(grad)
        return - self.mod.batch_size * torch.ones_like(grad)

# To add the batch_size attribute to the module for b update    
def forward_hook_fn(module, input, output):
    # Assuming the input is a tuple of tensors, where the first tensor is the input tensor
    try:
        input_tensor = input[0]
        batch_size = input_tensor.size(0)
        module.batch_size = batch_size
    except:
        # For DenseNets as the input comes in a list torch.Size([64, 64, 56, 56])
        input_tensor = input[0][0]
        batch_size = input_tensor.size(0)
        module.batch_size = batch_size

# Called before the first layer to update the inputs
# Called in model.py 
# Alternative: BcosifyNormalize as a layer in bcosify.py
def pre_forward_bcosify_nomalize(module, input, inplace=False, dim: int = -3):
    return F.normalize(input[0], IMAGENET_MEAN_ADDINVERSE, IMAGENET_STD_ADDINVERSE, inplace)
   