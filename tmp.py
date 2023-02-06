import torch 
import torchvision

"""
This file is to test I/O operations in the model
Just a tmp file to see what goes in and out the model.
How input is processed through out the layers.
"""

from backbone import CLIPResNet

inp = torch.rand([4, 3, 224, 224])

model = CLIPResNet([3, 4, 6, 3]) # for resnet50

out = model(inp)

print(f"Out length: {len(out)}")
print(f"Out[0] shape: {out[0].shape}")
print(f"Out[1] shape: {out[1].shape}")
print(f"Out[2] shape: {out[2].shape}")
print(f"Out[3] shape: {out[3].shape}")

# print(model)
