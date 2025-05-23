import torch, gc
from models.model import Net

DEVICE = "cuda"

model = Net().to(DEVICE)

print('total param',  sum([param.nelement() for param in model.parameters()]))

gc.collect()
torch.cuda.empty_cache()