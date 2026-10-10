import torch
from train_D_midfusion_concat import RGBTYOLOv10MidConcat, BASE_MODEL_YAML

model = RGBTYOLOv10MidConcat(
    cfg=BASE_MODEL_YAML,
    ch=4,
    nc=2,
    verbose=True,
)

model.train()

x = torch.randn(2, 4, 640, 640)

out = model(x)

print("Forward success.")
print(type(out))