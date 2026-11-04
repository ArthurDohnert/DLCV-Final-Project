"""Passo 8 do guia: confirma que a GPU do nó funciona com PyTorch e mede velocidade."""
import time
import torch

print("torch", torch.__version__, "| CUDA disponível:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Sem GPU visível. Confira --gres/--partition no sbatch.")
dev = torch.device("cuda")
p = torch.cuda.get_device_properties(dev)
print(f"GPU: {p.name} | memória: {p.total_memory / 1e9:.1f} GB")

import segmentation_models_pytorch as smp
model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), 1e-4)
scaler = torch.cuda.amp.GradScaler()
x = torch.randn(8, 3, 512, 512, device=dev)
y = torch.randn(8, 1, 512, 512, device=dev)
for i in range(13):
    if i == 3:
        torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.l1_loss(model(x), y)
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
torch.cuda.synchronize()
print(f"U-Net ResNet34, batch 8 de 512x512, AMP: {(time.time() - t0) / 10 * 1000:.0f} ms/iteração")
print(f"Pico de memória: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print("TUDO OK")
