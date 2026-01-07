import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Dataloader import get_dataloader
from RoadExtractorNet import HSV_Guided_UNet, Plain_UNet

from HSV_Attention_UNet import MultiScale_HSV_UNet
import os

from tqdm import tqdm  # 导入tqdm

from Loss import hsv_unet_loss

# =============== 参数设置 ===============
DATA_ROOT = "D:/myDataManager/pycharmProject/Crack-Segmentation/road_roi_net/RoadDataset"
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 300
BATCH_SIZE = 1
LR = 1e-3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2

# =============== 数据加载 ===============
train_loader = get_dataloader(DATA_ROOT, split="train",
                              batch_size=BATCH_SIZE,
                              num_workers=0)

val_loader = get_dataloader(DATA_ROOT, split="val",
                            batch_size=BATCH_SIZE,
                            num_workers=0,
                            shuffle=False)

# =============== 模型 & 优化器 ===============
model = MultiScale_HSV_UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

start_epoch = 0
best_val_loss = float("inf")

# 恢复训练（如果需要）
ckpt_path = os.path.join(SAVE_DIR, "latest.pth")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_loss", float("inf"))
    print(f"[INFO] Resumed from epoch {start_epoch}")

# =============== 训练循环 ===============
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_att_loss = 0.0

    # 创建训练进度条
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS} [Train]')

    for imgs, masks_gt in train_pbar:
        imgs = imgs.to(DEVICE)
        masks_gt = masks_gt.to(DEVICE)

        # 添加维度检查，确保masks_gt是4维的
        if masks_gt.dim() == 3:
            masks_gt = masks_gt.unsqueeze(1)

        # 新网络输出：logits, attention_mask, thresholds
        logits = model(imgs)

        # 计算损失
        total_loss, loss_dict = hsv_unet_loss(
            logits, masks_gt, imgs,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_seg_loss += loss_dict['seg']

        # 更新进度条显示当前batch的损失
        train_pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Seg_Loss': f'{loss_dict["seg"]:.4f}',
        })

    avg_train_loss = running_loss / len(train_loader)
    avg_seg_loss = running_seg_loss / len(train_loader)

    # -------- 验证 --------
    model.eval()
    val_loss = 0.0
    val_seg_loss = 0.0
    val_att_loss = 0.0

    # 创建验证进度条
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS} [Val]')

    with torch.no_grad():
        for imgs, masks_gt in val_pbar:
            imgs = imgs.to(DEVICE)
            masks_gt = masks_gt.to(DEVICE)

            # 同样在验证时添加维度检查
            if masks_gt.dim() == 3:
                masks_gt = masks_gt.unsqueeze(1)

            logits = model(imgs)
            loss, loss_dict = hsv_unet_loss(
                logits, masks_gt, imgs
            )
            val_loss += loss.item()
            val_seg_loss += loss_dict['seg']

            # 更新验证进度条
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Seg': f'{loss_dict["seg"]:.4f}',
            })

    avg_val_loss = val_loss / len(val_loader)
    avg_val_seg_loss = val_seg_loss / len(val_loader)

    # 学习率调度
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # 打印每个epoch的汇总信息
    print(f"\n[Epoch {epoch + 1}/{NUM_EPOCHS}] LR: {current_lr:.2e}")
    print(f"  Train => Total: {avg_train_loss:.4f} | Seg: {avg_seg_loss:.4f}")
    print(f"  Val   => Total: {avg_val_loss:.4f} | Seg: {avg_val_seg_loss:.4f}")

    # -------- 保存模型 --------
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_val_loss,
        "val_loss": avg_val_loss
    }
    torch.save(ckpt, os.path.join(SAVE_DIR, "latest.pth"))

    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(ckpt, os.path.join(SAVE_DIR, "best.pth"))
        print(f"[INFO] Best model saved at epoch {epoch + 1} with val_loss: {best_val_loss:.4f}")

    # 每10个epoch保存一次检查点
    if (epoch + 1) % 10 == 0:
        torch.save(ckpt, os.path.join(SAVE_DIR, f"epoch_{epoch + 1}.pth"))