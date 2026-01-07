import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from RoadExtractorNet import HSV_Guided_UNet, Plain_UNet, MultiScale_HSV_UNet
# from HSV_Attention_UNet import MultiScale_HSV_UNet
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
import pandas as pd
from tabulate import tabulate




# ========== 后处理函数 (加入闭合 + 最大连通域) ==========
def postprocess_and_save(img_path, mask_tensor, thresh=0.5, close_kernel=16):
    """
    img_path: 原始图片路径
    mask_tensor: [1,H,W] 或 [H,W] sigmoid 输出
    """
    # ---- 1) 转 numpy ----
    mask = mask_tensor.squeeze().cpu().numpy()  # [H,W], float32

    # ---- 2) 阈值二值化 ----
    binary = (mask > thresh).astype(np.float32)  # 保持归一化

    # ---- 3) 形态学闭合 ----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    binary_uint8 = (binary * 1).astype(np.uint8)  # uint8 0/1
    closed = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)

    # ---- 4) 保留最大连通域 ----
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = 1 + np.argmax(areas)
        final_mask = np.zeros_like(closed, dtype=np.float32)
        final_mask[labels == max_idx] = 1.0
    else:
        final_mask = closed.astype(np.float32)

    # ---- 5) 读原图并 resize 回去 ----
    orig = cv2.imread(img_path)
    h, w = orig.shape[:2]
    mask_resized = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---- 6) 掩膜原图 ----
    masked_img = orig.copy()
    masked_img[mask_resized == 0] = 0

    return masked_img, mask_resized


# ========== 指标计算函数 ==========
def calculate_metrics(y_true, y_pred):
    """
    计算各种评估指标
    y_true: 真实标签 (H, W)
    y_pred: 预测标签 (H, W)
    """
    # 展平数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1]).ravel()

    # 计算各项指标
    metrics = {}

    # IOU (Jaccard)
    metrics['iou'] = jaccard_score(y_true_flat, y_pred_flat, average=None)
    metrics['miou'] = jaccard_score(y_true_flat, y_pred_flat, average='macro')

    # Precision, Recall, F1
    metrics['precision'] = precision_score(y_true_flat, y_pred_flat, average=None, zero_division=0)
    metrics['recall'] = recall_score(y_true_flat, y_pred_flat, average=None, zero_division=0)
    metrics['f1'] = f1_score(y_true_flat, y_pred_flat, average=None, zero_division=0)

    # 整体指标
    metrics['overall_precision'] = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    metrics['overall_recall'] = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    metrics['overall_f1'] = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)

    # FNR, FPR
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # 假阴性率
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假阳性率

    # Dice (与F1相同)
    metrics['dice'] = metrics['f1']

    return metrics, (tn, fp, fn, tp)


# # ========== 参数 ==========
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CKPT_PATH = "our/checkpoints/best.pth"
# TEST_DIR = "C:/Users/ycc/Desktop/paperList/paper0.8/all-data/--data2/images"
# MASK_DIR = "C:/Users/ycc/Desktop/paperList/paper0.8/all-data/--data2/road_masks"  # 真实掩码目录
# SAVE_DIR = "C:/Users/ycc/Desktop/paperList/paper0.8/all-data/--data2/road_masks_predict"
# SAVE_ATTENTION_DIR = "C:/Users/ycc/Desktop/paperList/paper0.8/all-data/--data2/road_masks_predict/attention_Feiwu"  # 保存注意力掩码
# os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 参数 ==========
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CKPT_PATH = "./checkpoints/best.pth"
# TEST_DIR = "D:/myDataManager/pycharmProject/Crack-Segmentation/road_roi_net/RoadDataset/train/images"
# MASK_DIR = "D:/myDataManager/pycharmProject/Crack-Segmentation/road_roi_net/RoadDataset/train/masks"
# SAVE_DIR = "C:/Users/ycc/Desktop/paperList/paper0.8/12_2/road_predict"
# SAVE_ATTENTION_DIR = "C:/Users/ycc/Desktop/paperList/paper0.8/12_2/attention_Feiwu"  # 保存注意力掩码
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ========== 参数 ==========
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CKPT_PATH = "./checkpoints/best.pth"
# TEST_DIR = "H:/DJI/DJI_202511281344_004/2"
# MASK_DIR = "H:/DJI/DJI_202511281344_004/2"  # 真实掩码目录
# SAVE_DIR = "H:/DJI/DJI_202511281344_004/2/road_masks_predict"
# SAVE_ATTENTION_DIR = "H:/DJI/DJI_202511281344_004/2/attention_Feiwu"  # 保存注意力掩码
# os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 参数 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "our/checkpoints/best.pth"
TEST_DIR = "C:/Users/ycc/Desktop/images1130/22_"
MASK_DIR = "C:/Users/ycc/Desktop/images1130/22road_masks"  # 真实掩码目录
SAVE_DIR = "C:/Users/ycc/Desktop/images1130/22__/road_masks_predict"
SAVE_ATTENTION_DIR = "D:/DJI/DJI_202511281344_004/2/attention_Feiwu"  # 保存注意力掩码
os.makedirs(SAVE_DIR, exist_ok=True)


IMG_SIZE = (256, 256)  # 与训练时保持一致
NUM_CLASSES = 2  # 与训练时保持一致

# ========== 预处理 ==========
transform_img = transforms.Compose([
    transforms.ToTensor()
])

# ========== 加载模型 ==========
model = MultiScale_HSV_UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"[INFO] Model loaded from {CKPT_PATH}")
print(f"[INFO] Testing on {DEVICE}")

# ========== 存储所有指标 ==========
all_metrics = {
    'background_iou': [],
    'road_iou': [],
    'miou': [],
    'background_precision': [],
    'road_precision': [],
    'overall_precision': [],
    'background_recall': [],
    'road_recall': [],
    'overall_recall': [],
    'background_f1': [],
    'road_f1': [],
    'overall_f1': [],
    'fnr': [],
    'fpr': [],
    'dice': []
}

# ========== 推理 ==========
with torch.no_grad():
    test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for name in test_images:
        img_path = os.path.join(TEST_DIR, name)
        mask_path = os.path.join(MASK_DIR, os.path.splitext(name)[0] + '.png')

        # 检查掩码文件是否存在
        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask file not found: {mask_path}")
            continue

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)

        # 读取真实掩码
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gt is None:
            print(f"[WARNING] Cannot read mask file: {mask_path}")
            continue

        # 调整掩码尺寸与预测结果一致
        h, w = img_bgr.shape[:2]
        mask_gt_resized = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_gt_binary = (mask_gt_resized > 127).astype(np.uint8)  # 二值化

        tensor = transform_img(img_resized).unsqueeze(0).to(DEVICE)

        # 新网络输出：logits, attention_mask, thresholds
        logits,_,_ = model(tensor)

        # 处理分割结果
        if NUM_CLASSES == 1:
            # 二分类情况，使用sigmoid
            seg_prob = torch.sigmoid(logits)[0]  # [1, H, W]
        else:
            # 多分类情况，使用softmax并取道路类别（假设道路是类别1）
            seg_prob = F.softmax(logits, dim=1)[0, 1]  # [H, W]，取类别1的概率

        # 后处理分割结果
        seg_result, seg_mask_resized = postprocess_and_save(img_path, seg_prob.unsqueeze(0))

        # 计算指标
        metrics, conf_matrix = calculate_metrics(mask_gt_binary, seg_mask_resized)

        # 存储指标
        all_metrics['background_iou'].append(metrics['iou'][0])
        all_metrics['road_iou'].append(metrics['iou'][1])
        all_metrics['miou'].append(metrics['miou'])
        all_metrics['background_precision'].append(metrics['precision'][0])
        all_metrics['road_precision'].append(metrics['precision'][1])
        all_metrics['overall_precision'].append(metrics['overall_precision'])
        all_metrics['background_recall'].append(metrics['recall'][0])
        all_metrics['road_recall'].append(metrics['recall'][1])
        all_metrics['overall_recall'].append(metrics['overall_recall'])
        all_metrics['background_f1'].append(metrics['f1'][0])
        all_metrics['road_f1'].append(metrics['f1'][1])
        all_metrics['overall_f1'].append(metrics['overall_f1'])
        all_metrics['fnr'].append(metrics['fnr'])
        all_metrics['fpr'].append(metrics['fpr'])
        all_metrics['dice'].append(metrics['dice'][1])  # 道路类别的dice

        # ---- 保存分割结果 ----
        fname = os.path.splitext(name)[0]
        seg_out_path = os.path.join(SAVE_DIR, f"{fname}_seg.jpg")
        cv2.imwrite(seg_out_path, seg_result)

        # ---- 保存原始预测掩码（用于可视化） ----
        seg_mask_vis = (seg_mask_resized * 255).astype(np.uint8)
        seg_mask_path = os.path.join(SAVE_DIR, f"{fname}_mask.png")
        cv2.imwrite(seg_mask_path, seg_mask_vis)

        print(f"[Processed] {name} | mIOU: {metrics['miou']:.4f} | Road F1: {metrics['f1'][1]:.4f}")

print(f"\n[INFO] Testing completed! Results saved in {SAVE_DIR} and {SAVE_ATTENTION_DIR}")


# ========== 计算平均指标并制表输出 ==========
def create_metrics_table(metrics_dict):
    """创建美化的指标表格"""
    # 计算平均值
    avg_metrics = {key: np.mean(values) for key, values in metrics_dict.items()}

    # 创建表格数据
    table_data = [
        ["指标", "背景", "道路", "整体"],
        ["IOU", f"{avg_metrics['background_iou']:.4f}", f"{avg_metrics['road_iou']:.4f}", f"{avg_metrics['miou']:.4f}"],
        ["Precision", f"{avg_metrics['background_precision']:.4f}", f"{avg_metrics['road_precision']:.4f}",
         f"{avg_metrics['overall_precision']:.4f}"],
        ["Recall", f"{avg_metrics['background_recall']:.4f}", f"{avg_metrics['road_recall']:.4f}",
         f"{avg_metrics['overall_recall']:.4f}"],
        ["F1-Score", f"{avg_metrics['background_f1']:.4f}", f"{avg_metrics['road_f1']:.4f}",
         f"{avg_metrics['overall_f1']:.4f}"],
        ["", "", "", ""],
        ["FNR", "-", f"{avg_metrics['fnr']:.4f}", "-"],
        ["FPR", "-", f"{avg_metrics['fpr']:.4f}", "-"],
        ["Dice", "-", f"{avg_metrics['dice']:.4f}", "-"]
    ]

    return table_data, avg_metrics


# 生成并打印表格
print("\n" + "=" * 60)
print("               模型评估指标汇总")
print("=" * 60)

table_data, avg_metrics = create_metrics_table(all_metrics)
print(tabulate(table_data, headers="firstrow", tablefmt="grid", stralign="center"))

# 额外信息
print(f"\n测试图像数量: {len(test_images)}")
print(f"平均指标基于 {len(all_metrics['miou'])} 张有效图像计算")

# 保存详细指标到CSV文件
detailed_metrics_df = pd.DataFrame(all_metrics)
detailed_metrics_df.to_csv(os.path.join(SAVE_DIR, "detailed_metrics.csv"), index=False)
print(f"\n详细指标已保存至: {os.path.join(SAVE_DIR, 'detailed_metrics.csv')}")

# 保存汇总指标
summary_metrics = {
    'Metric': ['mIOU', 'Road_IOU', 'Background_IOU', 'Road_F1', 'Overall_F1', 'Road_Precision', 'Road_Recall', 'FNR',
               'FPR', 'Dice'],
    'Value': [
        avg_metrics['miou'],
        avg_metrics['road_iou'],
        avg_metrics['background_iou'],
        avg_metrics['road_f1'],
        avg_metrics['overall_f1'],
        avg_metrics['road_precision'],
        avg_metrics['road_recall'],
        avg_metrics['fnr'],
        avg_metrics['fpr'],
        avg_metrics['dice']
    ]
}
summary_df = pd.DataFrame(summary_metrics)
summary_df.to_csv(os.path.join(SAVE_DIR, "summary_metrics.csv"), index=False)
print(f"汇总指标已保存至: {os.path.join(SAVE_DIR, 'summary_metrics.csv')}")