import torch
from importlib import import_module
import pandas as pd


def dice(y_true, y_pred, reduce_axes=[-2, 1], beta=1., epsilon=1e-6):
    beta2 = beta ** 2  # beta squared
    numerator = (1 + beta2) * (y_true * y_pred).sum(reduce_axes)
    denominator = beta2 * y_true.square().sum(reduce_axes) + y_pred.square().sum(reduce_axes)
    denominator = denominator.clamp(min=epsilon)
    return numerator / denominator


def soft_dice(pred, target, epsilon=1e-6):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice


def calculate_f1_score(pred_list, target_list):
    # 计算 F1 分数
    true_positives = torch.sum(pred_list * target_list)
    false_positives = torch.sum(pred_list * (1 - target_list))
    false_negatives = torch.sum((1 - pred_list) * target_list)
    precision = true_positives / (true_positives + false_positives + 1e-7)  # 添加一个很小的数以避免除以0
    recall = true_positives / (true_positives + false_negatives + 1e-7)  # 添加一个很小的数以避免除以0
    f1_score = (2 * precision * recall) / (precision + recall + 1e-7)  # 添加一个很小的数以避免除以0
    return f1_score


def calculate_normalized_surface_dice(pred_list, target_list):
    # 计算归一化表面 Dice 系数
    surface_distances = compute_surface_distances(pred_list, target_list)
    hausdorff_distance = compute_robust_hausdorff(surface_distances, 95)
    normalized_surface_dice = 1 - hausdorff_distance
    return normalized_surface_dice


def calculate_iou(pred_list, target_list):
    # 计算 IoU (Intersection over Union)
    intersection = torch.sum(pred_list * target_list)
    union = torch.sum(pred_list) + torch.sum(target_list) - intersection
    iou = intersection / (union + 1e-7)  # 添加一个很小的数以避免除以0
    return iou


def calculate_mae(pred_list, target_list):
    # 计算 MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(pred_list - target_list))
    return mae


def calculate_sensitivity(pred_list, target_list):
    # 计算敏感度 (Sensitivity)，也称为真阳性率（True Positive Rate）
    true_positives = torch.sum(pred_list * target_list)
    actual_positives = torch.sum(target_list)
    sensitivity = true_positives / (actual_positives + 1e-7)  # 添加一个很小的数以避免除以0
    return sensitivity


def calculate_recall(pred_list, target_list):
    # 计算召回率 (Recall)，也称为灵敏度（Sensitivity）或真阳性率（True Positive Rate）
    true_positives = torch.sum(pred_list * target_list)
    false_negatives = torch.sum((1 - pred_list) * target_list)
    recall = true_positives / (true_positives + false_negatives + 1e-7)  # 添加一个很小的数以避免除以0
    return recall


models = [
    {'file': 'Test_UNet', 'function': 'inference'},
    {'file': 'Test_UNetPlusPlus', 'function': 'inference'},
    {'file': 'Test_LungInf', 'function': 'inference'},
    {'file': 'mytest_VGG', 'function': 'inference'},
    {'file': 'mytest_VGG_CLAHE', 'function': 'inference'},
]

metrics = {
    'dice_coefficient': dice,
    'sensitivity': calculate_sensitivity,
    'f1_score': calculate_f1_score,
    'iou': calculate_iou,
    'mae': calculate_mae,
    # 'recall': calculate_recall
    # 'normalized_surface_dice': calculate_normalized_surface_dice,
}

results = {}
test_time = []
import time

for model in models:
    module = import_module(model['file'])
    inference_func = getattr(module, model['function'])
    star_time = time.time()
    pred, target = inference_func()
    end_time = time.time()
    test_time.append(end_time - star_time)

    for metric_name, compute_metric in metrics.items():
        if metric_name not in results:
            results[metric_name] = []
        metric_value = compute_metric(pred, target)
        results[metric_name].append(metric_value)

print(results)
print(test_time)
