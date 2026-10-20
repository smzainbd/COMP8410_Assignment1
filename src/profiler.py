import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

METRICS_CSV_PATH = "results/metrics/metrics.csv"


def load_fashion_data(data_dir="./data", train_n=2048, test_n=512):
    """Loads the first train_n and test_n Fashion-MNIST samples onto DEVICE."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    tf = transforms.Compose([transforms.ToTensor()])
    tr_set = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tf)
    te_set = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tf)

    x_tr = torch.stack([tr_set[i][0] for i in range(train_n)]).to(DEVICE)
    y_tr = torch.tensor([tr_set[i][1] for i in range(train_n)], device=DEVICE)
    x_te = torch.stack([te_set[i][0] for i in range(test_n)]).to(DEVICE)
    y_te = torch.tensor([te_set[i][1] for i in range(test_n)], device=DEVICE)

    return x_tr, y_tr, x_te, y_te


def compute_error_stats(pred_clean, target_clean, pred_perturbed, target_perturbed, y_labels, train_split_mse):
    """
    Computes:
    - test_mse: MSE(pred_clean, target_clean)
    - perturbed_mse: MSE(pred_perturbed, target_perturbed)
    - output_drift_mse: MSE(pred_perturbed, pred_clean)
    - gen_gap_abs: |train_split_mse - test_mse|
    - class_mse_std: std of per-class test MSE (or NaN when no class labels exist)
    """
    with torch.no_grad():
        test_mse = F.mse_loss(pred_clean, target_clean).item()
        perturbed_mse = F.mse_loss(pred_perturbed, target_perturbed).item()
        output_drift_mse = F.mse_loss(pred_perturbed, pred_clean).item()
        gen_gap_abs = abs(float(train_split_mse) - test_mse)

        if y_labels is not None and pred_clean.size(0) == y_labels.size(0):
            class_mses = []
            for c in torch.unique(y_labels):
                mask = (y_labels == c)
                if mask.sum() > 0:
                    class_mses.append(F.mse_loss(pred_clean[mask], target_clean[mask]).item())
            class_mse_std = round(float(np.std(class_mses)), 5) if len(class_mses) > 1 else np.nan
        else:
            class_mse_std = np.nan

    return (
        round(test_mse, 5),
        round(perturbed_mse, 5),
        round(output_drift_mse, 5),
        round(gen_gap_abs, 5),
        class_mse_std
    )


def record_run(pair, week, system, model_name, model, batch_unit, train_s, batch_lat_ms, vram_mb,
               train_split_mse, test_mse, perturbed_mse, output_drift_mse, gen_gap_abs, class_mse_std,
               aux_metric_name="None", aux_metric_val=np.nan,
               config_path="configs/sla_config.json", csv_path=METRICS_CSV_PATH):
    """Records measured stats, batch timing unit, output drift, and auxiliary metric to disk."""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    max_lat = float(cfg["max_latency_ms"])
    tdp_w = float(cfg["gpu_tdp_watts"])
    hr_usd = float(cfg["cloud_gpu_hour_usd"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    batches_per_sec = 1000.0 / max(batch_lat_ms, 0.001)

    est_energy_mwh_per_1k_batches = (tdp_w * (batch_lat_ms / 1000.0) * 1000.0) / 3.6
    est_cost_usd_per_100k_batches = ((batch_lat_ms / 1000.0) * 100000.0 / 3600.0) * hr_usd
    within_latency_target = bool(batch_lat_ms <= max_lat)

    row = {
        "Pair": pair,
        "Week": week,
        "System": system,
        "Model": model_name,
        "Total_Params": total_params,
        "Trainable_Params": trainable_params,
        "Train_Time_s": round(train_s, 3),
        "Batch_Unit": batch_unit,
        "Batch_Latency_ms": round(batch_lat_ms, 3),
        "Batches_Per_Sec": round(batches_per_sec, 1),
        "Peak_VRAM_MB": round(vram_mb, 2),
        "Train_Split_MSE": round(float(train_split_mse), 5),
        "Test_MSE": test_mse,
        "Perturbed_Test_MSE": perturbed_mse,
        "Output_Drift_MSE": output_drift_mse,
        "Gen_Gap_Abs": gen_gap_abs,
        "Class_MSE_Std": class_mse_std,
        "Aux_Metric_Name": aux_metric_name,
        "Aux_Metric_Val": round(float(aux_metric_val), 5) if not np.isnan(aux_metric_val) else np.nan,
        "Est_Energy_mWh_1kBatches_Formula": round(est_energy_mwh_per_1k_batches, 2),
        "Est_Cost_100kBatches_USD_Formula": round(est_cost_usd_per_100k_batches, 4),
        "Meets_Latency_Target": within_latency_target
    }

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        df_disk = pd.read_csv(csv_path)
        if set(df_disk.columns) != set(row.keys()):
            df_updated = pd.DataFrame([row])
        else:
            df_disk = df_disk[~((df_disk["Pair"] == pair) & (df_disk["System"] == system))]
            df_updated = pd.concat([df_disk, pd.DataFrame([row])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([row])

    df_updated = df_updated.sort_values(by=["Pair", "System"]).reset_index(drop=True)
    df_updated.to_csv(csv_path, index=False)

    print(
        f"[{pair} | {system}] {model_name:36s} | "
        f"TestMSE: {test_mse:.5f} | PertMSE: {perturbed_mse:.5f} | Drift: {output_drift_mse:.5f} | "
        f"Lat({batch_unit}): {batch_lat_ms:.3f}ms"
    )
    return row