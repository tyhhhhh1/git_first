import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from dataset import NMRSepDataset
from model import NMRSepFormer
from loss import AreaDominantQuantitativeLoss


PROJECT_DIR = r'/home/ty/PycharmProjects/数据集双'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 50
TRAIN_SIZE = 140000
VAL_SIZE = 10000
VAL_STEPS = VAL_SIZE // BATCH_SIZE
VIS_EVERY = 7
EXPERIMENTAL_AUX_WEIGHT = 0.0
EXPERIMENTAL_AUX_FRACTION_WEIGHT = 1.0
EXPERIMENTAL_AUX_RECON_WEIGHT = 0.35
EXPERIMENTAL_AUX_SILENCE_WEIGHT = 0.25
GLOBAL_SEED = 20260617
EVAL_PRESENCE_THRESHOLD = 0.02
TRAIN_ACTIVE_THRESHOLD = 1e-4
EARLY_STOP_PATIENCE = 10
PHASE1_END_RATIO = 0.25
PHASE2_END_RATIO = 0.60
SAVE_PATH = r'/home/ty/PycharmProjects/数据集双/checkpoints'
os.makedirs(SAVE_PATH, exist_ok=True)


CSV_PATH = r'/home/ty/PycharmProjects/数据集双/AminoAcids20_lib.csv'
PRETRAINED_PATH = r'/home/ty/PycharmProjects/数据集双/checkpoints/bbest_model_serum_falsepeak_refine.pth'
REAL_SERUM_RECIPE_CSV = r'/home/ty/PycharmProjects/结合loss'
EXPERIMENTAL_PURE_DIR = r'/home/ty/PycharmProjects/数据集双/network_ready'

FINETUNED_SAVE_PATH = r'/home/ty/PycharmProjects/数据集双/checkpoints/best_model_experimental_area_refine.pth'
BEST_BY_SCORE_PATH = os.path.join(SAVE_PATH, "best_by_score.pth")
BEST_BY_TABLE_S1_EXTERNAL_RAW_MAPE_PATH = os.path.join(SAVE_PATH, "best_by_table_s1_external_raw_mape.pth")
BEST_BY_LOW_FALSE_POSITIVE_PATH = os.path.join(SAVE_PATH, "best_by_low_false_positive.pth")
BEST_BY_OVERLAP_LOW_FP_PATH = os.path.join(SAVE_PATH, "best_by_overlap_low_fp.pth")
LATEST_EPOCH_PATH = os.path.join(SAVE_PATH, "latest_epoch.pth")

# Table S1 is used only as a calibration/holdout diagnostic. The main training
# stream remains random experimental-pure-spectrum superposition.
TABLE_S1_CALIBRATION_INDICES = (0, 2, 4, 6, 8)
TABLE_S1_EXTERNAL_INDICES = (1, 3, 5, 7, 9)
TABLE_S1_EVAL_EVERY = 1
TABLE_S1_EVAL_REPEATS = 1
TABLE_S1_EVAL_AUGMENT = False

SERUM_FOCUS = ['Asparagine', 'Glutamine', 'Glutamate', 'Isoleucine', 'Leucine', 'Proline', 'Valine']
HARD_FP_WEIGHTS = {
    'Proline': 3.0,
    'Leucine': 2.0,
    'Glutamate': 1.5,
    'Glutamine': 1.5,
    'Asparagine': 1.0,
    'Isoleucine': 1.5,
    'Valine': 1.5,
}
BCAA_FP_WEIGHTS = {
    'Leucine': 2.0,
    'Isoleucine': 1.5,
    'Valine': 1.5,
}
OVERLAP_GROUPS = [
    ('Glutamine', 'Glutamate', 'Asparagine'),
    ('Isoleucine', 'Leucine', 'Valine', 'Proline'),
    ('Alanine', 'Serine', 'Glycine', 'Threonine'),
]


def validate_training_inputs():
    required_files = [
        ("peak feature library CSV", CSV_PATH),
        ("dataset.py", os.path.join(PROJECT_DIR, "dataset.py")),
        ("model.py", os.path.join(PROJECT_DIR, "model.py")),
        ("loss.py", os.path.join(PROJECT_DIR, "loss.py")),
    ]
    required_dirs = [
        ("checkpoint directory", SAVE_PATH),
    ]

    missing = []
    for label, path in required_files:
        if not os.path.isfile(path):
            missing.append(f"{label}: {path}")
    for label, path in required_dirs:
        if not os.path.isdir(path):
            missing.append(f"{label}: {path}")

    if os.path.isdir(EXPERIMENTAL_PURE_DIR):
        network_ready_files = [
            os.path.join(EXPERIMENTAL_PURE_DIR, "metadata.json"),
            os.path.join(EXPERIMENTAL_PURE_DIR, "ppm_grid_float32.npy"),
            os.path.join(EXPERIMENTAL_PURE_DIR, "pure_spectra_float32.npy"),
        ]
        has_network_ready = all(os.path.exists(path) for path in network_ready_files)
        pure_csv_count = sum(
            1 for name in os.listdir(EXPERIMENTAL_PURE_DIR)
            if name.lower().endswith(".csv")
        )
        if not has_network_ready and pure_csv_count < 20:
            missing.append(
                f"experimental pure spectra not found as network_ready or 20 CSVs: {EXPERIMENTAL_PURE_DIR}"
            )
        if has_network_ready:
            print(f"  [file] experimental pure spectra: network_ready in {EXPERIMENTAL_PURE_DIR}")
        else:
            print(f"  [file] experimental pure spectrum CSVs: {pure_csv_count} in {EXPERIMENTAL_PURE_DIR}")
    else:
        missing.append(f"experimental pure spectrum directory: {EXPERIMENTAL_PURE_DIR}")

    if not os.path.exists(PRETRAINED_PATH):
        missing.append(f"selected pretrained checkpoint: {PRETRAINED_PATH}")

    if missing:
        raise FileNotFoundError("Missing training inputs:\n" + "\n".join(f"- {item}" for item in missing))

    print("Training input check passed:")
    for label, path in required_files:
        print(f"  [file] {label}: {path}")
    for label, path in required_dirs:
        print(f"  [dir]  {label}: {path}")
    print(f"  [file] selected pretrained checkpoint: {PRETRAINED_PATH}")
    if os.path.isdir(REAL_SERUM_RECIPE_CSV):
        recipe_count = sum(
            1 for name in os.listdir(REAL_SERUM_RECIPE_CSV)
            if name.startswith("def_mix_serum_") and name.lower().endswith(".csv")
        )
        print(f"  [optional] real serum recipe CSVs: {recipe_count} in {REAL_SERUM_RECIPE_CSV}")

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_fixed_validation_batches(dataset, steps, batch_size, seed=GLOBAL_SEED + 1000):
    rng_np = np.random.get_state()
    rng_py = random.getstate()
    rng_torch = torch.random.get_rng_state()

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    batches = []
    for step in range(steps):
        mixtures, targets = [], []
        for j in range(batch_size):
            mixture, target, _ = dataset[step * batch_size + j]
            mixtures.append(mixture)
            targets.append(target)
        batches.append((torch.stack(mixtures, dim=0), torch.stack(targets, dim=0)))

    np.random.set_state(rng_np)
    random.setstate(rng_py)
    torch.random.set_rng_state(rng_torch)
    return batches


def evaluate_classification_metrics(pred_maxes, target_maxes, names, threshold=0.02, verbose=True):
    pred_binary = (pred_maxes > threshold).astype(int)
    target_binary = (target_maxes > threshold).astype(int)
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    fpr = fp / (tn + fp + 1e-8)
    fnr = fn / (tp + fn + 1e-8)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)

    if verbose:
        print(f"   [classification] TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"   [rates] TPR={tpr * 100:.2f}% | TNR={tnr * 100:.2f}% | FPR={fpr * 100:.2f}% | FNR={fnr * 100:.2f}%")
        print(f"   [F1] {f1 * 100:.2f}%")
        print("   [per-compound recall]")
    per_compound = {}
    for idx, name in enumerate(names):
        if name not in SERUM_FOCUS:
            continue
        hit = ((pred_binary[:, idx] == 1) & (target_binary[:, idx] == 1)).sum()
        false_negative = ((pred_binary[:, idx] == 0) & (target_binary[:, idx] == 1)).sum()
        false_positive = ((pred_binary[:, idx] == 1) & (target_binary[:, idx] == 0)).sum()
        positives = hit + false_negative
        predicted = hit + false_positive
        recall = hit / (positives + 1e-8)
        compound_precision = hit / (predicted + 1e-8)
        per_compound[name] = {
            'tp': int(hit),
            'fp': int(false_positive),
            'fn': int(false_negative),
            'recall': float(recall),
            'precision': float(compound_precision),
        }
        if verbose:
            print(
                f"      {name:12s} recall={recall * 100:6.2f}% | "
                f"precision={compound_precision * 100:6.2f}% | "
                f"TP={hit:4d} FN={false_negative:4d} FP={false_positive:4d}"
            )

    # Serum recall is a better best-model signal than global TN-heavy accuracy.
    serum_indices = [i for i, name in enumerate(names) if name in SERUM_FOCUS]
    serum_tp = ((pred_binary[:, serum_indices] == 1) & (target_binary[:, serum_indices] == 1)).sum()
    serum_fn = ((pred_binary[:, serum_indices] == 0) & (target_binary[:, serum_indices] == 1)).sum()
    serum_recall = serum_tp / (serum_tp + serum_fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return f1, serum_recall, fpr, precision, per_compound


def calculate_hard_fp_score(per_compound):
    score = 0.0
    for name, weight in HARD_FP_WEIGHTS.items():
        score += weight * per_compound.get(name, {}).get('fp', 0)
    return float(score)


def calculate_bcaa_fp_score(per_compound):
    score = 0.0
    for name, weight in BCAA_FP_WEIGHTS.items():
        score += weight * per_compound.get(name, {}).get('fp', 0)
    return float(score)


def conditional_bcaa_false_positives(pred_maxes, target_maxes, names, threshold=0.02):
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    pred_binary = pred_maxes > threshold
    target_binary = target_maxes > threshold

    def count_false_positive_when(target_name, trigger_names):
        if target_name not in name_to_idx:
            return 0
        trigger_idx = [name_to_idx[name] for name in trigger_names if name in name_to_idx]
        if not trigger_idx:
            return 0
        target_idx = name_to_idx[target_name]
        target_absent = ~target_binary[:, target_idx]
        target_predicted = pred_binary[:, target_idx]
        trigger_active = target_binary[:, trigger_idx].any(axis=1)
        return int((target_absent & target_predicted & trigger_active).sum())

    return {
        'Leucine_FP_when_Ile_or_Val_active': count_false_positive_when(
            'Leucine', ('Isoleucine', 'Valine')
        ),
        'Isoleucine_FP_when_Leu_or_Val_active': count_false_positive_when(
            'Isoleucine', ('Leucine', 'Valine')
        ),
        'Valine_FP_when_Ile_or_Leu_active': count_false_positive_when(
            'Valine', ('Isoleucine', 'Leucine')
        ),
    }


def inactive_peak_metrics(pred_maxes, target_maxes):
    inactive = target_maxes <= TRAIN_ACTIVE_THRESHOLD
    values = pred_maxes[inactive]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    values = np.maximum(values, 0.0)
    return float(np.mean(values)), float(np.percentile(values, 95)), float(np.max(values))


def inactive_area_for_compounds(pred_areas, target_maxes, names, compounds):
    indices = [idx for idx, name in enumerate(names) if name in compounds]
    if not indices:
        return 0.0
    inactive = target_maxes[:, indices] <= TRAIN_ACTIVE_THRESHOLD
    values = np.maximum(pred_areas[:, indices], 0.0)
    if inactive.sum() <= 0:
        return 0.0
    return float(values[inactive].mean())


def inactive_peak_p95_for_compounds(pred_maxes, target_maxes, names, compounds):
    indices = [idx for idx, name in enumerate(names) if name in compounds]
    if not indices:
        return 0.0
    inactive = target_maxes[:, indices] <= TRAIN_ACTIVE_THRESHOLD
    values = np.maximum(pred_maxes[:, indices], 0.0)
    if inactive.sum() <= 0:
        return 0.0
    return float(np.percentile(values[inactive], 95))


def experimental_quantitative_aux_loss(predictions, background, targets, mixture_linear):
    pred_area = F.relu(predictions).sum(dim=-1)
    target_area = targets.sum(dim=-1)
    active = (target_area > TRAIN_ACTIVE_THRESHOLD).float()
    inactive = 1.0 - active

    pred_fraction = pred_area / torch.clamp(pred_area.sum(dim=-1, keepdim=True), min=1e-6)
    target_fraction = target_area / torch.clamp(target_area.sum(dim=-1, keepdim=True), min=1e-6)
    fraction_loss = (torch.abs(pred_fraction - target_fraction) * active).sum() / (active.sum() + 1e-8)

    pred_sum = predictions.sum(dim=1, keepdim=True) + background
    recon_loss = F.l1_loss(
        F.avg_pool1d(pred_sum, kernel_size=4, stride=4),
        F.avg_pool1d(mixture_linear, kernel_size=4, stride=4),
    )

    inactive_area_loss = (pred_area * inactive).sum() / (inactive.sum() + 1e-8)
    return (
        EXPERIMENTAL_AUX_FRACTION_WEIGHT * fraction_loss
        + EXPERIMENTAL_AUX_RECON_WEIGHT * recon_loss
        + EXPERIMENTAL_AUX_SILENCE_WEIGHT * inactive_area_loss
    )


def area_validation_metrics(predictions, targets):
    pred_area = F.relu(predictions).sum(dim=-1)
    target_area = targets.sum(dim=-1)
    active = (target_area > TRAIN_ACTIVE_THRESHOLD).float()
    if active.sum() <= 0:
        zero = predictions.new_tensor(0.0)
        return zero, zero

    pred_fraction = pred_area / torch.clamp(pred_area.sum(dim=-1, keepdim=True), min=1e-6)
    target_fraction = target_area / torch.clamp(target_area.sum(dim=-1, keepdim=True), min=1e-6)
    fraction_mae = (torch.abs(pred_fraction - target_fraction) * active).sum() / (active.sum() + 1e-8)

    target_total = torch.clamp(target_area.sum(dim=-1, keepdim=True), min=1e-6)
    rel_denom = torch.clamp(target_area + 0.015 * target_total, min=1e-6)
    recovery_mape = (
        torch.abs(pred_area - target_area) / rel_denom * active
    ).sum() / (active.sum() + 1e-8) * 100.0
    return fraction_mae, recovery_mape


def _area_matrix_from_batches(model, dataset, recipe_indices, repeats=TABLE_S1_EVAL_REPEATS,
                              augment=TABLE_S1_EVAL_AUGMENT):
    model.eval()
    rows = []
    with torch.no_grad():
        for recipe_idx in recipe_indices:
            if recipe_idx >= len(dataset.table_s1_recipes):
                continue
            recipe = dataset.table_s1_recipes[recipe_idx]
            for repeat in range(repeats):
                sample = dataset.generate_experimental_recipe_sample(recipe, augment=augment)
                if sample is None:
                    continue
                mixture, targets, _ = sample
                mixture = mixture.unsqueeze(0).to(DEVICE)
                targets = targets.unsqueeze(0).to(DEVICE)
                with torch.cuda.amp.autocast():
                    predictions, _ = model(mixture)
                pred_area = F.relu(predictions).sum(dim=-1).squeeze(0).float().cpu().numpy()
                target_area = targets.sum(dim=-1).squeeze(0).float().cpu().numpy()
                for name, idx in dataset.mapping.items():
                    if target_area[idx] <= TRAIN_ACTIVE_THRESHOLD:
                        continue
                    rows.append((recipe_idx + 1, name, float(pred_area[idx]), float(target_area[idx])))
    return rows


def _fit_global_area_calibration(rows):
    by_name = {}
    for _, name, pred_area, target_area in rows:
        if pred_area <= 1e-8 or target_area <= 1e-8:
            continue
        by_name.setdefault(name, []).append(target_area / pred_area)

    factors = {}
    for name, values in by_name.items():
        arr = np.asarray(values, dtype=np.float64)
        factors[name] = float(np.clip(np.median(arr), 0.25, 4.0))
    return factors


def _summarize_area_rows(rows, factors=None):
    if not rows:
        return {
            'count': 0,
            'mape': float('nan'),
            'mae_area': float('nan'),
            'bias': float('nan'),
            'recovery': float('nan'),
        }

    errors = []
    abs_errors = []
    signed_errors = []
    recoveries = []
    for _, name, pred_area, target_area in rows:
        factor = 1.0 if factors is None else factors.get(name, 1.0)
        pred_cal = pred_area * factor
        signed = pred_cal - target_area
        signed_errors.append(signed)
        abs_errors.append(abs(signed))
        errors.append(abs(signed) / max(target_area, 1e-8) * 100.0)
        recoveries.append(pred_cal / max(target_area, 1e-8) * 100.0)
    return {
        'count': len(rows),
        'mape': float(np.mean(errors)),
        'mae_area': float(np.mean(abs_errors)),
        'bias': float(np.mean(signed_errors)),
        'recovery': float(np.mean(recoveries)),
    }


def evaluate_table_s1_global_calibration(model, dataset, repeats=TABLE_S1_EVAL_REPEATS,
                                         augment=TABLE_S1_EVAL_AUGMENT):
    calib_rows = _area_matrix_from_batches(
        model, dataset, TABLE_S1_CALIBRATION_INDICES, repeats=repeats, augment=augment
    )
    external_rows = _area_matrix_from_batches(
        model, dataset, TABLE_S1_EXTERNAL_INDICES, repeats=repeats, augment=augment
    )
    factors = _fit_global_area_calibration(calib_rows)
    calib_raw = _summarize_area_rows(calib_rows)
    calib_cal = _summarize_area_rows(calib_rows, factors=factors)
    external_raw = _summarize_area_rows(external_rows)
    external_cal = _summarize_area_rows(external_rows, factors=factors)
    return factors, calib_raw, calib_cal, external_raw, external_cal


def print_table_s1_calibration_report(model, dataset):
    factors, calib_raw, calib_cal, external_raw, external_cal = evaluate_table_s1_global_calibration(model, dataset)
    print(
        "   [Fixed Table S1 global area calibration] "
        f"calib_raw_MAPE={calib_raw['mape']:.2f}% -> calib_cal_MAPE={calib_cal['mape']:.2f}% | "
        f"external_raw_MAPE={external_raw['mape']:.2f}% -> external_cal_MAPE={external_cal['mape']:.2f}%"
    )
    if factors:
        focus = {name: factors[name] for name in SERUM_FOCUS if name in factors}
        if focus:
            factor_text = ", ".join(f"{name}={value:.3f}" for name, value in focus.items())
            print(f"   [global factors from calibration split] {factor_text}")
    return factors, calib_raw, calib_cal, external_raw, external_cal


def save_global_calibration_factors(factors, epoch):
    if not factors:
        return
    out_path = os.path.join(SAVE_PATH, "table_s1_global_area_factors.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Epoch,Metabolite,Global_Area_Factor\n")
        for name in sorted(factors):
            f.write(f"{epoch},{name},{factors[name]:.8g}\n")
    print(f"Saved Table S1 global area factors: {out_path}")


def run_training():
    seed_everything(GLOBAL_SEED)
    validate_training_inputs()
    print(f"Starting serum-style area-refinement fine-tuning on {DEVICE}")

    train_dataset = NMRSepDataset(
        CSV_PATH,
        serum_finetune=False,
        serum_mode='standard',
        real_serum_recipe_csv=REAL_SERUM_RECIPE_CSV,
        real_serum_recipe_prob=0.0,
        real_serum_mix_prob=0.0,
        overlap_recipe_prob=0.0,
        low_conc_prob=0.0,
        experimental_pure_dir=EXPERIMENTAL_PURE_DIR,
        experimental_mix_prob=1.0,
        experimental_min_components=2,
        experimental_max_components=14,
        pdf_method_mix=True,
        pdf_method1_prob=1.0,
        dataset_length=TRAIN_SIZE,
        conc_range=(0.1, 1.0),
        lw_range=(500.0 / 550.0, 500.0 / 450.0),
        eta_range=(0.0, 1.0),
        global_shift_range=(-0.030, 0.030),
        jitter_range=(-0.0010, 0.0010),
        experimental_scale_range=(0.1, 1.0),
        experimental_shift_range=(-0.03, 0.03),
        experimental_noise_db_range=(35.0, 50.0),
        proline_hard_negative_prob=0.20,
        bcaa_hard_negative_prob=0.25,
    )
    if len(train_dataset.experimental_templates) < 20:
        raise RuntimeError(
            "Experimental pure spectra were not fully loaded. "
            f"Loaded {len(train_dataset.experimental_templates)} templates from {EXPERIMENTAL_PURE_DIR}. "
            "Please check EXPERIMENTAL_PURE_DIR and network_ready files."
        )
    print(
        f"Experimental-superposition training enabled: "
        f"{len(train_dataset.experimental_templates)} pure spectra | "
        f"components={train_dataset.experimental_min_components}-{train_dataset.experimental_max_components}"
    )
    experimental_aux_dataset = NMRSepDataset(
        CSV_PATH,
        serum_finetune=False,
        serum_mode='standard',
        experimental_pure_dir=EXPERIMENTAL_PURE_DIR,
        experimental_mix_prob=1.0,
        experimental_min_components=2,
        experimental_max_components=14,
        experimental_scale_range=(0.1, 1.0),
        experimental_shift_range=(-0.03, 0.03),
        experimental_noise_db_range=(35.0, 50.0),
        proline_hard_negative_prob=0.20,
        bcaa_hard_negative_prob=0.25,
        conc_range=(1.0, 10.0),
        global_shift_range=(-0.020, 0.020),
        jitter_range=(-0.0010, 0.0010),
    )
    val_dataset = NMRSepDataset(
        CSV_PATH,
        serum_finetune=False,
        serum_mode='standard',
        real_serum_recipe_csv=REAL_SERUM_RECIPE_CSV,
        real_serum_recipe_prob=0.0,
        real_serum_mix_prob=0.0,
        overlap_recipe_prob=0.0,
        low_conc_prob=0.0,
        experimental_pure_dir=EXPERIMENTAL_PURE_DIR,
        experimental_mix_prob=1.0,
        experimental_min_components=2,
        experimental_max_components=14,
        pdf_method_mix=True,
        pdf_method1_prob=1.0,
        dataset_length=VAL_SIZE,
        conc_range=(0.1, 1.0),
        lw_range=(500.0 / 550.0, 500.0 / 450.0),
        eta_range=(0.0, 1.0),
        global_shift_range=(-0.030, 0.030),
        jitter_range=(-0.0010, 0.0010),
        experimental_scale_range=(0.1, 1.0),
        experimental_shift_range=(-0.03, 0.03),
        experimental_noise_db_range=(35.0, 50.0),
        proline_hard_negative_prob=0.20,
        bcaa_hard_negative_prob=0.25,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, prefetch_factor=2,
        persistent_workers=True, worker_init_fn=worker_init_fn
    )
    experimental_aux_loader = DataLoader(
        experimental_aux_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, prefetch_factor=2,
        persistent_workers=True, worker_init_fn=worker_init_fn
    )
    experimental_aux_iter = iter(experimental_aux_loader) if EXPERIMENTAL_AUX_WEIGHT > 0 else None
    fixed_val_batches = build_fixed_validation_batches(val_dataset, VAL_STEPS, BATCH_SIZE)
    print(
        f"Fixed validation set: {len(fixed_val_batches) * BATCH_SIZE} spectra | "
        f"eval_threshold={EVAL_PRESENCE_THRESHOLD} | train_active_threshold={TRAIN_ACTIVE_THRESHOLD}"
    )

    model = NMRSepFormer(num_compounds=train_dataset.num_classes, d_model=128).to(DEVICE)
    prototype_templates = {
        name: train_dataset.experimental_templates.get(name, train_dataset.templates[name])
        for name in train_dataset.names
    }
    model.init_prototypes_from_templates(prototype_templates, DEVICE)

    state = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"Loaded selected pretrained weights: {PRETRAINED_PATH}")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    loss_fn = AreaDominantQuantitativeLoss(
        area_fraction_w=7.0,
        area_recovery_w=3.5,
        group_fraction_w=2.0,
        shape_w=0.20,
        recon_w=0.25,
        presence_w=0.30,
        inactive_w=2.0,
        local_false_w=0.30,
        excess_w=0.50,
        presence_threshold=EVAL_PRESENCE_THRESHOLD,
        target_presence_threshold=TRAIN_ACTIVE_THRESHOLD,
        negative_presence_weight=2.2,
        hard_channel_boost=1.35,
        proline_absent_w=3.0,
        proline_absent_peak_w=1.5,
        proline_peak_threshold=0.015,
        leucine_absent_w=1.5,
        leucine_absent_peak_w=0.75,
        leucine_peak_threshold=0.015,
        mapping=train_dataset.mapping,
    ).to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    best_score = -float('inf')
    best_external_raw_mape = float('inf')
    best_hard_fp_score = float('inf')
    best_bcaa_fp_score = float('inf')
    epochs_without_external_or_hard_improvement = 0
    phase1_end = max(1, int(EPOCHS * PHASE1_END_RATIO))
    phase2_end = max(phase1_end + 1, int(EPOCHS * PHASE2_END_RATIO))

    for epoch in range(EPOCHS):
        if epoch < phase1_end:
            phase = "Phase 1: area-ratio stabilization"
        elif epoch < phase2_end:
            phase = "Phase 2: overlap area disentanglement"
        else:
            phase = "Phase 3: quantitative calibration"

        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} | {phase} ---")
        model.train()
        total_train_loss = 0.0

        for mixture, targets, _ in train_loader:
            mixture = mixture.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                predictions, background = model(mixture)
                mixture_linear = mixture[:, 0:1, :]
                loss, loss_dict = loss_fn(predictions, targets, background, mixture_linear)

                if EXPERIMENTAL_AUX_WEIGHT > 0:
                    try:
                        exp_mixture, exp_targets, _ = next(experimental_aux_iter)
                    except StopIteration:
                        experimental_aux_iter = iter(experimental_aux_loader)
                        exp_mixture, exp_targets, _ = next(experimental_aux_iter)

                    exp_mixture = exp_mixture.to(DEVICE, non_blocking=True)
                    exp_targets = exp_targets.to(DEVICE, non_blocking=True)
                    exp_predictions, exp_background = model(exp_mixture)
                    exp_mixture_linear = exp_mixture[:, 0:1, :]
                    l_exp_aux = experimental_quantitative_aux_loss(
                        exp_predictions, exp_background, exp_targets, exp_mixture_linear
                    )
                    loss = loss + EXPERIMENTAL_AUX_WEIGHT * l_exp_aux

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        scheduler.step()

        model.eval()
        total_val_loss = 0.0
        total_val_false_peak = 0.0
        total_val_excess = 0.0
        total_val_inactive_area = 0.0
        total_val_area_fraction = 0.0
        total_val_area_recovery = 0.0
        all_pred_maxes, all_target_maxes = [], []
        all_pred_areas, all_target_areas = [], []

        with torch.no_grad():
            for m, t in fixed_val_batches:
                m = m.to(DEVICE, non_blocking=True)
                t = t.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast():
                    p, bg = model(m)
                    m_linear = m[:, 0:1, :]
                    val_loss, _ = loss_fn(p, t, bg, m_linear)

                total_val_loss += val_loss.item()
                area_fraction, area_recovery = area_validation_metrics(p, t)
                total_val_area_fraction += area_fraction.item()
                total_val_area_recovery += area_recovery.item()
                pred_pos = F.relu(p)
                target_support = (t > TRAIN_ACTIVE_THRESHOLD).float()
                target_support = F.max_pool1d(
                    target_support.reshape(-1, 1, t.shape[-1]),
                    kernel_size=31,
                    stride=1,
                    padding=15,
                ).reshape_as(t)
                active_ch = (t.amax(dim=2, keepdim=True) > TRAIN_ACTIVE_THRESHOLD).float()
                false_region = (1.0 - target_support) * active_ch
                total_val_false_peak += (
                    (pred_pos * false_region).sum() / (false_region.sum() + 1e-8)
                ).item()
                total_val_excess += F.relu(
                    pred_pos.sum(dim=1, keepdim=True) + F.relu(bg) - torch.clamp(m_linear, min=0.0)
                ).mean().item()
                inactive_ch = (t.amax(dim=2) <= TRAIN_ACTIVE_THRESHOLD).float()
                inactive_area = pred_pos.sum(dim=2) * inactive_ch
                total_val_inactive_area += (
                    inactive_area.sum() / (inactive_ch.sum() + 1e-8)
                ).item()
                all_pred_maxes.append(pred_pos.amax(dim=2).detach().cpu().numpy())
                all_target_maxes.append(t.amax(dim=2).detach().cpu().numpy())
                all_pred_areas.append(pred_pos.sum(dim=2).detach().cpu().numpy())
                all_target_areas.append(t.sum(dim=2).detach().cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / VAL_STEPS
        avg_val_false_peak = total_val_false_peak / VAL_STEPS
        avg_val_excess = total_val_excess / VAL_STEPS
        avg_val_inactive_area = total_val_inactive_area / VAL_STEPS
        avg_val_area_fraction = total_val_area_fraction / VAL_STEPS
        avg_val_area_recovery = total_val_area_recovery / VAL_STEPS
        print(
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
            f"Val AreaFracMAE: {avg_val_area_fraction:.6f} | "
            f"Val AreaRecoveryMAPE: {avg_val_area_recovery:.2f}% | "
            f"Val FalsePeak: {avg_val_false_peak:.6f} | Val Excess: {avg_val_excess:.6f} | "
            f"Val InactiveArea: {avg_val_inactive_area:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6e}"
        )

        all_pred_maxes = np.concatenate(all_pred_maxes, axis=0)
        all_target_maxes = np.concatenate(all_target_maxes, axis=0)
        all_pred_areas = np.concatenate(all_pred_areas, axis=0)
        all_target_areas = np.concatenate(all_target_areas, axis=0)
        f1, serum_recall, fpr, precision, per_compound = evaluate_classification_metrics(
            all_pred_maxes, all_target_maxes, train_dataset.names,
            threshold=EVAL_PRESENCE_THRESHOLD,
        )
        hard_fp_score = calculate_hard_fp_score(per_compound)
        bcaa_fp_score = calculate_bcaa_fp_score(per_compound)
        conditional_bcaa_fp = conditional_bcaa_false_positives(
            all_pred_maxes, all_target_maxes, train_dataset.names,
            threshold=EVAL_PRESENCE_THRESHOLD,
        )
        inactive_peak_mean, inactive_peak_p95, inactive_peak_max = inactive_peak_metrics(
            all_pred_maxes, all_target_maxes
        )
        proline_inactive_area = inactive_area_for_compounds(
            all_pred_areas, all_target_maxes, train_dataset.names, ('Proline',)
        )
        leucine_inactive_area = inactive_area_for_compounds(
            all_pred_areas, all_target_maxes, train_dataset.names, ('Leucine',)
        )
        gln_glu_pro_inactive_area = inactive_area_for_compounds(
            all_pred_areas, all_target_maxes, train_dataset.names,
            ('Glutamine', 'Glutamate', 'Proline')
        )
        ile_leu_val_pro_inactive_area = inactive_area_for_compounds(
            all_pred_areas, all_target_maxes, train_dataset.names,
            ('Isoleucine', 'Leucine', 'Valine', 'Proline')
        )
        bcaa_inactive_area = inactive_area_for_compounds(
            all_pred_areas, all_target_maxes, train_dataset.names,
            ('Isoleucine', 'Leucine', 'Valine')
        )
        leucine_inactive_peak_p95 = inactive_peak_p95_for_compounds(
            all_pred_maxes, all_target_maxes, train_dataset.names, ('Leucine',)
        )
        print(
            f"   [inactive peaks] Val InactivePeakMean={inactive_peak_mean:.6f} | "
            f"Val InactivePeakP95={inactive_peak_p95:.6f} | "
            f"Val InactivePeakMax={inactive_peak_max:.6f}"
        )
        print(
            f"   [overlap-fp] "
            f"Proline_FP={per_compound.get('Proline', {}).get('fp', 0)} | "
            f"Leucine_FP={per_compound.get('Leucine', {}).get('fp', 0)} | "
            f"Isoleucine_FP={per_compound.get('Isoleucine', {}).get('fp', 0)} | "
            f"Valine_FP={per_compound.get('Valine', {}).get('fp', 0)} | "
            f"hard_fp_score={hard_fp_score:.2f} | "
            f"bcaa_fp_score={bcaa_fp_score:.2f}"
        )
        print(
            f"   [conditional BCAA FP] "
            f"Leu_when_Ile_or_Val={conditional_bcaa_fp['Leucine_FP_when_Ile_or_Val_active']} | "
            f"Ile_when_Leu_or_Val={conditional_bcaa_fp['Isoleucine_FP_when_Leu_or_Val_active']} | "
            f"Val_when_Ile_or_Leu={conditional_bcaa_fp['Valine_FP_when_Ile_or_Leu_active']}"
        )
        print(
            f"   [hard inactive area] Proline={proline_inactive_area:.6f} | "
            f"Leucine={leucine_inactive_area:.6f} | "
            f"Gln/Glu/Pro={gln_glu_pro_inactive_area:.6f} | "
            f"Ile/Leu/Val/Pro={ile_leu_val_pro_inactive_area:.6f} | "
            f"BCAA={bcaa_inactive_area:.6f} | "
            f"LeucinePeakP95={leucine_inactive_peak_p95:.6f}"
        )
        overlap_terms = []
        for group in OVERLAP_GROUPS:
            idx = [train_dataset.mapping[n] for n in group if n in train_dataset.mapping]
            if len(idx) < 2:
                continue
            p_area = np.maximum(all_pred_areas[:, idx], 0.0)
            t_area = np.maximum(all_target_areas[:, idx], 0.0)
            group_target_total = t_area.sum(axis=1, keepdims=True)
            active = group_target_total.squeeze(1) > TRAIN_ACTIVE_THRESHOLD
            if not np.any(active):
                continue
            p_fraction = p_area / np.maximum(p_area.sum(axis=1, keepdims=True), 1e-6)
            t_fraction = t_area / np.maximum(group_target_total, 1e-6)
            overlap_terms.append(float(np.mean(np.abs(p_fraction[active] - t_fraction[active]).sum(axis=1))))
        overlap_focus = float(np.mean(overlap_terms)) if overlap_terms else 0.0
        score = (
            serum_recall * 0.8
            + precision * 0.8
            + f1 * 0.7
            - fpr * 2.5
            - avg_val_area_fraction * 4.0
            - (avg_val_area_recovery / 100.0) * 1.8
            - overlap_focus * 0.8
            - avg_val_false_peak * 0.8
            - avg_val_excess * 1.2
            - avg_val_inactive_area * 0.5
        )
        if fpr > 0.05:
            score -= 2.0
        latest_table_s1_factors = {}
        latest_external_raw_mape = float('nan')
        latest_external_cal_mape = float('nan')
        if (epoch + 1) % TABLE_S1_EVAL_EVERY == 0:
            latest_table_s1_factors, _, _, external_raw, external_cal = print_table_s1_calibration_report(
                model, val_dataset
            )
            latest_external_raw_mape = external_raw['mape']
            latest_external_cal_mape = external_cal['mape']

        print(
            f"   [selection] precision={precision * 100:.2f}% | "
            f"serum_recall={serum_recall * 100:.2f}% | "
            f"score={score:.6f} | "
            f"external_raw_MAPE={latest_external_raw_mape:.2f}% | "
            f"external_cal_MAPE={latest_external_cal_mape:.2f}% | "
            f"hard_fp_score={hard_fp_score:.2f} | "
            f"bcaa_fp_score={bcaa_fp_score:.2f} | "
            f"inactive_peak_p95={inactive_peak_p95:.6f} | "
            f"best_score={best_score:.6f} | "
            f"best_external_raw_MAPE={best_external_raw_mape:.2f}% | "
            f"best_hard_fp_score={best_hard_fp_score:.2f} | "
            f"best_bcaa_fp_score={best_bcaa_fp_score:.2f}"
        )

        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state, LATEST_EPOCH_PATH)
        print(f"Latest epoch saved: {LATEST_EPOCH_PATH}")
        if (epoch + 1) % VIS_EVERY == 0 or (epoch + 1) == EPOCHS:
            epoch_path = os.path.join(SAVE_PATH, f"epoch_{epoch + 1}.pth")
            torch.save(state, epoch_path)
            print(f"Periodic checkpoint saved: {epoch_path}")

        improved_external = False
        improved_hard_fp = False
        improved_bcaa_fp = False

        if score > best_score:
            best_score = score
            torch.save(state, BEST_BY_SCORE_PATH)
            print(f"Best-by-score saved: {BEST_BY_SCORE_PATH} | score={best_score:.6f}")

        if np.isfinite(latest_external_raw_mape) and latest_external_raw_mape < best_external_raw_mape:
            best_external_raw_mape = latest_external_raw_mape
            improved_external = True
            torch.save(state, BEST_BY_TABLE_S1_EXTERNAL_RAW_MAPE_PATH)
            torch.save(state, FINETUNED_SAVE_PATH)
            save_global_calibration_factors(latest_table_s1_factors, epoch + 1)
            print(
                f"Best-by-Table-S1-external-raw-MAPE saved: "
                f"{BEST_BY_TABLE_S1_EXTERNAL_RAW_MAPE_PATH} | "
                f"external_raw_MAPE={best_external_raw_mape:.2f}%"
            )
            print(f"Synced predict-compatible best model: {FINETUNED_SAVE_PATH}")

        if hard_fp_score < best_hard_fp_score and serum_recall >= 0.95:
            best_hard_fp_score = hard_fp_score
            improved_hard_fp = True
            torch.save(state, BEST_BY_LOW_FALSE_POSITIVE_PATH)
            print(
                f"Best-by-low-false-positive saved: {BEST_BY_LOW_FALSE_POSITIVE_PATH} | "
                f"hard_fp_score={best_hard_fp_score:.2f} | serum_recall={serum_recall * 100:.2f}%"
            )

        external_guard_ok = (
            np.isfinite(latest_external_raw_mape)
            and (
                not np.isfinite(best_external_raw_mape)
                or latest_external_raw_mape <= best_external_raw_mape * 1.5
            )
        )
        if bcaa_fp_score < best_bcaa_fp_score and serum_recall >= 0.95 and external_guard_ok:
            best_bcaa_fp_score = bcaa_fp_score
            improved_bcaa_fp = True
            torch.save(state, BEST_BY_OVERLAP_LOW_FP_PATH)
            print(
                f"Best-by-overlap-low-FP saved: {BEST_BY_OVERLAP_LOW_FP_PATH} | "
                f"bcaa_fp_score={best_bcaa_fp_score:.2f} | "
                f"external_raw_MAPE={latest_external_raw_mape:.2f}% | "
                f"serum_recall={serum_recall * 100:.2f}%"
            )

        if improved_external or improved_hard_fp or improved_bcaa_fp:
            epochs_without_external_or_hard_improvement = 0
        else:
            epochs_without_external_or_hard_improvement += 1
            print(
                f"No external_raw_MAPE or hard-FP improvement for "
                f"{epochs_without_external_or_hard_improvement}/{EARLY_STOP_PATIENCE} epochs | "
                f"best_external_raw_MAPE={best_external_raw_mape:.2f}% | "
                f"best_hard_fp_score={best_hard_fp_score:.2f} | "
                f"best_bcaa_fp_score={best_bcaa_fp_score:.2f}"
            )

        if (epoch + 1) % VIS_EVERY == 0 or (epoch + 1) == EPOCHS:
            save_visualization(model, fixed_val_batches, train_dataset.names, epoch + 1)
            save_false_positive_visualization(model, fixed_val_batches, train_dataset.names, epoch + 1)

        if epochs_without_external_or_hard_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}; best_score={best_score:.6f}")
            break


def save_visualization(model, fixed_val_batches, names, epoch):
    model.eval()
    with torch.no_grad():
        m_vis, t_vis = fixed_val_batches[0]
        m_vis = m_vis.to(DEVICE)
        p_vis, bg_vis = model(m_vis)

    mix_signal = m_vis[0, 0, :].cpu().numpy()
    t_signal = t_vis[0].cpu().numpy()
    p_signal = p_vis[0].cpu().numpy()
    bg_signal = bg_vis[0, 0, :].cpu().numpy()
    focus_channels = [
        i for i, name in enumerate(names)
        if name in SERUM_FOCUS and np.max(t_signal[i]) > TRAIN_ACTIVE_THRESHOLD
    ]
    other_channels = [
        i for i in range(len(names))
        if i not in focus_channels and np.max(t_signal[i]) > TRAIN_ACTIVE_THRESHOLD
    ]
    active_channels = (focus_channels + other_channels)[:8]
    if not active_channels:
        return

    fig, axes = plt.subplots(len(active_channels) + 1, 1, figsize=(15, 3 * (len(active_channels) + 1)), sharex=True)
    ppm = np.linspace(10.0, 0.0, 16384)
    axes[0].plot(ppm, mix_signal, color='black', lw=1, label='Input Mixture')
    axes[0].plot(ppm, bg_signal, color='m', lw=1.2, linestyle='--', label='Predicted Background')
    axes[0].set_title(f"Epoch {epoch} - Serum Fine-tune Reconstruction", fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for i, ch in enumerate(active_channels):
        ax = axes[i + 1]
        ax.plot(ppm, t_signal[ch], lw=2, label='Ground Truth', alpha=0.6)
        ax.plot(ppm, p_signal[ch], color='red', lw=1.5, linestyle='--', label='Prediction')
        ax.set_title(f"{names[ch]} / Channel {ch}")
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Chemical Shift (ppm)")
    axes[-1].invert_xaxis()
    plt.tight_layout()
    out_path = os.path.join(SAVE_PATH, f"epoch_{epoch}_serum_finetune.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved visualization: {out_path}")


def save_false_positive_visualization(model, fixed_val_batches, names, epoch):
    model.eval()
    with torch.no_grad():
        m_vis, t_vis = fixed_val_batches[0]
        m_vis = m_vis.to(DEVICE)
        p_vis, bg_vis = model(m_vis)

    mix_signal = m_vis[0, 0, :].cpu().numpy()
    t_signal = t_vis[0].cpu().numpy()
    p_signal = p_vis[0].cpu().numpy()
    bg_signal = bg_vis[0, 0, :].cpu().numpy()

    target_max = np.max(t_signal, axis=1)
    pred_max = np.max(p_signal, axis=1)
    inactive = target_max <= TRAIN_ACTIVE_THRESHOLD
    false_positive_candidates = [
        i for i in np.argsort(pred_max)[::-1]
        if inactive[i] and pred_max[i] > EVAL_PRESENCE_THRESHOLD
    ][:6]
    if not false_positive_candidates:
        return

    fig, axes = plt.subplots(
        len(false_positive_candidates) + 1,
        1,
        figsize=(15, 3 * (len(false_positive_candidates) + 1)),
        sharex=True,
    )
    ppm = np.linspace(10.0, 0.0, 16384)
    axes[0].plot(ppm, mix_signal, color='black', lw=1, label='Input Mixture')
    axes[0].plot(ppm, bg_signal, color='m', lw=1.2, linestyle='--', label='Predicted Background')
    axes[0].set_title(f"Epoch {epoch} - Top False Positive Channels", fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for i, ch in enumerate(false_positive_candidates):
        ax = axes[i + 1]
        ax.plot(ppm, t_signal[ch], lw=2, label='Ground Truth', alpha=0.6)
        ax.plot(ppm, p_signal[ch], color='red', lw=1.5, linestyle='--', label='Prediction')
        ax.set_title(f"{names[ch]} / Channel {ch} | pred_max={pred_max[ch]:.4f}")
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Chemical Shift (ppm)")
    axes[-1].invert_xaxis()
    plt.tight_layout()
    out_path = os.path.join(SAVE_PATH, f"epoch_{epoch}_false_positive_channels.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved false-positive visualization: {out_path}")


if __name__ == "__main__":
    run_training()
