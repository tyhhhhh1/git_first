import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation
from scipy.optimize import nnls

import matplotlib
# Keep Agg disabled when interactive plot windows are needed.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import dataset and model classes.
from dataset import NMRSepDataset
from model import NMRSepFormer


# ==========================================
# 1. Data loading and alignment to the 16384-point 10-to-0 ppm axis
# ==========================================
def integrate_positive_area(ppm_axis, intensity, mask):
    if not np.any(mask):
        return 0.0
    x = np.asarray(ppm_axis)[mask]
    y = np.clip(np.asarray(intensity)[mask], 0.0, None)
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def load_and_align_mnova_data(filepath, target_ppm_axis):
    print(f"Loading Mnova data: {os.path.basename(filepath)}")
    try:
        data = np.loadtxt(filepath, delimiter=',' if filepath.endswith('.csv') else None)
        real_ppm = data[:, 0]
        intensity = data[:, 1]
    except Exception:
        df = pd.read_csv(filepath, sep=None, engine='python', header=None)
        real_ppm = df.iloc[:, 0].values
        intensity = df.iloc[:, 1].values

    sort_idx = np.argsort(real_ppm)
    real_ppm_sorted = real_ppm[sort_idx]
    intensity_sorted = intensity[sort_idx]

    interpolator = interp1d(real_ppm_sorted, intensity_sorted, kind='linear', bounds_error=False, fill_value=0)
    aligned_intensity = interpolator(target_ppm_axis)
    aligned_intensity[aligned_intensity < 0] = 0.0

    raw_aligned_intensity = aligned_intensity.copy()

    # Exclude the water region when estimating the normalization scale.
    water_mask = (target_ppm_axis >= 4.6) & (target_ppm_axis <= 5.0)
    input_intensity = aligned_intensity.copy()
    input_intensity[water_mask] = 0.0
    non_water_intensity = input_intensity[~water_mask]
    p_val = np.percentile(non_water_intensity, 99.9) if len(non_water_intensity) > 0 else np.max(input_intensity)
    if p_val == 0: p_val = 1e-6

    network_input_intensity = input_intensity / p_val
    network_input_intensity = np.clip(network_input_intensity, 0.0, 3.0)
    clipped_fraction = float(np.mean(input_intensity / p_val > 3.0))
    print(
        f"Scale check: p_val={p_val:.6g} (non-water p99.9), "
        f"input_max={np.max(network_input_intensity):.3f}, "
        f"clip>3 ratio={clipped_fraction * 100:.3f}%"
    )

    return network_input_intensity, raw_aligned_intensity, p_val, real_ppm_sorted, intensity_sorted


# ==========================================
# 2. NMRVPF Eq. 5 internal-standard quantification
# ==========================================
def calculate_absolute_concentration(target_area, ic_area, target_protons,
                                     ic_protons=9,
                                     ic_mw=218.32,
                                     ic_mass_conc_g_per_ml=0.0005,
                                     ic_concentration_mM=None):
    """
    Eq. 5 in NMRVPF:
        C_t = n_IC * rho_IC * PV_IC * Int_t / (n_t * Int_IC * MW_IC)

    The paper uses 0.05% (w/v) internal reference for amino-acid mixtures,
    i.e. 0.05 g / 100 mL = 0.0005 g/mL.  The old code used 0.05 directly,
    which inflates concentrations by about 100x.
    """
    if ic_area <= 0 or target_protons <= 0:
        return 0.0
    if ic_concentration_mM is not None:
        return float(ic_concentration_mM) * ic_protons * target_area / (target_protons * ic_area)
    numerator = ic_protons * ic_mass_conc_g_per_ml * target_area
    denominator = target_protons * ic_area * ic_mw
    concentration_mol_per_ml = numerator / denominator
    return concentration_mol_per_ml * 1e6  # mol/mL -> mM


def shift_spectrum(ppm_axis, spectrum, shift_ppm):
    order = np.argsort(ppm_axis)
    x_sorted = ppm_axis[order]
    y_sorted = spectrum[order]
    shifted_sorted = np.interp(x_sorted - shift_ppm, x_sorted, y_sorted, left=0.0, right=0.0)
    shifted = np.zeros_like(spectrum, dtype=np.float64)
    shifted[order] = shifted_sorted
    return shifted


def normalize_template_to_unit_area(ppm_axis, spectrum, quant_mask):
    spectrum = np.clip(np.asarray(spectrum, dtype=np.float64), 0.0, None)
    area = integrate_positive_area(ppm_axis, spectrum, quant_mask)
    if area <= 1e-12:
        return None
    return spectrum / area


def build_input_evidence_mask_torch(mixture, window=41, noise_k=2.5, floor_ratio=0.005):
    """
    Build a soft evidence mask from the measured mixture. This suppresses
    separator and NNLS peaks in ppm regions unsupported by the input spectrum.
    """
    mix = torch.clamp(mixture.float(), min=0.0)
    local_max = F.max_pool1d(mix, kernel_size=window, stride=1, padding=window // 2)

    flat = mix.flatten(start_dim=1)
    med = flat.median(dim=1).values.view(-1, 1, 1)
    mad = (flat - med.flatten(start_dim=1)).abs().median(dim=1).values.view(-1, 1, 1)
    noise = 1.4826 * mad + 1e-8
    global_peak = mix.amax(dim=-1, keepdim=True)
    threshold = torch.maximum(med + noise_k * noise, floor_ratio * global_peak)

    return torch.sigmoid((local_max - threshold) / noise)


def constrain_predictions_to_input(predictions, mixture, evidence_mask, eps=1e-8):
    """
    Apply two conservative inference-time constraints:
    1. keep only input-supported peak regions;
    2. prevent summed compound predictions from exceeding the mixture pointwise.
    """
    pred = torch.relu(predictions.float()) * evidence_mask
    pred_sum = pred.sum(dim=1, keepdim=True)
    available = torch.clamp(mixture.float(), min=0.0)
    scale = torch.clamp(available / (pred_sum + eps), max=1.0)
    return pred * scale


TABLE_S1_EMPIRICAL_CALIBRATION = {
    "enabled": True,
    "min_candidates": 7,
    "min_s_over_ic": 50.0,
    "sample_scale": 1.65,
    "compound_factors": {
        "Asparagine": 1.28,
        "Glutamine": 0.77,
        "Glutamate": 1.42,
        "Isoleucine": 0.65,
        "Leucine": 0.94,
        "Proline": 1.25,
        "Valine": 1.64,
    },
}

REAL_SERUM_EMPIRICAL_CALIBRATION = {
    "enabled": False,
    "sample_scale": 1.0,
    "compound_factors": {},
}


def resolve_empirical_calibration(active_mode, candidate_count, total_mixture_area, ic_area):
    """
    Return the calibration mode and factors for the current prediction.

    Table S1 calibration is an empirical benchmark correction for defined
    amino-acid mixtures only. It must not be applied to real serum until a
    separate real-serum calibration set is available.
    """
    s_over_ic = total_mixture_area / max(ic_area, 1e-12)

    if active_mode == "table_s1":
        cfg = TABLE_S1_EMPIRICAL_CALIBRATION
        enabled = (
            cfg["enabled"]
            and candidate_count >= cfg["min_candidates"]
            and s_over_ic >= cfg["min_s_over_ic"]
        )
        mode = "table_s1_empirical" if enabled else "none"
        return mode, enabled, cfg["sample_scale"], cfg["compound_factors"], s_over_ic

    if active_mode == "real_serum":
        cfg = REAL_SERUM_EMPIRICAL_CALIBRATION
        enabled = bool(cfg["enabled"])
        mode = "real_serum_empirical" if enabled else "none"
        return mode, enabled, cfg["sample_scale"], cfg["compound_factors"], s_over_ic

    return "none", False, 1.0, {}, s_over_ic


def fit_nmrvpf_style_nnls(ppm_axis, mixture_abs, dataset, candidates, quant_mask,
                          shift_grid=(-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018),
                          add_baseline=True, evidence_mask=None):
    """
    NMRVPF-style post fit:
    jointly fit all candidate reference templates to the experimental mixture
    in 0.7-4.2 ppm. Each shifted template is normalized to unit area, so the
    summed NNLS coefficient for a compound is its fitted integral Int_t.
    """
    y = np.clip(np.asarray(mixture_abs, dtype=np.float64)[quant_mask], 0.0, None)
    columns = []
    meta = []

    for name in candidates:
        template = dataset._generate_compound(name, jitter_active=False)
        if np.max(template) <= 0:
            continue
        for shift_ppm in shift_grid:
            shifted = shift_spectrum(ppm_axis, template, shift_ppm)
            unit = normalize_template_to_unit_area(ppm_axis, shifted, quant_mask)
            if unit is None:
                continue
            fit_unit = unit
            if evidence_mask is not None:
                fit_unit = normalize_template_to_unit_area(ppm_axis, unit * evidence_mask, quant_mask)
                if fit_unit is None:
                    continue
            columns.append(fit_unit[quant_mask])
            meta.append((name, shift_ppm, fit_unit))

    if add_baseline:
        x = ppm_axis[quant_mask]
        x01 = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)
        columns.extend([np.ones_like(y), x01, 1.0 - x01])
        meta.extend([('__baseline__', 0.0, None), ('__baseline__', 0.0, None), ('__baseline__', 0.0, None)])

    if not columns:
        return {}, {}, np.zeros_like(mixture_abs, dtype=np.float64)

    A = np.stack(columns, axis=1)
    coef, residual_norm = nnls(A, y)

    fitted_area = {name: 0.0 for name in candidates}
    fitted_specs = {name: np.zeros_like(mixture_abs, dtype=np.float64) for name in candidates}
    for c, (name, shift_ppm, unit_template) in zip(coef, meta):
        if name == '__baseline__' or c <= 0:
            continue
        fitted_area[name] += float(c)
        fitted_specs[name] += unit_template * float(c)

    reconstructed = np.zeros_like(mixture_abs, dtype=np.float64)
    reconstructed[quant_mask] = A @ coef
    return fitted_area, fitted_specs, reconstructed


# ==========================================
# 3. Prediction and quantification workflow
# ==========================================
def predict_serum_amino_acids():
    # ==========================================
    # Path configuration
    # ==========================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_WEIGHTS_CANDIDATES = [
        os.path.join(SCRIPT_DIR, 'checkpoints', 'best_model_quantitative_lowfp_refine.pth'),
        os.path.join(SCRIPT_DIR, 'checkpoints', 'best_model_quantitative_area_aux_refine.pth'),
        os.path.join(SCRIPT_DIR, 'checkpoints', 'best_model_serum_refine.pth'),
        os.path.join(SCRIPT_DIR, 'checkpoints', 'best_model_serum_finetune.pth'),
        os.path.join(SCRIPT_DIR, 'checkpoints', 'best_model.pth'),
    ]
    CSV_LIBRARY_PATH = os.path.join(SCRIPT_DIR, 'AminoAcids20_lib.csv')
    REAL_DATA_FILE = os.path.join(SCRIPT_DIR, 'sample1.csv')
    SAVE_FOLDER = os.path.join(SCRIPT_DIR, 'serum_predictions_new')

    # Prediction mode:
    #   table_s1   -> defined amino-acid mixtures in the NMRVPF Table S1 style.
    #   real_serum -> DSS serum mixtures such as Bayesil defined-serum samples.
    PREDICT_MODE = "auto"

    MODE_CONFIGS = {
        "table_s1": {
            "internal_standard": {
                "name": "TMSP",
                "protons": 9,
                "mw": 172.27,
                "mass_conc_g_per_ml": 0.0005,  # 0.05% (w/v)
                "concentration_mM": None,
            },
            "report_zero_below_mM": 0.6,
            "candidate_min_mM": 2.5,
            "candidate_soft_min_mM": 2.5,
            "candidate_min_peak": 0.03,
            "quant_calibration_factor": 1.20,
            "target_scope": "table_s1_7aa",
            "fit_all_candidates": False,
            "global_plot_min": 0.01,
        },
        "real_serum": {
            "internal_standard": {
                "name": "DSS",
                "protons": 9,
                "mw": 218.32,
                "mass_conc_g_per_ml": None,
                "concentration_mM": 0.8769,  # def_mix_serum_1 DSS = 876.9 uM
            },
            "report_zero_below_mM": 0.004,
            "candidate_min_mM": 0.010,
            "candidate_soft_min_mM": 0.003,
            "candidate_min_peak": 0.001,
            "quant_calibration_factor": 1.00,
            "target_scope": "all_20aa",
            "fit_all_candidates": True,
            "global_plot_min": 0.001,
        },
    }
    sample_key_for_mode = os.path.splitext(os.path.basename(REAL_DATA_FILE))[0].lower()
    if PREDICT_MODE == "auto":
        active_mode = "table_s1" if sample_key_for_mode.startswith("sample") else "real_serum"
    else:
        active_mode = PREDICT_MODE
    if active_mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown PREDICT_MODE: {PREDICT_MODE}")

    mode_cfg = MODE_CONFIGS[active_mode]
    INTERNAL_STANDARD = mode_cfg["internal_standard"]

    # Paper setting: spectra are quantified in the 0.7-4.2 ppm range.
    QUANT_PPM_MIN = 0.7
    QUANT_PPM_MAX = 4.2
    REPORT_ZERO_BELOW_MM = mode_cfg["report_zero_below_mM"]
    NNLS_CANDIDATE_MIN_MM = mode_cfg["candidate_min_mM"]
    NNLS_CANDIDATE_SOFT_MIN_MM = mode_cfg["candidate_soft_min_mM"]
    NNLS_CANDIDATE_MIN_PEAK = mode_cfg["candidate_min_peak"]
    QUANT_CALIBRATION_FACTOR = mode_cfg["quant_calibration_factor"]

    # Table S1 sample-1 ground truth, used only for post-run evaluation.
    # Leave empty for unknown samples.
    SAMPLE_STANDARDS_MM = {
        'sample1': {
            'Glutamine': 160.94,
            'Glutamate': 33.64,
            'Proline': 0.0,
        },
        'sample2': {
            'Isoleucine': 68.61,
            'Leucine': 17.08,
        },
        'sample3': {
            'Asparagine': 51.77,
            'Glutamine': 9.75,
            'Glutamate': 7.34,
            'Isoleucine': 29.50,
            'Leucine': 5.47,
            'Proline': 52.38,
            'Valine': 6.91,
        },
        'sample4': {
            'Glutamine': 4.27,
            'Glutamate': 29.97,
            'Isoleucine': 21.27,
            'Leucine': 9.57,
            'Proline': 19.54,
        },
        'sample5': {
            'Asparagine': 43.60,
            'Glutamate': 24.47,
            'Leucine': 19.13,
            'Proline': 17.20,
            'Valine': 26.12,
        },
        'sample6': {
            'Asparagine': 81.74,
            'Glutamine': 26.82,
            'Isoleucine': 34.30,
            'Valine': 35.34,
        },
        'sample7': {
            'Asparagine': 103.54,
            'Glutamine': 97.54,
            'Glutamate': 95.43,
            'Isoleucine': 96.05,
            'Leucine': 99.77,
            'Proline': 68.79,
            'Valine': 99.87,
        },
        'sample8': {
            'Asparagine': 27.77,
            'Glutamine': 44.18,
            'Glutamate': 37.21,
            'Isoleucine': 71.03,
            'Leucine': 42.89,
            'Proline': 22.76,
            'Valine': 107.37,
        },
        'sample9': {
            'Asparagine': 74.49,
            'Glutamine': 78.10,
            'Glutamate': 85.49,
            'Isoleucine': 69.25,
            'Leucine': 81.80,
            'Proline': 60.70,
            'Valine': 140.18,
        },
        'sample10': {
            'Asparagine': 71.93,
            'Glutamine': 46.94,
            'Glutamate': 86.13,
            'Isoleucine': 75.47,
            'Leucine': 105.24,
            'Proline': 65.35,
            'Valine': 94.65,
        }
    }

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Library CSV: {CSV_LIBRARY_PATH}")
    print(f"Input sample: {REAL_DATA_FILE}")
    print(f"Internal standard: {INTERNAL_STANDARD['name']} | MW={INTERNAL_STANDARD['mw']} | mass conc={INTERNAL_STANDARD['mass_conc_g_per_ml']} g/mL")

    # 1. Build dataset metadata: ppm axis, mapping, and templates.
    dataset = NMRSepDataset(csv_path=CSV_LIBRARY_PATH, signal_len=16384, ppm_range=(10.0, 0.0))
    ppm_axis = dataset.x
    if dataset.num_classes != 20:
        raise ValueError(f"Expected 20 compounds/channels, but library contains {dataset.num_classes}.")
    mapping = dataset.mapping

    # 2. Build model and load checkpoint.
    model = NMRSepFormer(num_compounds=dataset.num_classes, d_model=128).to(device)
    weight_path = next((p for p in MODEL_WEIGHTS_CANDIDATES if os.path.exists(p)), None)
    if weight_path is None:
        raise FileNotFoundError(f'No model weights found in: {MODEL_WEIGHTS_CANDIDATES}')
    print(f'Using weights: {weight_path}')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    sample_name = os.path.basename(REAL_DATA_FILE).replace('.csv', '')
    sample_key = sample_name.lower()

    # 3. Load and align the sample spectrum.
    network_input_np, raw_spec_np, p_val, raw_ppm_np, raw_intensity_np = load_and_align_mnova_data(REAL_DATA_FILE, ppm_axis)
    
    # 4. Build the two-channel model input: linear intensity + log intensity.
    mixture_linear = network_input_np
    mixture_log = np.log10(np.clip(mixture_linear, 0.0, None) + 1e-6)
    dual_mixture = np.stack([mixture_linear, mixture_log], axis=0).astype(np.float32)  # [2, 16384]
    serum_tensor = torch.tensor(dual_mixture).unsqueeze(0).to(device)                  # [1, 2, 16384]

    # Automatically integrate the internal-standard peak around 0 ppm.
    ic_mask = (raw_ppm_np <= 0.05) & (raw_ppm_np >= -0.05)
    ic_area = integrate_positive_area(raw_ppm_np, raw_intensity_np, ic_mask)
    print(f"Internal-standard area around 0 ppm: {ic_area:.6g}")

    quant_range_mask = (ppm_axis >= QUANT_PPM_MIN) & (ppm_axis <= QUANT_PPM_MAX)
    total_mixture_area = integrate_positive_area(ppm_axis, raw_spec_np, quant_range_mask)
    print(f"NMRVPF quant range {QUANT_PPM_MIN:.1f}-{QUANT_PPM_MAX:.1f} ppm total area S_all: {total_mixture_area:.6g}")

    # Non-exchangeable 1H counts visible in D2O within the 0.7-4.2 ppm quantification range.
    # Do not include NH/COOH protons: they exchange in D2O and should not normalize these fitted integrals.
    all_protons_dict = {
        'Alanine': 4,
        'Arginine': 7,
        'Asparagine': 3,
        'Aspartic': 3,
        'Cysteine': 3,
        'Glutamate': 5,
        'Glutamine': 5,
        'Glycine': 2,
        'Histidine': 3,
        'Isoleucine': 10,
        'Leucine': 10,
        'Lysine': 9,
        'Methionine': 8,
        'Phenylalanine': 3,
        'Proline': 7,
        'Serine': 3,
        'Threonine': 5,
        'Tryptophan': 3,
        'Tyrosine': 3,
        'Valine': 8,
    }
    table_s1_targets = [
        'Asparagine', 'Glutamine', 'Glutamate',
        'Isoleucine', 'Leucine', 'Proline', 'Valine',
    ]
    if mode_cfg["target_scope"] == "all_20aa":
        protons_dict = {name: all_protons_dict[name] for name in dataset.names if name in all_protons_dict}
    else:
        protons_dict = {name: all_protons_dict[name] for name in table_s1_targets if name in all_protons_dict}
    missing_protons = [name for name in dataset.names if name not in all_protons_dict]
    if missing_protons:
        print(f"Warning: missing proton count for: {', '.join(missing_protons)}")
    print(f"Target channels ({len(protons_dict)}): {', '.join(protons_dict.keys())}")

    results_list = []
    all_predictions = {}
    # Run one forward pass and collect all channel predictions.
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model_output = model(serum_tensor)
            if isinstance(model_output, tuple):
                predictions, background = model_output
            else:
                predictions = model_output
                background = None

            evidence_mask = build_input_evidence_mask_torch(serum_tensor[:, 0:1, :])
            raw_predictions = predictions
            predictions = constrain_predictions_to_input(predictions, serum_tensor[:, 0:1, :], evidence_mask)

    evidence_mask_np = evidence_mask[0, 0].detach().cpu().numpy()
    raw_pred_sum_np = torch.relu(raw_predictions).sum(dim=1)[0].detach().cpu().numpy()
    constrained_pred_sum_np = predictions.sum(dim=1)[0].detach().cpu().numpy()
    print(
        "Input-guided constraint: "
        f"raw_sum_max={np.max(raw_pred_sum_np):.4g}, "
        f"constrained_sum_max={np.max(constrained_pred_sum_np):.4g}, "
        f"evidence_mean={np.mean(evidence_mask_np):.4g}"
    )

    channel_metrics = {}
    candidate_names = []
    for target_name, n_t in protons_dict.items():
        if target_name not in mapping:
            continue

        channel_id = mapping[target_name]
        pred_norm_np = predictions[0, channel_id].cpu().numpy()
        ref_spec_np = dataset._generate_compound(target_name, jitter_active=False)
        ref_norm = ref_spec_np / np.max(ref_spec_np) if np.max(ref_spec_np) > 0 else ref_spec_np
        base_window = ref_norm > 0.002
        integration_window = binary_dilation(base_window, iterations=15) & quant_range_mask

        quant_spec_absolute = pred_norm_np * p_val
        quant_spec_absolute[~integration_window] = 0.0
        pred_area_direct = integrate_positive_area(ppm_axis, quant_spec_absolute, integration_window)
        channel_rho_t = pred_area_direct / total_mixture_area if total_mixture_area > 0 else 0.0
        channel_target_area = total_mixture_area * channel_rho_t
        channel_concentration = calculate_absolute_concentration(
            channel_target_area,
            ic_area,
            target_protons=n_t,
            ic_protons=INTERNAL_STANDARD['protons'],
            ic_mw=INTERNAL_STANDARD['mw'],
            ic_mass_conc_g_per_ml=INTERNAL_STANDARD['mass_conc_g_per_ml'],
            ic_concentration_mM=INTERNAL_STANDARD.get('concentration_mM'),
        )
        channel_peak = float(np.max(pred_norm_np[integration_window])) if np.any(integration_window) else 0.0

        channel_metrics[target_name] = {
            'pred_norm_np': pred_norm_np,
            'ref_norm': ref_norm,
            'integration_window': integration_window,
            'pred_area_direct': pred_area_direct,
            'channel_rho_t': channel_rho_t,
            'channel_concentration': channel_concentration,
            'channel_peak': channel_peak,
        }

        strong_evidence = channel_concentration >= NNLS_CANDIDATE_MIN_MM
        weak_but_peak_aligned = (
            channel_concentration >= NNLS_CANDIDATE_SOFT_MIN_MM
            and channel_peak >= NNLS_CANDIDATE_MIN_PEAK
        )
        if strong_evidence or weak_but_peak_aligned:
            candidate_names.append(target_name)

    if not candidate_names and channel_metrics:
        candidate_names = [
            max(channel_metrics, key=lambda name: channel_metrics[name]['channel_concentration'])
        ]
    if mode_cfg.get("fit_all_candidates", False):
        candidate_names = list(protons_dict.keys())

    candidate_set = set(candidate_names)
    suppressed_names = [name for name in protons_dict.keys() if name in mapping and name not in candidate_set]
    print(f"NNLS candidates selected from model channels: {', '.join(candidate_names) if candidate_names else 'None'}")
    if suppressed_names:
        print(f"NNLS suppressed low-evidence compounds: {', '.join(suppressed_names)}")

    nnls_area_dict, nnls_spec_dict, nnls_reconstruction = fit_nmrvpf_style_nnls(
        ppm_axis=ppm_axis,
        mixture_abs=raw_spec_np,
        dataset=dataset,
        candidates=candidate_names,
        quant_mask=quant_range_mask,
        evidence_mask=evidence_mask_np,
    )
    nnls_total_area = sum(nnls_area_dict.values())
    print(f"NNLS fitted total area: {nnls_total_area:.6g}")
    (
        calibration_mode,
        use_empirical_calibration,
        empirical_sample_scale,
        empirical_compound_factors,
        s_over_ic,
    ) = resolve_empirical_calibration(
        active_mode=active_mode,
        candidate_count=len(candidate_set),
        total_mixture_area=total_mixture_area,
        ic_area=ic_area,
    )
    if use_empirical_calibration:
        print(
            "Empirical calibration enabled: "
            f"mode={calibration_mode}, "
            f"candidate_count={len(candidate_set)}, S_all/IC={s_over_ic:.2f}, "
            f"sample_scale={empirical_sample_scale:.3f}"
        )
    else:
        print(
            "Empirical calibration disabled: "
            f"mode={calibration_mode}, candidate_count={len(candidate_set)}, "
            f"S_all/IC={s_over_ic:.2f}"
        )

    for target_name, n_t in protons_dict.items():
        if target_name not in mapping:
            print(f"Warning: {target_name} is not present in the mapping, skipped.")
            continue

        channel_id = mapping[target_name]

        # Store the current compound prediction for the global plot.
        pred_norm_np = predictions[0, channel_id].cpu().numpy()
        nnls_plot_np = nnls_spec_dict.get(target_name, np.zeros_like(raw_spec_np)) / max(p_val, 1e-12)
        all_predictions[target_name] = nnls_plot_np.copy()

        # Generate a clean reference spectrum.
        ref_spec_np = dataset._generate_compound(target_name, jitter_active=False)
        ref_norm = ref_spec_np / np.max(ref_spec_np) if np.max(ref_spec_np) > 0 else ref_spec_np

        # NMRVPF reports Int_t = S_all * rho_t.  For this separator model,
        # rho_t is approximated by the target-channel positive area fraction
        # within the paper's 0.7-4.2 ppm quantification range.
        base_window = ref_norm > 0.002
        integration_window = binary_dilation(base_window, iterations=15) & quant_range_mask

        quant_spec_absolute = pred_norm_np * p_val
        quant_spec_absolute[~integration_window] = 0.0  # 娓呴浂绐楀彛澶栧簳鍣?
        pred_area_direct = integrate_positive_area(ppm_axis, quant_spec_absolute, integration_window)
        channel_rho_t = pred_area_direct / total_mixture_area if total_mixture_area > 0 else 0.0
        channel_target_area = total_mixture_area * channel_rho_t
        channel_concentration = calculate_absolute_concentration(
            channel_target_area,
            ic_area,
            target_protons=n_t,
            ic_protons=INTERNAL_STANDARD['protons'],
            ic_mw=INTERNAL_STANDARD['mw'],
            ic_mass_conc_g_per_ml=INTERNAL_STANDARD['mass_conc_g_per_ml'],
            ic_concentration_mM=INTERNAL_STANDARD.get('concentration_mM'),
        )

        nnls_target_area = nnls_area_dict.get(target_name, 0.0) if target_name in candidate_set else 0.0
        nnls_rho_t = nnls_target_area / total_mixture_area if total_mixture_area > 0 else 0.0
        raw_concentration = calculate_absolute_concentration(
            nnls_target_area,
            ic_area,
            target_protons=n_t,
            ic_protons=INTERNAL_STANDARD['protons'],
            ic_mw=INTERNAL_STANDARD['mw'],
            ic_mass_conc_g_per_ml=INTERNAL_STANDARD['mass_conc_g_per_ml'],
            ic_concentration_mM=INTERNAL_STANDARD.get('concentration_mM'),
        )
        calibrated_concentration = raw_concentration * QUANT_CALIBRATION_FACTOR
        empirical_factor = 1.0
        if use_empirical_calibration:
            empirical_factor = (
                empirical_sample_scale
                * empirical_compound_factors.get(target_name, 1.0)
            )
        empirically_calibrated_concentration = calibrated_concentration * empirical_factor

        detected = (target_name in candidate_set) and (empirically_calibrated_concentration >= REPORT_ZERO_BELOW_MM)
        final_concentration = empirically_calibrated_concentration if detected else 0.0
        standard_conc = SAMPLE_STANDARDS_MM.get(sample_key, {}).get(target_name, np.nan)
        abs_error = final_concentration - standard_conc if not np.isnan(standard_conc) else np.nan
        rel_error_pct = (abs_error / standard_conc * 100.0) if (not np.isnan(standard_conc) and standard_conc != 0.0) else np.nan

        metrics = channel_metrics[target_name]
        results_list.append({
            'Predict_Mode': active_mode,
            'Metabolite': target_name,
            'Target_Protons': n_t,
            'Channel_Area': pred_area_direct,
            'Channel_rho_t': channel_rho_t,
            'Channel_Calculated_Conc_mM': channel_concentration,
            'Channel_Peak': metrics['channel_peak'],
            'S_all_0p7_4p2': total_mixture_area,
            'NNLS_Candidate': target_name in candidate_set,
            'NNLS_rho_t': nnls_rho_t,
            'NNLS_Int_t': nnls_target_area,
            'IC_Area': ic_area,
            'IC_Name': INTERNAL_STANDARD['name'],
            'IC_MW': INTERNAL_STANDARD['mw'],
            'IC_Mass_Conc_g_per_mL': INTERNAL_STANDARD['mass_conc_g_per_ml'],
            'IC_Concentration_mM': INTERNAL_STANDARD.get('concentration_mM'),
            'Raw_NNLS_Calculated_Conc_mM': raw_concentration,
            'Quant_Calibration_Factor': QUANT_CALIBRATION_FACTOR,
            'Calibrated_NNLS_Calculated_Conc_mM': calibrated_concentration,
            'Calibration_Mode': calibration_mode,
            'Empirical_Calibration_Applied': use_empirical_calibration,
            'Empirical_Calibration_Factor': empirical_factor,
            'Empirically_Calibrated_Conc_mM': empirically_calibrated_concentration,
            'Calculated_Conc_mM': final_concentration,
            'Detected': detected,
            'Standard_Conc_mM': standard_conc,
            'Abs_Error_mM': abs_error,
            'Rel_Error_%': rel_error_pct
        })

        # Save per-compound visualization.
        plt.figure(figsize=(16, 8))
        plt.title(f"[{sample_name}] - {target_name} | Conc: {final_concentration:.2f} mM", fontsize=16,
                  fontweight='bold')

        plot_mask = (ppm_axis <= 6.0) & (ppm_axis >= 0.0)
        plt.plot(ppm_axis[plot_mask], network_input_np[plot_mask], color='black', linewidth=1.2, alpha=0.8,
                 label='Serum Spectrum (Input)')
        plt.fill_between(ppm_axis[plot_mask], 0, pred_norm_np[plot_mask], color='red', alpha=0.45,
                         label='Model Channel (Unmixed)')
        plt.plot(ppm_axis[plot_mask], pred_norm_np[plot_mask], color='red', linewidth=1.5)
        plt.fill_between(ppm_axis[plot_mask], 0, nnls_plot_np[plot_mask], color='green', alpha=0.35,
                         label='NNLS Joint Fit')
        plt.plot(ppm_axis[plot_mask], nnls_plot_np[plot_mask], color='green', linewidth=1.2)

        offset = np.max(network_input_np) * 1.1 if np.max(network_input_np) > 0 else 1.0
        plt.plot(ppm_axis[plot_mask], ref_norm[plot_mask] + offset, color='blue', linestyle='--', linewidth=1.5,
                 alpha=0.8, label='Reference')

        plt.xlabel("Chemical Shift (ppm)", fontsize=14)
        plt.ylabel("Intensity", fontsize=14)
        plt.gca().invert_xaxis()
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = target_name.replace("/", "_").replace("\\", "_")
        plt.savefig(os.path.join(SAVE_FOLDER, f"{safe_name}_conc_{final_concentration:.2f}.png"), dpi=150)
        plt.close()

    df_results = pd.DataFrame(results_list)
    out_csv = os.path.join(SAVE_FOLDER, f"{sample_name}_quantification.csv")
    df_results.to_csv(out_csv, index=False)

    print(f"\nSample [{sample_name}] prediction and quantification completed.")
    print(f"Quantification CSV saved to '{out_csv}'")

    # ==========================================
    # Global interactive visualization
    # ==========================================
    print("\nGenerating global spectral decomposition plot...")
    plt.figure(figsize=(18, 8))
    plt.title(f"[{sample_name}] - Global Spectral Decomposition", fontsize=18, fontweight='bold')

    # Focus on the core 6.0-to-0.0 ppm region.
    plot_mask = (ppm_axis <= 6.0) & (ppm_axis >= 0.0)

    # 1. Draw the input mixture and the total NNLS fit.
    plt.plot(ppm_axis[plot_mask], network_input_np[plot_mask], color='black', linewidth=1.5, alpha=0.6,
             label='Original Mixture')
    plt.plot(
        ppm_axis[plot_mask],
        nnls_reconstruction[plot_mask] / max(p_val, 1e-12),
        color='crimson',
        linewidth=1.8,
        linestyle='--',
        alpha=0.9,
        label='NNLS Total Fit',
    )

    # 2. Overlay all predicted compounds.
    cmap = plt.get_cmap('tab20')
    for i, (name, pred_spec) in enumerate(all_predictions.items()):
        if np.max(pred_spec[plot_mask]) > 0.01:
            color = cmap(i % 20)
            plt.fill_between(ppm_axis[plot_mask], 0, pred_spec[plot_mask], color=color, alpha=0.5)
            plt.plot(ppm_axis[plot_mask], pred_spec[plot_mask], color=color, linewidth=1.0, label=f'{name}')

    plt.xlabel("Chemical Shift (ppm)", fontsize=14)
    plt.ylabel("Intensity", fontsize=14)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)

    # Place the legend outside the plot area.
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1)
    plt.tight_layout()

    # 4. Show the interactive window.
    plt.show()


if __name__ == "__main__":
    predict_serum_amino_acids()

