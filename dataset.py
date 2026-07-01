import torch
import numpy as np
import pandas as pd
import os
import json
import glob
from torch.utils.data import Dataset

AA3_TO_FULL = {
    "Ala": "Alanine",
    "Arg": "Arginine",
    "Asn": "Asparagine",
    "Asp": "Aspartic",
    "Cys": "Cysteine",
    "Gln": "Glutamine",
    "Glu": "Glutamate",
    "Gly": "Glycine",
    "His": "Histidine",
    "Ile": "Isoleucine",
    "Leu": "Leucine",
    "Lys": "Lysine",
    "Met": "Methionine",
    "Phe": "Phenylalanine",
    "Pro": "Proline",
    "Ser": "Serine",
    "Thr": "Threonine",
    "Trp": "Tryptophan",
    "Tyr": "Tyrosine",
    "Val": "Valine",
}

def fast_pseudo_voigt(x, sigma, gamma, eta):
    """极速伪 Voigt：仅计算窗口内 (~120 点)，非全谱"""
    L = (gamma / np.pi) / (x ** 2 + gamma ** 2)
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)
    return eta * L + (1 - eta) * G

def add_baseline_distortion(signal_len=16384, max_amplitude=0.1):
    """平滑基线畸变"""
    xs = np.linspace(0, 1, signal_len)
    baseline = np.zeros(signal_len)
    poly_order = np.random.randint(1, 4)
    coeffs = np.random.uniform(-1, 1, poly_order)
    for i, c in enumerate(coeffs):
        baseline += c * (xs ** i)
    num_sines = np.random.randint(1, 4)
    if num_sines > 0:
        freqs = np.random.uniform(0.5, 5.0, num_sines)[:, None]
        phases = np.random.uniform(0, 2 * np.pi, num_sines)[:, None]
        amps = np.random.uniform(-1, 1, num_sines)[:, None]
        sines = amps * np.sin(2 * np.pi * freqs * xs + phases)
        baseline += np.sum(sines, axis=0)
    if np.max(np.abs(baseline)) > 0:
        baseline = baseline / np.max(np.abs(baseline))
    return baseline * max_amplitude

class NMRSepDataset(Dataset):
    def __init__(self, csv_path, mapping_file='compound_mapping.json', signal_len=16384,
                 ppm_range=(10.0, 0.0), lw_range=(0.5, 2.0), eta_range=(0.1, 1.0),
                 jitter_range=(-0.001, 0.001), global_shift_range=(-0.02, 0.02), conc_range=(0.1, 10.0),
                 serum_finetune=False, serum_recipe_prob=0.65, complex_mix_prob=0.25,
                 serum_mode='auto', real_serum_recipe_csv=None, real_serum_recipe_prob=0.60,
                 real_serum_mix_prob=0.55, background_hump_prob=0.75,
                 overlap_recipe_prob=0.55, low_conc_prob=0.45,
                 experimental_pure_dir=None, experimental_mix_prob=0.0,
                 experimental_min_components=2, experimental_max_components=12,
                 experimental_scale_range=(0.1, 1.0), experimental_shift_range=(-0.03, 0.03),
                 experimental_noise_db_range=(35.0, 50.0),
                 pdf_method_mix=False, pdf_method1_prob=0.5, pdf_balanced_sampling=True,
                 dataset_length=20000, deterministic_seed=None):
        """
        - global_shift_range: 表3-1的 整体偏移量 φ_i [-0.02, 0.02]
        - jitter_range: 表3-1的 峰偏移量 δ_{i,j,k} [-0.001, 0.001]
        """
        self.df = pd.read_csv(csv_path)
        self.cols = self.df.columns
        self.signal_len = signal_len
        self.x = np.linspace(ppm_range[0], ppm_range[1], signal_len)
        self.dx = (ppm_range[0] - ppm_range[1]) / (signal_len - 1)
        
        self.lw_range = lw_range
        self.eta_range = eta_range
        self.jitter_range = jitter_range
        self.global_shift_range = global_shift_range
        self.conc_range = conc_range
        self.serum_finetune = serum_finetune
        self.serum_recipe_prob = serum_recipe_prob
        self.complex_mix_prob = complex_mix_prob
        self.serum_mode = serum_mode
        self.real_serum_recipe_csv = real_serum_recipe_csv
        self.real_serum_recipe_prob = real_serum_recipe_prob
        self.real_serum_mix_prob = real_serum_mix_prob
        self.background_hump_prob = background_hump_prob
        self.overlap_recipe_prob = overlap_recipe_prob
        self.low_conc_prob = low_conc_prob
        self.experimental_pure_dir = experimental_pure_dir
        self.experimental_mix_prob = experimental_mix_prob
        self.experimental_min_components = int(experimental_min_components)
        self.experimental_max_components = int(experimental_max_components)
        self.experimental_scale_range = experimental_scale_range
        self.experimental_shift_range = experimental_shift_range
        self.experimental_noise_db_range = experimental_noise_db_range
        self.pdf_method_mix = pdf_method_mix
        self.pdf_method1_prob = pdf_method1_prob
        self.pdf_balanced_sampling = pdf_balanced_sampling
        self.dataset_length = int(dataset_length)

        current_csv_names = sorted(self.df.iloc[:, 1].dropna().unique().tolist())
        self.mapping = {name: i for i, name in enumerate(current_csv_names)}
        self.names = list(self.mapping.keys())
        self.num_classes = len(self.names)

        if not os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'w') as f:
                    json.dump(self.mapping, f, indent=4)
            except Exception:
                pass

        self.fast_lib = {}
        for name in self.names:
            sub_df = self.df[self.df[self.cols[1]] == name]
            self.fast_lib[name] = sub_df.iloc[:, [3, 4, 5]].values.astype(np.float32)
            
        self.templates = {}
        for name in self.names:
            self.templates[name] = self._generate_template(name)
        self.experimental_templates = self._load_experimental_templates(experimental_pure_dir)
        self.experimental_names = sorted(self.experimental_templates.keys())

        self.serum_panel = [
            'Asparagine', 'Glutamine', 'Glutamate',
            'Isoleucine', 'Leucine', 'Proline', 'Valine',
            'Alanine', 'Arginine', 'Aspartic', 'Cysteine', 'Glycine',
            'Histidine', 'Lysine', 'Methionine', 'Phenylalanine',
            'Serine', 'Threonine', 'Tryptophan', 'Tyrosine'
        ]
        self.serum_panel = [name for name in self.serum_panel if name in self.mapping]
        self.weak_focus = [name for name in ['Asparagine', 'Valine'] if name in self.mapping]
        self.table_s1_recipes = [
            {'Glutamine': 160.94, 'Glutamate': 33.64},
            {'Isoleucine': 68.61, 'Leucine': 17.08},
            {'Asparagine': 51.77, 'Glutamine': 9.75, 'Glutamate': 7.34,
             'Isoleucine': 29.50, 'Leucine': 5.47, 'Proline': 52.38, 'Valine': 6.91},
            {'Glutamine': 4.27, 'Glutamate': 29.97, 'Isoleucine': 21.27,
             'Leucine': 9.57, 'Proline': 19.54},
            {'Asparagine': 43.60, 'Glutamate': 24.47, 'Leucine': 19.13,
             'Proline': 17.20, 'Valine': 26.12},
            {'Asparagine': 74.93, 'Glutamine': 81.81, 'Glutamate': 82.09,
             'Isoleucine': 84.53, 'Leucine': 93.21, 'Proline': 60.19, 'Valine': 118.31},
            {'Isoleucine': 36.23, 'Leucine': 48.11, 'Threonine': 113.02, 'Valine': 92.96},
            {'Asparagine': 69.69, 'Aspartic': 3.68, 'Cysteine': 20.22,
             'Isoleucine': 17.36, 'Methionine': 25.21},
            {'Alanine': 53.34, 'Arginine': 20.46, 'Asparagine': 34.47,
             'Glutamine': 20.12, 'Glycine': 42.20, 'Histidine': 44.67,
             'Isoleucine': 18.11, 'Lysine': 16.09, 'Phenylalanine': 16.78,
             'Proline': 36.12, 'Serine': 45.22, 'Threonine': 19.94, 'Valine': 30.42},
            {'Glycine': 102.86, 'Histidine': 54.88, 'Serine': 56.52, 'Tryptophan': 4.85},
        ]
        self.table_s1_recipes = [
            {name: conc for name, conc in recipe.items() if name in self.mapping}
            for recipe in self.table_s1_recipes
        ]
        self.table_s1_recipes = [recipe for recipe in self.table_s1_recipes if recipe]
        self.real_serum_base_recipes = self._load_real_serum_base_recipes(real_serum_recipe_csv)
        self.real_serum_base_recipe = self.real_serum_base_recipes[0] if self.real_serum_base_recipes else None
        self.hard_overlap_groups = [
            [name for name in ['Glutamine', 'Glutamate', 'Asparagine'] if name in self.mapping],
            [name for name in ['Isoleucine', 'Leucine', 'Valine', 'Proline'] if name in self.mapping],
            [name for name in ['Alanine', 'Glycine', 'Serine', 'Threonine'] if name in self.mapping],
            [name for name in ['Tyrosine', 'Phenylalanine', 'Histidine'] if name in self.mapping],
        ]
        self.hard_overlap_groups = [g for g in self.hard_overlap_groups if len(g) >= 2]
        self.mode = self._resolve_mode()

    def _resolve_mode(self):
        mode = str(self.serum_mode).lower()
        if mode == 'auto':
            return 'hybrid' if self.serum_finetune else 'standard'
        return mode

    def _normalize_recipe_name(self, name):
        name = str(name).strip()
        name = name.replace('L-', '').replace('D-', '').strip()
        aliases = {
            'Glutamic': 'Glutamate',
            'Glutamic acid': 'Glutamate',
            'Glutamic Acid': 'Glutamate',
            'Aspartic acid': 'Aspartic',
            'Aspartic Acid': 'Aspartic',
            'Cystine': 'Cysteine',
        }
        if name in aliases:
            return aliases[name]
        return name.replace(' acid', '').replace(' Acid', '')

    def _iter_real_serum_recipe_paths(self, real_serum_recipe_csv):
        if real_serum_recipe_csv is None:
            return []
        if isinstance(real_serum_recipe_csv, (list, tuple)):
            paths = []
            for item in real_serum_recipe_csv:
                paths.extend(self._iter_real_serum_recipe_paths(item))
            return paths
        path = str(real_serum_recipe_csv)
        if os.path.isdir(path):
            return sorted(glob.glob(os.path.join(path, 'def_mix_serum_*.csv')))
        if any(ch in path for ch in ['*', '?', '[']):
            return sorted(glob.glob(path))
        return [path] if os.path.exists(path) else []

    def _load_real_serum_base_recipes(self, real_serum_recipe_csv):
        recipes = []
        for path in self._iter_real_serum_recipe_paths(real_serum_recipe_csv):
            recipe = self._load_real_serum_base_recipe(path)
            if recipe:
                recipes.append(recipe)
        return recipes

    def _load_real_serum_base_recipe(self, real_serum_recipe_csv):
        if not real_serum_recipe_csv or not os.path.exists(real_serum_recipe_csv):
            return None
        try:
            df = pd.read_csv(real_serum_recipe_csv)
        except Exception:
            return None

        cols = {str(c).lower().strip(): c for c in df.columns}
        name_col = None
        conc_col = None
        preferred_conc = None
        for key, col in cols.items():
            if key in ('model_channel', 'metabolite', 'compound name') or 'compound' in key or 'name' in key:
                name_col = col
            if key == 'bayesil_mm':
                preferred_conc = col
            elif 'bayesil' in key:
                conc_col = col
        if preferred_conc is not None:
            conc_col = preferred_conc
        if name_col is None or conc_col is None:
            return None

        recipe = {}
        conc_col_key = str(conc_col).lower()
        for _, row in df.iterrows():
            name = self._normalize_recipe_name(row[name_col])
            if name not in self.mapping:
                continue
            try:
                conc = float(row[conc_col])
            except Exception:
                continue
            if conc <= 0:
                continue
            if 'mm' not in conc_col_key:
                conc = conc / 1000.0
            recipe[name] = conc
        return recipe or None
    def _pvoigt_peak(self, center_ppm, height, lw_hz, eta):
        ppm_per_point = self.dx
        lw_ppm = lw_hz / 500.0
        sigma = (lw_ppm / 2.355) * (1 - eta)
        gamma = (lw_ppm / 2) * eta
        half_window_pts = int(np.ceil((5 * sigma + 10 * gamma) / ppm_per_point)) + 2
        half_window_pts = max(half_window_pts, 3)

        center_idx = int(round((self.x[0] - center_ppm) / ppm_per_point))
        start = max(0, center_idx - half_window_pts)
        end = min(self.signal_len, center_idx + half_window_pts + 1)

        x_local = self.x[start:end] - center_ppm
        peak_chunk = fast_pseudo_voigt(x_local, sigma, gamma, eta)
        peak_chunk *= height
        return peak_chunk, start, end

    def _generate_template(self, name):
        """预计算纯净模板 (无 jitter, conc=1.0)"""
        peaks_data = self.fast_lib[name]
        spectrum = np.zeros(self.signal_len, dtype=np.float32)
        if len(peaks_data) == 0:
            return spectrum
        eta = 0.5
        for pos, h, lw in peaks_data:
            chunk, s, e = self._pvoigt_peak(float(pos), float(h), float(lw), eta)
            spectrum[s:e] += chunk
        mx = np.max(spectrum)
        if mx > 0:
            spectrum /= mx
        return spectrum

    def _safe_area_normalize(self, spectrum):
        spectrum = np.clip(np.asarray(spectrum, dtype=np.float32), 0.0, None)
        area = float(np.trapz(spectrum, self.x))
        if abs(area) > 1e-12:
            return (spectrum / abs(area)).astype(np.float32)
        mx = float(np.max(spectrum))
        if mx > 0:
            return (spectrum / mx).astype(np.float32)
        return spectrum.astype(np.float32)

    def _load_experimental_templates(self, experimental_pure_dir):
        templates = {}
        if not experimental_pure_dir or not os.path.isdir(str(experimental_pure_dir)):
            return templates

        # Preferred format: network_ready directory with normalized pure spectra.
        # Files:
        #   metadata.json, ppm_grid_float32.npy, pure_spectra_float32.npy
        # The pure spectra are area-normalized on their native ppm grid and are
        # resampled to the model ppm grid here.
        metadata_path = os.path.join(str(experimental_pure_dir), "metadata.json")
        ppm_path = os.path.join(str(experimental_pure_dir), "ppm_grid_float32.npy")
        spectra_path = os.path.join(str(experimental_pure_dir), "pure_spectra_float32.npy")
        if os.path.exists(metadata_path) and os.path.exists(ppm_path) and os.path.exists(spectra_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                order = metadata.get("amino_acid_order", [])
                ppm = np.load(ppm_path).astype(np.float32)
                spectra = np.load(spectra_path).astype(np.float32)
                ppm_order = np.argsort(ppm)
                ppm_sorted = ppm[ppm_order]
                for row_idx, short_name in enumerate(order):
                    if row_idx >= spectra.shape[0]:
                        continue
                    name = AA3_TO_FULL.get(str(short_name))
                    if name is None or name not in self.mapping:
                        continue
                    spec_sorted = spectra[row_idx][ppm_order]
                    resampled = np.interp(self.x, ppm_sorted, spec_sorted, left=0.0, right=0.0)
                    templates[name] = self._safe_area_normalize(resampled)
            except Exception:
                templates = {}

        for path in sorted(glob.glob(os.path.join(str(experimental_pure_dir), "*.csv"))):
            short_name = os.path.splitext(os.path.basename(path))[0]
            name = AA3_TO_FULL.get(short_name)
            if name is None or name not in self.mapping:
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            cols = {str(c).lower().strip(): c for c in df.columns}
            ppm_col = cols.get("ppm") or cols.get("ppm_norm")
            if ppm_col is None:
                ppm_matches = [c for c in df.columns if "ppm" in str(c).lower()]
                ppm_col = ppm_matches[0] if ppm_matches else df.columns[0]

            spec_col = cols.get("spectrum")
            if spec_col is None:
                non_ppm_cols = [c for c in df.columns if c != ppm_col]
                spec_col = non_ppm_cols[0] if non_ppm_cols else None
            if spec_col is None:
                continue

            ppm = df[ppm_col].to_numpy(dtype=np.float32)
            spec = df[spec_col].to_numpy(dtype=np.float32)
            order = np.argsort(ppm)
            ppm = ppm[order]
            spec = spec[order]
            resampled = np.interp(self.x, ppm, spec, left=0.0, right=0.0).astype(np.float32)
            templates[name] = self._safe_area_normalize(resampled)

        return templates

    def _shift_experimental_spectrum(self, spectrum, shift_ppm):
        x_asc = self.x[::-1]
        s_asc = spectrum[::-1]
        shifted = np.interp(x_asc - shift_ppm, x_asc, s_asc, left=0.0, right=0.0)
        return shifted[::-1].astype(np.float32)

    def _add_snr_noise(self, mixture):
        snr_db = np.random.uniform(*self.experimental_noise_db_range)
        rms = float(np.sqrt(np.mean(np.square(mixture))) + 1e-12)
        noise_std = rms / (10.0 ** (snr_db / 20.0))
        return (mixture + np.random.normal(0.0, noise_std, mixture.shape)).astype(np.float32)

    def _generate_experimental_superposition(self):
        NUM_CHANNELS = 20
        max_n = min(max(1, self.experimental_max_components), len(self.experimental_names))
        min_n = min(max(1, self.experimental_min_components), max_n)
        if max_n < 1:
            return None

        n = np.random.randint(min_n, max_n + 1)
        selected = list(np.random.choice(self.experimental_names, n, replace=False))

        final_targets = torch.zeros(NUM_CHANNELS, self.signal_len)
        mixture = np.zeros(self.signal_len, dtype=np.float32)
        actual_names = ["None"] * NUM_CHANNELS

        for name in selected:
            scale = np.random.uniform(*self.experimental_scale_range)
            shift = np.random.uniform(*self.experimental_shift_range)
            spec = self._shift_experimental_spectrum(self.experimental_templates[name], shift)
            spec = (scale * spec).astype(np.float32)
            compound_id = self.mapping[name]
            final_targets[compound_id] = torch.FloatTensor(spec)
            actual_names[compound_id] = name
            mixture += spec

        mix_max = float(np.max(mixture))
        if mix_max > 0:
            mixture /= mix_max
            final_targets /= mix_max

        mixture = np.clip(self._add_snr_noise(mixture), 0.0, None)
        mixture_log = np.log10(mixture + 1e-6)
        dual_mixture = np.stack([mixture, mixture_log], axis=0).astype(np.float32)
        return torch.FloatTensor(dual_mixture), final_targets, actual_names

    def generate_experimental_recipe_sample(self, recipe, augment=True):
        """Build one sample from experimental pure spectra and an explicit recipe.

        The recipe values are treated as concentration-like linear coefficients.
        This is used for Table S1 calibration/evaluation only; it should not be
        used as the main training source unless the caller intentionally wants to
        train on fixed recipes.
        """
        NUM_CHANNELS = 20
        if not self.experimental_templates:
            return None

        final_targets = torch.zeros(NUM_CHANNELS, self.signal_len)
        mixture = np.zeros(self.signal_len, dtype=np.float32)
        actual_names = ["None"] * NUM_CHANNELS

        for name, concentration in recipe.items():
            if name not in self.mapping or name not in self.experimental_templates:
                continue
            shift = np.random.uniform(*self.experimental_shift_range) if augment else 0.0
            spec = self._shift_experimental_spectrum(self.experimental_templates[name], shift)
            spec = (float(concentration) * spec).astype(np.float32)
            compound_id = self.mapping[name]
            final_targets[compound_id] = torch.FloatTensor(spec)
            actual_names[compound_id] = name
            mixture += spec

        mix_max = float(np.max(mixture))
        if mix_max > 0:
            mixture /= mix_max
            final_targets /= mix_max

        if augment:
            if np.random.rand() < 0.35:
                baseline = add_baseline_distortion(self.signal_len, np.random.uniform(0.015, 0.08))
                mixture = np.clip(mixture + baseline, 0.0, None)
            mixture = np.clip(self._add_snr_noise(mixture), 0.0, None)

        mixture_log = np.log10(np.clip(mixture, 0.0, None) + 1e-6)
        dual_mixture = np.stack([mixture, mixture_log], axis=0).astype(np.float32)
        return torch.FloatTensor(dual_mixture), final_targets, actual_names

    def _pdf_use_method1(self, idx):
        ratio = float(self.pdf_method1_prob)
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        if self.pdf_balanced_sampling and abs(ratio - 0.5) < 1e-12:
            return (int(idx) % 2) == 0
        if self.pdf_balanced_sampling:
            value = ((int(idx) * 1103515245 + 12345) & 0x7FFFFFFF) / float(0x80000000)
            return value < ratio
        return np.random.rand() < ratio

    def _generate_pdf_peak_superposition(self):
        NUM_CHANNELS = 20
        max_n = min(7, len(self.names))
        min_n = min(2, max_n)
        n = np.random.randint(min_n, max_n + 1)
        selected = list(np.random.choice(self.names, n, replace=False))

        final_targets = torch.zeros(NUM_CHANNELS, self.signal_len)
        mixture = np.zeros(self.signal_len, dtype=np.float32)
        actual_names = ["None"] * NUM_CHANNELS

        for name in selected:
            concentration = np.random.uniform(*self.conc_range)
            spec = self._generate_compound(name, jitter_active=True, concentration=concentration)
            compound_id = self.mapping[name]
            final_targets[compound_id] = torch.FloatTensor(spec)
            actual_names[compound_id] = name
            mixture += spec

        mix_max = float(np.max(mixture))
        if mix_max > 0:
            mixture /= mix_max
            final_targets /= mix_max

        mixture = np.clip(self._add_snr_noise(mixture), 0.0, None)
        mixture_log = np.log10(mixture + 1e-6)
        dual_mixture = np.stack([mixture, mixture_log], axis=0).astype(np.float32)
        return torch.FloatTensor(dual_mixture), final_targets, actual_names

    def _generate_compound(self, name, jitter_active=True, concentration=None):
        peaks_data = self.fast_lib[name]
        n_peaks = peaks_data.shape[0]
        if n_peaks == 0:
            return np.zeros(self.signal_len, dtype=np.float32)

        eta = np.random.uniform(*self.eta_range) if jitter_active else 0.5
        lw_factor = np.random.uniform(*self.lw_range, size=(n_peaks,)) if jitter_active else np.ones(n_peaks)
        
        # 物理学解耦：整体漂移 vs 内部裂分微扰
        global_shift = np.random.uniform(*self.global_shift_range) if jitter_active else 0.0
        local_jitter = np.random.uniform(*self.jitter_range, size=(n_peaks,)) if jitter_active else np.zeros(n_peaks)

        spectrum = np.zeros(self.signal_len, dtype=np.float32)
        for i in range(n_peaks):
            pos = float(peaks_data[i, 0]) + global_shift + local_jitter[i]
            h = float(peaks_data[i, 1])
            lw = float(peaks_data[i, 2]) * lw_factor[i]
            
            if pos > self.x[0] or pos < self.x[-1]:
                continue
                
            chunk, s, e = self._pvoigt_peak(pos, h, lw, eta)
            spectrum[s:e] += chunk

        if jitter_active:
            if concentration is None:
                spectrum *= np.random.uniform(*self.conc_range)
            else:
                spectrum *= float(concentration)

        return spectrum

    def _random_overlap_recipe(self):
        groups = [g for g in self.hard_overlap_groups if len(g) >= 2]
        if not groups:
            return None

        main_group = list(groups[np.random.randint(0, len(groups))])
        selected = list(main_group)
        if np.random.rand() < 0.45 and len(groups) > 1:
            extra_group = list(groups[np.random.randint(0, len(groups))])
            for name in extra_group:
                if name not in selected:
                    selected.append(name)

        background_pool = [name for name in self.serum_panel if name not in selected]
        if background_pool:
            n_bg = np.random.randint(1, min(5, len(background_pool)) + 1)
            selected.extend(list(np.random.choice(background_pool, n_bg, replace=False)))
        selected = list(dict.fromkeys(selected))

        if self.mode == 'real_serum':
            low, high = 0.006, 0.35
        else:
            low, high = 2.0, 110.0

        recipe = {}
        group_base = np.random.uniform(low * 4.0, high)
        for name in selected:
            if name in main_group:
                if np.random.rand() < self.low_conc_prob:
                    recipe[name] = np.random.uniform(low, min(group_base * 0.45, high))
                else:
                    recipe[name] = group_base * np.random.uniform(0.55, 1.20)
            else:
                recipe[name] = np.random.uniform(low, high * 0.75)
        return recipe

    def _random_serum_recipe(self):
        if self.serum_finetune and np.random.rand() < self.overlap_recipe_prob:
            recipe = self._random_overlap_recipe()
            if recipe:
                return recipe

        if self.mode == 'real_serum':
            if self.real_serum_base_recipes and np.random.rand() < self.real_serum_recipe_prob:
                base_idx = np.random.randint(0, len(self.real_serum_base_recipes))
                recipe = dict(self.real_serum_base_recipes[base_idx])
                recipe = {name: conc * np.random.uniform(0.70, 1.35) for name, conc in recipe.items()}
                if np.random.rand() < 0.70:
                    overlap = self._random_overlap_recipe()
                    if overlap:
                        for name in overlap:
                            if name in recipe and np.random.rand() < 0.45:
                                recipe[name] = overlap[name]
                return recipe

            pool = [name for name in self.names if name in self.mapping]
            n = np.random.randint(min(8, len(pool)), min(15, len(pool)) + 1)
            selected = list(np.random.choice(pool, n, replace=False))
            for group in self.hard_overlap_groups:
                if any(name in selected for name in group) and np.random.rand() < 0.80:
                    for name in group:
                        if name not in selected and len(selected) < len(pool) and np.random.rand() < 0.5:
                            selected.append(name)
            selected = list(dict.fromkeys(selected))
            low = np.random.uniform(0.005, 0.03)
            high = np.random.uniform(0.05, 0.40)
            recipe = {}
            for name in selected:
                if name in ['Alanine', 'Serine', 'Glycine', 'Glutamine', 'Glutamate', 'Valine', 'Leucine']:
                    recipe[name] = np.random.uniform(low * 1.5, high)
                else:
                    recipe[name] = np.random.uniform(low, high)
            return recipe

        if self.table_s1_recipes and np.random.rand() < self.serum_recipe_prob:
            base_recipe = dict(self.table_s1_recipes[np.random.randint(0, len(self.table_s1_recipes))])
            return {name: conc * np.random.uniform(0.85, 1.15) for name, conc in base_recipe.items()}

        if self.serum_panel and np.random.rand() < self.complex_mix_prob:
            n = np.random.randint(min(5, len(self.serum_panel)), min(7, len(self.serum_panel)) + 1)
            selected = list(np.random.choice(self.serum_panel, n, replace=False))
            for name in self.weak_focus:
                if name not in selected and np.random.rand() < 0.75:
                    if len(selected) >= n:
                        selected[np.random.randint(0, len(selected))] = name
                    else:
                        selected.append(name)
            low = np.random.uniform(2.0, 12.0)
            high = min(110.0, low * np.random.uniform(4.0, 10.0))
            return {name: np.random.uniform(low, high) for name in selected}

        return None
    def _add_real_serum_background(self, mixture):
        if np.random.rand() < self.background_hump_prob:
            n_humps = np.random.randint(1, 4)
            for _ in range(n_humps):
                center = np.random.uniform(0.8, 4.2)
                width = np.random.uniform(0.015, 0.08)
                height = np.random.uniform(0.02, 0.10) * max(float(np.max(mixture)), 1e-6)
                mixture += height * np.exp(-0.5 * ((self.x - center) / width) ** 2).astype(np.float32)
        return mixture

    def __getitem__(self, idx):
        NUM_CHANNELS = 20

        if self.pdf_method_mix:
            if self.experimental_templates and self._pdf_use_method1(idx):
                experimental_sample = self._generate_experimental_superposition()
                if experimental_sample is not None:
                    return experimental_sample
            return self._generate_pdf_peak_superposition()

        recipe = self._random_serum_recipe() if self.serum_finetune else None

        if self.experimental_templates and np.random.rand() < self.experimental_mix_prob:
            experimental_sample = self._generate_experimental_superposition()
            if experimental_sample is not None:
                return experimental_sample

        if recipe is None:
            if self.mode == 'real_serum' and self.serum_finetune and self.serum_panel and np.random.rand() < self.real_serum_mix_prob:
                num_to_mix = np.random.randint(min(8, len(self.serum_panel)), min(15, len(self.serum_panel)) + 1)
                selected = list(np.random.choice(self.serum_panel, num_to_mix, replace=False))
                for group in self.hard_overlap_groups:
                    if any(name in selected for name in group) and np.random.rand() < 0.75:
                        for name in group:
                            if name in self.mapping and name not in selected and len(selected) < NUM_CHANNELS:
                                selected.append(name)
            elif self.serum_finetune and self.serum_panel and np.random.rand() < 0.70:
                num_to_mix = np.random.randint(5, min(7, len(self.serum_panel)) + 1)
                selected = list(np.random.choice(self.serum_panel, num_to_mix, replace=False))
            else:
                num_to_mix = np.random.randint(2, min(7, len(self.names)) + 1)
                selected = list(np.random.choice(self.names, num_to_mix, replace=False))

            hard_group = ['Glutamate', 'Glutamine', 'Proline', 'Asparagine', 'Valine',
                          'Alanine', 'Serine', 'Glycine', 'Leucine', 'Isoleucine']
            if any(h in selected for h in hard_group) and np.random.rand() < 0.70:
                for h in hard_group:
                    if h in self.mapping and h not in selected and len(selected) < NUM_CHANNELS:
                        selected.append(h)
            recipe = {name: None for name in selected}

        final_targets = torch.zeros(NUM_CHANNELS, self.signal_len)
        mixture = np.zeros(self.signal_len, dtype=np.float32)
        actual_names = ["None"] * NUM_CHANNELS

        for name, concentration in recipe.items():
            spec = self._generate_compound(name, jitter_active=True, concentration=concentration)
            compound_id = self.mapping[name]
            final_targets[compound_id] = torch.FloatTensor(spec)
            actual_names[compound_id] = name
            mixture += spec

        mix_max = float(np.max(mixture))
        if mix_max > 0:
            mixture /= mix_max
            final_targets /= mix_max

        if np.random.rand() < 0.5:
            baseline = add_baseline_distortion(self.signal_len, np.random.uniform(0.05, 0.15))
            mixture = np.clip(mixture + baseline, 0, None)

        if self.mode == 'real_serum':
            mixture = self._add_real_serum_background(mixture)

        noise_std = np.random.uniform(0.001, 0.03) * float(np.max(mixture))
        mixture += np.random.normal(0, noise_std, mixture.shape).astype(np.float32)

        if np.random.rand() < 0.9:
            dss_h = np.random.uniform(0.1, 0.5) * float(np.max(mixture))
            fake_dss = (np.exp(-0.5 * ((self.x - 0.0) / 0.0015) ** 2)) * dss_h
            mixture += fake_dss.astype(np.float32)

        mixture_log = np.log10(np.clip(mixture, 0.0, None) + 1e-6)
        dual_mixture = np.stack([mixture, mixture_log], axis=0).astype(np.float32)

        return torch.FloatTensor(dual_mixture), final_targets, actual_names

    def __len__(self):
        return self.dataset_length
