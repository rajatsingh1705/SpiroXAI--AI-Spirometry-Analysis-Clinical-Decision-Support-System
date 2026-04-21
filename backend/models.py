"""
Model definitions and ensemble predictor for lung disease classification.
Loads XGBoost, LightGBM, DNN, and FT-Transformer models at startup.
SHAP explanations computed via XGBoost TreeExplainer only (for speed).
"""

import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")
SUMMARY_PATH = os.path.join(PROJECT_DIR, "experiment_summary.json")

# ── Load experiment config ─────────────────────────────────────────────────────
_summary = {}
if os.path.exists(SUMMARY_PATH):
    with open(SUMMARY_PATH) as f:
        _summary = json.load(f)

CLASS_NAMES = _summary.get("class_names", ["Normal", "Obstruction", "Restriction"])
FEATURE_NAMES = _summary.get("feature_names", [])
N_FEATURES = len(FEATURE_NAMES)
ENSEMBLE_WEIGHTS = _summary.get("ensemble_weights", {"xgb": 0.3, "lgb": 0.35, "ft": 0.25, "dnn": 0.1})
ENSEMBLE_BIAS = np.array(_summary.get("ensemble_bias", [0.0, 0.0, 0.0]), dtype=np.float32)
FT_CONFIG = _summary.get("ft_config", {"d_token": 256, "n_layers": 6, "n_heads": 8, "ffn_factor": 4 / 3})


# ── PyTorch model architectures (must match training exactly) ──────────────────
_torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch_available = True
except ImportError:
    pass

if _torch_available:

    class FeatureTokenizer(nn.Module):
        def __init__(self, n_features, d_token):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.zeros(n_features, d_token))
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        def forward(self, x):
            return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, d_token, n_heads, dropout=0.1):
            super().__init__()
            assert d_token % n_heads == 0
            self.n_heads = n_heads
            self.d_head = d_token // n_heads
            self.scale = self.d_head ** -0.5
            self.qkv = nn.Linear(d_token, 3 * d_token, bias=False)
            self.out = nn.Linear(d_token, d_token)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            B, T, D = x.shape
            H, Dh = self.n_heads, self.d_head
            QKV = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
            Q, K, V = QKV[0], QKV[1], QKV[2]
            attn = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
            attn = self.drop(attn)
            out = (attn @ V).transpose(1, 2).reshape(B, T, D)
            return self.out(out), attn

    class FTBlock(nn.Module):
        def __init__(self, d_token, n_heads, ffn_factor=4 / 3, dropout=0.1):
            super().__init__()
            d_ffn = int(d_token * ffn_factor)
            self.norm1 = nn.LayerNorm(d_token)
            self.norm2 = nn.LayerNorm(d_token)
            self.attn = MultiHeadSelfAttention(d_token, n_heads, dropout)
            self.ffn = nn.Sequential(
                nn.Linear(d_token, d_ffn), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ffn, d_token), nn.Dropout(dropout),
            )

        def forward(self, x):
            a, w = self.attn(self.norm1(x))
            x = x + a
            x = x + self.ffn(self.norm2(x))
            return x, w

    class FTTransformer(nn.Module):
        def __init__(self, n_features, n_classes, d_token=256, n_layers=6,
                     n_heads=8, ffn_factor=4 / 3, attn_dropout=0.10, ffn_dropout=0.15):
            super().__init__()
            self.tokenizer = FeatureTokenizer(n_features, d_token)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.blocks = nn.ModuleList([
                FTBlock(d_token, n_heads, ffn_factor,
                        attn_dropout if i < n_layers - 1 else 0.0)
                for i in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Sequential(
                nn.ReLU(), nn.Dropout(ffn_dropout), nn.Linear(d_token, n_classes)
            )

        def forward(self, x, return_attn=False):
            B = x.shape[0]
            tokens = torch.cat([self.cls_token.expand(B, -1, -1), self.tokenizer(x)], dim=1)
            attns = []
            for block in self.blocks:
                tokens, w = block(tokens)
                attns.append(w)
            logits = self.head(self.norm(tokens[:, 0]))
            return (logits, attns) if return_attn else logits

    class VanillaDNN(nn.Module):
        def __init__(self, n_features, n_classes, dropout=0.20):
            super().__init__()
            dims = [n_features, 1024, 512, 256, 128]
            layers = []
            for in_d, out_d in zip(dims[:-1], dims[1:]):
                layers += [
                    nn.Linear(in_d, out_d),
                    nn.BatchNorm1d(out_d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            layers.append(nn.Linear(128, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


# ── Ensemble Predictor ─────────────────────────────────────────────────────────
class EnsemblePredictor:
    """Loads all 4 models at startup (singleton) and runs weighted ensemble."""

    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.ft_model = None
        self.dnn_model = None
        self.qt = None
        self.shap_explainer = None
        self.device = "cpu"
        self.loaded = False
        self._load()

    def _load(self):
        """Load all models from saved_models/ directory."""
        from feature_engineering import load_quantile_transformer

        print(f"[ML] Loading models from: {MODELS_DIR}")

        # QuantileTransformer
        self.qt = load_quantile_transformer()
        if self.qt:
            print("[ML] ✅ QuantileTransformer loaded")

        # XGBoost
        xgb_path = os.path.join(MODELS_DIR, "xgb_final.json")
        if os.path.exists(xgb_path):
            try:
                import xgboost as xgb
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(xgb_path)
                print("[ML] ✅ XGBoost loaded")
            except Exception as e:
                print(f"[ML] ⚠ XGBoost load failed: {e}")

        # LightGBM
        lgb_path = os.path.join(MODELS_DIR, "lgb_final.txt")
        if os.path.exists(lgb_path):
            try:
                import lightgbm as lgb
                self.lgb_model = lgb.Booster(model_file=lgb_path)
                print("[ML] ✅ LightGBM loaded")
            except Exception as e:
                print(f"[ML] ⚠ LightGBM load failed: {e}")

        # PyTorch models
        if _torch_available:
            ft_path = os.path.join(MODELS_DIR, "ft_transformer_final.pt")
            if os.path.exists(ft_path):
                try:
                    cfg = FT_CONFIG
                    self.ft_model = FTTransformer(
                        n_features=N_FEATURES, n_classes=3,
                        d_token=cfg["d_token"], n_layers=cfg["n_layers"],
                        n_heads=cfg["n_heads"], ffn_factor=cfg["ffn_factor"],
                    )
                    ckpt = torch.load(ft_path, map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        self.ft_model.load_state_dict(ckpt["model_state_dict"])
                    else:
                        self.ft_model.load_state_dict(ckpt)
                    self.ft_model.eval()
                    print("[ML] ✅ FT-Transformer loaded")
                except Exception as e:
                    print(f"[ML] ⚠ FT-Transformer load failed: {e}")

            dnn_path = os.path.join(MODELS_DIR, "dnn_final.pt")
            if os.path.exists(dnn_path):
                try:
                    self.dnn_model = VanillaDNN(N_FEATURES, 3)
                    ckpt = torch.load(dnn_path, map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        self.dnn_model.load_state_dict(ckpt["model_state_dict"])
                    else:
                        self.dnn_model.load_state_dict(ckpt)
                    self.dnn_model.eval()
                    print("[ML] ✅ DNN loaded")
                except Exception as e:
                    print(f"[ML] ⚠ DNN load failed: {e}")

        # SHAP explainer (XGBoost only for speed)
        if self.xgb_model is not None:
            try:
                import shap
                self.shap_explainer = shap.TreeExplainer(self.xgb_model)
                print("[ML] ✅ SHAP TreeExplainer ready")
            except Exception as e:
                print(f"[ML] ⚠ SHAP init failed: {e}")

        self.loaded = any([self.xgb_model, self.lgb_model, self.ft_model, self.dnn_model])
        if self.loaded:
            print("[ML] ✅ Ensemble ready")
        else:
            print("[ML] ⚠ No models loaded — heuristic fallback active")

    def _heuristic_fallback(self, raw_input: dict) -> dict:
        """ATS/ERS clinical heuristic when models are unavailable."""
        pef = raw_input.get("Baseline_PEF_Ls", 8.0)
        fef = raw_input.get("Baseline_FEF2575_Ls", 4.0)
        fet = raw_input.get("Baseline_Forced_Expiratory_Time", 3.0)

        if fef < 0.6 or (pef / (fef + 1e-9) > 5.0 and fet > 10.0):
            proba = [0.10, 0.80, 0.10]
        elif pef < 2.0 and fet < 4.0:
            proba = [0.15, 0.15, 0.70]
        elif pef > 5.0 and fef > 3.0:
            proba = [0.90, 0.05, 0.05]
        else:
            proba = [0.45, 0.30, 0.25]

        proba = np.array(proba, dtype=np.float32)
        return {
            "probabilities": proba,
            "predicted_class": int(np.argmax(proba)),
            "is_heuristic": True,
        }

    def predict(self, raw_input: dict) -> dict:
        """Run full ensemble prediction."""
        if not self.loaded:
            return self._heuristic_fallback(raw_input)

        from feature_engineering import prepare_features
        X_raw, X_scaled = prepare_features(raw_input, FEATURE_NAMES, self.qt)

        probas = []
        weights_used = {}

        # XGBoost (raw features)
        if self.xgb_model is not None:
            try:
                p = self.xgb_model.predict_proba(X_raw)
                probas.append(ENSEMBLE_WEIGHTS["xgb"] * p)
                weights_used["xgb"] = ENSEMBLE_WEIGHTS["xgb"]
            except Exception as e:
                print(f"[ML] XGB predict error: {e}")

        # LightGBM (raw features)
        if self.lgb_model is not None:
            try:
                p = self.lgb_model.predict(X_raw)
                if p.ndim == 1:
                    p = p.reshape(1, -1)
                probas.append(ENSEMBLE_WEIGHTS["lgb"] * p)
                weights_used["lgb"] = ENSEMBLE_WEIGHTS["lgb"]
            except Exception as e:
                print(f"[ML] LGB predict error: {e}")

        # FT-Transformer (scaled features)
        if self.ft_model is not None and _torch_available:
            try:
                with torch.no_grad():
                    logits = self.ft_model(torch.FloatTensor(X_scaled))
                    p = torch.softmax(logits, dim=-1).numpy()
                probas.append(ENSEMBLE_WEIGHTS["ft"] * p)
                weights_used["ft"] = ENSEMBLE_WEIGHTS["ft"]
            except Exception as e:
                print(f"[ML] FT predict error: {e}")

        # DNN (scaled features)
        if self.dnn_model is not None and _torch_available:
            try:
                with torch.no_grad():
                    self.dnn_model.eval()
                    logits = self.dnn_model(torch.FloatTensor(X_scaled))
                    p = torch.softmax(logits, dim=-1).numpy()
                probas.append(ENSEMBLE_WEIGHTS["dnn"] * p)
                weights_used["dnn"] = ENSEMBLE_WEIGHTS["dnn"]
            except Exception as e:
                print(f"[ML] DNN predict error: {e}")

        if not probas:
            return self._heuristic_fallback(raw_input)

        # Weighted average
        total_w = sum(weights_used.values())
        ensemble_proba = sum(probas) / total_w

        # Clinical rule boosters
        pef = float(raw_input.get("Baseline_PEF_Ls", 8.0))
        fef = float(raw_input.get("Baseline_FEF2575_Ls", 4.0))
        fet = float(raw_input.get("Baseline_Forced_Expiratory_Time", 3.0))

        biased_logits = ensemble_proba.copy()

        if (fef < 0.6) or (fef < 1.2 and fet > 8.0):
            biased_logits[0, 1] += 1.5

        if (pef < 1.5 and fet < 4.0) or (pef < 3.5 and fet < 2.0 and (fef / max(1e-9, pef) > 0.7)):
            biased_logits[0, 2] += 1.5

        pred_idx = int(np.argmax(biased_logits, axis=1)[0])

        # Temperature-based softmax for smooth UI probabilities
        exp_biased = np.exp(biased_logits[0] * 2.5)
        exp_biased /= np.sum(exp_biased)
        final_proba = exp_biased.astype(np.float32)

        return {
            "probabilities": final_proba,
            "predicted_class": pred_idx,
            "is_heuristic": False,
            "X_raw": X_raw,
        }

    def explain(self, raw_input: dict, pred_result: dict) -> dict:
        """Generate SHAP explanations using XGBoost model only. Returns top 5 features."""
        import pandas as pd

        top_features = []
        text_summary = ""
        X_raw = pred_result.get("X_raw")
        pred_class = pred_result["predicted_class"]
        class_name = CLASS_NAMES[pred_class]

        if self.shap_explainer is not None and X_raw is not None:
            try:
                X_df = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
                shap_vals = self.shap_explainer.shap_values(X_df)

                if isinstance(shap_vals, list):
                    sv = shap_vals[pred_class][0]
                else:
                    sv = shap_vals[0, :, pred_class]

                # Top 5 features by absolute SHAP value
                top_idx = np.argsort(np.abs(sv))[-5:][::-1]
                for idx in top_idx:
                    top_features.append({
                        "feature": FEATURE_NAMES[idx],
                        "contribution": round(float(sv[idx]), 4),
                        "direction": "positive" if sv[idx] > 0 else "negative",
                        "value": round(float(X_raw[0, idx]), 4),
                    })

                # Human-readable summary
                top2 = top_features[:2]
                parts = []
                for f in top2:
                    readable = f["feature"].replace("_", " ").replace("Baseline ", "")
                    direction = "low" if f["direction"] == "negative" else "high"
                    parts.append(f"{direction} {readable}")
                if parts:
                    text_summary = (
                        f"Prediction is {class_name} mainly due to "
                        + " and ".join(parts) + "."
                    )
                else:
                    text_summary = f"Prediction is {class_name}."

            except Exception as e:
                print(f"[ML] SHAP error: {e}")

        if not top_features:
            text_summary = f"Prediction is {class_name} based on spirometry pattern analysis."

        return {
            "top_features": top_features,
            "text_summary": text_summary,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
_predictor = None


def get_predictor() -> EnsemblePredictor:
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor
