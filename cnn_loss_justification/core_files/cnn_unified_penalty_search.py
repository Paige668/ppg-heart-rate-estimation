"""Run the retained CNN shared-penalty search from the current workspace layout."""

import importlib.util
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GENERAL_ROOT = SCRIPT_DIR.parent

module_path = GENERAL_ROOT / 'cnn_unified' / 'raw_robust_inherited_v2_unified_general_shared.py'
spec = importlib.util.spec_from_file_location('cnn_general_shared', module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

base_out = SCRIPT_DIR / 'cnn_penalty_search_runs'
base_out.mkdir(parents=True, exist_ok=True)

candidates = [
    {'name': 'cfg_a', 'bias_penalty': 0.40, 'over_penalty': 0.18},
    {'name': 'cfg_b', 'bias_penalty': 0.48, 'over_penalty': 0.24},
    {'name': 'cfg_c', 'bias_penalty': 0.55, 'over_penalty': 0.30},
]

results = []
for cand in candidates:
    for use_full_pipeline, case_name in [(False, 'no_prep'), (True, 'full_pipeline')]:
        config = mod.RouteConfig(
            model_name=f"CNN Unified General Search {cand['name']} ({case_name})",
            use_full_pipeline=use_full_pipeline,
            feature_high=3.0,
            lr=6e-4,
            motion_gate_scale=0.45,
            attn_temperature=0.58,
            delta_cap_bpm=5.0,
            scale_cap=0.06,
            bias_penalty=cand['bias_penalty'],
            over_penalty=cand['over_penalty'],
        )
        out_dir = base_out / cand['name'] / case_name
        out_dir.mkdir(parents=True, exist_ok=True)
        res = mod.run_experiment(
            config=config,
            out_dir=str(out_dir),
            prediction_file_name=f"predictions_{cand['name']}_{case_name}.csv",
        )
        res['case'] = case_name
        res['config_name'] = cand['name']
        res['bias_penalty'] = cand['bias_penalty']
        res['over_penalty'] = cand['over_penalty']
        results.append(res)

(Path(base_out) / 'search_results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
