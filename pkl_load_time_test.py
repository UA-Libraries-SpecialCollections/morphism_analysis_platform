import os
import pickle
import time
import numpy as np

path = r"C:\datasets\ISandT_published_test_sets\DPQ_opal_500_k5-8\DPQ_opal_500_k5-8_morphism_comparison_selected_u0020_0000001_0000001.pkl"

t0 = time.perf_counter()
with open(path, "rb") as f:
    obj = pickle.load(f)
t1 = time.perf_counter()

payload = obj.get("morphism_comparison", obj) if isinstance(obj, dict) else obj
matches = np.asarray(payload.get("matches", []))
plot_cache = payload.get("plot_cache", {}) if isinstance(payload, dict) else {}
diagnostics = payload.get("match_diagnostics", {}) if isinstance(payload, dict) else {}

print(f"file size: {os.path.getsize(path) / (1024**2):,.1f} MB")
print(f"pickle.load time: {t1 - t0:,.2f} seconds")
print(f"matches shape: {getattr(matches, 'shape', None)}")
print(f"matches dtype: {getattr(matches, 'dtype', None)}")
print(f"matches memory: {getattr(matches, 'nbytes', 0) / (1024**2):,.1f} MB")

if isinstance(plot_cache, dict) and "count" in plot_cache:
    count = np.asarray(plot_cache["count"])
    print(f"plot_cache count shape: {count.shape}")
    print(f"occupied cells: {np.count_nonzero(count):,}")

if isinstance(diagnostics, dict):
    approx_diag_mb = 0.0
    for v in diagnostics.values():
        try:
            approx_diag_mb += np.asarray(v).nbytes / (1024**2)
        except Exception:
            pass
    print(f"diagnostic arrays memory: {approx_diag_mb:,.1f} MB")