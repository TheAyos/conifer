'''
Convert a scikit-learn BDT to an AI Engine project.

write() needs no toolchain. compile(), decision_function() and build() need the Vitis
AI Engine tools on PATH; the platform is found from the Vitis environment.

Run the toolchain stages with:  python sklearn_to_aie.py --build
'''

import sys
import logging
import numpy as np
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import conifer

# Importing conifer installs a root handler, which makes basicConfig a no-op, so the
# level has to be set directly or the backend's INFO lines never appear.
logging.getLogger().setLevel(logging.INFO)

X, y = make_hastie_10_2(n_samples=10000, random_state=0)
X_train, y_train, X_test = X[:9000], y[:9000], X[9000:]

clf = GradientBoostingClassifier(n_estimators=32, max_depth=4, random_state=0)
clf.fit(X_train, y_train)

# 'auto' asks the backend to choose the mapping. Set any handle explicitly to pin it.
cfg = conifer.backends.aie.auto_config()
cfg['OutputDir'] = 'prj_aie'
cfg['Priority'] = 'latency'      # or 'throughput'

model = conifer.converters.convert_from_sklearn(clf, cfg)
model.write()

print('\nResolved mapping (pass this back to reproduce the same project):')
for key, value in model.resolved_config().items():
    if key in ('priority', 'n_tiles', 'split_axis', 'vector_width', 'tau', 'n_samples',
               'shard', 'feed', 'plio_rate'):
        print(f'  {key:14s} {value}')

report = model.read_report()
print(f"\nStage: {report['stage']}")
print(f"  estimated {report['estimate']['est_cyc_per_sample']:.1f} cyc/sample, "
      f"{report['estimate']['est_latency_ss_ns']:.0f} ns latency_ss, "
      f"{report['estimate']['est_throughput_ns_per_sample']:.2f} ns/sample")
print(f"  next: {report['next_step']}")

if '--build' not in sys.argv:
    print('\nPass --build to run the toolchain stages.')
    sys.exit()

assert model.compile(), 'aiecompiler failed, see the log named above'

y_aie = model.decision_function(X_test[:256])
y_skl = clf.decision_function(X_test[:256])
print(f'\nScored {len(y_aie)} samples, '
      f'{np.mean(np.sign(y_aie.ravel()) == np.sign(y_skl)) * 100:.1f}% sign agreement '
      f'with scikit-learn at this precision')

assert model.build(), 'aiesimulator failed, see the log named above'
report = model.read_report()
print(f"\nStage: {report['stage']}")
print(f"  {report['cyc_per_sample']:.2f} cyc/sample on {report['n_active_cores']} tile(s)")
print(f"  latency_ss {report['latency_ss_ns']:.1f} ns "
      f"(drift {report['latency_ss_drift_ns_per_group']:.2f} ns/group)")
print(f"  slowest tile {report['slowest_tile_ratio']:.4f}x the average")
print(f"  tile memory {report['tile_memory_bytes_max']} B of 65536")
