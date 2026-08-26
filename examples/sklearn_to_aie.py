'''
Convert a scikit-learn BDT to an AI Engine project.

write() needs no toolchain. compile(), decision_function() and build() need the Vitis
AI Engine tools on PATH and PLATFORM_REPO_PATHS set.
'''

import logging
import numpy as np
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import conifer

logging.basicConfig(level=logging.INFO)

X, y = make_hastie_10_2(n_samples=10000, random_state=0)
X_train, y_train, X_test = X[:9000], y[:9000], X[9000:]

clf = GradientBoostingClassifier(n_estimators=32, max_depth=4, random_state=0)
clf.fit(X_train, y_train)

# 'auto' asks the backend to choose the mapping. Set any handle explicitly to pin it.
cfg = conifer.backends.aie.auto_config()
cfg['OutputDir'] = 'prj_aie'
cfg['Priority'] = 'throughput'      # or 'throughput'

model = conifer.converters.convert_from_sklearn(clf, cfg)
model.write()

print('\nResolved mapping:')
print()
print()
print()
print(model.resolved_config())
for key in ['priority', 'n_tiles', 'split_axis', 'vector_width', 'tau', 'n_samples']:
    print(f'  {key:14s} {model.resolved_config()[key]}')

report = model.read_report()
print(f"\nStage: {report['stage']}")
print(f"  estimated {report['estimate']['est_cyc_per_sample']:.1f} cyc/sample")
print(f"  estimated latency_ss {report['estimate']['est_latency_ss_ns']:.0f} ns")
print(f"  estimated throughput {report['estimate']['est_throughput_ns_per_sample']:.2f} ns/sample")
if report['next_step']:
    print(f"  {report['next_step']}")

# # With the toolchain available:
# model.compile()
# print("Report after .compile()")
# print(model.read_report())
# y_aie = model.decision_function(X_test[:256])
# model.build()
# print("Report after .build()")
# print(model.read_report())
