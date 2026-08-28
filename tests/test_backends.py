'''
Test that the conifer backends yield the same output given the same model, data, and config
'''

import conifer
import pytest
import numpy as np
import datetime

def f_train_skl():
    # Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
    from sklearn.datasets import make_hastie_10_2
    from sklearn.ensemble import GradientBoostingClassifier

    # Make a random dataset from sklearn 'hastie'
    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    # Train a BDT
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,
        max_depth=3, random_state=0).fit(X_train, y_train)

    return clf, X_test, y_test.shape

def backend_predict(model, odir, X, y_shape, backend, backend_config):
  the_backend = conifer.backends.get_backend(backend)
  cfg = the_backend.auto_config()
  cfg.update(backend_config)
  cfg['OutputDir'] = odir
  cfg['ProjectName'] = "custom_project"
  model = conifer.converters.convert_from_sklearn(model, cfg)
  model.compile()
  y = model.decision_function(X).reshape(y_shape)
  return y

class Tester:
   def __init__(self, model, X, y_shape, backend_a, backend_b, config_a={}, config_b={}):
      self.model = model
      self.X = X
      self.y_shape = y_shape
      self.backend_a = backend_a
      self.backend_b = backend_b
      self.config_a = config_a
      self.config_b = config_b

model0 = f_train_skl()

# compare pairs of backends with different precision at different rounding modes
hls_cpp_precisions = ['ap_fixed<16,6>', 'ap_fixed<8,4,AP_TRN,AP_WRAP>', 'ap_fixed<8,4,AP_RND,AP_WRAP>', 'ap_fixed<8,4,AP_RND_ZERO,AP_WRAP>',
                      'ap_fixed<8,4,AP_RND,AP_SAT>', 'ap_fixed<18,8>', 'ap_fixed<18,8,AP_RND_CONV,AP_SAT>']
hls_hdl_precisions = ['ap_fixed<16,6>', 'ap_fixed<18,8>', 'ap_fixed<12,6>']
# compare configs with mixed precision
mixed_precision_cfg = {'InputPrecision' : 'ap_fixed<16,6>', 'ScorePrecision' : 'ap_fixed<12,5>'}

tests = [*[Tester(*model0, 'xilinxhls', 'cpp', {'Precision' : p}, {'Precision' : p}) for p in hls_cpp_precisions],
         *[Tester(*model0, 'xilinxhls', 'vhdl', {'Precision' : p}, {'Precision' : p}) for p in hls_hdl_precisions],
         Tester(*model0, 'xilinxhls', 'cpp', mixed_precision_cfg, mixed_precision_cfg),
         Tester(*model0, 'xilinxhls', 'vhdl', mixed_precision_cfg, mixed_precision_cfg),
         Tester(*model0, 'xilinxhls', 'xilinxhls', {'Unroll' : True}, {'Unroll' : False})]

@pytest.mark.parametrize('test', tests)
def test_backend_equality(test):
  stamp = int(datetime.datetime.now().timestamp())
  if test.backend_a == test.backend_b:
     name_a, name_b = test.backend_a + '_a', test.backend_a + '_b'
  else:
     name_a, name_b = test.backend_a, test.backend_b
  name_a, name_b = [f'prj_backends_{stamp}_{n}' for n in [name_a, name_b]]
  y_a = backend_predict(test.model, name_a, test.X, test.y_shape, test.backend_a, test.config_a)
  y_b = backend_predict(test.model, name_b, test.X, test.y_shape, test.backend_b, test.config_b)
  np.testing.assert_array_equal(y_a, y_b)
  assert((y_a is not None) and (y_b is not None)), "backend_predict returned None for the predictions: inconclusive test"

def test_py_backend():
   clf, X, _ = model0
   model = conifer.converters.convert_from_sklearn(clf)
   assert model.config.backend == 'python'
   y_skl = clf.decision_function(X)
   y_cnf = model.decision_function(X)
   np.testing.assert_allclose(y_skl, y_cnf, rtol=1e-6, atol=1e-6)

# AI Engine backend tests
# compile() runs aiecompiler and decision_function() runs x86simulator, so these fail
# without the Vitis AI Engine tools

# TODO: quantize() at precision.py add support for other types
aie_precision = {'Precision': 'ap_fixed<16,6,AP_RND_CONV,AP_SAT>',
                 'ScorePrecision': 'ap_fixed<32,16,AP_RND_CONV,AP_SAT>'}

# small, each configuration costs one aiecompiler run and one simulator run
aie_rows = 64

def f_train_ydf_oblique():
  ydf = pytest.importorskip('ydf')
  from sklearn.datasets import make_hastie_10_2
  X, y = make_hastie_10_2(random_state=0)
  model = ydf.GradientBoostedTreesLearner(
      num_trees=20, max_depth=3, split_axis='SPARSE_OBLIQUE',
      sparse_oblique_weights='BINARY',   # the only weights the kernel supports
      apply_link_function=False, label='y',
  ).train({'x': X[:2000], 'y': y[:2000] == 1}, verbose=0)
  return model, X[2000:2000 + aie_rows]

def aie_and_cpp(convert, model, X, odir, overrides):
  ys = []
  for backend in ('aie', 'cpp'):
    cfg = conifer.backends.get_backend(backend).auto_config()
    cfg.update(aie_precision)
    cfg['OutputDir'] = f'{odir}_{backend}'
    if backend == 'aie':
      cfg.update(overrides)
    m = convert(model, cfg)
    m.compile()
    ys.append(np.asarray(m.decision_function(X)).ravel())
  return ys

aie_axis_configs = [
    pytest.param({'NTiles': 1, 'SplitAxis': 'tree', 'NSamples': aie_rows}, id='one_tile'),
    pytest.param({'NTiles': 4, 'SplitAxis': 'tree', 'NSamples': aie_rows}, id='tree_split_4'),
    pytest.param({'NTiles': 4, 'SplitAxis': 'sample', 'NSamples': aie_rows}, id='sample_split_4'),
]

@pytest.mark.parametrize('overrides', aie_axis_configs)
def test_aie_axis_matches_cpp(overrides):
  clf, X, _ = model0
  stamp = int(datetime.datetime.now().timestamp())
  y_aie, y_cpp = aie_and_cpp(conifer.converters.convert_from_sklearn, clf, X[:aie_rows],
                             f'prj_aie_axis_{stamp}', overrides)
  np.testing.assert_array_equal(y_aie, y_cpp)

aie_oblique_configs = [
    pytest.param({'NTiles': 1, 'SplitAxis': 'tree', 'NSamples': aie_rows}, id='one_tile'),
    pytest.param({'NTiles': 4, 'SplitAxis': 'tree', 'NSamples': aie_rows}, id='tree_split_4'),
    pytest.param({'NTiles': 4, 'SplitAxis': 'sample', 'NSamples': aie_rows}, id='sample_split_4'),
]

@pytest.mark.parametrize('overrides', aie_oblique_configs)
def test_aie_oblique_matches_cpp(overrides):
  model, X = f_train_ydf_oblique()
  stamp = int(datetime.datetime.now().timestamp())
  y_aie, y_cpp = aie_and_cpp(conifer.converters.convert_from_ydf, model, X,
                             f'prj_aie_oblique_{stamp}', overrides)
  np.testing.assert_array_equal(y_aie, y_cpp)
