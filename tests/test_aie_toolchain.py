'''AI Engine tests that need the Vitis toolchain. Skipped when it is not on PATH.'''

import shutil
import numpy as np
import pytest
import conifer

pytestmark = pytest.mark.skipif(shutil.which('aiecompiler') is None,
                                reason='aiecompiler not on PATH')

COMPARE = 'ap_fixed<16,5,AP_RND_CONV,AP_SAT>'
SCORE = 'ap_fixed<32,16,AP_RND_CONV,AP_SAT>'


def _aie_config(tmp_path, **kwargs):
    cfg = conifer.backends.aie.auto_config()
    cfg['OutputDir'] = str(tmp_path)
    cfg.update(kwargs)
    return cfg


def _cpp_reference(model, tmp_path, precisions):
    '''The same ensemble on conifer's bit-exact cpp backend'''
    d = {k: getattr(model, k) for k in conifer.model.ModelBase._ensemble_fields}
    d['trees'] = [[{k: getattr(t, k) for k in conifer.model.DecisionTreeBase._tree_fields}
                   for t in tc] for tc in model.trees]
    cfg = {'Backend': 'cpp', 'ProjectName': 'golden', 'OutputDir': str(tmp_path)}
    cfg.update(precisions)
    ref = conifer.model.make_model(d, cfg)
    ref.compile()
    return ref


def test_axis_aligned_x86sim_matches_the_cpp_golden(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = make_classification(n_samples=600, n_features=16, n_informative=10,
                               random_state=0)
    clf = GradientBoostingClassifier(n_estimators=32, max_depth=4,
                                     random_state=0).fit(X[:500], y[:500])
    precisions = {'Precision': COMPARE, 'InputPrecision': COMPARE,
                  'ThresholdPrecision': COMPARE, 'WeightPrecision': COMPARE,
                  'ScorePrecision': SCORE}

    model = conifer.converters.convert_from_sklearn(
        clf, _aie_config(tmp_path / 'aie', **precisions))
    assert model.compile()
    y_aie = model.decision_function(X[500:756])

    ref = _cpp_reference(model, tmp_path / 'cpp', precisions)
    y_ref = np.asarray(ref.decision_function(X[500:756])).ravel()
    np.testing.assert_allclose(y_aie, y_ref, atol=0, rtol=0)


def test_oblique_x86sim_matches_the_cpp_golden(tmp_path):
    ydf = pytest.importorskip('ydf')
    import pandas as pd
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=800, n_features=16, n_informative=12,
                               random_state=1)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(16)])
    df['label'] = y
    ymodel = ydf.GradientBoostedTreesLearner(
        label='label', task=ydf.Task.CLASSIFICATION, num_trees=16, max_depth=5,
        split_axis='SPARSE_OBLIQUE', sparse_oblique_weights='BINARY',
        sparse_oblique_normalization='NONE', early_stopping='NONE').train(df, verbose=0)

    compare = 'ap_fixed<16,6,AP_RND_CONV,AP_SAT>'
    precisions = {'Precision': compare, 'InputPrecision': compare,
                  'ThresholdPrecision': compare,
                  'WeightPrecision': 'ap_fixed<16,3,AP_RND_CONV,AP_SAT>',
                  'ScorePrecision': SCORE}

    model = conifer.converters.convert_from_ydf(
        ymodel, _aie_config(tmp_path / 'aie', **precisions))
    assert model.compile()
    y_aie = model.decision_function(X[700:756])

    ref = _cpp_reference(model, tmp_path / 'cpp', precisions)
    y_ref = np.asarray(ref.decision_function(X[700:756])).ravel()
    np.testing.assert_allclose(y_aie, y_ref, atol=0, rtol=0)


def test_compile_reports_real_tile_memory(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = make_classification(n_samples=400, n_features=16, n_informative=10,
                               random_state=0)
    clf = GradientBoostingClassifier(n_estimators=32, max_depth=4,
                                     random_state=0).fit(X, y)
    model = conifer.converters.convert_from_sklearn(clf, _aie_config(tmp_path))
    assert model.compile()
    report = model.read_report()
    assert report['stage'] == 'compile'
    assert report['tile_memory_bytes_max'] > 0
