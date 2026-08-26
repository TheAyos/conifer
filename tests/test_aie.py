import os
import shutil
import numpy as np
import pytest
import conifer
from conifer.backends.aie import mapper
from conifer.backends.aie.precision import Precision

COMPARE = 'ap_fixed<16,5,AP_RND_CONV,AP_SAT>'
SCORE = 'ap_fixed<32,16,AP_RND_CONV,AP_SAT>'


def _config(tmp_path, **kwargs):
    cfg = conifer.backends.aie.auto_config()
    cfg['OutputDir'] = str(tmp_path)
    cfg.update(kwargs)
    return cfg


@pytest.fixture(scope='module')
def skl_model():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = make_classification(n_samples=600, n_features=16, n_informative=10,
                               random_state=0)
    clf = GradientBoostingClassifier(n_estimators=32, max_depth=4,
                                     random_state=0).fit(X[:500], y[:500])
    return clf, X[500:]


@pytest.fixture(scope='module')
def ydf_model():
    ydf = pytest.importorskip('ydf')
    import pandas as pd
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=800, n_features=16, n_informative=12,
                               random_state=1)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(16)])
    df['label'] = y
    model = ydf.GradientBoostedTreesLearner(
        label='label', task=ydf.Task.CLASSIFICATION, num_trees=16, max_depth=5,
        split_axis='SPARSE_OBLIQUE', sparse_oblique_weights='BINARY',
        sparse_oblique_normalization='NONE', early_stopping='NONE').train(df, verbose=0)
    return model, X[700:]


def _oblique_config(tmp_path):
    return _config(tmp_path,
                   Precision='ap_fixed<16,6,AP_RND_CONV,AP_SAT>',
                   InputPrecision='ap_fixed<16,6,AP_RND_CONV,AP_SAT>',
                   ThresholdPrecision='ap_fixed<16,6,AP_RND_CONV,AP_SAT>',
                   WeightPrecision='ap_fixed<16,3,AP_RND_CONV,AP_SAT>')


def test_backend_is_registered():
    assert 'aie' in conifer.backends.get_available_backends()


def test_write_emits_a_project(skl_model, tmp_path):
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path))
    model.write()
    for f in ['src/parameters.h', 'src/graph.hpp', 'src/tb.cpp', 'src/bdt_vec.hpp',
              'Makefile', 'aie_model.json']:
        assert os.path.exists(tmp_path / f), f'missing {f}'
    assert model.family == 'axis'


def test_oblique_selects_the_oblique_family(ydf_model, tmp_path):
    ymodel, _ = ydf_model
    model = conifer.converters.convert_from_ydf(ymodel, _oblique_config(tmp_path))
    model.write()
    assert model.family == 'oblique'
    assert model.n_tiles == 1
    assert os.path.exists(tmp_path / 'src/bdt_qs_oblique.hpp')


def test_tables_reproduce_the_ensemble(skl_model, tmp_path):
    '''The emitted tables must score what conifer scores, up to quantization'''
    clf, X = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path))
    reference = conifer.converters.convert_from_sklearn(clf)
    ref = np.asarray(reference.decision_function(X)).ravel()
    got = model.score_p.dequantize(
        model.tables.replay(X, model.threshold_p, model.score_p, model.init_predict[0],
                            norm=model.norm,
                            split_le=(model.splitting_convention == '<=')))
    assert np.sum(np.sign(ref) == np.sign(got)) == len(ref)
    assert np.max(np.abs(ref - got)) < 0.01


def test_auto_config_round_trips(skl_model, tmp_path):
    '''Resolving auto and passing the result back must give the same project'''
    clf, _ = skl_model
    a = conifer.converters.convert_from_sklearn(clf, _config(tmp_path / 'a'))
    a.write()
    resolved = a.resolved_config()
    assert 'auto' not in [str(v) for v in resolved.values()]

    cfg = _config(tmp_path / 'b')
    cfg.update({k: resolved[k] for k in
                ('priority', 'n_tiles', 'split_axis', 'vector_width', 'tau', 'n_samples')})
    b = conifer.converters.convert_from_sklearn(clf, cfg)
    b.write()
    for f in ['src/parameters.h', 'aie_model.json']:
        pa = open(tmp_path / 'a' / f).read().replace(str(tmp_path / 'a'), '')
        pb = open(tmp_path / 'b' / f).read().replace(str(tmp_path / 'b'), '')
        assert pa == pb, f'{f} differs between auto and resolved'


def test_report_reports_the_write_stage(skl_model, tmp_path):
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path))
    model.write()
    report = model.read_report()
    assert report['stage'] == 'write'
    assert report['next_step'] is not None
    assert report['estimate']['est_latency_ss_ns'] > 0


# ----- guards -----

def test_oblique_weights_must_be_binary(ydf_model, tmp_path):
    ymodel, _ = ydf_model
    model = conifer.converters.convert_from_ydf(ymodel, _oblique_config(tmp_path / 'ok'))
    d = {k: getattr(model, k) for k in conifer.model.ModelBase._ensemble_fields}
    d['trees'] = [[{k: getattr(t, k) for k in conifer.model.DecisionTreeBase._tree_fields}
                   for t in tc] for tc in model.trees]
    w = np.asarray(d['trees'][0][0]['weight'], dtype=float)
    row = next(i for i in range(len(w)) if np.count_nonzero(w[i]) > 1)
    w[row][np.nonzero(w[row])[0][0]] = 0.5
    d['trees'][0][0]['weight'] = w.tolist()
    with pytest.raises(NotImplementedError, match='binary'):
        conifer.model.make_model(d, _oblique_config(tmp_path / 'bad'))


def test_multiclass_is_rejected(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = make_classification(n_samples=300, n_features=8, n_informative=6,
                               n_classes=3, random_state=0)
    clf = GradientBoostingClassifier(n_estimators=4, max_depth=3, random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError, match='one value per sample'):
        conifer.converters.convert_from_sklearn(clf, _config(tmp_path))


def test_too_many_tiles_for_the_device(skl_model, tmp_path):
    clf, _ = skl_model
    with pytest.raises(ValueError, match='template limit|AI Engine tiles'):
        conifer.converters.convert_from_sklearn(clf, _config(tmp_path, NTiles=1000))


def test_oblique_rejects_multiple_tiles(ydf_model, tmp_path):
    ymodel, _ = ydf_model
    cfg = _oblique_config(tmp_path)
    cfg['NTiles'] = 4
    with pytest.raises(ValueError, match='multi-tile oblique'):
        conifer.converters.convert_from_ydf(ymodel, cfg)


@pytest.mark.parametrize('precision,match', [
    ('ap_fixed<18,8,AP_RND_CONV,AP_SAT>', 'width'),
    ('ap_fixed<16,5>', 'AP_RND_CONV'),
    ('ap_fixed<16,5,AP_TRN,AP_WRAP>', 'AP_RND_CONV'),
    ('float', 'ap_fixed'),
])
def test_unsupported_precisions_are_rejected(skl_model, tmp_path, precision, match):
    clf, _ = skl_model
    cfg = _config(tmp_path, Precision=precision, InputPrecision=precision,
                  ThresholdPrecision=precision, WeightPrecision=precision)
    with pytest.raises(ValueError, match=match):
        conifer.converters.convert_from_sklearn(clf, cfg)


def test_unknown_device_lists_the_known_ones():
    from conifer.backends.aie.devices import get_device_config
    with pytest.raises(ValueError, match='Known devices'):
        get_device_config('xcvu9p-flgb2104-2L-e')


def test_deep_model_is_rejected(tmp_path):
    from conifer.backends.aie import checks
    with pytest.raises(ValueError, match='max_depth'):
        checks.check_max_depth(7)


def test_score_range_warning_does_not_raise(caplog):
    from conifer.backends.aie import checks
    checks.warn_score_range(10000, 5.0, Precision(SCORE))
    assert any('saturate' in r.message for r in caplog.records)


# ----- mapper -----

def test_cost_model_reproduces_the_measured_anchors():
    '''Measured on VEK280 at t32-d4-f16 / int16 / W=32'''
    assert mapper.invocation_cycles(16, 4, 32, 32, 2) == pytest.approx(5354, abs=1)
    assert mapper.invocation_cycles(16, 4, 8, 32, 2) == pytest.approx(1610, abs=1)


def test_priority_picks_opposite_split_axes():
    lat = mapper.estimate(32, 4, 16, 8, 32, 2, 4, 'latency')
    thr = mapper.estimate(32, 4, 16, 8, 32, 2, 4, 'throughput')
    assert lat['split_axis'] == 'tree'
    assert thr['split_axis'] == 'sample'
    assert thr['est_throughput_ns_per_sample'] < lat['est_throughput_ns_per_sample']


def test_auto_tiles_explains_where_it_stopped():
    n, notes = mapper.choose_n_tiles(32, 4, 16, 32, 2, 4, 'latency', 304, 0x10000)
    assert 1 <= n <= mapper.MAX_TEMPLATE_TILES
    assert any('n_tiles' in s for s in notes)


def test_estimate_declares_its_extrapolations():
    e = mapper.estimate(32, 6, 16, 1, 32, 2, 4, 'latency', oblique=True)
    assert e['validity'], 'a depth-6 oblique estimate must say what it extrapolates'


def test_a_huge_ensemble_does_not_fit_one_tile():
    mem = mapper.table_bytes(10000, 4, 16, 16, 32, 2, 4, False)
    assert mem['total'] > 0x10000
