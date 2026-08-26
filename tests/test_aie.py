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


# ----- sharding -----

def _sharded(skl_model, tmp_path, n_tiles=8, **kw):
    clf, _ = skl_model
    kw.setdefault('SplitAxis', 'tree')
    return conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=n_tiles, **kw))


def test_tree_split_shards_and_feeds_from_a_memtile(skl_model, tmp_path):
    model = _sharded(skl_model, tmp_path)
    assert model.sharding is not None
    assert model.feed_memtile
    assert model.n_memtiles == 1
    model.write()
    header = open(tmp_path / 'src/parameters.h').read()
    assert '#define BDT_SHARDED 1' in header
    assert '#define BDT_FEED_MEMTILE 1' in header
    for sym in ['N_SHARDS', 'WINDOWED', 'T_BEGIN', 'T_COUNT', 'N_FEAT', 'OFFSET']:
        assert sym in header, f'missing bdtsh::{sym}'


@pytest.mark.parametrize('n_tiles', [2, 4, 8, 16])
def test_sharded_tables_score_exactly_what_unsharded_ones_do(skl_model, tmp_path, n_tiles):
    '''The partial scores the tiles sum must equal the whole-ensemble score'''
    model = _sharded(skl_model, tmp_path, n_tiles=n_tiles)
    X = np.random.default_rng(0).uniform(-8, 8, size=(48, model.n_features))
    bad = model.sharding.verify(X, model.threshold_p, model.score_p,
                                model.init_predict[0], norm=model.norm,
                                split_le=(model.splitting_convention == '<='))
    assert len(bad) == 0


def test_sharding_reduces_the_rows_a_tile_reads(skl_model, tmp_path):
    from conifer.backends.aie.shard import Sharding
    model = _sharded(skl_model, tmp_path, n_tiles=16)
    identity = Sharding(model.tables, model.n_trees_padded, model.n_features_padded,
                        16, fperm=list(range(model.n_features_padded)), optimize=False)
    assert model.sharding.total_rows < identity.total_rows


def test_sample_split_is_not_sharded(skl_model, tmp_path):
    '''A sample-split tile holds the whole ensemble, so it reads every row'''
    model = _sharded(skl_model, tmp_path, SplitAxis='sample', Priority='throughput')
    assert model.sharding is None
    assert not model.feed_memtile


def test_sharding_can_be_turned_off(skl_model, tmp_path):
    model = _sharded(skl_model, tmp_path, Shard=False)
    assert model.sharding is None
    model.write()
    assert '#define BDT_SHARDED 0' in open(tmp_path / 'src/parameters.h').read()


def test_memtile_can_be_declined_while_sharding(skl_model, tmp_path):
    model = _sharded(skl_model, tmp_path, Feed='plio')
    assert model.sharding is not None
    assert not model.feed_memtile


def test_tree_split_sums_the_per_tile_partial_scores(skl_model, tmp_path):
    '''Each tile writes its own partial; read_scores must add them up'''
    import os
    model = _sharded(skl_model, tmp_path, n_tiles=4)
    model.write()
    assert model.n_outputs == 4
    d = tmp_path / 'data'
    os.makedirs(d, exist_ok=True)
    n = model.n_samples
    parts = [np.arange(n) * (i + 1) for i in range(4)]
    for i, p in enumerate(parts):
        name = 'scores.dat' if i == 0 else f'scores.t{i}.dat'
        with open(d / name, 'w') as f:
            for v in p:
                f.write(f'T 100 ns\n{int(v)}\n')
    got = model.read_scores()
    np.testing.assert_allclose(got, model.score_p.dequantize(np.sum(parts, axis=0)))


# ----- platform discovery -----

def test_platform_is_found_from_the_vitis_environment(tmp_path, monkeypatch):
    '''settings64.sh sets XILINX_VITIS but not PLATFORM_REPO_PATHS'''
    from conifer.backends.aie import platforms
    root = tmp_path / 'Vitis'
    d = root / 'base_platforms' / 'xilinx_vek280_base_202610_1'
    os.makedirs(d)
    open(d / 'xilinx_vek280_base_202610_1.xpfm', 'w').close()
    monkeypatch.setenv('XILINX_VITIS', str(root))
    monkeypatch.delenv('PLATFORM_REPO_PATHS', raising=False)
    assert platforms.find_platform('vek280_base') == str(
        d / 'xilinx_vek280_base_202610_1.xpfm')


def test_platform_prefers_the_exactly_named_directory(tmp_path, monkeypatch):
    from conifer.backends.aie import platforms
    base = tmp_path / 'Vitis' / 'base_platforms'
    for name in ('vek280_base', 'xilinx_vek280_base_202610_1'):
        os.makedirs(base / name)
        open(base / name / f'{name}.xpfm', 'w').close()
    monkeypatch.setenv('XILINX_VITIS', str(tmp_path / 'Vitis'))
    monkeypatch.delenv('PLATFORM_REPO_PATHS', raising=False)
    assert platforms.find_platform('vek280_base').endswith('vek280_base/vek280_base.xpfm')


def test_missing_platform_says_where_it_looked(monkeypatch):
    from conifer.backends.aie import platforms
    monkeypatch.delenv('PLATFORM_REPO_PATHS', raising=False)
    monkeypatch.delenv('XILINX_VITIS', raising=False)
    monkeypatch.delenv('XILINX_HLS', raising=False)
    with pytest.raises(RuntimeError, match='settings64|PLATFORM_REPO_PATHS'):
        platforms.resolve_platform('vek280_base')


def test_scores_are_read_from_the_simulator_output_directory(skl_model, tmp_path):
    '''x86simulator writes into <build>/x86simulator_output, not a data/ below it'''
    model = _sharded(skl_model, tmp_path, n_tiles=2)
    model.write()
    d = tmp_path / 'build_x86' / 'x86simulator_output'
    os.makedirs(d)
    for i in range(2):
        name = 'scores.dat' if i == 0 else f'scores.t{i}.dat'
        with open(d / name, 'w') as f:
            for v in range(model.n_samples):
                f.write(f'{v * (i + 1)}\n')
    got = model.read_scores()
    assert len(got) == model.n_samples


# ----- the priority knob -----

def test_each_priority_wins_its_own_metric(skl_model, tmp_path):
    '''A throughput mapping must not be beaten on throughput by a latency mapping'''
    clf, _ = skl_model
    got = {}
    for priority in ('latency', 'throughput'):
        model = conifer.converters.convert_from_sklearn(
            clf, _config(tmp_path / priority, Priority=priority))
        got[priority] = model.estimate
    assert got['latency']['est_latency_ss_ns'] < got['throughput']['est_latency_ss_ns']
    assert (got['throughput']['est_throughput_ns_per_sample']
            < got['latency']['est_throughput_ns_per_sample'])


def test_both_priorities_compile_the_same_batch(skl_model, tmp_path):
    '''n_samples is a batch size, so two mappings of one model stay comparable'''
    clf, _ = skl_model
    n = {p: conifer.converters.convert_from_sklearn(
             clf, _config(tmp_path / p, Priority=p)).n_samples
         for p in ('latency', 'throughput')}
    assert n['latency'] == n['throughput']


def test_auto_uses_one_tile_ceiling_for_both_priorities(skl_model, tmp_path):
    clf, _ = skl_model
    n = {p: conifer.converters.convert_from_sklearn(
             clf, _config(tmp_path / p, Priority=p)).n_tiles
         for p in ('latency', 'throughput')}
    assert n['latency'] == n['throughput']


def test_auto_picks_a_width_the_study_measured():
    '''The cost model prices wider vectors but nothing on this device has measured them'''
    for priority in ('latency', 'throughput'):
        _, W, _ = mapper.choose_mapping(32, 4, 16, 2, 4, priority, 304, 0x10000)
        assert W in mapper.AUTO_VECTOR_WIDTHS


def test_latency_prefers_a_narrower_vector_than_throughput():
    '''A narrower vector is a smaller invocation; a wider one amortises it over more'''
    _, w_lat, _ = mapper.choose_mapping(32, 4, 16, 2, 4, 'latency', 304, 0x10000)
    _, w_thr, _ = mapper.choose_mapping(32, 4, 16, 2, 4, 'throughput', 304, 0x10000)
    assert w_lat < w_thr


def test_an_explicit_width_is_honoured(skl_model, tmp_path):
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, VectorWidth=32))
    assert model.W == 32


@pytest.mark.parametrize('depth,W,measured', [
    (4, 32, 518.0), (4, 16, 345.7),
    (5, 32, 790.4), (5, 16, 500.4),
    (6, 32, 1533.5), (6, 16, 981.9), (6, 8, 891.4),
])
def test_cost_model_tracks_the_measured_width_sweep(depth, W, measured):
    '''One tree per tile, 16 features, int16 - the shape the widths were swept at'''
    got = mapper.invocation_cycles(16, depth, 1, W, 2)
    assert abs(got - measured) / measured < 0.10


def test_an_unswept_width_says_so(skl_model, tmp_path):
    '''W=8 at depth 4 is priced by mechanism, not measured; the estimate must admit it'''
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path, VectorWidth=8))
    assert any('not swept' in v for v in model.estimate['validity'])


def test_a_swept_width_carries_no_caveat(skl_model, tmp_path):
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path, VectorWidth=32))
    assert not any('not swept' in v for v in model.estimate['validity'])
