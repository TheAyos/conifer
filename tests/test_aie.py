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
    with pytest.raises(ValueError, match='AI Engine tiles'):
        conifer.converters.convert_from_sklearn(clf, _config(tmp_path, NTiles=1000))


def test_more_tiles_than_the_shim_can_route(skl_model, tmp_path):
    """Every tile emits its partial on its own outgoing PLIO channel, so the shim runs
    out long before the cores do. aiecompiler only says so at -target=hw, minutes in.
    """
    clf, _ = skl_model
    with pytest.raises(ValueError, match='outgoing PLIO channels'):
        conifer.converters.convert_from_sklearn(clf, _config(tmp_path, NTiles=128))


def test_oblique_splits_across_tiles(ydf_model, tmp_path):
    """The tree range is a template parameter, so a tile scores its own shard and the
    ladder names one symbol per tile -- the same mechanism the axis-aligned fork uses.
    """
    ymodel, _ = ydf_model
    cfg = _oblique_config(tmp_path)
    cfg['NTiles'] = 4
    cfg['SplitAxis'] = 'tree'
    model = conifer.converters.convert_from_ydf(ymodel, cfg)
    model.write()
    assert model.n_tiles == 4 and model.n_trees_padded % 4 == 0
    ladder = (tmp_path / 'src/tile_roles.h').read_text()
    assert 'bdt_qs_tile_3' in ladder


def test_oblique_reads_a_partial_from_every_tile(ydf_model, tmp_path):
    """Every tile emits its own partial on its own port, oblique included, so the reader
    has to know there are n_tiles of them -- one output would silently score a fraction
    of the ensemble.
    """
    ymodel, _ = ydf_model
    cfg = _oblique_config(tmp_path)
    cfg['NTiles'] = 4
    cfg['SplitAxis'] = 'tree'
    model = conifer.converters.convert_from_ydf(ymodel, cfg)
    model.write()
    assert model.n_outputs == 4


def test_oblique_never_shards_or_takes_the_memtile(ydf_model, tmp_path):
    """An oblique node has a dense weight row and a basis over the global feature set,
    so there is no per-shard feature frame a memtile could hand a tile.
    """
    ymodel, _ = ydf_model
    cfg = _oblique_config(tmp_path)
    cfg['NTiles'] = 4
    cfg['SplitAxis'] = 'tree'
    cfg['Feed'] = 'memtile'
    model = conifer.converters.convert_from_ydf(ymodel, cfg)
    model.write()
    assert model.sharding is None and model.feed_memtile is False


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
    n, notes = mapper.choose_n_tiles(32, 4, 16, 32, 2, 4, 'latency', 112, 0x10000)
    assert 1 <= n <= 112
    assert any('n_tiles' in s for s in notes)


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


def test_declining_the_memtile_also_declines_sharding(skl_model, tmp_path):
    """Only a memtile can hand each tile its own rows; a PLIO multicasts one stream"""
    model = _sharded(skl_model, tmp_path, Feed='plio')
    assert model.sharding is None
    assert not model.feed_memtile
    model.write()
    assert '#define BDT_SHARDED 0' in open(tmp_path / 'src/parameters.h').read()


def test_plio_rate_defaults_to_the_device_and_is_capped(skl_model, tmp_path):
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path))
    assert model.plio_rate == model.device['plio_rate_mhz']
    with pytest.raises(ValueError, match='exceeds'):
        conifer.converters.convert_from_sklearn(
            clf, _config(tmp_path / 'x', PlioRate=99999))


def test_wide_feature_models_are_accepted(tmp_path):
    """The 64-feature bound was the feature-mask experiment, which no longer ships"""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = make_classification(n_samples=400, n_features=128, n_informative=20,
                               random_state=0)
    clf = GradientBoostingClassifier(n_estimators=8, max_depth=3, random_state=0).fit(X, y)
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=4, SplitAxis='tree'))
    model.write()
    assert model.n_features == 128


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
    # EVERY root the search reads, not just the two the docstring names. `_roots()` also
    # consults XILINX_HLS, so on a machine that has actually sourced settings64.sh this
    # test found the REAL platform and failed -- a test that only isolates part of the
    # environment is a test that passes on the machine it was written on.
    monkeypatch.delenv('PLATFORM_REPO_PATHS', raising=False)
    monkeypatch.delenv('XILINX_HLS', raising=False)
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


# ----- latency_ss is a fit, not a mean -----

def _drifting(n_groups, base=300.0, drift=6.0):
    return [base + drift * i for i in range(n_groups)]


def test_latency_intercept_does_not_move_with_run_length():
    """A pipelined mapping drifts, so the mean is a function of how long the run was"""
    from conifer.backends.aie.report import _summarise_latency
    got = []
    for n in (8, 16, 32, 64):
        r = {}
        _summarise_latency(_drifting(n), r)
        got.append(r['latency_ss_ns'])
    assert max(got) - min(got) < 1e-6, f'intercept moved with run length: {got}'
    means = [np.mean(_drifting(n)) for n in (8, 16, 32, 64)]
    assert max(means) - min(means) > 100, 'the mean should move, or this proves nothing'


def test_latency_reports_the_drift_beside_the_intercept():
    from conifer.backends.aie.report import _summarise_latency
    r = {}
    _summarise_latency(_drifting(32, base=300.0, drift=6.4), r)
    assert r['latency_ss_ns'] == pytest.approx(300.0)
    assert r['latency_ss_drift_ns_per_group'] == pytest.approx(6.4)


def test_a_trimmed_window_still_reports_group_zero():
    """Trimming the fill must not report the window's own accumulated skew"""
    from conifer.backends.aie.report import _summarise_latency
    full, trimmed = {}, {}
    lat = _drifting(32)
    _summarise_latency(lat, full)
    _summarise_latency(lat[4:28], trimmed, offset=4)
    assert trimmed['latency_ss_ns'] == pytest.approx(full['latency_ss_ns'])


def test_too_few_groups_is_declined_not_guessed():
    from conifer.backends.aie.report import _summarise_latency
    r = {}
    _summarise_latency([300.0, 306.0], r)
    assert 'latency_ss_ns' not in r
    assert 'unmeasured' in r['latency_ss_note']


# ----- n_samples is a batch, and any X length must work -----

@pytest.mark.parametrize('asked', [1, 7, 100, 333, 512])
def test_any_n_samples_is_accepted(skl_model, tmp_path, asked):
    """A run is a whole number of groups, so round up rather than refuse"""
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NSamples=asked))
    assert model.n_samples >= asked
    assert model.n_samples % model.batch_step == 0
    assert model.n_samples - asked < model.batch_step


def test_n_samples_is_left_alone_when_it_already_fits(skl_model, tmp_path):
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path, NSamples=8))
    step = model.batch_step
    exact = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path / 'b', NSamples=step * 3))
    assert exact.n_samples == step * 3


def test_a_short_batch_is_padded_not_refused(skl_model, tmp_path):
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path, NSamples=64))
    model.write()
    model.write_input(np.zeros((5, model.n_features)))
    lines = sum(1 for _ in open(tmp_path / 'data/x.dat'))
    assert lines * 4 == model.n_samples * model.n_features_padded


def test_an_oversized_batch_is_refused_by_write_input(skl_model, tmp_path):
    """decision_function splits a long X into runs; write_input takes one run"""
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path, NSamples=64))
    model.write()
    with pytest.raises(ValueError, match='exceeds'):
        model.write_input(np.zeros((model.n_samples + 1, model.n_features)))


# ----- decision_function must return one score per row, whatever the batch -----

def _stub_simulator(model, monkeypatch):
    """Stand in for x86simulator: score whatever write_input last wrote.

    Exercises the real path - write_input, the per-tile merge, read_scores and the
    chunking - without a toolchain, which is where the truncation bug lived.
    """
    from conifer.backends.aie import tools
    state = {}
    real_write = model.write_input

    def write_input(X):
        state['X'] = np.asarray(X, dtype=np.float64)
        real_write(X)

    def run_make(output_dir, target, **kwargs):
        X = state['X']
        if len(X) < model.n_samples:
            X = np.vstack([X, np.zeros((model.n_samples - len(X), X.shape[1]))])
        q = model.tables.replay(X, model.threshold_p, model.score_p,
                                model.init_predict[0], norm=model.norm,
                                split_le=(model.splitting_convention == '<='))
        d = os.path.join(output_dir, 'build_x86', 'x86simulator_output')
        os.makedirs(d, exist_ok=True)
        # Tree-split partials sum to the score; put the whole of it on tile 0.
        for i in range(model.n_outputs):
            vals = q if i == 0 else np.zeros_like(q)
            name = 'scores.dat' if i == 0 else f'scores.t{i}.dat'
            with open(os.path.join(d, name), 'w') as f:
                for v in vals:
                    f.write(f'{int(v)}\n')
        return True

    model.write_input = write_input
    model.platform = lambda: '/stub/platform.xpfm'
    monkeypatch.setattr(tools, 'run_make', run_make)


@pytest.mark.parametrize('n_rows', [1, 5, 32, 33, 64, 100])
def test_decision_function_returns_one_score_per_row(skl_model, tmp_path, monkeypatch,
                                                     n_rows):
    """A long X is split across runs; a short one is padded. Either way, len(X) scores"""
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NSamples=32, NTiles=2, SplitAxis='tree'))
    model.write()
    _stub_simulator(model, monkeypatch)

    X = np.random.default_rng(0).uniform(-4, 4, size=(n_rows, model.n_features))
    y = model.decision_function(X)
    assert len(y) == n_rows, f'asked for {n_rows} scores, got {len(y)}'


def test_decision_function_matches_a_reference_backend_exactly(skl_model, tmp_path,
                                                               monkeypatch):
    """Every returned score must equal the reference, not just the count"""
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NSamples=32, NTiles=2, SplitAxis='tree'))
    model.write()
    _stub_simulator(model, monkeypatch)

    X = np.random.default_rng(1).uniform(-4, 4, size=(70, model.n_features))
    y = model.decision_function(X)
    reference = model.score_p.dequantize(
        model.tables.replay(X, model.threshold_p, model.score_p, model.init_predict[0],
                            norm=model.norm,
                            split_le=(model.splitting_convention == '<=')))
    assert len(y) == len(X)
    np.testing.assert_array_equal(y, reference)


# ----- the per-tile role ladder -----

def test_the_role_ladder_is_generated_and_has_no_ceiling_of_its_own(skl_model, tmp_path):
    """64 tiles was how far somebody had written `#if` out by hand, not a device limit.

    A tile's tree range is baked into its symbol, so the enumeration is unavoidable and
    `kernel::create` needs a literal name. Generating it is what makes the ceiling the
    device's core count instead.
    """
    from conifer.backends.aie import roles
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=100, Priority='latency'))
    model.write()
    text = open(tmp_path / 'src' / 'tile_roles.h').read()
    assert model.n_tiles == 100
    assert 'bdt_qs_tile_99' in text
    for guard in ('BDT_LADDER_DECL', 'BDT_LADDER_DEF', 'BDT_LADDER_CREATE'):
        assert f'#ifdef {guard}' in text
    # No include guard: the file is included once per section.
    assert '#pragma once' not in text


def test_the_ladder_is_empty_where_one_symbol_serves_every_tile():
    """Sample-split and N=1 run one symbol on every tile; the graph loops instead."""
    from conifer.backends.aie import roles
    for axis, n in (('sample', 8), ('tree', 1)):
        decl, defn, create = roles.ladder(n, axis, 'plio')
        assert (decl, defn, create) == ([], [], [])


def test_the_ladder_marks_tile_zero_and_only_tile_zero():
    """Tile 0 carries the ensemble's base score, so its definition differs from its
    neighbours'. Its DECLARATION does not, and one name in both sections is what lets
    the generator emit a single list."""
    from conifer.backends.aie import roles
    decl, defn, _ = roles.ladder(4, 'tree', 'plio')
    assert decl[0] == 'BDT_DECL_ROLE0(0)' and defn[0] == 'BDT_DEF_ROLE0(0)'
    assert all('ROLE0' not in d for d in decl[1:] + defn[1:])


def _capture_logger(monkeypatch, module):
    """Record what a module logs, without emitting records the no-error fixture sees"""
    import types
    lines = []
    monkeypatch.setattr(module, 'logger', types.SimpleNamespace(
        info=lambda m, *a: lines.append(('info', str(m))),
        debug=lambda m, *a: lines.append(('debug', str(m))),
        warning=lambda m, *a: lines.append(('warning', str(m))),
        error=lambda m, *a: lines.append(('error', str(m)))))
    return lines


def test_a_failed_run_names_its_log_and_the_first_error(tmp_path, monkeypatch):
    """A tool that fails leaves hundreds of lines behind; the point of capturing them is
    to be told where they are and what the first one said.
    """
    from conifer.backends.aie import tools
    log = tmp_path / 'x86sim_build.log'
    monkeypatch.setattr(tools, 'require_tools', lambda *a: None)

    def fail(cmd):
        log.write_text('INFO: compiling\n'
                       '../src/params.h:99:15: error: static assertion failed\n'
                       'ERROR: [aiecompiler 77-753] cannot recover\n')
        return 512

    monkeypatch.setattr(tools.os, 'system', fail)
    lines = _capture_logger(monkeypatch, tools)

    assert tools.run_make(str(tmp_path), 'x86sim_build') is False
    errors = [m for lvl, m in lines if lvl == 'error']
    assert errors and str(log) in errors[0]
    assert 'static assertion failed' in errors[0], 'the first error, not the last'


def test_a_successful_run_still_says_where_the_log_is(tmp_path, monkeypatch):
    from conifer.backends.aie import tools
    monkeypatch.setattr(tools, 'require_tools', lambda *a: None)
    monkeypatch.setattr(tools.os, 'system', lambda cmd: 0)
    lines = _capture_logger(monkeypatch, tools)

    assert tools.run_make(str(tmp_path), 'aiesim') is True
    assert any(str(tmp_path / 'aiesim.log') in m for lvl, m in lines if lvl == 'info')
    assert not [m for lvl, m in lines if lvl == 'error']


def test_next_step_does_not_offer_what_the_report_already_holds(tmp_path):
    """A hardware compile leaves the mapping behind without simulating, so the compile
    stage can already carry the tile memory the generic hint tells a user to go and get.
    """
    from conifer.backends.aie.report import read_aie_report
    work = tmp_path / 'build_x86' / 'Work'
    work.mkdir(parents=True)
    plain = read_aie_report(str(tmp_path))
    assert plain['stage'] == 'compile' and 'tile memory' in plain['next_step']

    (tmp_path / 'build_x86' / 'Map_Report.csv').write_text(
        'CLUSTER,TILE\nPT0,"(1, 2)"\n\nBUFFER,MEMORY_GROUP,SIZE\nb0,"(1, 2)",256\n')
    mapped = read_aie_report(str(tmp_path))
    assert mapped['tile_memory_bytes_max'] == 256
    assert 'tile memory' not in mapped['next_step']


def test_cyc_per_sample_is_the_arrays_cost_on_both_axes(monkeypatch):
    """Sample-split tiles each score a slice, so dividing a tile's cycles by its own
    slice reports a per-tile number where tree-split reports an array one -- and the
    estimate, which is always an array number, then misses by exactly n_tiles.
    """
    from conifer.backends.aie import report as rpt
    cores = [{'cyc': 12000, 'calls': 16, 'total': 13000} for _ in range(8)]
    monkeypatch.setattr(rpt, '_cores', lambda d: cores)
    monkeypatch.setattr(rpt, '_tree_split_latency', lambda *a: None)
    for axis in ('sample', 'tree'):
        out = {}
        rpt._build_metrics('', {'config': {'n_tiles': 8, 'split_axis': axis},
                                'n_samples': 512, 'clock_ghz': 1.25}, out)
        assert out['cyc_per_sample'] == pytest.approx(12000 / 512), axis


def test_every_macro_the_ladder_emits_is_defined_in_the_firmware():
    """The generator names macros the kernels define, and nothing checks that pairing.

    A generated call to a macro nobody defined is a compile error, which the
    toolchain-free tests would otherwise never see -- and did not: BDT_DECL_ROLE0 was
    emitted for tile 0 with only BDT_DEF_ROLE0 defined.
    """
    import re
    import glob
    from conifer.backends.aie import roles
    firmware = os.path.join(os.path.dirname(conifer.backends.aie.__file__),
                            'firmware', 'axis')
    defined = set()
    for path in glob.glob(os.path.join(firmware, '*')):
        with open(path) as f:
            defined |= set(re.findall(r'#\s*define\s+(BDT_(?:DECL|DEF)_\w+)\s*\(', f.read()))
    emitted = set()
    for n_tiles in (2, 3, 8, 128):
        decl, defn, _ = roles.ladder(n_tiles, 'tree', 'plio')
        emitted |= {line.split('(')[0] for line in decl + defn}
    assert emitted and emitted <= defined, emitted - defined


def test_the_search_space_follows_the_device_not_the_old_ladder():
    """The ladder is generated per project, so nothing stops the search at 64 tiles"""
    assert mapper.tile_candidates(304) == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert mapper.tile_candidates(112)[-1] == 64, 'the shim bound still caps it'


# ----- things the phase 6.1 sweep caught -----

def test_sample_split_deals_a_file_per_tile(skl_model, tmp_path):
    """Each tile scores its own samples, so each reads its own cut of the input"""
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=4, SplitAxis='sample', Priority='throughput'))
    model.write()
    model.write_input(np.zeros((model.n_samples, model.n_features)))
    written = sorted(f for f in os.listdir(tmp_path / 'data') if f.endswith('.dat'))
    assert len(written) == model.n_tiles, f'expected one file per tile, got {written}'
    key = f'.n{model.n_tiles}d{model.delta}'
    for t in range(model.n_tiles):
        assert f'x{key}.t{t}.dat' in written


def test_tree_split_writes_one_input_file(skl_model, tmp_path):
    """A memtile shares one stream, so there is nothing to deal"""
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=4, SplitAxis='tree'))
    model.write()
    model.write_input(np.zeros((model.n_samples, model.n_features)))
    written = [f for f in os.listdir(tmp_path / 'data') if f.endswith('.dat')]
    assert written == ['x.dat']


def test_the_deal_covers_every_sample_once(skl_model, tmp_path):
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=4, SplitAxis='sample', Priority='throughput'))
    model.write()
    model.write_input(np.zeros((model.n_samples, model.n_features)))
    total = sum(sum(1 for _ in open(tmp_path / 'data' / f))
                for f in os.listdir(tmp_path / 'data') if f.endswith('.dat'))
    assert total * 4 == model.n_samples * model.n_features_padded


def test_leaves_are_stored_no_wider_than_they_need(skl_model, tmp_path):
    """The leaf select chain runs at the stored width, not the accumulator's"""
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path))
    model.write()
    header = open(tmp_path / 'src/parameters.h').read()
    assert 'typedef int32_t score_t;' in header
    assert 'typedef int32_t leaf_t;' not in header, \
        'leaves stored at the accumulator width doubles every broadcast and select'


def test_oblique_estimate_carries_the_measured_tax(ydf_model, tmp_path):
    ymodel, _ = ydf_model
    model = conifer.converters.convert_from_ydf(ymodel, _oblique_config(tmp_path))
    axis = mapper.estimate(model.tables.n_trees, model.tables.max_depth,
                           model.n_features_padded, model.n_tiles, model.W, 2, 4,
                           'latency', split_axis=model.split_axis)
    assert (model.estimate['est_cyc_per_sample']
            > 2 * axis['est_cyc_per_sample']), 'oblique must not be priced as axis-aligned'


def test_leaf_width_is_the_narrowest_that_holds_the_values(skl_model, tmp_path):
    """The select chain runs at the stored width, so leaves narrow when they can"""
    import re
    clf, _ = skl_model
    for integer_bits, tag in ((4, 'verywide'), (16, 'wide'), (19, 'narrow')):
        score = f'ap_fixed<32,{integer_bits},AP_RND_CONV,AP_SAT>'
        model = conifer.converters.convert_from_sklearn(
            clf, _config(tmp_path / tag, ScorePrecision=score))
        model.write()
        header = open(tmp_path / tag / 'src/parameters.h').read()
        declared = next(l for l in header.split('\n') if 'leaf_t;' in l)
        values = [int(v) for v in
                  re.search(r'LEAVES\[\d+\] = \{(.*?)\};', header, re.S).group(1)
                  .replace('\n', '').split(',') if v.strip()]
        peak = max(abs(v) for v in values)
        want = next(b for b in (8, 16, 32) if peak <= 2 ** (b - 1) - 1)
        assert f'int{want}_t' in declared, f'peak {peak} declared {declared}'


def test_a_leaf_too_wide_for_16_bits_says_what_it_costs(skl_model, tmp_path, caplog):
    """A wide binary point can block the narrowing; the user cannot see that alone"""
    import logging
    clf, _ = skl_model
    with caplog.at_level(logging.INFO):
        model = conifer.converters.convert_from_sklearn(
            clf, _config(tmp_path, ScorePrecision='ap_fixed<32,4,AP_RND_CONV,AP_SAT>'))
        model.write()
    assert any('select chain runs at the stored width' in r.message for r in caplog.records)
