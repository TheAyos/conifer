'''AI Engine backend tests. None of these need the Vitis toolchain.

The end-to-end checks against a real aiecompiler live in tests/test_backends.py.
'''

import os
import re
import glob
import types
import shutil
import numpy as np
import pytest
import conifer
from conifer.backends.aie import checks, mapper, roles, tools
from conifer.backends.aie import report as rpt
from conifer.backends.aie.precision import Precision

SCORE = 'ap_fixed<32,16,AP_RND_CONV,AP_SAT>'
OBLIQUE_P = 'ap_fixed<16,6,AP_RND_CONV,AP_SAT>'
FIRMWARE = os.path.join(os.path.dirname(conifer.backends.aie.__file__), 'firmware')


def _config(tmp_path, **kwargs):
    cfg = conifer.backends.aie.auto_config()
    cfg['OutputDir'] = str(tmp_path)
    cfg.update(kwargs)
    return cfg


def _oblique_config(tmp_path, **kwargs):
    return _config(tmp_path, Precision=OBLIQUE_P, InputPrecision=OBLIQUE_P,
                   ThresholdPrecision=OBLIQUE_P,
                   WeightPrecision='ap_fixed<16,3,AP_RND_CONV,AP_SAT>', **kwargs)


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


def _replay(model, X):
    '''What the emitted tables score, in float'''
    return model.score_p.dequantize(
        model.tables.replay(X, model.threshold_p, model.score_p, model.init_predict[0],
                            norm=model.norm,
                            split_le=(model.splitting_convention == '<=')))


# --------------------------------------------------------------------------- project

@pytest.mark.parametrize('family', ['axis', 'oblique'])
def test_the_project_is_written_and_reproduces_itself(family, skl_model, ydf_model,
                                                      tmp_path):
    '''write() emits a buildable project, and resolving 'auto' is idempotent.

    Passing resolved_config() back must give a byte-identical project, which is what
    makes a reported mapping a reproducible one.
    '''
    assert 'aie' in conifer.backends.get_available_backends()
    if family == 'axis':
        clf, _ = skl_model
        convert, cfg = conifer.converters.convert_from_sklearn, _config
    else:
        clf, _ = ydf_model
        convert, cfg = conifer.converters.convert_from_ydf, _oblique_config

    a = convert(clf, cfg(tmp_path / 'a', NTiles=4, SplitAxis='tree'))
    a.write()
    assert a.family == family
    for f in ('src/parameters.h', 'src/graph.hpp', 'src/tb.cpp', 'Makefile',
              'aie_model.json'):
        assert os.path.exists(tmp_path / 'a' / f), f'missing {f}'
    # Every tile emits its own partial, oblique included; one output would silently
    # score a fraction of the ensemble.
    assert a.n_outputs == 4
    assert 'bdt_qs_tile_3' in (tmp_path / 'a' / 'src/tile_roles.h').read_text()

    # Pass the resolved dict back whole. Merging its keys into a fresh auto_config()
    # would prove nothing: that dict still carries the CamelCase alternates set to
    # 'auto', those win, and both sides then resolve to the same mapping by
    # coincidence rather than because anything was carried over.
    back = a.resolved_config()
    assert 'auto' not in [str(v) for v in back.values()]
    back['output_dir'] = str(tmp_path / 'b')
    b = convert(clf, back)
    b.write()
    assert (b.n_tiles, b.W, b.split_axis) == (a.n_tiles, a.W, a.split_axis)
    for f in ('src/parameters.h', 'aie_model.json'):
        pa = (tmp_path / 'a' / f).read_text().replace(str(tmp_path / 'a'), '')
        pb = (tmp_path / 'b' / f).read_text().replace(str(tmp_path / 'b'), '')
        assert pa == pb, f'{f} differs between auto and resolved'

    # aiecompiler reuses <target>/Work and rebuilds only the cores whose kernel changed
    # name, so a directory rewritten for a different mapping simulates a mixture of the
    # two -- an oblique build failed on a stack the current kernel fits inside, against
    # cores four hours old.
    work = tmp_path / 'b' / 'build_x86' / 'Work' / 'aie' / '11_0'
    work.mkdir(parents=True)
    (work / 'stale.elf').write_text('an earlier mapping')
    b.write()
    assert work.exists(), 'the same sources: the build is still the right one'
    convert(clf, cfg(tmp_path / 'b', NTiles=8, SplitAxis='sample',
                     Priority='throughput')).write()
    assert not (tmp_path / 'b' / 'build_x86').exists(), \
        'a different mapping must drop the Work it would have built on'


# ---------------------------------------------------------------------------- tables

@pytest.mark.parametrize('max_depth,n_features', [(4, 16), (1, 10), (3, 128)])
def test_the_tables_score_what_conifer_scores(tmp_path, max_depth, n_features):
    '''The emitted tables against conifer's own python backend, up to quantization.

    A stump (depth 1, one node and two leaves) and a 128-feature model are here
    because both were once refused: the cost table started at depth 2, and a
    64-feature bound outlived the experiment that needed it.
    '''
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = make_classification(n_samples=600, n_features=n_features,
                               n_informative=n_features // 2, random_state=0)
    clf = GradientBoostingClassifier(n_estimators=16, max_depth=max_depth,
                                     random_state=0).fit(X[:500], y[:500])
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=4, SplitAxis='tree'))
    model.write()

    ref = np.asarray(
        conifer.converters.convert_from_sklearn(clf).decision_function(X[500:])).ravel()
    got = _replay(model, X[500:])
    assert np.all(np.sign(ref) == np.sign(got))
    assert np.max(np.abs(ref - got)) < 0.01
    assert model.n_features == n_features
    assert model.tables.nodes_per_tree == (1 << max_depth) - 1


@pytest.mark.parametrize('integer_bits', [4, 16, 19])
def test_leaves_are_stored_no_wider_than_they_need(skl_model, tmp_path, integer_bits,
                                                   caplog):
    '''The leaf select chain runs at the stored width, so a leaf that fits int16 must
    not be held at the accumulator's. When the binary point blocks the narrowing, the
    user cannot see why unless it is said.
    '''
    clf, _ = skl_model
    with caplog.at_level('INFO'):
        model = conifer.converters.convert_from_sklearn(clf, _config(
            tmp_path, ScorePrecision=f'ap_fixed<32,{integer_bits},AP_RND_CONV,AP_SAT>'))
        model.write()
    header = (tmp_path / 'src/parameters.h').read_text()
    declared = next(l for l in header.split('\n') if 'leaf_t;' in l)
    values = [int(v) for v in re.search(r'LEAVES\[\d+\] = \{(.*?)\};', header, re.S)
              .group(1).replace('\n', '').split(',') if v.strip()]
    peak = max(abs(v) for v in values)
    want = next(b for b in (8, 16, 32) if peak <= 2 ** (b - 1) - 1)
    assert f'int{want}_t' in declared, f'peak {peak} declared {declared}'
    if want == 32:
        assert any('select chain runs at the stored width' in r.message
                   for r in caplog.records)


@pytest.mark.parametrize('n_tiles', [2, 16])
def test_sharding_preserves_the_score_and_narrows_the_rows(skl_model, tmp_path, n_tiles):
    '''Sharding permutes trees and feature rows so a tile reads a window rather than
    every row. The partials it produces must still sum to the unsharded score.
    '''
    from conifer.backends.aie.shard import Sharding
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=n_tiles, SplitAxis='tree'))
    X = np.random.default_rng(0).uniform(-8, 8, size=(48, model.n_features))
    assert len(model.sharding.verify(
        X, model.threshold_p, model.score_p, model.init_predict[0], norm=model.norm,
        split_le=(model.splitting_convention == '<='))) == 0

    identity = Sharding(model.tables, model.n_trees_padded, model.n_features_padded,
                        n_tiles, fperm=list(range(model.n_features_padded)),
                        optimize=False)
    assert model.sharding.total_rows < identity.total_rows


# --------------------------------------------------------------------------- mapping

@pytest.mark.parametrize('overrides,expect', [
    ({'Priority': 'latency'}, dict(split_axis='tree', shard=True, feed_memtile=True)),
    ({'Priority': 'throughput'}, dict(split_axis='sample', shard=False,
                                      feed_memtile=False)),
    ({'Priority': 'latency', 'Shard': False}, dict(shard=False)),
    ({'Priority': 'latency', 'Feed': 'plio'}, dict(shard=False, feed_memtile=False)),
    ({'Priority': 'latency', 'VectorWidth': 32}, dict(W=32)),
])
def test_the_mapping_follows_the_priority(skl_model, tmp_path, overrides, expect):
    '''A priority is a choice of split axis, and the rest follows from it: only a
    tree-split can shard, and only a memtile can hand each tile its own rows -- a PLIO
    multicasts one stream, so declining it declines sharding too.
    '''
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=8, **overrides))
    got = dict(split_axis=model.split_axis, shard=model.sharding is not None,
               feed_memtile=model.feed_memtile, W=model.W)
    for k, v in expect.items():
        assert got[k] == v, f'{k}: expected {v}, got {got[k]}'
    assert model.plio_rate == model.device['plio_rate_mhz']

    # The mapping has to reach the kernels, not just the model object.
    model.write()
    header = (tmp_path / 'src/parameters.h').read_text()
    assert f'#define BDT_SHARDED {int(got["shard"])}' in header
    assert f'#define BDT_FEED_MEMTILE {int(got["feed_memtile"])}' in header
    if got['shard']:
        for sym in ('N_SHARDS', 'WINDOWED', 'T_BEGIN', 'T_COUNT', 'N_FEAT', 'OFFSET'):
            assert sym in header, f'missing bdtsh::{sym}'


def test_neither_priority_is_beaten_on_its_own_metric(skl_model, tmp_path):
    clf, _ = skl_model
    est = {p: conifer.converters.convert_from_sklearn(
               clf, _config(tmp_path / p, Priority=p)).estimate
           for p in ('latency', 'throughput')}
    assert est['latency']['est_latency_ss_ns'] < est['throughput']['est_latency_ss_ns']
    assert (est['throughput']['est_throughput_ns_per_sample']
            < est['latency']['est_throughput_ns_per_sample'])


@pytest.mark.parametrize('max_depth,feat_bytes,latency,throughput', [
    (4, 2, 32, 64),    # 16-bit bitvector, 16-bit compare
    (5, 2, 16, 32),    # the bitvector widens and the group halves
    (6, 2, 8, 16),
    (4, 4, 16, 32),    # a 32-bit compare binds where the bitvector does not
    (6, 4, 8, 16),     # and the bitvector binds again at depth 6
])
def test_the_vector_fills_the_register_the_priority_wants(max_depth, feat_bytes,
                                                          latency, throughput):
    '''W is not searched: it is the group whose inner-loop vector fills 512 bits for
    latency and 1024 for throughput. A lane carries the result bitvector -- one bit per
    leaf -- and the feature word, whichever is wider.

    Measured on one tile, int16, aiesimulator: depth 4 latency_ss is 4316 ns at W=16
    against 4292 at W=32, and depth 5 is 7979 against 10816. The winner flips exactly
    where a lane doubles, and each winner fills 512 bits.
    '''
    assert mapper.lane_bits(max_depth, feat_bytes) == max(
        mapper.bitvector_bits(max_depth), 8 * feat_bytes)
    for priority, expect in (('latency', latency), ('throughput', throughput)):
        assert mapper.vector_width(priority, max_depth, feat_bytes) == expect
        _, W, _ = mapper.choose_mapping(32, max_depth, 16, feat_bytes, 4, priority,
                                        304, 1.25)
        assert W == expect, 'the cost model must not get a vote on W'


@pytest.mark.parametrize('n_trees,depth,trees_per_tile,W,measured', [
    (16, 4, 32, 32, 5354),    # the anchor the law was fitted at
    (16, 4, 8, 32, 1610),
    (16, 4, 1, 32, 518),      # and the width sweep, one tree per tile
    (16, 4, 1, 16, 345.7),
    (16, 5, 1, 32, 790.4),
    (16, 5, 1, 16, 500.4),
    (16, 6, 1, 32, 1533.5),
    (16, 6, 1, 8, 891.4),
])
def test_the_cost_model_tracks_the_study(n_trees, depth, trees_per_tile, W, measured):
    '''Measured on VEK280, int16, 16 features'''
    got = mapper.invocation_cycles(n_trees, depth, trees_per_tile, W, 2)
    assert abs(got - measured) / measured < 0.10, f'{got} against {measured}'


def test_the_estimate_prices_oblique_apart_from_axis_aligned(ydf_model, tmp_path):
    '''Tree-split divides the ensemble but not the basis, which is built over the whole
    feature set on every tile. A single multiplier over the axis-aligned law would
    divide it too and promise a speedup no mapping reaches.

    That same dense weight row is why an oblique model never shards or takes the
    memtile feed: there is no per-shard feature frame to hand a tile, so asking for one
    declines it rather than half-applying it.
    '''
    ymodel, _ = ydf_model
    fed = conifer.converters.convert_from_ydf(
        ymodel, _oblique_config(tmp_path / 'fed', NTiles=4, SplitAxis='tree',
                                Feed='memtile'))
    fed.write()
    assert fed.sharding is None and fed.feed_memtile is False

    model = conifer.converters.convert_from_ydf(ymodel, _oblique_config(tmp_path))
    axis = mapper.estimate(model.tables.n_trees, model.tables.max_depth,
                           model.n_features_padded, model.n_tiles, model.W, 2, 4,
                           'latency', model.device['clock_ghz'],
                           split_axis=model.split_axis)
    assert model.estimate['est_cyc_per_sample'] > 2 * axis['est_cyc_per_sample']


def test_the_tile_search_follows_the_device(skl_model, tmp_path):
    '''The ladder is generated per project, so nothing stops the search at 64 tiles --
    that was how far somebody had written `#if` out by hand.
    '''
    assert mapper.tile_candidates(304) == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert mapper.tile_candidates(112)[-1] == 64, 'the shim bound still caps it'
    n, _, notes = mapper.choose_mapping(32, 4, 16, 2, 4, 'latency', 112, 1.25)
    assert 1 <= n <= 112 and any('n_tiles' in s for s in notes)

    clf, _ = skl_model
    n = {p: conifer.converters.convert_from_sklearn(
             clf, _config(tmp_path / p, Priority=p)).n_tiles
         for p in ('latency', 'throughput')}
    assert n['latency'] == n['throughput'], 'auto stops at one ceiling for both'


# ----------------------------------------------------------------------------- batch

@pytest.mark.parametrize('asked', [1, 7, 333, 512, None])
def test_the_batch_is_a_whole_number_of_runs(skl_model, tmp_path, asked):
    '''n_samples is a batch, and a run is W samples on one tile under tree-split or on
    every tile under sample-split. Round up to a whole number of runs rather than
    refuse, and stay near the default target so two mappings stay comparable.
    '''
    from conifer.backends.aie.writer import DEFAULT_BATCH
    clf, _ = skl_model
    kw = {} if asked is None else {'NSamples': asked}
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path, **kw))
    step = model.W * (model.n_tiles if model.split_axis == 'sample' else 1)
    assert model.batch_step == step
    assert model.n_samples % step == 0
    if asked is None:
        assert model.n_samples >= min(DEFAULT_BATCH, step)
        assert model.n_samples < DEFAULT_BATCH + step
    else:
        assert asked <= model.n_samples < asked + step


def test_a_sample_split_says_which_tiles_get_only_padding(skl_model, tmp_path, caplog,
                                                          monkeypatch):
    '''Groups are dealt to tiles in turn, so a short X does not pad them evenly: it
    starves whole tiles. A run that outgrows the batch target has to say so too.
    '''
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, Priority='throughput', NTiles=16, SplitAxis='sample'))
    model.write()
    assert model.n_samples == model.W * 16
    assert any('one run is' in n and 'nothing but padding' in n for n in model.notes)

    monkeypatch.setattr(type(model), 'platform', lambda self: '/none.xpfm')
    monkeypatch.setattr(tools, 'run_make', lambda *a, **k: False)
    with caplog.at_level('INFO'):
        model.decision_function(np.zeros((model.W, model.n_features)))
    assert '15 of the 16 tiles score only padding' in caplog.text


@pytest.mark.parametrize('axis,priority,files', [('tree', 'latency', 1),
                                                 ('sample', 'throughput', 4)])
def test_the_input_is_dealt_the_way_the_graph_reads_it(skl_model, tmp_path, axis,
                                                       priority, files):
    '''A memtile shares one stream, so a tree-split has nothing to deal. A sample-split
    tile scores its own samples and reads its own cut, dealt in whole turns.
    '''
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=4, SplitAxis=axis, Priority=priority))
    model.write()
    model.write_input(np.zeros((model.n_samples, model.n_features)))

    written = sorted(f for f in os.listdir(tmp_path / 'data') if f.endswith('.dat'))
    assert len(written) == files, written
    if files > 1:
        key = f'.n{model.n_tiles}d{model.delta}'
        assert all(f'x{key}.t{t}.dat' in written for t in range(4))
    # Four values a line for the 64-bit PLIO, every sample dealt exactly once.
    total = sum(sum(1 for _ in open(tmp_path / 'data' / f)) for f in written)
    assert total * 4 == model.n_samples * model.n_features_padded

    with pytest.raises(ValueError, match='exceeds'):
        model.write_input(np.zeros((model.n_samples + 1, model.n_features)))


# ------------------------------------------------------------------------- refusals

def _precision(p):
    return dict(Precision=p, InputPrecision=p, ThresholdPrecision=p, WeightPrecision=p)


@pytest.mark.parametrize('overrides,match', [
    (dict(NTiles=1000), 'AI Engine tiles'),
    (dict(NTiles=128), 'outgoing PLIO channels'),
    (dict(PlioRate=99999), 'exceeds'),
    (_precision('ap_fixed<18,8,AP_RND_CONV,AP_SAT>'), 'width'),
    (_precision('ap_fixed<16,5>'), 'AP_RND_CONV'),
    (_precision('ap_fixed<16,5,AP_TRN,AP_WRAP>'), 'AP_RND_CONV'),
    (_precision('float'), 'ap_fixed'),
])
def test_every_refusal_names_what_to_change(skl_model, tmp_path, overrides, match):
    '''What a project cannot express, refused before any toolchain runs'''
    clf, _ = skl_model
    with pytest.raises(ValueError, match=match):
        conifer.converters.convert_from_sklearn(clf, _config(tmp_path, **overrides))


def test_the_shapes_the_backend_cannot_map_are_refused(ydf_model, tmp_path, caplog):
    '''Refusals that need a model of their own, and the one case that only warns'''
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from conifer.backends.aie.devices import get_device_config

    X, y = make_classification(n_samples=300, n_features=8, n_informative=6,
                               n_classes=3, random_state=0)
    multi = GradientBoostingClassifier(n_estimators=4, max_depth=3,
                                       random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError, match='one value per sample'):
        conifer.converters.convert_from_sklearn(multi, _config(tmp_path))

    with pytest.raises(ValueError, match='Known devices'):
        get_device_config('xcvu9p-flgb2104-2L-e')
    with pytest.raises(ValueError, match='not supported yet'):
        checks.check_max_depth(7)
    with pytest.raises(ValueError, match='at least 1'):
        checks.check_max_depth(0)

    # An oblique kernel signs a feature pair; it has no multiplier for a general weight.
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

    # A score that would saturate is a warning, not a refusal: the user may know the
    # range is wider than the data.
    checks.warn_score_range(10000, 5.0, Precision(SCORE))
    assert any('saturate' in r.message for r in caplog.records)


def test_the_required_rounding_mode_is_the_one_the_tables_are_quantized_in():
    '''Not that no other mode could agree -- AP_RND differs only at exact ties and
    AP_WRAP only on overflow -- but that quantize() rounds half to even and saturates,
    so AP_RND_CONV,AP_SAT is the mode describing the tables the writer emits.
    '''
    p = Precision('ap_fixed<16,6,AP_RND_CONV,AP_SAT>')
    half = (0.5 + np.arange(4)) / (1 << p.shift)
    assert list(p.quantize(half)) == [0, 2, 2, 4], 'ties to even, not away from zero'
    assert p.quantize([1e9])[0] == 2 ** 15 - 1, 'and the ends saturate rather than wrap'


# ----------------------------------------------------------------------------- scores

def _stub_simulator(model, monkeypatch):
    '''Stand in for x86simulator: score whatever write_input last wrote.

    Exercises write_input, the per-tile merge, read_scores and the chunking without a
    toolchain -- which is where a truncation bug lived.
    '''
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
            name = 'scores.dat' if i == 0 else f'scores.t{i}.dat'
            with open(os.path.join(d, name), 'w') as f:
                for v in (q if i == 0 else np.zeros_like(q)):
                    f.write(f'{int(v)}\n')
        return True

    model.write_input = write_input
    model.platform = lambda: '/stub/platform.xpfm'
    monkeypatch.setattr(tools, 'run_make', run_make)


@pytest.mark.parametrize('n_rows', [1, 33, 70])
def test_decision_function_returns_the_reference_scores(skl_model, tmp_path,
                                                        monkeypatch, n_rows):
    '''One score per row whatever the batch: a long X is split across runs and a short
    one padded, and every score must equal the reference rather than merely count.
    '''
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NSamples=32, NTiles=2, SplitAxis='tree'))
    model.write()
    _stub_simulator(model, monkeypatch)

    X = np.random.default_rng(1).uniform(-4, 4, size=(n_rows, model.n_features))
    y = model.decision_function(X)
    assert len(y) == n_rows
    np.testing.assert_array_equal(y, _replay(model, X))


def test_the_reader_sums_the_tiles_and_honours_the_simulator_asked_for(skl_model,
                                                                      tmp_path):
    '''Every tile writes its own partial on its own port, so read_scores adds them up.
    It must also read the run that was asked for: decision_function reads its own x86
    output even when an older build() left an aiesimulator one behind, and asking for
    'aie' must not fall back to the x86 scores -- that would read as agreement between
    two runs of the same thing.
    '''
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=4, SplitAxis='tree'))
    model.write()
    d = tmp_path / 'data'
    os.makedirs(d, exist_ok=True)
    parts = [np.arange(model.n_samples) * (i + 1) for i in range(4)]
    for i, part in enumerate(parts):
        name = 'scores.dat' if i == 0 else f'scores.t{i}.dat'
        (d / name).write_text(''.join(f'T 100 ns\n{int(v)}\n' for v in part))
    np.testing.assert_allclose(
        model.read_scores(), model.score_p.dequantize(np.sum(parts, axis=0)))

    for sub, val in (('build_x86/x86simulator_output', 16),
                     ('build_hw/aiesimulator_output', 32)):
        os.makedirs(tmp_path / sub, exist_ok=True)
        for i in range(4):
            name = 'scores.dat' if i == 0 else f'scores.t{i}.dat'
            (tmp_path / sub / name).write_text(f'{val if i == 0 else 0}\n')
    assert model.read_scores()[0] == model.score_p.dequantize([16])[0]
    assert model.read_scores(simulator='aie')[0] == model.score_p.dequantize([32])[0]

    shutil.rmtree(tmp_path / 'build_hw')
    with pytest.raises(FileNotFoundError):
        model.read_scores(simulator='aie')


# ----------------------------------------------------------------------------- build

def test_build_names_the_rows_and_the_stage_it_runs(skl_model, tmp_path, monkeypatch):
    '''build() ran the simulator on whatever data/x.dat held, so its result depended on
    a prior decision_function() -- which leaves its LAST batch. simulate=False stops
    after the hardware compile, which needs no stimulus at all.
    '''
    clf, X = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path))
    ran = []
    monkeypatch.setattr(tools, 'run_make',
                        lambda out, target, **kw: ran.append(target) or True)
    monkeypatch.setattr(type(model), 'platform', lambda self: 'x.xpfm')

    model.build(X[:8])
    assert ran == ['aiesim']
    first = (tmp_path / 'data' / 'x.dat').read_text()

    ran.clear()
    model.build(X[8:16])
    assert (tmp_path / 'data' / 'x.dat').read_text() != first

    ran.clear()
    model.build(simulate=False)
    assert ran == ['hw_build']

    with pytest.raises(ValueError):
        model.build(X[:8], simulate=False)


# ---------------------------------------------------------------------------- report

def test_the_report_reads_whatever_stage_is_on_disk(skl_model, tmp_path):
    '''Never raises for a stage that has not run, and the hint offers only what the
    report does not already hold: a hardware compile leaves the mapping behind without
    simulating, so it can already carry the tile memory the generic hint points at.
    '''
    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path))
    model.write()
    written = model.read_report()
    assert written['stage'] == 'write' and written['next_step']
    assert written['estimate']['est_latency_ss_ns'] > 0

    (tmp_path / 'build_x86' / 'Work').mkdir(parents=True)
    compiled = rpt.read_aie_report(str(tmp_path))
    assert compiled['stage'] == 'compile' and 'tile memory' in compiled['next_step']

    (tmp_path / 'build_x86' / 'Map_Report.csv').write_text(
        'CLUSTER,TILE\nPT0,"(1, 2)"\n\nBUFFER,MEMORY_GROUP,SIZE\nb0,"(1, 2)",256\n')
    mapped = rpt.read_aie_report(str(tmp_path))
    assert mapped['tile_memory_bytes_max'] == 256
    assert 'tile memory' not in mapped['next_step']


@pytest.mark.parametrize('split_axis', ['tree', 'sample'])
def test_the_measured_metrics_are_the_arrays_and_not_the_runs(tmp_path, monkeypatch,
                                                              split_axis):
    '''Two numbers that were both wrong, and in opposite directions.

    cyc_per_sample divides by what the ARRAY retired, so both axes report the same
    thing and the estimate can be read against it. throughput is the steady-state
    invocation period over the samples an invocation retires -- not total_cycle_count,
    which exceeds the kernel time by a fixed graph startup (7300-9600 cycles whatever
    the model) and so reports that constant divided by the batch.
    '''
    W, n_tiles, n_samples, ghz = 32, 4, 128, 1.25
    period, kernel_cyc, startup = 400.0, 8 * 1024, 8000
    monkeypatch.setattr(rpt, '_cores', lambda d: [
        {'col': i, 'row': 0, 'name': f'bdt_qs_tile_{i}', 'calls': n_samples // W,
         'cyc': kernel_cyc, 'total': kernel_cyc + startup} for i in range(n_tiles)])
    with open(tmp_path / 'scores.dat', 'w') as f:
        for g in range(n_samples // W):
            for j in range(W):
                f.write(f'T {g * period + j:g} ns\n0\n')

    out = {}
    rpt._build_metrics(str(tmp_path), {'config': {'n_tiles': n_tiles, 'vector_width': W,
                                                  'split_axis': split_axis},
                                       'n_samples': n_samples, 'clock_ghz': ghz}, out)
    per_invocation = W * (n_tiles if split_axis == 'sample' else 1)
    assert out['cyc_per_sample'] == pytest.approx(kernel_cyc / n_samples)
    assert out['throughput_ns_per_sample'] == pytest.approx(period / per_invocation)
    assert out['run_ns_per_sample'] == pytest.approx(
        (kernel_cyc + startup) / ghz / n_samples)
    assert out['run_ns_per_sample'] > 3 * out['throughput_ns_per_sample'], \
        'the whole run is the number that used to be reported as throughput'
    assert out['io_ns_per_sample'] == pytest.approx(
        out['throughput_ns_per_sample'] - out['cyc_per_sample'] / ghz)


def test_latency_ss_is_a_fit_not_a_mean():
    '''A pipelined mapping can hold a steady period while residence climbs, so the mean
    is a function of the run length and the intercept is not. A trimmed window must
    still report group zero rather than its own accumulated skew.
    '''
    def drifting(n, base=300.0, drift=6.0):
        return [base + drift * i for i in range(n)]

    fits = []
    for n in (8, 16, 32, 64):
        r = {}
        rpt._summarise_latency(drifting(n), r)
        fits.append(r['latency_ss_ns'])
    assert max(fits) - min(fits) < 1e-6, f'intercept moved with run length: {fits}'
    means = [np.mean(drifting(n)) for n in (8, 16, 32, 64)]
    assert max(means) - min(means) > 100, 'the mean should move, or this proves nothing'

    r = {}
    rpt._summarise_latency(drifting(32, drift=6.4), r)
    assert r['latency_ss_ns'] == pytest.approx(300.0)
    assert r['latency_ss_drift_ns_per_group'] == pytest.approx(6.4)

    full, trimmed = {}, {}
    rpt._summarise_latency(drifting(32), full)
    rpt._summarise_latency(drifting(32)[4:28], trimmed, offset=4)
    assert trimmed['latency_ss_ns'] == pytest.approx(full['latency_ss_ns'])

    too_few = {}
    rpt._summarise_latency([300.0, 306.0], too_few)
    assert 'latency_ss_ns' not in too_few and 'unmeasured' in too_few['latency_ss_note']


# ------------------------------------------------------------------------------ tools

def _capture_logger(monkeypatch, module):
    lines = []
    monkeypatch.setattr(module, 'logger', types.SimpleNamespace(
        info=lambda m, *a: lines.append(('info', str(m))),
        debug=lambda m, *a: lines.append(('debug', str(m))),
        warning=lambda m, *a: lines.append(('warning', str(m))),
        error=lambda m, *a: lines.append(('error', str(m)))))
    return lines


@pytest.mark.parametrize('outcome,expect', [
    ('ok', True),
    ('error', False),
    ('signal', False),   # subprocess reports a signal death as a NEGATIVE return code,
])                       # so testing it for > 0 would call a killed build a success
def test_a_tool_run_reports_where_it_failed(tmp_path, monkeypatch, outcome, expect):
    '''A failed tool leaves hundreds of lines behind, so a run always names its log and
    quotes the first real error. An interrupt is not a failure at all: os.system is
    system(3) and ignores SIGINT for the child's lifetime, so a Ctrl-C during a build
    that runs for minutes came back as a toolchain failure with an empty log.
    '''
    def call(cmd, shell=None, stdout=None, stderr=None):
        if outcome == 'error':
            stdout.write('INFO: compiling\n'
                         '../src/params.h:99:15: error: static assertion failed\n'
                         'ERROR: [aiecompiler 77-753] cannot recover\n')
            return 2
        return 0 if outcome == 'ok' else -9

    monkeypatch.setattr(tools, 'require_tools', lambda *a: None)
    monkeypatch.setattr(tools.subprocess, 'call', call)
    lines = _capture_logger(monkeypatch, tools)

    assert tools.run_make(str(tmp_path), 'aiesim') is expect
    log = str(tmp_path / 'aiesim.log')
    assert any(log in m for lvl, m in lines if lvl == 'info'), 'always name the log'
    errors = [m for lvl, m in lines if lvl == 'error']
    assert bool(errors) is not expect
    if outcome == 'error':
        assert log in errors[0]
        assert 'static assertion failed' in errors[0], 'the first error, not the last'

    def interrupted(cmd, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(tools.subprocess, 'call', interrupted)
    lines.clear()
    with pytest.raises(KeyboardInterrupt):
        tools.run_make(str(tmp_path), 'aiesim')
    assert not [m for lvl, m in lines if lvl == 'error']

    # The tools finish with "(WARNING:3, CRITICAL-WARNING:0, ERROR:0)", which contains
    # the word and reports none; quoting that as the first error hides the real one.
    log_file = tmp_path / 'tally.log'
    log_file.write_text('(WARNING:3, CRITICAL-WARNING:0, ERROR:0)\n'
                        'ERROR: [aiecompiler 77-753] cannot recover\n')
    assert tools._first_error(str(log_file)) == \
        'ERROR: [aiecompiler 77-753] cannot recover'
    log_file.write_text('(WARNING:3, CRITICAL-WARNING:0, ERROR:0)\nall good\n')
    assert tools._first_error(str(log_file)) is None


# ---------------------------------------------- what the generator and the kernels share

def test_the_generated_ladder_pairs_with_the_kernels_it_calls(skl_model, tmp_path):
    '''A tile's tree range is baked into its symbol, so kernel::create needs a literal
    name and the enumeration cannot be a loop. Generating it makes the ceiling the
    device's core count rather than however far somebody wrote `#if` out by hand.

    Nothing else checks that the macros it emits are the ones the firmware defines: a
    generated call to an undefined macro is a compile error the toolchain-free tests
    would never see, and did not -- BDT_DECL_ROLE0 was emitted with only BDT_DEF_ROLE0
    defined, which broke every multi-tile build.
    '''
    defined = set()
    for path in glob.glob(os.path.join(FIRMWARE, 'axis', '*')):
        defined |= set(re.findall(r'#\s*define\s+(BDT_(?:DECL|DEF)_\w+)\s*\(',
                                  open(path).read()))
    emitted = set()
    for n_tiles in (2, 3, 8, 128):
        decl, defn, _ = roles.ladder(n_tiles, 'tree', 'plio')
        emitted |= {line.split('(')[0] for line in decl + defn}
    assert emitted and emitted <= defined, emitted - defined

    # Tile 0 carries the ensemble's base score, so its definition differs from its
    # neighbours'. Its declaration does not, which is what lets one list serve both.
    decl, defn, _ = roles.ladder(4, 'tree', 'plio')
    assert decl[0] == 'BDT_DECL_ROLE0(0)' and defn[0] == 'BDT_DEF_ROLE0(0)'
    assert all('ROLE0' not in d for d in decl[1:] + defn[1:])
    # One symbol serves every tile of a sample-split and of N=1; the graph loops.
    assert roles.ladder(8, 'sample', 'plio') == ([], [], [])
    assert roles.ladder(1, 'tree', 'plio') == ([], [], [])

    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path, NTiles=100))
    model.write()
    text = (tmp_path / 'src' / 'tile_roles.h').read_text()
    assert model.n_tiles == 100 and 'bdt_qs_tile_99' in text
    assert '#pragma once' not in text, 'included once per section, so no include guard'

    # The Makefile the writer emits keeps the two toolchain stages apart, so asking
    # what a design costs does not require the simulator.
    makefile = (tmp_path / 'Makefile').read_text()
    assert 'hw_build: check-platform' in makefile
    assert 'aiesim: hw_build' in makefile, 'the simulator still needs a compiled graph'
    body = makefile.split('aiesim: hw_build')[1].split('\n\n')[0]
    assert 'aiecompiler' not in body, 'the compile belongs to hw_build now'


def test_the_kernels_match_what_the_generator_promises():
    '''Two pairings nothing else checks, both once broken.

    table_bytes is the memory a user is shown and stack_size is what the tile gets:
    ((X + n * KIB - 1) / KIB) * KIB reads like "X plus n KiB, rounded" and gives one
    KiB less. And the graphs kept the research tree's stimulus path as an #ifndef
    fallback -- dead, since parameters.h always defines XIN_FILE, but a dead default
    naming a directory no conifer user has is a trap waiting for the day it is not.
    '''
    b = mapper.table_bytes(n_trees=32, max_depth=4, max_leaves=16, n_features=16,
                           W=32, feat_bytes=2, leaf_bytes=4, oblique=False)
    tables = sum(v for k, v in b.items() if k not in ('heap', 'stack', 'total'))
    text = open(os.path.join(FIRMWARE, 'axis', 'graph.hpp')).read()
    got = {}
    for kind in ('heap', 'stack'):
        expr = re.search(rf'{kind}_size\(kk\) = ([^;]+);', text).group(1)
        got[kind] = eval(expr.replace('/', '//'), {},
                         dict(KIB=1024, TABLES=tables, XBYTES=16 * 32 * 2))
    assert (got['heap'], got['stack']) == (b['heap'], b['stack']), (got, b)

    for path in glob.glob(os.path.join(FIRMWARE, '**', '*'), recursive=True):
        if os.path.isfile(path):
            body = open(path, errors='ignore').read()
            for residue in ('gen/out', 'xin_file.h', 'X_fm_'):
                assert residue not in body, f'{os.path.basename(path)} names {residue}'


# --------------------------------------------------------------------------- platform

def test_the_platform_is_found_or_says_where_it_looked(tmp_path, monkeypatch):
    '''settings64.sh sets XILINX_VITIS but not PLATFORM_REPO_PATHS. Isolate every root
    the search reads, not just the ones the docstring names -- a test that isolates
    part of the environment is one that passes only on the machine it was written on.
    '''
    from conifer.backends.aie import platforms
    for var in ('PLATFORM_REPO_PATHS', 'XILINX_VITIS', 'XILINX_HLS'):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match='settings64|PLATFORM_REPO_PATHS'):
        platforms.resolve_platform('vek280_base')

    base = tmp_path / 'Vitis' / 'base_platforms'
    for name in ('vek280_base', 'xilinx_vek280_base_202610_1'):
        os.makedirs(base / name)
        open(base / name / f'{name}.xpfm', 'w').close()
    monkeypatch.setenv('XILINX_VITIS', str(tmp_path / 'Vitis'))
    # The exactly named directory wins over the versioned one.
    assert platforms.find_platform('vek280_base').endswith(
        'vek280_base/vek280_base.xpfm')

    shutil.rmtree(base / 'vek280_base')
    assert platforms.find_platform('vek280_base') == str(
        base / 'xilinx_vek280_base_202610_1' / 'xilinx_vek280_base_202610_1.xpfm')
