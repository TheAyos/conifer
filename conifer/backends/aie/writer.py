import os
import copy
import json
import math
import shutil
import datetime
import numpy as np
from conifer.utils import copydocstring
from conifer.backends.common import MultiPrecisionConfig
from conifer.model import ModelBase, ConfigBase
from conifer.backends.aie import checks, mapper, tables as _tables
from conifer.backends.aie.precision import Precision, COMPARE_WIDTH, SCORE_WIDTH
from conifer.backends.aie.devices import get_device_config
from conifer.backends.aie.report import read_aie_report
from conifer.backends.aie import tools
import logging
logger = logging.getLogger(__name__)

AUTO = 'auto'

_DEFAULT_COMPARE = 'ap_fixed<16,5,AP_RND_CONV,AP_SAT>'
_DEFAULT_SCORE = 'ap_fixed<32,16,AP_RND_CONV,AP_SAT>'


class AIEConfig(MultiPrecisionConfig):
    backend = 'aie'
    _config_fields = MultiPrecisionConfig._config_fields + [
        'priority', 'n_tiles', 'split_axis', 'vector_width', 'tau', 'n_samples',
        'xilinx_part', 'platform', 'elfgen_jobs']
    _aie_alts = {'priority': ['Priority'],
                 'n_tiles': ['NTiles'],
                 'split_axis': ['SplitAxis'],
                 'vector_width': ['VectorWidth', 'W'],
                 'tau': ['Tau'],
                 'n_samples': ['NSamples'],
                 'xilinx_part': ['XilinxPart'],
                 'platform': ['Platform'],
                 'elfgen_jobs': ['ElfgenJobs'],
                 }
    _alternates = {**MultiPrecisionConfig._alternates, **_aie_alts}
    _aie_defaults = {'precision': _DEFAULT_COMPARE,
                     'score_precision': _DEFAULT_SCORE,
                     'priority': 'latency',
                     'n_tiles': AUTO,
                     'split_axis': AUTO,
                     'vector_width': AUTO,
                     'tau': AUTO,
                     'n_samples': AUTO,
                     'xilinx_part': 'xcve2802-vsvh1760-2MP-e-S',
                     'platform': None,
                     'elfgen_jobs': None,
                     }
    _defaults = {**MultiPrecisionConfig._defaults, **_aie_defaults}
    _allow_undefined = [*MultiPrecisionConfig._allow_undefined] + ['platform', 'elfgen_jobs']

    def __init__(self, configDict, validate=True):
        super(AIEConfig, self).__init__(configDict, validate=False)
        for key, val in AIEConfig._aie_defaults.items():
            if getattr(self, key, None) is None and key in AIEConfig._config_fields:
                setattr(self, key, val)
        if self.score_precision is None or 'ap_' not in str(self.score_precision):
            self.score_precision = _DEFAULT_SCORE
        if validate:
            self._validate()

    def default_config():
        return copy.deepcopy(AIEConfig._defaults)

    def _extra_validate(self):
        if self.priority not in ('latency', 'throughput'):
            raise ValueError(f"priority must be 'latency' or 'throughput', got "
                             f"'{self.priority}'")
        if self.split_axis not in (AUTO, 'tree', 'sample'):
            raise ValueError(f"split_axis must be 'tree', 'sample' or '{AUTO}', got "
                             f"'{self.split_axis}'")
        assert self.input_precision == self.threshold_precision, \
            (f'input & threshold precision must be equal, got: {self.input_precision} & '
             f'{self.threshold_precision}')


class AIEModel(ModelBase):

    def __init__(self, ensembleDict, config, metadata=None):
        super(AIEModel, self).__init__(ensembleDict, config, metadata)
        self.config = AIEConfig(config)
        cfg = self.config

        self.threshold_p = Precision(cfg.threshold_precision)
        self.score_p = Precision(cfg.score_precision)
        self.weight_p = Precision(cfg.weight_precision)
        self.threshold_p.validate('threshold', allowed=(COMPARE_WIDTH,))
        self.score_p.validate('score', allowed=(SCORE_WIDTH,))
        self.weight_p.validate('weight', allowed=(COMPARE_WIDTH,))

        checks.check_n_classes(self.n_classes)
        self.oblique = checks.check_weights(self.trees)
        self.device = get_device_config(cfg.xilinx_part)

        flat = [tc[0] for tc in self.trees]
        self.n_features_padded = checks.padded_n_features(self.n_features, self.oblique)
        if self.n_features_padded != self.n_features:
            logger.info(f'padding {self.n_features} features to {self.n_features_padded}: '
                        f'the oblique kernel loads a weight row as one vector')
            flat = [_pad_tree_features(t, self.n_features_padded) for t in flat]

        self.tables = _tables.QuickScorerTables(
            flat, self.n_features_padded, oblique=self.oblique,
            weight_precision=self.weight_p if self.oblique else None)
        checks.check_max_depth(self.tables.max_depth)
        self.basis = self.tables.basis()

        self._resolve_mapping()
        self._check_shape()
        self.notes = self._notes

    @property
    def family(self):
        return 'oblique' if self.oblique else 'axis'

    def _resolve_mapping(self):
        cfg = self.config
        d = self.tables.max_depth
        self._notes = []

        self.priority = cfg.priority
        self.split_axis = (('tree' if self.priority == 'latency' else 'sample')
                           if cfg.split_axis == AUTO else cfg.split_axis)
        self.W = (mapper.vector_width(self.priority, d) if cfg.vector_width == AUTO
                  else int(cfg.vector_width))
        checks.check_vector_width(self.W, self.threshold_p.n_bytes)

        if cfg.n_tiles == AUTO:
            n, notes = mapper.choose_n_tiles(
                self.tables.n_trees, d, self.n_features_padded, self.W,
                self.threshold_p.n_bytes, self.score_p.n_bytes, self.priority,
                self.device['n_tiles'], self.device['tile_memory_bytes'], self.oblique)
            self.n_tiles, self._notes = n, notes
        else:
            self.n_tiles = int(cfg.n_tiles)
        checks.check_n_tiles(self.n_tiles, self.oblique, self.device['n_tiles'],
                             mapper.MAX_TEMPLATE_TILES)

        if self.split_axis == 'tree' and self.n_tiles > 1:
            self.tau = (int(math.ceil(self.tables.n_trees / self.n_tiles))
                        if cfg.tau == AUTO else int(cfg.tau))
        else:
            self.tau = self.tables.n_trees
        # Every shard runs t_count = TAU trees, so a ragged split overruns the last one.
        # Null trees make the split exact and contribute zero.
        self.n_trees_padded = (self.tau * self.n_tiles if self.split_axis == 'tree'
                               else self.tables.n_trees)
        if self.n_trees_padded > self.tables.n_trees:
            self._notes.append(
                f'padding {self.tables.n_trees} trees to {self.n_trees_padded} with null '
                f'trees so every tile takes exactly tau={self.tau}')

        self.n_samples = (self._auto_n_samples() if cfg.n_samples == AUTO
                          else int(cfg.n_samples))
        self.estimate = mapper.estimate(
            self.n_trees_padded, self.tables.max_depth, self.n_features_padded,
            self.n_tiles, self.W, self.threshold_p.n_bytes, self.score_p.n_bytes,
            self.priority, self.oblique)

    def _auto_n_samples(self):
        step = self.W * (self.n_tiles if self.split_axis == 'sample' else 1)
        return step * max(1, 256 // step)

    def _check_shape(self):
        f = self.n_features_padded
        if not self.oblique and f > 64:
            raise ValueError(
                f'n_features {f} exceeds 64, which is where the kernel indexes its feature '
                f'mask with a 64-bit shift')
        if self.n_samples % self.W:
            raise ValueError(f'n_samples {self.n_samples} must be a multiple of W {self.W}')
        if self.split_axis == 'sample' and (self.n_samples // self.W) % self.n_tiles:
            raise ValueError(
                f'sample-split needs (n_samples / W) divisible by n_tiles, got '
                f'({self.n_samples} / {self.W}) % {self.n_tiles}')
        p = self.tables.nodes_per_tree
        if self.oblique:
            terms = self.basis['max_terms']
            if p * terms > 64:
                raise ValueError(
                    f'the oblique kernel loads one 64-lane vector of terms per tree, so '
                    f'QS_NODES_PER_TREE * MAX_TERMS must be <= 64, got {p} * {terms}. '
                    f'Reduce max_depth to {int(math.log2(64 // terms + 1))} or fewer')
            qt_lanes = max(1 << (p - 1).bit_length(), 8)
            if qt_lanes * self.tables.bv_bits > 1024:
                raise ValueError(
                    f'the oblique kernel has no chunked node loop, which max_depth '
                    f'{self.tables.max_depth} would need')
        mem = mapper.table_bytes(
            self.n_trees_padded, self.tables.max_depth, self.tables.max_leaves,
            self.n_features_padded, self.W, self.threshold_p.n_bytes,
            self.score_p.n_bytes, self.oblique,
            self.basis['basis_n'] if self.oblique else 0,
            self.basis['max_terms'] if self.oblique else 1)
        self.memory = mem
        checks.warn_tile_memory(mem, self.device['tile_memory_bytes'])
        checks.warn_score_range(self.n_trees_padded, float(np.max(np.abs(self.tables.leaves))),
                                self.score_p)

    def resolved_config(self):
        '''The configuration with every "auto" filled in, ready to be passed back'''
        cfg = self.config._to_dict()
        cfg.update({'priority': self.priority,
                    'n_tiles': self.n_tiles,
                    'split_axis': self.split_axis,
                    'vector_width': self.W,
                    'tau': self.tau,
                    'n_samples': self.n_samples,
                    'xilinx_part': self.config.xilinx_part,
                    'platform': self.config.platform or self.device['platform'],
                    })
        return cfg

    @copydocstring(ModelBase.write)
    def write(self):
        cfg = self.config
        out = cfg.output_dir
        os.makedirs(f'{out}/src', exist_ok=True)
        os.makedirs(f'{out}/data', exist_ok=True)
        self.save()

        firmware = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'firmware')
        for src in ('common', self.family):
            d = f'{firmware}/{src}'
            for name in sorted(os.listdir(d)):
                if os.path.isfile(f'{d}/{name}'):
                    shutil.copyfile(f'{d}/{name}', f'{out}/src/{name}')

        with open(f'{out}/src/parameters.h', 'w') as f:
            f.write(self._parameters_h())
        with open(f'{out}/aie_model.json', 'w') as f:
            json.dump(self._model_json(), f, indent=2)
        self._write_makefile()

        for note in self._notes:
            logger.info(note)
        logger.info(f'estimated {self.estimate["est_cyc_per_sample"]:.1f} cyc/sample, '
                    f'latency_ss {self.estimate["est_latency_ss_ns"]:.0f} ns on '
                    f'{self.n_tiles} tile(s)')
        for v in self.estimate['validity']:
            logger.info(f'estimate caveat: {v}')

    def _write_makefile(self):
        cfg = self.config
        template = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template',
                                'Makefile')
        platform = cfg.platform or self.device['platform']
        subs = {'@PLATFORM@': platform,
                '@TOP@': '../src/tb.cpp',
                '@ELFGEN_JOBS@': '' if cfg.elfgen_jobs is None else str(cfg.elfgen_jobs),
                '@XIN@': f'{cfg.output_dir}/data/x.dat',
                }
        with open(template) as f:
            text = f.read()
        for k, v in subs.items():
            text = text.replace(k, v)
        with open(f'{cfg.output_dir}/Makefile', 'w') as f:
            f.write(text)

    def _model_json(self):
        return {'family': self.family,
                'oblique': self.oblique,
                'n_trees': self.tables.n_trees,
                'n_trees_padded': self.n_trees_padded,
                'n_features': self.n_features,
                'n_features_padded': self.n_features_padded,
                'max_depth': self.tables.max_depth,
                'max_leaves': self.tables.max_leaves,
                'n_samples': self.n_samples,
                'fx_shift': self.threshold_p.shift,
                'val_shift': self.score_p.shift,
                'wgt_shift': self.weight_p.shift,
                'norm': self.norm,
                'config': self.resolved_config(),
                'estimate': self.estimate,
                'memory': self.memory,
                'notes': self._notes,
                }

    def _parameters_h(self):
        t, cfg = self.tables, self.config
        T, P = self.n_trees_padded, t.nodes_per_tree
        L, F = t.max_leaves, self.n_features_padded
        bv_c = {16: 'uint16_t', 32: 'uint32_t', 64: 'uint64_t'}[t.bv_bits]
        q = t.quantize(self.threshold_p, self.score_p, self.norm)
        init_q = int(self.score_p.quantize([float(self.init_predict[0]) * self.norm])[0])
        all_ones = (1 << t.bv_bits) - 1

        qt_feat = _pad(t.qt_group, T * P, 0)
        qt_thr = _pad(list(q['qt_thr']), T * P, 0)
        qt_bv = _pad(list(t.qt_bv), T * P, all_ones)
        init_v = _pad(list(t.init_v), T, all_ones)
        leaves = np.zeros((T, L), dtype=np.int64)
        leaves[:t.n_trees] = q['leaves']

        s = ['// Generated by conifer. Do not edit.',
             '#pragma once', '#include <cstdint>', '']
        s.append(f'#define BDT_W {self.W}')
        s.append(f'#define XIN_FILE "{cfg.output_dir}/data/x.dat"')
        if self.family == 'axis':
            s += [f'#define BDT_N_TILES {self.n_tiles}',
                  f'#define BDT_SPLIT_TREE {1 if self.split_axis == "tree" else 0}',
                  f'#define BDT_MERGE_PLIO 1',
                  f'#define BDT_FEED_PLIO 0',
                  f'#define BDT_TAU {self.tau if self.split_axis == "tree" else 0}',
                  '#define BDT_SHARDED 0',
                  '#define BDT_FEED_MEMTILE 0',
                  '#define BDT_TAP 0',
                  '#define BDT_PLACE 0',
                  '#define BDT_MERGE_REDUCE 0']
        s += ['', 'namespace bdtm {', '',
              f'typedef {self.threshold_p.c_type} feat_t;',
              f'typedef {self.score_p.c_type} score_t;',
              f'typedef {self.score_p.c_type} leaf_t;',
              f'typedef {bv_c} bv_t;', '',
              f'constexpr unsigned N_FEATURES = {F};',
              f'constexpr unsigned N_TREES    = {T};',
              f'constexpr unsigned MAX_LEAVES = {L};',
              f'constexpr unsigned N_SAMPLES  = {self.n_samples};',
              f'constexpr unsigned QS_NODES_PER_TREE = {P};',
              f'constexpr bool SPLIT_LE = {"true" if self.splitting_convention == "<=" else "false"};',
              f'constexpr score_t INIT_PREDICT = {init_q};']
        if self.oblique:
            b = self.basis
            s += [f'constexpr int WGT_SHIFT = {self.weight_p.shift};',
                  f'constexpr unsigned BASIS_N = {b["basis_n"]};',
                  f'constexpr unsigned MAX_TERMS = {b["max_terms"]};']
        s.append('')
        s.append(_array('int16_t', 'QT_FEAT', qt_feat))
        s.append(_array(self.threshold_p.c_type, 'QT_THR', qt_thr))
        s.append(_array('bv_t', 'QT_BV', qt_bv, hexw=t.bv_bits))
        s.append(_array('bv_t', 'INIT_V', init_v, hexw=t.bv_bits))
        s.append(_array('leaf_t', 'LEAVES', leaves.ravel()))
        if self.oblique:
            b = self.basis
            n = T * P * b['max_terms'] + 64
            s.append(_array('int16_t', 'BASIS_I', b['basis_i']))
            s.append(_array('int16_t', 'BASIS_J', b['basis_j']))
            s.append(_array(self.threshold_p.c_type, 'BASIS_WI', b['basis_wi']))
            s.append(_array(self.threshold_p.c_type, 'BASIS_WJ', b['basis_wj']))
            s.append(_array('int16_t', 'QT_BTERM', _pad(list(b['qt_bterm']), n, 0)))
            s.append(_array(self.threshold_p.c_type, 'QT_BSIGN',
                            _pad(list(b['qt_bsign']), n, 0)))
        s += ['}  // namespace bdtm', '']
        return '\n'.join(s)

    @copydocstring(ModelBase.compile)
    def compile(self):
        self.write()
        return tools.run_make(self.config.output_dir, 'x86sim_build')

    @copydocstring(ModelBase.decision_function)
    def decision_function(self, X, trees=False):
        if trees:
            logger.warn('Individual tree output (trees=True) is not implemented for the aie '
                        'backend')
        X = np.asarray(X)
        assert X.shape[1] == self.n_features, \
            f'Wrong number of features, expected {self.n_features}, got {X.shape[1]}'
        n = len(X)
        self.write_input(X)
        if not tools.run_make(self.config.output_dir, 'x86sim'):
            return None
        y = self.read_scores()
        return y[:n]

    def write_input(self, X):
        '''Feature-major, W-blocked, four values a line for the 64-bit PLIO'''
        cfg = self.config
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        if n % self.n_samples:
            pad = self.n_samples - (n % self.n_samples)
            logger.info(f'padding {n} samples to {n + pad}: the graph is compiled for '
                        f'{self.n_samples} rows a run')
            X = np.vstack([X, np.zeros((pad, X.shape[1]))])
        if X.shape[1] != self.n_features_padded:
            X = np.hstack([X, np.zeros((len(X), self.n_features_padded - X.shape[1]))])
        xq = self.threshold_p.quantize(X)
        toks = []
        for g in range(0, len(xq), self.W):
            block = xq[g:g + self.W]
            for k in range(self.n_features_padded):
                toks.extend(int(v) for v in block[:, k])
        per_line = 4
        os.makedirs(f'{cfg.output_dir}/data', exist_ok=True)
        with open(f'{cfg.output_dir}/data/x.dat', 'w') as f:
            for i in range(0, len(toks), per_line):
                f.write(' '.join(str(v) for v in toks[i:i + per_line]) + '\n')

    def read_scores(self, filename=None):
        '''Integer scores from the simulator output, dequantized'''
        cfg = self.config
        paths = ([filename] if filename else
                 [f'{cfg.output_dir}/x86simulator_output/data/scores.dat',
                  f'{cfg.output_dir}/data/scores.dat',
                  f'{cfg.output_dir}/aiesimulator_output/data/scores.dat'])
        path = next((p for p in paths if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(f'No scores.dat found, looked in {paths}')
        vals = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('T '):
                    continue
                vals.extend(int(v) for v in line.split())
        return self.score_p.dequantize(np.asarray(vals, dtype=np.int64))

    @copydocstring(ModelBase.build)
    def build(self, **kwargs):
        self.write()
        start = datetime.datetime.now()
        logger.info(f'build starting {start:%H:%M:%S}')
        ok = tools.run_make(self.config.output_dir, 'aiesim')
        stop = datetime.datetime.now()
        logger.info(f'build finished {stop:%H:%M:%S} - took {str(stop - start)}')
        return ok

    def read_report(self) -> dict:
        '''Read whatever stage of report is on disk

        Returns
        ----------
        dictionary of extracted report contents, with a 'stage' key naming what was found
        '''
        return read_aie_report(self.config.output_dir)


def _pad(values, n, fill):
    values = list(values)
    return values + [fill] * (n - len(values))


def _array(ctype, name, values, hexw=None):
    values = list(values)
    if hexw:
        digits = hexw // 4
        suffix = 'ull' if hexw == 64 else 'u'
        toks = [f'0x{int(v) & ((1 << hexw) - 1):0{digits}x}{suffix}' for v in values]
    else:
        toks = [str(int(v)) for v in values]
    lines = [f'constexpr {ctype} {name}[{len(toks)}] = {{']
    for i in range(0, len(toks), 12):
        lines.append('  ' + ', '.join(toks[i:i + 12]) + ',')
    lines.append('};\n')
    return '\n'.join(lines)


def _pad_tree_features(tree, n_features):
    '''Widen every weight row with zero columns, which no split references'''
    t = copy.copy(tree)
    rows = np.asarray(tree.weight, dtype=np.float64)
    wide = np.zeros((rows.shape[0], n_features))
    wide[:, :rows.shape[1]] = rows
    t.weight = wide.tolist()
    return t


def make_model(ensembleDict, config):
    return AIEModel(ensembleDict, config)


def auto_config(granularity='simple'):
    '''Create an initial configuration for the aie backend'''
    config = {'Backend': 'aie',
              'ProjectName': 'my_prj',
              'OutputDir': 'my-conifer-prj',
              'Precision': _DEFAULT_COMPARE,
              'ScorePrecision': _DEFAULT_SCORE,
              'Priority': 'latency',
              'NTiles': AUTO,
              'SplitAxis': AUTO,
              'VectorWidth': AUTO,
              'Tau': AUTO,
              'NSamples': AUTO,
              'XilinxPart': 'xcve2802-vsvh1760-2MP-e-S',
              }
    return config
