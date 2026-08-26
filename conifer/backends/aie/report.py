import os
import re
import csv
import glob
import json
import xml.etree.ElementTree as ET
import numpy as np
import logging
logger = logging.getLogger(__name__)

_TS = re.compile(r'^T\s+(\S+)\s*(\S*)\s*$')
# aiesimulator mixes units within one file, so the unit column has to be read.
_UNIT_NS = {'': 1.0, 's': 1e9, 'ms': 1e6, 'us': 1e3, 'ns': 1.0, 'ps': 1e-3}
_RUNTIME = {'main', '_main_no_exit_init', '_main_init', '_start'}

_NEXT_STAGE = {
    'write': 'run compile() to map the design and get its real tile memory',
    'compile': 'run build() for cycle counts and latency',
    'build': None,
}


def read_plio(path):
    '''Timestamps in ns and integer values from a PLIO .dat'''
    ts, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line == 'TLAST':
                continue
            m = _TS.match(line)
            if m:
                ts.append(float(m.group(1)) * _UNIT_NS[m.group(2)])
            else:
                vals.extend(int(t) for t in line.split())
    ts = np.asarray(ts)
    if np.any(np.diff(ts) < 0):
        raise RuntimeError(f'{path}: timestamps go backwards, a unit was mishandled')
    return ts, np.asarray(vals, dtype=np.int64)


def _cores(sim_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(sim_dir, 'profile_funct_*.xml'))):
        root = ET.parse(p).getroot()
        if int(root.findtext('total_cycle_count') or 0) <= 0:
            continue
        fns = []
        for f in root.iter('function'):
            name = (f.findtext('function_name') or '').strip()
            cyc = int(f.findtext('function_and_descendants_time/total_cycle_count') or 0)
            if name and name not in _RUNTIME and cyc > 0:
                fns.append((name, int(f.findtext('calls') or 0), cyc))
        if not fns:
            continue
        name, calls, cyc = max(fns, key=lambda f: f[2])
        col, row = os.path.basename(p)[len('profile_funct_'):-len('.xml')].split('_')
        out.append({'col': int(col), 'row': int(row), 'name': name, 'calls': calls,
                    'cyc': cyc, 'total': int(root.findtext('total_cycle_count'))})
    return out


def _map_sections(path):
    with open(path) as f:
        blocks = [b for b in re.split(r'\n\s*\n', f.read()) if b.strip()]
    out = {}
    for b in blocks:
        rows = list(csv.DictReader(b.strip().splitlines()))
        if rows:
            out[next(iter(rows[0]))] = rows
    return out


def _tile(s):
    m = re.fullmatch(r'\((\d+),\s*(\d+)\)', s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def _sizes(path, key):
    out, core = {}, None
    pat = re.compile(key + r'\s*=\s*(\d+)')
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r'Core (\S+)', line)
            if m:
                core = m.group(1)
            m = pat.search(line)
            if m and core:
                out[core] = int(m.group(1))
    return out


def _compile_metrics(work_dir, report):
    map_path = os.path.join(os.path.dirname(work_dir), 'Map_Report.csv')
    if os.path.exists(map_path):
        mr = _map_sections(map_path)
        per_tile = {}
        for r in mr.get('BUFFER', []):
            t = _tile(r.get('MEMORY_GROUP', ''))
            if t:
                per_tile[t] = per_tile.get(t, 0) + int(r['SIZE'])
        if per_tile:
            report['tile_memory_bytes'] = dict(sorted(per_tile.items()))
            report['tile_memory_bytes_max'] = max(per_tile.values())
        clusters = mr.get('CLUSTER', [])
        if clusters:
            report['placement'] = [(c.get('CLUSTER'), _tile(c.get('TILE', '')))
                                   for c in clusters]
            report['n_tiles'] = len(clusters)

    reports = os.path.join(work_dir, 'reports')
    pm = _sizes(os.path.join(reports, 'report_pm.txt'), r'PM Size Used')
    if pm:
        report['program_memory_bytes'] = max(pm.values())
    heap = _sizes(os.path.join(reports, 'report_heap.txt'),
                  r'Heap Size Used \(after alignment\)')
    if heap:
        report['heap_bytes'] = max(heap.values())
    stack = _sizes(os.path.join(reports, 'report_stack.txt'), r'Stack Size Used')
    if stack:
        report['stack_bytes'] = max(stack.values())


def _build_metrics(sim_dir, meta, report):
    cores = _cores(sim_dir)
    if not cores:
        return
    n_tiles = meta.get('config', {}).get('n_tiles', len(cores))
    split_axis = meta.get('config', {}).get('split_axis', 'tree')
    n_samples = meta.get('n_samples', 0)
    ghz = meta.get('clock_ghz', 1.25)

    report['n_active_cores'] = len(cores)
    if len(cores) != n_tiles:
        logger.warning(f'{len(cores)} cores active but the model declares {n_tiles} tiles; '
                       f'an idle tile would read as a speedup')
    if n_samples:
        per_tile = n_samples / max(1, len(cores)) if split_axis == 'sample' else n_samples
        cps = np.asarray([c['cyc'] for c in cores]) / per_tile
        report['cyc_per_sample'] = float(cps.max())
        report['cyc_per_sample_avg'] = float(cps.mean())
        report['straggler_ratio'] = float(cps.max() / cps.mean()) if cps.mean() else None
        total = max(c['total'] for c in cores)
        report['total_cyc'] = total
        report['throughput_ns_per_sample'] = float(total / ghz / n_samples)

    # One tile, or sample-split: nothing is shared, so a group's residence is the
    # invocation that scored it.
    per_call_ns = np.asarray([c['cyc'] / c['calls'] / ghz for c in cores if c['calls']])
    if per_call_ns.size and (len(cores) == 1 or split_axis == 'sample'):
        report['latency_ss_ns'] = float(per_call_ns.mean())
        report['latency_ss_sd_ns'] = float(per_call_ns.std())
    elif per_call_ns.size:
        report['latency_ss_ns'] = float(per_call_ns.max())
        report['latency_ss_note'] = ('approximated by the straggler invocation: a tree-split '
                                     'graph does not expose the head tile acceptance')

    tp = os.path.join(os.path.dirname(sim_dir), 'throughput_info.json')
    if os.path.exists(tp):
        with open(tp) as f:
            ports = json.load(f).get('plio_throughput', {})
        report['port_throughput_mbps'] = {
            v.get('port name'): float(str(v.get('Throughput', '0')).split()[0])
            for v in ports.values()}


def read_aie_report(output_dir) -> dict:
    '''Read whatever stage of AI Engine report is present

    Never raises for a missing later stage: the returned 'stage' says what was found,
    and 'next_step' says what to run for more.
    '''
    report = {'stage': 'write'}
    meta_path = os.path.join(output_dir, 'aie_model.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        report['estimate'] = meta.get('estimate', {})
        report['n_tiles'] = meta.get('config', {}).get('n_tiles')
        report['estimated_tile_memory_bytes'] = meta.get('memory', {}).get('total')

    for d in ('build_hw', 'build_x86'):
        work = os.path.join(output_dir, d, 'Work')
        if os.path.isdir(work):
            report['stage'] = 'compile'
            _compile_metrics(work, report)
            break

    sim_dir = os.path.join(output_dir, 'build_hw', 'aiesimulator_output')
    if os.path.isdir(sim_dir) and glob.glob(os.path.join(sim_dir, 'profile_funct_*.xml')):
        report['stage'] = 'build'
        _build_metrics(sim_dir, meta, report)

    report['next_step'] = _NEXT_STAGE.get(report['stage'])
    return report
