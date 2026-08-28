import os
import datetime
import logging
logger = logging.getLogger(__name__)

# AIE tools that a project needs, in the order a build uses them.
_AIE_TOOLS = {
    'aiecompiler': 'aiecompiler',
    'x86simulator': 'x86simulator',
    'aiesimulator': 'aiesimulator',
}


def get_tool_exe_in_path(tool):

    if tool not in _AIE_TOOLS.keys():
        return None

    tool_exe = _AIE_TOOLS[tool]

    if os.system('type {} > /dev/null 2>/dev/null'.format(tool_exe)) != 0:
        return None

    return tool_exe


def require_tools(*tools):
    '''Raise with an actionable message if any tool is not on the path'''
    missing = [t for t in tools if get_tool_exe_in_path(t) is None]
    if missing:
        raise RuntimeError(
            f'Could not find {", ".join(missing)} on the path. Source the Vitis settings '
            f'script (settings64.sh) for a release with AI Engine support, and set '
            f'PLATFORM_REPO_PATHS to a platform repository')


def _first_error(log_path):
    '''The first error the tools reported, to say more than "it failed"'''
    try:
        with open(log_path, errors='ignore') as f:
            for line in f:
                if 'ERROR' in line or 'error:' in line:
                    return line.strip()[:300]
    except OSError:
        pass
    return None


def run_make(output_dir, target, **variables):
    '''Run one target of the project Makefile, capturing its output to a log'''
    tools = {'x86sim_build': ('aiecompiler',),
             'x86sim': ('aiecompiler', 'x86simulator'),
             'aiesim': ('aiecompiler', 'aiesimulator')}.get(target, ())
    require_tools(*tools)
    args = ' '.join(f"{k}='{v}'" for k, v in variables.items() if v is not None)
    log_path = os.path.join(output_dir, f'{target}.log')
    cmd = f'make -C {output_dir} {target} {args}'.strip()
    logger.debug(f'Running build with command "{cmd}"')

    start = datetime.datetime.now()
    logger.info(f'{target} starting {start:%H:%M:%S}')
    success = os.system(f'{cmd} > {log_path} 2>&1')
    stop = datetime.datetime.now()
    logger.info(f'{target} finished {stop:%H:%M:%S} - took {str(stop - start)}, '
                f'log in {log_path}')

    if success > 0:
        error = _first_error(log_path)
        logger.error(f'{target} failed, check the log in {log_path}'
                     + (f'. First error: {error}' if error else ''))
        return False
    return True
