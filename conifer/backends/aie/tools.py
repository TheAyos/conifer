import os
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


def get_aiecompiler():
    return get_tool_exe_in_path('aiecompiler')


def get_x86simulator():
    return get_tool_exe_in_path('x86simulator')


def get_aiesimulator():
    return get_tool_exe_in_path('aiesimulator')


def require_tools(*tools):
    '''Raise with an actionable message if any tool is not on the path'''
    missing = [t for t in tools if get_tool_exe_in_path(t) is None]
    if missing:
        raise RuntimeError(
            f'Could not find {", ".join(missing)} on the path. Source the Vitis settings '
            f'script (settings64.sh) for a release with AI Engine support, and set '
            f'PLATFORM_REPO_PATHS to a platform repository')


def run_make(output_dir, target, **variables):
    '''Run one target of the project Makefile'''
    tools = {'x86sim_build': ('aiecompiler',),
             'x86sim': ('aiecompiler', 'x86simulator'),
             'aiesim': ('aiecompiler', 'aiesimulator')}.get(target, ())
    require_tools(*tools)
    args = ' '.join(f"{k}='{v}'" for k, v in variables.items() if v is not None)
    cmd = f'make -C {output_dir} {target} {args}'.strip()
    logger.debug(f'Running build with command "{cmd}"')
    success = os.system(cmd)
    if success > 0:
        logger.error(f'make {target} failed, check the output in {output_dir}')
        return False
    return True
