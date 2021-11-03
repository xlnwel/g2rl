import importlib


def pkg_str(root_dir, separator, base_name=None):
    if base_name is None:
        return root_dir
    return f'{root_dir}{separator}{base_name}'


def get_package_from_algo(algo, place=0, separator='.'):
    algo = algo.split('-', 1)[place]
    
    pkg = get_package('algo', algo, separator)
    if pkg is None:
        pkg = get_package('distributed', algo, separator)
    return pkg


def get_package(root_dir, base_name=None, separator='.'):
    for i in range(1, 10):
        indexed_root_dir = root_dir if i == 1 else f'{root_dir}{i}'
        pkg = pkg_str(indexed_root_dir, '.', base_name)
        try:
            if importlib.util.find_spec(pkg) is not None:
                pkg = pkg_str(indexed_root_dir, separator, base_name)
                return pkg
        except:
            return None
    return None


def import_module(name=None, pkg=None, algo=None, *, config=None, place=0):
    """ import <name> module from pkg, 
    if pkg is not provided, import <name> module
    according to algo or algorithm in config """
    if pkg is None:
        algo = algo or config['algorithm']
        assert isinstance(algo, str), algo
        pkg = get_package_from_algo(algo=algo, place=place)
        m = importlib.import_module(f'{pkg}.{name}')
    else:
        pkg = get_package(root_dir=pkg, base_name=name)
        m = importlib.import_module(pkg)

    return m


def import_main(module, algo=None, *, config=None):
    algo = algo or config['algorithm']
    assert isinstance(algo, str), algo
    pkg = get_package_from_algo(algo, place={'train': 0, 'eval': -1}[module])
    m = importlib.import_module(f'{pkg}.{module}')

    return m.main
