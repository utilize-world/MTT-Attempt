import imp
import os.path as osp
# imp
import importlib.util


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location(name, pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
    #return imp.load_source('', pathname)


# path_name = osp.join(osp.dirname(__file__), "MTT.py")
# print(osp.exists(path_name))
# print(load(path_name))
