import sys


import aikit.functional.frontends.torch as torch
import aikit
from aikit.functional.frontends import set_frontend_to_specific_version


from . import ops


tensor = _frontend_array = torch.tensor


# setting to specific version #
# --------------------------- #

if aikit.is_local():
    module = aikit.utils._importlib.import_cache[__name__]
else:
    module = sys.modules[__name__]

set_frontend_to_specific_version(module)
