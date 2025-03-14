from typing import Any as Any
from typing import Callable as Callable
from typing import Optional as Optional
from typing import Tuple as Tuple
from typing import TypeVar as TypeVar
from typing import Union as Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray as NDArray

SparseArr = Union[sp.csc_array, sp.csr_array]
FArr = NDArray[np.floating]
AnyArray = Union[FArr, SparseArr]
