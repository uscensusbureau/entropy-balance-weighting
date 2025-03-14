import numpy as np
import scipy.sparse as sp

import entropy_balance_weighting as wgt


def create_seperable_data_set() -> None:
    """
    Create and reweight a simulated group-specific separable data set.

    These types of data sets give the upper limit on the size of data sets that can be
    reweighted, as the memory requirements are not large when stored in sparse format.
    The routine below also prints out the amount of memory that would be required to
    store this matrix in a dense format.
    """
    n_per_group = 108000
    n_groups = 3144
    n = n_per_group * n_groups
    k = 2
    unit_data = (np.random.random(size=(n, k)) < 0.05).astype(np.float64)
    x = sp.block_diag(
        [sp.csr_array(z) for z in np.split(unit_data, n_groups, axis=0)], format="csr"
    )
    print(x)
    print(
        "Sparse matrix mem (GB)",
        (x.data.nbytes + x.indptr.nbytes + x.indices.nbytes) / 1024**3,
    )
    print("Dense matrix mem (GB)", (x.shape[0] * x.shape[1] * 8) / 1024**3)

    moments = np.tile(np.mean(unit_data, 0), n_groups)
    print(moments)

    weights0 = np.ones(n)
    wgt.setup_logging("seperable.log")
    res = wgt.entropy_balance(
        mean_population_moments=moments,
        x_sample=x,
        weights0=weights0,
        options={"max_steps": 1000},
    )
    print(res.converged)


if __name__ == "__main__":
    create_seperable_data_set()
