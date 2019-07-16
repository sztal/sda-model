"""Estimation functions."""
import warnings
import numpy as np
from .exceptions import HillEstimatorWarning, MomentsEstimatorWarning
from .exceptions import KernelTypeEstimatorWarning, KernelTypeEstimatorError


def estimate_moments_1(data, sort=True):
    """Estimate first moments (Hill estimator) from a data array.

    Parameters
    ----------
    data: (N, ) array_like
        _Numpy_ data array. Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    (N, ) array_like
        numpy array of 1st moments (Hill estimator)
        corresponding to all possible order statistics
        of the dataset.
    """
    if sort:
        data = np.sort(data)[::-1]
    logs_1 = np.log(data)
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    k_vector = np.arange(1, len(data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    return M1

def estimate_moments_2(data, sort=True):
    """Estimate first and second moments from a data array.

    Parameters
    ----------
    data: (N, ) array_like
        _Numpy_ data array. Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    (N, ) array_like
        numpy array of 1st moments (Hill estimator)
        corresponding to all possible order statistics
        of the dataset.
    (N, ) array_like
        numpy array of 2nd moments corresponding to all
        possible order statistics of the dataset.
    """
    if sort:
        data = np.sort(data)[::-1]
    logs_1 = np.log(data)
    logs_2 = (np.log(data))**2
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    k_vector = np.arange(1, len(data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    M2 = (1./k_vector)*logs_2_cumsum \
        - (2.*logs_1[1:]/k_vector)*logs_1_cumsum \
        + logs_2[1:]
    return M1, M2

def estimate_moments_3(data, sort=True):
    """Estimate first, second and third moments from a data array.

    Parameters
    ----------
    data: (N, ) array_like
        _Numpy_ data array. Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    (N, ) array_like
        numpy array of 1st moments (Hill estimator)
        corresponding to all possible order statistics
        of the dataset.
    (N, ) array_like
        numpy array of 2nd moments corresponding to all
        possible order statistics of the dataset.
    (N, ) array_like
        numpy array of 3nd moments corresponding to all
        possible order statistics of the dataset.
    """
    if sort:
        data = np.sort(data)[::-1]
    logs_1 = np.log(data)
    logs_2 = (np.log(data))**2
    logs_3 = (np.log(data))**3
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    logs_3_cumsum = np.cumsum(logs_3[:-1])
    k_vector = np.arange(1, len(data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    M2 = (1./k_vector)*logs_2_cumsum - (2.*logs_1[1:]/k_vector)*logs_1_cumsum\
         + logs_2[1:]
    M3 = (1./k_vector)*logs_3_cumsum - (3.*logs_1[1:]/k_vector)*logs_2_cumsum\
         + (3.*logs_2[1:]/k_vector)*logs_1_cumsum - logs_3[1:]
    # cleaning exceptional cases
    clean_indices = np.where(
        (M2 <= 0) | (M3 == 0) | (np.abs(1.-(M1**2)/M2) < 1e-10) \
        | (np.abs(1.-(M1*M2)/M3) < 1e-10)
    )
    M1[clean_indices] = np.nan
    M2[clean_indices] = np.nan
    M3[clean_indices] = np.nan
    return M1, M2, M3


# Hill estimator --------------------------------------------------------------

def hill_dbs(data, t_bootstrap=0.5, r_bootstrap=500, eps_stop = 1.0,
             verbose=False, diagn_plots=False, sort=True):
    """Double-bootstrap procedure for Hill estimator.

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    t_bootstrap : float
        Parameter controlling the size of the 2nd
        bootstrap. Defined from n2 = n*(t_bootstrap).
    r_bootstrap : int
        Number of bootstrap resamplings for the 1st and 2nd bootstraps.
    eps_stop : float
        Parameter controlling range of AMSE minimization.
        Defined as the fraction of order statistics to consider
        during the AMSE minimization step.
    verbose : int or bool
        Flag controlling bootstrap verbosity.
    diagn_plots : bool
        Flag to switch on/off generation of AMSE diagnostic plots.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    k_star : int
        number of order statistics optimal for estimation
        according to the double-bootstrap procedure.
    x1_arr : (N, ) array_like
        Array of fractions of order statistics used for the
        1st bootstrap sample.
    n1_amse : (N, ) array_like
        Array of AMSE values produced by the 1st bootstrap sample.
    k1_min : float
        Value of fraction of order statistics corresponding
        to the minimum of AMSE for the 1st bootstrap sample.
    max_index1 : int
        Index of the 1st bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter.
    x2_arr : (N, ) array_like
        array of fractions of order statistics used for the
        2nd bootstrap sample.
    n2_amse : (N, ) array_like
        array of AMSE values produced by the 2nd bootstrap sample.
    k2_min : float
        Value of fraction of order statistics corresponding
        to the minimum of AMSE for the 2nd bootstrap sample.
    max_index2 : int
        Index of the 2nd bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter.
    """
    # pylint: disable=too-many-locals,too-many-statements
    if sort:
        data = np.sort(data)[::-1]
    if verbose > 0:
        print("Performing Hill double-bootstrap...")
    n = len(data)
    eps_bootstrap = 0.5*(1+np.log(int(t_bootstrap*n))/np.log(n))
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(n1-1)
    good_counts1 = np.zeros(n1-1)
    k1 = None
    k2 = None
    min_index1 = 1
    min_index2 = 1
    while k2 is None:
        # first bootstrap with n1 sample size
        for _ in range(r_bootstrap):
            sample = np.random.choice(data, n1, replace=True)
            sample[::-1].sort()
            M1, M2 = estimate_moments_2(sample, sort=False)
            current_amse1 = (M2 - 2.*(M1)**2)**2
            samples_n1 += current_amse1
            good_counts1[np.where(current_amse1 != np.nan)] += 1
        averaged_delta = samples_n1 / good_counts1

        max_index1 = (np.abs(np.linspace(1./n1, 1.0, n1) - eps_stop)).argmin()
        # take care of indexing
        k1 = np.nanargmin(averaged_delta[min_index1:max_index1]) + 1 + min_index1
        if diagn_plots:
            n1_amse = averaged_delta
            x1_arr = np.linspace(1./n1, 1.0, n1)

        # second bootstrap with n2 sample size
        n2 = int(n1*n1/float(n))
        samples_n2 = np.zeros(n2-1)
        good_counts2 = np.zeros(n2-1)

        for _ in range(r_bootstrap):
            sample = np.random.choice(data, n2, replace=True)
            sample[::-1].sort()
            M1, M2 = estimate_moments_2(sample, sort=False)
            current_amse2 = (M2 - 2.*(M1**2))**2
            samples_n2 += current_amse2
            good_counts2[np.where(current_amse2 != np.nan)] += 1
        max_index2 = (np.abs(np.linspace(1./n2, 1.0, n2) - eps_stop)).argmin()
        averaged_delta = samples_n2 / good_counts2

        max_index1 = (np.abs(np.linspace(1./n1, 1.0, n1) - eps_stop)).argmin()
        # take care of indexing
        k2 = np.nanargmin(averaged_delta[min_index2:max_index2]) + 1 + min_index2
        if diagn_plots:
            n2_amse = averaged_delta
            x2_arr = np.linspace(1./n2, 1.0, n2)

        if k2 > k1:
            warnings.warn("k2 > k1, AMSE false minimum suspected, resampling ...",
                          HillEstimatorWarning)
            # move left AMSE boundary to avoid numerical issues
            min_index1 = min_index1 + int(0.005*n)
            min_index2 = min_index2 + int(0.005*n)
            k2 = None

    # this constant is provided in the Danielsson's paper
    # use instead of rho below if needed
    # rho = (np.log(k1)/(2.*np.log(n1) - np.log(k1))) \
    #       **(2.*(np.log(n1) - np.log(k1))/(np.log(n1)))

    # this constant is provided in Qi's paper
    rho = (1. - (2*(np.log(k1) - np.log(n1))/(np.log(k1)))) \
          **(np.log(k1)/np.log(n1) - 1.)

    k_star = (k1*k1/float(k2)) * rho
    k_star = int(np.round(k_star))

    # enforce k_star to pick 2nd value (rare cases of extreme cutoffs)
    if k_star == 0:
        k_star = 2
    if int(k_star) >= len(data):
        warnings.warn("estimated threshold k is larger than the size of data",
                      HillEstimatorWarning)
        k_star = len(data)-1
    if verbose > 0:
        print("--- Hill double-bootstrap information ---")
        print("Size of the 1st bootstrap sample n1:", n1)
        print("Size of the 2nd bootstrap sample n2:", n2)
        print("Estimated k1:", k1)
        print("Estimated k2:", k2)
        print("Estimated constant rho:", rho)
        print("Estimated optimal k:", k_star)
        print("-----------------------------------------")
    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    return k_star, x1_arr, n1_amse, k1/float(n1), max_index1, x2_arr, n2_amse, k2/float(n2), max_index2

def hill_estimator(data, bootstrap=True, verbose=False, sort=True, **kwds):
    """Compute Hill estimator.

    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    bootstrap : bool
        Flag to switch on/off double-bootstrap procedure.
    verbose : int or bool
        Flag controlling bootstrap verbosity.
    sort : bool
        Should data be copied and sorted.
    **kwds :
        Keyword arguments passed to :py:func:`hill_dbs`.

    Returns
    -------
    list
        List containing an array of order statistics,
        an array of corresponding tail index estimates,
        the optimal order statistic estimated by double-bootstrap
        and the corresponding tail index,
        an array of fractions of order statistics used for
        the 1st bootstrap sample with an array of corresponding
        AMSE values, value of fraction of order statistics
        corresponding to the minimum of AMSE for the 1st bootstrap
        sample, index of the 1st bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter; and the same characteristics for the
        2nd bootstrap sample.
    """
    # pylint: disable=too-many-locals
    if sort:
        data = np.sort(data)[::-1]
    k_arr = np.arange(1, len(data))
    xi_arr = estimate_moments_1(data, sort=False)
    if bootstrap:
        (k_star, x1_arr, n1_amse, k1, max_index1,
         x2_arr, n2_amse, k2, max_index2) = \
            hill_dbs(data, verbose=verbose, **kwds)
        while k_star is None:
            if verbose > 0:
                print("Resampling...")
            (k_star, x1_arr, n1_amse, k1, max_index1,
             x2_arr, n2_amse, k2, max_index2) = \
                 hill_dbs(data, verbose=verbose, **kwds)
        xi_star = xi_arr[k_star-1]
        if verbose > 0:
            print("Adjusted Hill estimated gamma:", 1 + 1./xi_star)
            print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, k1, max_index1 = 4*[None]
        x2_arr, n2_amse, k2, max_index2 = 4*[None]
    results = [k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1, max_index1,\
               x2_arr, n2_amse, k2, max_index2]
    return results

def smooth_hill_estimator(data, r_smooth=2, sort=True):
    """Compute smooth Hill estimator.

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    sort : bool
        Should data be copied and sorted.
    r_smooth : float
        Integer parameter controlling the width of smoothing window.
        Typically small value such as 2 or 3.

    Returns
    -------
    k_arr : (N, ) array_like
        _Numpy_ array of order statistics based on the data provided.
    xi_arr : (N, ) array_like
        _Numpy_ array of tail index estimates corresponding to
        the order statistics array k_arr.
    """
    if sort:
        data = np.sort(data)[::-1]
    n = len(data)
    M1 = estimate_moments_1(data, sort=False)
    xi_arr = np.zeros(int(np.floor(float(n)/r_smooth)))
    k_arr = np.arange(1, int(np.floor(float(n)/r_smooth))+1)
    xi_arr[0] = M1[0]
    bin_lengths = np.array([1.]+[float((r_smooth-1)*k) for k in k_arr[:-1]])
    cum_sum = 0.0
    for i in range(1, r_smooth*int(np.floor(float(n)/r_smooth))-1):
        k = i
        cum_sum += M1[k]
        if (k+1) % (r_smooth) == 0:
            xi_arr[int(k+1)//int(r_smooth)] = cum_sum
            cum_sum -= M1[int(k+1)//int(r_smooth)]
    xi_arr = xi_arr/bin_lengths
    return k_arr, xi_arr


# Moments Tail Index estimator ------------------------------------------------

def moments_dbs_prefactor(xi_n, n1, k1):
    """Calculate pre-factor used in moments double-bootstraping procedure.

    Parameters
    ----------
    xi_n : float
        Moments tail index estimate corresponding to
        sqrt(n)-th order statistic.
    n1 : int
        Size of the 1st bootstrap in double-bootstrap procedure.
    k1 : int
        Estimated optimal order statistic based on the 1st bootstrap sample.

    Returns
    -------
    float
        Constant used in estimation of the optimal
        stopping order statistic for moments estimator.
    """
    def V_sq(xi_n):
        if xi_n >= 0:
            V = 1. + (xi_n)**2
            return V
        a = (1.-xi_n)**2
        b = (1-2*xi_n)*(6*((xi_n)**2)-xi_n+1)
        c = (1.-3*xi_n)*(1-4*xi_n)
        V = a*b/c
        return V

    def V_bar_sq(xi_n):
        if xi_n >= 0:
            V = 0.25*(1+(xi_n)**2)
            return V
        a = 0.25*((1-xi_n)**2)
        b = 1-8*xi_n+48*(xi_n**2)-154*(xi_n**3)
        c = 263*(xi_n**4)-222*(xi_n**5)+72*(xi_n**6)
        d = (1.-2*xi_n)*(1-3*xi_n)*(1-4*xi_n)
        e = (1.-5*xi_n)*(1-6*xi_n)
        V = a*(b+c)/(d*e)
        return V

    def b(xi_n, rho):
        if xi_n < rho:
            a1 = (1.-xi_n)*(1-2*xi_n)
            a2 = (1.-rho-xi_n)*(1.-rho-2*xi_n)
            return a1/a2
        if xi_n >= rho and xi_n < 0:
            return 1./(1-xi_n)
        b = (xi_n/(rho*(1.-rho))) + (1./((1-rho)**2))
        return b

    def b_bar(xi_n, rho):
        if xi_n < rho:
            a1 = 0.5*(-rho*(1-xi_n)**2)
            a2 = (1.-xi_n-rho)*(1-2*xi_n-rho)*(1-3*xi_n-rho)
            return a1/a2
        if xi_n >= rho and xi_n < 0:
            a1 = 1-2*xi_n-np.sqrt((1-xi_n)*(1-2.*xi_n))
            a2 = (1.-xi_n)*(1-2*xi_n)
            return a1/a2
        b = (-1.)*((rho + xi_n*(1-rho))/(2*(1-rho)**3))
        return b

    rho = np.log(k1)/(2*np.log(k1) - 2.*np.log(n1))
    a = (V_sq(xi_n)) * (b_bar(xi_n, rho)**2)
    b = V_bar_sq(xi_n) * (b(xi_n, rho)**2)
    prefactor = (a/b)**(1./(1. - 2*rho))
    return prefactor


def moments_dbs(data, xi_n, t_bootstrap=0.5, r_bootstrap=500, eps_stop=1.0,
                verbose=False, diagn_plots=False, sort=True):
    """Double-bootstrap procedure for moments estimator.

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    xi_n : float
        Moments tail index estimate corresponding to
        sqrt(n)-th order statistic.
    t_bootstrap : float
        Parameter controlling the size of the 2nd
        bootstrap. Defined from n2 = n*(t_bootstrap).
    r_bootstrap : int
        Number of bootstrap resamplings for the 1st and 2nd bootstraps.
    eps_stop : float
        Parameter controlling range of AMSE minimization.
        Defined as the fraction of order statistics to consider
        during the AMSE minimization step.
    verbose : int or bool
        Flag controlling bootstrap verbosity.
    diagn_plots : bool
        Flag to switch on/off generation of AMSE diagnostic plots.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    k_star : int
        number of order statistics optimal for estimation
        according to the double-bootstrap procedure.
    x1_arr : (N, ) array_like
        Array of fractions of order statistics used for the
        1st bootstrap sample.
    n1_amse : (N, ) array_like
        Array of AMSE values produced by the 1st bootstrap sample.
    k1_min : float
        Value of fraction of order statistics corresponding
        to the minimum of AMSE for the 1st bootstrap sample.
    max_index1 : int
        Index of the 1st bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter.
    x2_arr : (N, ) array_like
        array of fractions of order statistics used for the
        2nd bootstrap sample.
    n2_amse : (N, ) array_like
        array of AMSE values produced by the 2nd bootstrap sample.
    k2_min : float
        Value of fraction of order statistics corresponding
        to the minimum of AMSE for the 2nd bootstrap sample.
    max_index2 : int
        Index of the 2nd bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter.
    """
    # pylint: disable=too-many-locals,too-many-statements
    if sort:
        data = np.sort(data)[::-1]
    if verbose > 0:
        print("Performing moments double-bootstrap ...")
    n = len(data)
    eps_bootstrap = 0.5*(1+np.log(int(t_bootstrap*n))/np.log(n))

    # first bootstrap with n1 sample size
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(n1-1)
    good_counts1 = np.zeros(n1-1)
    for _ in range(r_bootstrap):
        sample = np.random.choice(data, n1, replace=True)
        sample[::-1].sort()
        M1, M2, M3 = estimate_moments_3(sample, sort=False)
        xi_2 = M1 + 1. - 0.5*((1. - (M1*M1)/M2))**(-1.)
        xi_3 = np.sqrt(0.5*M2) + 1. - (2./3.)*(1. / (1. - M1*M2/M3))
        samples_n1 += (xi_2 - xi_3)**2
        good_counts1[np.where((xi_2 - xi_3)**2 != np.nan)] += 1
    max_index1 = (np.abs(np.linspace(1./n1, 1.0, n1) - eps_stop)).argmin()
    averaged_delta = samples_n1 / good_counts1
    k1 = np.nanargmin(averaged_delta[:max_index1]) + 1 #take care of indexing
    if diagn_plots:
        n1_amse = averaged_delta
        x1_arr = np.linspace(1./n1, 1.0, n1)

    # r second bootstrap with n2 sample size
    n2 = int(n1*n1/float(n))
    samples_n2 = np.zeros(n2-1)
    good_counts2 = np.zeros(n2-1)
    for _ in range(r_bootstrap):
        sample = np.random.choice(data, n2, replace=True)
        sample[::-1].sort()
        M1, M2, M3 = estimate_moments_3(sample, sort=False)
        xi_2 = M1 + 1. - 0.5*(1. - (M1*M1)/M2)**(-1.)
        xi_3 = np.sqrt(0.5*M2) + 1. - (2./3.)*(1. / (1. - M1*M2/M3))
        samples_n2 += (xi_2 - xi_3)**2
        good_counts2[np.where((xi_2 - xi_3)**2 != np.nan)] += 1
    max_index2 = (np.abs(np.linspace(1./n2, 1.0, n2) - eps_stop)).argmin()
    averaged_delta = samples_n2 / good_counts2
    k2 = np.nanargmin(averaged_delta[:max_index2]) + 1 #take care of indexing
    if diagn_plots:
        n2_amse = averaged_delta
        x2_arr = np.linspace(1./n2, 1.0, n2)
    if k2 > k1:
        warnings.warn(
            "estimated k2 is greater than k1! Re-doing bootstrap ...",
            MomentsEstimatorWarning
        )
        return 9*[None]

    #calculate estimated optimal stopping k
    prefactor = moments_dbs_prefactor(xi_n, n1, k1)
    k_star = int((k1*k1/float(k2)) * prefactor)

    if int(k_star) >= len(data):
        warnings.warn(
            "estimated threshold k is larger than the size of data",
            MomentsEstimatorWarning
        )
        k_star = len(data) - 1
    if verbose > 0:
        print("--- Moments double-bootstrap information ---")
        print("Size of the 1st bootstrap sample n1:", n1)
        print("Size of the 2nd bootstrap sample n2:", n2)
        print("Estimated k1:", k1)
        print("Estimated k2:", k2)
        print("Estimated constant:", prefactor)
        print("Estimated optimal k:", k_star)
        print("--------------------------------------------")
    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    return k_star, x1_arr, n1_amse, k1/float(n1), max_index1, x2_arr, n2_amse, k2/float(n2), max_index2

def moments_estimator(data, bootstrap=True, sort=True, verbose=False, **kwds):
    """Compute moments estimator.

    If bootstrap flag is ``True``, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    bootstrap : bool
        Flag to switch on/off double-bootstrap procedure.
    verbose : int or bool
        Flag controlling bootstrap verbosity.
    sort : bool
        Should data be copied and sorted.
    **kwds :
        Keyword arguments passed to :py:func:`moments_dbs`.

    Returns
    -------
    list
        Listcontaining an array of order statistics,
        an array of corresponding tail index estimates,
        the optimal order statistic estimated by double-
        bootstrap and the corresponding tail index,
        an array of fractions of order statistics used for
        the 1st bootstrap sample with an array of corresponding
        AMSE values, value of fraction of order statistics
        corresponding to the minimum of AMSE for the 1st bootstrap
        sample, index of the 1st bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter; and the same characteristics for the
        2nd bootstrap sample.
    """
    if sort:
        data = np.sort(data)[::-1]
    n =  len(data)
    M1, M2 = estimate_moments_2(data, sort=False)
    xi_arr = M1 + 1. - 0.5*(1. - (M1*M1)/M2)**(-1)
    k_arr = np.arange(1, len(data))
    if bootstrap:
        xi_n = xi_arr[int(np.floor(n**0.5))-1]
        results = moments_dbs(data, xi_n, **kwds)

        while results[0] is None:
            if verbose > 0:
                print("Resampling...")
            results = moments_dbs(data, xi_n, **kwds)
        (k_star, x1_arr, n1_amse, k1, max_index1,
         x2_arr, n2_amse, k2, max_index2) = results
        xi_star = xi_arr[k_star-1]
        if verbose > 0:
            if xi_star <= 0:
                print ("Moments estimated gamma: infinity (xi <= 0).")
            else:
                print ("Moments estimated gamma:", 1 + 1./xi_star)
            print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, k1, max_index1 = 4*[None]
        x2_arr, n2_amse, k2, max_index2 = 4*[None]
    results = [
        k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1,
        max_index1, x2_arr, n2_amse, k2, max_index2
    ]
    return results


# Kernel-type Tail Index estimator --------------------------------------------

def get_biweight_kernel_estimates(data, hsteps, alpha, sort=True):
    """Calculate biweight kernel-type estimates for tail index.

    Biweight kernel is defined as:
    ``phi(u) = (15/8) * (1 - u^2)^2``

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    hsteps : int
        Parameter controlling number of bandwidth steps
        of the kernel-type estimator.
    alpha : float
        Parameter controlling the amount of "smoothing"
        for the kernel-type estimator.
        Should be greater than 0.5.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    h_arr : (N, ) array_like
        _Numpy_ array of fractions of order statistics included
        in kernel-type tail index estimation.
    xi_arr: (N, ) array_like
        _Numpy_ array with tail index estimated corresponding
        to different fractions of order statistics included
        listed in h_arr array.
    """
    # pylint: disable=too-many-locals
    if sort:
        data = np.sort(data)[::-1]
    n = len(data)
    logs = np.log(data)
    differences = logs[:-1] - logs[1:]
    i_arr = np.arange(1, n)/float(n)
    i3_arr = i_arr**3
    i5_arr = i_arr**5
    i_alpha_arr = i_arr**alpha
    i_alpha2_arr = i_arr**(2.+alpha)
    i_alpha4_arr = i_arr**(4.+alpha)
    t1 = np.cumsum(i_arr*differences)
    t2 = np.cumsum(i3_arr*differences)
    t3 = np.cumsum(i5_arr*differences)
    t4 = np.cumsum(i_alpha_arr*differences)
    t5 = np.cumsum(i_alpha2_arr*differences)
    t6 = np.cumsum(i_alpha4_arr*differences)
    h_arr = np.logspace(np.log10(1./n), np.log10(1.0), hsteps)
    max_i_vector = (np.floor((n*h_arr))-2.).astype(int)
    gamma_pos = (15./(8*h_arr))*t1[max_i_vector]\
                - (15./(4*(h_arr**3)))*t2[max_i_vector]\
                + (15./(8*(h_arr**5)))*t3[max_i_vector]

    q1 = (15./(8*h_arr))*t4[max_i_vector]\
         + (15./(8*(h_arr**5)))*t6[max_i_vector]\
         - (15./(4*(h_arr**3)))*t5[max_i_vector]

    q2 = (15.*(1+alpha)/(8*h_arr))*t4[max_i_vector]\
         + (15.*(5+alpha)/(8*(h_arr**5)))*t6[max_i_vector]\
         - (15.*(3+alpha)/(4*(h_arr**3)))*t5[max_i_vector]

    xi_arr = gamma_pos -1. + q2/q1
    return h_arr, xi_arr

def get_triweight_kernel_estimates(data, hsteps, alpha, sort=True):
    """Calculate triweight kernel-type estimates for tail index.

    Triweight kernel is defined as:
    ``phi(u) = (35/16) * (1 - u^2)^3``

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    hsteps : int
        Parameter controlling number of bandwidth steps
        of the kernel-type estimator.
    alpha : float
        Parameter controlling the amount of "smoothing"
        for the kernel-type estimator.
        Should be greater than 0.5.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    h_arr : (N, ) array_like
        _Numpy_ array of fractions of order statistics included
        in kernel-type tail index estimation.
    xi_arr : (N, ) array_like
        _Numpy_ array with tail index estimated corresponding
        to different fractions of order statistics included
        listed in h_arr array.
    """
    # pylint: disable=too-many-locals
    if sort:
        data = np.sort(data)[::-1]
    n = len(data)
    logs = np.log(data)
    differences = logs[:-1] - logs[1:]
    i_arr = np.arange(1, n)/float(n)
    i3_arr = i_arr**3
    i5_arr = i_arr**5
    i7_arr = i_arr**7
    i_alpha_arr = i_arr**alpha
    i_alpha2_arr = i_arr**(2.+alpha)
    i_alpha4_arr = i_arr**(4.+alpha)
    i_alpha6_arr = i_arr**(6.+alpha)
    t1 = np.cumsum(i_arr*differences)
    t2 = np.cumsum(i3_arr*differences)
    t3 = np.cumsum(i5_arr*differences)
    t4 = np.cumsum(i7_arr*differences)
    t5 = np.cumsum(i_alpha_arr*differences)
    t6 = np.cumsum(i_alpha2_arr*differences)
    t7 = np.cumsum(i_alpha4_arr*differences)
    t8 = np.cumsum(i_alpha6_arr*differences)
    h_arr = np.logspace(np.log10(1./n), np.log10(1.0), hsteps)
    max_i_vector = (np.floor((n*h_arr))-2.).astype(int)

    gamma_pos = (35./(16*h_arr))*t1[max_i_vector]\
                - (105./(16*(h_arr**3)))*t2[max_i_vector]\
                + (105./(16*(h_arr**5)))*t3[max_i_vector]\
                - (35./(16*(h_arr**7)))*t4[max_i_vector]

    q1 = (35./(16*h_arr))*t5[max_i_vector]\
         + (105./(16*(h_arr**5)))*t7[max_i_vector]\
         - (105./(16*(h_arr**3)))*t6[max_i_vector]\
         - (35./(16*(h_arr**7)))*t8[max_i_vector]

    q2 = (35.*(1+alpha)/(16*h_arr))*t5[max_i_vector] \
        + (105.*(5+alpha)/(16*(h_arr**5)))*t7[max_i_vector] \
        - (105.*(3+alpha)/(16*(h_arr**3)))*t6[max_i_vector] \
        - (35.*(7+alpha)/(16*(h_arr**7)))*t8[max_i_vector]

    xi_arr = gamma_pos - 1. + q2/q1
    return h_arr, xi_arr

def kernel_type_dbs(data, hsteps, alpha=0.6, t_bootstrap=0.5, r_bootstrap=500,
                    eps_stop=1.0, verbose=False, diagn_plots=False, sort=True):
    """Double-bootstrap for Kernel-type Tail Index estimator.

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    hsteps : int
        Parameter controlling number of bandwidth steps
        of the kernel-type estimator.
    alpha : float
        Parameter controlling the amount of "smoothing"
        for the kernel-type estimator.
        Should be greater than 0.5.
    t_bootstrap : float
        Parameter controlling the size of the 2nd
        bootstrap. Defined from n2 = n*(t_bootstrap).
    r_bootstrap : int
        Number of bootstrap resamplings for the 1st and 2nd bootstraps.
    eps_stop : float
        Parameter controlling range of AMSE minimization.
        Defined as the fraction of order statistics to consider
        during the AMSE minimization step.
    verbose : int or bool
        Flag controlling bootstrap verbosity.
    diagn_plots : bool
        Flag to switch on/off generation of AMSE diagnostic plots.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    h_star : float
        Fraction of order statistics optimal for estimation
        according to the double-bootstrap procedure.
    x1_arr : (N, ) array_like
        Array of fractions of order statistics used for the
        1st bootstrap sample.
    n1_amse : (N, ) array_like
        Array of AMSE values produced by the 1st bootstrap sample.
    h1 : float
        Value of fraction of order statistics corresponding
        to the minimum of AMSE for the 1st bootstrap sample.
    max_k_index1 : int
        Index of the 1st bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter.
    x2_arr : (N, ) array_like
        Array of fractions of order statistics used for the
        2nd bootstrap sample.
    n2_amse : (N, ) array_like
        Array of AMSE values produced by the 2nd bootstrap sample.
    h2 : float
        Value of fraction of order statistics corresponding
        to the minimum of AMSE for the 2nd bootstrap sample.
    max_k_index2 : int
        Index of the 2nd bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter.
    """
    if sort:
        data = np.sort(data)[::-1]
    if verbose:
        print("Performing kernel double-bootstrap...")
    n = len(data)
    eps_bootstrap = 0.5*(1+np.log(int(t_bootstrap*n))/np.log(n))

    # first bootstrap with n1 sample size
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(hsteps)
    good_counts1 = np.zeros(hsteps)
    for _ in range(r_bootstrap):
        sample = np.random.choice(data, n1, replace = True)
        sample[::-1].sort()
        _, xi2_arr = get_biweight_kernel_estimates(sample, hsteps, alpha)
        _, xi3_arr = get_triweight_kernel_estimates(sample, hsteps, alpha)
        samples_n1 += (xi2_arr - xi3_arr)**2
        good_counts1[np.where((xi2_arr - xi3_arr)**2 != np.nan)] += 1
    max_index1 = \
        (np.abs(np.logspace(np.log10(1./n1), np.log10(1.0), hsteps) - eps_stop)) \
        .argmin()
    x1_arr = np.logspace(np.log10(1./n1), np.log10(1.0), hsteps)
    averaged_delta = samples_n1 / good_counts1
    h1 = x1_arr[np.nanargmin(averaged_delta[:max_index1])]
    if diagn_plots:
        n1_amse = averaged_delta

    # second bootstrap with n2 sample size
    n2 = int(n1*n1/float(n))
    if n2 < hsteps:
        msg = "Number of h points is larger than number " \
            + "of order statistics! Please either increase " \
            + "the size of 2nd bootstrap or decrease number " \
            + "of h grid points."
        raise KernelTypeEstimatorError(msg)
    samples_n2 = np.zeros(hsteps)
    good_counts2 = np.zeros(hsteps)
    for _ in range(r_bootstrap):
        sample = np.random.choice(data, n2, replace = True)
        sample[::-1].sort()
        _, xi2_arr = get_biweight_kernel_estimates(sample, hsteps, alpha)
        _, xi3_arr = get_triweight_kernel_estimates(sample, hsteps, alpha)
        samples_n2 += (xi2_arr - xi3_arr)**2
        good_counts2[np.where((xi2_arr - xi3_arr)**2 != np.nan)] += 1
    max_index2 = (np.abs(np.logspace(np.log10(1./n2), np.log10(1.0), hsteps) - eps_stop)).argmin()
    x2_arr = np.logspace(np.log10(1./n2), np.log10(1.0), hsteps)
    averaged_delta = samples_n2 / good_counts2
    h2 = x2_arr[np.nanargmin(averaged_delta[:max_index2])]
    if diagn_plots:
        n2_amse = averaged_delta

    A = (143.*((np.log(n1) + np.log(h1))**2) \
        /(3*(np.log(n1) - 13. * np.log(h1))**2)) \
        **(-np.log(h1)/np.log(n1))

    h_star = (h1*h1/float(h2)) * A

    if h_star > 1:
        msg = "estimated threshold is larger than the size of data! " \
            + "Optimal 'h' is set to 1 ..."
        warnings.warn(msg, KernelTypeEstimatorWarning)
        h_star = 1.

    if verbose:
        print("--- Kernel-type double-bootstrap information ---")
        print("Size of the 1st bootstrap sample n1:", n1)
        print("Size of the 2nd bootstrap sample n2:", n2)
        print("Estimated h1:", h1)
        print("Estimated h2:", h2)
        print("Estimated constant A:", A)
        print("Estimated optimal h:", h_star)
        print("------------------------------------------------")
    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    if x1_arr is not None:
        max_k_index1 = x1_arr[max_index1]
    else:
        max_k_index1 = None
    if x2_arr is not None:
        max_k_index2 = x2_arr[max_index2]
    else:
        max_k_index2 = None
    return h_star, x1_arr, n1_amse, h1, max_k_index1, x2_arr, n2_amse, h2, max_k_index2

def kernel_type_estimator(data, hsteps=200, alpha=0.6, bootstrap=True,
                          verbose=False, sort=True, **kwds):
    """Compute Kernelt-type estimator.

    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

     Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    hsteps : int
        Parameter controlling number of bandwidth steps
        of the kernel-type estimator.
    alpha : float
        Parameter controlling the amount of "smoothing"
        for the kernel-type estimator.
        Should be greater than 0.5.
    bootstrap : bool
        Flag to switch on/off double-bootstrap procedure.
    sort : bool
        Should data be copied and sorted.
    **kwds :
        Keyword arguments passed to :py:func:`kernel_type_dbs`.

    Returns
    -------
    list
        List containing an array of fractions of order statistics,
        an array of corresponding tail index estimates,
        the optimal order statistic estimated by double-
        bootstrap and the corresponding tail index,
        an array of fractions of order statistics used for
        the 1st bootstrap sample with an array of corresponding
        AMSE values, value of fraction of order statistics
        corresponding to the minimum of AMSE for the 1st bootstrap
        sample, index of the 1st bootstrap sample's order statistics
        array corresponding to the minimization boundary set
        by eps_stop parameter; and the same characteristics for the
        2nd bootstrap sample.
    """
    if sort:
        data = np.sort(data)[::-1]
    n = len(data)
    h_arr, xi_arr = get_biweight_kernel_estimates(data, hsteps, alpha=alpha)
    if bootstrap:
        results = kernel_type_dbs(data, hsteps, alpha=alpha, verbose=verbose,
                                  sort=False, **kwds)
        (h_star, x1_arr, n1_amse, h1, max_index1,
         x2_arr, n2_amse, h2, max_index2) = results
        while h_star is None:
            if verbose > 0:
                print("Resampling...")
            results = kernel_type_dbs(data, hsteps, alpha=alpha, verbose=verbose,
                                      sort=False, **kwds)
            (h_star, x1_arr, n1_amse, h1, max_index1,
             x2_arr, n2_amse, h2, max_index2) = results

        #get k index which corresponds to h_star
        k_star = np.argmin(np.abs(h_arr - h_star))
        xi_star = xi_arr[k_star]
        k_arr = []
        k_star = int(np.floor(h_arr[k_star]*n))-1
        k_arr = np.floor((h_arr * n))
        if verbose > 0:
            if xi_star <= 0:
                print ("Kernel-type estimated gamma: infinity (xi <= 0).")
            else:
                print ("Kernel-type estimated gamma:", 1 + 1./xi_star)
            print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, h1, max_index1 = 4*[None]
        x2_arr, n2_amse, h2, max_index2 = 4*[None]
        k_arr = np.floor(h_arr * n)
    results = [
        np.array(k_arr), xi_arr, k_star, xi_star, x1_arr, n1_amse, h1,
        max_index1, x2_arr, n2_amse, h2, max_index2
    ]
    return results
