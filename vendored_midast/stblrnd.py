from typing import Union, Tuple, List
import numpy as np


def stblrnd(alpha: float, beta: float, gamma: float, delta: float, size: Union[None, Tuple[int, ...]] = None) -> np.ndarray:
    """
    Generate random samples from an alpha-stable distribution.

    :param alpha: Stability parameter (0 < alpha <= 2).
    :param beta: Skewness parameter (-1 <= beta <= 1).
    :param gamma: Scale parameter (gamma >= 0).
    :param delta: Location parameter.
    :param size: Shape of the output array. Default is a scalar.
    :return: Array of samples from the alpha-stable distribution.
    """
    if alpha <= 0 or alpha > 2 or not np.isscalar(alpha):
        raise ValueError("alpha must be a scalar which lies in the interval (0,2]")
    if abs(beta) > 1 or not np.isscalar(beta):
        raise ValueError("beta must be a scalar which lies in the interval [-1,1]")
    if gamma < 0 or not np.isscalar(gamma):
        raise ValueError("gamma must be a non-negative scalar")
    if not np.isscalar(delta):
        raise ValueError("delta must be a scalar")

    if size is None:
        size = ()
    else:
        size = tuple(size)

    if alpha == 2:  # Gaussian distribution
        r = np.sqrt(2) * np.random.randn(*size)
    elif alpha == 1 and beta == 0:  # Cauchy distribution
        r = np.tan(np.pi / 2 * (2 * np.random.rand(*size) - 1))
    elif alpha == 0.5 and abs(beta) == 1:  # Levy distribution (a.k.a. Pearson V)
        r = beta / np.random.randn(*size) ** 2
    elif beta == 0:  # Symmetric alpha-stable
        V = np.pi / 2 * (2 * np.random.rand(*size) - 1)
        W = -np.log(np.random.rand(*size))
        r = (
            np.sin(alpha * V)
            / np.cos(V) ** (1 / alpha)
            * np.cos((1 - alpha) * V - np.arctan(beta * np.tan(np.pi * alpha / 2)))
            / W ** ((1 - alpha) / alpha)
        )
    elif alpha != 1:  # General case, alpha not 1
        V = np.pi / 2 * (2 * np.random.rand(*size) - 1)
        W = -np.log(np.random.rand(*size))
        const = beta * np.tan(np.pi * alpha / 2)
        B = np.arctan(const)
        S = (1 + const**2) ** (1 / (2 * alpha))
        r = (
            S
            * np.sin(alpha * V + B)
            / np.cos(V) ** (1 / alpha)
            * np.cos((1 - alpha) * V - B)
            / W ** ((1 - alpha) / alpha)
        )
    else:  # General case, alpha = 1
        V = np.pi / 2 * (2 * np.random.rand(*size) - 1)
        W = -np.log(np.random.rand(*size))
        piover2 = np.pi / 2
        sclshftV = piover2 + beta * V
        r = 1 / piover2 * (sclshftV * np.tan(V) - beta * np.log((piover2 * W * np.cos(V)) / sclshftV))

    if alpha != 1:
        r = gamma * r + delta
    else:
        r = gamma * r + (2 / np.pi) * beta * gamma * np.log(gamma) + delta

    return r


def sub_gaussian(alpha: float, d: int) -> np.ndarray:
    """
    Generate a sub-Gaussian random vector.

    :param alpha: Stability parameter for the alpha-stable distribution.
    :param d: Dimension of the Gaussian vector.
    :return: Sub-Gaussian random vector.
    """
    G = np.random.normal(0, 1, d)
    A = stblrnd(
        alpha=alpha / 2,
        gamma=(np.cos(np.pi * alpha / 4)) ** (2 / alpha),
        beta=1,
        delta=0,
        size=(1,),
    )
    return np.sqrt(A) * G


def sub_gaussian_vect(alpha: float, d: int, n: int) -> np.ndarray:
    """
    Generate multiple sub-Gaussian random vectors.

    :param alpha: Stability parameter for the alpha-stable distribution.
    :param d: Dimension of each Gaussian vector.
    :param n: Number of vectors to generate.
    :return: Array of sub-Gaussian random vectors.
    """
    return np.array([sub_gaussian(alpha, d) for _ in range(n)])


def sub_gaussian_vect_with_corr_change_v2(
    alpha: float,
    d: int,
    n: int,
    n_star: int,
    rho_before: float,
    rho_after: float,
    alpha2: Union[float, None] = None,
) -> List[List[float]]:
    """
    Generate sub-Gaussian vectors with a correlation change.

    :param alpha: Stability parameter for the alpha-stable distribution.
    :param d: Dimension of each Gaussian vector.
    :param n: Total number of vectors.
    :param n_star: Number of vectors before the correlation change.
    :param rho_before: Correlation coefficient before the change.
    :param rho_after: Correlation coefficient after the change.
    :param alpha2: Stability parameter for the second segment. Default is None.
    :return: List of sub-Gaussian vectors.
    """
    mean = [0, 0]
    corr_before = np.array([[1, rho_before], [rho_before, 1]])
    corr_after = np.array([[1, rho_after], [rho_after, 1]])

    A = stblrnd(
        alpha=alpha / 2,
        gamma=(np.cos(np.pi * alpha / 4)) ** (2 / alpha),
        beta=1,
        delta=0,
        size=(n_star,),
    )
    N1 = np.random.multivariate_normal(mean, corr_before, n_star)
    vector_before = [list(A[i] * N1[i]) for i in range(n_star)]

    if not alpha2:
        alpha2 = alpha

    A2 = stblrnd(
        alpha=alpha2 / 2,
        gamma=(np.cos(np.pi * alpha2 / 4)) ** (2 / alpha2),
        beta=1,
        delta=0,
        size=(int(n - n_star),),
    )
    N2 = np.random.multivariate_normal(mean, corr_after, int(n - n_star))
    vector_after = [list(A2[i] * N2[i]) for i in range(int(n - n_star))]

    return vector_before + vector_after


def sub_gaussian_vect_with_corr_change_case3(
    alpha: float,
    d: int,
    n1: int,
    n2: int,
    n3: int,
    rho1: float,
    rho2: float,
    rho3: float,
    alpha2: Union[float, None] = None,
    alpha3: Union[float, None] = None,
) -> List[List[float]]:
    """
    Generate sub-Gaussian vectors with two correlation changes.

    :param alpha: Stability parameter for the alpha-stable distribution.
    :param d: Dimension of each Gaussian vector.
    :param n1: Number of vectors in the first segment.
    :param n2: Number of vectors in the second segment.
    :param n3: Number of vectors in the third segment.
    :param rho1: Correlation coefficient in the first segment.
    :param rho2: Correlation coefficient in the second segment.
    :param rho3: Correlation coefficient in the third segment.
    :param alpha2: Stability parameter for the second segment. Default is None.
    :param alpha3: Stability parameter for the third segment. Default is None.
    :return: List of sub-Gaussian vectors.
    """
    mean = [0, 0]

    corr1 = np.array([[1, rho1], [rho1, 1]])
    corr2 = np.array([[1, rho2], [rho2, 1]])
    corr3 = np.array([[1, rho3], [rho3, 1]])

    A = stblrnd(
        alpha=alpha / 2,
        gamma=(np.cos(np.pi * alpha / 4)) ** (2 / alpha),
        beta=1,
        delta=0,
        size=(n1,),
    )
    N1 = np.random.multivariate_normal(mean, corr1, n1)
    vector_1 = [list(A[i] * N1[i]) for i in range(n1)]

    if not alpha2:
        alpha2 = alpha

    if not alpha3:
        alpha3 = alpha

    A2 = stblrnd(
        alpha=alpha2 / 2,
        gamma=(np.cos(np.pi * alpha2 / 4)) ** (2 / alpha2),
        beta=1,
        delta=0,
        size=(n2,),
    )
    N2 = np.random.multivariate_normal(mean, corr2, n2)
    vector_2 = [list(A2[i] * N2[i]) for i in range(n2)]

    A = stblrnd(
        alpha=alpha3 / 2,
        gamma=(np.cos(np.pi * alpha3 / 4)) ** (2 / alpha3),
        beta=1,
        delta=0,
        size=(n3,),
    )
    N1 = np.random.multivariate_normal(mean, corr3, n3)
    vector_3 = [list(A[i] * N1[i]) for i in range(n3)]

    return vector_1 + vector_2 + vector_3


def multivariate_gaussian_vect_with_alpha_stable_noise(
    alpha: float,
    rho: float,
    n: int,
    p: float,
    sigma1: float = 1,
    sigma2: float = 1,
    mu1: float = 0,
    mu2: float = 0,
    gamma: float = 1,
    beta: float = 1,
    delta: float = 0,
) -> np.ndarray:
    """
    Generate multivariate Gaussian vectors with alpha-stable noise.

    :param alpha: Stability parameter for the alpha-stable noise.
    :param rho: Correlation coefficient between Gaussian coordinates.
    :param n: Number of samples to generate.
    :param p: Probability of adding alpha-stable noise.
    :param sigma1: Standard deviation of the first Gaussian coordinate. Default is 1.
    :param sigma2: Standard deviation of the second Gaussian coordinate. Default is 1.
    :param mu1: Mean of the first Gaussian coordinate. Default is 0.
    :param mu2: Mean of the second Gaussian coordinate. Default is 0.
    :param gamma: Scale parameter for the alpha-stable noise. Default is 1.
    :param beta: Skewness parameter for the alpha-stable noise. Default is 1.
    :param delta: Location parameter for the alpha-stable noise. Default is 0.
    :return: Array of multivariate Gaussian vectors with alpha-stable noise.
    """
    mean = [mu1, mu2]
    corr_matrix = np.array([[sigma1, sigma1 * sigma2 * rho], [sigma1 * sigma2 * rho, sigma2]])
    probability = np.random.random(n)
    A = stblrnd(alpha=alpha, beta=beta, gamma=gamma, delta=delta, size=(n,)) * (probability < p)
    N = np.random.multivariate_normal(mean, corr_matrix, n)
    return N + np.tile(A, (2, 1)).T


def gaussian_vect_with_stable_noise_corr_change(
    alpha: float, rho1: float, rho2: float, n: int, n_star: int, p: float
) -> List[List[float]]:
    """
    Generate Gaussian vectors with alpha-stable noise and a correlation change.

    :param alpha: Stability parameter for the alpha-stable noise.
    :param rho1: Correlation coefficient in the first segment.
    :param rho2: Correlation coefficient in the second segment.
    :param n: Total number of vectors.
    :param n_star: Number of vectors in the first segment.
    :param p: Probability of adding alpha-stable noise.
    :return: List of Gaussian vectors with alpha-stable noise.
    """
    vect1 = multivariate_gaussian_vect_with_alpha_stable_noise(alpha=alpha, n=n_star, p=p, rho=rho1)

    vect2 = multivariate_gaussian_vect_with_alpha_stable_noise(alpha=alpha, n=(n - n_star), p=p, rho=rho2)

    return list(vect1) + list(vect2)


def gaussian_vect_with_stable_noise_corr_change_case3(
    alpha: float, rho1: float, rho2: float, n1: int, n2: int, n3: int, p: float
) -> List[List[float]]:
    """
    Generate Gaussian vectors with alpha-stable noise and two correlation changes.

    :param alpha: Stability parameter for the alpha-stable noise.
    :param rho1: Correlation coefficient in the first and third segments.
    :param rho2: Correlation coefficient in the second segment.
    :param n1: Number of vectors in the first segment.
    :param n2: Number of vectors in the second segment.
    :param n3: Number of vectors in the third segment.
    :param p: Probability of adding alpha-stable noise.
    :return: List of Gaussian vectors with alpha-stable noise.
    """
    vect1 = multivariate_gaussian_vect_with_alpha_stable_noise(alpha=alpha, n=n1, p=p, rho=rho1)

    vect2 = multivariate_gaussian_vect_with_alpha_stable_noise(alpha=alpha, n=n2, p=p, rho=rho2)

    vect3 = multivariate_gaussian_vect_with_alpha_stable_noise(alpha=alpha, n=n3, p=p, rho=rho1)

    return list(vect1) + list(vect2) + list(vect3)


def t_student_vector(rho: float, dof: int, n: int) -> np.ndarray:
    """
    Generate two-dimensional Student's t-distributed random variables.

    :param rho: Correlation coefficient between coordinates.
    :param dof: Degrees of freedom.
    :param n: Number of samples to generate.
    :return: Array of Student's t-distributed random variables.
    """
    mean = [0, 0]
    corr_matrix = np.array([[1, rho], [rho, 1]])
    N = np.random.multivariate_normal(mean, corr_matrix, n)
    chi = np.random.chisquare(df=dof, size=n)
    return N * np.tile(chi, (2, 1)).T


def t_student_vect_with_corr_change(
    rho1: float,
    rho2: float,
    dof1: int,
    n: int,
    n_star: int,
    dof2: Union[int, None] = None,
) -> List[List[float]]:
    """
    Generate Student's t-distributed variables with one correlation change.

    :param rho1: Correlation coefficient in the first segment.
    :param rho2: Correlation coefficient in the second segment.
    :param dof1: Degrees of freedom in the first segment.
    :param dof2: Degrees of freedom in the second segment. Default is None.
    :param n: Total number of samples.
    :param n_star: Number of samples in the first segment.
    :return: List of Student's t-distributed variables.
    """
    if not dof2:
        dof2 = dof1

    vect1 = t_student_vector(n=n_star, dof=dof1, rho=rho1)

    vect2 = t_student_vector(n=(n - n_star), dof=dof2, rho=rho2)

    return list(vect1) + list(vect2)


def t_student_vect_with_corr_change_case3(
    rho1: float,
    rho2: float,
    rho3: float,
    dof1: int,
    n1: int,
    n2: int,
    n3: int,
    dof2: Union[int, None] = None,
    dof3: Union[int, None] = None,
) -> List[List[float]]:
    """
    Generate Student's t-distributed variables with two correlation changes.

    :param rho1: Correlation coefficient in the first segment.
    :param rho2: Correlation coefficient in the second segment.
    :param rho3: Correlation coefficient in the third segment.
    :param dof1: Degrees of freedom in the first segment.
    :param dof2: Degrees of freedom in the second segment. Default is None.
    :param dof3: Degrees of freedom in the third segment. Default is None.
    :param n1: Number of samples in the first segment.
    :param n2: Number of samples in the second segment.
    :param n3: Number of samples in the third segment.
    :return: List of Student's t-distributed variables.
    """
    if not dof2:
        dof2 = dof1

    if not dof3:
        dof3 = dof1

    vect1 = t_student_vector(n=n1, dof=dof1, rho=rho1)

    vect2 = t_student_vector(n=n2, dof=dof2, rho=rho2)

    vect3 = t_student_vector(n=n3, dof=dof3, rho=rho3)

    return list(vect1) + list(vect2) + list(vect3)
