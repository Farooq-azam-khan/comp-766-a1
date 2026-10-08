import numpy as np

from convolution import ORIENTATION_COUNT


def gamma(beta, gamma):
    if 0 <= beta <= np.pi:
        raise ValueError(f'{beta=} not in rage 0 to pi.')

    if 0 <= gamma<= np.pi:
        raise ValueError(f'{gamma=} not in rage 0 to pi.')

    if np.abs(gamma - beta) <= np.pi/2:
        return gamma - beta
    if np.pi/2 < gamma - beta <= np.pi / 2:
        return gamma - beta - np.pi
    if -np.pi <= gamma - beta < -np.pi/2:
        return gamma - beta + np.pi
    raise ValueError('gamma func invalid condition.')

def is_cocircular(posi, posj, theta_lam, theta_lam_prime):
    """
    Co-circularity in discrete case:
        - given position x, y draw a circle of radius 1/2
        - given epsilon = pi/16
        - two tangents are cocircular if there exists AT LEAST ONE ASSIGNEMNT of the position and orientation variables given the def. of cocircular.
    """
    epsilon = np.pi/ORIENTATION_COUNT
    d_ij = np.sqrt((posi[1]-posj[1])**2 + (posi[0]-posj[0])**2) # sqrt[ (y2-y1)^2 + (x2-x1)^2 ]
    alpha = np.arcsin(1/d_ij)
    theta_ij = np.arctan((posj[1]-posi[1])/(posj[0]-posi[0])) # arctan[ (y2-y1)/(x2-x1)]
    return np.abs(gamma(theta_lam, theta_ij) - gamma(theta_ij, theta_lam_prime)) < epsilon + 2*alpha


def cocircularity_coeff(posi, posj, lam, lam_prime, c_min=0.1):
    theta_lam, theta_lam_prime = lam*np.pi/ORIENTATION_COUNT,lam_prime*np.pi/ORIENTATION_COUNT
    if is_cocircular(posi, posj, theta_lam, theta_lam_prime):
        return 1
    # fig 9: not doing too much geometry
    # drop_off_slope = 0.1 # eta in the paper.
    return c_min

def generate_support():
    pass
