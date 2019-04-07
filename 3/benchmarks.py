import numpy as np


def griewank(P):
    return (
        1
        + (P**2).sum(axis=1) / 4000
        - np.prod(
            np.cos(
                P/np.sqrt(
                    np.arange(1, P.shape[-1] + 1)
                )
            ),
            axis=1
        )
    )


def rast(P):
    n = P.shape[-1]
    return (
        P**2 - 10 * np.cos(2*np.pi * P)
    ).sum(axis=1) + 10*n


def schw(P):
    n = P.shape[-1]
    return np.sum(
        P * np.sin(
            np.sqrt(
                np.abs(P)
            )
        ),
        axis=1
    ) + 418.9829*n


def dp(P):
    return (P[..., 0]-1)**2 + np.sum(
        (
            2 * P[..., 1:]**2 - P[..., :-1]**2
        )**2 *
        np.arange(2, P.shape[-1] + 1), axis=1
    )

def sphere(P):
    return np.sum(
        P**2,
        axis=1
    )
