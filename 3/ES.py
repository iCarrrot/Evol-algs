import numpy as np
import sys

def parent_selection(P, size, scores, i=-1):
    if size == scores.size:
        return P

    probs = 1 / (scores - scores.min() + sys.float_info.epsilon)

    ids = np.random.choice(
        range(scores.size),
        size=size,
        replace=True,
        p=(probs / probs.sum()).squeeze()
    )
    return P[ids]


def mutation(Pc, tau, tau0, d):
    eps0 = tau0 * np.random.randn(1)
    eps = tau * np.random.randn(d)
    Pc[:, d:] = Pc[:, d:] * (eps + eps0)
    eps2 = Pc[:, d:] * np.random.randn(d)
    Pc[:, :d] += eps2
    return Pc


def ES(mi, Lambda, d, population_evaluation, number_of_iterations, K, plus=True, domain=(0, 1), **kwargs):
    history = np.zeros((number_of_iterations, mi + Lambda if plus else Lambda))
    tau = K/np.sqrt(2*d)
    tau0 = K/np.sqrt(2*np.sqrt(d))
    P = np.hstack(
        (np.random.uniform(high=domain[1], low=domain[0], size=(mi, d)),
         np.random.rand(mi, d))
    )
    scores = population_evaluation(P[:, :d])
    for i in range(number_of_iterations):
        Pc = parent_selection(P, Lambda, scores, i)
        Pc = mutation(Pc, tau, tau0, d)
        if plus:
            P = np.vstack((parent_selection(P, mi, scores, i), Pc))
        else:
            P = Pc
        history[i, :] = scores = population_evaluation(P[:, :d])

    return history
