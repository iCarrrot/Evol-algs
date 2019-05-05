import numpy as np
import sys


MAX_SIGMA = 100000.


def parent_selection(P, size, scores, i=-1):

    probs = scores.max() - scores

    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(scores.size) / scores.size

    ids = np.random.choice(
        range(scores.size),
        size=size,
        replace=True,
        p=probs
    )
    return P[ids]


def mutation(Pc, tau, tau0, d, **kwargs):
    # eps0 = tau0 * np.random.randn(1)
    # eps = tau * np.random.randn(d)
    # Pc[:, d:] = Pc[:, d:] * np.exp(eps + eps0)
    # eps2 = Pc[:, d:] * np.random.randn(d)
    # Pc[:, :d] += eps2
    ind_size = (Pc.shape[-1] // 2)
    xs = Pc[:, :ind_size]
    sigmas = Pc[:, ind_size:]

    epsilons_i = np.random.randn(*sigmas.shape) * (tau)
    epsilons_0 = np.random.randn(sigmas.shape[0], 1) * (tau0)
    sigmas *= np.exp(epsilons_0 + epsilons_i)

    xs += np.random.randn(*xs.shape) * (sigmas)
    Pc[:, :ind_size] = xs
    Pc[:, ind_size:] = sigmas
    return Pc


def replacement(pop, scores, mi):
    chosen = np.argsort(scores)[:mi]
    return pop[chosen], scores[chosen]


def ES(mi, Lambda, d, population_evaluation, number_of_iterations, K, plus=True, domain=(0, 1), **kwargs):
    history = np.zeros((number_of_iterations, mi))
    sigma_history = np.zeros((number_of_iterations, d))

    tau = K/np.sqrt(2*d)
    tau0 = K/np.sqrt(2*np.sqrt(d))
    P = np.hstack(
        (
            np.random.uniform(high=domain[1], low=domain[0], size=(mi, d)),
            # np.random.rand(mi, d)*100
            np.random.uniform(
                low=1,
                high=domain[1] * 3 / 4,
                size=(mi, d)
            )
        )
    )

    scores = population_evaluation(P[:, :d])
    for i in range(number_of_iterations):
        Pc = parent_selection(P, Lambda, scores, i)
        Pc = mutation(Pc, tau, tau0, d)
        if plus:
            P, scores = replacement(np.vstack((P, Pc)),
                                    np.hstack(
                                        (scores, population_evaluation(Pc[:, :d]))),
                                    mi)
        else:
            P, scores = replacement(Pc, population_evaluation(Pc[:, :d]), mi)
        history[i, :] = scores.copy()
        sigma_history[i, :] = np.mean(P[:, d:], axis=0)
    return {'costs': history, 'sigmas': sigma_history}
