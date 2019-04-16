import numpy as np
import matplotlib.pyplot as plt
import time


def plot_scores(ev_func, tries=10, barplot=True, file=False, filename='', title='', verbose=True, **kwargs):
    t0 = time.time()
    bests = []
    costss = []
    sigmass = []
    if file:
        with open("output_"+filename, "a") as f:
            f.write(f'\n\n\n\n\n\n\n\n{t0}\n\n')

    for i in range(tries):
        t1 = time.time()
        data = ev_func(**kwargs)
        costs = data['costs']
        if 'sigmas' in data:
            sigmass.append(data["sigmas"])
        best = np.min(costs)
        logs = f'{title}: próba {i}; wynik: {best}; czas: {time.time() - t1}'
        if verbose:
            print(logs)
        if file:
            with open("output_"+filename, "a") as f:
                f.write(logs)

        bests.append(best)
        costss.append(costs)

    costs = costss[np.argmin(bests)]
    x, y = costs.shape
    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.scatter([np.argmin(costs.min(axis=1))], [np.min(costs)])
    plt.plot(range(x), costs.min(axis=1))
    plt.plot(range(x), costs.max(axis=1))
    plt.plot(range(x), costs.mean(axis=1))
    plt.show()
    if len(sigmass) > 0 :
        sigmas = sigmass[np.argmin(bests)]
        plt.figure(figsize=(15, 5))
        plt.title(title + ' sigmas')
        plt.plot(range(x), sigmas.min(axis=1))
        plt.plot(range(x), sigmas.max(axis=1))
        plt.plot(range(x), sigmas.mean(axis=1))
        plt.show()
    logs = f'\nWynik {title}: {min(bests)}; czas: {time.time() - t0}, \n\n\n'
    if verbose:
        print(logs)
    if file:
        with open("output_"+filename, "a") as f:
            f.write(logs)

    if barplot:
        plt.hist(bests)
        plt.show()


def plot_test(F, domain = (-50,50), num=3000):
    x = np.linspace(*domain, num)
    z = np.linspace(*domain, num)
    x, z = np.meshgrid(x,z)
    vec = np.hstack((x.reshape(-1,1),z.reshape(-1,1)))
    y = (F(vec)).reshape(num,num)
    plt.contour(x,z,y)
    plt.show()