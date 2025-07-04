from scipy.stats import qmc

def quassi_random_search(param_ranges, n_trials):
    """
    param_ranges: 是每个超参数取值范围(min, max)的列表
    n_trials： 试验的次数
    """
    dim = len(param_ranges)
    sampler = qmc.Halton(d=dim, scramble=True)
    samples = sampler.random(n=n_trials)
    scaled_samples = []
    for s in samples:
        params = []
        for i, (low, high) in enumerate(param_ranges):
            params.append(low + (high - low) * s[i])
        scaled_samples.append(params)
    return scaled_samples

