"""
This code was downloaded from github.com/betanalpha/jupyter_case_studies
on Mon July 2, 2018

Associated license.txt file contents:
    The code in this repository is copyrighted by Columbia University and licensed under the new BSD (3-clause) license:

      https://opensource.org/licenses/BSD-3-Clause

    The text in this repository is copyrighted by Michael Betancourt and licensed under the CC BY-NC 4.0 license:

      https://creativecommons.org/licenses/by-nc/4.0/


"""

import pystan
import pickle
import numpy as np
import pandas as pd
import os


def check_div(fit):
    """Check transitions that ended with a divergence"""
    no_warning = True
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = [x for y in sampler_params for x in y['divergent__']]
    n = sum(divergent)
    N = len(divergent)
    if n > 0:
        print('  Try running with larger adapt_delta to remove the divergences')
        print('{} of {} iterations ended with a divergence ({}%)'.format(n, N,
                100 * n / N))
        no_warning = False
    return no_warning


def check_treedepth(fit, max_depth = 10):
    """Check transitions that ended prematurely due to maximum tree depth limit"""
    no_warning = True
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    depths = [x for y in sampler_params for x in y['treedepth__']]
    n = sum(1 for x in depths if x == max_depth)
    N = len(depths)
    #print(('{} of {} iterations saturated the maximum tree depth of {}'
    #        + ' ({}%)').format(n, N, max_depth, 100 * n / N))
    if n > 0:
        print('  Run again with max_depth set to a larger value to avoid saturation')
        print(('{} of {} iterations saturated the maximum tree depth of {}'
                + ' ({}%)').format(n, N, max_depth, 100 * n / N))
        no_warning = False
    return no_warning


def check_energy(fit):
    """Checks the energy Bayesian fraction of missing information (E-BFMI)"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    no_warning = True
    for chain_num, s in enumerate(sampler_params):
        energies = s['energy__']
        numer = sum((energies[i] - energies[i - 1])**2 for i in range(1, len(energies))) / len(energies)
        denom = np.var(energies)
        if numer / denom < 0.2:
            print('Chain {}: E-BFMI = {}'.format(chain_num, numer / denom))
            no_warning = False
    if no_warning:
        print('E-BFMI indicated no pathological behavior')
    else:
        print('  E-BFMI below 0.2 indicates you may need to reparameterize your model')
    return no_warning


def check_n_eff(fit, ratio_threshold=0.001, verbose=True):
    """Checks the effective sample size per iteration"""
    fit_summary = fit.summary(probs=[0.5])
    n_effs = [x[4] for x in fit_summary['summary']]
    names = fit_summary['summary_rownames']
    n_iter = len(fit.extract()['lp__'])

    no_warning = True
    for n_eff, name in zip(n_effs, names):
        ratio = n_eff / n_iter
        if (ratio < ratio_threshold):
            if verbose: print('n_eff / iter for parameter {} is {}!'.format(name, ratio))
            no_warning = False
    if no_warning:
        if verbose: print('n_eff / iter looks reasonable for all parameters')
    else:
        if verbose: print('  n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated')
    return no_warning


def check_rhat(fit, rhat_threshold=1.1, verbose=True):
    """Checks the potential scale reduction factors"""
    from math import isnan
    from math import isinf

    fit_summary = fit.summary(probs=[0.5])
    rhats = [x[5] for x in fit_summary['summary']]
    names = fit_summary['summary_rownames']

    no_warning = True
    for rhat, name in zip(rhats, names):
        if (rhat > rhat_threshold or isnan(rhat) or isinf(rhat)):
            if verbose: print('Rhat for parameter {} is {}!'.format(name, rhat))
            no_warning = False
    if no_warning:
        if verbose: print('Rhat looks reasonable for all parameters')
    else:
        if verbose: print('  Rhat above 1.1 indicates that the chains very likely have not mixed')
    return no_warning


def check_all_diagnostics(fit, max_depth=10):
    """Checks all MCMC diagnostics"""
    check_n_eff(fit)
    check_rhat(fit)
    check_div(fit)
    check_treedepth(fit, max_depth=max_depth)
    check_energy(fit)


def _by_chain(unpermuted_extraction):
    num_chains = len(unpermuted_extraction[0])
    result = [[] for _ in range(num_chains)]
    for c in range(num_chains):
        for i in range(len(unpermuted_extraction)):
            result[c].append(unpermuted_extraction[i][c])
    return np.array(result)


def _shaped_ordered_params(fit):
    ef = fit.extract(permuted=False, inc_warmup=False) # flattened, unpermuted, by (iteration, chain)
    ef = _by_chain(ef)
    ef = ef.reshape(-1, len(ef[0][0]))
    ef = ef[:, 0:len(fit.flatnames)] # drop lp__
    shaped = {}
    idx = 0
    for dim, param_name in zip(fit.par_dims, fit.extract().keys()):
        length = int(np.prod(dim))
        shaped[param_name] = ef[:,idx:idx + length]
        shaped[param_name].reshape(*([-1] + dim))
        idx += length
    return shaped


def partition_div(fit):
    """ Returns parameter arrays separated into divergent and non-divergent transitions"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    div = np.concatenate([x['divergent__'] for x in sampler_params]).astype('int')
    params = _shaped_ordered_params(fit)
    nondiv_params = dict((key, params[key][div == 0]) for key in params)
    div_params = dict((key, params[key][div == 1]) for key in params)
    return nondiv_params, div_params


def compile_model(filename, model_name=None, force_recompile=False, verbose=True):
    """This will automatically cache models - great if you're just running a
    script on the command line.

    See http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html"""
    from hashlib import md5

    return_directory = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Stan models'))

    with open(filename) as f:
        model_code = f.read()
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        if model_name is None:
            cache_fn = 'cached-model-{}.pkl'.format(code_hash)
        else:
            cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
        if force_recompile:
            print(f"Compiling StanModel from file: {filename}")
            sm = pystan.StanModel(model_code=model_code)
            with open(cache_fn, 'wb') as f:
                pickle.dump(sm, f)
        else:
            try:
                sm = pickle.load(open(cache_fn, 'rb'))
            except:
                print(f"Compiling StanModel from file: {filename}")
                sm = pystan.StanModel(model_code=model_code)
                with open(cache_fn, 'wb') as f:
                    pickle.dump(sm, f)
            else:
                if verbose:
                    print("Using cached StanModel")

        os.chdir(return_directory)

        return sm


def rhat_from_dataframe(df, split_chains=True):
    df = df.copy()
    if split_chains and ('draw' in list(df.columns)):
        chains = np.unique(df.chain)
        add_chain = max(chains) + 1
        cut_draw = (df.draw.max()+1)/2
        new_chain = []
        new_draw = []
        for c, d in zip(df.chain, df.draw):
            if d >= cut_draw:
                new_chain.append(c + add_chain)
                new_draw.append(d - cut_draw)
            else:
                new_chain.append(c)
                new_draw.append(d)
        df['chain'] = new_chain
        df['draw'] = new_draw
        
    chains = np.unique(df.chain)
    num_chains = len(chains)
    num_samples = len(df)/num_chains
    
    columns = list(df.columns)
    ignore_columns = ['chain', 'draw', 'warmup'] + [x for x in columns if x[-2:]=='__']
    columns = [x for x in columns if x not in ignore_columns]
    
    chain_mean = [df[df.chain==n][columns].mean() for n in chains]
    
    chain_mean = pd.concat(chain_mean, axis=1)
    
    grand_mean = chain_mean.mean(axis=1)
    
    x3 = chain_mean.sub(grand_mean, axis=0)**2
    between_chains_var = num_samples/(num_chains-1)*x3.sum(axis=1)
    
    within_chains_var = []
    for n in chains:
        d2 = df[df.chain==n]
        d2 = d2[columns]
        x = ((d2 - chain_mean[n])**2).sum()
        within_chains_var.append(1/(num_samples-1)*x)
    
    within_chains_var = pd.concat(within_chains_var, axis=1).mean(axis=1)
    
    rh1 = ((num_samples-1)/num_samples)*within_chains_var
    rh2 = between_chains_var/num_samples
    
    return np.sqrt((rh1 + rh2)/within_chains_var)