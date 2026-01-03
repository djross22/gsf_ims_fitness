
import cmdstanpy
import numpy as np
import pandas as pd
import os


def check_all_diagnostics(fit):
    print(fit.diagnose())


def file_to_list(file_name):
    text_file = open(file_name, "r")
    lines = text_file.readlines()
    return lines


def compile_model(filename, file_in_repository_models=True, check_includes=True, incl_stan_save_file=None):

    return_directory = os.getcwd()
    if file_in_repository_models:
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stan_models'))
    
    lines = file_to_list(filename)
    
    has_includes = False
    stan_file = filename
    while check_includes:
        check_includes = False
        new_lines = []
        for line in lines:
            if line.strip().startswith('#include'):
                include_file = line[line.find('#include')+9:line.rfind('.stan')+5]
                include_file = include_file.strip()
                include_lines = file_to_list(include_file)
                new_lines += include_lines
                check_includes = True
                has_includes = True
            else:
                new_lines += [line]
        lines = new_lines
        
    if has_includes:
        if incl_stan_save_file is not None:
            os.chdir(return_directory)
            stan_file = incl_stan_save_file
        else:
            stan_file = filename.replace('.stan', '.incl.stan')

        with open(stan_file, "w") as out_file:
            out_file.writelines(lines)
    
    sm = cmdstanpy.CmdStanModel(stan_file=stan_file)
    
    os.chdir(return_directory)

    return sm


def check_rhat_by_params(fit, rhat_cutoff, stan_parameters=None):
    df = fit.summary()
    if stan_parameters is not None:
        key_params = np.array(df.index)
        sel = []
        for p in key_params:
            s = False
            for p2 in stan_parameters:
                if p2 in p:
                    s = True
                    break
            sel.append(s)
        key_params = key_params[sel]
            
        df = df.loc[key_params]
     
    df = df[df.R_hat>rhat_cutoff]
    
    return list(df.index)
    

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