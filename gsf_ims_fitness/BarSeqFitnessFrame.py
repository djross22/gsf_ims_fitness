# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:29:34 2019

@author: djross
"""

import glob  # filenames and pathnames utility
import os    # operating sytem utility
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
#from scipy import special
#from scipy import misc

#import pystan
import pickle

import seaborn as sns
sns.set()

from IPython.display import display
#import ipywidgets as widgets
#from ipywidgets import interact#, interact_manual

from . import fitness
from . import stan_utility

sns.set_style("white")
sns.set_style("ticks", {'xtick.direction':'in', 'xtick.top':True, 'ytick.direction':'in', 'ytick.right':True})

class BarSeqFitnessFrame:
        
    def __init__(self, notebook_dir, experiment=None, barcode_file=None, high_tet=20, inducer_conc_list=None, inducer="IPTG",
                 inducer_2=None, inducer_conc_list_2=None, low_tet=None):
        
        self.notebook_dir = notebook_dir
        
        self.high_tet = high_tet
        if low_tet is not None:
            self.low_tet = low_tet
        
        if experiment is None:
            experiment = fitness.get_exp_id(notebook_dir)
        
        self.experiment = experiment
        
        print(f"Importing BarSeq count data and plotting histogram for thresholding for experiment: {experiment}")

        self.data_directory = notebook_dir + "\\barcode_analysis"
        os.chdir(self.data_directory)
    
        if barcode_file is None:
            barcode_file = glob.glob("*.sorted_counts.csv")[0]
        print(f"Importing BarSeq count data from file: {barcode_file}")
        barcode_frame = pd.read_csv(barcode_file, skipinitialspace=True)
    
        barcode_frame.sort_values('total_counts', ascending=False, inplace=True)
        #barcode_frame.reset_index(drop=True, inplace=True)
        
        self.barcode_frame = barcode_frame
        
        if inducer_conc_list is None:
            inducer_conc_list = [0, 2]
            for i in range(10):
                inducer_conc_list.append(2*inducer_conc_list[-1])
        self.inducer_conc_list = inducer_conc_list
        
        if inducer_conc_list_2 is not None:
            self.inducer_conc_list_2 = inducer_conc_list_2
        
        self.inducer = inducer
        if inducer_2 is not None:
            self.inducer_2 = inducer_2
        
        self.fit_fitness_difference_params = None
        self.fit_fitness_difference_funct = None
            
    def trim_and_sum_barcodes(self, cutoff=None, export_trimmed_file=False, trimmed_export_file=None, auto_save=True):
        
        barcode_frame = self.barcode_frame
        
        if cutoff is not None:
            barcode_frame = barcode_frame[barcode_frame["total_counts"]>cutoff].copy()
            #barcode_frame.reset_index(drop=True, inplace=True)
            
        print(f"Calculating read fraction for each barcode in each sample")
        for w in fitness.wells():
            label = 'fraction_' + w
            barcode_frame[label] = barcode_frame[w]/barcode_frame[w].sum()
        
        barcode_frame['fraction_total'] = barcode_frame['total_counts']/barcode_frame['total_counts'].sum()
        
        print(f"Calculating read totals and fractions for each barcode in samples from first time point")
        total = []
        for index, row in barcode_frame[fitness.wells_by_column()[:24]].iterrows():
            counts = 0
            for t in fitness.wells_by_column()[:24]:
                counts += row[t]
            total.append(counts)
        barcode_frame['total_counts_plate_2'] = total
        barcode_frame['fraction_total_p2'] = barcode_frame['total_counts_plate_2']/barcode_frame['total_counts_plate_2'].sum()  
        
        fraction_list = ["fraction_" + w for w in fitness.wells_by_column()[:24] ]
        barcode_frame["fraction_p2_std"] = barcode_frame[fraction_list].std(axis=1)
        
        self.barcode_frame = barcode_frame
        
        if export_trimmed_file:
            if trimmed_export_file is None:
                trimmed_export_file = f"{self.experiment}.trimmed_sorted_counts.csv"
            print(f"Exporting trimmed barcode counts data to: {trimmed_export_file}")
            barcode_frame.to_csv(trimmed_export_file)
            
        if auto_save:
            self.save_as_pickle()
            
    def label_reference_sequences(self, ref_seq_file_path=None, show_output=True, auto_save=True):
        
        barcode_frame = self.barcode_frame
        
        if ref_seq_file_path is None:
            ref_seq_file = "reference_sequences.csv"
            ref_seq_file_found = False
            top_directory = self.notebook_dir
            while not ref_seq_file_found:
                find_result = top_directory.rfind("\\")
                if find_result == -1:
                    break
                else:
                    top_directory = top_directory[:find_result]
                    os.chdir(top_directory)
                    ref_seq_file_found = os.path.isfile(ref_seq_file)
            
            if ref_seq_file_found:
                ref_seq_frame = pd.read_csv(ref_seq_file, skipinitialspace=True)
            else:
                ref_seq_frame = None
        else:
            ref_seq_frame = pd.read_csv(ref_seq_file_path, skipinitialspace=True)
            
        os.chdir(self.data_directory)
        
        name_list = [""]*len(barcode_frame)
        barcode_frame["RS_name"] = name_list
    
        if ref_seq_frame is not None:
            for index, row in ref_seq_frame.iterrows():
                display_frame = barcode_frame[barcode_frame["forward_BC"].str.contains(row["forward_lin_tag"])]
                display_frame = display_frame[display_frame["reverse_BC"].str.contains(row["reverse_lin_tag"])]
                display_frame = display_frame[["RS_name", "forward_BC", "reverse_BC", "total_counts"]]
                if len(display_frame)>0:
                    display_frame["RS_name"].iloc[0] = row["RS_name"]
                    barcode_frame.loc[display_frame.index[0], "RS_name"] = row["RS_name"]
                if show_output:
                    display(display_frame)
        
            total_reads = barcode_frame["total_counts"].sum()
            print(f"total reads: {total_reads}")
            total_RS_reads = barcode_frame[barcode_frame["RS_name"]!=""]["total_counts"].sum()
            print(f"reference sequence reads: {total_RS_reads} ({total_RS_reads/total_reads*100}%)")
            
        self.barcode_frame = barcode_frame
        
        if auto_save:
            self.save_as_pickle()
            
    def mark_chimera_parents(self):
        
        barcode_frame = self.barcode_frame
        
        new_columns_list = ["forward_parent", "reverse_parent", "parent_geo_mean_p2"]
        for col in new_columns_list:
            if col not in barcode_frame.columns:
                barcode_frame[col] = np.nan
        
        for index, row in barcode_frame[barcode_frame["possibleChimera"]].iterrows():
            comp_data = barcode_frame[barcode_frame.index<index]
                
            for_matches = comp_data[comp_data["forward_BC"]==row["forward_BC"]]
            for_matches = for_matches[for_matches["possibleChimera"]==False]
            rev_matches = comp_data[comp_data["reverse_BC"]==row["reverse_BC"]]
            rev_matches = rev_matches[rev_matches["possibleChimera"]==False]
            if ( ( len(for_matches)>0 ) & ( len(rev_matches)>0 ) ):
                #barcode_frame.at[index, "possibleChimera"] = True
                barcode_frame.at[index, "forward_parent"] = for_matches.index[0]
                barcode_frame.at[index, "reverse_parent"] = rev_matches.index[0]
                geo_mean_p2 = np.sqrt(rev_matches["total_counts_plate_2"].values[0]*for_matches["total_counts_plate_2"].values[0])
                barcode_frame.at[index, "parent_geo_mean_p2"] = geo_mean_p2
        
        self.barcode_frame = barcode_frame
            
    def flag_possible_chimeras(self, use_faster_search=True, faster_search_ratio=10):
        
        barcode_frame = self.barcode_frame
        if "possibleChimera" not in barcode_frame.columns:
            barcode_frame["possibleChimera"] = False
        new_columns_list = ["forward_parent", "reverse_parent", "parent_geo_mean", "parent_geo_mean_p2"]
        for col in new_columns_list:
            if col not in barcode_frame.columns:
                barcode_frame[col] = np.nan
        
        for index, row in barcode_frame[1:].iterrows():
            if use_faster_search:
                data_counts = row["total_counts"]
                comp_data = barcode_frame[barcode_frame["total_counts"]>faster_search_ratio*data_counts]
            else:
                comp_data = barcode_frame.loc[:index-1]
                
            for_matches = comp_data[comp_data["forward_BC"]==row["forward_BC"]]
            for_matches = for_matches[for_matches["possibleChimera"]==False]
            rev_matches = comp_data[comp_data["reverse_BC"]==row["reverse_BC"]]
            rev_matches = rev_matches[rev_matches["possibleChimera"]==False]
            if ( ( len(for_matches)>0 ) & ( len(rev_matches)>0 ) ):
                barcode_frame.at[index, "possibleChimera"] = True
                barcode_frame.at[index, "forward_parent"] = for_matches.index[0]
                barcode_frame.at[index, "reverse_parent"] = rev_matches.index[0]
                geo_mean = np.sqrt(rev_matches["total_counts"].values[0]*for_matches["total_counts"].values[0])
                barcode_frame.at[index, "parent_geo_mean"] = geo_mean
                geo_mean_p2 = np.sqrt(rev_matches["total_counts_plate_2"].values[0]*for_matches["total_counts_plate_2"].values[0])
                barcode_frame.at[index, "parent_geo_mean_p2"] = geo_mean_p2
        
        chimera_frame = barcode_frame[barcode_frame["possibleChimera"]]
        print(f"Number of potential chimera barcodes identified: {len(chimera_frame)}")
        
        self.barcode_frame = barcode_frame
        
    def mark_actual_chimeras(self, chimera_cut_line, auto_save=True):
        
        barcode_frame = self.barcode_frame
        
        barcode_frame["isChimera"] = False
        
        for index, row in barcode_frame[barcode_frame["possibleChimera"]].iterrows():
            geo_mean = row["parent_geo_mean"]/96
            count = row["total_counts"]/96
            if count<chimera_cut_line(geo_mean):
                barcode_frame.at[index, "isChimera"] = True
        
        self.barcode_frame = barcode_frame
        
        if auto_save:
            self.save_as_pickle()
        
            
    def fit_barcode_fitness(self,
                            auto_save=True,
                            ignore_samples=[],
                            refit_index=None,
                            ref_slope_to_average=True,
                            bi_linear_alpha=np.log(5)):
        
        barcode_frame = self.barcode_frame
        high_tet = self.high_tet
        low_tet = getattr(self, 'low_tet', None)
            
        #os.chdir(self.data_directory)
    
        inducer_conc_list = self.inducer_conc_list
        inducer = self.inducer
        inducer_2 = getattr(self, 'inducer_2', None)
        inducer_conc_list_2 = getattr(self, 'inducer_conc_list_2', None)
        
        if (low_tet is None) and (inducer_2 is None):
            sample_plate_map = fitness.get_sample_plate_map(inducer, inducer_conc_list, tet_conc_list=[high_tet])
        else:
            sample_plate_map = fitness.get_sample_plate_map(inducer, inducer_conc_list,
                                                            inducer_2=inducer_2, inducer_conc_list_2=inducer_conc_list_2, tet_conc_list=[low_tet, high_tet])
        
        # ignore_samples should be a list of 2-tuples: (sample_id, growth_plate) to ignore.
        # In the old version of the code, ignore_samples was a list of 3-tuples: e.g., ("no-tet", growth_plate, inducer_conc)
        # For backward compatibility, check if the old version is used, and change it to the new version
        if len(ignore_samples)>0:
            if len(ignore_samples[0])==3:
                new_ignore = []
                for ig in ignore_samples:
                    w = ig[0]=="tet"
                    gp = ig[1]
                    x = ig[2]
                    df = sample_plate_map
                    df = df[df.with_tet==w]
                    df = df[df.growth_plate==gp]
                    df = df[df[inducer]==x]
                    if len(df)>1:
                        print("problem converting ignore_samples")
                    elif len(df)==1:
                        row = df.iloc[0]
                        new_ignore.append((row.sample_id, gp))
                        
                ignore_samples = new_ignore
            for ig in ignore_samples:
                print(f"ignoring sample {ig[0]}, time point {ig[1]-1}")
            print()
        
        sample_list = np.unique(sample_plate_map.sample_id)
        # dictionary where each entry is a list of 4 booleans indicating whether or not 
        #     the sample should be ignored for each time point:
        sample_keep_dict = {}
        for s in sample_list:
            v = [(s, i+2) not in ignore_samples for i in range(4)]
            sample_keep_dict[s] = v
        
        if refit_index is None:
            print(f"Fitting to log(barcode ratios) to find fitness for each barcode in {self.experiment}")
            
            # for each sample_id, get a list of counts for each barcode at the 4 time points
            samples_with_tet = []
            samples_without_tet = []
            for s in sample_list:
                df = sample_plate_map
                df = df[df["sample_id"]==s]
                
                has_tet = df.iloc[0].with_tet
                if has_tet:
                    samples_with_tet.append(s)
                else:
                    samples_without_tet.append(s)
                
                df = df.sort_values('growth_plate')
                well_list = list(df.well.values)
                
                count_list = []
                for ind, row in barcode_frame.iterrows():
                    row_by_sample = row[well_list]
                    count_list.append(row_by_sample.values)
                    
                barcode_frame[f"read_count_S{s}"] = count_list
        
        print(f"samples_with_tet: {samples_with_tet}")
        print(f"samples_without_tet: {samples_without_tet}")
        print()
        
        # Dictionary of dictionaries
        #     first key is tet concentration
        #     second key is spike-in name
        spike_in_fitness_dict = {}
        tet_list = [0, 1.25, 10, 20]
        # TODO: Fitness for 1.25 and 10 are preliminary estimates based on weighted average of [tet] = zero and 20
        # TODO: move fitness values for spike-ins to somewhere else (not hard coded)
        fitness_dicts = [{"AO-B": 0.9637, "AO-E": 0.9666}, {"AO-B": 0.9587125, "AO-E": 0.9597825}, 
                         {"AO-B": 0.93045, "AO-E": 0.92115}, {"AO-B": 0.8972, "AO-E": 0.8757}]
        for t, d in zip(tet_list, fitness_dicts):
            spike_in_fitness_dict[t] = d
        
        if low_tet is None:
            ref_fit_str_B = str(spike_in_fitness_dict[0]["AO-B"]) + ';' + str(spike_in_fitness_dict[high_tet]["AO-B"])
            ref_fit_str_E = str(spike_in_fitness_dict[0]["AO-E"]) + ';' + str(spike_in_fitness_dict[high_tet]["AO-E"])
        else:
            ref_fit_str_B = str(spike_in_fitness_dict[0]["AO-B"]) + ';' + str(spike_in_fitness_dict[low_tet]["AO-B"]) + ';' + str(spike_in_fitness_dict[high_tet]["AO-B"])
            ref_fit_str_E = str(spike_in_fitness_dict[0]["AO-E"]) + ';' + str(spike_in_fitness_dict[low_tet]["AO-E"]) + ';' + str(spike_in_fitness_dict[high_tet]["AO-E"])
            
        print(f'Reference fitness values, AO-B: {ref_fit_str_B}, AO-E: {ref_fit_str_E}')
        print()
    
        #ref_index_b = barcode_frame[barcode_frame["RS_name"]=="AO-B"].index[0]
        #ref_index_e = barcode_frame[barcode_frame["RS_name"]=="AO-E"].index[0]
    
        #spike_in_row_dict = {"AO-B": barcode_frame[ref_index_b:ref_index_b+1], "AO-E": barcode_frame[ref_index_e:ref_index_e+1]}
        spike_in_row_dict = {"AO-B": barcode_frame[barcode_frame["RS_name"]=="AO-B"].iloc[0],
                             "AO-E": barcode_frame[barcode_frame["RS_name"]=="AO-E"].iloc[0]}
        
        sp_b = spike_in_row_dict["AO-B"][["RS_name", "read_count_S6"]]
        sp_e = spike_in_row_dict["AO-E"][["RS_name", "read_count_S6"]]
        print(f"AO-B: {sp_b}")
        print(f"AO-E: {sp_e}")
        print()
        
        # Fit to barcode log(ratios) over time to get slopes = fitness
        #     use both AO-B and AO-E as reference (separately)
        # Samples without tet are fit to simple linear function.
        # Samples with tet are fit to bi-linear function, with initial slope equal to corresponding without-tet sample (or average)
        
        if refit_index is None:
            fit_frame = barcode_frame
        else:
            fit_frame = barcode_frame.loc[refit_index:refit_index]
        
        x0 = np.array([2, 3, 4, 5])
        print()
        for spike_in, initial in zip(["AO-B", "AO-E"], ["b", "e"]):
            no_tet_slope_lists = []
            
            for samp in samples_without_tet:
                df = sample_plate_map
                df = df[df["sample_id"]==samp]
                df = df.sort_values('growth_plate')
                well_list = list(df.well.values)
            
                spike_in_reads = np.array(spike_in_row_dict[spike_in][well_list], dtype='int64')
                spike_in_fitness = spike_in_fitness_dict[0][spike_in]
            
                f_est_list = []
                f_err_list = []
                slope_list = []
                for index, row in fit_frame.iterrows(): # iterate over barcodes
                    n_reads = np.array(row[well_list], dtype='int64')
                    
                    sel = (n_reads>0)&(spike_in_reads>0)&sample_keep_dict[samp]
                    x = x0[sel]
                    y = (np.log(n_reads[sel]) - np.log(spike_in_reads[sel]))
                    s = np.sqrt(1/n_reads[sel] + 1/spike_in_reads[sel])
                                
                    if len(x)>1:
                        popt, pcov = curve_fit(fitness.line_funct, x, y, sigma=s, absolute_sigma=True)
                        slope_list.append(popt[0])
                        f_est_list.append(spike_in_fitness + popt[0]/np.log(10))
                        f_err_list.append(np.sqrt(pcov[0,0])/np.log(10))
                    else:
                        slope_list.append(np.nan)
                        f_est_list.append(np.nan)
                        f_err_list.append(np.nan)
                
                fit_frame[f'fitness_S{samp}_{initial}'] = f_est_list
                fit_frame[f'fitness_S{samp}_err_{initial}'] = f_err_list
            
                no_tet_slope_lists.append(slope_list)
                
            no_tet_slope_lists = np.array(no_tet_slope_lists)
            if ref_slope_to_average:
                no_tet_slope = no_tet_slope_lists.mean(axis=0)
            else:
                # Previous version used the zero-inducer zer-tet sample as the reference slope
                no_tet_slope = [x[0] for x in no_tet_slope_lists.transpose()]
            
            for samp in samples_with_tet:
                df = sample_plate_map
                df = df[df["sample_id"]==samp]
                df = df.sort_values('growth_plate')
                well_list = list(df.well.values)
            
                spike_in_reads = np.array(spike_in_row_dict[spike_in][well_list], dtype='int64')
                if low_tet is None:
                    tet_conc = high_tet
                else:
                    tet_conc = df.antibiotic_conc.iloc[0]
                spike_in_fitness = spike_in_fitness_dict[tet_conc][spike_in]
            
                f_est_list = []
                f_err_list = []
                for (index, row), slope_0 in zip(fit_frame.iterrows(), no_tet_slope): # iterate over barcodes
                    n_reads = np.array(row[well_list], dtype='int64')
                    
                    sel = (n_reads>0)&(spike_in_reads>0)&sample_keep_dict[samp]
                    x = x0[sel]
                    y = (np.log(n_reads[sel]) - np.log(spike_in_reads[sel]))
                    s = np.sqrt(1/n_reads[sel] + 1/spike_in_reads[sel])
                                
                    if len(x)>1:
                        def fit_funct(xp, mp, bp): return fitness.bi_linear_funct(xp-2, mp, bp, slope_0, alpha=bi_linear_alpha)
                        popt, pcov = curve_fit(fit_funct, x, y, sigma=s, absolute_sigma=True)
                        f_est_list.append(spike_in_fitness + popt[0]/np.log(10))
                        f_err_list.append(np.sqrt(pcov[0,0])/np.log(10))
                    else:
                        f_est_list.append(np.nan)
                        f_err_list.append(np.nan)
                
                fit_frame[f'fitness_S{samp}_{initial}'] = f_est_list
                fit_frame[f'fitness_S{samp}_err_{initial}'] = f_err_list
            
        self.barcode_frame = barcode_frame
        
        #os.chdir(self.notebook_dir)
        #pickle_file = self.experiment + '_inducer_conc_list.pkl'
        #with open(pickle_file, 'wb') as f:
        #    pickle.dump(inducer_conc_list, f)
            
        if auto_save:
            self.save_as_pickle()
        
        
    def stan_fitness_difference_curves(self,
                                      includeChimeras=False,
                                      fit_fitness_difference_params=None,
                                      control=dict(adapt_delta=0.9),
                                      iterations=1000,
                                      chains=4,
                                      auto_save=True,
                                      refit_index=None,
                                      plasmid="pVER",
                                      return_fit=False,
                                      include_lactose_zero=False):
            
        print(f"Using Stan to fit to fitness curves to find sensor parameters for {self.experiment}")
        print(f"  Using fitness parameters for {plasmid}")
        print("      Version from 2022-11-25")
        #os.chdir(self.notebook_dir)
        
        barcode_frame = self.barcode_frame
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
        
        fitness_columns_setup = self.get_fitness_columns_setup()
        
        if fitness_columns_setup[0]:
            old_style_columns, x, linthresh, fit_plot_colors, ligand_list, antibiotic_conc_list = fitness_columns_setup
            
            plot_df = None
        else:
            old_style_columns, linthresh, fit_plot_colors, antibiotic_conc_list, plot_df, ligand_list = fitness_columns_setup
        
        if len(ligand_list) == 1:
            sm_file = 'Double Hill equation fit.stan'
        
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = fitness.fit_fitness_difference_params(plasmid=plasmid, tet_conc=antibiotic_conc_list[1])
            
            params_list = ['log_g0', 'log_ginf_1', 'log_ec50_1', 'log_sensor_n_1', 'log_ginf_g0_ratio_1',
                           'low_fitness_high_tet', 'mid_g_high_tet', 'fitness_n_high_tet']
            log_g0_ind = params_list.index('log_g0')
            log_ginf_g0_ind = params_list.index('log_ginf_g0_ratio_1')
            params_dim = len(params_list)
                
        elif len(ligand_list) == 2:
            sm_file = 'Double Hill equation fit.two-lig.two-tet.stan'
        
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = [fitness.fit_fitness_difference_params(plasmid=plasmid, tet_conc=x) for x in antibiotic_conc_list[1:]]
            
            params_list = ['log_g0', 'log_ginf_1', 'log_ec50_1', 'log_sensor_n_1', 'log_ginf_g0_ratio_1',
                           'log_ginf_2', 'log_ec50_2', 'log_sensor_n_2', 'log_ginf_g0_ratio_2',
                           'low_fitness_low_tet', 'mid_g_low_tet', 'fitness_n_low_tet',
                           'low_fitness_high_tet', 'mid_g_high_tet', 'fitness_n_high_tet']
            log_g0_ind = params_list.index('log_g0')
            log_ginf_g0_ind_1 = params_list.index('log_ginf_g0_ratio_1')
            log_ginf_g0_ind_2 = params_list.index('log_ginf_g0_ratio_2')
            params_dim = len(params_list)
            
        quantile_params_list = [x for x in params_list if 'log_' in x]
        quantile_params_dim = len(quantile_params_list)
                
        stan_model = stan_utility.compile_model(sm_file)
        self.fit_fitness_difference_params = fit_fitness_difference_params
        
        quantile_list = [0.05, 0.25, 0.5, 0.75, 0.95]
        quantile_dim = len(quantile_list)
        
        log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf = fitness.log_g_limits(plasmid=plasmid)
        
        rng = np.random.default_rng()
        def stan_fit_row(st_row, st_index, lig_list, return_fit=False):
            print()
            print(f"fitting row index: {st_index}, for ligands: {lig_list}")
            
            stan_data = get_stan_data(st_row, plot_df, antibiotic_conc_list, lig_list, fit_fitness_difference_params, old_style_columns=old_style_columns, initial="b", plasmid=plasmid)

            if True:#try:
                if len(lig_list) == 1:
                    stan_init = [ init_stan_fit_single_ligand(stan_data, fit_fitness_difference_params) for i in range(chains) ]
                else:
                    stan_init = [ init_stan_fit_two_lig_two_tet(stan_data, fit_fitness_difference_params) for i in range(chains) ]
                
                #print("stan_data:")
                #for k, v in stan_data.items():
                #    print(f"{k}: {v}")
                #print()
                stan_fit = stan_model.sampling(data=stan_data, iter=iterations, init=stan_init, chains=chains, control=control)
                if return_fit:
                    return stan_fit
        
                stan_samples_arr = np.array([stan_fit[key] for key in params_list ])
                stan_popt = np.array([np.median(s) for s in stan_samples_arr ])
                stan_pcov = np.cov(stan_samples_arr, rowvar=True)
                stan_resid = np.median(stan_fit["rms_resid"])
                
                # Only save the quantiles and samples for the sensor params (not the fitness vs. g params)
                stan_quant_arr = stan_samples_arr[:quantile_params_dim]
                stan_samples_out = rng.choice(stan_quant_arr, size=32, replace=False, axis=1, shuffle=False)
                stan_quantiles = np.array([np.quantile(x, quantile_list) for x in stan_quant_arr ])
                
                # Also save these posterior probabilities: the sensor is on at zero, sensor is inverted (for each ligand)
                g0_samples = 10**stan_samples_arr[log_g0_ind]
                hill_on_at_zero_prob = len(g0_samples[g0_samples>wild_type_ginf/4])/len(g0_samples)
                if len(lig_list) == 1:
                    g_ratio_samples = stan_samples_arr[log_ginf_g0_ind]
                    hill_invert_prob = len(g_ratio_samples[g_ratio_samples<0])/len(g_ratio_samples)
                else:
                    g_ratio_samples = [stan_samples_arr[k] for k in [log_ginf_g0_ind_1, log_ginf_g0_ind_2]]
                    hill_invert_prob = [len(s[s<0])/len(s) for s in g_ratio_samples]
            '''
            except:
                stan_popt = np.full((params_dim), np.nan)
                stan_pcov = np.full((params_dim, params_dim), np.nan)
                stan_resid = np.nan
                stan_samples_out = np.full((quantile_params_dim, 32), np.nan)
                stan_quantiles = np.full((quantile_params_dim, quantile_dim), np.nan)
                print(f"Error during Stan fitting for index {st_index}:", sys.exc_info()[0])
                hill_invert_prob = np.nan
                hill_on_at_zero_prob = np.nan
            '''
                
            return (stan_popt, stan_pcov, stan_resid, stan_samples_out, stan_quantiles, hill_invert_prob, hill_on_at_zero_prob)
        
        if refit_index is None:
            fit_list = [ stan_fit_row(row, index, ligand_list) for (index, row) in barcode_frame.iterrows() ]
            
            popt_list = []
            pcov_list = []
            residuals_list = []
            samples_out_list = []
            quantiles_list = []
            invert_prob_list = []
            on_at_zero_prob_list = []
            
            for item in fit_list: # iterate over barcodes
                stan_popt, stan_pcov, stan_resid, stan_samples_out, stan_quantiles, hill_invert_prob, hill_on_at_zero_prob = item
                
                popt_list.append(stan_popt)
                pcov_list.append(stan_pcov)
                residuals_list.append(stan_resid)
                samples_out_list.append(stan_samples_out)
                quantiles_list.append(stan_quantiles)
                invert_prob_list.append(hill_invert_prob)
                on_at_zero_prob_list.append(hill_on_at_zero_prob)
            
            perr_list = [np.diagonal(x) for x in pcov_list]
            
            for param, v, err in zip(params_list, np.transpose(popt_list), np.transpose(perr_list)):
                col_name = param
                for i, lig in enumerate(ligand_list):
                    col_name = col_name.replace(f"_{i+1}", f"_{lig}")
                barcode_frame[col_name] = v
                barcode_frame[f"{col_name}_err"] = err
                
            for param, q, samp in zip(quantile_params_list, np.array(quantiles_list).transpose([1, 0, 2]), np.array(samples_out_list).transpose([1, 0, 2])):
                col_name = param
                for i, lig in enumerate(ligand_list):
                    col_name = col_name.replace(f"_{i+1}", f"_{lig}")
                barcode_frame[f"{col_name}_quantiles"] = list(q)
                barcode_frame[f"{col_name}_samples"] = list(samp)
                
            for i, lig in enumerate(ligand_list):
                barcode_frame[f"hill_invert_prob_{lig}"] = np.array(invert_prob_list).transpose()[i]
            
            barcode_frame["sensor_params_cov_all"] = pcov_list
            barcode_frame["hill_on_at_zero_prob"] = on_at_zero_prob_list
            barcode_frame["sensor_rms_residuals"] = residuals_list
        else:
            # TODO: update refits to Nov 2022, handle multiple ligands
            row_to_fit = barcode_frame.loc[refit_index]
            if return_fit:
                return stan_fit_row(row_to_fit, refit_index, ligand_list, return_fit=True)
            stan_popt, stan_pcov, stan_resid, stan_samples_out, stan_quantiles, hill_invert_prob, hill_on_at_zero_prob = stan_fit_row(row_to_fit, refit_index)
            arr_1 = barcode_frame.loc[refit_index, "sensor_params"]
            print(f"old: {arr_1}")
            arr_1 *= 0
            arr_1 += stan_popt
            new_test = barcode_frame.loc[refit_index, "sensor_params"]
            print(f"new: {new_test}")
            arr_2 = barcode_frame.loc[refit_index, "sensor_params_cov"]
            arr_2 *= 0
            arr_2 += stan_pcov
            arr_3 = barcode_frame.loc[refit_index, "sensor_rms_residuals"]
            arr_3 *= 0
            arr_3 += stan_resid
            arr_4 = barcode_frame.loc[refit_index, "sensor_stan_samples"]
            arr_4 *= 0
            arr_4 += stan_samples_out
            arr_5 = barcode_frame.loc[refit_index, "sensor_params_quantiles"]
            arr_5 *= 0
            arr_5 += stan_quantiles
            barcode_frame.loc[refit_index, "hill_invert_prob"] = hill_invert_prob
            barcode_frame.loc[refit_index, "hill_on_at_zero_prob"] = hill_on_at_zero_prob
        
        self.barcode_frame = barcode_frame
        
        if auto_save:
            self.save_as_pickle()
        
            
    def stan_GP_curves(self,
                       includeChimeras=False,
                       stan_GP_model='gp-hill-nomean-constrained.stan',
                       fit_fitness_difference_params=None,
                       control=dict(adapt_delta=0.9),
                       iterations=1000,
                       chains=4,
                       auto_save=True,
                       refit_index=None,
                       plasmid="pVER",
                       return_fit=False):
            
        print(f"Using Stan to fit to fitness curves with GP model for {self.experiment}")
        print(f"  Using fitness parameters for {plasmid}")
        print("      Method version from 2022-11-25")
        
        barcode_frame = self.barcode_frame
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
        
        fitness_columns_setup = self.get_fitness_columns_setup()
        
        if fitness_columns_setup[0]:
            old_style_columns, x, linthresh, fit_plot_colors, ligand_list, antibiotic_conc_list = fitness_columns_setup
            
            low_fitness = fit_fitness_difference_params[0]
            mid_g = fit_fitness_difference_params[1]
            fitness_n = fit_fitness_difference_params[2]
        else:
            old_style_columns, linthresh, fit_plot_colors, antibiotic_conc_list, plot_df, ligand_list = fitness_columns_setup
            
        if len(ligand_list) == 1:
            stan_GP_model = 'gp-hill-nomean-constrained.stan'
        
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = fitness.fit_fitness_difference_params(plasmid=plasmid, tet_conc=antibiotic_conc_list[1])
            
            params_list = ['low_fitness_high_tet', 'mid_g_high_tet', 'fitness_n_high_tet', 'log_rho', 'log_alpha', 'log_sigma']
            
            g_arr_list = ['constr_log_g']
            dg_arr_list = ['dlog_g']
            f_arr_list = ['mean_y']
            
            x_dim = 12 # number of concentrations for each ligand, including zero
                
        elif len(ligand_list) == 2:
            stan_GP_model = 'gp-hill-nomean-constrained.two-lig.two-tet.stan'
        
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = [fitness.fit_fitness_difference_params(plasmid=plasmid, tet_conc=x) for x in antibiotic_conc_list[1:]]
            
            params_list = ['low_fitness_low_tet', 'mid_g_low_tet', 'fitness_n_low_tet', 
                           'low_fitness_high_tet', 'mid_g_high_tet', 'fitness_n_high_tet', 
                           'log_rho', 'log_alpha', 'log_sigma']
            g_arr_list = [f'log_g_{i}' for i in [1, 2]]
            dg_arr_list = [f'dlog_g_{i}' for i in [1, 2]]
            f_arr_list = ['y_1_out_low_tet',  'y_1_out_high_tet',  'y_2_out_low_tet', 'y_2_out_high_tet']
            
            x_dim = 6 # number of concentrations for each ligand, including zero
            
        params_dim = len(params_list)
        
        quantile_list = [0.05, 0.25, 0.5, 0.75, 0.95]
        quantile_dim = len(quantile_list)
        
        stan_model = stan_utility.compile_model(stan_GP_model)
        
        if old_style_columns:
            high_tet = antibiotic_conc_list[1]
        
        log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf = fitness.log_g_limits(plasmid=plasmid)
        
        rng = np.random.default_rng()
        def stan_fit_row(st_row, st_index, lig_list, return_fit=False):
            print()
            print(f"fitting row index: {st_index}, for ligands: {lig_list}")
            
            stan_data = get_stan_data(st_row, plot_df, antibiotic_conc_list, lig_list, fit_fitness_difference_params, old_style_columns=old_style_columns, initial="b", plasmid=plasmid, is_gp_model=True)
        
            try:
            single_tet = len(antibiotic_conc_list)==2
            single_ligand = len(lig_list) == 1
                stan_init = [ init_stan_GP_fit(fit_fitness_difference_params, single_tet=single_tet, single_ligand=single_ligand) for i in range(chains) ]
                
                stan_fit = stan_model.sampling(data=stan_data, iter=iterations, init=stan_init, chains=chains, control=control)
                if return_fit:
                    return stan_fit
                    
                g_arr = [stan_fit[x] for x in g_arr_list]
                dg_arr = [stan_fit[x] for x in dg_arr_list]
                f_arr = [stan_fit[x] for x in f_arr_list]
                
                stan_g = [np.array([ np.quantile(a, q, axis=0) for q in quantile_list ]) for a in g_arr]
                stan_dg = [np.array([ np.quantile(a, q, axis=0) for q in quantile_list ]) for a in dg_arr]
                stan_f = [np.array([ np.quantile(a, q, axis=0) for q in quantile_list ]) for a in f_arr]
                
                stan_g_var = [np.var(a, axis=0) for a in g_arr]
                stan_dg_var = [np.var(a, axis=0) for a in dg_arr]
                
                stan_g_samples = [rng.choice(a, size=32, replace=False, axis=0, shuffle=False).transpose() for a in g_arr]
                stan_dg_samples = [rng.choice(a, size=32, replace=False, axis=0, shuffle=False).transpose() for a in dg_arr]
                    
                params_arr = np.array([stan_fit[x] for x in params_list])
        
                stan_popt = np.array([np.median(s) for s in params_arr ])
                stan_pcov = np.cov(params_arr, rowvar=True)
                
                
                stan_resid = np.median(stan_fit["rms_resid"])
            except:
                stan_g = [np.full((quantile_dim, x_dim), np.nan) for i in range(len(g_arr_list))]
                stan_dg = [np.full((quantile_dim, x_dim), np.nan) for i in range(len(dg_arr_list))]
                stan_f = [np.full((quantile_dim, x_dim), np.nan) for i in range(len(f_arr_list))]
                
                stan_g_var = [np.full(x_dim, np.nan) for i in range(len(g_arr_list))]
                stan_dg_var = [np.full(x_dim, np.nan) for i in range(len(dg_arr_list))]
                
                stan_g_samples = [np.full(x_dim, 32), np.nan) for i in range(len(g_arr_list))]
                stan_dg_samples = [np.full(x_dim, 32), np.nan) for i in range(len(dg_arr_list))]
                
                stan_popt = np.full(params_dim, np.nan)
                stan_pcov = np.full((params_dim, params_dim), np.nan)
                
                stan_resid = np.nan
                print(f"Error during Stan fitting for index {st_index}:", sys.exc_info()[0])
                
            return (stan_popt, stan_pcov, stan_resid, stan_g, stan_dg, stan_f, stan_g_var, stan_dg_var, stan_g_samples, stan_dg_samples)
        
        if refit_index is None:
            fit_list = [ stan_fit_row(row, index, ligand_list) for (index, row) in barcode_frame.iterrows() ]
            
            popt_list = []
            pcov_list = []
            
            stan_g_list = []
            stan_dg_list = []
            stan_f_list = []
            
            residuals_list = []
                
            stan_g_var_list = []
            stan_dg_var_list = []
            
            stan_g_samples_list = []
            stan_dg_samples_list = []
            
            for item in fit_list: # iterate over barcodes
                stan_popt, stan_pcov, stan_resid, stan_g, stan_dg, stan_f, stan_g_var, stan_dg_var, stan_g_samples, stan_dg_samples = item
                
                popt_list.append(stan_popt)
                pcov_list.append(stan_pcov)
                
                stan_g_list.append(stan_g)
                stan_dg_list.append(stan_dg)
                stan_f_list.append(stan_f)
                
                residuals_list.append(stan_resid)
                
                stan_g_var_list.append(stan_g_var)
                stan_dg_var_list.append(stan_dg_var)
                
                stan_g_samples_list.append(stan_g_samples)
                stan_dg_samples_list.append(stan_dg_samples)
            
            stan_g_list = np.array(stan_g_list).transpose([1,0,2,3])
            stan_g_var_list = np.array(stan_g_var_list).transpose([1,0,2])
            stan_g_samples_list = np.array(stan_g_samples_list).transpose([1,0,2,3])
            
            stan_dg_list = np.array(stan_dg_list).transpose([1,0,2,3])
            stan_dg_var_list = np.array(stan_dg_var_list).transpose([1,0,2])
            stan_dg_samples_list = np.array(stan_dg_samples_list).transpose([1,0,2,3])
            
            stan_f_list = np.array(stan_f_list).transpose([1,0,2,3])
            
            barcode_frame["GP_params"] = popt_list
            barcode_frame["GP_cov"] = pcov_list
            
            for g_list, param, g_var, g_samp in zip(stan_g_list, g_arr_list, stan_g_var_list, stan_g_samples_list):
                col_name = param
                if col_name == 'constr_log_g':
                    col_name = 'log_g_1'
                for i, lig in enumerate(ligand_list):
                    col_name = col_name.replace(f"_{i+1}", f"_{lig}")
                col_name = f"GP_{col_name}"
                barcode_frame[col_name] = list(g_list)
                barcode_frame[f"{col_name}_var"] = list(g_var)
                barcode_frame[f"{col_name}_samp"] = list(g_samp)
            
            for dg_list, param, dg_var, dg_samp in zip(stan_dg_list, dg_arr_list, stan_dg_var_list, stan_dg_samples_list):
                col_name = param
                if col_name == 'dlog_g':
                    col_name = 'dlog_g_1'
                for i, lig in enumerate(ligand_list):
                    col_name = col_name.replace(f"_{i+1}", f"_{lig}")
                col_name = f"GP_{col_name}"
                barcode_frame[col_name] = list(dg_list)
                barcode_frame[f"{col_name}_var"] = list(dg_var)
                barcode_frame[f"{col_name}_samp"] = list(dg_samp)
            
            for f_list, param in zip(stan_f_list, f_arr_list):
                col_name = param
                if col_name == 'mean_y':
                    col_name = 'y_1_out_high_tet'
                for i, lig in enumerate(ligand_list):
                    col_name = col_name.replace(f"_{i+1}_out", f"_{lig}")
                col_name = f"GP_{col_name}"
                barcode_frame[col_name] = list(f_list)
            
            barcode_frame["GP_residuals"] = residuals_list
        else:
            # TODO: update refits to Nov 2022, handle multiple ligands
            row_to_fit = barcode_frame.loc[refit_index]
            if return_fit:
                return stan_fit_row(row_to_fit, refit_index, ligand_list, return_fit=True)
            stan_popt, stan_pcov, stan_resid, stan_g, stan_dg, stan_f, stan_g_var, stan_dg_var, stan_g_samples, stan_dg_samples = stan_fit_row(row_to_fit, refit_index)
            
            arr_1 = barcode_frame.loc[refit_index, "sensor_GP_params"]
            print(f"old: {arr_1}")
            arr_1 *= 0
            arr_1 += stan_popt
            new_test = barcode_frame.loc[refit_index, "sensor_GP_params"]
            print(f"new: {new_test}")
            
            arr_2 = barcode_frame.loc[refit_index, "sensor_GP_cov"]
            arr_2 *= 0
            arr_2 += stan_pcov
            
            arr_3 = barcode_frame.loc[refit_index, "sensor_GP_residuals"]
            arr_3 *= 0
            arr_3 += stan_resid
            
            arr_4 = barcode_frame.loc[refit_index, "sensor_GP_g_quantiles"]
            arr_4 *= 0
            arr_4 += stan_g
            
            arr_5 = barcode_frame.loc[refit_index, "sensor_GP_Df_quantiles"]
            arr_5 *= 0
            arr_5 += stan_f
            
            arr_6 = barcode_frame.loc[refit_index, "sensor_GP_Dg_quantiles"]
            arr_6 *= 0
            arr_6 += stan_dg
            
            arr_7 = barcode_frame.loc[refit_index, "sensor_GP_g_var"]
            arr_7 *= 0
            arr_7 += stan_g_var
            
            arr_8 = barcode_frame.loc[refit_index, "sensor_GP_dg_var"]
            arr_8 *= 0
            arr_8 += stan_dg_var
            
            arr_9 = barcode_frame.loc[refit_index, "sensor_GP_g_samples"]
            arr_9 *= 0
            arr_9 += stan_g_samples
            
            arr_10 = barcode_frame.loc[refit_index, "sensor_GP_dg_samples"]
            arr_10 *= 0
            arr_10 += stan_dg_samples
        
        self.barcode_frame = barcode_frame
        
        if auto_save:
            self.save_as_pickle()
            
        
            
    def merge_barcodes(self, small_bc_index, big_bc_index, auto_refit=True, auto_save=True, ignore_samples=[]):
        # merge small barcode into big barcode (add read counts)
        # remove small barcode from dataframe
        
        barcode_frame = self.barcode_frame
        high_tet = self.high_tet
        
        columns_to_sum = fitness.wells()
        columns_to_sum += [ 'total_counts', 'fraction_total', 'total_counts_plate_2', 'fraction_total_p2' ]
        columns_to_sum += [ f'fraction_{w}' for w in fitness.wells() ]
        columns_to_sum += [ f'read_count_{0}_{plate_num}' for plate_num in range(2,6) ]
        columns_to_sum += [ f'read_count_{high_tet}_{plate_num}' for plate_num in range(2,6) ]
        
        
        print(f"merging {small_bc_index} into {big_bc_index}")
        
        for col in columns_to_sum:
            if col in barcode_frame.columns:
                if type(barcode_frame.loc[big_bc_index, col]) != type(np.array([1])):
                    barcode_frame.loc[big_bc_index, col] += barcode_frame.loc[small_bc_index, col]
                else:
                    big_bc_arr = barcode_frame.loc[big_bc_index, col]
                    small_bc_arr = barcode_frame.loc[small_bc_index, col]
                    big_bc_arr += small_bc_arr
                
        barcode_frame.drop(small_bc_index, inplace=True)
        print(f"dropping {small_bc_index}")
        
        self.barcode_frame = barcode_frame
        
        if auto_save:
            self.save_as_pickle()
            
        if auto_refit:
            self.fit_barcode_fitness(auto_save=auto_save, ignore_samples=ignore_samples, refit_index=big_bc_index)
            self.stan_fitness_difference_curves(auto_save=auto_save, refit_index=big_bc_index)
        
        
            
    def plot_count_hist(self, hist_bin_max=None, num_bins=50, save_plots=False, pdf_file=None):
        
        barcode_frame = self.barcode_frame
        
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        os.chdir(self.data_directory)
        if save_plots:
            if pdf_file is None:
                pdf_file = 'barcode histogram plot.pdf'
            pdf = PdfPages(pdf_file)
        
        if hist_bin_max is None:
            hist_bin_max = barcode_frame[int(len(barcode_frame)/50):int(len(barcode_frame)/50)+1]["total_counts"].values[0]
        
        #Plot histogram of Barcode counts to enable decision about threshold
        plt.rcParams["figure.figsize"] = [16,8]
        fig, axs = plt.subplots(1, 2)
        bins = np.linspace(-0.5, hist_bin_max + 0.5, num_bins)
        for ax in axs.flatten():
            ax.hist(barcode_frame['total_counts'], bins=bins);
            ax.set_xlabel('Barcode Count', size=20)
            ax.set_ylabel('Number of Barcodes', size=20)
            ax.tick_params(labelsize=16);
        axs[0].hist(barcode_frame['total_counts'], bins=bins, histtype='step', cumulative=-1);
        axs[0].set_yscale('log');
        axs[1].set_yscale('log');
        axs[1].set_xlim(0,hist_bin_max/3);
            
        if save_plots:
            pdf.savefig()
                
        if save_plots:
            pdf.close()
        
    def plot_read_counts(self, num_to_plot=None, save_plots=False, pdf_file=None, vmin=None):
        
        barcode_frame = self.barcode_frame
        
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        os.chdir(self.data_directory)
        if save_plots:
            if pdf_file is None:
                pdf_file = 'barcode read count plots.pdf'
            pdf = PdfPages(pdf_file)
    
        BC_totals = []
        index_list = []
        for i, w in enumerate(fitness.wells()):
            BC_totals.append(barcode_frame[w].sum())
            index_list.append(i+1)
        
        BC_total_arr = []
        for r in fitness.rows():
            subarr = []
            for c in fitness.columns():
                subarr.append(barcode_frame[r + str(c)].sum())
            BC_total_arr.append(subarr)
    
        #Plot barcode read counts across plate
        plt.rcParams["figure.figsize"] = [12,16]
        fig, axs = plt.subplots(2, 1)
    
        r12 = np.asarray(np.split(np.asarray(BC_totals), 8)).transpose().flatten()
    
        axs[0].scatter(index_list, r12, c=plot_colors96(), s=70);
        for i in range(13):
            axs[0].plot([i*8+0.5, i*8+0.5],[min(BC_totals), max(BC_totals)], color='gray');
        axs[0].set_title("Total Read Counts Per Sample", fontsize=32)
        #axs[0].set_yscale('log');
    
        axs[0].set_xlim(0,97);
        axs[0].set_xlabel('Sample Number', size=20)
        axs[0].set_ylabel('Total Reads per Sample', size=20);
        axs[0].tick_params(labelsize=16);
    
        axs[1].matshow(BC_total_arr, cmap="inferno", vmin=vmin);
        axs[1].grid(b=False);
        axs[1].set_xticklabels([i+1 for i in range(12)], size=16);
        axs[1].set_xticks([i for i in range(12)]);
        axs[1].set_yticklabels([ r + " " for r in fitness.rows()[::-1] ], size=16);
        axs[1].set_yticks([i for i in range(8)]);
        axs[1].set_ylim(-0.5, 7.5);
        axs[1].tick_params(length=0);
        
        if save_plots:
            pdf.savefig()
                
        if save_plots:
            pdf.close()
            
    def plot_read_fractions(self,
                            save_plots=False,
                            num_to_plot=5,
                            plot_range=None,
                            plot_size=16,
                            plot_fraction=True,
                            marker_size=70):
    
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        if save_plots:
            pdf_file = 'barcode read fraction plots.pdf'
            pdf = PdfPages(pdf_file)
    
        os.chdir(self.data_directory)
        
        if plot_range is None:
            f_data = self.barcode_frame.iloc[:num_to_plot]
        else:
            f_data = self.barcode_frame.loc[plot_range[0]:plot_range[1]]
        
        #Plot read fraction across all samples for first several barcodes
        plt.rcParams["figure.figsize"] = [plot_size,6*len(f_data)*plot_size/16]
        fig, axs = plt.subplots(len(f_data), 1)
        
        inducer = self.inducer
        inducer_conc_list = self.inducer_conc_list
        high_tet = self.high_tet
        
        low_tet = getattr(self, 'low_tet', None)
        inducer_2 = getattr(self, 'inducer_2', None)
        inducer_conc_list_2 = getattr(self, 'inducer_conc_list_2', None)
        if (low_tet is None) and (inducer_2 is None):
            sample_plate_map = fitness.get_sample_plate_map(inducer, inducer_conc_list, tet_conc_list=[high_tet])
        else:
            sample_plate_map = fitness.get_sample_plate_map(inducer, inducer_conc_list,
                                                            inducer_2=inducer_2, inducer_conc_list_2=inducer_conc_list_2, tet_conc_list=[low_tet, high_tet])
        
        tet_list = np.unique(sample_plate_map.antibiotic_conc)
        if plot_fraction:
            plot_param = "fraction_"
        else:
            plot_param = ""
        for (index, row), ax in zip(f_data.iterrows(), axs):
            y_for_scale = []
            for marker, tet in zip(['o', '<', '>'], tet_list):
                y = []
                x = []
                c = []
                for i, w in enumerate(fitness.wells_by_column()):
                    col = int(w[1:])
                    df = sample_plate_map
                    df = df[df.well==w]
                    df = df[df.antibiotic_conc==tet]
                    if len(df)>0:
                        y.append(row[plot_param + w])
                        x.append(i+1)
                        c.append(plot_colors()[col-1])
                    if (row[plot_param + w])>0:
                        y_for_scale.append(row[plot_param + w])
        
                ax.scatter(x, y, c=c, s=marker_size, marker=marker);
            ax.set_ylim(0.5*min(y_for_scale), 2*max(y));
            ax.set_yscale("log")
            barcode_str = str(index) + ', '
            if row['RS_name'] != "": barcode_str += row['RS_name'] + ", "
            barcode_str += row['forward_BC'] + ', ' + row['reverse_BC']
            ax.text(x=0.05, y=0.95, s=barcode_str, horizontalalignment='left', verticalalignment='top',
                     transform=ax.transAxes, fontsize=plot_size)
        
            for i in range(13):
                ax.plot([i*8+0.5, i*8+0.5],[0.6*min(y_for_scale), 1.2*max(y)], color='gray');
        if plot_fraction:
            axs[0].set_title("Read Fraction Per Barcode", fontsize=2*plot_size);
        else:
            axs[0].set_title("Read Count Per Barcode", fontsize=2*plot_size);
        if save_plots:
            pdf.savefig()
    
        if save_plots:
            pdf.close()
            
    def plot_stdev(self,
                   save_plots=False,
                   count_cutoff=500,
                   experiment=None,
                   includeChimeras=False):
        
        if experiment is None:
            experiment = self.experiment
    
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        os.chdir(self.data_directory)
        
        if save_plots:
            pdf_file = 'barcode read standard deviation plots.pdf'
            pdf = PdfPages(pdf_file)
            
        #Plot Barcode fraction for each well in time point 1 vs. mean fraction in time point 1
        plt.rcParams["figure.figsize"] = [16,16]
        fig, axs = plt.subplots(2, 2)
        
        f_data = self.barcode_frame[self.barcode_frame["total_counts"]>count_cutoff]
        if (not includeChimeras) and ("isChimera" in f_data.columns):
            f_data = f_data[f_data["isChimera"] == False]
            
        f_x = f_data['fraction_total_p2']
        f_x_min = f_data[f_data['fraction_total_p2']>0]['fraction_total_p2'].min()
        for i, w in enumerate(fitness.wells_by_column()[:24]):
            c = [(plot_colors()*8)[i]]*len(f_data)
            for ax in axs.flatten()[:2]:
                ax.scatter(f_x, f_data['fraction_' + w], c=c)
            for ax in axs.flatten()[2:4]:
                ax.scatter(f_x, (f_data['fraction_' + w] - f_x)*100, c=c)
                
        for ax in axs.flatten()[:2]:
            x_lim_0 = ax.get_xlim()
            ax.plot([0, x_lim_0[1]], [0, x_lim_0[1]], color='k')
        for ax in axs.flatten()[2:4]:
            x_lim_0 = ax.get_xlim()
            ax.plot([0, x_lim_0[1]], [0,0], color='k')
            
        axs.flatten()[1].set_xscale("log");
        axs.flatten()[1].set_yscale("log");
        axs.flatten()[1].set_xlim(f_x_min/1.3, x_lim_0[1]*1.3);
        axs.flatten()[1].set_ylim(f_x_min/1.3, x_lim_0[1]*1.3);
    
        axs.flatten()[3].set_xscale("log");
        axs.flatten()[3].set_xlim(f_x_min/1.3, x_lim_0[1]*1.3);
        fig.suptitle('Fraction from Each Dual Barcode (Plate 2)', fontsize=24, position=(0.5, 0.905))
    
        for ax in axs.flatten()[:2]:
            ax.set_xlabel('Fraction Total', size=20)
            ax.set_ylabel('Fraction per Sample', size=20);
            ax.tick_params(labelsize=16);
        
        for ax in axs.flatten()[2:4]:
            ax.set_xlabel('Fraction Total', size=20)
            ax.set_ylabel('Fraction per Sample - Fraction Total (%)', size=20);
            ax.tick_params(labelsize=16);
        if save_plots:
            pdf.savefig()
    
        # data from 2019-10-02: #################################################################################
        x_test = np.asarray([0.23776345382258504, 0.21428834768303265, 0.14955568743012018, 0.10527042635253019, 0.08814193520270863,
                             0.07140559171457407, 0.032268913991628186, 0.02486533840744069, 0.009370452839984682, 0.0021539027931815613,
                             0.0001936817014361814])
        y_test = np.asarray([0.0019726945744597706, 0.0028398295224567756, 0.0027140121666701543, 0.0016422861817864806,
                             0.0012364410886752844, 0.0014467832918787287, 0.0009412184378809117, 0.0007090217957749182,
                             0.00034552377974558844, 0.00017198555940160456, 4.958998052635534e-05])
        poisson_err_test = np.asarray([0.001391130466104952, 0.001320415964490587, 0.0011032026255463198, 0.0009247685041703838,
                                       0.0008466282838575875, 0.0007620910541483005, 0.0005123905962175842, 0.000449754496329767,
                                       0.00027605091052578906, 0.0001323496187650663, 3.929704870026295e-05])
        #####################################################################################################
    
        # data from 2019-10-08: #############################################################################
        x_small = np.asarray([0.08251274176535274, 0.0962239061597132, 0.08539004578198717, 0.08675701439383578, 0.07400424816228543,
                              0.07566109361860245, 0.0699367739242362, 0.06963680434271374, 0.06384195016208481, 0.06321931248609224,
                              0.06334894239678983, 0.02536420185939611, 0.03923343837910993, 0.020238576239101202])
        y_small = np.asarray([0.003020200426682457, 0.003374150359051314, 0.00374541788260866, 0.0035764736646941536,
                              0.002598176841078495, 0.003669639858790278, 0.0021759993522437074, 0.002827475646549457,
                              0.0038335541520843315, 0.002201298340428577, 0.008012477386731139, 0.001454772893578839,
                              0.0012788004626381614, 0.0021763030793714206])
        poisson_err_small = np.asarray([0.0008661333092282185, 0.0009340439480853888, 0.0008821889073372234, 0.0008856945951456786,
                                        0.000820757229296616, 0.000830315430739499, 0.0007963057526756344, 0.0007963629310250612,
                                        0.000763102677224598, 0.0007575749124137182, 0.0007546065015548847, 0.0004797418835729835,
                                        0.000596486425619687, 0.00042833165436399073])
        ##############################################################################################
    
        #Plot standard deviation of barcode read fractions (across wells in time point 1) vs mean read fraction 
        plt.rcParams["figure.figsize"] = [16,8]
        fig, axs = plt.subplots(1, 2)
        #fig.suptitle('First Time Point Only (Plate 2)', fontsize=24, position=(0.5, 0.925))
    
        axs[0].plot(x_test, y_test, "o", ms=10, label="Library Prep Test, 2019-10-02");
        axs[0].plot(x_test, poisson_err_test, c="gray");
        axs[0].plot(x_small, y_small, "o", ms=10, label="Small Library Selection, 2019-10-08");
        axs[1].plot(x_test, y_test/x_test, "o", ms=10, label="Library Prep Test, 2019-10-02");
        axs[1].plot(x_test, poisson_err_test/x_test, c="gray");
        axs[1].plot(x_small, y_small/x_small, "o", ms=10, label="Small Library Selection, 2019-10-08");
    
        y = f_data["fraction_p2_std"]
        x = f_data["fraction_total_p2"]
        err_est = f_data["fraction_total_p2"] / np.sqrt(f_data["total_counts_plate_2"]/24)
        
        axs[0].plot(x, y, "o", ms=5, label = experiment);
        axs[0].plot(x, err_est, c="darkgreen");
        axs[0].set_ylabel('Stdev(barcode fraction per sample)', size=20);
        axs[1].plot(x, y/x, "o", ms=5, label = experiment);
        axs[1].plot(x, err_est/x, c="darkgreen");
        axs[1].set_ylabel('Relative Stdev(barcode fraction per sample)', size=20);
    
        for ax in axs.flatten():
            ax.set_xlabel('Mean(barcode fraction per sample)', size=20);
            ax.tick_params(labelsize=16);
            ax.set_xscale("log");
            ax.set_yscale("log");
        leg = axs[0].legend(loc='lower right', bbox_to_anchor= (0.99, 0.01), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
        leg.get_frame().set_edgecolor('k');
        leg = axs[1].legend(loc='lower left', bbox_to_anchor= (0.01, 0.01), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
        leg.get_frame().set_edgecolor('k');
        if save_plots:
            pdf.savefig()
    
        if save_plots:
            pdf.close()
    
    def plot_fitness_curves(self,
                            save_plots=False,
                            plot_range=None,
                            include_ref_seqs=True,
                            includeChimeras=False,
                            ylim=None,
                            plot_size=[8, 6],
                            fontsize=13,
                            ax_label_size=14,
                            show_bc_str=False,
                            real_fitness_units=False):
        
        if plot_range is None:
            barcode_frame = self.barcode_frame
        else:
            barcode_frame = self.barcode_frame.loc[plot_range[0]:plot_range[1]]
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        if include_ref_seqs:
            RS_count_frame = self.barcode_frame[self.barcode_frame["RS_name"]!=""]
            barcode_frame = pd.concat([barcode_frame, RS_count_frame])
            
        if real_fitness_units:
            fit_scale = fitness.fitness_scale()
            fit_units = '1/h'
        else:
            fit_scale = 1
            fit_units = 'log(10)/plate'
            
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        if save_plots:
            os.chdir(self.data_directory)
            pdf_file = 'barcode fitness plots.pdf'
            pdf = PdfPages(pdf_file)
        
        #plot fitness curves
        num_plot_rows = int(np.round(len(barcode_frame)/2 + 0.1))
        plt.rcParams["figure.figsize"] = [2*plot_size[0],plot_size[1]*num_plot_rows]
        fig, axs_grid = plt.subplots(num_plot_rows, 2)
        plt.subplots_adjust(hspace = .35)
        axs = axs_grid.flatten()
        #if len(barcode_frame)==1:
        #    axs = [ axs ]
        
        fitness_columns_setup = self.get_fitness_columns_setup()
        if fitness_columns_setup[0]:
            old_style_plots, x, linthresh, fit_plot_colors, ligand_list, antibiotic_conc_list = fitness_columns_setup
        else:
            old_style_plots, linthresh, fit_plot_colors, antibiotic_conc_list, plot_df, ligand_list = fitness_columns_setup
        
        for (index, row), ax in zip(barcode_frame.iterrows(), axs): # iterate over barcodes
            for initial in ["b", "e"]:
                fill_style = "full" if initial=="b" else "none"
                if old_style_plots:
                    for tet, color in zip(antibiotic_conc_list, fit_plot_colors):
                        y = row[f"fitness_{tet}_estimate_{initial}"]*fit_scale
                        s = row[f"fitness_{tet}_err_{initial}"]*fit_scale
                        ax.errorbar(x, y, s, marker='o', ms=8, color=color, fillstyle=fill_style)
                else:
                    for tet, color in zip(antibiotic_conc_list, fit_plot_colors):
                        for lig, marker in zip(ligand_list, ['o', '<', '>']):
                            df = plot_df
                            df = df[(df.ligand==lig)|(df.ligand=='none')]
                            df = df[df.antibiotic_conc==tet]
                            x = df[lig]
                            y = [row[f"fitness_S{i}_{initial}"]*fit_scale for i in df.sample_id]
                            s = [row[f"fitness_S{i}_err_{initial}"]*fit_scale for i in df.sample_id]
                            ax.errorbar(x, y, s, marker=marker, ms=8, color=color, fillstyle=fill_style)
            
                if initial == "b":
                    barcode_str = str(index) + ': '
                    barcode_str += format(row[f'total_counts'], ",") + "; "
                    barcode_str += row['RS_name']
                    if show_bc_str:
                        barcode_str += ": " + row['forward_BC'] + ",\n"
                        barcode_str += row['reverse_BC'] + " "
                        fontfamily = "Courier New"
                    else:
                        fontfamily = None
                    ax.text(x=1, y=1.1, s=barcode_str, horizontalalignment='right', verticalalignment='top',
                            transform=ax.transAxes, fontsize=fontsize, fontfamily=fontfamily)
                    ax.set_xscale('symlog', linthresh=linthresh)
                    x_lab = '], ['.join(ligand_list)
                    ax.set_xlabel(f'[{x_lab}] (umol/L)', size=ax_label_size)
                    ax.set_ylabel(f'Growth Rate ({fit_units})', size=ax_label_size)
                    ax.tick_params(labelsize=ax_label_size-2);
                    if ylim is not None:
                        ax.set_ylim(ylim);
            
        if save_plots:
            pdf.savefig()
    
        if save_plots:
            pdf.close()
            
        return fig, axs_grid
        
    def get_fitness_columns_setup(self):
        barcode_frame = self.barcode_frame
        
        high_tet = self.high_tet
        low_tet = getattr(self, 'low_tet', None)
        
        inducer = self.inducer
        inducer_2 = getattr(self, 'inducer_2', None)
        
        inducer_conc_list = self.inducer_conc_list
        inducer_conc_list_2 = getattr(self, 'inducer_conc_list_2', None)
        
        # old_style_plots indicates whether to use the old style column headings (i.e., f"fitness_{high_tet}_estimate_{initial}")
        #     or the new style (i.e., f"fitness_S{i}_{initial}"
        # The new style is preferred, so will be used if both are possible
        old_style_plots = False
        for initial in ['b', 'e']:
            for i  in range(1, 25):
                c = f"fitness_S{i}_{initial}"
                old_style_plots = old_style_plots or (c not in barcode_frame.columns)
        if old_style_plots:
            print("Using old style column headings")
        else:
            print("Using new style column headings")
        
        fit_plot_colors = sns.color_palette()
        
        if old_style_plots:
            x = np.array(self.inducer_conc_list)
            linthresh = min(x[x>0])
            
            ligand_list = [inducer]
            
            antibiotic_conc_list = np.array([0, high_tet])
            
            return old_style_plots, x, linthresh, fit_plot_colors, ligand_list, antibiotic_conc_list
        else:
            if (low_tet is None) and (inducer_2 is None):
                sample_plate_map = fitness.get_sample_plate_map(inducer, inducer_conc_list, tet_conc_list=[high_tet])
            else:
                sample_plate_map = fitness.get_sample_plate_map(inducer, inducer_conc_list,
                                                                inducer_2=inducer_2, inducer_conc_list_2=inducer_conc_list_2, tet_conc_list=[low_tet, high_tet])
            ligand_list = list(np.unique(sample_plate_map.ligand))
            if 'none' in ligand_list:
                ligand_list.remove('none')
            antibiotic_conc_list = np.unique(sample_plate_map.antibiotic_conc)
            
            plot_df = sample_plate_map
            plot_df = plot_df[plot_df.growth_plate==2].sort_values(by=ligand_list)
            
            x_list = np.array([np.array(plot_df[x]) for x in ligand_list]).flatten()
            linthresh = min(x_list[x_list>0])
            
            return old_style_plots, linthresh, fit_plot_colors, antibiotic_conc_list, plot_df, ligand_list
    

    def plot_fitness_and_difference_curves(self,
                            save_plots=False,
                            plot_range=None,
                            include_ref_seqs=True,
                            includeChimeras=False,
                            ylim = None,
                            show_fits=True,
                            show_GP=False,
                            log_g_scale=False,
                            box_size=6,
                            show_bc_str=False):
        
        if plot_range is None:
            barcode_frame = self.barcode_frame
        else:
            barcode_frame = self.barcode_frame.loc[plot_range[0]:plot_range[1]]
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        if include_ref_seqs:
            RS_count_frame = self.barcode_frame[self.barcode_frame["RS_name"]!=""]
            barcode_frame = pd.concat([barcode_frame, RS_count_frame])
            
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        if save_plots:
            os.chdir(self.data_directory)
            pdf_file = 'barcode fitness plots.pdf'
            pdf = PdfPages(pdf_file)
        
        #plot fitness curves
        fitness_columns_setup = self.get_fitness_columns_setup()
        if fitness_columns_setup[0]:
            old_style_plots, x, linthresh, fit_plot_colors, ligand_list, antibiotic_conc_list = fitness_columns_setup
            if "sensor_params" not in barcode_frame.columns:
                show_fits = False
        else:
            old_style_plots, linthresh, fit_plot_colors, antibiotic_conc_list, plot_df, ligand_list = fitness_columns_setup
            if "log_g0" not in barcode_frame.columns:
                show_fits = False
        
        if show_GP:
            plt.rcParams["figure.figsize"] = [2*box_size, 3*box_size/2]
        else:
            plt.rcParams["figure.figsize"] = [2*box_size, 3*box_size/4]
        
        if show_fits:
            fit_fitness_difference_params = self.fit_fitness_difference_params
            
            if old_style_plots:
                if len(barcode_frame["sensor_params"].iloc[0])==7:
                    # ['log_g0', 'log_ginf', 'log_ec50', 'log_sensor_n', 'low_fitness', 'mid_g',
                    #  'fitness_n']
                    def fit_funct(x, log_g_min, log_g_max, log_x_50, log_nx, low_fitness, mid_g, fitness_n, *argv):
                        return double_hill_funct(x, 10**log_g_min, 10**log_g_max, 10**log_x_50, 10**log_nx,
                                                 low_fitness, 0, mid_g, fitness_n)
                elif len(barcode_frame["sensor_params"].iloc[0])>7:
                    # ['log_g0', 'log_ginf', 'log_ec50', 'log_sensor_n', 'log_ginf_g0_ratio',
                    #  'low_fitness', 'mid_g', 'fitness_n']
                    def fit_funct(x, log_g_min, log_g_max, log_x_50, log_nx, log_ginf_g0_ratio,
                                  low_fitness, mid_g, fitness_n, *argv):
                        return double_hill_funct(x, 10**log_g_min, 10**log_g_max, 10**log_x_50, 10**log_nx,
                                                 low_fitness, 0, mid_g, fitness_n)
                else:
                    def fit_funct(x, g_min, g_max, x_50, nx):
                        return double_hill_funct(x, g_min, g_max, x_50, nx, fit_fitness_difference_params[0], 0,
                                                 fit_fitness_difference_params[1], fit_fitness_difference_params[2])
            else:
                def fit_funct(x, log_g0, log_ginf, log_ec50, log_nx, low_fitness, mid_g, fitness_n):
                    return double_hill_funct(x, 10**log_g0, 10**log_ginf, 10**log_ec50, 10**log_nx,
                                             low_fitness, 0, mid_g, fitness_n)
        
        
        fig_axs_list = []
        for index, row in barcode_frame.iterrows(): # iterate over barcodes
            if show_GP:
                fig, axs_grid = plt.subplots(2, 2)
                axl = axs_grid.flatten()[0]
                axr = axs_grid.flatten()[2]
                axg = axs_grid.flatten()[1]
                axdg = axs_grid.flatten()[3]
            else:
                fig, axs_grid = plt.subplots(1, 2)
                axl = axs_grid.flatten()[0]
                axr = axs_grid.flatten()[1]
                axg = axs_grid.flatten()[0]
                axdg = axs_grid.flatten()[0]
            plt.subplots_adjust(hspace = .35)
            
            for ax in axs_grid.flatten():
                ax.set_xscale('symlog', linthresh=linthresh)
                x_lab = '], ['.join(ligand_list)
                ax.set_xlabel(f'[{x_lab}] (umol/L)', size=14)
            
            for initial in ["b", "e"]:
                fill_style = "full" if initial=="b" else "none"
                if old_style_plots:
                    for tet, color in zip(antibiotic_conc_list, fit_plot_colors):
                        y = row[f"fitness_{tet}_estimate_{initial}"]
                        s = row[f"fitness_{tet}_err_{initial}"]
                        axl.errorbar(x, y, s, marker='o', ms=8, color=color, fillstyle=fill_style)
                    
                    y_zero = row[f"fitness_{0}_estimate_{initial}"]
                    s_zero = row[f"fitness_{0}_err_{initial}"]
                    y_high = row[f"fitness_{antibiotic_conc_list[1]}_estimate_{initial}"]
                    s_high = row[f"fitness_{antibiotic_conc_list[1]}_err_{initial}"]
                    
                    y = (y_high - y_zero)/y_zero.mean()
                    s = np.sqrt( s_high**2 + s_zero**2 )/y_zero.mean()
                    fill_style = "full" if initial=="b" else "none"
                    axr.errorbar(x, y, s, marker='o', ms=8, color=fit_plot_colors[0], fillstyle=fill_style)
                else:
                    y_ref_list = []
                    s_ref_list = []
                    for tet, color in zip(antibiotic_conc_list, fit_plot_colors):
                        for lig, marker in zip(ligand_list, ['o', '<', '>']):
                            df = plot_df
                            df = df[(df.ligand==lig)|(df.ligand=='none')]
                            df = df[df.antibiotic_conc==tet]
                            x = df[lig]
                            y = [row[f"fitness_S{i}_{initial}"] for i in df.sample_id]
                            s = [row[f"fitness_S{i}_err_{initial}"] for i in df.sample_id]
                            axl.errorbar(x, y, s, marker=marker, ms=8, color=color, fillstyle=fill_style)
                            if tet == 0:
                                y_ref_list += list(y)
                                s_ref_list += list(s)
                    
                    y_ref_list = np.array(y_ref_list)
                    w = 1/np.array(s_ref_list)**2
                    y_ref = np.average(y_ref_list, weights=w)
                    s_ref = np.average((y_ref_list-y_ref)**2, weights=w)
                    v_1 = np.sum(w)
                    v_2 = np.sum(w**2)
                    s_ref = np.sqrt(s_ref/(1 - (v_2/v_1**2)))
                    for tet, color in zip(antibiotic_conc_list, fit_plot_colors):
                        for lig, marker in zip(ligand_list, ['o', '<', '>']):
                            df = plot_df
                            df = df[(df.ligand==lig)|(df.ligand=='none')]
                            df = df[df.antibiotic_conc==tet]
                            x = df[lig]
                            y = np.array([row[f"fitness_S{i}_{initial}"] for i in df.sample_id])
                            y = (y - y_ref)/y_ref
                            s = np.array([row[f"fitness_S{i}_err_{initial}"] for i in df.sample_id])
                            s = np.sqrt(s**2 + s_ref**2)/y_ref
                            marker = marker if show_fits else '-' + marker
                            axr.errorbar(x, y, s, fmt=marker, ms=8, color=color, fillstyle=fill_style)
                
                if initial == "b":
                    barcode_str = str(index) + ': '
                    barcode_str += format(row[f'total_counts'], ",") + "; "
                    barcode_str += row['RS_name']
                    if show_bc_str:
                        barcode_str += ": " + row['forward_BC'] + ",\n"
                        barcode_str += row['reverse_BC'] + " "
                        fontfamily = "Courier New"
                    else:
                        fontfamily = None
                    if not old_style_plots:
                        barcode_str += f"\ny_ref: {y_ref:.3f} +- {s_ref:.3f}"
                    axl.text(x=1, y=1.025, s=barcode_str, horizontalalignment='center', verticalalignment='bottom',
                            transform=axl.transAxes, fontsize=13, fontfamily=fontfamily)
                    axl.set_ylabel('Fitness (log(10)/plate)', size=14)
                    axl.tick_params(labelsize=12);
                    axr.set_ylabel('Fitness with Tet - Fitness without Tet', size=14)
                    axr.tick_params(labelsize=12);
                    if ylim is not None:
                        axl.set_ylim(ylim);
                    
            if show_fits:
                if old_style_plots:
                    x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
                    x_fit = np.insert(x_fit, 0, 0)
                    params = row["sensor_params"]
                    y_fit = fit_funct(x_fit, *params)
                    axr.plot(x_fit, y_fit, color='k', zorder=1000);
                else:
                    tet_level_list = ['high'] if len(antibiotic_conc_list)==2 else ['low', 'high']
                    for tet, color in zip(tet_level_list, fit_plot_colors[1:]):
                        for lig in ligand_list:
                            df = plot_df
                            df = df[df.ligand==lig]
                            x = df[lig]
                            x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
                            x_fit = np.insert(x_fit, 0, 0)
                            
                            # fit_funct(x, log_g0, log_ginf, log_ec50, log_nx, low_fitness, mid_g, fitness_n)
                            params_list = ['log_g0', f'log_ginf_{lig}', f'log_ec50_{lig}', f'log_sensor_n_{lig}', 
                                           f'low_fitness_{tet}_tet', f'mid_g_{tet}_tet', f'fitness_n_{tet}_tet']
                            
                            params = [row[p] for p in params_list]
                            y_fit = fit_funct(x_fit, *params)
                            
                            axr.plot(x_fit, y_fit, color=color, zorder=1000);
                
            if show_GP:
                if old_style_plots:
                    stan_g = 10**row["sensor_GP_g_quantiles"]
                    stan_dg = row["sensor_GP_Dg_quantiles"]
                    stan_f = row["sensor_GP_Df_quantiles"]
                    
                    axr.plot(x, stan_f[2], color=fit_plot_colors[2])
                    
                    axg.plot(x, stan_g[2], color=fit_plot_colors[2])
                    axg.set_ylabel('GP Gene Epxression Estimate (MEF)', size=14)
                    axg.tick_params(labelsize=12);
                    if log_g_scale: axg.set_yscale("log")
                    
                    axdg.plot([x[0],x[-1]], [0,0], c='k');
                    axdg.plot(x, stan_dg[2], color=fit_plot_colors[3])
                    axdg.set_ylabel('GP d(log(g))/d(log(x))', size=14)
                    axdg.tick_params(labelsize=12);
                    for i in range(1,3):
                        axr.fill_between(x, stan_f[2-i], stan_f[2+i], alpha=.3, color=fit_plot_colors[2]);
                        axg.fill_between(x, stan_g[2-i], stan_g[2+i], alpha=.3, color=fit_plot_colors[2]);
                        axdg.fill_between(x, stan_dg[2-i], stan_dg[2+i], alpha=.3, color=fit_plot_colors[3]);
                        
                    # Also plot Hill fit result for g
                    x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
                    x_fit = np.insert(x_fit, 0, 0)
                    hill_params = 10**row["sensor_params"][:4]
                    axg.plot(x_fit, hill_funct(x_fit, *hill_params), c='k', zorder=1000)
                else:
                    gp_color = fit_plot_colors[4]
                    slope_color = fit_plot_colors[5]
                    
                    tet_level_list = ['high'] if len(antibiotic_conc_list)==2 else ['low', 'high']
                    for lig in ligand_list:
                        stan_g = 10**row[f"GP_log_g_{lig}"]
                        stan_dg = row[f"GP_dlog_g_{lig}"]
                        
                        df = plot_df
                        df = df[(df.ligand==lig)|(df.ligand=='none')]
                        x = np.unique(df[lig])
                    
                        axg.plot(x, stan_g[2], color=gp_color)
                        axdg.plot(x, stan_dg[2], color=gp_color)
                        
                        for tet in tet_level_list:
                            
                            stan_f = row[f"GP_y_{lig}_{tet}_tet"]
                            axr.plot(x, stan_f[2], color=gp_color)
                            
                        axg.set_ylabel('GP Gene Epxression Estimate (MEF)', size=14)
                        axg.tick_params(labelsize=12);
                        if log_g_scale: axg.set_yscale("log")
                        
                        axdg.plot([x[0],x[-1]], [0,0], c='k');
                        axdg.set_ylabel('GP d(log(g))/d(log(x))', size=14)
                        axdg.tick_params(labelsize=12);
                        for i in range(1,3):
                            axg.fill_between(x, stan_g[2-i], stan_g[2+i], alpha=.3, color=gp_color);
                            axdg.fill_between(x, stan_dg[2-i], stan_dg[2+i], alpha=.3, color=slope_color);
                            for tet in tet_level_list:
                                axr.fill_between(x, stan_f[2-i], stan_f[2+i], alpha=.3, color=gp_color);
                        
            fig_axs_list.append((fig, axs_grid))
            
        if save_plots:
            pdf.savefig()
    
        if save_plots:
            pdf.close()
            
        return fig_axs_list
    
    def plot_count_ratios_vs_time(self, plot_range,
                                  with_tet=None,
                                  mark_samples=[],
                                  show_spike_ins=["b", "e"]):
        if with_tet is None:
            plot_tet = True
            plot_no_tet = True
        else:
            plot_tet = with_tet
            plot_no_tet = not with_tet
            
        barcode_frame = self.barcode_frame
        high_tet = self.high_tet
        
        if plot_range is None:
            plot_range = [0, max(barcode_frame.index)]
        
        plot_count_frame = barcode_frame.loc[plot_range[0]:plot_range[1]]
        plt.rcParams["figure.figsize"] = [10,6*(len(plot_count_frame))]
        fig, axs = plt.subplots(len(plot_count_frame), 1)
    
        inducer_conc_list = self.inducer_conc_list
            
        inducer = self.inducer
    
        inducer_conc_list_in_plate = np.asarray(np.split(np.asarray(inducer_conc_list),4)).transpose().flatten().tolist()*8
        inducer_conc_list_in_plate = np.asarray([(inducer_conc_list[j::4]*4)*2 for j in range(4)]*1).flatten()
        
        with_tet = []
        plate_list = []
        for r in fitness.rows():
            for c in fitness.columns():
                plate_list.append( int(2+(c-1)/3) )
                with_tet.append(r in fitness.rows()[1::2])
    
        sample_plate_map = pd.DataFrame({"well": fitness.wells()})
        sample_plate_map['with_tet'] = with_tet
        sample_plate_map[inducer] = inducer_conc_list_in_plate
        sample_plate_map['growth_plate'] = plate_list
        sample_plate_map.set_index('well', inplace=True, drop=False)
    
        wells_with_high_tet = []
        wells_with_zero_tet = []
    
        for i in range(2,6):
            df = sample_plate_map[(sample_plate_map["with_tet"]) & (sample_plate_map["growth_plate"]==i)]
            df = df.sort_values([inducer])
            wells_with_high_tet.append(df["well"].values)
            df = sample_plate_map[(sample_plate_map["with_tet"] != True) & (sample_plate_map["growth_plate"]==i)]
            df = df.sort_values([inducer])
            wells_with_zero_tet.append(df["well"].values)
    
        for i in range(2,6):
            counts_0 = []
            counts_tet = []
            for index, row in barcode_frame.iterrows():
                row_0 = row[wells_with_zero_tet[i-2]]
                counts_0.append(row_0.values)
                row_tet = row[wells_with_high_tet[i-2]]
                counts_tet.append(row_tet.values)
            barcode_frame[f"read_count_{0}_" + str(i)] = counts_0
            barcode_frame[f"read_count_{high_tet}_" + str(i)] = counts_tet

        spike_in_row_dict = {"AO-B": barcode_frame[barcode_frame["RS_name"]=="AO-B"],
                        "AO-E": barcode_frame[barcode_frame["RS_name"]=="AO-E"]}
        
        #Run for both AO-B and AO-E
        for spike_in, initial in zip(["AO-B", "AO-E"], ["b", "e"]):
            if initial in show_spike_ins:
                spike_in_reads_0 = [ spike_in_row_dict[spike_in][f'read_count_{0}_{plate_num}'].values[0] for plate_num in range(2,6) ]
                spike_in_reads_tet = [ spike_in_row_dict[spike_in][f'read_count_{high_tet}_{plate_num}'].values[0] for plate_num in range(2,6) ]
            
                x0 = [2, 3, 4, 5]
            
                for (index, row), ax in zip(plot_count_frame.iterrows(), axs): # iterate over barcodes
                    x_mark = []
                    y_mark = []
                    
                    if plot_no_tet:
                        n_reads = [ row[f'read_count_{0}_{plate_num}'] for plate_num in range(2,6) ]
                        for j in range(len(n_reads[0])): # iteration over IPTG concentrations 0-11
                            x = []
                            y = []
                            s = []
                            for i in range(len(n_reads)): # iteration over time points 0-3
                                if (n_reads[i][j]>0 and spike_in_reads_0[i][j]>0):
                                    x.append(x0[i])
                                    y.append(np.log10(n_reads[i][j]) - np.log10(spike_in_reads_0[i][j]))
                                    s.append( (np.sqrt(1/n_reads[i][j] + 1/spike_in_reads_0[i][j]))/np.log(10) )
                                    
                                    if ("no-tet", x0[i], inducer_conc_list[j]) in mark_samples:
                                        x_mark.append(x0[i])
                                        y_mark.append(np.log10(n_reads[i][j]) - np.log10(spike_in_reads_0[i][j]))
                                    
                            label = inducer_conc_list[j] if initial=="b" else None
                            fillstyle = "full" if initial=="b" else "none"
                            ax.errorbar(x, y, s, c=plot_colors()[j], marker='o', ms=8, fillstyle=fillstyle, label=label)
                
                    if plot_tet:
                        n_reads = [ row[f'read_count_{high_tet}_{plate_num}'] for plate_num in range(2,6) ]
                        for j in range(len(n_reads[0])): # iteration over IPTG concentrations 0-11
                            x = []
                            y = []
                            s = []
                            for i in range(len(n_reads)): # iteration over time points 0-3
                                if (n_reads[i][j]>0 and spike_in_reads_tet[i][j]>0):
                                    x.append(x0[i])
                                    y.append(np.log10(n_reads[i][j]) - np.log10(spike_in_reads_tet[i][j]))
                                    s.append( (np.sqrt(1/n_reads[i][j] + 1/spike_in_reads_tet[i][j]))/np.log(10) )
                                    
                                    if ("tet", x0[i], inducer_conc_list[j]) in mark_samples:
                                        x_mark.append(x0[i])
                                        y_mark.append(np.log10(n_reads[i][j]) - np.log10(spike_in_reads_tet[i][j]))
                            
                            if plot_no_tet:
                                label = None
                            else:
                                label = inducer_conc_list[j] if initial=="b" else None
                            fillstyle = "full" if initial=="b" else "none"
                            ax.errorbar(x, y, s, c=plot_colors()[j], marker='v', ms=8, fillstyle=fillstyle, label=label)
                        
                    barcode_str = str(index) + ', '
                    barcode_str += row['RS_name'] + ": "
                    barcode_str += row['forward_BC'] + ", "
                    barcode_str += row['reverse_BC']
                    ax.text(x=0.0, y=1.05, s=barcode_str, horizontalalignment='left', verticalalignment='top',
                            transform=ax.transAxes, fontsize=10, fontfamily="Courier New")
                    ax.set_xlabel('Plate Number', size=16)
                    ax.set_ylabel('Log10(count ÷ spike-in count)', size=16)
                    ax.set_xticks([2, 3, 4, 5])
                    leg = ax.legend(loc='lower left', bbox_to_anchor= (1.03, 0.07), ncol=3, borderaxespad=0, frameon=True, fontsize=10)
                    leg.get_frame().set_edgecolor('k');
                    
                    ax.plot(x_mark, y_mark, c='k', marker='o', ms=18, fillstyle="none", markeredgewidth=3, zorder=1000, linestyle="none")
        
    def plot_counts_vs_time(self, plot_range,
                                  with_tet=None,
                                  mark_samples=[]):

        if with_tet is None:
            plot_tet = True
            plot_no_tet = True
        else:
            plot_tet = with_tet
            plot_no_tet = not with_tet
            
        barcode_frame = self.barcode_frame
        high_tet = self.high_tet
        
        if plot_range is None:
            plot_range = [0, max(barcode_frame.index)]
        
        plot_count_frame = barcode_frame.loc[plot_range[0]:plot_range[1]].copy()
        plt.rcParams["figure.figsize"] = [10,6*(len(plot_count_frame))]
        fig, axs = plt.subplots(len(plot_count_frame), 1)
    
        inducer_conc_list = self.inducer_conc_list
            
        inducer = self.inducer
    
        inducer_conc_list_in_plate = np.asarray(np.split(np.asarray(inducer_conc_list),4)).transpose().flatten().tolist()*8
        inducer_conc_list_in_plate = np.asarray([(inducer_conc_list[j::4]*4)*2 for j in range(4)]*1).flatten()
        
        with_tet = []
        plate_list = []
        for r in fitness.rows():
            for c in fitness.columns():
                plate_list.append( int(2+(c-1)/3) )
                with_tet.append(r in fitness.rows()[1::2])
    
        sample_plate_map = pd.DataFrame({"well": fitness.wells()})
        sample_plate_map['with_tet'] = with_tet
        sample_plate_map[inducer] = inducer_conc_list_in_plate
        sample_plate_map['growth_plate'] = plate_list
        sample_plate_map.set_index('well', inplace=True, drop=False)
    
        wells_with_high_tet = []
        wells_with_zero_tet = []
    
        for i in range(2,6):
            df = sample_plate_map[(sample_plate_map["with_tet"]) & (sample_plate_map["growth_plate"]==i)]
            df = df.sort_values([inducer])
            wells_with_high_tet.append(df["well"].values)
            df = sample_plate_map[(sample_plate_map["with_tet"] != True) & (sample_plate_map["growth_plate"]==i)]
            df = df.sort_values([inducer])
            wells_with_zero_tet.append(df["well"].values)
    
        for i in range(2,6):
            counts_0 = []
            counts_tet = []
            for index, row in plot_count_frame.iterrows():
                row_0 = row[wells_with_zero_tet[i-2]]
                counts_0.append(row_0.values)
                row_tet = row[wells_with_high_tet[i-2]]
                counts_tet.append(row_tet.values)
            plot_count_frame[f"read_count_{0}_" + str(i)] = counts_0
            plot_count_frame[f"read_count_{high_tet}_" + str(i)] = counts_tet

        x0 = [2, 3, 4, 5]
    
        for (index, row), ax in zip(plot_count_frame.iterrows(), axs): # iterate over barcodes
            x_mark = []
            y_mark = []
            
            if plot_no_tet:
                n_reads = [ row[f'read_count_{0}_{plate_num}'] for plate_num in range(2,6) ]
                for j in range(len(n_reads[0])): # iteration over IPTG concentrations 0-11
                    x = []
                    y = []
                    s = []
                    for i in range(len(n_reads)): # iteration over time points 0-3
                        if n_reads[i][j]>0:
                            x.append(x0[i])
                            y.append(np.log10(n_reads[i][j]))
                            s.append( (np.sqrt(1/n_reads[i][j])/np.log(10)) )
                            
                            if ("no-tet", x0[i], inducer_conc_list[j]) in mark_samples:
                                x_mark.append(x0[i])
                                y_mark.append(np.log10(n_reads[i][j]))
                            
                    label = inducer_conc_list[j]
                    fillstyle = "full"
                    ax.errorbar(x, y, s, c=plot_colors()[j], marker='o', ms=8, fillstyle=fillstyle, label=label)
        
            if plot_tet:
                n_reads = [ row[f'read_count_{high_tet}_{plate_num}'] for plate_num in range(2,6) ]
                for j in range(len(n_reads[0])): # iteration over IPTG concentrations 0-11
                    x = []
                    y = []
                    s = []
                    for i in range(len(n_reads)): # iteration over time points 0-3
                        if n_reads[i][j]>0:
                            x.append(x0[i])
                            y.append(np.log10(n_reads[i][j]))
                            s.append( (np.sqrt(1/n_reads[i][j])/np.log(10)) )
                            
                            if ("tet", x0[i], inducer_conc_list[j]) in mark_samples:
                                x_mark.append(x0[i])
                                y_mark.append(np.log10(n_reads[i][j]))
                    
                    if plot_no_tet:
                        label = None
                    else:
                        label = inducer_conc_list[j]
                    fillstyle = "full"
                    ax.errorbar(x, y, s, c=plot_colors()[j], marker='v', ms=8, fillstyle=fillstyle, label=label)
                
            barcode_str = str(index) + ', '
            barcode_str += row['RS_name'] + ": "
            barcode_str += row['forward_BC'] + ", "
            barcode_str += row['reverse_BC']
            ax.text(x=0.0, y=1.05, s=barcode_str, horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, fontsize=10, fontfamily="Courier New")
            ax.set_xlabel('Plate Number', size=16)
            ax.set_ylabel('Log10(count)', size=16)
            ax.set_xticks([2, 3, 4, 5])
            leg = ax.legend(loc='lower left', bbox_to_anchor= (1.03, 0.07), ncol=3, borderaxespad=0, frameon=True, fontsize=10)
            leg.get_frame().set_edgecolor('k');
            
            ax.plot(x_mark, y_mark, c='k', marker='o', ms=18, fillstyle="none", markeredgewidth=3, zorder=1000, linestyle="none")
        
         
    
    def plot_chimera_plot(self,
                          save_plots=False,
                          chimera_cut_line=None,
                          plot_size=8):
            
        barcode_frame = self.barcode_frame[self.barcode_frame["possibleChimera"]]
        
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        os.chdir(self.data_directory)
        if save_plots:
            pdf_file = 'barcode fitness plots.pdf'
            pdf = PdfPages(pdf_file)
        
        plt.rcParams["figure.figsize"] = [plot_size, plot_size]
        fig, axs = plt.subplots(1, 1)
        axs.set_ylabel('Chimera Read Count per Sample', size=20)
        axs.set_xlabel('Geometric Mean of Parental Read Counts', size=20);
        axs.tick_params(labelsize=16);
    
        #axs.plot(np.sqrt(for_parent_count_list_96*rev_parent_count_list_96), chimera_count_list_96, 'o', ms=5,
        #        label="Individual Sample Counts");
        x = barcode_frame["parent_geo_mean"]/96
        y = barcode_frame["total_counts"]/96
        axs.plot(x, y, 'o', ms=7, label="Possible Chimeras, Total Counts ÷ 96");
        
        if "parent_geo_mean_p2" in barcode_frame.columns:
            x = barcode_frame["parent_geo_mean_p2"]/24
            y = barcode_frame["total_counts_plate_2"]/24
            axs.plot(x, y, 'o', ms=5, label="Total from Time Point 1 ÷ 24");
            leg = axs.legend(loc='upper left', bbox_to_anchor= (0.03, 0.97), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
            leg.get_frame().set_edgecolor('k');
        
        if "isChimera" in barcode_frame.columns:
            plot_frame = barcode_frame[barcode_frame["isChimera"]]
            x = plot_frame["parent_geo_mean"]/96
            y = plot_frame["total_counts"]/96
            axs.plot(x, y, 'o', ms=5, label="Actual Chimeras, Total Counts ÷ 96");
            leg = axs.legend(loc='upper left', bbox_to_anchor= (0.03, 0.97), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
            leg.get_frame().set_edgecolor('k');
        
        axs.set_xscale("log");
        axs.set_yscale("log");
        ylim = axs.get_ylim()
        xlim = axs.get_xlim()
        
        if chimera_cut_line is not None: 
            x_line = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]))
            axs.plot(x_line, chimera_cut_line(x_line), color='k');
            axs.set_ylim(ylim);
        
        if save_plots:
            pdf.savefig()
    
        if save_plots:
            pdf.close()
            
        return fig, axs
            
    # Method for plotting sub-frame on background of full library distribution
    def plot_hill_params(self, input_frames, in_labels=None, in_colors=None, in_alpha=0.7,
                         error_bars=True, log_high_err_cutoff=0.71, legend=True,
                         everything_color=None, box_size=8):
        
        if in_labels is None:
            in_labels = [""] * len (input_frames)
        
        if in_colors is None:
            in_colors = [fitness.gray_out("indigo")] * len (input_frames)
            
        if everything_color is None:
            everything_color = fitness.gray_out("xkcd:tea green", s_factor=0.7, v_factor=0.8)
            
        plt.rcParams["figure.figsize"] = [2*box_size, 2*box_size]
        fig, axs_grid = plt.subplots(2, 2)
        axs = axs_grid.flatten()
    
        y_label_list = ["G0", "Ginf", "Ginf/G0", "n"]
        if 'log_g0' in self.barcode_frame.columns.values:
            param_names = ["log_g0", "log_ginf", "log_ginf_g0_ratio", "log_n"]
            x_param = f'log_ec50'
            x_err_label = f'log_ec50 error'
        else:
            param_names = ["log_low_level", "log_high_level", "log_high_low_ratio", "log_n"]
            x_param = f'log_ic50'
            x_err_label = f'log_ic50 error'
    
        x_label = f'EC50'
    
        # This part plots the input input_frames
        for input_frame, c, lab in zip(input_frames, in_colors, in_labels):
            for ax, name in zip(axs, param_names):
                y_err_label = f'{name} error'
        
                params_x = input_frame[x_param]
                params_y = input_frame[name]
                err_x = input_frame[x_err_label]
                err_y = input_frame[y_err_label]
                
                if error_bars:
                    yerr = err_y
                    xerr = err_x
                    xerr = log_plot_errorbars(params_x, xerr)
                    yerr = log_plot_errorbars(params_y, yerr)
                    #xerr = np.array([xerr]).transpose()
                    #yerr = np.array([yerr]).transpose()
                
                    ax.errorbar(10**params_x, 10**params_y, yerr=yerr, xerr=xerr, fmt="o", ms=4, color=c,
                                label=lab, alpha=in_alpha);
                else:
                    ax.plot(10**params_x, 10**params_y, "o", ms=4, color=c,
                            label=lab, alpha=in_alpha);
    
        # This part plots all the rest
        plot_frame = self.barcode_frame[3:]
        plot_frame = plot_frame[plot_frame["total_counts"]>3000]
        if 'log_g0' in self.barcode_frame.columns.values:
            plot_frame = plot_frame[plot_frame["log_ginf error"]<log_high_err_cutoff]
        else:
            plot_frame = plot_frame[plot_frame["log_high_level error"]<log_high_err_cutoff]
        for ax, name, y_label in zip(axs, param_names, y_label_list):
            
            params_x = 10**plot_frame[x_param]
            params_y = 10**plot_frame[name]
            
            ax.set_xscale("log");
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot(params_x, params_y, "o", ms=3, color=everything_color, zorder=0, alpha=0.3, label="everything");
            #ax.set_xlim(xlim);
            #ax.set_ylim(ylim);
            ax.set_xlabel(x_label, size=20)
            ax.set_ylabel(y_label, size=20)
            ax.tick_params(labelsize=16);
            
            if y_label!="n":
                ax.set_yscale("log")
        
        if legend:
            leg = axs[0].legend(loc='lower center', bbox_to_anchor= (1.07, 1.02), ncol=6, borderaxespad=0, frameon=True, fontsize=12)
            leg.get_frame().set_edgecolor('k');
        y_max = axs[3].get_ylim()[1]
        #axs[0].set_ylim(-500, 3000);
        #axs[1].set_ylim(0.5, 2.75);
        ylim2 = axs[0].get_ylim();
        ylim3 = axs[1].get_ylim();
        axs[0].set_ylim(min(ylim2[0], ylim3[0]), max(ylim2[1], ylim3[1]));
        axs[1].set_ylim(min(ylim2[0], ylim3[0]), max(ylim2[1], ylim3[1]));
        #for ax in axs:
        #    ax.set_xlim(6,1000);
        
        return axs
    
    def save_as_pickle(self, notebook_dir=None, experiment=None, pickle_file=None):
        if notebook_dir is None:
            notebook_dir = self.notebook_dir
        if experiment is None:
            experiment = self.experiment
        if pickle_file is None:
            pickle_file = experiment + '_BarSeqFitnessFrame.pkl'
            
        os.chdir(notebook_dir)
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)
        print(f"BarSeqFitnessFrame saved as: {pickle_file}")
        
    def cleaned_frame(self, count_threshold=3000, log_high_error_cutoff=0.7, num_good_hill_points=12, exclude_mut_regions=None):
        frame = self.barcode_frame
        frame = frame[frame["total_counts"]>count_threshold]
        if 'log_ginf' in frame.columns.values:
            frame = frame[frame["log_ginf error"]<log_high_error_cutoff]
        else:
            frame = frame[frame["log_high_level error"]<log_high_error_cutoff]
        frame = frame[frame["good_hill_fit_points"]>=num_good_hill_points]
        
        if exclude_mut_regions is None:
            exclude_mut_regions = ["KAN", "Ori", "tetA", "YFP", "insulator"]
            
        for reg in exclude_mut_regions:
            frame = frame[frame["pacbio_" + reg + "_mutations"]<=0]
    
        return frame

def plot_colors():
    return sns.hls_palette(12, l=.4, s=.8)
    
def plot_colors96():
    p_c12 = [ ]
    for c in plot_colors():
        for i in range(8):
            p_c12.append(c)
    return p_c12
    
def plot_colors48():
    p_c12 = [ ]
    for c in plot_colors():
        for i in range(4):
            p_c12.append(c)
    return p_c12

def log_plot_errorbars(log_mu, log_sig):
    mu = np.array(10**log_mu)
    mu_low = np.array(10**(log_mu - log_sig))
    mu_high = np.array(10**(log_mu + log_sig))
    sig_low = mu - mu_low
    sig_high = mu_high - mu
    return np.array([sig_low, sig_high])
                
def hill_funct(x, low, high, mid, n):
    return low + (high-low)*( x**n )/( mid**n + x**n )

# Hill function of Hill function to describe fitness_difference(gene_expression([inducer]))
def double_hill_funct(x, g0, ginf, ec50, nx, f_min, f_max, g_50, ng):
    # g0, ginf, ec50, and nx are characteristics of individual sensor variants
        # g0 is the gene epxression level at zero inducer
        # ginf is the gene expresion level at full induction
        # ec50 is the inducer concentration of 1/2 max gene expression
        # nx is the exponent that describes the steepness of the sensor response curve
    # f_min, f_max, g_50, and ng are characteristics of the selection system
    # they are estimated from the fits above
        # f_min is the minimum fitness level, at zero gene expression
        # f_max is the maximum fitness level, at infinite gene expression (= 0)
        # g_50 is the gene expression of 1/2 max fitness
        # ng is the exponent that describes the steepness of the fitness vs. gene expression curve
    return hill_funct( hill_funct(x, g0, ginf, ec50, nx), f_min, f_max, g_50, ng )
    

def init_stan_fit_single_ligand(stan_data, fit_fitness_difference_params):
    x_data = stan_data['x']
    y_data = stan_data['y']
    log_g0 = log_level(np.mean(y_data[:2]))
    log_ginf = log_level(np.mean(y_data[-2:]))
    
    min_ic = np.log10(min([i for i in x_data if i>0]))
    max_ic = np.log10(max(x_data))
    log_ec50 = np.random.uniform(min_ic, max_ic)
    
    n = np.random.uniform(1.3, 1.7)
    
    sig = np.random.uniform(1, 3)
    
    low_fitness = fit_fitness_difference_params[0]
    mid_g = fit_fitness_difference_params[1]
    fitness_n = fit_fitness_difference_params[2]
    
    return dict(log_g0=log_g0, log_ginf=log_ginf, log_ec50=log_ec50,
                sensor_n=n, sigma=sig, low_fitness=low_fitness, mid_g=mid_g, fitness_n=fitness_n)
                
def init_stan_fit_two_lig_two_tet(stan_data, fit_fitness_difference_params):
    

    min_ic = np.log10(min(stan_data['x_1']))
    max_ic = np.log10(max(stan_data['x_1']))
    log_ec50_1 = np.random.uniform(min_ic, max_ic)
    log_ec50_2 = np.random.uniform(min_ic, max_ic)
    
    n_1 = np.random.uniform(1.3, 1.7)
    n_2 = np.random.uniform(1.3, 1.7)
    
    sig = np.random.uniform(1, 3)
    
    # Indices for x_y_s_list[ligand][tet][x,y,s][n]
    return dict(log_g0=log_level(stan_data['y_0_low_tet']), 
                log_ginf_1=log_level(np.mean(stan_data['y_1_high_tet'][-2:])), 
                log_ginf_2=log_level(np.mean(stan_data['y_2_high_tet'][-2:])), 
                log_ec50_1=log_ec50_1, 
                log_ec50_2=log_ec50_2, 
                sensor_n_1=n_1, 
                sensor_n_2=n_2, 
                sigma=sig, 
                low_fitness_low_tet=fit_fitness_difference_params[0][0],
                mid_g_low_tet=fit_fitness_difference_params[0][1],
                fitness_n_low_tet=fit_fitness_difference_params[0][2],
                low_fitness_high_tet=fit_fitness_difference_params[1][0],
                mid_g_high_tet=fit_fitness_difference_params[1][1],
                fitness_n_high_tet=fit_fitness_difference_params[1][2],
                )
    
def init_stan_GP_fit(fit_fitness_difference_params, single_tet, single_ligand):
    sig = np.random.uniform(1, 3)
    rho = np.random.uniform(0.9, 1.1)
    alpha = np.random.uniform(0.009, 0.011)
    
    if single_tet:
        low_fitness = fit_fitness_difference_params[0]
        mid_g = fit_fitness_difference_params[1]
        fitness_n = fit_fitness_difference_params[2]
        
        return dict(sigma=sig, low_fitness=low_fitness, mid_g=mid_g, fitness_n=fitness_n, rho=rho, alpha=alpha)
    else:
        return dict(sigma=sig, rho=rho, alpha=alpha,
                    low_fitness_low_tet=fit_fitness_difference_params[0][0],
                    mid_g_low_tet=fit_fitness_difference_params[0][1],
                    fitness_n_low_tet=fit_fitness_difference_params[0][2],
                    low_fitness_high_tet=fit_fitness_difference_params[1][0],
                    mid_g_high_tet=fit_fitness_difference_params[1][1],
                    fitness_n_high_tet=fit_fitness_difference_params[1][2],
                    )
    
    
def log_level(fitness_difference):
    log_g = 1.439*fitness_difference + 3.32
    log_g = log_g*np.random.uniform(0.9,1.1)
    if log_g<1.5:
        log_g = 1.5
    if log_g>4:
        log_g = 4
    return log_g
        
def get_stan_data(st_row, plot_df, antibiotic_conc_list, 
                  lig_list, fit_fitness_difference_params, 
                  old_style_columns=False, initial="b", plasmid="pVER",
                  is_gp_model=False):
    
    log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf = fitness.log_g_limits(plasmid=plasmid)
    
    if old_style_columns:
        high_tet = antibiotic_conc_list[1]
        
        y_zero = st_row[f"fitness_{0}_estimate_{initial}"]
        s_zero = st_row[f"fitness_{0}_err_{initial}"]
        y_high = st_row[f"fitness_{high_tet}_estimate_{initial}"]
        s_high = st_row[f"fitness_{high_tet}_err_{initial}"]
        
        y = (y_high - y_zero)/y_zero
        s = np.sqrt( s_high**2 + s_zero**2 )/y_zero
        
        if include_lactose_zero:
            print(f"      including zero lactose data for {st_index}")
            y = np.insert(y, 0, st_row['lactose_fitness_diff'])
            s = np.insert(s, 0, st_row['lactose_fitness_err'])
            x_fit = np.insert(x, 0, 0)
        else:
            x_fit = x
        x_y_s_list = [[x_fit, y, s]]
    else:
        df = plot_df
        df = df[df.antibiotic_conc==0] # use all data at zero tet for reference fitness
        y = [st_row[f"fitness_S{i}_{initial}"] for i in df.sample_id]
        s = [st_row[f"fitness_S{i}_err_{initial}"] for i in df.sample_id]
        y_ref_list = np.array(y)
        s_ref_list = np.array(s)
        
        w = 1/s_ref_list**2
        y_ref = np.average(y_ref_list, weights=w)
        s_ref = np.average((y_ref_list-y_ref)**2, weights=w)
        v_1 = np.sum(w)
        v_2 = np.sum(w**2)
        s_ref = np.sqrt(s_ref/(1 - (v_2/v_1**2)))
            
        tet_list = antibiotic_conc_list[antibiotic_conc_list>0]
        x_y_s_list = []
        for lig in lig_list:
            sub_list = []
            for tet in tet_list:
                df = plot_df
                df = df[(df.ligand==lig)|(df.ligand=='none')]
                df = df[df.antibiotic_conc==tet]
                x = np.array(df[lig])
                y = np.array([st_row[f"fitness_S{i}_{initial}"] for i in df.sample_id])
                y = (y - y_ref)/y_ref
                s = np.array([st_row[f"fitness_S{i}_err_{initial}"] for i in df.sample_id])
                s = np.sqrt(s**2 + s_ref**2)/y_ref
                
                sub_list.append([x, y, s])
            x_y_s_list.append(sub_list)
            
        if len(lig_list) == 1:
            # Case for single tet concentration
    
            low_fitness = fit_fitness_difference_params[0]
            mid_g = fit_fitness_difference_params[1]
            fitness_n = fit_fitness_difference_params[2]
            
            x_fit = x_y_s_list[0][0][0]
            y = x_y_s_list[0][0][1]
            s = x_y_s_list[0][0][2]
            
            if is_gp_model:
                # For GP model, can't have missing data. So, if either y or s is nan, replace with values that won't affect GP model results (i.e. s=100)
                invalid = (np.isnan(y) | np.isnan(s))
                y[invalid] = low_fitness/2
                s[invalid] = 100
                x = x_fit
                y_err = s
            else:
                valid = ~(np.isnan(y) | np.isnan(s))
                x = x_fit[valid]
                y = y[valid]
                y_err = s[valid]
            
            stan_data = dict(x=x, y=y, N=len(y), y_err=y_err,
                             low_fitness_mu=low_fitness, mid_g_mu=mid_g, fitness_n_mu=fitness_n,
                             log_g_min=log_g_min, log_g_max=log_g_max, log_g_prior_scale=log_g_prior_scale)
        
        else:
            # Case for two-tet, two-ligand
            y_0_med = x_y_s_list[0][0][1][0]
            s_0_med = x_y_s_list[0][0][2][0]
            
            x_1 = x_y_s_list[0][0][0]
            y_1_med = x_y_s_list[0][0][1]
            s_1_med = x_y_s_list[0][0][2]
            y_1_med = y_1_med[x_1>0]
            s_1_med = s_1_med[x_1>0]
            x_1 = x_1[x_1>0]
            
            x_1_high = x_y_s_list[0][1][0]
            y_1_high = x_y_s_list[0][1][1]
            s_1_high = x_y_s_list[0][1][2]
            y_1_high = y_1_high[x_1_high>0]
            s_1_high = s_1_high[x_1_high>0]
            x_1_high = x_1_high[x_1_high>0]
            
            x_2 = x_y_s_list[1][0][0]
            y_2_med = x_y_s_list[1][0][1]
            s_2_med = x_y_s_list[1][0][2]
            y_2_med = y_2_med[x_2>0]
            s_2_med = s_2_med[x_2>0]
            x_2 = x_2[x_2>0]
            
            x_2_high = x_y_s_list[1][1][0]
            y_2_high = x_y_s_list[1][1][1]
            s_2_high = x_y_s_list[1][1][2]
            y_2_high = y_2_high[x_2_high>0]
            s_2_high = s_2_high[x_2_high>0]
            x_2_high = x_2_high[x_2_high>0]
            
            stan_data = dict(N_lig=len(x_1), x_1=x_1, x_2=x_2, 
                             y_0_low_tet=y_0_med, y_0_low_tet_err=s_0_med,
                             y_1_low_tet=y_1_med, y_1_low_tet_err=s_1_med,
                             y_2_low_tet=y_2_med, y_2_low_tet_err=s_2_med,
                             y_1_high_tet=y_1_high, y_1_high_tet_err=s_1_high,
                             y_2_high_tet=y_2_high, y_2_high_tet_err=s_2_high,
                             log_g_min=log_g_min, log_g_max=log_g_max, log_g_prior_scale=log_g_prior_scale,
                             low_fitness_mu_low_tet=fit_fitness_difference_params[0][0],
                             mid_g_mu_low_tet=fit_fitness_difference_params[0][1],
                             fitness_n_mu_low_tet=fit_fitness_difference_params[0][2],
                             low_fitness_std_low_tet=fit_fitness_difference_params[0][3],
                             mid_g_std_low_tet=fit_fitness_difference_params[0][4],
                             fitness_n_std_low_tet=fit_fitness_difference_params[0][5],
                             low_fitness_mu_high_tet=fit_fitness_difference_params[1][0],
                             mid_g_mu_high_tet=fit_fitness_difference_params[1][1],
                             fitness_n_mu_high_tet=fit_fitness_difference_params[1][2],
                             low_fitness_std_high_tet=fit_fitness_difference_params[1][3],
                             mid_g_std_high_tet=fit_fitness_difference_params[1][4],
                             fitness_n_std_high_tet=fit_fitness_difference_params[1][5],
                             )
    
    return stan_data


    
