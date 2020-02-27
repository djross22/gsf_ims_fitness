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
        
    def __init__(self, notebook_dir, experiment=None, barcode_file=None, low_tet=0, high_tet=20, inducer_conc_list=None, inducer="IPTG"):
        
        self.notebook_dir = notebook_dir
        
        self.low_tet = low_tet
        self.high_tet = high_tet
        
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
        
        self.inducer = inducer
        
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
                            inducer_conc_list=None,
                            inducer=None,
                            auto_save=True,
                            ignore_samples=[],
                            refit_index=None):
        
        barcode_frame = self.barcode_frame
        low_tet = self.low_tet
        high_tet = self.high_tet
            
        #os.chdir(self.data_directory)
    
        if inducer_conc_list is None:
            inducer_conc_list = self.inducer_conc_list
            
        if inducer is None:
            inducer = self.inducer
        
        if refit_index is None:
            print(f"Fitting to log(barcode ratios) to find fitness for each barcode in {self.experiment}")
        
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
            wells_with_low_tet = []
        
            for i in range(2,6):
                df = sample_plate_map[(sample_plate_map["with_tet"]) & (sample_plate_map["growth_plate"]==i)]
                df = df.sort_values([inducer])
                wells_with_high_tet.append(df["well"].values)
                df = sample_plate_map[(sample_plate_map["with_tet"] != True) & (sample_plate_map["growth_plate"]==i)]
                df = df.sort_values([inducer])
                wells_with_low_tet.append(df["well"].values)
        
            for i in range(2,6):
                counts_0 = []
                counts_tet = []
                for index, row in barcode_frame.iterrows():
                    row_0 = row[wells_with_low_tet[i-2]]
                    counts_0.append(row_0.values)
                    row_tet = row[wells_with_high_tet[i-2]]
                    counts_tet.append(row_tet.values)
                barcode_frame[f"read_count_{low_tet}_" + str(i)] = counts_0
                barcode_frame[f"read_count_{high_tet}_" + str(i)] = counts_tet
        
        spike_in_fitness_0 = {"AO-B": np.array([0.9637]*12), "AO-E": np.array([0.9666]*12)}
        spike_in_fitness_tet = {"AO-B": np.array([0.8972]*12), "AO-E": np.array([0.8757]*12)}
    
        #ref_index_b = barcode_frame[barcode_frame["RS_name"]=="AO-B"].index[0]
        #ref_index_e = barcode_frame[barcode_frame["RS_name"]=="AO-E"].index[0]
    
        #spike_in_row = {"AO-B": barcode_frame[ref_index_b:ref_index_b+1], "AO-E": barcode_frame[ref_index_e:ref_index_e+1]}
        spike_in_row = {"AO-B": barcode_frame[barcode_frame["RS_name"]=="AO-B"],
                        "AO-E": barcode_frame[barcode_frame["RS_name"]=="AO-E"]}
        
        sp_b = spike_in_row["AO-B"][["RS_name", "read_count_0_2"]]
        sp_e = spike_in_row["AO-E"][["RS_name", "read_count_0_2"]]
        print(f"AO-B: {sp_b}")
        print(f"AO-E: {sp_e}")
        
        #Fit to barcode log(ratios) over time to get slopes = fitness
        #Run for both AO-B and AO-E
        for spike_in, initial in zip(["AO-B", "AO-E"], ["b", "e"]):
            f_tet_est_list = []
            f_0_est_list = []
            f_tet_err_list = []
            f_0_err_list = []
        
            spike_in_reads_0 = [ spike_in_row[spike_in][f'read_count_{low_tet}_{plate_num}'].values[0] for plate_num in range(2,6) ]
            spike_in_reads_tet = [ spike_in_row[spike_in][f'read_count_{high_tet}_{plate_num}'].values[0] for plate_num in range(2,6) ]
        
            x0 = [2, 3, 4, 5]
            
            if refit_index is None:
                fit_frame = barcode_frame
            else:
                fit_frame = barcode_frame.loc[refit_index:refit_index]
        
            for index, row in fit_frame.iterrows(): # iterate over barcodes
                slopes = []
                errors = []
                n_reads = [ row[f'read_count_{low_tet}_{plate_num}'] for plate_num in range(2,6) ]
            
                for j in range(len(n_reads[0])): # iteration over IPTG concentrations 0-11
                    x = []
                    y = []
                    s = []
                    for i in range(len(n_reads)): # iteration over time points 0-3
                        if (n_reads[i][j]>0 and spike_in_reads_0[i][j]>0):
                            x.append(x0[i])
                            y.append(np.log(n_reads[i][j]) - np.log(spike_in_reads_0[i][j]))
                            sigma = np.sqrt(1/n_reads[i][j] + 1/spike_in_reads_0[i][j])
                            if ("no-tet", x0[i], inducer_conc_list[j]) in ignore_samples:
                                sigma = np.inf
                            s.append(sigma)
                                
                    if len(x)>1:
                        popt, pcov = curve_fit(fitness.line_funct, x, y, sigma=s, absolute_sigma=True)
                        slopes.append(popt[0])
                        errors.append(np.sqrt(pcov[0,0]))
                    else:
                        slopes.append(np.nan)
                        errors.append(np.nan)
                
                    if j==0:
                        if len(x)>1:
                            slope_0 = popt[0]
                        else:
                            slope_0 = 0
                
                slopes = np.asarray(slopes)
                errors = np.asarray(errors)
                f_0_est = spike_in_fitness_0[spike_in] + slopes/np.log(10)
                f_0_err = errors/np.log(10)
            
                slopes = []
                errors = []
                n_reads = [ row[f'read_count_{high_tet}_{plate_num}'] for plate_num in range(2,6) ]
            
                for j in range(len(n_reads[0])): # iteration over IPTG concentrations 0-11
                    x = []
                    y = []
                    s = []
                    for i in range(len(n_reads)): # iteration over time points 0-3
                        if (n_reads[i][j]>0 and spike_in_reads_tet[i][j]>0):
                            x.append(x0[i])
                            y.append(np.log(n_reads[i][j]) - np.log(spike_in_reads_tet[i][j]))
                            sigma = np.sqrt(1/n_reads[i][j] + 1/spike_in_reads_tet[i][j])
                            if ("tet", x0[i], inducer_conc_list[j]) in ignore_samples:
                                sigma = np.inf
                            s.append(sigma)
                            
                    if len(x)>1:
                        def fit_funct(xp, mp, bp): return fitness.bi_linear_funct(xp-2, mp, bp, slope_0, alpha=np.log(5))
                        #bounds = ([-np.log(10), -50], [slope_0, 50])
                        popt, pcov = curve_fit(fit_funct, x, y, sigma=s, absolute_sigma=True)#, bounds=bounds)
                        slopes.append(popt[0])
                        errors.append(np.sqrt(pcov[0,0]))
                    else:
                        slopes.append(np.nan)
                        errors.append(np.nan)
                
                slopes = np.asarray(slopes)
                errors = np.asarray(errors)
                f_tet_est = spike_in_fitness_tet["AO-B"] + slopes/np.log(10)
                f_tet_err = errors/np.log(10)
                
                f_tet_est_list.append(f_tet_est)
                f_0_est_list.append(f_0_est)
                f_tet_err_list.append(f_tet_err)
                f_0_err_list.append(f_0_err)
            
            if refit_index is None:
                barcode_frame[f'fitness_{low_tet}_estimate_{initial}'] = f_0_est_list
                barcode_frame[f'fitness_{low_tet}_err_{initial}'] = f_0_err_list
                barcode_frame[f'fitness_{high_tet}_estimate_{initial}'] = f_tet_est_list
                barcode_frame[f'fitness_{high_tet}_err_{initial}'] = f_tet_err_list
            else:
                fit_arr_1 = barcode_frame.loc[refit_index, f'fitness_{low_tet}_estimate_{initial}']
                fit_arr_1 *= 0
                fit_arr_1 += f_0_est_list[0]
                fit_arr_2 = barcode_frame.loc[refit_index, f'fitness_{low_tet}_err_{initial}']
                fit_arr_2 *= 0
                fit_arr_2 += f_0_err_list[0]
                fit_arr_3 = barcode_frame.loc[refit_index, f'fitness_{high_tet}_estimate_{initial}']
                fit_arr_3 *= 0
                fit_arr_3 += f_tet_est_list[0]
                fit_arr_4 = barcode_frame.loc[refit_index, f'fitness_{high_tet}_err_{initial}']
                fit_arr_4 *= 0
                fit_arr_4 += f_tet_err_list[0]
            

        self.barcode_frame = barcode_frame
        
        #os.chdir(self.notebook_dir)
        #pickle_file = self.experiment + '_inducer_conc_list.pkl'
        #with open(pickle_file, 'wb') as f:
        #    pickle.dump(inducer_conc_list, f)
            
        if auto_save:
            self.save_as_pickle()
        
            
    def fit_fitness_difference_curves(self,
                                      includeChimeras=False,
                                      fit_fitness_difference_funct=None,
                                      fit_fitness_difference_params=None,
                                      auto_save=True):
            
        print(f"Fitting to fitness curves to find sensor parameters for {self.experiment}")
        
        if fit_fitness_difference_params is None:
            fit_fitness_difference_params = np.array([-7.41526290e-01,  7.75447318e+02,  2.78019804e+00])
        
        self.fit_fitness_difference_params = fit_fitness_difference_params
            
        def default_funct(x, g_min, g_max, x_50, nx):
            return double_hill_funct(x, g_min, g_max, x_50, nx, fit_fitness_difference_params[0], 0, fit_fitness_difference_params[1], fit_fitness_difference_params[2])
        
        if fit_fitness_difference_funct is None:
            if self.fit_fitness_difference_funct is None:
                fit_fitness_difference_funct = default_funct
            else:
                fit_fitness_difference_funct = self.fit_fitness_difference_funct
        
        barcode_frame = self.barcode_frame
        low_tet = self.low_tet
        high_tet = self.high_tet
        
        if "sensor_params" not in barcode_frame.columns:
            barcode_frame["sensor_params"] = [ np.full((4), np.nan) for i in range(len(barcode_frame))]
        
        if "sensor_params_err" not in barcode_frame.columns:
            barcode_frame["sensor_params_cov"] = [ np.full((4, 4), np.nan) for i in range(len(barcode_frame))]
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        inducer_conc_list = self.inducer_conc_list
        
        x = np.array(inducer_conc_list)
        x_min = min([i for i in inducer_conc_list if i>0])
        
        popt_list = []
        pcov_list = []
        residuals_list = []
        
        for (index, row) in barcode_frame.iterrows(): # iterate over barcodes
            initial = "b"
            y_low = row[f"fitness_{low_tet}_estimate_{initial}"]
            s_low = row[f"fitness_{low_tet}_err_{initial}"]
            y_high = row[f"fitness_{high_tet}_estimate_{initial}"]
            s_high = row[f"fitness_{high_tet}_err_{initial}"]
            
            y = (y_high - y_low)/y_low
            s = np.sqrt( s_high**2 + s_low**2 )/y_low
            
            valid = ~(np.isnan(y) | np.isnan(s))
            
            p0 = [100, 1500, 200, 1.5]
            bounds = [[0.1, 0.1, x_min/4, 0.1], [5000, 5000, 4*max(x), 5]]
            try:
                popt, pcov = curve_fit(fit_fitness_difference_funct, x[valid], y[valid], sigma=s[valid], p0=p0, maxfev=len(x)*10000, bounds=bounds)
                resid = np.sqrt(np.sum((y[valid] - fit_fitness_difference_funct(x[valid], *popt))**2)/len(x[valid]))
            except (RuntimeError, ValueError) as err:
                popt = np.full((4), np.nan)
                pcov = np.full((4, 4), np.nan)
                resid = np.nan
                print(f"Error fitting curve for index {index}: {err}")
            
            popt_list.append(popt)
            pcov_list.append(pcov)
            residuals_list.append(resid)
                
        barcode_frame["sensor_params"] = popt_list
        barcode_frame["sensor_params_cov"] = pcov_list
        barcode_frame["sensor_rms_residuals"] = residuals_list
        
        self.barcode_frame = barcode_frame
        
        if auto_save:
            self.save_as_pickle()
        
            
    def stan_fitness_difference_curves(self,
                                      includeChimeras=False,
                                      stan_fitness_difference_model='Double Hill equation fit.stan',
                                      fit_fitness_difference_params=None,
                                      control=dict(adapt_delta=0.9),
                                      iterations=1000,
                                      chains=4,
                                      auto_save=True,
                                      refit_index=None,
                                      plasmid="pVER"):
            
        print(f"Using Stan to fit to fitness curves to find sensor parameters for {self.experiment}")
        print(f"  Using fitness parameters for {plasmid}")
        #os.chdir(self.notebook_dir)
        stan_model = stan_utility.compile_model(stan_fitness_difference_model)
        
        if fit_fitness_difference_params is None:
            if plasmid == "pVER":
                fit_fitness_difference_params = np.array([-0.72246,  13328,  3.2374])
            else:
                fit_fitness_difference_params = np.array([-7.41526290e-01,  7.75447318e+02,  2.78019804e+00])
        
        self.fit_fitness_difference_params = fit_fitness_difference_params
        
        barcode_frame = self.barcode_frame
        low_tet = self.low_tet
        high_tet = self.high_tet
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        inducer_conc_list = self.inducer_conc_list
        
        x = np.array(inducer_conc_list)
        x_min = min([i for i in inducer_conc_list if i>0])
        
        low_fitness = fit_fitness_difference_params[0]
        mid_g = fit_fitness_difference_params[1]
        fitness_n = fit_fitness_difference_params[2]
        
        params_list = ['log_low_level', 'log_high_level', 'log_IC_50', 'log_sensor_n', 'log_high_low_ratio',
                       'low_fitness', 'mid_g', 'fitness_n']
        log_high_low_ind = params_list.index('log_high_low_ratio')
        params_dim = len(params_list)
        
        quantile_params_list = params_list[:-len(fit_fitness_difference_params)]
        quantile_list = [0.05, 0.25, 0.5, 0.75, 0.95]
        quantile_dim = len(quantile_list)
        
        if "sensor_params" not in barcode_frame.columns:
            barcode_frame["sensor_params"] = [ np.full((params_dim), np.nan) for i in range(len(barcode_frame))]
        
        if "sensor_params_cov" not in barcode_frame.columns:
            barcode_frame["sensor_params_cov"] = [ np.full((params_dim, params_dim), np.nan) for i in range(len(barcode_frame))]
        
        

        def stan_fit_row(st_row, st_index):
            print()
            print(f"fitting row index: {st_index}")
            initial = "b"
            y_low = st_row[f"fitness_{low_tet}_estimate_{initial}"]
            s_low = st_row[f"fitness_{low_tet}_err_{initial}"]
            y_high = st_row[f"fitness_{high_tet}_estimate_{initial}"]
            s_high = st_row[f"fitness_{high_tet}_err_{initial}"]
            
            y = (y_high - y_low)/y_low
            s = np.sqrt( s_high**2 + s_low**2 )/y_low
            
            valid = ~(np.isnan(y) | np.isnan(s))
            
            if plasmid == "pVER":
                log_g_min = 1.2
                log_g_max = 5.4
                log_g_prior_scale = 0.15
            else:
                log_g_min = 1
                log_g_max = 4.5
                log_g_prior_scale = 0.3
            stan_data = dict(x=x[valid], y=y[valid], N=len(y[valid]), y_err=s[valid],
                             low_fitness_mu=low_fitness, mid_g_mu=mid_g, fitness_n_mu=fitness_n,
                             log_g_min=log_g_min, log_g_max=log_g_max, log_g_prior_scale=log_g_prior_scale)
        
            try:
                stan_init = [ init_stan_fit(x[valid], y[valid], fit_fitness_difference_params) for i in range(4) ]
                
                stan_fit = stan_model.sampling(data=stan_data, iter=iterations, init=stan_init, chains=chains, control=control)
                stan_samples = stan_fit.extract(permuted=False, pars=params_list)
        
                stan_samples_arr = np.array([stan_samples[key].flatten() for key in params_list ])
                stan_popt = np.array([np.median(s) for s in stan_samples_arr ])
                stan_pcov = np.cov(stan_samples_arr, rowvar=True)
                stan_resid = np.median(stan_fit["rms_resid"])
                stan_samples_out = np.array([stan_samples[key][::71,:].flatten() for key in params_list ])
                stan_quantiles = np.array([np.quantile(stan_samples[key], quantile_list) for key in quantile_params_list ])
                high_low_samples = stan_samples_arr[log_high_low_ind]
                hill_invert_prob = len(high_low_samples[high_low_samples<0])/len(high_low_samples)
            except:
                stan_popt = np.full((params_dim), np.nan)
                stan_pcov = np.full((params_dim, params_dim), np.nan)
                stan_resid = np.nan
                stan_samples_out = np.full((params_dim, 32), np.nan)
                stan_quantiles = np.full((len(quantile_params_list), quantile_dim), np.nan)
                print(f"Error during Stan fitting for index {st_index}:", sys.exc_info()[0])
                hill_invert_prob = np.nan
                
            return (stan_popt, stan_pcov, stan_resid, stan_samples_out, stan_quantiles, hill_invert_prob)
        
        if refit_index is None:
            fit_list = [ stan_fit_row(row, index) for (index, row) in barcode_frame.iterrows() ]
            
            popt_list = []
            pcov_list = []
            residuals_list = []
            samples_out_list = []
            quantiles_list = []
            invert_prob_list = []
            
            for item in fit_list: # iterate over barcodes
                stan_popt, stan_pcov, stan_resid, stan_samples_out, stan_quantiles, hill_invert_prob = item
                
                popt_list.append(stan_popt)
                pcov_list.append(stan_pcov)
                residuals_list.append(stan_resid)
                samples_out_list.append(stan_samples_out)
                quantiles_list.append(stan_quantiles)
                invert_prob_list.append(hill_invert_prob)
                    
            barcode_frame["sensor_params"] = popt_list
            barcode_frame["sensor_params_cov"] = pcov_list
            barcode_frame["sensor_rms_residuals"] = residuals_list
            barcode_frame["sensor_stan_samples"] = samples_out_list
            barcode_frame["sensor_params_quantiles"] = quantiles_list
            barcode_frame["hill_invert_prob"] = invert_prob_list
        else:
            row_to_fit = barcode_frame.loc[refit_index]
            stan_popt, stan_pcov, stan_resid, stan_samples_out, stan_quantiles, hill_invert_prob = stan_fit_row(row_to_fit, refit_index)
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
                       plasmid="pVER"):
            
        print(f"Using Stan to fit to fitness curves with GP model for {self.experiment}")
        print(f"  Using fitness parameters for {plasmid}")
        
        stan_model = stan_utility.compile_model(stan_GP_model)
        
        if fit_fitness_difference_params is None:
            if plasmid == "pVER":
                fit_fitness_difference_params = np.array([-0.72246,  13328,  3.2374])
            else:
                fit_fitness_difference_params = np.array([-7.41526290e-01,  7.75447318e+02,  2.78019804e+00])
        
        self.fit_fitness_difference_params = fit_fitness_difference_params
        
        barcode_frame = self.barcode_frame
        low_tet = self.low_tet
        high_tet = self.high_tet
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        inducer_conc_list = self.inducer_conc_list
        
        x = np.array(inducer_conc_list)
        x_min = min([i for i in inducer_conc_list if i>0])
        
        low_fitness = fit_fitness_difference_params[0]
        mid_g = fit_fitness_difference_params[1]
        fitness_n = fit_fitness_difference_params[2]
        
        params_list = ['low_fitness', 'mid_g', 'fitness_n', 'log_rho', 'log_alpha', 'log_sigma']
        params_dim = len(params_list)
        
        quantile_list = [0.05, 0.25, 0.5, 0.75, 0.95]
        quantile_dim = len(quantile_list)
        
        if "sensor_GP_params" not in barcode_frame.columns:
            barcode_frame["sensor_GP_params"] = [ np.full((params_dim), np.nan) for i in range(len(barcode_frame))]
        
        if "sensor_GP_cov" not in barcode_frame.columns:
            barcode_frame["sensor_GP_cov"] = [ np.full((params_dim, params_dim), np.nan) for i in range(len(barcode_frame))]
        
        if plasmid == "pVER":
            log_g_min = 1.2
            log_g_max = 5.4
        else:
            log_g_min = 1
            log_g_max = 4.5

        def stan_fit_row(st_row, st_index):
            print()
            print(f"fitting row index: {st_index}")
            initial = "b"
            y_low = st_row[f"fitness_{low_tet}_estimate_{initial}"]
            s_low = st_row[f"fitness_{low_tet}_err_{initial}"]
            y_high = st_row[f"fitness_{high_tet}_estimate_{initial}"]
            s_high = st_row[f"fitness_{high_tet}_err_{initial}"]
            
            y = (y_high - y_low)/y_low
            s = np.sqrt( s_high**2 + s_low**2 )/y_low
            
            # if either y or s is nan, replace with values that won't affect GP model results (i.e. s=10)
            invalid = (np.isnan(y) | np.isnan(s))
            y[invalid] = low_fitness/2
            s[invalid] = 10
            
            stan_data = dict(x=x, y=y, N=len(y), y_err=s,
                             low_fitness_mu=low_fitness, mid_g_mu=mid_g, fitness_n_mu=fitness_n,
                             log_g_min=log_g_min, log_g_max=log_g_max)
        
            try:
                stan_init = [ init_stan_GP_fit(fit_fitness_difference_params) for i in range(chains) ]
                
                stan_fit = stan_model.sampling(data=stan_data, iter=iterations, init=stan_init, chains=chains, control=control)
                stan_samples = stan_fit.extract(permuted=True)
                
                g_arr = stan_samples['constr_log_g']
                dg_arr = stan_samples['dlog_g']
                f_arr = stan_samples['mean_y']
                params_arr = np.array([stan_samples[x] for x in params_list])
        
                stan_popt = np.array([np.median(s) for s in params_arr ])
                stan_pcov = np.cov(params_arr, rowvar=True)
                
                stan_g = np.array([ np.quantile(g_arr, q, axis=0) for q in quantile_list ])
                stan_dg = np.array([ np.quantile(dg_arr, q, axis=0) for q in quantile_list ])
                stan_f = np.array([ np.quantile(f_arr, q, axis=0) for q in quantile_list ])
                
                stan_g_var = np.var(g_arr, axis=0)
                stan_dg_var = np.var(dg_arr, axis=0)
                
                temp_arr = stan_fit.extract(permuted=False, pars=["constr_log_g"])["constr_log_g"][::71,:,:]
                stan_g_samples = np.array([ a.flatten() for a in temp_arr.transpose() ])
                temp_arr = stan_fit.extract(permuted=False, pars=["dlog_g"])["dlog_g"][::71,:,:]
                stan_dg_samples = np.array([ a.flatten() for a in temp_arr.transpose() ])
                
                stan_resid = np.median(stan_fit["rms_resid"])
            except:
                stan_popt = np.full((params_dim), np.nan)
                stan_pcov = np.full((params_dim, params_dim), np.nan)
                
                stan_g = np.full((quantile_dim, len(x)), np.nan)
                stan_dg = np.full((quantile_dim, len(x)), np.nan)
                stan_f = np.full((quantile_dim, len(x)), np.nan)
                
                stan_g_var = np.full((params_dim), np.nan)
                stan_dg_var = np.full((params_dim), np.nan)
                
                stan_g_samples = np.full((len(x), 32), np.nan)
                stan_dg_samples = np.full((len(x), 32), np.nan)
                
                stan_resid = np.nan
                print(f"Error during Stan fitting for index {st_index}:", sys.exc_info()[0])
                
            return (stan_popt, stan_pcov, stan_resid, stan_g, stan_dg, stan_f, stan_g_var, stan_dg_var, stan_g_samples, stan_dg_samples)
        
        if refit_index is None:
            fit_list = [ stan_fit_row(row, index) for (index, row) in barcode_frame.iterrows() ]
            
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
                    
            barcode_frame["sensor_GP_params"] = popt_list
            barcode_frame["sensor_GP_cov"] = pcov_list
            
            barcode_frame["sensor_GP_g_quantiles"] = stan_g_list
            barcode_frame["sensor_GP_Dg_quantiles"] = stan_dg_list
            barcode_frame["sensor_GP_Df_quantiles"] = stan_f_list
            
            barcode_frame["sensor_GP_residuals"] = residuals_list
            
            barcode_frame["sensor_GP_g_var"] = stan_g_var_list
            barcode_frame["sensor_GP_dg_var"] = stan_dg_var_list
            
            barcode_frame["sensor_GP_g_samples"] = stan_g_samples_list
            barcode_frame["sensor_GP_dg_samples"] = stan_dg_samples_list
        else:
            row_to_fit = barcode_frame.loc[refit_index]
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
        low_tet = self.low_tet
        high_tet = self.high_tet
        
        columns_to_sum = fitness.wells()
        columns_to_sum += [ 'total_counts', 'fraction_total', 'total_counts_plate_2', 'fraction_total_p2' ]
        columns_to_sum += [ f'fraction_{w}' for w in fitness.wells() ]
        columns_to_sum += [ f'read_count_{low_tet}_{plate_num}' for plate_num in range(2,6) ]
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
        
    def plot_read_counts(self, num_to_plot=None, save_plots=False, pdf_file=None):
        
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
    
        axs[0].scatter(index_list, r12, c=plot_colors12(), s=70);
        for i in range(13):
            axs[0].plot([i*8+0.5, i*8+0.5],[min(BC_totals), max(BC_totals)], color='gray');
        axs[0].set_title("Total Read Counts Per Sample", fontsize=32)
        #axs[0].set_yscale('log');
    
        axs[0].set_xlim(0,97);
        axs[0].set_xlabel('Sample Number', size=20)
        axs[0].set_ylabel('Total Reads per Sample', size=20);
        axs[0].tick_params(labelsize=16);
    
        axs[1].matshow(BC_total_arr, cmap="inferno");
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
                            num_to_plot=None):
    
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        if save_plots:
            pdf_file = 'barcode read fraction plots.pdf'
            pdf = PdfPages(pdf_file)
    
        os.chdir(self.data_directory)
        
        #Plot read fraction across all samples for first several barcodes
        plt.rcParams["figure.figsize"] = [16,6*num_to_plot]
        fig, axs = plt.subplots(num_to_plot, 1)
    
        f_data = self.barcode_frame[:num_to_plot]
    
        for (index, row), ax in zip(f_data.iterrows(), axs):
            y = []
            x = []
            y_for_scale = []
            for i, t in enumerate(fitness.wells_by_column()):
                y.append(row["fraction_" + t])
                x.append(i+1)
                if (row["fraction_" + t])>0:
                    y_for_scale.append(row["fraction_" + t])
    
            ax.scatter(x, y, c=plot_colors12(), s=70);
            ax.set_ylim(0.5*min(y_for_scale), 2*max(y));
            ax.set_yscale("log")
            barcode_str = str(index) + ', '
            if row['RS_name'] != "": barcode_str += row['RS_name'] + ", "
            barcode_str += row['forward_BC'] + ', ' + row['reverse_BC']
            ax.text(x=0.05, y=0.95, s=barcode_str, horizontalalignment='left', verticalalignment='top',
                     transform=ax.transAxes, fontsize=14)
        
            for i in range(13):
                ax.plot([i*8+0.5, i*8+0.5],[0.6*min(y_for_scale), 1.2*max(y)], color='gray');
        axs[0].set_title("Read Fraction Per Barcode", fontsize=32);
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
                            inducer_conc_list=None,
                            plot_range=None,
                            inducer=None,
                            include_ref_seqs=True,
                            includeChimeras=False,
                            ylim = None):
        
        low_tet = self.low_tet
        high_tet = self.high_tet
        
        if plot_range is None:
            barcode_frame = self.barcode_frame
        else:
            barcode_frame = self.barcode_frame.loc[plot_range[0]:plot_range[1]]
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        if include_ref_seqs:
            RS_count_frame = self.barcode_frame[self.barcode_frame["RS_name"]!=""]
            barcode_frame = pd.concat([barcode_frame, RS_count_frame])
        
        if inducer_conc_list is None:
            inducer_conc_list = self.inducer_conc_list
            
        if inducer is None:
            inducer = self.inducer
            
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        os.chdir(self.data_directory)
        if save_plots:
            pdf_file = 'barcode fitness plots.pdf'
            pdf = PdfPages(pdf_file)
        
        #plot fitness curves
        num_plot_rows = int(np.round(len(barcode_frame)/2 + 0.1))
        plt.rcParams["figure.figsize"] = [16,6*num_plot_rows]
        fig, axs_grid = plt.subplots(num_plot_rows, 2)
        plt.subplots_adjust(hspace = .35)
        axs = axs_grid.flatten()
        #if len(barcode_frame)==1:
        #    axs = [ axs ]
        x = inducer_conc_list
        linthreshx = min([i for i in inducer_conc_list if i>0])
        
        fit_plot_colors = sns.color_palette()
        
        for (index, row), ax in zip(barcode_frame.iterrows(), axs): # iterate over barcodes
            for initial in ["b", "e"]:
                y = row[f"fitness_{low_tet}_estimate_{initial}"]
                s = row[f"fitness_{low_tet}_err_{initial}"]
                fill_style = "full" if initial=="b" else "none"
                ax.errorbar(x, y, s, marker='o', ms=8, color=fit_plot_colors[0], fillstyle=fill_style)
                y = row[f"fitness_{high_tet}_estimate_{initial}"]
                s = row[f"fitness_{high_tet}_err_{initial}"]
                ax.errorbar(x, y, s, marker='^', ms=8, color=fit_plot_colors[1], fillstyle=fill_style)
                if ylim is not None:
                    ax.set_ylim(ylim);
            
                if initial == "b":
                    barcode_str = str(index) + ': '
                    barcode_str += format(row[f'total_counts'], ",") + "; "
                    barcode_str += row['RS_name'] + ": "
                    barcode_str += row['forward_BC'] + ",\n"
                    barcode_str += row['reverse_BC'] + " "
                    ax.text(x=1, y=1.1, s=barcode_str, horizontalalignment='right', verticalalignment='top',
                            transform=ax.transAxes, fontsize=13, fontfamily="Courier New")
                    ax.set_xscale('symlog', linthreshx=linthreshx)
                    ax.set_xlim(-linthreshx/10, 2*max(x));
                    ax.set_xlabel(f'[{inducer}] (umol/L)', size=14)
                    ax.set_ylabel('Fitness (log(10)/plate)', size=14)
                    ax.tick_params(labelsize=12);
            
        if save_plots:
            pdf.savefig()
    
        if save_plots:
            pdf.close()
    
    def plot_fitness_difference_curves(self,
                                       save_plots=False,
                                       inducer_conc_list=None,
                                       plot_range=None,
                                       inducer=None,
                                       include_ref_seqs=True,
                                       includeChimeras=False,
                                       show_fits=True):
        
        low_tet = self.low_tet
        high_tet = self.high_tet
        
        if plot_range is None:
            barcode_frame = self.barcode_frame
        else:
            barcode_frame = self.barcode_frame.loc[plot_range[0]:plot_range[1]]
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        if include_ref_seqs:
            RS_count_frame = self.barcode_frame[self.barcode_frame["RS_name"]!=""]
            barcode_frame = pd.concat([barcode_frame, RS_count_frame])
        
        if inducer_conc_list is None:
            inducer_conc_list = self.inducer_conc_list
            
        if inducer is None:
            inducer = self.inducer
        
        if "sensor_params" not in barcode_frame.columns:
            show_fits = False
            
        if show_fits:
            fit_fitness_difference_params = self.fit_fitness_difference_params
            
            if len(barcode_frame["sensor_params"].iloc[0])==7:
                # ['log_low_level', 'log_high_level', 'log_IC_50', 'log_sensor_n', 'low_fitness', 'mid_g', 'fitness_n']
                def fit_funct(x, log_g_min, log_g_max, log_x_50, log_nx, low_fitness, mid_g, fitness_n):
                    return double_hill_funct(x, 10**log_g_min, 10**log_g_max, 10**log_x_50, 10**log_nx,
                                             low_fitness, 0, mid_g, fitness_n)
            else:
                def fit_funct(x, g_min, g_max, x_50, nx):
                    return double_hill_funct(x, g_min, g_max, x_50, nx, fit_fitness_difference_params[0], 0,
                                             fit_fitness_difference_params[1], fit_fitness_difference_params[2])
            
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        os.chdir(self.data_directory)
        if save_plots:
            pdf_file = 'barcode fitness difference plots.pdf'
            pdf = PdfPages(pdf_file)
        
        #plot fitness curves
        num_plot_rows = int(np.round(len(barcode_frame)/2 + 0.1))
        plt.rcParams["figure.figsize"] = [16,6*num_plot_rows]
        fig, axs_grid = plt.subplots(num_plot_rows, 2)
        plt.subplots_adjust(hspace = .35)
        axs = axs_grid.flatten()
        #if len(barcode_frame)==1:
        #    axs = [ axs ]
        x = inducer_conc_list
        linthreshx = min([i for i in inducer_conc_list if i>0])
        
        fit_plot_colors = sns.color_palette()
        
        for (index, row), ax in zip(barcode_frame.iterrows(), axs): # iterate over barcodes
            for initial in ["b", "e"]:
                y_low = row[f"fitness_{low_tet}_estimate_{initial}"]
                s_low = row[f"fitness_{low_tet}_err_{initial}"]
                y_high = row[f"fitness_{high_tet}_estimate_{initial}"]
                s_high = row[f"fitness_{high_tet}_err_{initial}"]
                
                y = (y_high - y_low)/y_low.mean()
                s = np.sqrt( s_high**2 + s_low**2 )/y_low.mean()
                fill_style = "full" if initial=="b" else "none"
                ax.errorbar(x, y, s, marker='o', ms=8, color=fit_plot_colors[0], fillstyle=fill_style)
            
                if initial == "b":
                    barcode_str = str(index) + ': '
                    barcode_str += format(row[f'total_counts'], ",") + "; "
                    barcode_str += row['RS_name'] + ": "
                    barcode_str += row['forward_BC'] + ",\n"
                    barcode_str += row['reverse_BC'] + " "
                    ax.text(x=1., y=1.1, s=barcode_str, horizontalalignment='right', verticalalignment='top',
                            transform=ax.transAxes, fontsize=13, fontfamily="Courier New")
                    ax.set_xscale('symlog', linthreshx=linthreshx)
                    ax.set_xlim(-linthreshx/10, 2*max(x));
                    ax.set_xlabel(f'[{inducer}] (umol/L)', size=14)
                    ax.set_ylabel('Fitness with Tet - Fitness without Tet', size=14)
                    ax.tick_params(labelsize=12);
                    
            if show_fits:
                x_fit = np.logspace(np.log10(linthreshx/10), np.log10(2*max(x)))
                x_fit = np.insert(x_fit, 0, 0)
                params = row["sensor_params"]
                y_fit = fit_funct(x_fit, *params)
                ax.plot(x_fit, y_fit, color='k', zorder=1000);
            
        if save_plots:
            pdf.savefig()
    
        if save_plots:
            pdf.close()
    
    def plot_fitness_and_difference_curves(self,
                            save_plots=False,
                            inducer_conc_list=None,
                            plot_range=None,
                            inducer=None,
                            include_ref_seqs=True,
                            includeChimeras=False,
                            ylim = None,
                            show_fits=True,
                            show_GP=False,
                            log_g_scale=False):
        
        low_tet = self.low_tet
        high_tet = self.high_tet
        
        if plot_range is None:
            barcode_frame = self.barcode_frame
        else:
            barcode_frame = self.barcode_frame.loc[plot_range[0]:plot_range[1]]
        
        if "sensor_params" not in barcode_frame.columns:
            show_fits = False
            
        if show_fits:
            fit_fitness_difference_params = self.fit_fitness_difference_params
            
            if len(barcode_frame["sensor_params"].iloc[0])==7:
                # ['log_low_level', 'log_high_level', 'log_IC_50', 'log_sensor_n', 'low_fitness', 'mid_g',
                #  'fitness_n']
                def fit_funct(x, log_g_min, log_g_max, log_x_50, log_nx, low_fitness, mid_g, fitness_n, *argv):
                    return double_hill_funct(x, 10**log_g_min, 10**log_g_max, 10**log_x_50, 10**log_nx,
                                             low_fitness, 0, mid_g, fitness_n)
            elif len(barcode_frame["sensor_params"].iloc[0])>7:
                # ['log_low_level', 'log_high_level', 'log_IC_50', 'log_sensor_n', 'log_high_low_ratio',
                #  'low_fitness', 'mid_g', 'fitness_n']
                def fit_funct(x, log_g_min, log_g_max, log_x_50, log_nx, log_high_low_ratio,
                              low_fitness, mid_g, fitness_n, *argv):
                    return double_hill_funct(x, 10**log_g_min, 10**log_g_max, 10**log_x_50, 10**log_nx,
                                             low_fitness, 0, mid_g, fitness_n)
            else:
                def fit_funct(x, g_min, g_max, x_50, nx):
                    return double_hill_funct(x, g_min, g_max, x_50, nx, fit_fitness_difference_params[0], 0,
                                             fit_fitness_difference_params[1], fit_fitness_difference_params[2])
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        if include_ref_seqs:
            RS_count_frame = self.barcode_frame[self.barcode_frame["RS_name"]!=""]
            barcode_frame = pd.concat([barcode_frame, RS_count_frame])
        
        if inducer_conc_list is None:
            inducer_conc_list = self.inducer_conc_list
            
        if inducer is None:
            inducer = self.inducer
            
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        if save_plots:
            os.chdir(self.data_directory)
            pdf_file = 'barcode fitness plots.pdf'
            pdf = PdfPages(pdf_file)
        
        #plot fitness curves
        
        x = inducer_conc_list
        linthreshx = min([i for i in inducer_conc_list if i>0])
        
        fit_plot_colors = sns.color_palette()
        
        for index, row in barcode_frame.iterrows(): # iterate over barcodes
            if show_GP:
                plt.rcParams["figure.figsize"] = [16,12]
                fig, axs_grid = plt.subplots(2, 2)
                axl = axs_grid.flatten()[0]
                axr = axs_grid.flatten()[2]
                axg = axs_grid.flatten()[1]
                axdg = axs_grid.flatten()[3]
            else:
                plt.rcParams["figure.figsize"] = [16,6]
                fig, axs_grid = plt.subplots(1, 2)
                axl = axs_grid.flatten()[0]
                axr = axs_grid.flatten()[1]
                axg = axs_grid.flatten()[0]
                axdg = axs_grid.flatten()[0]
            plt.subplots_adjust(hspace = .35)
            
            for ax in axs_grid.flatten():
                ax.set_xscale('symlog', linthreshx=linthreshx)
                ax.set_xlim(-linthreshx/7, 1.5*max(x));
                ax.set_xlabel(f'[{inducer}] (umol/L)', size=14)
            
            for initial in ["b", "e"]:
                y = row[f"fitness_{low_tet}_estimate_{initial}"]
                s = row[f"fitness_{low_tet}_err_{initial}"]
                fill_style = "full" if initial=="b" else "none"
                axl.errorbar(x, y, s, marker='o', ms=8, color=fit_plot_colors[0], fillstyle=fill_style)
                y = row[f"fitness_{high_tet}_estimate_{initial}"]
                s = row[f"fitness_{high_tet}_err_{initial}"]
                axl.errorbar(x, y, s, marker='^', ms=8, color=fit_plot_colors[1], fillstyle=fill_style)
                
                y_low = row[f"fitness_{low_tet}_estimate_{initial}"]
                s_low = row[f"fitness_{low_tet}_err_{initial}"]
                y_high = row[f"fitness_{high_tet}_estimate_{initial}"]
                s_high = row[f"fitness_{high_tet}_err_{initial}"]
                
                y = (y_high - y_low)/y_low.mean()
                s = np.sqrt( s_high**2 + s_low**2 )/y_low.mean()
                fill_style = "full" if initial=="b" else "none"
                axr.errorbar(x, y, s, marker='o', ms=8, color=fit_plot_colors[0], fillstyle=fill_style)
            
                
                if ylim is not None:
                    axl.set_ylim(ylim);
            
                if initial == "b":
                    barcode_str = str(index) + ': '
                    barcode_str += format(row[f'total_counts'], ",") + "; "
                    barcode_str += row['RS_name'] + ": "
                    barcode_str += row['forward_BC'] + ", "
                    barcode_str += row['reverse_BC'] + " "
                    axl.text(x=1, y=1.05, s=barcode_str, horizontalalignment='center', verticalalignment='top',
                            transform=axl.transAxes, fontsize=13, fontfamily="Courier New")
                    axl.set_ylabel('Fitness (log(10)/plate)', size=14)
                    axl.tick_params(labelsize=12);
                    axr.set_ylabel('Fitness with Tet - Fitness without Tet', size=14)
                    axr.tick_params(labelsize=12);
                    
            if show_fits:
                x_fit = np.logspace(np.log10(linthreshx/10), np.log10(2*max(x)))
                x_fit = np.insert(x_fit, 0, 0)
                params = row["sensor_params"]
                y_fit = fit_funct(x_fit, *params)
                axr.plot(x_fit, y_fit, color='k', zorder=1000);
                
            if show_GP:
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
                x_fit = np.logspace(np.log10(linthreshx/10), np.log10(2*max(x)))
                x_fit = np.insert(x_fit, 0, 0)
                hill_params = 10**row["sensor_params"][:4]
                axg.plot(x_fit, hill_funct(x_fit, *hill_params), c='k', zorder=1000)
            
        if save_plots:
            pdf.savefig()
    
        if save_plots:
            pdf.close()
    
    def plot_count_ratios_vs_time(self, plot_range,
                                  inducer=None,
                                  inducer_conc_list=None,
                                  with_tet=None,
                                  mark_samples=[]):
        if with_tet is None:
            plot_tet = True
            plot_no_tet = True
        else:
            plot_tet = with_tet
            plot_no_tet = not with_tet
            
        barcode_frame = self.barcode_frame
        low_tet = self.low_tet
        high_tet = self.high_tet
        
        plot_count_frame = barcode_frame.iloc[plot_range[0]:plot_range[1]]
        plt.rcParams["figure.figsize"] = [10,6*(len(plot_count_frame))]
        fig, axs = plt.subplots(len(plot_count_frame), 1)
    
        if inducer_conc_list is None:
            inducer_conc_list = self.inducer_conc_list
            
        if inducer is None:
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
        wells_with_low_tet = []
    
        for i in range(2,6):
            df = sample_plate_map[(sample_plate_map["with_tet"]) & (sample_plate_map["growth_plate"]==i)]
            df = df.sort_values([inducer])
            wells_with_high_tet.append(df["well"].values)
            df = sample_plate_map[(sample_plate_map["with_tet"] != True) & (sample_plate_map["growth_plate"]==i)]
            df = df.sort_values([inducer])
            wells_with_low_tet.append(df["well"].values)
    
        for i in range(2,6):
            counts_0 = []
            counts_tet = []
            for index, row in barcode_frame.iterrows():
                row_0 = row[wells_with_low_tet[i-2]]
                counts_0.append(row_0.values)
                row_tet = row[wells_with_high_tet[i-2]]
                counts_tet.append(row_tet.values)
            barcode_frame[f"read_count_{low_tet}_" + str(i)] = counts_0
            barcode_frame[f"read_count_{high_tet}_" + str(i)] = counts_tet

        spike_in_row = {"AO-B": barcode_frame[barcode_frame["RS_name"]=="AO-B"],
                        "AO-E": barcode_frame[barcode_frame["RS_name"]=="AO-E"]}
        
        #Run for both AO-B and AO-E
        for spike_in, initial in zip(["AO-B", "AO-E"], ["b", "e"]):
            spike_in_reads_0 = [ spike_in_row[spike_in][f'read_count_{low_tet}_{plate_num}'].values[0] for plate_num in range(2,6) ]
            spike_in_reads_tet = [ spike_in_row[spike_in][f'read_count_{high_tet}_{plate_num}'].values[0] for plate_num in range(2,6) ]
        
            x0 = [2, 3, 4, 5]
        
            for (index, row), ax in zip(plot_count_frame.iterrows(), axs): # iterate over barcodes
                x_mark = []
                y_mark = []
                
                if plot_no_tet:
                    n_reads = [ row[f'read_count_{low_tet}_{plate_num}'] for plate_num in range(2,6) ]
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
                        ax.errorbar(x, y, s, c=plot_colors()[j], marker='^', ms=8, fillstyle=fillstyle, label=label)
                    
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
        
            
    def plot_chimera_plot(self,
                          save_plots=False,
                          chimera_cut_line=None):
            
        barcode_frame = self.barcode_frame[self.barcode_frame["possibleChimera"]]
        
        # Turn interactive plotting on or off depending on show_plots
        plt.ion()
        
        os.chdir(self.data_directory)
        if save_plots:
            pdf_file = 'barcode fitness plots.pdf'
            pdf = PdfPages(pdf_file)
        
        plt.rcParams["figure.figsize"] = [8,8]
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
            
    # Method for plotting sub-frame on background of full library distribution
    def plot_hill_params(self, input_frames, in_labels=None, in_colors=None, in_alpha=0.7,
                         error_bars=True, log_high_err_cutoff=0.71, legend=True,
                         everything_color=None):
        
        if in_labels is None:
            in_labels = [""] * len (input_frames)
        
        if in_colors is None:
            in_colors = [fitness.gray_out("indigo")] * len (input_frames)
            
        if everything_color is None:
            everything_color = fitness.gray_out("xkcd:tea green", s_factor=0.7, v_factor=0.8)
            
        plt.rcParams["figure.figsize"] = [16, 16]
        fig, axs_grid = plt.subplots(2, 2)
        axs = axs_grid.flatten()
    
        param_names = ["Low Level", "High Level", "High Low Level Ratio", "n"]
    
        x_label = f'IC50'
        x_err_label = f'IC50 error'
    
        # This part plots the input input_frames
        for input_frame, c, lab in zip(input_frames, in_colors, in_labels):
            for ax, name in zip(axs, param_names):
                y_label = f'{name}'
                y_err_label = f'{name} error'
        
                params_x = input_frame[x_label]
                params_y = input_frame[y_label]
                err_x = input_frame[x_err_label]
                err_y = input_frame[y_err_label]
                
                if error_bars:
                    ax.errorbar(params_x, params_y, yerr=err_y, xerr=err_x, fmt="o", ms=4, color=c,
                                label=lab, alpha=in_alpha);
                else:
                    ax.plot(params_x, params_y, "o", ms=4, color=c,
                            label=lab, alpha=in_alpha);
    
        # This part plots all the rest
        plot_frame = self.barcode_frame[3:]
        plot_frame = plot_frame[plot_frame["total_counts"]>3000]
        plot_frame = plot_frame[plot_frame["log_high_level error"]<log_high_err_cutoff]
        for ax, name in zip(axs, param_names):
            y_label = f'{name}'
    
            params_x = plot_frame[x_label]
            params_y = plot_frame[y_label]
            
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
        
    def cleaned_frame(self, count_threshold=3000, log_high_error_cutoff=0.71, num_good_hill_points=12):
        frame = self.barcode_frame[3:]
        frame = frame[frame["total_counts"]>count_threshold]
        frame = frame[frame["log_high_level error"]<log_high_error_cutoff]
        frame = frame[frame["good_hill_fit_points"]==num_good_hill_points]
        return frame

def plot_colors():
    return sns.hls_palette(12, l=.4, s=.8)
    
def plot_colors12():
    p_c12 = [ ]
    for c in plot_colors():
        for i in range(8):
            p_c12.append(c)
    return p_c12
                
def hill_funct(x, low, high, mid, n):
    return low + (high-low)*( x**n )/( mid**n + x**n )

# Hill function of Hill function to describe fitness_difference(gene_expression([inducer]))
def double_hill_funct(x, g_min, g_max, x_50, nx, f_min, f_max, g_50, ng):
    # g_min, g_max, x_50, and nx are characteristics of individual sensor variants
        # g_min is the minimum gene epxression level, at zero inducer
        # g_max is the maximum gene expresion level, at full induction
        # x_50 is the inducer concentration of 1/2 max gene expression
        # nx is the exponent that describes the steepness of the sensor response curve
    # f_min, f_max, g_50, and ng are characteristics of the selection system
    # they are estimated from the fits above
        # f_min is the minimum fitness level, at zero gene expression
        # f_max is the maximum fitness level, at infinite gene expression (= 0)
        # g_50 is the gene expression of 1/2 max fitness
        # ng is the exponent that describes the steepness of the fitness vs. gene expression curve
    return hill_funct( hill_funct(x, g_min, g_max, x_50, nx), f_min, f_max, g_50, ng )
    

def init_stan_fit(x_data, y_data, fit_fitness_difference_params):
    log_low_level = log_level(np.mean(y_data[:2]))
    log_high_level = log_level(np.mean(y_data[-2:]))
    
    min_ic = np.log10(min([i for i in x_data if i>0]))
    max_ic = np.log10(max(x_data))
    log_IC_50 = np.random.uniform(min_ic, max_ic)
    
    n = np.random.uniform(1.3, 1.7)
    
    sig = np.random.uniform(1, 3)
    
    low_fitness = fit_fitness_difference_params[0]
    mid_g = fit_fitness_difference_params[1]
    fitness_n = fit_fitness_difference_params[2]
    
    return dict(log_low_level=log_low_level, log_high_level=log_high_level, log_IC_50=log_IC_50,
                sensor_n=n, sigma=sig, low_fitness=low_fitness, mid_g=mid_g, fitness_n=fitness_n)
    
def init_stan_GP_fit(fit_fitness_difference_params):
    sig = np.random.uniform(1, 3)
    rho = np.random.uniform(0.9, 1.1)
    alpha = np.random.uniform(0.009, 0.011)
    
    low_fitness = fit_fitness_difference_params[0]
    mid_g = fit_fitness_difference_params[1]
    fitness_n = fit_fitness_difference_params[2]
    
    return dict(sigma=sig, low_fitness=low_fitness, mid_g=mid_g, fitness_n=fitness_n, rho=rho, alpha=alpha)
    
def log_level(fitness_difference):
    log_g = 1.439*fitness_difference + 3.32
    log_g = log_g*np.random.uniform(0.9,1.1)
    if log_g<1.5:
        log_g = 1.5
    if log_g>4:
        log_g = 4
    return log_g


    
