# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:29:34 2019

@author: djross
"""

import glob  # filenames and pathnames utility
import os    # operating sytem utility

import matplotlib.pyplot as plt
#from matplotlib import colors
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
from ipywidgets import interact#, interact_manual

from . import fitness

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
            comp_data = barcode_frame.loc[:index-1]
                
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
        
    def mark_actual_chimeras(self, chimera_cut_line):
        
        barcode_frame = self.barcode_frame
        
        barcode_frame["isChimera"] = False
        
        for index, row in barcode_frame[barcode_frame["possibleChimera"]].iterrows():
            geo_mean = row["parent_geo_mean"]/96
            count = row["total_counts"]/96
            if count<chimera_cut_line(geo_mean):
                barcode_frame.at[index, "isChimera"] = True
        
        self.barcode_frame = barcode_frame
        
            
    def fit_barcode_fitness(self,
                            inducer_conc_list=None,
                            inducer=None,
                            auto_save=True,
                            ignore_samples=[]):
        
        barcode_frame = self.barcode_frame
        low_tet = self.low_tet
        high_tet = self.high_tet
            
        print(f"Fitting to log(barcode ratios) to find fitness for each barcode in {self.experiment}")
            
        os.chdir(self.data_directory)
    
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
        
            fit_frame = barcode_frame
        
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
            
            barcode_frame[f'fitness_{low_tet}_estimate_{initial}'] = f_0_est_list
            barcode_frame[f'fitness_{low_tet}_err_{initial}'] = f_0_err_list
            barcode_frame[f'fitness_{high_tet}_estimate_{initial}'] = f_tet_est_list
            barcode_frame[f'fitness_{high_tet}_err_{initial}'] = f_tet_err_list

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
            barcode_frame["sensor_params"] = [ np.full((3), np.nan) ] * len(barcode_frame)
        
        if "sensor_params_err" not in barcode_frame.columns:
            barcode_frame["sensor_params_cov"] = [ np.full((3, 3), np.nan) ] * len(barcode_frame)
            
        if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
            barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
            
        inducer_conc_list = self.inducer_conc_list
        
        x = np.array(inducer_conc_list)
        
        popt_list = []
        pcov_list = []
        
        for (index, row) in barcode_frame.iterrows(): # iterate over barcodes
            initial = "b"
            y_low = row[f"fitness_{low_tet}_estimate_{initial}"]
            s_low = row[f"fitness_{low_tet}_err_{initial}"]
            y_high = row[f"fitness_{high_tet}_estimate_{initial}"]
            s_high = row[f"fitness_{high_tet}_err_{initial}"]
            
            y = (y_high - y_low)/y_low.mean()
            s = np.sqrt( s_high**2 + s_low**2 )/y_low.mean()
            
            valid = ~(np.isnan(y) | np.isnan(s))
            
            p0 = [100, 1500, 200, 1.5]
            bounds = [[0.1, 0.1, 0.1, 0.1], [2000, 5000, max(x), 5]]
            try:
                popt, pcov = curve_fit(fit_fitness_difference_funct, x[valid], y[valid], sigma=s[valid], p0=p0, maxfev=len(x)*10000, bounds=bounds)
            except (RuntimeError, ValueError) as err:
                popt = np.full((3), np.nan)
                pcov = np.full((3, 3), np.nan)
                print(f"Error fitting curve for index {index}: {err}")
            
            popt_list.append(popt)
            pcov_list.append(pcov)
                
        barcode_frame["sensor_params"] = popt_list
        barcode_frame["sensor_params_cov"] = pcov_list
        
        self.barcode_frame = barcode_frame
        
        if auto_save:
            self.save_as_pickle()
            
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
                            includeChimeras=False):
        
        low_tet = self.low_tet
        high_tet = self.high_tet
        
        if plot_range is None:
            barcode_frame = self.barcode_frame
        else:
            barcode_frame = self.barcode_frame.iloc[plot_range[0]:plot_range[1]]
            
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
        plt.rcParams["figure.figsize"] = [12,8*(len(barcode_frame))]
        fig, axs = plt.subplots(len(barcode_frame), 1)
        if len(barcode_frame)==1:
            axs = [ axs ]
        x = inducer_conc_list
        linthreshx = min([i for i in inducer_conc_list if i>0])
        
        fit_plot_colors = sns.color_palette()
        
        for (index, row), ax in zip(barcode_frame.iterrows(), axs): # iterate over barcodes
            for initial in ["b", "e"]:
                y = row[f"fitness_{low_tet}_estimate_{initial}"]
                s = row[f"fitness_{low_tet}_err_{initial}"]
                fill_style = "full" if initial=="b" else "none"
                ax.errorbar(x, y, s, marker='o', ms=10, color=fit_plot_colors[0], fillstyle=fill_style)
                y = row[f"fitness_{high_tet}_estimate_{initial}"]
                s = row[f"fitness_{high_tet}_err_{initial}"]
                ax.errorbar(x, y, s, marker='^', ms=10, color=fit_plot_colors[1], fillstyle=fill_style)
            
                if initial == "b":
                    barcode_str = str(index) + ', '
                    barcode_str += str(row['total_counts']) + ", "
                    barcode_str += row['RS_name'] + ": "
                    barcode_str += row['forward_BC'] + ", "
                    barcode_str += row['reverse_BC']
                    ax.text(x=0.0, y=1.03, s=barcode_str, horizontalalignment='left', verticalalignment='top',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_xscale('symlog', linthreshx=linthreshx)
                    ax.set_xlim(-linthreshx/10, 2*max(x));
                    ax.set_xlabel(f'[{inducer}] (umol/L)', size=20)
                    ax.set_ylabel('Fitness (log(10)/plate)', size=20)
                    ax.tick_params(labelsize=16);
            
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
            barcode_frame = self.barcode_frame.iloc[plot_range[0]:plot_range[1]]
            
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
        plt.rcParams["figure.figsize"] = [12,8*(len(barcode_frame))]
        fig, axs = plt.subplots(len(barcode_frame), 1)
        if len(barcode_frame)==1:
            axs = [ axs ]
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
                ax.errorbar(x, y, s, marker='o', ms=10, color=fit_plot_colors[0], fillstyle=fill_style)
            
                if initial == "b":
                    barcode_str = str(index) + ', '
                    barcode_str += str(row['total_counts']) + ", "
                    barcode_str += row['RS_name'] + ": "
                    barcode_str += row['forward_BC'] + ", "
                    barcode_str += row['reverse_BC']
                    ax.text(x=0.0, y=1.03, s=barcode_str, horizontalalignment='left', verticalalignment='top',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_xscale('symlog', linthreshx=linthreshx)
                    ax.set_xlim(-linthreshx/10, 2*max(x));
                    ax.set_xlabel(f'[{inducer}] (umol/L)', size=20)
                    ax.set_ylabel('Fitness with Tet - Fitness without Tet', size=20)
                    ax.tick_params(labelsize=16);
                    
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
    



    
