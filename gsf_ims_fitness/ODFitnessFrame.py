# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:09:08 2019

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

from . import fitness

sns.set_style("white")
sns.set_style("ticks", {'xtick.direction':'in', 'xtick.top':True, 'ytick.direction':'in', 'ytick.right':True})

class ODFitnessFrame:
        
    def __init__(self, notebook_dir, experiment=None, wavelength='600', auto_save=True):
        
        os.chdir(notebook_dir)
    
        self.notebook_dir = notebook_dir
        self.experiment = experiment
        self.wavelength = wavelength
        
        if self.experiment is None:
            self.experiment = fitness.get_exp_id(self.notebook_dir)
        
        print(f"Loading plate reader data and calculating fitness for experiment: {self.experiment}")
        
        worklist_files = glob.glob("*growth-plate*.csv")
        worklist_files.sort()
        if len(worklist_files)!=5:
            print(f"Warning! Unexpected number of worklist .csv files. Expected 5 files; found {len(worklist_files)} files")
            print(worklist_files)
        file_warning = False
        for f in worklist_files:
            if self.experiment not in f:
                file_warning = True
        if file_warning:
            print(f"Warning! Experiment name, {self.experiment}, not a substring of the worklist filenames.")
            print(worklist_files)
            
        sample_frames = [ pd.read_csv(file) for file in worklist_files ]
        
        gen5_files = glob.glob("*growth-plate*a_Neo.txt")
        gen5_files.sort()
        if len(gen5_files)!=5:
            print(f"Warning! Unexpecte number of plate reader output .txt files. Expected 5 files; found {len(gen5_files)} files")
            print(gen5_files)
        
        plate_read_frames = []
        for file in gen5_files:
            read_line_numbers = []
            with open(file) as f:
                for line_num, line in enumerate(f):
                    number_of_lines = line_num
                    if line.startswith('Time,T° '):
                        line = line[:-1]
                        #print(line)
                        read_line_numbers.append(line_num)
                        if self.wavelength in line:
                            read_number_to_use = len(read_line_numbers)
            read_line_numbers.append(number_of_lines)
            skiprows = read_line_numbers[read_number_to_use-1]
            nrows = read_line_numbers[read_number_to_use]-read_line_numbers[read_number_to_use-1] - 4
            gen5_frame = pd.read_csv(file, skiprows=skiprows, nrows=nrows, skip_blank_lines=False, encoding ="latin1")
            plate_read_frames.append(gen5_frame)
            
        for frame in plate_read_frames:
            frame['Time'] = pd.to_timedelta(frame['Time'])
            frame['Time (min)'] = [ td.total_seconds()/60 for td in frame['Time'] ]
            frame['Time (h)'] = frame['Time (min)']/60
            
        #parameters for fitting to growth curves to estimate starting density
        fit_start = 0.5
        fit_end = 2.25
        select_end = 3.42
        
        prior_OD_mean = 1.6662
        prior_OD_var = 0.0502**2
        
        #2nd set of offset priors is with drift subtracted
        prior_OD_mean = 1.6637
        prior_OD_var = 0.0502**2
        
        drift_rate = 0.001974264705882297
        
        plate_num = 0
        for plate, layout in zip(plate_read_frames, sample_frames):
            plate_num += 1
            popt_list = []
            back_like_list = []
            back_list = []
            back_var_list = []
            back_err_list = []
            doubling_list = []
            end_density_list = []
            
            fit_frame = plate[plate['Time (h)']>fit_start]
            
            select_frame = fit_frame[fit_frame['Time (h)']<select_end]
            x2 = select_frame['Time (h)']
               
            fit_frame = fit_frame[fit_frame['Time (h)']<fit_end]
            x = fit_frame['Time (h)']
            
            p0 = [1.65, 0.1, 1]
            bounds = ([1.4, 0, 0], [1.9, 1., 10])
            
            for well in layout['well']:
                y = fit_frame[well] - drift_rate*x
                y2 = select_frame[well] - drift_rate*x2
                popt, pcov = curve_fit(fitness.exp_funct, x, y, p0=p0, bounds=bounds, max_nfev=len(x)*1000)
                popt_list.append(popt)
                back_like_list.append(popt[0])
                back_var_list.append(pcov[0, 0])
                
                posterior_offset = (prior_OD_var*popt[0] + pcov[0, 0]*prior_OD_mean)
                posterior_offset = posterior_offset/(prior_OD_var + pcov[0, 0])
                back_list.append(posterior_offset)
                
                posterior_err = prior_OD_var*pcov[0, 0]
                posterior_err = posterior_err/(prior_OD_var + pcov[0, 0])
                posterior_err = posterior_err**0.5
                back_err_list.append(posterior_err)
                
                doubling_list.append(popt[2])
                end_density_list.append(y2.iloc[-1] - posterior_offset)
            
            #fits_popt.append(popt_list)
            layout['fit_params'] = popt_list
            layout['OD_offset'] = back_list
            layout['OD_offset_like'] = back_like_list
            layout['OD_offset_var'] = back_var_list
            layout['doubling_time'] = doubling_list
            layout['end_density'] = end_density_list
            layout['end_density_err'] = back_err_list
        
        frame_0 = sample_frames[0]
        y_lists = [ [] for i in range(len(frame_0)) ]
        x_lists = [ [] for i in range(len(frame_0)) ]
        sig_lists = [ [] for i in range(len(frame_0)) ]
        slopes = []
        slopes_err = []
        
        #create fitness_frame and add density lists (list of final density for plates 2-5 for each sample)
        frame_0 = sample_frames[0]
        y_lists = [ [] for i in range(len(frame_0)) ]
        x_lists = [ [] for i in range(len(frame_0)) ]
        sig_lists = [ [] for i in range(len(frame_0)) ]
        density_list = []
        density_err_list = []
        
        for i, plate in enumerate(sample_frames):
            if i>0:
                for x_l, y_l, s_l, density, dens_err in zip(x_lists, y_lists, sig_lists, plate['end_density'], plate['end_density_err']):
                    if density>0:
                        y_l.append(density)
                        x_l.append(i+1);
                        s_l.append(dens_err)
                    else:
                        y_l.append(np.nan)
                        s_l.append(np.nan)
        
        for x, y, y_err, well in zip(x_lists, y_lists, sig_lists, frame_0['well']):
            density_list.append(np.asarray(y))
            density_err_list.append(np.asarray(y_err))
        
        fitness_frame = pd.DataFrame(frame_0['well'])
        fitness_frame['density_list'] = density_list
        fitness_frame['density_err_list'] = density_err_list
        
        fitness_frame['inducerConcentration'] = sample_frames[-1]['inducerConcentration']
        fitness_frame['tet_concentration'] = sample_frames[-1]['selectorConc']
        fitness_frame['plasmid'] = sample_frames[-1]['plasmid']
        
        #start with linear fit for fitness estimate
        slopes = []
        x_list = [2, 3, 4, 5]
        bounds = ([-np.log(10), -15], [10, 15])
        
        for index, row in fitness_frame.iterrows():
            y = row['density_list']
            y_err = row['density_err_list']
            
            sigma = np.asarray(y_err)/np.asarray(y)
            
            x_fit = []
            y_fit = []
            sigma = []
            for x_f, y_f, s_f in zip(x_list, y, y_err):
                if not np.isnan(y_f):
                    x_fit.append(x_f)
                    y_fit.append(y_f)
                    sigma.append(s_f/y_f)
            
            popt, pcov = curve_fit(fitness.line_funct, x_fit, np.log(y_fit), bounds=bounds, sigma=sigma)
            slopes.append(popt[0])
            slopes_err.append( np.sqrt(pcov[0,0]) )
            
        fitness_frame['fitness'] = slopes
        fitness_frame['fitness_err'] = slopes_err
        
        #then redo fits for non-zero tet, using bi-linear form
        slopes = []
        slopes_err = []
        
        no_tet_frame = fitness_frame[fitness_frame["tet_concentration"] == 0]
        
        for index, row in fitness_frame.iterrows():
            if row["tet_concentration"] == 0:
                slopes.append(row["fitness"])
                slopes_err.append(row["fitness_err"])
            else:
                y = row['density_list']
                y_err = row['density_err_list']
        
                indConc = row['inducerConcentration']
        
                sigma = np.asarray(y_err)/np.asarray(y)
        
                x_fit = []
                y_fit = []
                sigma = []
                for x_f, y_f, s_f in zip(x_list, y, y_err):
                    if not np.isnan(y_f):
                        x_fit.append(x_f)
                        y_fit.append(y_f)
                        sigma.append(s_f/y_f)
        
                slope_0 = no_tet_frame[no_tet_frame["inducerConcentration"] == indConc]["fitness"].mean()
                def fit_funct(xp, mp, bp): return fitness.bi_linear_funct(xp-2, mp, bp, m1=slope_0, alpha=np.log(5))
                bounds = ([-np.log(10), -15], [slope_0, 15])
                popt, pcov = curve_fit(fit_funct, x_fit, np.log(y_fit), bounds=bounds, sigma=sigma, absolute_sigma=True)
                slopes.append(popt[0])
                slopes_err.append( np.sqrt(pcov[0,0]) )
            
        ref_fitness = np.log(10)
        fitness_frame['fitness'] = slopes/ref_fitness + 1
        fitness_frame['fitness_err'] = slopes_err/ref_fitness
        
        self.fitness_frame = fitness_frame
        
        if auto_save:
            self.save_as_pickle()
    
    def save_as_pickle(self, notebook_dir=None, experiment=None, pickle_file=None):
        if notebook_dir is None:
            notebook_dir = self.notebook_dir
        if experiment is None:
            experiment = self.experiment
        if pickle_file is None:
            pickle_file = experiment + '_ODFitnessFrame.pkl'
            
        os.chdir(notebook_dir)
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)
        print(f"ODFitnessFrame saved as: {pickle_file}")
        
    def plot_cell_density(self,
                          show_plots=True,
                          save_plots=False,
                          y_min=-0.05,
                          y_max=0.7,
                          log_yscale=False,
                          clone_name=None):
        # Turn interactive plotting on or off depending on show_plots
        if show_plots:
            plt.ion()
        else:
            plt.ioff()
        
        if save_plots:
            os.chdir(self.notebook_dir)
            pdf_file = 'cell density plot.pdf'
            pdf = PdfPages(pdf_file)
        
        fitness_frame = self.fitness_frame
        
        #plot cell density for different inducer concentrations
        inducer_concentrations = np.unique(fitness_frame['inducerConcentration'].values)
        
        plt.rcParams["figure.figsize"] = [16,12]
        fig, axs = plt.subplots(3, 4)
        
        axes_array = axs.flatten()
        
        if clone_name is None:
            plasmids = np.unique(fitness_frame['plasmid'].values)
            fig.suptitle(self.experiment, fontsize=20, position=(0.5, 0.92))
        else:
            plasmids = np.array([clone_name])
        
        x = [ i for i in range(2,len(fitness_frame["density_list"].iloc[0]) + 2) ]
        for conc, ax in zip(inducer_concentrations, axes_array):
            plot_frame = fitness_frame[fitness_frame['inducerConcentration']==conc]
            for index, row in plot_frame.iterrows():
                y = row["density_list"]
                y_err = row["density_err_list"]
                tet_conc = row["tet_concentration"]
                label = row["plasmid"]
                if label in plasmids:
                    if len(plasmids)>1:
                        color_ind = np.where(plasmids==label)[0][0]
                    else:
                        color_ind = 0 if tet_conc ==0 else 1
                    color = sns.color_palette()[color_ind]
                    fmt = '-o' if tet_conc ==0 else '-^'
                    ax.errorbar(np.array(x)-1, y, yerr=y_err, fmt=fmt, color=color, markersize=10)
            if log_yscale:
                ax.set_yscale("log")
            ax.set_ylim(y_min,y_max);
            ax.tick_params(labelsize=16);
            ax.text(0.95, 0.95, f'{conc*1000} µmol/L', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, size=14);
            
        axes_array[9].set_xlabel('Growth Plate Number', size=20)
        axes_array[9].xaxis.set_label_coords(1.05, -0.15)
        axes_array[4].set_ylabel('Cell Density (OD$\mathregular{_{600}}$)', size=20, labelpad=10);
        if save_plots:
            pdf.savefig()
        if not show_plots:
            plt.close(fig)
            
        if save_plots:
            pdf.close()
            
        return fig, axs
            
    def plot_fitness_curves(self,
                            show_plots=True,
                            save_plots=False,
                            y_min=None,
                            y_max=None,
                            inducer="IPTG",
                            si_plot=False,
                            clone_name=None):
        # Turn interactive plotting on or off depending on show_plots
        if show_plots:
            plt.ion()
        else:
            plt.ioff()
        
        notebook_dir = self.notebook_dir
        if save_plots:
            os.chdir(notebook_dir)
            pdf_file = 'fitness from OD plot.pdf'
            pdf = PdfPages(pdf_file)
            
        fitness_frame = self.fitness_frame
        
        #plot fitness curves
        Tet_concentrations = np.unique(fitness_frame['tet_concentration'].values)
        
        if clone_name is None:
            plasmids = np.unique(fitness_frame['plasmid'].values)
        else:
            plasmids = [clone_name]
        
        plt.rcParams["figure.figsize"] = [8,6]
        fig, axs = plt.subplots(1, 1)
        
        fitness_scale = fitness.fitness_scale()
        
        current_palette = sns.color_palette()
        plot_colors = current_palette
        for j, conc in enumerate(Tet_concentrations):
            if len(plasmids)==1:
                plot_colors = [ current_palette[j] ]
            for c, plas in zip(plot_colors, plasmids):
                frame = fitness_frame[(fitness_frame['tet_concentration']==conc)]
                frame = frame[(frame['plasmid']==plas)]
                x = list(1000*frame['inducerConcentration'])
                y = frame['fitness']
                y_err = frame['fitness_err']
                if si_plot:
                    y = y * fitness_scale
                    y_err = y_err * fitness_scale
                size = 10 #if conc==0 else 6
                if si_plot:
                    marker = "o" if conc==0 else "^"
                    c = plot_colors[0] if conc==0 else plot_colors[1]
                else:
                    marker = "-o" if conc==0 else "-^"
                    print(f"{plas}: {np.mean(y)} +- {np.std(y)}")
                if len(plasmids)>1:
                    label=plas + f', {conc}' if conc==0 else ""
                else:
                    label=plas + f', [tet] = {conc}'
                axs.errorbar(x, y, yerr=y_err, fmt=marker, label=label, markersize=size, color=c)
        linthresh = min([i for i in x if i>0])
        axs.set_xscale('symlog', linthresh=linthresh)
        if (y_min is not None) and (y_max is not None):
            axs.set_ylim(y_min, y_max);
        #axs.set_xlim(-linthreshx/10, 2*max(x));
        if si_plot==False:
            leg = axs.legend(loc='lower right', bbox_to_anchor= (0.975, 0.03), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
            leg.get_frame().set_edgecolor('k');
            axs.text(0.5, 1.025, self.experiment, horizontalalignment='center', verticalalignment='center', transform=axs.transAxes, size=20);
        axs.set_xlabel(f'[{inducer}] (µmol/L)', size=20)
        
        if si_plot:
            axs.plot(x, [0.9637 * fitness_scale]*len(x), c='k', linestyle="--")
            axs.plot(x, [0.8972 * fitness_scale]*len(x), c='k', linestyle="--")
            axs.set_ylabel('Fitness (hour$\mathregular{^{-1}}$)', size=20)
        else:
            axs.set_ylabel('Fitness (log(10)/plate)', size=20)
        
        axs.tick_params(labelsize=16);
        if save_plots:
            pdf.savefig()
        if not show_plots:
            plt.close(fig)
            
        if save_plots:
            pdf.close()
            
        return fig, axs
        