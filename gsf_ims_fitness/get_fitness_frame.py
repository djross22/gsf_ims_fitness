# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:31:07 2019

@author: djross
"""

import glob  # filenames and pathnames utility
import os    # operating sytem utility

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

def get_fitness_frame(notebook_dir, experiment=None, show_plots=False, wavelength='600'):
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    
    pdf_file = 'fitness frame plots.pdf'
    pdf = PdfPages(pdf_file)
        
    os.chdir(notebook_dir)
    
    if experiment is None:
        experiment = notebook_dir[notebook_dir.rfind('\\')+1:]
    
    print(f"Analyzing plate reader data and calculating fitness for experiment: {experiment}")
    
    worklist_files = glob.glob("*growth-plate*.csv")
    worklist_files.sort()
    if len(worklist_files)!=5:
        print(f"Warning! Unexpecte number of worklist .csv files. Expected 5 files; found {len(worklist_files)} files")
        print(worklist_files)
        
    sample_frames = [ pd.read_csv(file) for file in worklist_files ]
    
    rows = np.unique(sample_frames[-1]['row'].values)
    columns = np.unique(sample_frames[-1]['column'].values)
    
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
                    if wavelength in line:
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
    
    wells = sample_frames[-1]['well']
    
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
            popt, pcov = curve_fit(exp_funct, x, y, p0=p0, bounds=bounds, max_nfev=len(x)*1000)
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
    bounds = ([-np.log(10), -10], [10, 10])
    
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
        
        popt, pcov = curve_fit(line_funct, x_fit, np.log(y_fit), bounds=bounds, sigma=sigma)
        slopes.append(popt[0])
        slopes_err.append( np.sqrt(pcov[0,0]) )
        
    fitness_frame['fitness'] = slopes
    fitness_frame['fitness_err'] = slopes_err
    
    #then redo fits for non-zero tet, using bi-linear form
    slopes = []
    slopes_err = []
    bounds = ([-np.log(10), -10], [10, 10])
    
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
            def fit_funct(xp, mp, bp): return bi_linear_funct(xp-2, mp, bp, m1=slope_0, alpha=np.log(5))
            popt, pcov = curve_fit(fit_funct, x_fit, np.log(y_fit), bounds=bounds, sigma=sigma, absolute_sigma=True)
            slopes.append(popt[0])
            slopes_err.append( np.sqrt(pcov[0,0]) )
        
    fitness_frame['fitness'] = slopes
    fitness_frame['fitness_err'] = slopes_err
    
    #plot cell density for different inducer concentrations
    plt.rcParams["figure.figsize"] = [16,12]
    fig, axs = plt.subplots(3, 4)
    fig.suptitle(experiment, fontsize=20, position=(0.5, 0.92))
    axes_array = axs.flatten()
        
    for column, ax in zip(range(1,13), axes_array):
        frame_0 = sample_frames[0][sample_frames[0]['column']==column]
        y_lists = [ [] for i in range(len(frame_0)) ]
        x = []
    
        for i, plate in enumerate(sample_frames):
            x.append(i+1);
            frame = plate[plate['column']==column]
            for y_l, density in zip(y_lists, frame['end_density']):
                y_l.append(density)
    
        for y, well in zip(y_lists, frame_0['well']):
            lev = fitness_frame[fitness_frame['well']==well]['inducerConcentration'].mean()
            ax.plot(x, y, '-o', label=well + f', {lev:.2}')
            ax.set_ylim(-0.05,0.7);
    
        conc = plate[(plate['well']==f'A{column}')]['inducerConcentration'].values[0]
        ax.text(0.95, 0.95, f'{conc*1000} umol/L', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, size=14);
    axes_array[9].set_xlabel('Plate Number', size=20)
    axes_array[9].xaxis.set_label_coords(1.05, -0.15)
    axes_array[4].set_ylabel('Cell Density (OD600)', size=20, labelpad=10);
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    
    #plot fitness curves
    Tet_concentrations = np.unique(fitness_frame['tet_concentration'].values)
    plasmids = np.unique(fitness_frame['plasmid'].values)
    
    plt.rcParams["figure.figsize"] = [12, 9]
    fig, axs = plt.subplots(1, 1)
    
    ref_fitness = np.log(10)
    current_palette = sns.color_palette()
    plot_colors = current_palette
    for conc in Tet_concentrations:
        for c, plas in zip(plot_colors, plasmids):
            frame = fitness_frame[(fitness_frame['tet_concentration']==conc)]
            frame = frame[(frame['plasmid']==plas)]
            x = list(1000*frame['inducerConcentration'])
            y = frame['fitness']/ref_fitness + 1
            y_err = frame['fitness_err']/ref_fitness
            size = 10 #if conc==0 else 6
            marker = "-o" if conc==0 else "-^"
            label=plas + f', {conc}' if conc==0 else ""
            axs.errorbar(x, y, yerr=y_err, fmt=marker, label=label, markersize=size, color=c)
    axs.set_xscale('symlog', linthreshx=2)
    #axs.set_ylim(0.15, 1.05);
    axs.set_xlim(-x[4]/10, 2*max(x));
    leg = axs.legend(loc='lower right', bbox_to_anchor= (0.975, 0.1), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
    leg.get_frame().set_edgecolor('k');
    axs.set_xlabel('[IPTG] (umol/L)', size=20)
    axs.set_ylabel('Fitness (log(10)/plate)', size=20)
    axs.text(0.5, 1.025, experiment, horizontalalignment='center', verticalalignment='center', transform=axs.transAxes, size=20);
    axs.tick_params(labelsize=16);
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
        
    pdf.close()
    
    os.chdir(notebook_dir)
    pickle_file = experiment + '_fitness_from_OD.df_pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(fitness_frame, f)
    print(f"fitness_frame saved as: {pickle_file}")
        
    return fitness_frame
        
def exp_funct(x, background, A, doubling_time):
    return background + A*(2**(x/doubling_time) )

def line_funct(x, m, b):
    return m*x + b

def bi_linear_funct(z, m2, b, m1, alpha):
    return b + m2*z + ( m1 - m2 + (m2-m1)*np.exp(-z*alpha) )/alpha
        