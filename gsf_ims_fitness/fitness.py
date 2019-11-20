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

def analyze_bar_seq(notebook_dir,experiment=None, show_plots=False, cutoff=None, hist_bin_max=None, num_to_plot=20):
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    
    pdf_file = 'barcode frequency plots.pdf'
    pdf = PdfPages(pdf_file)
        
    os.chdir(notebook_dir)
    
    if experiment is None:
        experiment = notebook_dir[notebook_dir.find("Sequencing_data_downloads"):]
		experiment = experiment[experiment.find("\\")+1:]
		experiment = experiment[:experiment.find("\\")]
    
    print(f"Analyzing BarSeq data and calculating fitness for experiment: {experiment}")

	top_directory = notebook_dir[:notebook_dir.rfind("\\")]
	os.chdir(top_directory)
	ref_seq_file = "reference_sequences.csv"
	ref_seq_frame = pd.read_csv(ref_seq_file, skipinitialspace=True)

	data_directory = notebook_dir + "\\barcode_analysis"
	os.chdir(data_directory)

	barcode_file = glob.glob("*.sorted_counts.csv")[0]
	barcode_frame_0 = pd.read_csv(barcode_file, skipinitialspace=True)

	barcode_frame_0.sort_values('total_counts', ascending=False, inplace=True)
	barcode_frame_0.reset_index(drop=True, inplace=True)

	plt.rcParams["figure.figsize"] = [16,8]
	fig, axs = plt.subplots(1, 2)
	if hist_bin_max is None:
    	hist_bin_max = barcode_frame_0[int(len(barcode_frame_0)/50):int(len(barcode_frame_0)/50)+1]["total_counts"].values[0]
	bins = np.linspace(-0.5,hist_bin_max + 0.5,int(hist_bin_max/2)+1)
	for ax in axs.flatten():
		ax.hist(barcode_frame_0['total_counts'], bins=bins);
		ax.set_xlabel('Barcode Count', size=20)
		ax.set_ylabel('Number of Barcodes', size=20)
		ax.tick_params(labelsize=16);
	axs[0].hist(barcode_frame_0['total_counts'], bins=bins, histtype='step', cumulative=-1);
	axs[0].set_yscale('log');
	axs[1].set_yscale('log');
	axs[1].set_xlim(0,hist_bin_max/3);
	pdf.savefig()
    if not show_plots:
        plt.close(fig)

	if cutoff is None:
	    cutoff = int(hist_bin_max/10)
	print(f"Barcode frequency cutoff: {cutoff}")

	barcode_frame = barcode_frame_0[barcode_frame_0["total_counts"]>cutoff].copy()
	barcode_frame.sort_values('total_counts', ascending=False, inplace=True)
	barcode_frame.reset_index(drop=True, inplace=True)

	rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
	columns = [i for i in range(1,13)]
	wells = []
	for r in rows:
		for c in columns:
			wells.append(r + str(c))
        
	wells_by_column = []
	for c in columns:
		for r in rows:
			wells_by_column.append(r + str(c))

	for w in wells:
		fractions = []
		label = 'fraction_' + w
		barcode_frame[label] = barcode_frame[w]/barcode_frame[w].sum()
    
	barcode_frame['fraction_total'] = barcode_frame['total_counts']/barcode_frame['total_counts'].sum()

	plot_colors = sns.hls_palette(12, l=.4, s=.8)

	plot_colors12 = [ ]
	for c in plot_colors:
		for i in range(8):
			plot_colors12.append(c)

	BC_totals = []
	index_list = []
	for i, w in enumerate(wells):
		BC_totals.append(barcode_frame[w].sum())
		index_list.append(i+1)
    
	BC_total_arr = []
	for r in rows:
		subarr = []
		for c in columns:
			subarr.append(barcode_frame[r + str(c)].sum())
		BC_total_arr.append(subarr)

	plt.rcParams["figure.figsize"] = [12,16]
	fig, axs = plt.subplots(2, 1)

	r12 = np.asarray(np.split(np.asarray(BC_totals), 8)).transpose().flatten()

	axs.scatter(index_list, r12, c=plot_colors12, s=70);
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
	axs[1].set_yticklabels([ r + " " for r in rows[::-1] ], size=16);
	axs[1].set_yticks([i for i in range(8)]);
	axs[1].set_ylim(-0.5, 7.5);
	axs[1].tick_params(length=0);
	pdf.savefig()
    if not show_plots:
        plt.close(fig)

	RS_names = ref_seq_frame["RS_name"]

	name_list = [""]*len(barcode_frame)
	barcode_frame["RS_name"] = name_list

	for index, row in ref_seq_frame.iterrows():
		display_frame = barcode_frame[barcode_frame["forward_BC"].str.contains(row["forward_lin_tag"])]
		display_frame = display_frame[display_frame["reverse_BC"].str.contains(row["reverse_lin_tag"])]
		display_frame = display_frame[["RS_name", "forward_BC", "reverse_BC", "total_counts"]]
		if len(display_frame)>0:
			display_frame["RS_name"].iloc[0] = row["RS_name"]
			#barcode_frame["RS_name"][display_frame.index[0]] = row["RS_name"]
			barcode_frame.loc[display_frame.index[0], "RS_name"] = row["RS_name"]
		#print(row["RS_name"])
		display(display_frame)

	total_reads = barcode_frame["total_counts"].sum()
	print(f"total reads: {total_reads}")
	total_RS_reads = barcode_frame[barcode_frame["RS_name"]!=""]["total_counts"].sum()
	print(f"reference sequence reads: {total_RS_reads}")

	total = []
	for index, row in barcode_frame[wells_by_column[:24]].iterrows():
		counts = 0
		for t in wells_by_column[:24]:
			counts += row[t]
		total.append(counts)
	barcode_frame['total_counts_plate_2'] = total
	barcode_frame['fraction_total_p2'] = barcode_frame['total_counts_plate_2']/barcode_frame['total_counts_plate_2'].sum()

	plt.rcParams["figure.figsize"] = [16,16]
	fig, axs = plt.subplots(2, 2)
	f_data = barcode_frame[barcode_frame["RS_name"]!=""]
	f_x = f_data['fraction_total_p2']

	for ax in axs.flatten()[:2]:
		ax.plot([0,.125], [0,.125], color='k')
	for ax in axs.flatten()[2:4]:
		ax.plot([0,.125], [0,0], color='k')
	for i, w in enumerate(wells_by_column[:24]):
		c = [(plot_colors*8)[i]]*len(f_data)
		for ax in axs.flatten()[:2]:
			ax.scatter(f_x, f_data['fraction_' + w], c=c)
		for ax in axs.flatten()[2:4]:
			ax.scatter(f_x, (f_data['fraction_' + w] - f_x)*100, c=c)
        
	axs.flatten()[1].set_xscale("log");
	axs.flatten()[1].set_yscale("log");
	axs.flatten()[1].set_xlim(0.01, 0.125);
	axs.flatten()[1].set_ylim(0.01, 0.125);

	axs.flatten()[3].set_xscale("log");
	#axs.flatten()[3].set_yscale("log");
	axs.flatten()[3].set_xlim(0.01, 0.125);
	#axs.flatten()[3].set_ylim(0.00007, 0.5);
	fig.suptitle('Fraction from Each Dual Barcode (Plate 2)', fontsize=24, position=(0.5, 0.905))

	for ax in axs.flatten()[:2]:
		ax.set_xlabel('Fraction Total', size=20)
		ax.set_ylabel('Fraction per Sample', size=20);
		ax.tick_params(labelsize=16);
    
	for ax in axs.flatten()[2:4]:
		ax.set_xlabel('Fraction Total', size=20)
		ax.set_ylabel('Fraction per Sample - Fraction Total (%)', size=20);
		ax.tick_params(labelsize=16);
	pdf.savefig()
    if not show_plots:
        plt.close(fig)

	plt.rcParams["figure.figsize"] = [16,6*num_to_plot]
	fig, axs = plt.subplots(num_to_plot, 1)

	f_data = barcode_frame[:num_to_plot]

	for index, row in f_data.iterrows():
		y = []
		x = []
		y_for_scale = []
		for i, t in enumerate(wells_by_column):
			y.append(row["fraction_" + t])
			x.append(i+1)
			if (row["fraction_" + t])>0:
				y_for_scale.append(row["fraction_" + t])

		axs[index].scatter(x, y, c=plot_colors12, s=70);
		#axs[index].set_ylim(min(y) - 0.1*( max(y) - min(y) ), max(y) + 0.17*( max(y) - min(y) ));
		axs[index].set_ylim(0.5*min(y_for_scale), 2*max(y));
		axs[index].set_yscale("log")
		barcode_str = str(index) + ', '
		if row['RS_name'] != "": barcode_str += row['RS_name'] + ", "
		barcode_str += row['forward_BC'] + ', ' + row['reverse_BC']
		axs[index].text(x=0.05, y=0.95, s=barcode_str, horizontalalignment='left', verticalalignment='top',
						transform=axs[index].transAxes, fontsize=14)
    
		for i in range(13):
			axs[index].plot([i*8+0.5, i*8+0.5],[0.6*min(y_for_scale), 1.2*max(y)], color='gray');
	axs[0].set_title("Read Fraction Per Barcode", fontsize=32);
	pdf.savefig()
    if not show_plots:
        plt.close(fig)


	# data from 2019-10-02:
	x_test = [0.23776345382258504, 0.21428834768303265, 0.14955568743012018, 0.10527042635253019, 0.08814193520270863,
			  0.07140559171457407, 0.032268913991628186, 0.02486533840744069, 0.009370452839984682, 0.0021539027931815613,
			  0.0001936817014361814]
	y_test = [0.0019726945744597706, 0.0028398295224567756, 0.0027140121666701543, 0.0016422861817864806,
			  0.0012364410886752844, 0.0014467832918787287, 0.0009412184378809117, 0.0007090217957749182,
			  0.00034552377974558844, 0.00017198555940160456, 4.958998052635534e-05]
	err_test = [0.001391130466104952, 0.001320415964490587, 0.0011032026255463198, 0.0009247685041703838,
				0.0008466282838575875, 0.0007620910541483005, 0.0005123905962175842, 0.000449754496329767,
				0.00027605091052578906, 0.0001323496187650663, 3.929704870026295e-05]
	####

	# data from 2019-10-08:
	x_small = [0.08251274176535274, 0.0962239061597132, 0.08539004578198717, 0.08675701439383578, 0.07400424816228543,
			   0.07566109361860245, 0.0699367739242362, 0.06963680434271374, 0.06384195016208481, 0.06321931248609224,
			   0.06334894239678983, 0.02536420185939611, 0.03923343837910993, 0.020238576239101202]
	y_small = [0.003020200426682457, 0.003374150359051314, 0.00374541788260866, 0.0035764736646941536,
			   0.002598176841078495, 0.003669639858790278, 0.0021759993522437074, 0.002827475646549457,
			   0.0038335541520843315, 0.002201298340428577, 0.008012477386731139, 0.001454772893578839,
			   0.0012788004626381614, 0.0021763030793714206]
	err_small = [0.0008661333092282185, 0.0009340439480853888, 0.0008821889073372234, 0.0008856945951456786,
				 0.000820757229296616, 0.000830315430739499, 0.0007963057526756344, 0.0007963629310250612,
				 0.000763102677224598, 0.0007575749124137182, 0.0007546065015548847, 0.0004797418835729835,
				 0.000596486425619687, 0.00042833165436399073]
	###


	fraction_list = ["fraction_" + w for w in wells_by_column[:24] ]

	plt.rcParams["figure.figsize"] = [8,8]
	fig, axs = plt.subplots(1, 1)
	#fig.suptitle('First Time Point Only (Plate 2)', fontsize=24, position=(0.5, 0.925))

	axs.plot(x_test, y_test, "o", ms=10, label="Library Prep Test, 2019-10-02");
	axs.plot(x_test, err_test, c="gray");

	axs.plot(x_small, y_small, "o", ms=10, label="Small Library Selection, 2019-10-08");

	f_data = barcode_frame[barcode_frame["total_counts"]>500]

	y = [ f_data[fraction_list].iloc[i].std() for i in range(len(f_data)) ]
	x = [ f_data[fraction_list].iloc[i].mean() for i in range(len(f_data)) ]
	err_est = [ ( f_data[fraction_list].iloc[i].mean() ) / ( np.sqrt( f_data[wells_by_column[:24]].iloc[i].mean() ) ) for i in range(len(f_data)) ]

	axs.plot(x, y, "o", ms=5, label = experiment);
	axs.plot(x, err_est, c="darkgreen");
	axs.set_xscale("log");
	axs.set_yscale("log");
	axs.set_xlabel('Mean(barcode fraction per sample)', size=20)
	axs.set_ylabel('Stdev(barcode fraction per sample)', size=20);
	axs.tick_params(labelsize=16);

	leg = axs.legend(loc='upper left', bbox_to_anchor= (0.025, 0.93), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
	leg.get_frame().set_edgecolor('k');
	pdf.savefig()
    if not show_plots:
        plt.close(fig)

	pdf.close()

	return barcode_frame


def fit_and_plot_barcode_fitness(barcode_frame, notebook_dir, show_plots=False, inducer_conc_list=None, max_fits=None):
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    
    pdf_file = 'barcode fitness plots.pdf'
    pdf = PdfPages(pdf_file)
        
	data_directory = notebook_dir + "\\barcode_analysis"
	os.chdir(data_directory)

	ref_index_b = barcode_frame[barcode_frame["RS_name"]=="AO-B"].index[0]
	ref_index_e = barcode_frame[barcode_frame["RS_name"]=="AO-E"].index[0]

	if inducer_conc_list is None:
		inducer_conc_list = [0, 2]
		for i in range(10):
			inducer_conc_list.append(2*inducer_conc_list[-1])

	inducer_conc_list_in_plate = np.asarray(np.split(np.asarray(inducer_conc_list),4)).transpose().flatten().tolist()*8
	inducer_conc_list_in_plate = np.asarray([(inducer_conc_list[j::4]*4)*2 for j in range(4)]*1).flatten()
	
	with_tet = []
	plate_list = []
	rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
	columns = [i for i in range(1,13)]
	for r in rows:
		for c in columns:
			plate_list.append( int(2+(c-1)/3) )
			with_tet.append(r in rows[1::2])

	sample_plate_map = pd.DataFrame({"well": wells})
	sample_plate_map['with_tet'] = with_tet
	sample_plate_map['IPTG'] = inducer_conc_list_in_plate
	sample_plate_map['growth_plate'] = plate_list
	sample_plate_map.set_index('well', inplace=True, drop=False)

	wells_with_tet = []
	wells_without_tet = []

	for i in range(2,6):
		df = sample_plate_map[(sample_plate_map["with_tet"]) & (sample_plate_map["growth_plate"]==i)]
		df = df.sort_values(['IPTG'])
		wells_with_tet.append(df["well"].values)
		df = sample_plate_map[(sample_plate_map["with_tet"] != True) & (sample_plate_map["growth_plate"]==i)]
		df = df.sort_values(['IPTG'])
		wells_without_tet.append(df["well"].values)

	RS_count_frame = barcode_frame[barcode_frame["RS_name"]!=""]
	barcode_plot_frame = barcode_frame.iloc[:num_to_plot]
    barcode_plot_frame = pd.concat([barcode_plot_frame, RS_count_frame[2:]])

	for i in range(2,6):
		counts_0 = []
		counts_tet = []
		for index, row in barcode_frame.iterrows():
			row_0 = row[wells_without_tet[i-2]]
			counts_0.append(row_0.values)
			row_tet = row[wells_with_tet[i-2]]
			counts_tet.append(row_tet.values)
		barcode_frame["read_count_0_" + str(i)] = counts_0
		barcode_frame["read_count_tet_" + str(i)] = counts_tet

	spike_in_fitness_0 = np.array([0.97280301, 0.97280301, 0.97280301, 0.97280301, 0.97280301,
		                           0.97280301, 0.97280301, 0.97280301, 0.97280301, 0.97280301,
		                           0.97280301, 0.97280301])
	spike_in_fitness_tet = np.array([0.93202979, 0.93202979, 0.93202979, 0.93202979, 0.93202979,
		                             0.93202979, 0.93202979, 0.93202979, 0.93202979, 0.93202979,
		                             0.93202979, 0.93202979])

	spike_in_row_b = barcode_frame[ref_index_b:ref_index_b+1]
	spike_in_row_e = barcode_frame[ref_index_e:ref_index_e+1]

	#Fit to barcode log(ratios) over time to get slopes = fitness
	x = [1, 2, 3, 4]
	f_tet_est_list = []
	f_0_est_list = []
	f_tet_err_list = []
	f_0_err_list = []

	spike_in_reads_0_b = [ spike_in_row_b[f'read_count_0_{plate_num}'].values[0] for plate_num in range(2,6) ]
	spike_in_reads_tet_b = [ spike_in_row_b[f'read_count_tet_{plate_num}'].values[0] for plate_num in range(2,6) ]
	spike_in_reads_0_e = [ spike_in_row_e[f'read_count_0_{plate_num}'].values[0] for plate_num in range(2,6) ]
	spike_in_reads_tet_e = [ spike_in_row_e[f'read_count_tet_{plate_num}'].values[0] for plate_num in range(2,6) ]

	x0 = [2, 3, 4, 5]

	if max_fits is None:
	    fit_frame = barcode_frame
	else:
	    fit_frame = barcode_frame[:max_fits]

	for index, row in fit_frame.iterrows(): # iterate over barcodes
		slopes_b = []
		errors_b = []
		n_reads = [ row[f'read_count_0_{plate_num}'] for plate_num in range(2,6) ]
    
		for j in range(len(n_reads[0])): # iteration over IPTG concentrations 0-11
			x = []
			y = []
			s = []
			for i in range(len(n_reads)): # iteration over time points 0-3
				if (n_reads[i][j]>0 and spike_in_reads_0[i][j]>0):
					x.append(x0[i])
					y.append(np.log(n_reads[i][j]) - np.log(spike_in_reads_0[i][j]))
					s.append(np.sqrt(1/n_reads[i][j] + 1/spike_in_reads_0[i][j]))
			if len(x)>1:
				popt, pcov = curve_fit(line_funct, x, y, sigma=s, absolute_sigma=True)
				slopes_b.append(popt[0])
				errors_b.append(np.sqrt(pcov[0,0]))
			else:
				slopes_b.append(np.nan)
				errors_b.append(np.nan)
        
			if j==0:
				if len(x)>1:
					slope_0 = popt[0]
				else:
					slope_0 = 0
        
		slopes_b = np.asarray(slopes_b)
		errors_b = np.asarray(errors_b)
		f_0_est = spike_in_fitness_0 + slopes_b/np.log(10)
		f_0_err = errors_b/np.log(10)
    
		slopes_b = []
		errors_b = []
		n_reads = [ row[f'read_count_tet_{plate_num}'] for plate_num in range(2,6) ]
    
		for j in range(len(n_reads[0])): # iteration over IPTG concentrations 0-11
			x = []
			y = []
			s = []
			for i in range(len(n_reads)): # iteration over time points 0-3
				if (n_reads[i][j]>0 and spike_in_reads_tet[i][j]>0):
					x.append(x0[i])
					y.append(np.log(n_reads[i][j]) - np.log(spike_in_reads_tet[i][j]))
					s.append(np.sqrt(1/n_reads[i][j] + 1/spike_in_reads_tet[i][j]))
			if len(x)>1:
				def fit_funct(xp, mp, bp): return bi_linear_funct(xp, mp, bp, slope_0, alpha=np.log(5))
            
				popt, pcov = curve_fit(fit_funct, x, y, sigma=s, absolute_sigma=True)
				slopes_b.append(popt[0])
				errors_b.append(np.sqrt(pcov[0,0]))
			else:
				slopes_b.append(np.nan)
				errors_b.append(np.nan)
        
		slopes_b = np.asarray(slopes_b)
		errors_b = np.asarray(errors_b)
		f_tet_est = spike_in_fitness_tet + slopes_b/np.log(10)
		f_tet_err = errors_b/np.log(10)
        
		f_tet_est_list.append(f_tet_est)
		f_0_est_list.append(f_0_est)
		f_tet_err_list.append(f_tet_err)
		f_0_err_list.append(f_0_err)
    
	barcode_frame['fitness_tet_estimate'] = f_tet_est_list
	barcode_frame['fitness_0_estimate'] = f_0_est_list
	barcode_frame['fitness_tet_err'] = f_tet_err_list
	barcode_frame['fitness_0_err'] = f_0_err_list
	
	#plot fitness curves
	plt.rcParams["figure.figsize"] = [12,8*(len(ref_count_frame))]
	fig, axs = plt.subplots(len(ref_count_frame), 1)
	x = inducer_conc_list
	linthreshx = min([i for i in inducer_conc_list if i>0])
	for index, row in ref_count_frame.iterrows(): # iterate over reference sequences
		y = row["fitness_0_estimate"]
		s = row["fitness_0_err"]
		axs[index].errorbar(x, y, s, marker='o', ms=10)
		y = row["fitness_tet_estimate"]
		s = row["fitness_tet_err"]
		axs[index].errorbar(x, y, s, marker='^', ms=10)
    
		barcode_str = str(index) + ', '
		barcode_str += str(row['total_counts']) + ", "
		barcode_str += row['RS_name'] + ": "
		barcode_str += row['forward_BC'] + ", "
		barcode_str += row['reverse_BC']
		axs[index].text(x=0.0, y=1.03, s=barcode_str, horizontalalignment='left', verticalalignment='top',
						transform=axs[index].transAxes, fontsize=12)
		axs[index].set_xscale('symlog', linthreshx=linthreshx)
		axs[index].set_xlim(-linthreshx/10, 2*max(x));
		axs[index].set_xlabel('[IPTG] (umol/L)', size=20)
		axs[index].set_ylabel('Fitness (log(10)/plate)', size=20)
		axs[index].tick_params(labelsize=16);
	pdf.savefig()
    if not show_plots:
        plt.close(fig)

	pdf.close()

	os.chdir(data_directory)
	pickle_file = experiment + '_counts_and_fitness.df_pkl'
	with open(pickle_file, 'wb') as f:
		pickle.dump(ref_count_frame, f)

	pickle_file = experiment + '_iptg_list.pkl'
	with open(pickle_file, 'wb') as f:
		pickle.dump(iptg_list, f)

	return barcode_frame

        
def exp_funct(x, background, A, doubling_time):
    return background + A*(2**(x/doubling_time) )

def line_funct(x, m, b):
    return m*x + b

def bi_linear_funct(z, m2, b, m1, alpha):
    return b + m2*z + ( m1 - m2 + (m2-m1)*np.exp(-z*alpha) )/alpha
        