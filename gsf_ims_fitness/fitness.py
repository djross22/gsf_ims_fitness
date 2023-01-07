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
from scipy.interpolate import interpn
#from scipy import special
#from scipy import misc

#import pystan
import pickle

import seaborn as sns
sns.set()
import palettable
import cmocean

#from IPython.display import display

#import ipywidgets as widgets
#from ipywidgets import interact#, interact_manual

def get_sample_plate_map(inducer, inducer_conc_list, tet_conc_list, inducer_2=None, inducer_conc_list_2=None):

    """
    This method returns a dataframe that has the growth conditions for each well in the BarSeq output plate.
    Each row is a well in the output plate.
    The columns indicate which time point ('growth_plate'), whether or not the sample included antibiotic ('with_tet'),
        and the concentration(s) of inducer(s) (inducer, inducer_2)
        For plate layouts with >1 antibiotic concentration, there is also a column with the antibiotic concentration ('antibiotic_conc') 
    
    Parameters
    ----------
    inducer : string
        Identifier for the inducer used in the experiment
        
    inducer_2 : string
        Identifier for the second inducer used in the experiment
        
    inducer_conc_list : list or array of float
        A list of inducer concentrations used in the experiment
        
    inducer_conc_list_2 : list or array of float
        A list of inducer_2 concentrations used in the experiment
        
    tet_conc_list : list or array of float
        A list of antibiotic concentrations used in the experiment, including zero

    Returns
    -------
    sample_plate_map : Pandas.DataFrame
        A dataframe with the growth conditions for each well in the BarSeq output plate
    """
    
    if inducer_2 is None:
        # This handles the case for the original plate layout, with 12 inducer concentrations, each measured with and without antibiotic
        inducer_conc_list_in_plate = np.asarray(np.split(np.asarray(inducer_conc_list),4)).transpose().flatten().tolist()*8
        inducer_conc_list_in_plate = np.asarray([(inducer_conc_list[j::4]*4)*2 for j in range(4)]*1).flatten()
        
        ligand_list = [inducer if x>0 else 'none' for x in inducer_conc_list_in_plate]
        
        layout_dict = {}
        for zip_tup in zip(['A', 'C', 'E', 'G', 'A', 'C', 'E', 'G', 'A', 'C', 'E', 'G'],
                           ['B', 'D', 'F', 'H', 'B', 'D', 'F', 'H', 'B', 'D', 'F', 'H'],
                           [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                           [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]):
            r1, r2, c, s1, s2 = zip_tup
            for y in [0, 3, 6, 9]:
                w1 = f"{r1}{c + y}"
                w2 = f"{r2}{c + y}"
                layout_dict[w1] = s1
                layout_dict[w2] = s2
        
        with_tet = []
        plate_list = []
        well_list = []
        sample_id = []
        for r in rows():
            for c in columns():
                w = r + str(c)
                plate_list.append( int(2+(c-1)/3) )
                with_tet.append(r in rows()[1::2])
                well_list.append(f"{r}{c}")
                sample_id.append(layout_dict[w])
        
        antibiotic_conc = [tet_conc_list[-1] if x else 0 for x in with_tet]
    else:
        # This handles the case for the plate layout with 2 inducers and 2 non-zero antibiotic concentrations
        inducer_conc_list.sort()
        inducer_conc_list_2.sort()
        
        zero_tet_inducer_conc = max(inducer_conc_list)/5
        zero_tet_inducer_conc_2 = max(inducer_conc_list_2)/5
        layout_dict = {}
        for zip_tup in zip(['A', 'C', 'E', 'G', 'A'], ['G', 'E', 'C', 'A', 'G'],
                           ['B', 'D', 'F', 'H', 'B'], ['H', 'F', 'D', 'B', 'H'],
                           [1, 1, 1, 1, 2], [3, 3, 3, 3, 2],
                           inducer_conc_list[::-1], inducer_conc_list_2[::-1],
                           [1, 2, 3, 4, 5], [12, 11, 10, 9, 8], [13, 14, 15, 16, 17], [24, 23, 22, 21, 20]):
            r1, r2, r3, r4, c1, c2, x1, x2, s1, s2, s3, s4 = zip_tup
            for y in [0, 3, 6, 9]:
                w1 = f"{r1}{c1 + y}"
                w2 = f"{r2}{c2 + y}"
                w3 = f"{r3}{c1 + y}"
                w4 = f"{r4}{c2 + y}"
                layout_dict[w1] = [x1, inducer, tet_conc_list[1], s1]
                layout_dict[w2] = [x1, inducer, tet_conc_list[2], s2]
                layout_dict[w3] = [x2, inducer_2, tet_conc_list[1], s3]
                layout_dict[w4] = [x2, inducer_2, tet_conc_list[2], s4]
        for y in [0, 3, 6, 9]:
            w = f"C{2 + y}"
            layout_dict[w] = [0, 'none', 0, 6]
            
            w = f"D{2 + y}"
            layout_dict[w] = [0, 'none', tet_conc_list[1], 18]
            
            w = f"E{2 + y}"
            layout_dict[w] = [zero_tet_inducer_conc, inducer, 0, 7]
            
            w = f"F{2 + y}"
            layout_dict[w] = [zero_tet_inducer_conc_2, inducer_2, 0, 19]
        
        with_tet = []
        plate_list = []
        well_list = []
        inducer_conc_list_in_plate = []
        inducer_2_conc_list_in_plate = []
        antibiotic_conc = []
        sample_id = []
        for r in rows():
            for c in columns():
                w = f"{r}{c}"
                plate_list.append( int(2+(c-1)/3) )
                v = layout_dict[w]
                with_tet.append(v[2]>0)
                antibiotic_conc.append(v[2])
                well_list.append(w)
                sample_id.append(v[3])
                
                if v[1] == inducer:
                    inducer_conc_list_in_plate.append(v[0])
                else:
                    inducer_conc_list_in_plate.append(0)
                
                if v[1] == inducer_2:
                    inducer_2_conc_list_in_plate.append(v[0])
                else:
                    inducer_2_conc_list_in_plate.append(0)
        
        ligand_list = [inducer if x>0 else inducer_2 if y>0 else 'none' for x, y in zip(inducer_conc_list_in_plate, inducer_2_conc_list_in_plate)]

    sample_plate_map = pd.DataFrame({"well": well_list}, dtype='string')
    sample_plate_map['sample_id'] = sample_id
    sample_plate_map['ligand'] = ligand_list
    
    sample_plate_map['with_tet'] = with_tet
    sample_plate_map['antibiotic_conc'] = antibiotic_conc
        
    sample_plate_map[inducer] = inducer_conc_list_in_plate
    if inducer_2 is not None:
        sample_plate_map[inducer_2] = inducer_2_conc_list_in_plate
        
    sample_plate_map['growth_plate'] = plate_list
    sample_plate_map.set_index('well', inplace=True, drop=False)
    
    return sample_plate_map

def bar_seq_threshold_plot(notebook_dir,
                           experiment=None,
                           save_plots=False,
                           cutoff=None,
                           hist_bin_max=None,
                           num_bins=50,
                           barcode_file=None):
    
    # Turn interactive plotting on or off depending on show_plots
    plt.ion()
    
    if save_plots:
        pdf_file = 'barcode histogram plot.pdf'
        pdf = PdfPages(pdf_file)
        
    #os.chdir(notebook_dir)
    
    if experiment is None:
        experiment = get_exp_id(notebook_dir)
    
    print(f"Importing BarSeq count data and plotting histogram for thresholding for experiment: {experiment}")

    data_directory = notebook_dir + "\\barcode_analysis"
    os.chdir(data_directory)

    if barcode_file is None:
        barcode_file = glob.glob("*.sorted_counts.csv")[0]
    print(f"Importing BarSeq count data from file: {barcode_file}")
    barcode_frame_0 = pd.read_csv(barcode_file, skipinitialspace=True)

    barcode_frame_0.sort_values('total_counts', ascending=False, inplace=True)
    barcode_frame_0.reset_index(drop=True, inplace=True)
    
    if hist_bin_max is None:
        hist_bin_max = barcode_frame_0[int(len(barcode_frame_0)/50):int(len(barcode_frame_0)/50)+1]["total_counts"].values[0]

    #Allow user to replot with different hist_bin_max
    interact_hist = interact.options(manual=True, manual_name="(re)plot histogram")
    @interact_hist()
    def plot_histogram(hist_max=str(hist_bin_max)):
        #Plot histogram of Barcode counts to enable decision about threshold
        
        plt.rcParams["figure.figsize"] = [16,8]
        fig, axs = plt.subplots(1, 2)
        try:
            hist_bin_max = float(hist_max)
            bins = np.linspace(-0.5, hist_bin_max + 0.5, num_bins)
            for ax in axs.flatten():
                ax.hist(barcode_frame_0['total_counts'], bins=bins);
                ax.set_xlabel('Barcode Count', size=20)
                ax.set_ylabel('Number of Barcodes', size=20)
                ax.tick_params(labelsize=16);
            axs[0].hist(barcode_frame_0['total_counts'], bins=bins, histtype='step', cumulative=-1);
            axs[0].set_yscale('log');
            axs[1].set_yscale('log');
            axs[1].set_xlim(0,hist_bin_max/3);
        except:
            print("hist_max needs to be a number")
        
    if save_plots:
        pdf.savefig()
            
    if save_plots:
        pdf.close()
        
    return barcode_frame_0


def total_plate_2_and_plot_bar_seq_quality(barcode_frame,
                                           notebook_dir,
                                           experiment=None,
                                           show_plots=True,
                                           save_plots=False,
                                           cutoff=None,
                                           num_to_plot=None,
                                           export_trimmed_file=False,
                                           trimmed_export_file=None):
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    
    if save_plots:
        pdf_file = 'barcode quality plots.pdf'
        pdf = PdfPages(pdf_file)
        
    ref_seq_file = "reference_sequences.csv"
    ref_seq_file_found = False
    top_directory = notebook_dir
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

    data_directory = notebook_dir + "\\barcode_analysis"
    os.chdir(data_directory)
    
    if experiment is None:
        experiment = get_exp_id(notebook_dir)

    if cutoff is None:
        hist_bin_max = barcode_frame[int(len(barcode_frame)/50):int(len(barcode_frame)/50)+1]["total_counts"].values[0]
        cutoff = int(hist_bin_max/10)
    print(f"Barcode frequency cutoff: {cutoff}")

    #drop_list = list(barcode_frame[barcode_frame["total_counts"]<cutoff].index)
    #barcode_frame.drop(drop_list, inplace=True)
    
    barcode_frame = barcode_frame[barcode_frame["total_counts"]>cutoff].copy()
    barcode_frame.reset_index(drop=True, inplace=True)
    if export_trimmed_file:
        if trimmed_export_file is None:
            trimmed_export_file = f"{experiment}.trimmed_sorted_counts.csv"
        print(f"Exporting trimmed barcode counts data to: {trimmed_export_file}")
        barcode_frame.to_csv(trimmed_export_file)

    for w in wells():
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
    for i, w in enumerate(wells()):
        BC_totals.append(barcode_frame[w].sum())
        index_list.append(i+1)
    
    BC_total_arr = []
    for r in rows():
        subarr = []
        for c in columns():
            subarr.append(barcode_frame[r + str(c)].sum())
        BC_total_arr.append(subarr)

    #Plot barcode read counts across plate
    plt.rcParams["figure.figsize"] = [12,16]
    fig, axs = plt.subplots(2, 1)

    r12 = np.asarray(np.split(np.asarray(BC_totals), 8)).transpose().flatten()

    axs[0].scatter(index_list, r12, c=plot_colors12, s=70);
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
    axs[1].set_yticklabels([ r + " " for r in rows()[::-1] ], size=16);
    axs[1].set_yticks([i for i in range(8)]);
    axs[1].set_ylim(-0.5, 7.5);
    axs[1].tick_params(length=0);
    if save_plots:
        pdf.savefig()
    if not show_plots:
        plt.close(fig)
    
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
            display(display_frame)
    
        total_reads = barcode_frame["total_counts"].sum()
        print(f"total reads: {total_reads}")
        total_RS_reads = barcode_frame[barcode_frame["RS_name"]!=""]["total_counts"].sum()
        print(f"reference sequence reads: {total_RS_reads}")

    total = []
    for index, row in barcode_frame[wells_by_column()[:24]].iterrows():
        counts = 0
        for t in wells_by_column()[:24]:
            counts += row[t]
        total.append(counts)
    barcode_frame['total_counts_plate_2'] = total
    barcode_frame['fraction_total_p2'] = barcode_frame['total_counts_plate_2']/barcode_frame['total_counts_plate_2'].sum()

    #Plot Barcode fraction for each well in time point 1 vs. mean fraction in time point 1
    plt.rcParams["figure.figsize"] = [16,16]
    fig, axs = plt.subplots(2, 2)
    if num_to_plot is None:
        f_data = barcode_frame
    else:
        f_data = barcode_frame[:num_to_plot]
        
    f_x = f_data['fraction_total_p2']
    for ax in axs.flatten()[:2]:
        ax.plot([0,.125], [0,.125], color='k')
    for ax in axs.flatten()[2:4]:
        ax.plot([0,.125], [0,0], color='k')
    for i, w in enumerate(wells_by_column()[:24]):
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
    axs.flatten()[3].set_xlim(0.01, 0.125);
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
    if not show_plots:
        plt.close(fig)
        
    return barcode_frame


def plot_bar_seq_read_fractions(barcode_frame,
                                notebook_dir,
                                experiment=None,
                                show_plots=True,
                                save_plots=False,
                                num_to_plot=None):

    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    
    if save_plots:
        pdf_file = 'barcode read fraction plots.pdf'
        pdf = PdfPages(pdf_file)

    data_directory = notebook_dir + "\\barcode_analysis"
    os.chdir(data_directory)
    
    if experiment is None:
        experiment = get_exp_id(notebook_dir)
    
    #Plot read fraction across all samples for first several barcodes
    plt.rcParams["figure.figsize"] = [16,6*num_to_plot]
    fig, axs = plt.subplots(num_to_plot, 1)

    f_data = barcode_frame[:num_to_plot]
            
    plot_colors = sns.hls_palette(12, l=.4, s=.8)

    plot_colors12 = [ ]
    for c in plot_colors:
        for i in range(8):
            plot_colors12.append(c)

    for index, row in f_data.iterrows():
        y = []
        x = []
        y_for_scale = []
        for i, t in enumerate(wells_by_column()):
            y.append(row["fraction_" + t])
            x.append(i+1)
            if (row["fraction_" + t])>0:
                y_for_scale.append(row["fraction_" + t])

        axs[index].scatter(x, y, c=plot_colors12, s=70);
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
    if save_plots:
        pdf.savefig()
    if not show_plots:
        plt.close(fig)

    if save_plots:
        pdf.close()


def plot_bar_seq_stdev(barcode_frame,
                       notebook_dir,
                       experiment=None,
                       show_plots=True,
                       save_plots=False,
                       count_cutoff=500):

    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    
    if save_plots:
        pdf_file = 'barcode read standard deviation plot.pdf'
        pdf = PdfPages(pdf_file)

    data_directory = notebook_dir + "\\barcode_analysis"
    os.chdir(data_directory)
    
    if experiment is None:
        experiment = get_exp_id(notebook_dir)

    #Plot standard deviation of barcode read fractions (across wells in time point 1) vs mean read fraction 
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


    fraction_list = ["fraction_" + w for w in wells_by_column()[:24] ]

    plt.rcParams["figure.figsize"] = [16,8]
    fig, axs = plt.subplots(1, 2)
    #fig.suptitle('First Time Point Only (Plate 2)', fontsize=24, position=(0.5, 0.925))

    axs[0].plot(x_test, y_test, "o", ms=10, label="Library Prep Test, 2019-10-02");
    axs[0].plot(x_test, poisson_err_test, c="gray");
    axs[0].plot(x_small, y_small, "o", ms=10, label="Small Library Selection, 2019-10-08");
    axs[1].plot(x_test, y_test/x_test, "o", ms=10, label="Library Prep Test, 2019-10-02");
    axs[1].plot(x_test, poisson_err_test/x_test, c="gray");
    axs[1].plot(x_small, y_small/x_small, "o", ms=10, label="Small Library Selection, 2019-10-08");

    f_data = barcode_frame[barcode_frame["total_counts"]>count_cutoff]

    y = np.asarray([ f_data[fraction_list].iloc[i].std() for i in range(len(f_data)) ])
    x = np.asarray([ f_data[fraction_list].iloc[i].mean() for i in range(len(f_data)) ])
    err_est = np.asarray([ ( f_data[fraction_list].iloc[i].mean() ) / ( np.sqrt( f_data[wells_by_column()[:24]].iloc[i].mean() ) ) for i in range(len(f_data)) ])

    axs[0].plot(x, y, "o", ms=5, label = experiment);
    axs[0].plot(x, err_est, c="darkgreen");
    axs[0].set_ylabel('Stdev(barcode fraction per sample)', size=20);
    axs[0].plot(x, y/x, "o", ms=5, label = experiment);
    axs[0].plot(x, err_est/x, c="darkgreen");
    axs[0].set_ylabel('Relative Stdev(barcode fraction per sample)', size=20);

    for ax in axs.flatten():
        ax.set_xlabel('Mean(barcode fraction per sample)', size=20);
        ax.tick_params(labelsize=16);
        ax.set_xscale("log");
        ax.set_yscale("log");
        leg = ax.legend(loc='upper left', bbox_to_anchor= (0.025, 0.93), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
        leg.get_frame().set_edgecolor('k');
    if save_plots:
        pdf.savefig()
    if not show_plots:
        plt.close(fig)

    if save_plots:
        pdf.close()

def fit_barcode_fitness(barcode_frame,
                        notebook_dir,
                        experiment=None,
                        inducer_conc_list=None,
                        max_fits=None):
    
    if max_fits is not None:
        barcode_frame = barcode_frame.iloc[:max_fits]
        
    if experiment is None:
        experiment = get_exp_id(notebook_dir)
        
    print(f"Fitting to log(barcode ratios) to find fitness for each barcode in {experiment}")
        
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
    for r in rows():
        for c in columns():
            plate_list.append( int(2+(c-1)/3) )
            with_tet.append(r in rows()[1::2])

    sample_plate_map = pd.DataFrame({"well": wells()})
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

    spike_in_fitness_0 = {"AO-B": np.array([0.9637]*12), "AO-E": np.array([0.9666]*12)}
    spike_in_fitness_tet = {"AO-B": np.array([0.8972]*12), "AO-E": np.array([0.8757]*12)}

    spike_in_row = {"AO-B": barcode_frame[ref_index_b:ref_index_b+1], "AO-E": barcode_frame[ref_index_e:ref_index_e+1]}

    #Fit to barcode log(ratios) over time to get slopes = fitness
    #Run for both AO-B and AO-E
    for spike_in, initial in zip(["AO-B", "AO-E"], ["b", "e"]):
        f_tet_est_list = []
        f_0_est_list = []
        f_tet_err_list = []
        f_0_err_list = []
    
        spike_in_reads_0 = [ spike_in_row[spike_in][f'read_count_0_{plate_num}'].values[0] for plate_num in range(2,6) ]
        spike_in_reads_tet = [ spike_in_row[spike_in][f'read_count_tet_{plate_num}'].values[0] for plate_num in range(2,6) ]
    
        x0 = [2, 3, 4, 5]
    
        if max_fits is None:
            fit_frame = barcode_frame
        else:
            fit_frame = barcode_frame[:max_fits]
    
        for index, row in fit_frame.iterrows(): # iterate over barcodes
            slopes = []
            errors = []
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
                    def fit_funct(xp, mp, bp): return bi_linear_funct(xp-2, mp, bp, slope_0, alpha=np.log(5))
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
        
        barcode_frame[f'fitness_tet_estimate_{initial}'] = f_tet_est_list
        barcode_frame[f'fitness_0_estimate_{initial}'] = f_0_est_list
        barcode_frame[f'fitness_tet_err_{initial}'] = f_tet_err_list
        barcode_frame[f'fitness_0_err_{initial}'] = f_0_err_list
    

    os.chdir(data_directory)
    pickle_file = experiment + '_counts_and_fitness.df_pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(barcode_frame, f)

    pickle_file = experiment + '_inducer_conc_list.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(inducer_conc_list, f)
        
    return barcode_frame

def plot_barcode_fitness(barcode_frame,
                         notebook_dir,
                         experiment=None,
                         show_plots=True,
                         save_plots=False,
                         inducer_conc_list=None,
                         plot_range=None,
                         inducer="IPTG"):
    
    if plot_range is not None:
        barcode_frame = barcode_frame.iloc[plot_range[0]:plot_range[1]]
        
    if experiment is None:
        experiment = get_exp_id(notebook_dir)
    
    if inducer_conc_list is None:
        inducer_conc_list = [0, 2]
        for i in range(10):
            inducer_conc_list.append(2*inducer_conc_list[-1])
        
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    
    if save_plots:
        pdf_file = 'barcode fitness plots.pdf'
        pdf = PdfPages(pdf_file)
        
    data_directory = notebook_dir + "\\barcode_analysis"
    os.chdir(data_directory)
    
    
    #plot fitness curves
    plt.rcParams["figure.figsize"] = [12,8*(len(barcode_frame))]
    fig, axs = plt.subplots(len(barcode_frame), 1)
    x = inducer_conc_list
    linthreshx = min([i for i in inducer_conc_list if i>0])
    
    plot_colors = sns.color_palette()
    
    for (index, row), ax in zip(barcode_frame.iterrows(), axs): # iterate over barcodes
        for initial in ["b", "e"]:
            y = row[f"fitness_0_estimate_{initial}"]
            s = row[f"fitness_0_err_{initial}"]
            fill_style = "full" if initial=="b" else "none"
            ax.errorbar(x, y, s, marker='o', ms=10, color=plot_colors[0], fillstyle=fill_style)
            y = row[f"fitness_tet_estimate_{initial}"]
            s = row[f"fitness_tet_err_{initial}"]
            ax.errorbar(x, y, s, marker='^', ms=10, color=plot_colors[1], fillstyle=fill_style)
        
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
    if not show_plots:
        plt.close(fig)

    if save_plots:
        pdf.close()

        
def exp_funct(x, background, A, doubling_time):
    return background + A*(2**(x/doubling_time) )

def line_funct(x, m, b):
    return m*x + b

def bi_linear_funct(z, m2, b, m1, alpha):
    return b + m2*z + ( m1 - m2 + (m2-m1)*np.exp(-z*alpha) )/alpha

def get_exp_id(notebook_dir):
    experiment = notebook_dir[notebook_dir.rfind("\\20")+1:]
    find_ind = experiment.find("\\")
    if find_ind != -1:
        experiment = experiment[:find_ind]
    return experiment

def wells():
    w = []
    for r in rows():
        for c in columns():
            w.append(r + str(c))
    return w

def wells_by_column():        
    w = []
    for c in columns():
        for r in rows():
            w.append(r + str(c))
    return w

def rows():
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

def columns():
    return [i for i in range(1,13)]

def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def hamming_distance(SEQ1, SEQ2, MAX = float("inf"), IGNORE_N = False ):
	"""Returns the number of mismatches between two strings.
	MAX sets the number of max number of mismatches that are reported.
	Lowering MAX increases performance.
	IGNORE_N = 1 will ignore mismatches with N."""
	mismatches = 0
	if SEQ1 != SEQ2: #first check for exact match
		if IGNORE_N:
			for i in range(len(SEQ1)):
				if SEQ1[i] != 'N' and SEQ2[i] != 'N':
					if SEQ1[i] != SEQ2[i]:
						mismatches += 1
				if mismatches >= MAX:
					break
			return mismatches
		else:
			for i in range(len(SEQ1)):
				if SEQ1[i] != SEQ2[i]:
					mismatches += 1
				if mismatches >= MAX:
					break
			return mismatches
	else:
		return mismatches
    
def fitness_scale():
    return np.log(10)/165*60
            
def gray_out(color, s_factor=0.5, v_factor=1):
    hsv_color = colors.rgb_to_hsv(colors.to_rgb(color)) * np.array([1, s_factor, v_factor])
    return colors.hsv_to_rgb(hsv_color)

def fitness_calibration_dict(plasmid="pVER"):
    # Dictionary of dictionaries
    #     first key is tet concentration
    #     second key is spike-in name
    spike_in_fitness_dict = {}
    tet_list = [0, 1.25, 10, 20]
    # Fitness for 0, 1.25 and 10 are from 2022-11-22_two-lig_two-sel_OD-test-5-plates, 
    # Fitness for 20 is from 2019 data, rescaled to match older zero-tet from 2022-11-22
    # TODO: move fitness values for spike-ins to somewhere else (not hard coded)
    # old: fitness_dicts = [{"AO-B": 0.9637, "AO-E": 0.9666}, {"AO-B": 0.9587125, "AO-E": 0.9597825}, 
    # old:                  {"AO-B": 0.93045, "AO-E": 0.92115}, {"AO-B": 0.8972, "AO-E": 0.8757}]
                     
    fitness_dicts = [{"AO-B": 0.9288, "AO-E": 0.9282}, {"AO-B": 0.9199, "AO-E": 0.9244}, 
                     {"AO-B": 0.9063, "AO-E": 0.9014}, {"AO-B": 0.8972*0.9288/0.9637, "AO-E": 0.8757*0.9282/0.9666}]
    '''
        pTY1-AO-B, tet: 0.0
            0.9288 +- 0.0026

        pTY1-AO-B, tet: 1.25
            0.9199 +- 0.0036

        pTY1-AO-B, tet: 10.0
            0.9063 +- 0.0055

        pTY1-AO-E, tet: 0.0
            0.9282 +- 0.0045

        pTY1-AO-E, tet: 1.25
            0.9244 +- 0.0030

        pTY1-AO-E, tet: 10.0
            0.9014 +- 0.0040
    '''
    
    for t, d in zip(tet_list, fitness_dicts):
        spike_in_fitness_dict[t] = d
    
    return spike_in_fitness_dict
    

def fit_fitness_difference_params(plasmid="pVER", tet_conc=20, use_geo_mean=False):
    # params are: low_fitness, mid_g, fitness_n, low_fitness_err, mid_g_err, fitness_n_err, 
    if plasmid == "pVER":
        if use_geo_mean:
            if tet_conc==20:
                params = np.array([-0.72246,  13328,  3.2374])
            elif tet_conc==10:
                #params = np.array([-0.8102, 4955, 1.817])
                #params = np.array([-0.818, 4.96e+03, 1.812, 0.009651, 133.5, 0.05171]) # from fit to 5 RSs using barseq data from 2021-12-12_IPTG_Select-DNA-5-plates
                params = np.array([-0.7473, 4.538e+03, 1.849, 0.008271, 93.44, 0.03828]) # from fit to 8 RSs and several other numbered variants using 4-lane barseq data from 2022-11-08_two-lig_two-sel_DNA-5-plates
                [-0.7473, 4.538e+03, 1.849]
            elif tet_conc==1.25:
                params = np.array([-0.7949, 239.3, 0.9777, 0.05126, 35.01, 0.04737]) # from fit to 8 RSs and several other numbered variants using 4-lane barseq data from 2022-11-08_two-lig_two-sel_DNA-5-plates
        else:
            if tet_conc==10:
                # from fit to 8 RSs and several other numbered variants using 4-lane barseq data from 2022-11-08_two-lig_two-sel_DNA-5-plates
                params = np.array([-0.7445, 4.967e+03, 1.891, 0.008442, 104.2, 0.04059]) 
                params = np.array([-0.7400, 4.983e+03, 1.871, 0.00597, 71.95, 0.0277]) 
                params = np.array([-0.7222, 4.76e+03, 1.762, 0.00897, 111.9, 0.04116]) 
            elif tet_conc==1.25:
                # from fit to 8 RSs and several other numbered variants using 4-lane barseq data from 2022-11-08_two-lig_two-sel_DNA-5-plates
                params = np.array([-0.9812, 186.4, 0.9386, 0.1216, 45.1, 0.05396]) 
                params = np.array([-0.9083, 221.8, 0.9936, 0.05355, 27.19, 0.03435]) 
                params = np.array([-0.7681, 311.1, 1.127, 0.03107, 25.54, 0.04042]) 
            elif tet_conc==20:
                params = np.array([-0.72246,  13328,  3.2374]) #place-holder values, for testing
    else:
        params = np.array([-7.41526290e-01,  7.75447318e+02,  2.78019804e+00])
        
    return params

def log_g_limits(plasmid="pVER"):
    if plasmid == "pVER":
        log_g_min = 0
        log_g_max = 4.7
        log_g_prior_scale = 0.15
        wild_type_ginf = 2.44697108e+04
    else:
        log_g_min = 1
        log_g_max = 4.5
        log_g_prior_scale = 0.3
        wild_type_ginf = 1839
    
    return (log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf)

def log_plot_errorbars(log_mu, log_sig):
    mu = np.array(10**log_mu)
    mu_low = np.array(10**(log_mu - log_sig))
    mu_high = np.array(10**(log_mu + log_sig))
    sig_low = mu - mu_low
    sig_high = mu_high - mu
    return np.array([sig_low, sig_high])


def density_scatter_plot(x , y, ax=None, sort=True, bins=50, log_x=True, log_y=False, log_z=True, z_cutoff=None, **kwargs)   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
        
    x_data = x
    y_data = y
    if log_x:
        x_data = np.log10(x)
    if log_y:
        y_data = np.log10(y)
        
    data, x_e, y_e = np.histogram2d( x_data, y_data, bins=bins)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1]) ), data, np.vstack([x_data,y_data]).T,
                method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    if z_cutoff is not None:
        z[z<z_cutoff] = z_cutoff/10

    if log_z:
        #z = np.log(z)
        norm = colors.LogNorm()
    else:
        norm = None
    sc = ax.scatter( x, y, c=z, norm=norm, **kwargs )
    return ax, sc


def density_scatter_cmap():
    # Diverging colormap from darkened sns.color_palette()[0] (blue) to sns.color_palette()[1] (orange),
    #     with yellow/off-white in the middle (taken from palettable.lightbartlein.diverging.BlueDarkOrange12_3)
    color_0 = colors.to_rgb((sns.color_palette()[0]))
    color_1 = colors.to_rgb((sns.color_palette()[1]))

    cmap = palettable.lightbartlein.diverging.BlueDarkOrange12_3.mpl_colormap
    c_arr = np.array(cmap(range(cmap.N)))
    new_c_arr = np.array([ c_arr[0], c_arr[127], c_arr[-20] ])
    for i, c in enumerate(new_c_arr):
        if i<1:
            #c[0] = c[0] * 0
            #c[1] = c[1] * 0
            #c[2] = c[2] * 0.75
            c[0] = color_0[0] * 0.75 #0.75 factor darkens the starting color
            c[1] = color_0[1] * 0.75
            c[2] = color_0[2] * 0.75
        if i>1:
            c[0] = color_1[0]
            c[1] = color_1[1]
            c[2] = color_1[2]
    new_cmap = colors.LinearSegmentedColormap.from_list("test_map", new_c_arr)
    
    return new_cmap

