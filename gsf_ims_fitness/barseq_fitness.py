# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:29:34 2019

@author: Andi Dhroso
Refactored and based from the original implementation from djross
"""

import glob  # filenames and pathnames utility
import os    # operating sytem utility
import sys
import warnings
import datetime
import logging
import traceback
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
import cmocean
import seaborn as sns
sns.set()

from . import fitness_utils
from . import stan_utils
from .state_io import save_state_v1

sns.set_style("white")
sns.set_style("ticks", {'xtick.direction':'in', 'xtick.top':True, 'ytick.direction':'in', 'ytick.right':True})


def load_barseq_data(
    manifest,
    notebook_dir=None,
    experiment=None,
    barcode_file=None,
    antibiotic_conc_list=(0, 20),
    inducer_conc_lists=None,
    ligand_list=("IPTG",),
    antibiotic="tet",
    min_read_count=500,
    ref_samples=None,
    plasmid="pVER",
    get_layout_from_file=False,
    growth_plate_layout_file=None,
    single_barcode=False,
    merge_dist_cutoff=2,
):
    # --- Resolve paths safely ---
    if notebook_dir is None:
        notebook_dir = manifest.get("notebook_dir")
    notebook_dir = Path(notebook_dir)
    manifest["notebook_dir"] = str(notebook_dir)

    if experiment is None:
        experiment = fitness_utils.get_exp_id(str(notebook_dir))
    manifest["experiment"] = experiment

    print(f"Importing BarSeq count data for experiment: {experiment}")

    data_directory = notebook_dir / "barcode_analysis"
    manifest["data_directory"] = str(data_directory)

    # --- Load barcode counts ---
    if barcode_file is None:
        barcode_file = next(data_directory.glob("*.trimmed_sorted_counts.csv"))
    barcode_file = Path(barcode_file)

    print(f"Importing BarSeq count data from file: {barcode_file.name}")
    barcode_frame = pd.read_csv(barcode_file, skipinitialspace=True)

    # --- Load cluster annotations ---
    f_cluster_file = next(data_directory.glob("*forward_merged_cluster.csv"))
    r_cluster_file = next(data_directory.glob("*reverse_merged_cluster.csv"))

    f_bc_cluster_frame = pd.read_csv(f_cluster_file)
    r_bc_cluster_frame = pd.read_csv(r_cluster_file)

    # --- Map cluster IDs + scores ---
    barcode_frame["for_BC_ID"] = barcode_frame["forward_BC"].map(
        dict(zip(f_bc_cluster_frame["Center"], f_bc_cluster_frame["Cluster.ID"]))
    )
    barcode_frame["rev_BC_ID"] = barcode_frame["reverse_BC"].map(
        dict(zip(r_bc_cluster_frame["Center"], r_bc_cluster_frame["Cluster.ID"]))
    )
    barcode_frame["for_BC_Score"] = barcode_frame["forward_BC"].map(
        dict(zip(f_bc_cluster_frame["Center"], f_bc_cluster_frame["Cluster.Score"]))
    )
    barcode_frame["rev_BC_Score"] = barcode_frame["reverse_BC"].map(
        dict(zip(r_bc_cluster_frame["Center"], r_bc_cluster_frame["Cluster.Score"]))
    )

    # --- Optional single-barcode merging (unchanged logic) ---
    if single_barcode:
        barcode_frame = _merge_single_barcodes(
            barcode_frame,
            f_bc_cluster_frame,
            r_bc_cluster_frame,
            experiment,
            merge_dist_cutoff,
            data_directory,
        )

    # --- Filter low-count barcodes ---
    barcode_frame = (
        barcode_frame
        .sort_values("total_counts", ascending=False)
        .loc[barcode_frame.total_counts >= min_read_count]
        .reset_index(drop=True)
    )

    # --- Manifest metadata only ---
    manifest["plasmid"] = plasmid
    manifest["antibiotic"] = antibiotic

    # --- Sample plate layout ---
    if get_layout_from_file:
        if growth_plate_layout_file is None:
            growth_plate_layout_file = find_growth_plate_layout_file(
                manifest, str(data_directory)
            )

        sample_plate_map = set_sample_plate_map(
            manifest,
            barcode_frame,
            auto_save=False,
            growth_plate_layout_file=growth_plate_layout_file,
            plasmid=plasmid,
        )

        manifest["antibiotic_conc_list"] = sorted(
            sample_plate_map.antibiotic_conc.unique().tolist()
        )

        ligands = sorted(
            set(sample_plate_map.ligand) - {"none"}
            if "ligand" in sample_plate_map
            else []
        )
        manifest["ligand_list"] = ligands

        inducer_conc_lists = [
            sorted(sample_plate_map[lig][sample_plate_map[lig] > 0].unique())
            for lig in ligands
        ]
        manifest["inducer_conc_lists"] = inducer_conc_lists

    else:
        manifest["antibiotic_conc_list"] = list(antibiotic_conc_list)
        manifest["ligand_list"] = list(ligand_list)
        manifest["inducer_conc_lists"] = inducer_conc_lists

        set_sample_plate_map(manifest, barcode_frame, auto_save=False)

    set_ref_samples(manifest, ref_samples)

    return barcode_frame

def load_barseq_data(manifest, notebook_dir=None, experiment=None, barcode_file=None, 
                antibiotic_conc_list=[0, 20], 
                inducer_conc_lists=None, 
                ligand_list=["IPTG"],
                antibiotic='tet',
                min_read_count=500,
                ref_samples=None,
                plasmid='pVER', #A designation for the plasmid used for the flow cytometry calibration data
                get_layout_from_file=False,
                growth_plate_layout_file=None,
                single_barcode=False,
                merge_dist_cutoff=2):
                #inducer_2=None, inducer_conc_list_2=None,
    
    # Get notebook_dir from manifest if not provided
    if notebook_dir is None:
        notebook_dir = manifest.get('notebook_dir')
    manifest['notebook_dir'] = notebook_dir
    
    if experiment is None:
        experiment = fitness_utils.get_exp_id(notebook_dir)
    
    manifest['experiment'] = experiment
    
    print(f"Importing BarSeq count data for experiment: {experiment}")
    
    data_directory = notebook_dir + "\\barcode_analysis"
    manifest['data_directory'] = data_directory
    os.chdir(data_directory)
    
    if barcode_file is None:
        barcode_file = glob.glob("*.trimmed_sorted_counts.csv")[0]
    print(f"Importing BarSeq count data from file: {barcode_file}")
    print()
    barcode_frame = pd.read_csv(barcode_file, skipinitialspace=True)
    
    # Add barcode cluster IDs and scores
    f_cluster_file = glob.glob('*forward_merged_cluster.csv')[0]
    r_cluster_file = glob.glob('*reverse_merged_cluster.csv')[0]
    f_bc_cluster_frame = pd.read_csv(f_cluster_file)
    r_bc_cluster_frame = pd.read_csv(r_cluster_file)
    
    for_barcode_clusterID_dict = dict(zip(f_bc_cluster_frame["Center"], f_bc_cluster_frame["Cluster.ID"]))
    rev_barcode_clusterID_dict = dict(zip(r_bc_cluster_frame["Center"], r_bc_cluster_frame["Cluster.ID"]))

    for_barcode_score_dict = dict(zip(f_bc_cluster_frame["Center"], f_bc_cluster_frame["Cluster.Score"]))
    rev_barcode_score_dict = dict(zip(r_bc_cluster_frame["Center"], r_bc_cluster_frame["Cluster.Score"]))
    
    barcode_frame["for_BC_ID"] = [ for_barcode_clusterID_dict[x] for x in barcode_frame["forward_BC"] ]
    barcode_frame["rev_BC_ID"] = [ rev_barcode_clusterID_dict[x] for x in barcode_frame["reverse_BC"] ]

    barcode_frame["for_BC_Score"] = [ for_barcode_score_dict[x] for x in barcode_frame["forward_BC"] ]
    barcode_frame["rev_BC_Score"] = [ rev_barcode_score_dict[x] for x in barcode_frame["reverse_BC"] ]
    
    # if single_barcode, merge barcodes that result from obvious read errors (differences between forward and reverse reads)
    if single_barcode:
        print('Automatically merging barcodes based on comparison between forward and reverse BC reads.')
        print('This could take a several minutes.')
        
        merge_log_file = f'{experiment}_single_barcode_merging.log'
        with open(merge_log_file, 'w') as log_file:
        
            # First, identify possible forward barcodes that should be merged:
            #     the same reverse barcode but different forward barcodes
            rev_id_list = barcode_frame.rev_BC_ID.values
            merge_sets = []
            for rev_id in np.unique(barcode_frame.rev_BC_ID):
                df = barcode_frame
                df = df[rev_id_list==rev_id]
                if len(df)>1:
                    merge_sets.append(df.for_BC_ID.values)
                    
            for_barcode_center_dict = dict(zip(f_bc_cluster_frame["Cluster.ID"], f_bc_cluster_frame["Center"]))
            rev_barcode_center_dict = dict(zip(r_bc_cluster_frame["Cluster.ID"], r_bc_cluster_frame["Center"]))
            for_barcode_count_dict = dict(zip(f_bc_cluster_frame["Cluster.ID"], f_bc_cluster_frame["time_point_1"]))
            rev_barcode_count_dict = dict(zip(r_bc_cluster_frame["Cluster.ID"], r_bc_cluster_frame["time_point_1"]))

            # Next, Create dictionary of forward barcode merges:
            #     key : barcode ID to be merged (and removed)
            #     value : barcode ID that will be merged into (and kept)
            bc_merge_dict = {}
            for x in merge_sets:
                bc_list = [for_barcode_center_dict[y] for y in x]
                for bc, y in zip(bc_list[1:], x[1:]):
                    if len(bc)==len(bc_list[0]):
                        dist = fitness_utils.hamming_distance(bc_list[0], bc)
                        if dist > merge_dist_cutoff:
                            log_file.write(f'    Warning, forward barcodes NOT merged with Hamming distance > cutoff, {dist}:\n')
                            log_file.write(f'        {bc_list[0]}, ID: {x[0]}, count: {for_barcode_count_dict[x[0]]}\n')
                            log_file.write(f'        {bc}, ID: {y}, count: {for_barcode_count_dict[y]}\n')
                        else:
                            bc_merge_dict[y] = x[0]
                            if dist > 1:
                                log_file.write(f'    Warning, merging forward barcodes with Hamming distance {dist}:\n')
                            else:
                                log_file.write(f'    Merging forward barcodes with Hamming distance {dist}:\n')
                            log_file.write(f'        {bc_list[0]}, ID: {x[0]}, count: {for_barcode_count_dict[x[0]]}\n')
                            log_file.write(f'        {bc}, ID: {y}, count: {for_barcode_count_dict[y]}\n')
                    else:
                        dist = fitness_utils.levenshtein_distance(bc_list[0], bc)
                        len_diff = np.abs(len(bc_list[0]) - len(bc))
                        if dist - len_diff > merge_dist_cutoff:
                            log_file.write(f'    Warning, forward barcodes NOT merged with Levenshtein distance > cutoff, {dist}:\n')
                            log_file.write(f'        {bc_list[0]}, ID: {x[0]}, count: {for_barcode_count_dict[x[0]]}\n')
                            log_file.write(f'        {bc}, ID: {y}, count: {for_barcode_count_dict[y]}\n')
                        else:
                            bc_merge_dict[y] = x[0]
                            if dist > 1:
                                log_file.write(f'    Warning, merging forward barcodes with Levenshtein distance {dist}:\n')
                            else:
                                log_file.write(f'    Merging forward barcodes with Levenshtein distance {dist}:\n')
                            log_file.write(f'        {bc_list[0]}, ID: {x[0]}, count: {for_barcode_count_dict[x[0]]}\n')
                            log_file.write(f'        {bc}, ID: {y}, count: {for_barcode_count_dict[y]}\n')
            
            new_bc_id_list = [bc_merge_dict[x] if x in bc_merge_dict else x for x in barcode_frame.for_BC_ID]
            barcode_frame['for_BC_ID'] = new_bc_id_list
            
            log_file.write('\n')
            
            # Finally, merge rows that have the same forward barcode ID
            new_row_list = []
            #rev_merge_list = []
            for bc_id in np.unique(barcode_frame.for_BC_ID):
                df = barcode_frame[barcode_frame.for_BC_ID==bc_id]
                for_bc = df.iloc[0].forward_BC
                if len(df) == 1:
                    new_row = df.iloc[0].copy()
                    new_row['reverse_BC'] = ''
                    new_row_list.append(new_row)
                else:
                    # Default action is to merge all barcodes in df with no warnings
                    #     So, if all the rows in df have the same reverse barcode, they will be merged.
                    merge = [True]*len(df)
                    merge_warning = [False]*len(df)
                    
                    rev_bc_list = df.reverse_BC
                    # If the rows in df have different reverse barcodes, consider merging based on distance between forward and RC of reverse barcodes
                    #     the case for all rows in df having the same reverse barcode was covered in the previous setp (merging forward BCs)
                    
                    # Check rev comp against forw BC
                    rc_rev_bc = [fitness_utils.rev_complement(s) for s in rev_bc_list]
                    if len(np.unique(rev_bc_list)) > 1:
                        merge = []
                        metric_list = []
                        dist_list = []
                        for rc_bc in rc_rev_bc:
                            len_diff = np.abs(len(for_bc) - len(rc_bc))
                            if len_diff == 0:
                                dist = fitness_utils.hamming_distance(for_bc, rc_bc)
                                dist_metric = 'Hamming'
                            else:
                                dist = fitness_utils.levenshtein_distance(for_bc, rc_bc)
                                dist_metric = 'Levenshtein'

                            merge.append(dist - len_diff <= merge_dist_cutoff)
                            metric_list.append(dist_metric)
                            dist_list.append(dist)
                    else:
                        metric_list = ['NA']*len(merge)
                        dist_list = ['NA']*len(merge)
                    
                    
                    if np.any(merge):
                        merge_df = df[merge]
                        new_row = merge_df.iloc[0].copy()
                        for w in fitness_utils.wells():
                            new_row[w] = merge_df[w].sum()
                        new_row['total_counts'] = merge_df['total_counts'].sum()
                        new_row['reverse_BC'] = ''
                        new_row_list.append(new_row)
                        
                        #rev_merge_list.append(list(merge_df.rev_BC_ID))
                        
                    log_file.write(f'    Reverse barcodes considered for merging based on distance of reverse complement to forward BC:\n')
                    log_file.write(f'                   Forward BC: {for_bc}\n')
                    for m, rc_bc, bc, r_bc_id, dist_metric, dist in zip(merge, rc_rev_bc, rev_bc_list, df.rev_BC_ID, metric_list, dist_list):
                        log_str = 'Yes merging' if m else 'NOT merging'
                        log_file.write(f'        {log_str}: Rev-comp: {rc_bc}, BC: {bc}, ID: {r_bc_id}, count: {rev_barcode_count_dict[r_bc_id]}, {dist_metric} distance: {dist}\n')
        
        with open(f'{experiment}_bc_merge_dict.pkl', 'wb') as f:
            pickle.dump(bc_merge_dict, f)
        
        barcode_frame = pd.DataFrame(new_row_list)
        print()   
    
    barcode_frame.sort_values('total_counts', ascending=False, inplace=True)
    barcode_frame = barcode_frame[barcode_frame.total_counts>=min_read_count]
    #barcode_frame.reset_index(drop=True, inplace=True)
    
    # Store barcode_frame in manifest and return it
    manifest['fit_fitness_difference_params'] = None
    manifest['fit_fitness_difference_funct'] = None
    
    manifest['plasmid'] = plasmid
    
    if get_layout_from_file:
        if growth_plate_layout_file is None:
            growth_plate_layout_file = find_growth_plate_layout_file(manifest, data_directory)
        sample_plate_map = set_sample_plate_map(manifest, barcode_frame, auto_save=False, growth_plate_layout_file=growth_plate_layout_file, plasmid=plasmid)
        
        manifest['antibiotic_conc_list'] = list(np.unique(sample_plate_map.antibiotic_conc))
        
        if plasmid == 'Align-Protease':
            lig_id_list = list(np.unique(sample_plate_map['inducer1'])) + list(np.unique(sample_plate_map['inducer2']))
        elif plasmid == 'Align-T7RNAP_1':
            lig_id_list = []
        else:
            lig_id_list = list(np.unique(sample_plate_map['ligand']))
        if 'none' in lig_id_list:
            lig_id_list.remove('none')
        inducer_conc_lists = []
        for lig in lig_id_list:
            sub_list = list(np.unique(sample_plate_map[lig]))
            sub_list.remove(0)
            inducer_conc_lists.append(sub_list)
        manifest['inducer_conc_lists'] = inducer_conc_lists
        manifest['ligand_list'] = lig_id_list
    else:
        manifest['antibiotic_conc_list'] = antibiotic_conc_list
        manifest['antibiotic'] = antibiotic
        
        if inducer_conc_lists is None:
            conc_list = [0, 2]
            for i in range(10):
                conc_list.append(2*conc_list[-1])
            inducer_conc_lists = [conc_list]
        else:
            manifest['inducer_conc_lists'] = inducer_conc_lists
        
        manifest['ligand_list'] = ligand_list
        
        sample_plate_map = set_sample_plate_map(manifest, barcode_frame, auto_save=False)
    
    set_ref_samples(manifest, ref_samples)
    
    return barcode_frame
    
        
def find_growth_plate_layout_file(manifest, data_directory=None):
    notebook_dir = manifest.get('notebook_dir')
    experiment = manifest.get('experiment')
    
    exp_directory = notebook_dir[:notebook_dir.find(experiment)] + experiment
    os.chdir(exp_directory)
    
    ret_file = glob.glob('*growth-plate_5.csv')[0]
    
    ret_file = exp_directory + '\\' + ret_file
    
    return ret_file


def set_ref_samples(manifest, ref_samples):
    if ref_samples is None:
        ref_samples = manifest.get('samples_without_tet')
        
    manifest['ref_samples'] = ref_samples
    
def trim_and_sum_barcodes(
    manifest,
    data,
    cutoff=None,
    export_trimmed_file=False,
    trimmed_export_file=None,
    auto_save=True,
    overwrite=False,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Trim and sum barcode counts, then optionally persist state.

    Parameters
    ----------
    manifest : dict
        Pipeline manifest
    data : pd.DataFrame
        Barcode counts table
    cutoff : int, optional
        Minimum total_counts threshold
    export_trimmed_file : bool
        Whether to export trimmed CSV (legacy behavior)
    trimmed_export_file : str, optional
        CSV export path
    auto_save : bool
        Whether to persist state
    overwrite : bool
        Reserved for future use (kept for compatibility)
    state_dir : str or Path, optional
        Root state directory (required if auto_save=True)
    state_tag : str, optional
        Namespace for outputs (e.g. 'stage_0_ingest')

    Returns
    -------
    barcode_frame : pd.DataFrame
        Processed barcode table
    """

    # ----------------------------
    # Core computation (UNCHANGED)
    # ----------------------------
    barcode_frame = data.copy()

    if cutoff is not None:
        barcode_frame = barcode_frame[barcode_frame["total_counts"] > cutoff].copy()

    print(f"Calculating read fraction for each barcode in each sample")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for w in fitness_utils.wells():
            label = 'fraction_' + w
            barcode_frame[label] = barcode_frame[w] / barcode_frame[w].sum()

        barcode_frame['fraction_total'] = (
            barcode_frame['total_counts'] /
            barcode_frame['total_counts'].sum()
        )

    print(f"Calculating read totals and fractions for each barcode in samples from first time point")
    total = []
    for _, row in barcode_frame[fitness_utils.wells_by_column()[:24]].iterrows():
        counts = 0
        for t in fitness_utils.wells_by_column()[:24]:
            counts += row[t]
        total.append(counts)

    barcode_frame['total_counts_plate_2'] = total
    barcode_frame['fraction_total_p2'] = (
        barcode_frame['total_counts_plate_2'] /
        barcode_frame['total_counts_plate_2'].sum()
    )

    fraction_list = ["fraction_" + w for w in fitness_utils.wells_by_column()[:24]]
    barcode_frame["fraction_p2_std"] = barcode_frame[fraction_list].std(axis=1)

    # ----------------------------
    # Optional legacy CSV export
    # ----------------------------
    if export_trimmed_file:
        if trimmed_export_file is None:
            trimmed_export_file = f"{manifest.get('experiment')}.trimmed_sorted_counts.csv"
        print(f"Exporting trimmed barcode counts data to: {trimmed_export_file}")
        barcode_frame.to_csv(trimmed_export_file, index=False)

    # ----------------------------
    # NEW: State persistence (V1)
    # ----------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError(
                "state_dir must be provided when auto_save=True"
            )

        tag = state_tag or "trim_and_sum_barcodes"
        table_name = f"{tag}_barcode_frame"

        # Record intent in manifest
        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "tables": [table_name],
            "cutoff": cutoff,
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={
                table_name: barcode_frame,
            },
        )

    return barcode_frame


def label_reference_sequences(
    manifest,
    data,
    ref_seq_file_path=None,
    show_output=True,
    auto_save=True,
    overwrite=False,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Label reference sequences in barcode table.

    Parameters
    ----------
    manifest : dict
        Pipeline manifest
    data : pd.DataFrame
        Barcode table
    ref_seq_file_path : str, optional
        Path to reference_sequences.csv
    show_output : bool
        Whether to display matched reference sequences
    auto_save : bool
        Whether to persist state
    overwrite : bool
        Reserved for compatibility
    state_dir : str or Path, optional
        Root state directory (required if auto_save=True)
    state_tag : str, optional
        Namespace for outputs

    Returns
    -------
    barcode_frame : pd.DataFrame
        Updated barcode table with reference labels
    """

    barcode_frame = data.copy()

    # --------------------------------------------------
    # Locate and load reference_sequences.csv (UNCHANGED)
    # --------------------------------------------------
    if ref_seq_file_path is None:
        ref_seq_file = "reference_sequences.csv"
        ref_seq_file_found = False
        top_directory = manifest.get('notebook_dir')

        while not ref_seq_file_found and top_directory:
            find_result = top_directory.rfind("\\")
            if find_result == -1:
                break
            top_directory = top_directory[:find_result]
            ref_seq_file_found = Path(top_directory, ref_seq_file).is_file()

        if ref_seq_file_found:
            ref_seq_frame = pd.read_csv(
                Path(top_directory, ref_seq_file),
                skipinitialspace=True
            )
        else:
            ref_seq_frame = None
    else:
        ref_seq_frame = pd.read_csv(ref_seq_file_path, skipinitialspace=True)

    # --------------------------------------------------
    # Label initialization (UNCHANGED)
    # --------------------------------------------------
    barcode_frame["RS_name"] = [""] * len(barcode_frame)

    plasmid = manifest.get('plasmid')
    if plasmid == 'Align-TF':
        barcode_frame["transcription_factor"] = [""] * len(barcode_frame)

    double_barcodes = (
        ref_seq_frame is not None and
        "reverse_lin_tag" in ref_seq_frame.columns
    )

    if double_barcodes:
        disp_cols = ["RS_name", "forward_BC", "reverse_BC", "rev_BC_ID", "total_counts"]
    else:
        disp_cols = ["RS_name", "forward_BC", "for_BC_ID", "total_counts"]

    if plasmid == 'Align-TF':
        disp_cols = ["transcription_factor"] + disp_cols

    # --------------------------------------------------
    # Reference sequence matching (UNCHANGED)
    # --------------------------------------------------
    if ref_seq_frame is not None:
        no_match_list = []

        for _, row in ref_seq_frame.iterrows():
            display_frame = barcode_frame[
                barcode_frame["forward_BC"].str.contains(row["forward_lin_tag"])
            ]

            if double_barcodes:
                display_frame = display_frame[
                    display_frame["reverse_BC"].str.contains(row["reverse_lin_tag"])
                ]

            display_frame = display_frame[disp_cols]

            if len(display_frame) == 0:
                no_match_list.append(row["RS_name"])

            elif len(display_frame) == 1:
                idx = display_frame.index[0]
                barcode_frame.loc[idx, "RS_name"] = row["RS_name"]

                if plasmid == 'Align-TF':
                    barcode_frame.loc[idx, "transcription_factor"] = row["transcription_factor"]

                if show_output:
                    display(display_frame)

            else:
                print(f"found more than one possible match for {row['RS_name']}")

        print(f"no matches found for:")
        for n in np.unique(no_match_list):
            print(f"    {n}")

        total_reads = barcode_frame["total_counts"].sum()
        total_RS_reads = barcode_frame[barcode_frame["RS_name"] != ""]["total_counts"].sum()
        print(f"total reads: {total_reads}")
        print(
            f"reference sequence reads: "
            f"{total_RS_reads} ({total_RS_reads / total_reads * 100:.2f}%)"
        )

    # --------------------------------------------------
    # NEW: State persistence (V1)
    # --------------------------------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "label_reference_sequences"
        table_name = f"{tag}_barcode_frame"

        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "tables": [table_name],
            "reference_sequences_file": (
                ref_seq_file_path or "reference_sequences.csv"
            ),
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={
                table_name: barcode_frame,
            },
        )

    return barcode_frame
        
def mark_chimera_parents(manifest, data):
    
    barcode_frame = data
    
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
    
    return barcode_frame
        
def flag_possible_chimeras(manifest, data, use_faster_search=True, faster_search_ratio=10):
    
    barcode_frame = data
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
    
    return barcode_frame

def mark_actual_chimeras(
    manifest,
    data,
    chimera_cut_line,
    auto_save=True,
    overwrite=False,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Mark actual chimeras based on cutoff function.

    Parameters
    ----------
    manifest : dict
        Pipeline manifest
    data : pd.DataFrame
        Barcode table
    chimera_cut_line : callable
        Function mapping geo_mean -> cutoff
    auto_save : bool
        Whether to persist state
    overwrite : bool
        Reserved for compatibility
    state_dir : str or Path, optional
        Root state directory (required if auto_save=True)
    state_tag : str, optional
        Namespace for outputs

    Returns
    -------
    barcode_frame : pd.DataFrame
        Barcode table with isChimera column populated
    """

    # ----------------------------
    # Core computation (UNCHANGED)
    # ----------------------------
    barcode_frame = data.copy()

    barcode_frame["isChimera"] = False

    for index, row in barcode_frame[barcode_frame["possibleChimera"]].iterrows():
        geo_mean = row["parent_geo_mean"] / 96
        count = row["total_counts"] / 96
        if count < chimera_cut_line(geo_mean):
            barcode_frame.at[index, "isChimera"] = True

    # ----------------------------
    # NEW: State persistence (V1)
    # ----------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "mark_actual_chimeras"
        table_name = f"{tag}_barcode_frame"

        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "tables": [table_name],
            "chimera_cut_line": getattr(chimera_cut_line, "__name__", "callable"),
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={
                table_name: barcode_frame,
            },
        )

    return barcode_frame


def stan_barcode_slope(
    manifest,
    data,
    index=None,
    spike_in_name=None,
    iterations=1000,
    iter_warmup=None,
    iter_sampling=None,
    chains=4,
    adapt_delta=0.9,
    tau_default=0.01,
    tau_de_weight=10,
    ref_tau_factor=1,
    return_fits=True,
    use_all_samples_model=True,
    slope_ref_prior_std=0.1,
    auto_save=True,
    overwrite=False,
    bi_linear_alpha=np.log(5),
    early_slope=False,
    dilution_factor=10,
    show_progress=False,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Run Stan barcode slope model.

    Full-table runs (index=None) are persisted.
    Single-index runs are returned only and NOT persisted.
    """

    if iter_warmup is None:
        iter_warmup = int(iterations / 2)
    if iter_sampling is None:
        iter_sampling = int(iterations / 2)

    plasmid = manifest.get('plasmid')
    if spike_in_name is None:
        if plasmid == 'pVER':
            spike_in_name = "AO-B"
        elif plasmid == 'pRamR':
            spike_in_name = "ON-01"

    arg_dict = dict(
        spike_in_name=spike_in_name,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        chains=chains,
        adapt_delta=adapt_delta,
        tau_default=tau_default,
        tau_de_weight=tau_de_weight,
        ref_tau_factor=ref_tau_factor,
        return_fits=return_fits,
        use_all_samples_model=use_all_samples_model,
        slope_ref_prior_std=slope_ref_prior_std,
        bi_linear_alpha=bi_linear_alpha,
        dilution_factor=dilution_factor,
        show_progress=show_progress,
    )

    # ============================================================
    # FULL DATASET MODE (persisted)
    # ============================================================
    if index is None:
        print("Using Stan model to determine slope for all barcodes")

        ignore_samples = manifest.get('ignore_samples', [])
        for ig in ignore_samples:
            print(f"ignoring or de-weighting sample {ig[0]}, time point {ig[1]-1}")

        arg_dict['return_fits'] = False
        arg_dict['verbose'] = False

        early_initial = '.ea.' if early_slope else ''

        if spike_in_name == "AO-B":
            initial = f'{early_initial}b'
        elif spike_in_name == "AO-E":
            initial = f'{early_initial}e'
        elif spike_in_name == "ON-01":
            initial = f'{early_initial}sp01'
        elif spike_in_name == "ON-02":
            initial = f'{early_initial}sp02'
        else:
            raise ValueError(f'spike_in_name not recognized: {spike_in_name}')

        fit_frame = data.copy()

        fitness_list_dict = {}
        err_list_dict = {}
        resid_list_dict = {}
        log_ratio_q_dict = {}

        ind0 = fit_frame.index[0]
        ret_dict = stan_barcode_slope_index(manifest, data, index=ind0, **arg_dict)

        key_list = list(ret_dict.keys())
        for key in key_list:
            p = ret_dict[key]
            fitness_list_dict[key] = [p[0]]
            err_list_dict[key] = [p[1]]
            resid_list_dict[key] = [p[2]]
            log_ratio_q_dict[key] = [p[3]]

        print_interval = 10 ** (np.round(np.log10(len(fit_frame))) - 1)

        for j, ind in enumerate(fit_frame.iloc[1:].index):
            if j % print_interval == 0:
                print(j + 1, ind)

            ret_dict = stan_barcode_slope_index(manifest, data, index=ind, **arg_dict)
            for key in key_list:
                p = ret_dict[key]
                fitness_list_dict[key].append(p[0])
                err_list_dict[key].append(p[1])
                resid_list_dict[key].append(p[2])
                log_ratio_q_dict[key].append(p[3])

        stan_str = 'sa' if use_all_samples_model else 's'

        for samp in fitness_list_dict.keys():
            fit_frame[f'fit_slope_S{samp}_{stan_str}{initial}'] = fitness_list_dict[samp]
            fit_frame[f'fit_slope_S{samp}_err_{stan_str}{initial}'] = err_list_dict[samp]
            fit_frame[f'fit_slope_S{samp}_resid_{stan_str}{initial}'] = list(resid_list_dict[samp])
            fit_frame[f'fit_slope_S{samp}_log_ratio_out_{stan_str}{initial}'] = list(log_ratio_q_dict[samp])

        # ----------------------------
        # NEW: State persistence (V1)
        # ----------------------------
        if auto_save:
            if state_dir is None:
                raise ValueError("state_dir must be provided when auto_save=True")

            tag = state_tag or "stan_barcode_slope"
            table_name = f"{tag}_fit_frame"

            manifest.setdefault("outputs", {})
            manifest["outputs"][tag] = {
                "tables": [table_name],
                "spike_in_name": spike_in_name,
                "iterations": iterations,
                "chains": chains,
                "adapt_delta": adapt_delta,
                "early_slope": early_slope,
                "use_all_samples_model": use_all_samples_model,
            }

            save_state_v1(
                manifest=manifest,
                state_dir=state_dir,
                tables={table_name: fit_frame},
            )

        return fit_frame

    # ============================================================
    # SINGLE-INDEX MODE (NOT persisted)
    # ============================================================
    else:
        arg_dict['index'] = index
        return stan_barcode_slope_index(manifest, data, **arg_dict)


def display_viewable_plate_layouts(manifest, data):
    sample_plate_map = manifest.get('sample_plate_map')
    col_contents = []
    antibiotic = manifest.get('antibiotic')
    plasmid = manifest.get('plasmid')
    for row in fitness_utils.rows():
        sel = [row in w for w in sample_plate_map.well]
        df = sample_plate_map[sel]
        st_list = []
        for ind, row2 in df.iterrows():
            st = f'S{row2.sample_id}, '
            if plasmid == 'Align-Protease':
                for ind_col in ['inducer1', 'inducer2']:
                    ind_id = row2[ind_col]
                    ind_conc = row2[ind_id]
                    if ind_conc > 0:
                        st += f'{ind_conc:.2f} {ind_id}, '
            if plasmid == 'Align-T7RNAP_1':
                pass
            else:
                lig = row2.ligand
                if lig != 'none':
                    st += f'{row2[lig]} {lig}, '
                
            st += f'{row2.antibiotic_conc} {antibiotic}'
            
            if plasmid == 'Align-TF':
                st += f', TF = {row2.transcription_factor}'
            
            st_list.append(st)
        col_contents.append(st_list)
    plate_layout_frame_2 = pd.DataFrame({r:cont for r, cont in zip(fitness_utils.rows(), col_contents)}, 
                                        index=[i+1 for i in range(12)])
    plate_layout_frame_2 = plate_layout_frame_2.transpose()

    plate_layout_frame_2.columns.name = 'BarSeq Samples Layout'
    display(plate_layout_frame_2)

    samp_dict = {}
    for st in plate_layout_frame_2[[1,2,3]].values.flatten():
        k = int(st[1:st.find(',')])
        v = st#[st.find(',')+2:]
        samp_dict[k] = v

    gp_samps_arr = np.array([[n for n in range(1, 13)],
                                [n for n in range(13, 25)]])

    gp_layout_arr = []
    for samps_row in gp_samps_arr:
        sub_arr = []
        for samp in samps_row:
            sub_arr.append(samp_dict[samp])
        gp_layout_arr.append(sub_arr)

    plate_layout_frame_3 = pd.DataFrame(gp_layout_arr, index=['A', 'B'])
    plate_layout_frame_3.rename(columns={n: n+1 for n in plate_layout_frame_3.columns}, inplace=True)
    plate_layout_frame_3.columns.name = 'Growth Plate Layout'
    display(plate_layout_frame_3)



def set_sample_plate_map(
    manifest,
    data,
    ignore_samples=[],
    verbose=True,
    auto_save=True,
    overwrite=False,
    growth_plate_layout_file=None,
    plasmid=None,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Set sample/plate mapping and annotate barcode_frame accordingly.

    Persists:
    - barcode_frame (Parquet)
    - sample_plate_map (Parquet)
    - structured metadata in manifest
    """

    # ----------------------------
    # Core logic (UNCHANGED)
    # ----------------------------
    manifest['ignore_samples'] = ignore_samples
    barcode_frame = data.copy()

    antibiotic_conc_list = manifest.get('antibiotic_conc_list', None)
    ligand_list = manifest.get('ligand_list', None)
    inducer_conc_lists = manifest.get('inducer_conc_lists', None)

    sample_plate_map = manifest.get('sample_plate_map', None)
    if sample_plate_map is None:
        sample_plate_map, anti_out = fitness_utils.get_sample_plate_map(
            growth_plate_layout_file=growth_plate_layout_file,
            inducer_list=ligand_list,
            inducer_conc_lists=inducer_conc_lists,
            tet_conc_list=antibiotic_conc_list,
            plasmid=plasmid,
        )
        if anti_out is not None:
            manifest['antibiotic'] = anti_out

    # Backward compatibility for ignore_samples
    if len(ignore_samples) > 0 and len(ignore_samples[0]) == 3:
        new_ignore = []
        for ig in ignore_samples:
            w = ig[0] == "tet"
            gp = ig[1]
            x = ig[2]
            df = sample_plate_map
            df = df[df.with_tet == w]
            df = df[df.growth_plate == gp]
            df = df[df[ligand_list[0]] == x]
            if len(df) == 1:
                row = df.iloc[0]
                new_ignore.append((row.sample_id, gp))
        ignore_samples = new_ignore

    if verbose and ignore_samples:
        for ig in ignore_samples:
            print(f"ignoring or de-weighting sample {ig[0]}, time point {ig[1]-1}")
        print()

    sample_list = np.unique(sample_plate_map.sample_id)

    sample_keep_dict = {}
    for s in sample_list:
        v = [(s, i + 2) not in ignore_samples for i in range(4)]
        sample_keep_dict[s] = v

    samples_with_tet = []
    samples_without_tet = []

    for s in sample_list:
        df = sample_plate_map[sample_plate_map["sample_id"] == s]
        has_tet = df.iloc[0].with_tet

        if has_tet:
            samples_with_tet.append(s)
        else:
            samples_without_tet.append(s)

        df = df.sort_values('growth_plate')
        well_list = list(df.well.values)

        if f"read_count_S{s}" not in barcode_frame.columns:
            bc_arr = np.array([barcode_frame[w].values for w in well_list]).transpose()
            barcode_frame[f"read_count_S{s}"] = [list(x) for x in bc_arr]

    if verbose:
        print(f"samples_with_tet: {samples_with_tet}")
        print(f"samples_without_tet: {samples_without_tet}")
        print()

    existing_plate_map = manifest.get('sample_plate_map', None)
    if existing_plate_map is not None:
        if np.any((existing_plate_map != sample_plate_map).values):
            raise Exception(
                "Attempting to change existing sample_plate_map. "
                "Set manifest['sample_plate_map'] = None to recompute."
            )

    # ----------------------------
    # Update manifest (STRUCTURED STATE)
    # ----------------------------
    manifest['sample_plate_map'] = "data/sample_plate_map.parquet"
    manifest['samples_with_tet'] = samples_with_tet
    manifest['samples_without_tet'] = samples_without_tet
    manifest['sample_keep_dict'] = sample_keep_dict
    manifest['ignore_samples'] = ignore_samples

    # ----------------------------
    # NEW: State persistence (V1)
    # ----------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "set_sample_plate_map"

        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "tables": [
                f"{tag}_barcode_frame",
                "sample_plate_map",
            ]
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={
                f"{tag}_barcode_frame": barcode_frame,
                "sample_plate_map": sample_plate_map,
            },
        )

    return sample_plate_map


def stan_barcode_slope_index(manifest,
                                data,
                                index,
                                spike_in_name=None,
                                iterations=1000,
                                iter_warmup=None,
                                iter_sampling=None,
                                chains=4,
                                adapt_delta=0.9,
                                stan_output_dir=None,
                                tau_default=0.01,
                                tau_de_weight=10,
                                ref_tau_factor=1,
                                return_fits=False,
                                verbose=True,
                                use_all_samples_model=True,
                                slope_ref_prior_std=0.1,
                                bi_linear_alpha=np.log(5),
                                early_slope=False,
                                dilution_factor=10,
                                show_progress=True):
    
    if iter_warmup is None:
        iter_warmup = int(iterations/2)
    if iter_sampling is None:
        iter_sampling = int(iterations/2)
    
    barcode_frame = data
    
    sample_plate_map = manifest.get('sample_plate_map')
    samples_with_tet = manifest.get('samples_with_tet')
    samples_without_tet = manifest.get('samples_without_tet')
    sample_keep_dict = manifest.get('sample_keep_dict')
    
    ref_samples = manifest.get('ref_samples')
    
    if ref_samples is None:
        ref_samples = samples_without_tet
        non_ref_without_tet = []
    else:
        non_ref_without_tet = list(set(samples_without_tet) - set(ref_samples))
    
    if verbose:
        print(f'Using these samples as reference samples: {ref_samples}')
    
    row = barcode_frame.loc[index]
    
    if spike_in_name is None:
        if manifest.get('plasmid') == 'pVER':
            spike_in_name = "AO-B"
        elif manifest.get('plasmid') == 'pRamR':
            spike_in_name = "ON-01"
    
    spike_in_row = barcode_frame[barcode_frame["RS_name"]==spike_in_name].iloc[0]
    
    x0 = np.array([i for i in range(4)])
    
    if use_all_samples_model:
        if ref_samples == samples_without_tet:
            # model with all zero-tet samples in reference group
            sm_file = 'Barcode_fitness_all_samples_2.stan'
        else:
            # model with some some zero-tet samples in refernce group and some in no_tet group
            sm_file = 'Barcode_fitness_all_samples.stan'
        stan_model = stan_utils.compile_model(sm_file, verbose=verbose)
    else:
        sm_no_tet_file = 'Barcode_fitness_no_tet.stan'
        stan_model_no_tet = stan_utils.compile_model(sm_no_tet_file, verbose=verbose)
    
    if use_all_samples_model:
        stan_data = dict(N=len(x0),
                            x=x0,
                            M_ref=len(ref_samples),
                            M_no_tet=len(non_ref_without_tet),
                            M_with_tet=len(samples_with_tet),
                            alpha=bi_linear_alpha, 
                            dilution_factor=dilution_factor, 
                            lower_bound_width=0.3,
                            slope_ref_prior_std=slope_ref_prior_std)
        n_reads_ref = []
        n_spike_ref = []
        tau_ref = []
        
        n_reads_no_tet = []
        n_spike_no_tet = []
        tau_no_tet = []
        
        n_reads_with_tet = []
        n_spike_with_tet = []
        tau_with_tet = []
    else:
        x_no_tet = []
        n_reads_no_tet = []
        spike_ins_no_tet = []
        tau_no_tet = []
        stan_fit_list = []
    fitness_out_dict = {}
    for samp_list in [ref_samples, non_ref_without_tet]:
        for samp in samp_list:
            if verbose: print(f'    sample {samp}')
            df = sample_plate_map
            df = df[df["sample_id"]==samp]
            df = df.sort_values('growth_plate')
            well_list = list(df.well.values)
        
            spike_in_reads = np.array(spike_in_row[well_list], dtype='int64')
            #spike_in_fitness = spike_in_fitness_dict[0][spike_in_name]
            
            n_reads = np.array(row[well_list], dtype='int64')
                    
            sel = sample_keep_dict[samp]
            tau = np.array([tau_default if s else tau_de_weight for s in sel])
            
            if use_all_samples_model:
                if samp_list is ref_samples:
                    n_reads_ref.append(n_reads)
                    n_spike_ref.append(spike_in_reads)
                    tau_ref.append(tau)
                else:
                    n_reads_no_tet.append(n_reads)
                    n_spike_no_tet.append(spike_in_reads)
                    tau_no_tet.append(tau)
            else:
                stan_data = dict(N=len(x0), x=x0, n_reads=n_reads, spike_in_reads=spike_in_reads, tau=tau)
                
                stan_fit = stan_model_no_tet.sample(data=stan_data, iter_sampling=iter_sampling, iter_warmup=iter_warmup, chains=chains, 
                                                    adapt_delta=adapt_delta, show_progress=show_progress, output_dir=stan_output_dir)
                if return_fits:
                    stan_fit_list.append(stan_fit)
            
                if samp_list is ref_samples:
                    x_no_tet += list(x0)
                    n_reads_no_tet += list(n_reads)
                    tau_no_tet += list(tau)
                    spike_ins_no_tet += list(spike_in_reads)
                
                fit_result = stan_fit.stan_variable('log_slope')
                fit_mu = np.mean(fit_result)
                fit_sig = np.std(fit_result)
                
                fit_result = stan_fit.stan_variable('log_ratio_out')
                fit_resid = np.log(n_reads) - np.log(spike_in_reads) - fit_mu
                log_ratio_out_quantiles = np.quantile(fit_result, [0.05, .25, .5, .75, .95], axis=0)
                
                fitness_out_dict[samp] = [fit_mu, fit_sig, fit_resid, log_ratio_out_quantiles]
    
    if not use_all_samples_model:
        if early_slope or (bi_linear_alpha is None):
            stan_model_with_tet = stan_model_no_tet
        else:
            # For bi_linear_alpha != None, run fit again, with counts from all zero-tet samples, and use result as slope_0 for fits to samples with tet
            x = np.array(x_no_tet)
            n_reads = np.array(n_reads_no_tet)
            spike_in_reads = np.array(spike_ins_no_tet)
            tau = np.array(tau_no_tet)*ref_tau_factor
            
            stan_data = dict(N=len(x), x=x, n_reads=n_reads, spike_in_reads=spike_in_reads, tau=tau)
            
            stan_fit = stan_model_no_tet.sample(data=stan_data, iter_sampling=iter_sampling, iter_warmup=iter_warmup, chains=chains, 
                                                adapt_delta=adapt_delta, show_progress=show_progress, output_dir=stan_output_dir)
            
            if return_fits:
                stan_fit_list.append(stan_fit)
            # Run fits for samples with antibiotic
            sm_with_tet_file = 'Barcode_fitness_with_tet.stan'
            stan_model_with_tet = stan_utils.compile_model(sm_with_tet_file, verbose=verbose)
            
            fit_result = stan_fit.stan_variable('log_slope')
            slope_0_mu = np.mean(fit_result)
            slope_0_sig = np.std(fit_result)
    
    rng = np.random.default_rng()
    for samp in samples_with_tet:
        if verbose: print(f'    sample {samp}')
        df = sample_plate_map
        df = df[df["sample_id"]==samp]
        df = df.sort_values('growth_plate')
        well_list = list(df.well.values)
    
        spike_in_reads = np.array(spike_in_row[well_list], dtype='int64')
        #spike_in_fitness = spike_in_fitness_dict[0][spike_in_name]
        
        n_reads = np.array(row[well_list], dtype='int64')
                
        sel = sample_keep_dict[samp]
        tau = np.array([tau_default if s else tau_de_weight for s in sel])
        
        if use_all_samples_model:
            n_reads_with_tet.append(n_reads)
            n_spike_with_tet.append(spike_in_reads)
            tau_with_tet.append(tau)
        else:
            x = x0
            # If early_slope == True, use only the first two time points (x = 2, 3)
            if early_slope:
                x = x[:2]
                n_reads = n_reads[:2]
                spike_in_reads = spike_in_reads[:2]
                tau = tau[:2]
            # If bi_linear_alpha is set to None, then use linear fit to time points 2, 3, 4 (x = 3, 4, 5).
            elif bi_linear_alpha is None:
                x = x[1:]
                n_reads = n_reads[1:]
                spike_in_reads = spike_in_reads[1:]
                tau = tau[1:]
            
            if early_slope or (bi_linear_alpha is None):
                stan_data = dict(N=len(x), x=x, 
                                    n_reads=n_reads, 
                                    spike_in_reads=spike_in_reads, 
                                    tau=tau)
            else:
                stan_data = dict(N=len(x), x=x, 
                                    n_reads=n_reads, 
                                    spike_in_reads=spike_in_reads, 
                                    tau=tau,
                                    slope_0_mu=slope_0_mu, 
                                    slope_0_sig=slope_0_sig, 
                                    alpha=bi_linear_alpha, 
                                    dilution_factor=dilution_factor, 
                                    lower_bound_width=0.3)
            
            try:
                stan_fit = stan_model_with_tet.sample(data=stan_data, iter_sampling=iter_sampling, iter_warmup=iter_warmup, chains=chains, 
                                                        adapt_delta=adapt_delta, output_dir=stan_output_dir)
                last_good_stan_fit = stan_fit
            except RuntimeError as err:
                if 'Initialization failed' in f'{err}':
                    print(f'Stan random init failed, re-trying with defined init.')
                    print(f'spike_in_reads: {spike_in_reads}')
                    
                    n_mean = n_reads.astype(float)
                    n_mean[n_mean==0] = 0.1
                    
                    log_ratios = np.log(n_mean) - np.log(spike_in_reads)
                    log_starting_ratio = log_ratios[0]
                    
                    y = (np.log(n_mean) - np.log(spike_in_reads))
                    s = np.sqrt(1/n_mean + 1/spike_in_reads)
                    print(f'x: {x}')
                    print(f'y: {y}')
                    print(f's: {s}')
                    
                    popt, pcov = curve_fit(fitness_utils.line_funct, x, y, sigma=s, absolute_sigma=True)
                    print(f'popt: {popt}')
                    log_slope = popt[0]
                    log_intercept = popt[1] - log_starting_ratio
                    
                    
                    log_err = (log_ratios - log_starting_ratio - log_intercept) - log_slope*x
                    print(f'log_err: {log_err}')
                    tau_rev = tau.copy()
                    tau_rev[n_reads==0] *= 30
                    log_err_tilda = log_err/tau_rev
                    
                    stan_init = {}
                    stan_init['log_slope'] = log_slope
                    stan_init['log_intercept'] = log_intercept
                    stan_init['log_err_tilda'] = log_err_tilda
                    print(stan_init)
                    
                    n_mean_test = spike_in_reads*np.exp(log_starting_ratio + log_intercept + log_slope*x + log_err)
                    print(f'n_mean_test: {n_mean_test}')
                    
                    #print(f'last good stan: {last_good_stan_fit.get_last_position()}')
                    
                    try:
                        stan_fit = stan_model_with_tet.sample(data=stan_data, iter_sampling=iter_sampling, iter_warmup=iter_warmup, chains=chains, 
                                                                adapt_delta=adapt_delta, inits=stan_init, output_dir=stan_output_dir)
                        
                    except Exception as err:
                        print(f'Stan fit failed again, giving up: {err}')
                        stan_fit = 'failed'
                else:
                    print(f'Stan fit error: {err}')
            except:
                print('Stan fit failed')
                stan_fit = 'failed'
                
            if return_fits:
                stan_fit_list.append(stan_fit)
                
            if stan_fit != 'failed':
                fit_mu = np.mean(stan_fit.stan_variable('log_slope'))
                fit_sig = np.std(stan_fit.stan_variable('log_slope'))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_resid = np.log(n_reads) - np.log(spike_in_reads) - np.mean(stan_fit.stan_variable('log_ratio_out'), axis=0)
                log_ratio_out_quantiles = np.quantile(stan_fit.stan_variable('log_ratio_out'), [0.05, .25, .5, .75, .95], axis=0)
            else:
                return
            
            fitness_out_dict[samp] = [fit_mu, fit_sig, fit_resid, log_ratio_out_quantiles]
    
    if use_all_samples_model:
        stan_data['n_reads_ref'] = np.array(n_reads_ref).transpose()
        stan_data['n_spike_ref'] = np.array(n_spike_ref).transpose()
        stan_data['tau_ref'] = np.array(tau_ref).transpose()
        
        stan_data['n_reads_no_tet'] = np.array(n_reads_no_tet).transpose()
        stan_data['n_spike_no_tet'] = np.array(n_spike_no_tet).transpose()
        stan_data['tau_no_tet'] = np.array(tau_no_tet).transpose()
        
        stan_data['n_reads_with_tet'] = np.array(n_reads_with_tet).transpose()
        stan_data['n_spike_with_tet'] = np.array(n_spike_with_tet).transpose()
        stan_data['tau_with_tet'] = np.array(tau_with_tet).transpose()
        
        stan_fit = stan_model.sample(data=stan_data, iter_sampling=iter_sampling, iter_warmup=iter_warmup, chains=chains, 
                                        adapt_delta=adapt_delta, show_progress=show_progress, output_dir=stan_output_dir)
        
        if not return_fits:
            if len(non_ref_without_tet) == 0:
                sample_lists = [ref_samples, samples_with_tet]
                sample_str_list = ['ref', 'with_tet']
            else:
                sample_lists = [ref_samples, non_ref_without_tet, samples_with_tet]
                sample_str_list = ['ref', 'no_tet', 'with_tet']
            for samp_list, samp_str in zip(sample_lists, sample_str_list):
                fit_result = stan_fit.stan_variable(f'slope_{samp_str}')
                fit_mu_list = np.mean(fit_result, axis=0)
                fit_sig_list = np.std(fit_result, axis=0)
                if samp_list is ref_samples:
                    fit_mu_list = [fit_mu_list]*len(ref_samples)
                    fit_sig_list = [fit_sig_list]*len(ref_samples)
                
                fit_result = stan_fit.stan_variable(f'residuals_{samp_str}')
                fit_resid_list = np.mean(fit_result, axis=0).transpose()
                
                fit_result = stan_fit.stan_variable(f'log_ratio_out_{samp_str}')
                log_ratio_out_list = np.quantile(fit_result, [0.05, .25, .5, .75, .95], axis=0).transpose([2,0,1])
                for samp, mu, sig, res, log_rat in zip(samp_list, fit_mu_list, fit_sig_list, fit_resid_list, log_ratio_out_list):
                    fitness_out_dict[samp] = [mu, sig, res, log_rat]
                
    if return_fits:
        if use_all_samples_model:
            return stan_fit
        else:
            return stan_fit_list
    else:
        return fitness_out_dict
    

def fit_barcode_slope(manifest,
                        data,
                        auto_save=True,
                        overwrite=False,
                        refit_index=None,
                        ref_slope_to_average=True,
                        bi_linear_alpha=np.log(5),
                        bi_linear_x0=None,
                        early_slope=False,
                        mid_slope=False,
                        all_slope=False,
                        use_all_ref_samples=True,
                        float_replace_zero=0.1,
                        min_log_count_error=0):
                        
    for ig in manifest.get('ignore_samples', []):
        print(f"ignoring or de-weighting sample {ig[0]}, time point {ig[1]-1}")
    
    return plot_or_fit_barcode_ratios(manifest,
                                            data,
                                            auto_save=auto_save,
                                            overwrite=overwrite,
                                            refit_index=refit_index,
                                            ref_slope_to_average=ref_slope_to_average,
                                            bi_linear_alpha=bi_linear_alpha,
                                            bi_linear_x0=bi_linear_x0,
                                            plots_not_fits=False,
                                            early_slope=early_slope,
                                            mid_slope=mid_slope,
                                            all_slope=all_slope,
                                            float_replace_zero=float_replace_zero,
                                            use_all_ref_samples=use_all_ref_samples,
                                            min_log_count_error=min_log_count_error)
    

def plot_count_ratio_per_sample(manifest,
                                data,
                                plot_range=None,
                                spike_in_initial=None,
                                max_plots=20,
                                log_scale=None,
                                plot_raw_counts=False):
                                
    plt.rcParams["figure.figsize"] = [26, 13]

    
    
    if spike_in_initial is None:
        spike_in_initial = get_default_initial(manifest)
        
    spike_in = fitness_utils.get_spike_in_name_from_inital(manifest.get('plasmid'), spike_in_initial)
    
    barcode_frame = data
    spike_in_row = barcode_frame[barcode_frame.RS_name==spike_in].iloc[0]
    
    plot_frame = barcode_frame
    if plot_range is not None:
        plot_frame = plot_frame.loc[plot_range[0]:plot_range[1]]
    if len(plot_frame) > max_plots:
        plot_frame = plot_frame.iloc[:max_plots]

    x = np.array([i for i in range(4)])
    plot_list_0 = manifest.get('samples_without_tet', []) + manifest.get('samples_with_tet', [])
    sample_plate_map = manifest.get('sample_plate_map')

    for ind, row in plot_frame.iterrows():
        fig, axs = plt.subplots(4, 6)
        suptitle = f'index: {ind}'
        if row.RS_name != '':
            suptitle += f', {row.RS_name}'
        fig.suptitle(suptitle, size=24, y=0.925)
        axs = axs.flatten()
        
        plot_list = plot_list_0
        if (manifest.get('plasmid') == 'Align-TF') and ('norm' not in row.RS_name):
            tf = align_tf_from_RS_name(row.RS_name)
            df_tf = sample_plate_map
            df_tf = df_tf[df_tf.transcription_factor==tf]
            samples_with_tf = np.unique(df_tf.sample_id)
            plot_list = [s for s in plot_list_0 if s in samples_with_tf] + [s for s in plot_list_0 if s not in samples_with_tf]
        
        for samp, ax in zip(plot_list, axs):
            if samp in manifest.get('samples_with_tet', []):
                for s in ['top', 'bottom', 'left', 'right']:
                    ax.spines[s].set_color('red')
            df = sample_plate_map
            df = df[df["sample_id"]==samp]
            df = df.sort_values('growth_plate')
            well_list = list(df.well.values)

            spike_in_reads = np.array(spike_in_row[well_list], dtype='int64')
            n_reads = np.array(row[well_list], dtype='int64')
            
            if plot_raw_counts:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = np.log(n_reads)
                    s = np.sqrt(1/n_reads)
                    s[n_reads == 0] = np.log(10)
                    if log_scale is not None:
                        y = y/np.log(log_scale)
                        s = s/np.log(log_scale)
                ax.errorbar(x[n_reads>0], y[n_reads>0], s[n_reads>0], fmt='o', ms=10);
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = np.log(n_reads) - np.log(spike_in_reads)
                    s = np.sqrt(1/n_reads + 1/spike_in_reads)
                    if log_scale is not None:
                        y = y/np.log(log_scale)
                        s = s/np.log(log_scale)
                ax.errorbar(x[n_reads>0], y[n_reads>0], s[n_reads>0], fmt='o', ms=10);
                
                log_ratio = row[f'fit_slope_S{samp}_log_ratio_out_{spike_in_initial}']
                if log_scale is not None:
                    log_ratio = log_ratio/np.log(log_scale)
                    
                if len(log_ratio.shape) ==  1:
                    ax.plot(x, log_ratio, '--k')
                else:
                    if log_ratio.shape == (5, 3):
                        x_plt = x[1:]
                    else:
                        x_plt = x
                    for q in log_ratio:
                        ax.plot(x_plt, q);
            
            ax.set_title(f'sample {samp}')




def plot_or_fit_barcode_ratios(
    manifest,
    data,
    auto_save=True,
    overwrite=False,
    refit_index=None,
    ref_slope_to_average=True,
    bi_linear_alpha=np.log(5),
    bi_linear_x0=None,
    plots_not_fits=False,
    plot_range=None,
    show_spike_ins=None,
    show_bc_str=False,
    plot_samples=None,
    early_slope=False,
    mid_slope=False,
    all_slope=False,
    use_all_ref_samples=True,
    float_replace_zero=0.1,
    min_log_count_error=0,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Plot or fit barcode log-ratios.

    Only the *fit* mode (plots_not_fits=False) produces persisted state.
    """

    barcode_frame = data

    if show_spike_ins is None:
        show_spike_ins = [get_default_initial(manifest)]

    if refit_index is None:
        if not plots_not_fits:
            print(
                f"Fitting log(barcode ratios) to find fitness for each barcode "
                f"in {manifest.get('experiment')}"
            )

        sample_plate_map = manifest.get('sample_plate_map')
        samples_with_tet = manifest.get('samples_with_tet')
        samples_without_tet = manifest.get('samples_without_tet')
        sample_keep_dict = manifest.get('sample_keep_dict')

    if plot_samples is None:
        plot_samples = samples_with_tet + samples_without_tet

    antibiotic_conc_list = manifest.get('antibiotic_conc_list')
    high_tet = antibiotic_conc_list[-1]

    # --- plasmid-specific spike-ins (UNCHANGED) ---
    if manifest.get('plasmid') == 'pVER':
        spike_1, spike_2 = "AO-B", "AO-E"
        spike_1_init, spike_2_init = 'b', 'e'
    elif manifest.get('plasmid') == 'pRamR':
        spike_1, spike_2 = "ON-01", "ON-02"
        spike_1_init, spike_2_init = 'sp01', 'sp02'
    elif manifest.get('plasmid') == 'pCymR':
        spike_1, spike_2 = 'AO-09', 'RS-20'
        spike_1_init, spike_2_init = 'sp09', 'rs20'
    elif manifest.get('plasmid') == 'Align-TF':
        spike_1, spike_2 = 'pRamR-norm-02', 'pLacI-norm-02'
        spike_1_init, spike_2_init = 'ramr', 'laci'
    elif manifest.get('plasmid') in ['Align-Protease', 'Align-T7RNAP_1']:
        spike_1, spike_2 = 'pRamR-norm-02', 'pNorm-mDHFR-03'
        spike_1_init, spike_2_init = 'nrm02', 'nrm03'

    # --- slope mode prefix ---
    if early_slope:
        early_initial = 'ea.'
    elif mid_slope:
        early_initial = 'mid.'
    elif all_slope:
        early_initial = 'all.'
    else:
        early_initial = ''

    # --- reference rows ---
    spike_in_row_dict = {
        spike_1: barcode_frame[barcode_frame["RS_name"] == spike_1].iloc[0],
        spike_2: barcode_frame[barcode_frame["RS_name"] == spike_2].iloc[0],
    }

    # --- plotting or fitting frame ---
    if plots_not_fits:
        if plot_range is None:
            plot_range = [0, max(barcode_frame.index)]
        fit_frame = barcode_frame.loc[plot_range[0]:plot_range[1]]

        plt.rcParams["figure.figsize"] = [10, 4 * len(fit_frame)]
        fig, axs = plt.subplots(len(fit_frame), 1)
    else:
        fit_frame = barcode_frame if refit_index is None else barcode_frame.loc[refit_index:refit_index]
        axs = [None for _ in range(len(fit_frame))]

    # --- core fitting loops ---
    x0 = np.array([2, 3, 4, 5])

    for spike_in, initial in zip(
        [spike_1, spike_2],
        [spike_1_init, spike_2_init],
    ):
        no_tet_slope_lists = []

        # ============================
        # WITHOUT TET
        # ============================
        for samp in samples_without_tet:
            df = sample_plate_map[sample_plate_map["sample_id"] == samp].sort_values("growth_plate")
            well_list = list(df.well.values)

            spike_in_reads = np.array(spike_in_row_dict[spike_in][well_list], dtype='int64')

            f_est_list, f_err_list, slope_list = [], [], []
            resids_list, log_ratio_out_list = [], []

            for (index, row), ax in zip(fit_frame.iterrows(), axs):
                n_reads = np.array(row[well_list], dtype='int64')
                x = x0

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = np.log(n_reads) - np.log(spike_in_reads)
                    s = np.sqrt(1 / n_reads + 1 / spike_in_reads + min_log_count_error ** 2)

                sel = np.array(sample_keep_dict[samp])
                s[~sel] = 10
                sel = (n_reads > 0) & (spike_in_reads > 0)

                x = x[sel]
                y = y[sel]
                s = s[sel]

                if plots_not_fits:
                    slope_list.append(row[f'fit_slope_S{samp}_{initial}'])
                else:
                    if len(x) > 1:
                        popt, pcov = curve_fit(fitness_utils.line_funct, x, y, sigma=s, absolute_sigma=True)
                        f_est_list.append(popt[0])
                        f_err_list.append(np.sqrt(pcov[0, 0]))
                    else:
                        f_est_list.append(np.nan)
                        f_err_list.append(np.nan)

            if not plots_not_fits:
                fit_frame[f'fit_slope_S{samp}_{early_initial}{initial}'] = f_est_list
                fit_frame[f'fit_slope_S{samp}_err_{early_initial}{initial}'] = f_err_list

            no_tet_slope_lists.append(f_est_list if not plots_not_fits else slope_list)

        no_tet_slope_lists = np.array(no_tet_slope_lists)
        no_tet_slope = (
            no_tet_slope_lists.mean(axis=0)
            if ref_slope_to_average
            else [x[0] for x in no_tet_slope_lists.T]
        )

        # ============================
        # WITH TET
        # ============================
        for samp in samples_with_tet:
            df = sample_plate_map[sample_plate_map["sample_id"] == samp].sort_values("growth_plate")
            well_list = list(df.well.values)

            spike_in_reads = np.array(spike_in_row_dict[spike_in][well_list], dtype='int64')

            f_est_list, f_err_list = [], []

            for (index, row), slope_0, ax in zip(fit_frame.iterrows(), no_tet_slope, axs):
                n_reads = np.array(row[well_list], dtype='int64')

                if early_slope or mid_slope or bi_linear_alpha is None:
                    n_reads = n_reads.astype(float)
                    n_reads[n_reads == 0] = float_replace_zero

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = np.log(n_reads) - np.log(spike_in_reads)
                    s = np.sqrt(1 / n_reads + 1 / spike_in_reads + min_log_count_error ** 2)

                sel = np.array(sample_keep_dict[samp])
                s[~sel] = 10
                sel = (n_reads > 0) & (spike_in_reads > 0)

                x = x0[sel]
                y = y[sel]
                s = s[sel]

                if not plots_not_fits:
                    if len(x) > 1:
                        fit_funct = (
                            fitness_utils.line_funct
                            if early_slope or mid_slope
                            else lambda xp, mp, bp: fitness_utils.bi_linear_funct(
                                xp - 2, mp, bp, slope_0, alpha=bi_linear_alpha
                            )
                        )
                        popt, pcov = curve_fit(fit_funct, x, y, sigma=s, absolute_sigma=True)
                        f_est_list.append(popt[0])
                        f_err_list.append(np.sqrt(pcov[0, 0]))
                    else:
                        f_est_list.append(np.nan)
                        f_err_list.append(np.nan)

            if not plots_not_fits:
                fit_frame[f'fit_slope_S{samp}_{early_initial}{initial}'] = f_est_list
                fit_frame[f'fit_slope_S{samp}_err_{early_initial}{initial}'] = f_err_list

    # ============================
    # Persistence (FIT MODE ONLY)
    # ============================
    if not plots_not_fits:
        if auto_save:
            if state_dir is None:
                raise ValueError("state_dir must be provided when auto_save=True")

            tag = state_tag or "plot_or_fit_barcode_ratios"
            table_name = f"{tag}_barcode_frame"

            manifest.setdefault("outputs", {})
            manifest["outputs"][tag] = {
                "tables": [table_name],
                "early_slope": early_slope,
                "mid_slope": mid_slope,
                "all_slope": all_slope,
                "bi_linear_alpha": bi_linear_alpha,
            }

            save_state_v1(
                manifest=manifest,
                state_dir=state_dir,
                tables={table_name: fit_frame},
            )

        return fit_frame

    # plots_only
    return data


def add_fitness_from_slopes(
    manifest,
    data,
    initial=None,
    auto_save=True,
    overwrite=False,
    is_on_aws=False,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Convert fitted slopes into fitness values using spike-in calibration.

    Persists the updated fit_frame as Parquet.
    """

    fit_frame = data.copy()
    plasmid = manifest.get('plasmid')
    sample_plate_map = manifest.get('sample_plate_map')
    sample_list = np.unique(sample_plate_map.sample_id)

    # spike_in_fitness_dict:
    #   {tet_conc -> {spike_in_name -> float or (interp_fn, interp_err_fn)}}
    spike_in_fitness_dict = fitness_utils.fitness_calibration_dict(
        plasmid=plasmid,
        barseq_directory=manifest.get('notebook_dir'),
        is_on_aws=is_on_aws,
    )

    k1 = list(spike_in_fitness_dict.keys())[0]
    k2 = list(spike_in_fitness_dict[k1].keys())[0]
    constant_spike_in = isinstance(spike_in_fitness_dict[k1][k2], float)

    if initial is None:
        initial = get_default_initial(manifest)

    spike_in = fitness_utils.get_spike_in_name_from_inital(plasmid, initial)

    for samp in sample_list:
        df = sample_plate_map[sample_plate_map["sample_id"] == samp]
        tet_conc = df.antibiotic_conc.iloc[0]

        if constant_spike_in:
            spike_in_fitness = spike_in_fitness_dict[tet_conc][spike_in]
            spike_in_fitness_err = 0.0
        else:
            if plasmid == 'pVER':
                ligand = df.ligand.iloc[0]
                lig_conc = df[ligand].iloc[0] if ligand != 'none' else 0
                spike_in_fitness, spike_in_fitness_err = (
                    spike_in_fitness_dict[tet_conc][spike_in](ligand, lig_conc)
                )
            elif plasmid == 'pRamR':
                lig_conc = max(df[manifest.get('ligand_list')].iloc[0].values)
                spike_in_fitness = spike_in_fitness_dict[tet_conc][spike_in][0](lig_conc)
                spike_in_fitness_err = spike_in_fitness_dict[tet_conc][spike_in][1](lig_conc)
            elif plasmid == 'pCymR':
                ligand = df.ligand.iloc[0]
                lig_conc = df[ligand].iloc[0] if ligand != 'none' else 0
                spike_in_fitness, spike_in_fitness_err = (
                    spike_in_fitness_dict[tet_conc][spike_in](ligand, lig_conc)
                )
            elif plasmid in ['Align-TF', 'Align-Protease', 'Align-T7RNAP_1']:
                ligand = 'none'
                lig_conc = 0
                spike_in_fitness, spike_in_fitness_err = (
                    spike_in_fitness_dict[tet_conc][spike_in](ligand, lig_conc)
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_frame[f'fitness_S{samp}_{initial}'] = (
                spike_in_fitness +
                fit_frame[f'fit_slope_S{samp}_{initial}'] / np.log(10)
            )
            fit_frame[f'fitness_S{samp}_err_{initial}'] = np.sqrt(
                spike_in_fitness_err**2 +
                (fit_frame[f'fit_slope_S{samp}_err_{initial}'] / np.log(10))**2
            )

    data = fit_frame.copy()

    # ----------------------------
    # NEW: State persistence (V1)
    # ----------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "add_fitness_from_slopes"
        table_name = f"{tag}_fit_frame"

        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "tables": [table_name],
            "initial": initial,
            "plasmid": plasmid,
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={table_name: data},
        )

    return data

def plot_fit_residuals(manifest, data, initial=None, alpha=0.3):
    if initial is None:
        initial = get_default_initial(manifest)
    
    barcode_frame = data
    
    antibiotic_conc_list = manifest.get('antibiotic_conc_list')

    ligand_list = manifest.get('ligand_list')
    antibiotic = manifest.get('antibiotic')
    
    sample_plate_map = manifest.get('sample_plate_map')
    
    plt.rcParams["figure.figsize"] = [16, 3]
    sample_list = np.unique(sample_plate_map.sample_id)
    if manifest.get('plasmid') == 'Align-TF':
        tf_dict = {}
        for samp in sample_list:
            df = sample_plate_map
            df = df[df.sample_id==samp]
            tf = df.iloc[0].transcription_factor
            tf_dict[samp] = tf
    
    mean_resid_lists = []
    rms_resid_lists = []
    tet_list = []
    for samp in sample_list:
        mean_sub_list = []
        rms_sub_list = []
        df = sample_plate_map
        df = df[df.sample_id==samp]
        tet = df.antibiotic_conc.iloc[0]
        tet_list.append(tet)
        
        df_bc = barcode_frame
        if manifest.get('plasmid') == 'Align-TF':
            # Only plot residuals for rows/variants that are in each sample
            sel_tf = [(align_tf_from_RS_name(x) == tf_dict[samp]) or ('norm' in x) for x in df_bc.RS_name]
            df_bc = df_bc[sel_tf]
        
        resid_array = df_bc[f'fit_slope_S{samp}_resid_{initial}']
        sel = [len(x)==4 for x in resid_array]
        resid_array = np.array(list(resid_array[sel]))
        x_list = [df_bc[sel][x] for x in df.well]
        w_list = df.well
        y_list = resid_array.transpose()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, axs = plt.subplots(1, 4)
        txt = f'Sample {samp}: {tet} {antibiotic}' 
        for lig in ligand_list:
            c = df[lig].iloc[0]
            txt += f', {c} {lig}'
        fig.suptitle(txt, size=20, y=1.1)
        for x, y, w, ax in zip(x_list, y_list, w_list, axs):
            ax.set_title(w, size=16)
            #ax.plot(x, y, 'o', alpha=0.2);
            s = (x>10)&(~np.isnan(y))
            y = y[s]
            x = x[s]
            if len(x) == 0:
                pass
            elif len(x) > 1000:
                ax.hist2d(np.log10(x), y, bins=50, norm=colors.LogNorm())
            else:
                ax.plot(np.log10(x), y, 'o', alpha=alpha)
            ax.set_xlabel('log10(read count)')
            if ax is axs[0]:
                ax.set_ylabel('fit residual')
            if len(x) != 0:
                mean_sub_list.append(np.mean(y))
                rms_sub_list.append(np.sqrt(np.mean(y**2)))
            else:
                mean_sub_list.append(np.nan)
                rms_sub_list.append(np.nan)
        mean_resid_lists.append(mean_sub_list)
        rms_resid_lists.append(rms_sub_list)

    plt.rcParams["figure.figsize"] = [12, 3]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Residuals vs. time for each sample', size=20)
    marker_list = ['-o']*8 + ['-^']*8 + ['-v']*8
    x = [i for i in range(1, 5)]
    for samp, mean_list, rms_list, marker, tet in zip(sample_list, mean_resid_lists, rms_resid_lists, marker_list, tet_list):
        for ax, y in zip(axs, [mean_list, rms_list]):
            ax.plot(x, y, marker, label=f'S{samp}', fillstyle=None, ms=12);
    ax.legend(loc='upper left', bbox_to_anchor= (1.05, 1.05), ncol=3);
    axs[0].set_xlabel('time point')
    axs[1].set_xlabel('time point')
    axs[0].set_ylabel('mean residuals')
    axs[1].set_ylabel('rms residuals');
    
    return (sample_list, mean_resid_lists, rms_resid_lists)



def set_fit_fitness_difference_params(
    manifest,
    fit_fitness_difference_params=None,
    params_file=None,
    auto_save=True,
    overwrite=False,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Set parameters used to fit fitness differences.

    This function updates manifest-only state.
    No tabular data or models are produced.
    """

    antibiotic_conc_list = manifest.get('antibiotic_conc_list')
    plasmid = manifest.get('plasmid')

    # --------------------------------------------------
    # Resolve parameters
    # --------------------------------------------------
    if fit_fitness_difference_params is not None:
        params = fit_fitness_difference_params

    elif params_file is not None:
        # NEW: YAML instead of pickle
        with open(params_file, "r") as f:
            params = yaml.safe_load(f)

    else:
        params = [
            fitness_utils.fit_fitness_difference_params(
                plasmid=plasmid,
                tet_conc=x
            )
            for x in antibiotic_conc_list[1:]
        ]

    # Normalize shape (list-of-lists)
    if params and not isinstance(params[0], list):
        params = [params]

    # --------------------------------------------------
    # Update manifest (STRUCTURED STATE ONLY)
    # --------------------------------------------------
    manifest['fit_fitness_difference_params'] = params

    # --------------------------------------------------
    # NEW: State persistence (manifest only)
    # --------------------------------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "set_fit_fitness_difference_params"

        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "manifest_only": True,
            "source": (
                "explicit"
                if fit_fitness_difference_params is not None
                else "file"
                if params_file is not None
                else "computed"
            ),
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables=None,
            models=None,
        )

    return manifest



def stan_fitness_difference_curves(
    manifest,
    data,
    adapt_delta=0.9,
    iterations=1000,
    iter_warmup=None,
    iter_sampling=None,
    chains=4,
    stan_output_dir=None,
    show_progress=False,
    auto_save=True,
    overwrite=False,
    refit_indexes=None,
    return_fit=False,
    initial=None,
    re_stan_on_rhat=True,
    rhat_cutoff=1.05,
    log_x_max=None,
    *,
    state_dir=None,
    state_tag=None,
):
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.disabled = True

    plasmid = manifest.get('plasmid')
    fit_fitness_difference_params = manifest.get('fit_fitness_difference_params')

    if iter_warmup is None:
        iter_warmup = int(iterations / 2)
    if iter_sampling is None:
        iter_sampling = int(iterations / 2)

    if initial is None:
        initial = get_default_initial(manifest)

    print(f"Using Stan to fit to fitness curves to find sensor parameters for {manifest.get('experiment')}")
    print(f"  Using fitness parameters for {plasmid}:")
    print(f"      {fit_fitness_difference_params}")
    print("      Method version from 2023-06-04")

    barcode_frame = data

    fitness_columns_setup = get_fitness_columns_setup(manifest, data, plot_initials=[initial])
    ligand_list = manifest.get('ligand_list')

    if fitness_columns_setup[0]:
        old_style_columns, x, linthresh, fit_plot_colors = fitness_columns_setup
        plot_df = None
    else:
        old_style_columns, linthresh, fit_plot_colors, plot_df = fitness_columns_setup

    antibiotic_conc_list = manifest.get('antibiotic_conc_list')

    if len(ligand_list) == 1:
        sm_file = 'Double_Hill_equation_fit.stan'
        params_list = [
            'log_g0', 'log_ginf_1', 'log_ec50_1', 'sensor_n_1', 'log_ginf_g0_ratio_1',
            'low_fitness_high_tet', 'mid_g_high_tet', 'fitness_n_high_tet'
        ]
        log_g0_ind = params_list.index('log_g0')
        log_ginf_g0_ind = params_list.index('log_ginf_g0_ratio_1')
        params_dim = len(params_list)

    elif len(ligand_list) == 2:
        sm_file = 'Double_Hill_equation_fit.two-lig.two-tet.stan'
        params_list = [
            'log_g0',
            'log_ginf_1', 'log_ec50_1', 'sensor_n_1', 'log_ginf_g0_ratio_1',
            'log_ginf_2', 'log_ec50_2', 'sensor_n_2', 'log_ginf_g0_ratio_2',
            'low_fitness_low_tet', 'mid_g_low_tet', 'fitness_n_low_tet',
            'low_fitness_high_tet', 'mid_g_high_tet', 'fitness_n_high_tet'
        ]
        log_g0_ind = params_list.index('log_g0')
        log_ginf_g0_ind_1 = params_list.index('log_ginf_g0_ratio_1')
        log_ginf_g0_ind_2 = params_list.index('log_ginf_g0_ratio_2')
        params_dim = len(params_list)

    elif len(ligand_list) == 3:
        params_list = [
            'log_g0',
            'log_ginf_1', 'log_ec50_1', 'sensor_n_1', 'log_ginf_g0_ratio_1', 'spec_1',
            'log_ginf_2', 'log_ec50_2', 'sensor_n_2', 'log_ginf_g0_ratio_2', 'spec_2',
            'log_ginf_3', 'log_ec50_3', 'sensor_n_3', 'log_ginf_g0_ratio_3', 'spec_3',
            'mean_log_ec50'
        ]
        if plasmid == 'pRamR':
            params_list += ['high_fitness', 'mid_g', 'fitness_n']
            sm_file = 'Double_Hill_equation_fit.three-lig.inverted.stan'
        else:
            params_list += ['low_fitness', 'mid_g', 'fitness_n']
            sm_file = 'Double_Hill_equation_fit.three-lig.stan'

        log_g0_ind = params_list.index('log_g0')
        log_ginf_g0_ind_1 = params_list.index('log_ginf_g0_ratio_1')
        log_ginf_g0_ind_2 = params_list.index('log_ginf_g0_ratio_2')
        log_ginf_g0_ind_3 = params_list.index('log_ginf_g0_ratio_3')
        params_dim = len(params_list)

    fit_ind = np.where(['fitness' in x for x in params_list])[0][0]
    quantile_params_list = params_list[:fit_ind]
    quantile_params_dim = len(quantile_params_list)

    key_params = params_list

    print(f'    Using model from file: {sm_file}')
    stan_model = stan_utils.compile_model(sm_file)

    quantile_list = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantile_dim = len(quantile_list)

    log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf = fitness_utils.log_g_limits(plasmid=plasmid)
    print(f'log_g_limits: {log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf}')

    rng = np.random.default_rng()

    def stan_fit_row(st_row, st_index, lig_list, return_fit=False):
        print()
        now = datetime.datetime.now()
        print(f"{now}, fitting row index: {st_index}, for ligands: {lig_list}")

        is_gp_model = (len(lig_list) > 1)

        try:
            stan_data = bs_frame_stan_data(manifest, data, st_row, initial=initial, is_gp_model=is_gp_model)

            if log_x_max is not None:
                stan_data['log_x_max'] = log_x_max
                print(f'Manually setting log_x_max: {log_x_max}')

            if len(lig_list) == 1:
                stan_init = init_stan_fit_single_ligand(stan_data, fit_fitness_difference_params)
            elif len(lig_list) == 2:
                stan_init = init_stan_fit_two_lig_two_tet(stan_data, fit_fitness_difference_params)
            elif len(lig_list) == 3:
                stan_init = init_stan_fit_three_ligand(stan_data, fit_fitness_difference_params, plasmid=plasmid)

            stan_fit = stan_model.sample(
                data=stan_data,
                iter_sampling=iter_sampling,
                iter_warmup=iter_warmup,
                inits=stan_init,
                chains=chains,
                adapt_delta=adapt_delta,
                show_progress=show_progress,
                output_dir=stan_output_dir,
            )

            if re_stan_on_rhat:
                rhat_params = stan_utils.check_rhat_by_params(
                    stan_fit, rhat_cutoff=rhat_cutoff, stan_parameters=key_params
                )
                if len(rhat_params) > 0:
                    print(f'Re-running Stan fit because r_hat > {rhat_cutoff} for: {rhat_params}')
                    stan_fit = stan_model.sample(
                        data=stan_data,
                        iter_sampling=iter_sampling * 10,
                        iter_warmup=iter_warmup * 10,
                        inits=stan_init,
                        chains=chains,
                        adapt_delta=adapt_delta,
                        show_progress=show_progress,
                        output_dir=stan_output_dir,
                    )

                    rhat_params = stan_utils.check_rhat_by_params(
                        stan_fit, rhat_cutoff=rhat_cutoff, stan_parameters=key_params
                    )
                    if len(rhat_params) > 0:
                        print(f'    Still r_hat > {rhat_cutoff} for: {rhat_params}')
                    else:
                        print(f'    All parameters now have r_hat < {rhat_cutoff}')

            if return_fit:
                return stan_fit

            stan_samples_arr = np.array([stan_fit.stan_variable(key) for key in params_list])
            stan_popt = np.array([np.mean(s) for s in stan_samples_arr])
            stan_pcov = np.cov(stan_samples_arr, rowvar=True)
            stan_resid = np.mean(stan_fit.stan_variable("rms_resid"))

            stan_quant_arr = stan_samples_arr[:quantile_params_dim]
            stan_samples_out = rng.choice(stan_quant_arr, size=32, replace=False, axis=1, shuffle=False)
            stan_quantiles = np.array([np.quantile(x, quantile_list) for x in stan_quant_arr])

            g0_samples = 10 ** stan_samples_arr[log_g0_ind]
            hill_on_at_zero_prob = len(g0_samples[g0_samples > wild_type_ginf / 4]) / len(g0_samples)

            if len(lig_list) == 1:
                g_ratio_samples = stan_samples_arr[log_ginf_g0_ind]
                hill_invert_prob = len(g_ratio_samples[g_ratio_samples < 0]) / len(g_ratio_samples)
            elif len(ligand_list) == 2:
                g_ratio_samples = [stan_samples_arr[k] for k in [log_ginf_g0_ind_1, log_ginf_g0_ind_2]]
                hill_invert_prob = [len(s[s < 0]) / len(s) for s in g_ratio_samples]
            elif len(ligand_list) == 3:
                g_ratio_samples = [stan_samples_arr[k] for k in [log_ginf_g0_ind_1, log_ginf_g0_ind_2, log_ginf_g0_ind_3]]
                hill_invert_prob = [len(s[s < 0]) / len(s) for s in g_ratio_samples]

        except Exception as err:
            stan_popt = np.full((params_dim), np.nan)
            stan_pcov = np.full((params_dim, params_dim), np.nan)
            stan_resid = np.nan
            stan_samples_out = np.full((quantile_params_dim, 32), np.nan)
            stan_quantiles = np.full((quantile_params_dim, quantile_dim), np.nan)
            hill_on_at_zero_prob = np.nan
            hill_invert_prob = np.nan if len(lig_list) == 1 else [np.nan] * len(lig_list)

            print(f"Error during Stan fitting for index {st_index}: {err}", sys.exc_info()[0])
            tb_str = ''.join(traceback.format_exception(None, err, err.__traceback__))
            print(tb_str)

        return (stan_popt, stan_pcov, stan_resid, stan_samples_out, stan_quantiles, hill_invert_prob, hill_on_at_zero_prob, st_index)

    # ----------------------------
    # Run fits
    # ----------------------------
    if refit_indexes is None:
        print(f'Running Stan fits for all rows in dataframe, number of rows: {len(barcode_frame)}')
        fit_list = [stan_fit_row(row, index, ligand_list) for (index, row) in barcode_frame.iterrows()]
    else:
        # If caller wants the raw fit object, we return early and DO NOT persist.
        if return_fit:
            row_to_fit = barcode_frame.loc[refit_indexes[0]]
            return stan_fit_row(row_to_fit, refit_indexes[0], ligand_list, return_fit=True)

        print(f'Running Stan fits for selected rows in dataframe, number of rows: {len(refit_indexes)}')
        print(f'    selected rows: {refit_indexes}')
        row_list = [barcode_frame.loc[index] for index in refit_indexes]
        fit_list = [stan_fit_row(row, index, ligand_list) for (index, row) in zip(refit_indexes, row_list)]

    popt_list, pcov_list, residuals_list = [], [], []
    samples_out_list, quantiles_list = [], []
    invert_prob_list, on_at_zero_prob_list = [], []
    index_list = []

    for item in fit_list:
        stan_popt, stan_pcov, stan_resid, stan_samples_out, stan_quantiles, hill_invert_prob, hill_on_at_zero_prob, ind = item
        popt_list.append(stan_popt)
        pcov_list.append(stan_pcov)
        residuals_list.append(stan_resid)
        samples_out_list.append(stan_samples_out)
        quantiles_list.append(stan_quantiles)
        invert_prob_list.append(hill_invert_prob)
        on_at_zero_prob_list.append(hill_on_at_zero_prob)
        index_list.append(ind)

    perr_list = [np.sqrt(np.diagonal(x)) for x in pcov_list]

    # ----------------------------
    # Write results back into dataframe (UNCHANGED)
    # ----------------------------
    if refit_indexes is None:
        if index_list == list(barcode_frame.index):
            print("index lists match")
        else:
            print("Warning!! index lists do not match!")

        for param, v, err in zip(params_list, np.transpose(popt_list), np.transpose(perr_list)):
            col_name = param
            for i, lig in enumerate(ligand_list):
                col_name = col_name.replace(f"_{i+1}", f"_{lig}")
            barcode_frame[col_name] = v
            barcode_frame[f"{col_name}_err"] = err

        for param, q, samp in zip(
            quantile_params_list,
            np.array(quantiles_list).transpose([1, 0, 2]),
            np.array(samples_out_list).transpose([1, 0, 2]),
        ):
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
        if index_list == list(refit_indexes):
            print("index lists match")
        else:
            print("Warning!! index lists do not match!")

        for param, v, err in zip(params_list, np.transpose(popt_list), np.transpose(perr_list)):
            col_name = param
            for i, lig in enumerate(ligand_list):
                col_name = col_name.replace(f"_{i+1}", f"_{lig}")
            barcode_frame.loc[index_list, col_name] = v
            barcode_frame.loc[index_list, f"{col_name}_err"] = err

        for param, q, samp in zip(
            quantile_params_list,
            np.array(quantiles_list).transpose([1, 0, 2]),
            np.array(samples_out_list).transpose([1, 0, 2]),
        ):
            col_name = param
            for i, lig in enumerate(ligand_list):
                col_name = col_name.replace(f"_{i+1}", f"_{lig}")
            for ind, new_q, new_samp in zip(index_list, list(q), list(samp)):
                barcode_frame.at[ind, f"{col_name}_quantiles"] = new_q
                barcode_frame.at[ind, f"{col_name}_samples"] = new_samp

        for i, lig in enumerate(ligand_list):
            barcode_frame.loc[index_list, f"hill_invert_prob_{lig}"] = np.array(invert_prob_list).transpose()[i]

        barcode_frame.loc[index_list, "sensor_params_cov_all"] = pcov_list
        barcode_frame.loc[index_list, "hill_on_at_zero_prob"] = on_at_zero_prob_list
        barcode_frame.loc[index_list, "sensor_rms_residuals"] = residuals_list

    out_df = barcode_frame

    # ----------------------------
    # NEW: Persist state (table only)
    # ----------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "stan_fitness_difference_curves"
        table_name = f"{tag}_barcode_frame"

        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "tables": [table_name],
            "initial": initial,
            "iterations": iterations,
            "chains": chains,
            "adapt_delta": adapt_delta,
            "re_stan_on_rhat": re_stan_on_rhat,
            "rhat_cutoff": rhat_cutoff,
            "log_x_max": log_x_max,
            "stan_output_dir": stan_output_dir,
            "refit_indexes": None if refit_indexes is None else list(refit_indexes),
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={table_name: out_df},
        )

    return out_df



def stan_GP_curves(
    manifest,
    data,
    stan_GP_model='gp-hill-nomean-constrained.stan',
    adapt_delta=0.9,
    iterations=1000,
    iter_warmup=None,
    iter_sampling=None,
    show_progress=False,
    chains=4,
    stan_output_dir=None,
    auto_save=True,
    overwrite=False,
    refit_indexes=None,
    return_fit=False,
    initial=None,
    re_stan_on_rhat=True,
    rhat_cutoff=1.05,
    *,
    state_dir=None,
    state_tag=None,
):
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.disabled = True

    plasmid = manifest.get('plasmid')
    fit_fitness_difference_params = manifest.get('fit_fitness_difference_params')

    if iter_warmup is None:
        iter_warmup = int(iterations / 2)
    if iter_sampling is None:
        iter_sampling = int(iterations / 2)

    if initial is None:
        initial = get_default_initial(manifest)

    print(f"Using Stan to fit to fitness curves with GP model for {manifest.get('experiment')}")
    print(f"  Using fitness parameters for {plasmid}")
    print(f"      {fit_fitness_difference_params}")
    print("      Method version from 2022-11-25")

    barcode_frame = data
    fitness_columns_setup = get_fitness_columns_setup(manifest, data, plot_initials=[initial])
    ligand_list = manifest.get('ligand_list')
    antibiotic_conc_list = manifest.get('antibiotic_conc_list')

    if fitness_columns_setup[0]:
        old_style_columns, x, linthresh, fit_plot_colors = fitness_columns_setup
    else:
        old_style_columns, linthresh, fit_plot_colors, plot_df = fitness_columns_setup

    # ----------------------------
    # Stan model / params
    # ----------------------------
    if len(ligand_list) == 1:
        stan_GP_model = 'gp-hill-nomean-constrained.stan'
        params_list = [
            'low_fitness_high_tet', 'mid_g_high_tet', 'fitness_n_high_tet',
            'log_rho', 'log_alpha', 'log_sigma'
        ]
        g_arr_list = ['constr_log_g']
        g_ratio_arr_list = ['log_g_ratio']
        dg_arr_list = ['dlog_g']
        f_arr_list = ['mean_y']
        x_dim = 12

    elif len(ligand_list) == 2:
        stan_GP_model = 'gp-hill-nomean-constrained.two-lig.two-tet.stan'
        params_list = [
            'low_fitness_low_tet', 'mid_g_low_tet', 'fitness_n_low_tet',
            'low_fitness_high_tet', 'mid_g_high_tet', 'fitness_n_high_tet',
            'log_rho', 'log_alpha', 'log_sigma'
        ]
        g_arr_list = [f'log_g_{i}' for i in [1, 2]]
        g_ratio_arr_list = [f'log_g_ratio_{i}' for i in [1, 2]]
        dg_arr_list = [f'dlog_g_{i}' for i in [1, 2]]
        f_arr_list = [
            'y_1_out_low_tet', 'y_1_out_high_tet',
            'y_2_out_low_tet', 'y_2_out_high_tet'
        ]
        x_dim = 6

    elif len(ligand_list) == 3:
        if plasmid == 'pRamR':
            stan_GP_model = 'gp-hill-nomean-constrained.three-ligand.inverted.stan'
            params_list = ['high_fitness', 'mid_g', 'fitness_n', 'log_rho', 'log_alpha', 'log_sigma']
        else:
            stan_GP_model = 'gp-hill-nomean-constrained.three-ligand.stan'
            params_list = ['low_fitness', 'mid_g', 'fitness_n', 'log_rho', 'log_alpha', 'log_sigma']

        g_arr_list = [f'log_g_{i}' for i in [1, 2, 3]]
        g_ratio_arr_list = [f'log_g_ratio_{i}' for i in [1, 2, 3]]
        dg_arr_list = [f'dlog_g_{i}' for i in [1, 2, 3]]
        f_arr_list = [f'y_{i}_out' for i in [1, 2, 3]]
        x_dim = 7

    key_params = params_list + g_arr_list + g_ratio_arr_list + dg_arr_list + f_arr_list
    params_dim = len(params_list)

    print(f'    Using model from file: {stan_GP_model}')
    stan_model = stan_utils.compile_model(stan_GP_model)

    rng = np.random.default_rng()

    # ----------------------------
    # Fit runner
    # ----------------------------
    def stan_fit_row(st_row, st_index, lig_list, return_fit=False):
        ret = {'st_index': st_index}
        print(datetime.datetime.now(), f"fitting row {st_index}")

        try:
            stan_data = bs_frame_stan_data(
                manifest, data, st_row, initial=initial, is_gp_model=True
            )

            stan_init = init_stan_GP_fit(
                fit_fitness_difference_params,
                single_tet=len(antibiotic_conc_list) == 2,
                single_ligand=len(lig_list) == 1,
                plasmid=plasmid,
            )

            stan_fit = stan_model.sample(
                data=stan_data,
                iter_sampling=iter_sampling,
                iter_warmup=iter_warmup,
                inits=stan_init,
                chains=chains,
                adapt_delta=adapt_delta,
                show_progress=show_progress,
                output_dir=stan_output_dir,
            )

            if re_stan_on_rhat:
                rhat = stan_utils.check_rhat_by_params(
                    stan_fit, rhat_cutoff=rhat_cutoff, stan_parameters=key_params
                )
                if rhat:
                    stan_fit = stan_model.sample(
                        data=stan_data,
                        iter_sampling=iter_sampling * 10,
                        iter_warmup=iter_warmup * 10,
                        inits=stan_init,
                        chains=chains,
                        adapt_delta=adapt_delta,
                        show_progress=show_progress,
                        output_dir=stan_output_dir,
                    )

            if return_fit:
                return stan_fit

            ret['stan_popt'] = np.array([np.mean(stan_fit.stan_variable(p)) for p in params_list])
            ret['stan_pcov'] = np.cov([stan_fit.stan_variable(p) for p in params_list], rowvar=True)
            ret['stan_resid'] = np.mean(stan_fit.stan_variable("rms_resid"))

            ret['stan_g'] = [
                np.quantile(stan_fit.stan_variable(p), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
                for p in g_arr_list
            ]

        except Exception as err:
            ret['stan_popt'] = np.full(params_dim, np.nan)
            ret['stan_pcov'] = np.full((params_dim, params_dim), np.nan)
            ret['stan_resid'] = np.nan
            ret['stan_g'] = [np.full((5, x_dim), np.nan) for _ in g_arr_list]
            print("Stan GP error:", err)

        return ret

    # ----------------------------
    # Run fits
    # ----------------------------
    if refit_indexes is None:
        fit_list = [
            stan_fit_row(row, idx, ligand_list)
            for idx, row in barcode_frame.iterrows()
        ]
    else:
        if return_fit:
            return stan_fit_row(
                barcode_frame.loc[refit_indexes[0]],
                refit_indexes[0],
                ligand_list,
                return_fit=True,
            )
        fit_list = [
            stan_fit_row(barcode_frame.loc[idx], idx, ligand_list)
            for idx in refit_indexes
        ]

    # ----------------------------
    # Write results back
    # ----------------------------
    for k in fit_list[0]:
        if k == 'st_index':
            continue
        barcode_frame[k] = [x[k] for x in fit_list]

    out_df = barcode_frame

    # ----------------------------
    # Persist state (table only)
    # ----------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "stan_GP_curves"
        table_name = f"{tag}_barcode_frame"

        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "tables": [table_name],
            "plasmid": plasmid,
            "stan_model": stan_GP_model,
            "chains": chains,
            "iterations": iterations,
            "adapt_delta": adapt_delta,
            "re_stan_on_rhat": re_stan_on_rhat,
            "rhat_cutoff": rhat_cutoff,
            "refit_indexes": None if refit_indexes is None else list(refit_indexes),
            "stan_output_dir": stan_output_dir,
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={table_name: out_df},
        )

    return out_df
        



def merge_barcodes(
    manifest,
    data,
    small_bc_index_list,
    big_bc_index,
    auto_save=True,
    overwrite=False,
    *,
    state_dir=None,
    state_tag=None,
):
    """
    Merge multiple barcode rows into a single barcode row by summing read counts.
    """

    print(f"Merging {small_bc_index_list} into {big_bc_index}")
    print(f"Remember to run trim_and_sum_barcodes() and set_sample_plate_map()")
    print(f"    after all merges are completed!!!")

    barcode_frame = data.copy()

    if 'was_merged' not in barcode_frame.columns:
        barcode_frame['was_merged'] = False

    merge_ind_list = [big_bc_index] + list(small_bc_index_list)
    merge_df = barcode_frame.loc[merge_ind_list]

    new_row = barcode_frame.loc[big_bc_index].copy()

    for w in fitness_utils.wells():
        new_row[w] = merge_df[w].sum()

    new_row['total_counts'] = merge_df['total_counts'].sum()
    new_row['was_merged'] = True

    # Reset nearest-neighbor distance (invalid after merge)
    if 'nearest_neighbor_dist' in new_row:
        new_row['nearest_neighbor_dist'] = np.nan

    barcode_frame.loc[big_bc_index] = new_row

    for bc in small_bc_index_list:
        barcode_frame.drop(bc, inplace=True)

    out_df = barcode_frame

    # ----------------------------
    # NEW: Persist state (table only)
    # ----------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "merge_barcodes"
        table_name = f"{tag}_barcode_frame"

        manifest.setdefault("outputs", {})
        manifest["outputs"][tag] = {
            "tables": [table_name],
            "big_bc_index": big_bc_index,
            "small_bc_index_list": list(small_bc_index_list),
            "n_merged": len(small_bc_index_list),
        }

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={table_name: out_df},
        )

    return out_df
    
        
def plot_count_hist(manifest, data, hist_bin_max=None, num_bins=50, save_plots=False, pdf_file=None, quantile_for_qc_ratio=0.99):
    
    barcode_frame = data
    
    # Turn interactive plotting on or off depending on show_plots
    plt.ion()
    os.chdir(manifest.get('data_directory'))
    if save_plots:
        if pdf_file is None:
            pdf_file = 'barcode histogram plot.pdf'
        pdf = PdfPages(pdf_file)
    
    if hist_bin_max is None:
        hist_bin_max = np.quantile(barcode_frame.total_counts, .99)
    
    #Plot histogram of Barcode counts to enable decision about threshold
    plt.rcParams["figure.figsize"] = [10, 3]
    fig, axs = plt.subplots(1, 2)
    bins = np.linspace(-0.5, hist_bin_max + 0.5, num_bins)
    x = barcode_frame['total_counts_plate_2'].values
    count_mode = stats.mode(x).mode
    count_quantile = np.quantile(x, quantile_for_qc_ratio)
    print(f'plate 2 count mode: {count_mode}')
    print(f'plate 2 {quantile_for_qc_ratio} quantile: {count_quantile}')
    print(f'plate 2 QC ratio: {count_quantile/count_mode:.2f}')
    for ax in axs.flatten():
        ax.hist(barcode_frame['total_counts'], bins=bins, label='All Time Points', alpha=0.7);
        ax.hist(x, bins=bins, label='Time Point 1', alpha=0.7);
        ax.set_xlabel('Barcode Count')
        ax.set_ylabel('Number of Variants')
        #ax.tick_params(labelsize=16);
    axs[0].hist(barcode_frame['total_counts'], bins=bins, histtype='step', cumulative=-1);
    axs[0].set_yscale('log');
    axs[1].set_yscale('log');
    axs[1].set_xlim(0,hist_bin_max/3);
    axs[0].legend(loc='lower left', bbox_to_anchor= (0.05, 1.02), ncol=2, borderaxespad=0, frameon=True)
        
    if save_plots:
        pdf.savefig()
            
    if save_plots:
        pdf.close()
    
def plot_read_counts(manifest, data, save_plots=False, pdf_file=None, vmin=0):
    
    barcode_frame = data
    
    # Turn interactive plotting on or off depending on show_plots
    plt.ion()
    os.chdir(manifest.get('data_directory'))
    if save_plots:
        if pdf_file is None:
            pdf_file = 'barcode read count plots.pdf'
        pdf = PdfPages(pdf_file)

    BC_totals = []
    index_list = []
    for i, w in enumerate(fitness_utils.wells()):
        BC_totals.append(barcode_frame[w].sum())
        index_list.append(i+1)
    
    BC_total_arr = []
    for r in fitness_utils.rows():
        subarr = []
        for c in fitness_utils.columns():
            subarr.append(barcode_frame[r + str(c)].sum())
        BC_total_arr.append(subarr)

    #Plot barcode read counts across plate
    plt.rcParams["figure.figsize"] = [8, 5]
    fig, ax = plt.subplots()

    r12 = np.asarray(np.split(np.asarray(BC_totals), 8)).transpose().flatten()
    for i, split_count in enumerate(np.split(r12, 4)):
        geo_std = np.exp(np.std(np.log(split_count)))
        geo_max = np.exp(np.ptp(np.log(split_count)))
        print(f'Time point {i+1}, geometric stdev: {geo_std:.2f}-fold')
        print(f'            maximum differnce: {geo_max:.2f}-fold')

    ax.scatter(index_list, r12, c=plot_colors96(), s=50);
    for i in range(13):
        ax.plot([i*8+0.5, i*8+0.5],[min(BC_totals), max(BC_totals)], color='gray');
    ax.set_yscale('log');

    ax.set_xlim(0,97);
    ax.set_xlabel('Sample Number', size=20)
    ax.set_ylabel('Total Reads per Sample', size=20);
    ax.tick_params(labelsize=16);
    
    fig, ax = plt.subplots()
    ax.matshow(BC_total_arr[::-1], cmap="inferno", vmin=vmin);
    ax.grid(visible=False);
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_xticklabels([i+1 for i in range(12)], size=16);
        ax.set_xticks([i for i in range(12)]);
        ax.set_yticklabels([ r + " " for r in fitness_utils.rows()[::-1] ], size=16);
        ax.set_yticks([i for i in range(8)]);
    ax.set_ylim(-0.5, 7.5);
    ax.tick_params(length=0);
    
    if save_plots:
        pdf.savefig()
            
    if save_plots:
        pdf.close()
        
def plot_read_fractions(manifest, data,
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

    os.chdir(manifest.get('data_directory'))
    
    if plot_range is None:
        f_data = data.iloc[:num_to_plot]
    else:
        f_data = data.loc[plot_range[0]:plot_range[1]]
    
    #Plot read fraction across all samples for first several barcodes
    plt.rcParams["figure.figsize"] = [plot_size,6*len(f_data)*plot_size/16]
    fig, axs = plt.subplots(len(f_data), 1)
            
    sample_plate_map = manifest.get('sample_plate_map')
    
    tet_list = np.unique(sample_plate_map.antibiotic_conc)
    if plot_fraction:
        plot_param = "fraction_"
    else:
        plot_param = ""
    for (index, row), ax in zip(f_data.iterrows(), axs):
        y_for_scale = []
        for marker, tet in zip(['o', '<', '>', '^', 'v'], tet_list):
            y = []
            x = []
            c = []
            for i, w in enumerate(fitness_utils.wells_by_column()):
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
        #ax.set_ylim(0.5*min(y_for_scale), 2*max(y));
        ax.set_yscale("log")
        barcode_str = str(index) + ', '
        if row['RS_name'] != "": barcode_str += row['RS_name'] + ", "
        barcode_str += row['forward_BC'] + ', ' + row['reverse_BC']
        ax.text(x=0.05, y=1.02, s=barcode_str, horizontalalignment='left', verticalalignment='bottom',
                    transform=ax.transAxes, fontsize=plot_size/1.5)
    
        ylim = ax.get_ylim()
        ax.set_ylim(ylim)
        for i in range(13):
            ax.plot([i*8+0.5, i*8+0.5],ylim, color='gray');
    if plot_fraction:
        axs[0].set_title("Read Fraction Per Barcode", fontsize=2*plot_size);
    else:
        axs[0].set_title("Read Count Per Barcode", fontsize=2*plot_size);
    if save_plots:
        pdf.savefig()

    if save_plots:
        pdf.close()
        
def plot_stdev(manifest, data,
                save_plots=False,
                count_cutoff=500,
                experiment=None,
                includeChimeras=False,
                reverse_well_order=False):
    
    if experiment is None:
        experiment = manifest.get('experiment')

    # Turn interactive plotting on or off depending on show_plots
    plt.ion()
    os.chdir(manifest.get('data_directory'))
    
    if save_plots:
        pdf_file = 'barcode read standard deviation plots.pdf'
        pdf = PdfPages(pdf_file)
        
    #Plot Barcode fraction for each well in time point 1 vs. mean fraction in time point 1
    plt.rcParams["figure.figsize"] = [16,16]
    fig, axs = plt.subplots(2, 2)
    
    f_data = data[data["total_counts"]>count_cutoff]
    if (not includeChimeras) and ("isChimera" in f_data.columns):
        f_data = f_data[f_data["isChimera"] == False]
        
    f_x = f_data['fraction_total_p2']
    f_x_min = f_data[f_data['fraction_total_p2']>0]['fraction_total_p2'].min()
    wells_to_plot = fitness_utils.wells_by_column()[:24]
    if reverse_well_order:
        wells_to_plot = wells_to_plot[::-1]
    per_well_log_std = []
    per_well_log_mu = []
    for i, w in enumerate(wells_to_plot):
        c = [(plot_colors()*8)[i]]*len(f_data)
        f_y = f_data['fraction_' + w]
        for ax in axs.flatten()[:2]:
            ax.scatter(f_x, f_y, c=c)
        
        x = f_x[(~np.isnan(f_y))&(f_y>0)]
        y = f_y[(~np.isnan(f_y))&(f_y>0)]
        per_well_log_std.append(np.std(np.log(y/x)))
        per_well_log_mu.append(np.mean(np.log(y/x)))
        for ax in axs.flatten()[2:4]:
            ax.scatter(f_x, (f_y - f_x)*100, c=c)
            
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
    #Plot mean and std of log ratio for each well
    plt.rcParams["figure.figsize"] = [16,6]
    fig, axs = plt.subplots(2, 1)
    for ax, y, lab in zip(axs, [per_well_log_mu, per_well_log_std], ['mean', 'std']):
        df_plot = pd.DataFrame({'well':wells_to_plot, lab:y})
        sns.barplot(ax=ax, data=df_plot, x="well", y=lab)
    
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
    
    axs[0].plot(x, y, "o", ms=5, label = experiment, alpha=0.1);
    axs[0].plot(x, err_est, c="darkgreen");
    axs[0].set_ylabel('Stdev(barcode fraction per sample)', size=20);
    axs[1].plot(x, y/x, "o", ms=5, label = experiment, alpha=0.1);
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

def plot_fitness_curves(manifest, data,
                        save_plots=False,
                        plot_range=None,
                        include_ref_seqs=False,
                        includeChimeras=False,
                        ylim=None,
                        plot_size=[6, 4],
                        fontsize=13,
                        ax_label_size=14,
                        show_bc_str=False,
                        real_fitness_units=False,
                        plot_initials=None,
                        plot_slope_not_fitness=False,
                        plot_stan_data=[False, False],
                        plot_w_ramr_correction=[True, False]):
    
    if plot_initials is None:
        if manifest.get('plasmid') == 'pVER':
            plot_initials=["b", "e"]
        elif manifest.get('plasmid') == 'pRamR':
            plot_initials=["sp01", "sp02"]
        elif manifest.get('plasmid') == 'pCymR':
            plot_initials=["sp09", "rs20"]
        elif manifest.get('plasmid') == 'Align-TF':
            plot_initials=["ramr", "laci"]
        elif manifest.get('plasmid') in ['Align-Protease', 'Align-T7RNAP_1']:
            plot_initials=["nrm02", "nrm03"]
    
    if plot_range is None:
        barcode_frame = data
    else:
        barcode_frame = data.loc[plot_range[0]:plot_range[1]]
    
    ligand_list = manifest.get('ligand_list')
        
    if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
        barcode_frame = barcode_frame[barcode_frame["isChimera"] == False]
        
    if include_ref_seqs:
        RS_count_frame = data[data["RS_name"]!=""]
        barcode_frame = pd.concat([barcode_frame, RS_count_frame])
        
    if real_fitness_units:
        fit_scale = fitness_utils.fitness_scale()
        fit_units = '1/h'
    else:
        fit_scale = 1
        fit_units = 'log(10)/plate'
        
    # Turn interactive plotting on or off depending on show_plots
    plt.ion()
    
    if save_plots:
        os.chdir(manifest.get('data_directory'))
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
    
    fitness_columns_setup = get_fitness_columns_setup(manifest, data, plot_initials=plot_initials)
    if fitness_columns_setup[0]:
        old_style_plots, x, linthresh, fit_plot_colors = fitness_columns_setup
    else:
        old_style_plots, linthresh, fit_plot_colors, plot_df = fitness_columns_setup
    
    antibiotic_conc_list = manifest.get('antibiotic_conc_list')
    
    for (index, row), ax in zip(barcode_frame.iterrows(), axs): # iterate over barcodes
        if manifest.get('plasmid') == 'Align-TF':
            tf = row.transcription_factor
        for initial, fill_style, plot_st, plot_corr in zip(plot_initials, ['full', 'none', 'right', 'left'], plot_stan_data, plot_w_ramr_correction):
            if old_style_plots:
                for tet, color in zip(antibiotic_conc_list, fit_plot_colors):
                    y = row[f"fitness_{tet}_estimate_{initial}"]*fit_scale
                    s = row[f"fitness_{tet}_err_{initial}"]*fit_scale
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ax.errorbar(x, y, s, marker='o', ms=8, color=color, fillstyle=fill_style)
            else:
                for tet, marker, antibiotic_color in zip(antibiotic_conc_list, ['o', '<', '>', '^', 'v'], fit_plot_colors):
                    for j, (lig, lig_color) in enumerate(zip(ligand_list, fit_plot_colors)):
                        df = plot_df
                        df = df[(df.ligand==lig)|(df.ligand=='none')]
                        df = df[df.antibiotic_conc==tet]
                        if (manifest.get('plasmid') == 'Align-TF') and (tf != 'all'):
                            df = df[df.transcription_factor==tf]
                        x = df[lig]
                        if plot_slope_not_fitness:
                            y = [row[f"fit_slope_S{i}_{initial}"]*fit_scale for i in df.sample_id]
                            s = [row[f"fit_slope_S{i}_err_{initial}"]*fit_scale for i in df.sample_id]
                        else:
                            if plot_st and (tet > 0):
                                stan_data = bs_frame_stan_data(manifest, data, row, initial=initial, apply_ramr_correction=plot_corr)
                                if len(antibiotic_conc_list) == 2:
                                    # Single non-zero antibiotic concentration
                                    if 'y_0' in stan_data:
                                        st_y_0 = list(stan_data[f'y_0'])
                                        st_y_0_err = list(stan_data[f'y_0_err'])
                                        x = np.array([0]*len(st_y_0) + list(stan_data[f'x_{j+1}']))
                                        y = np.array(st_y_0 + list(stan_data[f'y_{j+1}'])) + stan_data['y_ref']
                                        s = np.array(st_y_0_err + list(stan_data[f'y_{j+1}_err']))
                                    else:
                                        x = stan_data['x']
                                        y = stan_data['y']
                                        s = stan_data['y_err']
                                elif len(antibiotic_conc_list) == 3:
                                    # Two non-zero antibiotic concentrations
                                    if tet == antibiotic_conc_list[1]:
                                        tet_str = 'low'
                                        st_y_0 = [stan_data[f'y_0_low_tet']]
                                        st_y_0_err = [stan_data[f'y_0_low_tet_err']]
                                    else:
                                        tet_str = 'high'
                                        st_y_0 = []
                                        st_y_0_err = []
                                    x = np.array([0]*len(st_y_0) + list(stan_data[f'x_{j+1}']))
                                    y = np.array(st_y_0 + list(stan_data[f'y_{j+1}_{tet_str}_tet'])) + stan_data['y_ref']
                                    s = np.array(st_y_0_err + list(stan_data[f'y_{j+1}_{tet_str}_tet_err']))
                            else:
                                y = [row[f"fitness_S{i}_{initial}"]*fit_scale for i in df.sample_id]
                                s = [row[f"fitness_S{i}_err_{initial}"]*fit_scale for i in df.sample_id]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if manifest.get('plasmid') == 'Align-TF':
                                color = antibiotic_color
                            else:
                                color = lig_color
                            ax.errorbar(x, y, s, marker=marker, ms=8, color=color, fillstyle=fill_style)
        
            if initial == plot_initials[0]:
                barcode_str = str(index) + ': '
                barcode_str += format(row[f'total_counts'], ",") + "; "
                barcode_str += row['RS_name']
                if 'cytom_variant' in barcode_frame.columns:
                    barcode_str += ', ' + row['cytom_variant']
                if show_bc_str:
                    barcode_str += ": " + row['forward_BC'] + ",\n"
                    barcode_str += row['reverse_BC'] + " "
                    fontfamily = "Courier New"
                else:
                    fontfamily = None
                ax.text(x=1, y=1.1, s=barcode_str, horizontalalignment='right', verticalalignment='top',
                        transform=ax.transAxes, fontsize=fontsize, fontfamily=fontfamily)
                ax.set_xscale('symlog', linthresh=linthresh)
                if (manifest.get('plasmid') == 'Align-TF') and (tf != 'all'):
                    x_lab = align_ligand_from_tf(tf)
                else:
                    x_lab = '], ['.join(ligand_list)
                ax.set_xlabel(f'[{x_lab}] (umol/L)', size=ax_label_size)
                if plot_slope_not_fitness:
                    ax.set_ylabel(f'Log slope relative to {plot_initials}', size=ax_label_size)
                else:
                    ax.set_ylabel(f'Growth Rate ({fit_units})', size=ax_label_size)
                ax.tick_params(labelsize=ax_label_size-2);
                if ylim is not None:
                    ax.set_ylim(ylim);
        
    if save_plots:
        pdf.savefig()

    if save_plots:
        pdf.close()
        
    return fig, axs_grid
    
def get_fitness_columns_setup(manifest, data, plot_initials):
    barcode_frame = data
                    
    # old_style_plots indicates whether to use the old style column headings (i.e., f"fitness_{high_tet}_estimate_{initial}")
    #     or the new style (i.e., f"fitness_S{i}_{initial}"
    # The new style is preferred, so will be used if both are possible
    old_style_plots = False
    if manifest.get('plasmid') == 'pVER':
        for initial in plot_initials:
            for i  in range(1, 25):
                c = f"fitness_S{i}_{initial}"
                old_style_plots = old_style_plots or (c not in barcode_frame.columns)
    if old_style_plots:
        print("Using old style column headings")
    #else:
    #    print("Using new style column headings")
    
    fit_plot_colors = sns.color_palette()
    
    if old_style_plots:
        x = np.array(manifest.get('inducer_conc_list'))
        linthresh = min(x[x>0])
        
        return old_style_plots, x, linthresh, fit_plot_colors
    else:
        sample_plate_map = manifest.get('sample_plate_map')
        lig_list = list(np.unique(sample_plate_map.ligand))
        if 'none' in lig_list:
            lig_list.remove('none')
        
        plot_df = sample_plate_map
        plot_df = plot_df[plot_df.growth_plate==2].sort_values(by=lig_list)
        
        x_list = np.array([np.array(plot_df[x]) for x in lig_list]).flatten()
        linthresh = min(x_list[x_list>0])
        
        return old_style_plots, linthresh, fit_plot_colors, plot_df


def plot_dose_response(manifest, data,
                        plot_index,
                        show_GP=True,
                        log_g_scale=True,
                        box_size=6,
                        ax=None):
    
    plot_row = data.loc[plot_index]
    
    antibiotic_conc_list = manifest.get('antibiotic_conc_list')
    ligand_list = manifest.get('ligand_list')
    
    if ax is None:
        plt.rcParams["figure.figsize"] = [box_size, box_size*2/3]
        fig, axg = plt.subplots()
    else:
        axg = ax
    
    fit_fitness_difference_params = manifest.get('fit_fitness_difference_params')
    
    if manifest.get('plasmid') == 'pVER':
        plot_initials = ['b', 'e']
    elif manifest.get('plasmid') == 'pRamR':
        plot_initials = ['sp01']
    elif manifest.get('plasmid') == 'pCymR':
        plot_initials = ['sp09', 'rs20']
    
    fitness_columns_setup = get_fitness_columns_setup(manifest, data, plot_initials=plot_initials)
    old_style_plots, linthresh, fit_plot_colors, plot_df = fitness_columns_setup
    
    if manifest.get('plasmid') in ['pVER', 'pCymR']:
        def fit_funct(x, log_g0, log_ginf, log_ec50, nx, low_fitness, mid_g, fitness_n):
            return double_hill_funct(x, 10**log_g0, 10**log_ginf, 10**log_ec50, nx,
                                        low_fitness, 0, mid_g, fitness_n)
    elif manifest.get('plasmid') == 'pRamR':
        def fit_funct(x, log_g0, log_ginf, log_ec50, nx, high_fitness, mid_g, fitness_n):
            return double_hill_funct(x, 10**log_g0, 10**log_ginf, 10**log_ec50, nx,
                                        0, high_fitness, mid_g, fitness_n)
    

    fill_alpha = 0.2
    
    tet_level_list = ['high'] if len(antibiotic_conc_list)==2 else ['low', 'high']
    for lig, color in zip(ligand_list, fit_plot_colors):
        stan_g = 10**plot_row[f"GP_log_g_{lig}"]
        stan_dg = plot_row[f"GP_dlog_g_{lig}"]
        
        df = plot_df
        df = df[(df.ligand==lig)|(df.ligand=='none')]
        df = df[df.with_tet]
        x = np.unique(df[lig])
    
        axg.plot(x, stan_g[2], '--', color=color, lw=3, label=lig)
            
        axg.set_ylabel('Output (MEF)', size=14)
        axg.set_xlabel('[ligand]', size=14)
        axg.tick_params(labelsize=12);
        if log_g_scale: axg.set_yscale("log")
        
        for i in range(1,3):
            axg.fill_between(x, stan_g[2-i], stan_g[2+i], alpha=fill_alpha, color=color);
        axg.legend(loc='upper left', bbox_to_anchor= (1.03, 0.97), ncol=1, borderaxespad=0, frameon=True, fontsize=10)
        
        # Also plot Hill fit result for g
        x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
        x_fit = np.insert(x_fit, 0, 0)
        params_list = ['log_g0', f'log_ginf_{lig}', f'log_ec50_{lig}', f'sensor_n_{lig}']
        hill_params = [10**plot_row[p] for p in params_list[:-1]] + [plot_row[params_list[-1]]]
        y_fit = hill_funct(x_fit, *hill_params)
        axg.plot(x_fit, y_fit, c=color, zorder=1000)
    
    axg.set_xscale('symlog', linthresh=linthresh)
    
    if ax is None:
        return fig, axg


def plot_fitness_and_difference_curves(manifest, data,
                        save_plots=False,
                        plot_range=None,
                        include_ref_seqs=False,
                        includeChimeras=False,
                        ylim = None,
                        show_fits=True,
                        show_GP=False,
                        fitness_scale_factor=1,
                        show_hill_fit=False,
                        log_g_scale=False,
                        box_size=6,
                        show_bc_str=False,
                        show_mut_codes=True,
                        show_variant=False,
                        plot_initials=None):
    
    if plot_initials is None:
        if manifest.get('plasmid') in ['pVER', 'pCymR']:
            plot_initials = [get_default_initial(manifest)]
        elif manifest.get('plasmid') == 'pRamR':
            plot_initials = [get_default_initial(manifest), f'ea.{get_default_initial(manifest)}']
    
    if plot_range is None:
        barcode_frame = data
    else:
        barcode_frame = data.loc[plot_range[0]:plot_range[1]]
        
    if (not includeChimeras) and ("isChimera" in barcode_frame.columns):
        barcode_frame = barcode_frame[barcode_frame["isChimera"] != True]
        
    if include_ref_seqs:
        RS_count_frame = data[data["RS_name"]!=""]
        barcode_frame = pd.concat([barcode_frame, RS_count_frame])
        
    # Turn interactive plotting on or off depending on show_plots
    plt.ion()
    
    if save_plots:
        os.chdir(manifest.get('data_directory'))
        pdf_file = 'barcode fitness plots.pdf'
        pdf = PdfPages(pdf_file)
        
    antibiotic_conc_list = manifest.get('antibiotic_conc_list')
    ligand_list = manifest.get('ligand_list')
    
    #plot fitness curves
    fitness_columns_setup = get_fitness_columns_setup(manifest, data, plot_initials=plot_initials)
    if fitness_columns_setup[0]:
        old_style_plots, x, linthresh, fit_plot_colors = fitness_columns_setup
        if "sensor_params" not in barcode_frame.columns:
            show_fits = False
    else:
        old_style_plots, linthresh, fit_plot_colors, plot_df = fitness_columns_setup
        if "log_g0" not in barcode_frame.columns:
            show_fits = False
    
    if show_GP or show_hill_fit:
        plt.rcParams["figure.figsize"] = [2*box_size, 3*box_size/3]
    else:
        plt.rcParams["figure.figsize"] = [2*box_size, 3*box_size/6]
    
    if show_fits:
        fit_fitness_difference_params = manifest.get('fit_fitness_difference_params')
        
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
            if manifest.get('plasmid') in ['pVER', 'pCymR']:
                def fit_funct(x, log_g0, log_ginf, log_ec50, nx, low_fitness, mid_g, fitness_n):
                    return double_hill_funct(x, 10**log_g0, 10**log_ginf, 10**log_ec50, nx,
                                                low_fitness, 0, mid_g, fitness_n)
            elif manifest.get('plasmid') == 'pRamR':
                def fit_funct(x, log_g0, log_ginf, log_ec50, nx, high_fitness, mid_g, fitness_n):
                    return double_hill_funct(x, 10**log_g0, 10**log_ginf, 10**log_ec50, nx,
                                                0, high_fitness, mid_g, fitness_n)
    
    show_mut_codes = ('mutation_codes' in barcode_frame.columns) and show_mut_codes
    show_variant = ('variant' in barcode_frame.columns) and show_variant
    fig_axs_list = []
    for index, row in barcode_frame.iterrows(): # iterate over barcodes
        if show_GP or show_hill_fit:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig, axs_grid = plt.subplots(2, 2)
            axl = axs_grid.flatten()[0]
            axr = axs_grid.flatten()[2]
            axg = axs_grid.flatten()[1]
            axdg = axs_grid.flatten()[3]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
        
        for initial in plot_initials:
            fill_style = "full" if initial==plot_initials[0] else "none"
            if old_style_plots:
                for tet, color in zip(antibiotic_conc_list, fit_plot_colors):
                    y = row[f"fitness_{tet}_estimate_{initial}"]*fitness_scale_factor
                    s = row[f"fitness_{tet}_err_{initial}"]*fitness_scale_factor
                    axl.errorbar(x, y, s, marker='o', ms=8, color=color, fillstyle=fill_style)
                
                y_zero = row[f"fitness_{0}_estimate_{initial}"]
                s_zero = row[f"fitness_{0}_err_{initial}"]
                y_high = row[f"fitness_{antibiotic_conc_list[1]}_estimate_{initial}"]
                s_high = row[f"fitness_{antibiotic_conc_list[1]}_err_{initial}"]
                
                y = (y_high - y_zero)/y_zero.mean()
                s = np.sqrt( s_high**2 + s_zero**2 )/y_zero.mean()
                fill_style = "full" if initial==plot_initials[0] else "none"
                axr.errorbar(x, y, s, marker='o', ms=8, color=fit_plot_colors[0], fillstyle=fill_style)
            else:
                for tet, marker in zip(antibiotic_conc_list, ['o', '<', '>']):
                    for lig, color in zip(ligand_list, fit_plot_colors):
                        df = plot_df
                        df = df[(df.ligand==lig)|(df.ligand=='none')]
                        df = df[df.antibiotic_conc==tet]
                        x = df[lig]
                        y = np.array([row[f"fitness_S{i}_{initial}"] for i in df.sample_id])*fitness_scale_factor
                        s = np.array([row[f"fitness_S{i}_err_{initial}"] for i in df.sample_id])*fitness_scale_factor
                        axl.errorbar(x, y, s, marker=marker, ms=8, color=color, fillstyle=fill_style)
                
                if 'ea' not in initial:
                    for tet, marker in zip(antibiotic_conc_list, ['o', '<', '>']):
                        marker = marker if show_fits else '-' + marker
                        if tet > 0:
                            for j, (lig, color) in enumerate(zip(ligand_list, fit_plot_colors)):
                                # use is_gp_model=True to get data for all concentrations:
                                stan_data = bs_frame_stan_data(manifest, data, row, initial=initial, is_gp_model=True) 
                                
                                if len(antibiotic_conc_list) == 2:
                                    # Single non-zero antibiotic concentration
                                    if 'y_0' in stan_data:
                                        # case for multiple ligands
                                        st_y_0 = list(stan_data[f'y_0'])
                                        st_y_0_err = list(stan_data[f'y_0_err'])
                                        x = np.array([0]*len(st_y_0) + list(stan_data[f'x_{j+1}']))
                                        y = np.array(st_y_0 + list(stan_data[f'y_{j+1}']))
                                        s = np.array(st_y_0_err + list(stan_data[f'y_{j+1}_err']))
                                    else:
                                        # case for single ligand
                                        x = stan_data['x']
                                        y = stan_data['y']
                                        s = stan_data['y_err']
                                elif len(antibiotic_conc_list) == 3:
                                    # Two non-zero antibiotic concentrations
                                    if tet == antibiotic_conc_list[1]:
                                        tet_str = 'low'
                                        st_y_0 = [stan_data[f'y_0_low_tet']]
                                        st_y_0_err = [stan_data[f'y_0_low_tet_err']]
                                    else:
                                        tet_str = 'high'
                                        st_y_0 = []
                                        st_y_0_err = []
                                    x = np.array([0]*len(st_y_0) + list(stan_data[f'x_{j+1}']))
                                    y = np.array(st_y_0 + list(stan_data[f'y_{j+1}_{tet_str}_tet']))
                                    s = np.array(st_y_0_err + list(stan_data[f'y_{j+1}_{tet_str}_tet_err']))
                                    
                                axr.plot(x[s<10], y[s<10], marker, ms=8, color=color, fillstyle=fill_style)
                                ylimr = axr.get_ylim()
                                ylimr = (ylimr[0] - 0.1, ylimr[1] + 0.1)
                                axr.set_ylim(ylimr)
                                axr.errorbar(x[s<10], y[s<10], s[s<10], fmt=marker, ms=8, color=color, fillstyle=fill_style)
            
            if initial == plot_initials[0]:
                barcode_str = str(index) + ': '
                barcode_str += format(row[f'total_counts'], ",") + "; "
                barcode_str += row['RS_name']
                if show_variant:
                    barcode_str += f", {row.variant}"
                if show_mut_codes:
                    barcode_str += f", {row.mutation_codes}"
                if show_bc_str:
                    barcode_str += ": " + row['forward_BC'] + ",\n"
                    barcode_str += row['reverse_BC'] + " "
                    fontfamily = "Courier New"
                else:
                    fontfamily = None
                if not old_style_plots:
                    y_ref = stan_data['y_ref']
                    barcode_str += f"\ny_ref: {y_ref:.3f}"
                axl.text(x=0, y=1.025, s=barcode_str, horizontalalignment='left', verticalalignment='bottom',
                        transform=axl.transAxes, fontsize=13, fontfamily=fontfamily)
                fitness_units = 'log(10)/plate' if fitness_scale_factor==1 else '1/h'
                axl.set_ylabel(f'Fitness ({fitness_units})', size=14)
                axl.tick_params(labelsize=12);
                axr.set_ylabel(f'Fitness effect of {manifest.get("antibiotic")}', size=14)
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
                for tet, marker in zip(tet_level_list, ['<', '>']):
                    for lig, color in zip(ligand_list, fit_plot_colors):
                        df = plot_df
                        df = df[df.ligand==lig]
                        x = df[lig]
                        x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
                        x_fit = np.insert(x_fit, 0, 0)
                        
                        # LacI or CymR: fit_funct(x, log_g0, log_ginf, log_ec50, log_nx, low_fitness, mid_g, fitness_n)
                        # RamR: fit_funct(x, log_g0, log_ginf, log_ec50, log_nx, high_fitness, mid_g, fitness_n)
                        if manifest.get('plasmid') == 'pVER':
                            params_list = ['log_g0', f'log_ginf_{lig}', f'log_ec50_{lig}', f'sensor_n_{lig}', 
                                            f'low_fitness_{tet}_tet', f'mid_g_{tet}_tet', f'fitness_n_{tet}_tet']
                        elif manifest.get('plasmid') == 'pRamR':
                            params_list = ['log_g0', f'log_ginf_{lig}', f'log_ec50_{lig}', f'sensor_n_{lig}', 
                                            f'high_fitness', f'mid_g', f'fitness_n']
                        elif manifest.get('plasmid') == 'pCymR':
                            params_list = ['log_g0', f'log_ginf_{lig}', f'log_ec50_{lig}', f'sensor_n_{lig}', 
                                            f'low_fitness', f'mid_g', f'fitness_n']
                        
                        params = [row[p] for p in params_list]
                        y_fit = fit_funct(x_fit, *params)
                        
                        axr.plot(x_fit, y_fit, color=color, zorder=1000);
                        ylimr = axr.get_ylim()
                        ylimr = (min(ylimr[0], min(y_fit)-0.2), max(ylimr[1], max(y_fit)+0.2))
                        axr.set_ylim(ylimr)
            
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
                    if i == 3:
                        ylimr = axr.get_ylim()
                        ylimr = (min(ylimr[0], min(stan_f[2-i])-0.2), max(ylimr[1], max(stan_f[2+i])+0.2))
                        axr.set_ylim(ylim)
                    axg.fill_between(x, stan_g[2-i], stan_g[2+i], alpha=.3, color=fit_plot_colors[2]);
                    axdg.fill_between(x, stan_dg[2-i], stan_dg[2+i], alpha=.3, color=fit_plot_colors[3]);
                    
                # Also plot Hill fit result for g
                x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
                x_fit = np.insert(x_fit, 0, 0)
                hill_params = 10**row["sensor_params"][:4]
                axg.plot(x_fit, hill_funct(x_fit, *hill_params), c='k', zorder=1000)
            else:
                fill_alpha = 0.2
                
                tet_level_list = ['high'] if len(antibiotic_conc_list)==2 else ['low', 'high']
                for lig, color in zip(ligand_list, fit_plot_colors):
                    stan_g = 10**row[f"GP_log_g_{lig}"]
                    stan_dg = row[f"GP_dlog_g_{lig}"]
                    
                    df = plot_df
                    df = df[(df.ligand==lig)|(df.ligand=='none')]
                    df = df[df.with_tet]
                    x = np.unique(df[lig])
                
                    axg.plot(x, stan_g[2], '--', color=color, lw=3, label=lig)
                    axdg.plot(x, stan_dg[2], '--', color=color, lw=3)
                    
                    if manifest.get('plasmid') == 'pVER':
                        for tet in tet_level_list:
                            stan_f = row[f"GP_y_{lig}_{tet}_tet"]
                            axr.plot(x, stan_f[2], '--', color=color, lw=3)
                    elif manifest.get('plasmid') in ['pRamR', 'pCymR']:
                        for tet in tet_level_list:
                            stan_f = row[f"GP_y_{lig}"]
                            axr.plot(x, stan_f[2], '--', color=color, lw=3)
                        
                    axg.set_ylabel('Dose Response (MEF)', size=14)
                    axg.tick_params(labelsize=12);
                    if log_g_scale: axg.set_yscale("log")
                    
                    axdg.plot([x[0],x[-1]], [0,0], c='k');
                    axdg.set_ylabel('GP d(log(g))/d(log(x))', size=14)
                    axdg.tick_params(labelsize=12);
                    for i in range(1,3):
                        axg.fill_between(x, stan_g[2-i], stan_g[2+i], alpha=fill_alpha, color=color);
                        axdg.fill_between(x, stan_dg[2-i], stan_dg[2+i], alpha=fill_alpha, color=color);
                        for tet in tet_level_list:
                            axr.fill_between(x, stan_f[2-i], stan_f[2+i], alpha=fill_alpha, color=color);
                    axg.legend(loc='upper left', bbox_to_anchor= (1.03, 0.97), ncol=1, borderaxespad=0, frameon=True, fontsize=10)
                    
                    # Also plot Hill fit result for g
                    x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
                    x_fit = np.insert(x_fit, 0, 0)
                    params_list = ['log_g0', f'log_ginf_{lig}', f'log_ec50_{lig}', f'sensor_n_{lig}']
                    hill_params = [10**row[p] for p in params_list[:-1]] + [row[params_list[-1]]]
                    y_fit = hill_funct(x_fit, *hill_params)
                    axg.plot(x_fit, y_fit, c=color, zorder=1000)
                    
            
        if show_hill_fit and (not show_GP):
            if old_style_plots:
                # Only plot Hill fit result for g
                x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
                x_fit = np.insert(x_fit, 0, 0)
                hill_params = 10**row["sensor_params"][:4]
                axg.plot(x_fit, hill_funct(x_fit, *hill_params), c='k', zorder=1000)
            else:
                fill_alpha = 0.2
                
                tet_level_list = ['high'] if len(antibiotic_conc_list)==2 else ['low', 'high']
                for lig, color in zip(ligand_list, fit_plot_colors):
                    stan_g = 10**row[f"GP_log_g_{lig}"]
                    stan_dg = row[f"GP_dlog_g_{lig}"]
                    
                    df = plot_df
                    df = df[(df.ligand==lig)|(df.ligand=='none')]
                    df = df[df.with_tet]
                    x = np.unique(df[lig])
                    
                    # Only plot Hill fit result for g
                    x_fit = np.logspace(np.log10(linthresh/10), np.log10(2*max(x)))
                    x_fit = np.insert(x_fit, 0, 0)
                    params_list = ['log_g0', f'log_ginf_{lig}', f'log_ec50_{lig}', f'sensor_n_{lig}']
                    hill_params = [10**row[p] for p in params_list[:-1]] + [row[params_list[-1]]]
                    y_fit = hill_funct(x_fit, *hill_params)
                    axg.plot(x_fit, y_fit, c=color, zorder=1000)
                    
        fig_axs_list.append((fig, axs_grid))
        
    if save_plots:
        pdf.savefig()

    if save_plots:
        pdf.close()
        
    return fig_axs_list


def calibrate_fitness_difference_params(manifest, data,
                                        calibration_data_table,
                                        spike_in_initial=None,
                                        run_stan_fit=False,
                                        plot_raw_fitness=False,
                                        include_zero_antibiotic=False,
                                        color_by_ligand_conc=None,
                                        save_fitness_difference_params=False,
                                        rs_exclude_list=[],
                                        show_exclude_data=True,
                                        use_only_rs_variants=False,
                                        RS_list=None,
                                        wt_cutoff=0,
                                        min_err=0.05, # Either a single value (float), or a dictionary with keys equal to the antibiotic concentrations and values equal to the min_err for that concentration
                                        show_old_fit=True,
                                        apply_ramr_correction=None,
                                        turn_off_cmdstanpy_logger=True,
                                        robust_error_model=False,
                                        robust_nu=4,
                                        repeat_after_dropping_outliers=False,
                                        re_stan_on_rhat=True,
                                        rhat_cutoff=1.05,
                                        outlier_cutoff=2.5,
                                        return_fig=False,
                                        return_resid_table=False,
                                        return_fit_data=False,
                                        fig_size=[12, 6],
                                        alpha=0.7,
                                        plot_ligands=None,
                                        show_progress=True):
    if spike_in_initial is None:
        spike_in_initial = get_default_initial(manifest)
    spike_in = fitness_utils.get_spike_in_name_from_inital(manifest.get('plasmid'), spike_in_initial)
    print(f'Calibrating with counts normalized to {spike_in}, initial: {spike_in_initial}')
    plasmid = manifest.get('plasmid')
    
    if plot_raw_fitness:
        run_stan_fit = False
        show_old_fit = False
    
    if plasmid == 'pVER':
        stan_model_file = "Hill_equation_fit-zero_high.stan"
        
        ligand_plot_list = manifest.get('ligand_list')[:1]
        
        def init_fitness_fit(y_data):
            low = -0.8 #np.mean(y_data[:2])
            mid = 1000 #np.random.normal(1, 0.2) * 10000
            n = 1.1 #np.random.normal(1, 0.2) * 3
            sig = np.random.normal(1, 0.2) * 0.1
            return dict(low_level=low, IC_50=mid, hill_n=n, sigma=sig)
    elif plasmid == 'pRamR':
        stan_model_file = "Hill_equation_fit-zero_low.stan"
        
        ligand_plot_list = manifest.get('ligand_list')
        
        if apply_ramr_correction is None:
            apply_ramr_correction = True
        
        def init_fitness_fit(y_data):
            low = -1.5 #np.mean(y_data[:2])
            high = -0.5
            mid = 1000 #np.random.normal(1, 0.2) * 10000
            n = 1.1 #np.random.normal(1, 0.2) * 3
            sig = np.random.normal(1, 0.2) * 0.1
            return dict(low_level=low, IC_50=mid, hill_n=n, sigma=sig, high_level=high)
    elif plasmid == 'pCymR':
        stan_model_file = "Hill_equation_fit-zero_high.stan"
        
        ligand_plot_list = manifest.get('ligand_list')
        
        def init_fitness_fit(y_data):
            low = -0.8 #np.mean(y_data[:2])
            mid = 100 #np.random.normal(1, 0.2) * 10000
            n = 1.1 #np.random.normal(1, 0.2) * 3
            sig = np.random.normal(1, 0.2) * 0.1
            return dict(low_level=low, IC_50=mid, hill_n=n, sigma=sig)
    elif plasmid == 'Align-TF':
        stan_model_file = "Hill_equation_fit-zero_high.stan"
        
        ligand_plot_list = manifest.get('ligand_list')
        
        def init_fitness_fit(y_data):
            low = -0.8 #np.mean(y_data[:2])
            mid = 100 #np.random.normal(1, 0.2) * 10000
            n = 1.1 #np.random.normal(1, 0.2) * 3
            sig = np.random.normal(1, 0.2) * 0.1
            return dict(low_level=low, IC_50=mid, hill_n=n, sigma=sig)
    elif plasmid == 'Align-T7RNAP_1':
        stan_model_file = "Hill_equation_fit.basic.stan"
        
        ligand_plot_list = [None]
        
        def init_fitness_fit(y_data):
            low = -0.8 #np.mean(y_data[:2])
            high = -0.2 #np.mean(y_data[:2])
            mid = 100 #np.random.normal(1, 0.2) * 10000
            n = 1.1 #np.random.normal(1, 0.2) * 3
            sig = np.random.normal(1, 0.2) * 0.1
            return dict(low_level=low, high_level=high, IC_50=mid, hill_n=n, sigma=sig)
    
    if plot_ligands is not None:
        ligand_plot_list = [x for x in ligand_plot_list if x in plot_ligands]
    
    if robust_error_model:
        stan_model_file = stan_model_file[:-4] + "robust.stan"
        
    if run_stan_fit:
        fitness_model = stan_utils.compile_model(stan_model_file)
    
    bs_frame = data
    if RS_list is None:
        RS_list = [x for x in bs_frame.RS_name if (x != '')]
        if use_only_rs_variants:
            RS_list = [x for x in RS_list if ('RS' in x)]
        RS_list = np.unique(RS_list)
    
    sample_plate_map = manifest.get('sample_plate_map')
    plot_df = sample_plate_map
    if 'ligand' in sample_plate_map.columns:
        lig_list = list(np.unique(sample_plate_map.ligand))
        if 'none' in lig_list:
            lig_list.remove('none')
        plot_df = plot_df[plot_df.growth_plate==2].sort_values(by=lig_list)
    else:
        plot_df = plot_df[plot_df.growth_plate==2].sort_values(by='sample_id')
    
    
    fit_plot_colors = sns.color_palette()
    
    def cytom_variant_from_rs_name(rs_name):
        if plasmid == 'pVER':
            if 'RS' in rs_name:
                plas = 'pVER-RS-' + rs_name[2:]
            elif 'wt' in rs_name:
                plas = 'pVER-IPTG-WT'
            else:
                plas = rs_name.replace('DT_IPTG_', 'pVER-IPTG-')
            if '(' in rs_name:
                plas = rs_name.replace('WT', 'pVER-IPTG-WT')
                
        elif plasmid == 'pRamR':
            if 'RS' in rs_name:
                plas = 'RamR-' + rs_name
            elif 'wt' in rs_name:
                plas = 'pRamR-WT'
            else:
                plas = rs_name
                
        elif plasmid == 'pCymR':
            if 'RS' in rs_name:
                plas = 'pCymR-' + rs_name
            elif 'wt' in rs_name:
                plas = 'pCymR-WT'
            else:
                plas = rs_name
                
        elif plasmid == 'Align-TF':
            var = f'{rs_name}_mScar'
            
            if var == 'pRamR-WT-fin_mScar':
                var = 'pRamR-WT_P150_2.3k_mScar' # short-term fix
                
        elif plasmid == 'Align-T7RNAP_1':
            # RS names:       'T7_D240E', 'T7_D240G', 'T7_E207K', 'T7_F21Y', 'T7_N748D', 'T7_P266L', 'T7_WT'
            # cytom variants: 'pT7-78-D240E', 'pT7-78-D240G', 'pT7-78-E207K', 'pT7-78-F21Y', 'pT7-78-N748D', 'pT7-78-P266L', 'pT7-78-WT'
            var = rs_name.replace('T7_', 'pT7-78-')
            
        return var
        
    # Fitness calibration function is Hill function with either low or high value to zero:
    def hill_funct(x, low, high, mid, n):
        return low + (high-low)*( x**n )/( mid**n + x**n )

    if plasmid in ['pVER', 'pCymR', 'Align-TF']:
        def fit_funct(x, low, mid, n):
            return hill_funct(x, low, 0, mid, n)
    elif plasmid in ['Align-T7RNAP_1']:
        def fit_funct(x, low, high, mid, n):
            return hill_funct(x, low, high, mid, n)
    elif plasmid == 'pRamR':
        def fit_funct(x, high, mid, n):
            return hill_funct(x, 0, high, mid, n)
    
    if include_zero_antibiotic:# and plot_raw_fitness:
        plot_antibiotic_list = manifest.get('antibiotic_conc_list')
    else:
        plot_antibiotic_list = manifest.get('antibiotic_conc_list')[1:]
    plt.rcParams["figure.figsize"] = fig_size
    if len(plot_antibiotic_list)==1:
        fig, axs = plt.subplots()
        axs = [axs]
    elif len(plot_antibiotic_list)==2:
        fig, axs = plt.subplots(1, 2)
    else:
        plt.rcParams["figure.figsize"] = [fig_size[0], fig_size[1]*len(plot_antibiotic_list)]
        fig, axs = plt.subplots(len(plot_antibiotic_list), 1, layout='tight')
    
    
    if show_old_fit:
        if plasmid in ['Align-TF', 'Align-T7RNAP_1']:
            if (type(manifest.get('fit_fitness_difference_params')) is dict) and (spike_in_initial in manifest.get('fit_fitness_difference_params')):
                params = manifest.get('fit_fitness_difference_params')[spike_in_initial]
                plot_fit_params = [params[tet]['popt'] + params[tet]['perr'] for tet in plot_antibiotic_list]
            else:
                plot_fit_params = [None]*len(plot_antibiotic_list)
        else:
            if manifest.get('fit_fitness_difference_params') is None:
                plot_fit_params = [None]*len(plot_antibiotic_list)
            else:
                plot_fit_params = manifest.get('fit_fitness_difference_params')
                if type(plot_fit_params[0]) is not list:
                    plot_fit_params = [plot_fit_params]
    else:
        plot_fit_params = [None]*len(axs)
    
    if plasmid in ['Align-TF', 'Align-T7RNAP_1']:
        stan_params_to_save = {}
    else:
        stan_params_to_save = []
    if return_fit_data:
        fit_data_ret = []
    if return_resid_table:
        resid_table_list = []
    for ax, tet, old_fit_params in zip(axs, plot_antibiotic_list, plot_fit_params):
        fmt = 'o'
        ms = 8
        color_ind = -1
        fmt_list = ['o', '^', 'v', '<', '>', 'd', 'p', '*', 's', 'h', '+', 'x']
        fmt_ind = 0
        
        x_fit_list = []
        y_fit_list = []
        y_err_list = []
        identifier_list = []
        
        resid_frame_lists = {}
        if len(self.ligand_list) == 0:
            resid_frame_keys = ['rs_name', 'sample', 'ref_fitness', 'fitness']
        else:
            resid_frame_keys = ['rs_name', 'sample', 'ligand', 'lig_conc', 'ref_fitness', 'fitness']
        for k in resid_frame_keys:
            resid_frame_lists[k] = []
        resid_frame_lists['antibiotic_conc'] = tet

        if f"fitness_S1_ea.{spike_in_initial}" in bs_frame.columns:
            resid_frame_lists['early_fitness'] = []
        
        if color_by_ligand_conc is not None:
            lig_color_conc_list = []
                    
        for RS_name in RS_list:
            for lig in ligand_plot_list:
                if lig is not None:
                    i = np.where(np.array(self.ligand_list)==lig)[0][0]
                var = cytom_variant_from_rs_name(RS_name)

                df = calibration_data_table
                df = df[df.variant==var]
                if (len(ligand_plot_list)>1) or (plasmid == 'pCymR'):
                    df = df[df.ligand==lig]
                if lig is None:
                    lab = f'{RS_name}'
                else:
                    lab = f'{RS_name}, {lig}'

                if len(df)==1:
                    calibration_row = df.iloc[0]
                    
                    if lig is not None:
                        calibration_params = [10**calibration_row[x] for x in ['log_g0', 'log_ginf', 'log_ec50']] + [calibration_row['n']]
                    else:
                        calibration_params = [calibration_row['transcription_rate_mean']]
                    if not np.any(np.isnan(calibration_params)):
                        HiSeq_df = bs_frame[bs_frame['RS_name']==RS_name]
                        if 'wt' in RS_name:
                            HiSeq_df = HiSeq_df[HiSeq_df.total_counts>wt_cutoff]
                            
                        if len(HiSeq_df) > 0:
                            color_ind += 1
                            if color_ind >= len(fit_plot_colors):
                                color_ind=0
                                fmt_ind += 1
                                if fmt_ind >= len(fmt_list):
                                    fmt_ind = 0
                                fmt = fmt_list[fmt_ind]
                            color=fit_plot_colors[color_ind]
                            
                            var_labeled = False
                            
                            # For some datasets, there will be more than one entry for some RS_name variants - particularly for the wild-type.
                            #     So, use a loop here instead of just taking the single row that matches RS_name.
                            for ind, HiSeq_row in HiSeq_df.iterrows(): 
                                if var_labeled:
                                    lab = None
                                var_labeled = True
                                
                                if plasmid not in ['Align-TF', 'Align-T7RNAP_1']:
                                    stan_data = self.bs_frame_stan_data(HiSeq_row, 
                                                                        initial=spike_in_initial,
                                                                        min_err=min_err,
                                                                        apply_ramr_correction=apply_ramr_correction)
                                    xerr = None
                                
                                if plasmid in ['Align-TF', 'Align-T7RNAP_1']: # Can't use else here.
                                    plot_df_align = plot_df
                                    plot_df_align = plot_df_align[plot_df_align.antibiotic_conc==tet]
                                    
                                    if plasmid == 'Align-TF':
                                        # For Align-TF project, measurements at zero ligand and one non-zero ligand per TF
                                        tf = align_tf_from_ligand(lig)
                                        plot_df_align = plot_df_align[(plot_df_align.transcription_factor==tf)|(plot_df_align.ligand=='none')]
                                    
                                    samples = np.array(plot_df_align['sample_id'])
                                    if plasmid == 'Align-TF':
                                        sample_ligand_list = np.array(plot_df_align['ligand'])
                                        sample_tf_list = np.array(plot_df_align['transcription_factor'])
                                        ligand_conc_list = np.array(plot_df_align[lig])
                                    
                                        # For Align-TF, make array of ref_samples in the same order as samples (using ligand ID and tf to set ordering):
                                        ref_samples = []
                                        for lig_ref, tf_ref in zip(sample_ligand_list, sample_tf_list):
                                            df_ref = plot_df
                                            df_ref = df_ref[(df_ref.transcription_factor==tf)|(df_ref.ligand=='none')]
                                            df_ref = df_ref[df_ref.antibiotic_conc==0]
                                            df_ref = df_ref[df_ref.ligand==lig_ref]
                                            df_ref = df_ref[df_ref.transcription_factor==tf_ref]
                                            if len(df_ref) != 1:
                                                raise ValueError('length of df_ref != 1')
                                            ref_samples.append(df_ref.iloc[0].sample_id)
                                        ref_samples = np.array(ref_samples)
                                    else:
                                        # For Align-T7RNAP_1, make array of ref_samples in the same order as samples
                                        df_ref = plot_df
                                        df_ref = df_ref[df_ref.antibiotic_conc==0]
                                        ref_samples = np.array(df_ref['sample_id'])
                                    
                                    # samples and ref_samples should each have the same number of values:
                                    #    For Align-TF: (4), three without ligand and one with
                                    #    For Align-T7RNAP_1: (12), which are all replicates
                                    
                                    y = np.array([HiSeq_row[f"fitness_S{s}_{spike_in_initial}"] for s in samples])
                                    y_err = np.array([HiSeq_row[f"fitness_S{s}_err_{spike_in_initial}"] for s in samples])
                                    
                                    if not plot_raw_fitness:
                                        y_ref = np.array([HiSeq_row[f"fitness_S{s}_{spike_in_initial}"] for s in ref_samples])
                                        y_ref_err = np.array([HiSeq_row[f"fitness_S{s}_err_{spike_in_initial}"] for s in ref_samples])
                                        y_err = np.sqrt((y_err/y_ref)**2 + (y*y_ref_err/y_ref**2)**2)
                                        y = (y - y_ref)/y_ref
                                    
                                    # Get x values (gene expression) from asymmetric Hill fit for Align-TF data:
                                    if plasmid == 'Align-TF':
                                        x = []
                                        xerr = []
                                        min_xerr = np.log10(1.2)
                                        for lig_conc in ligand_conc_list:
                                            if lig_conc == 0:
                                                g = calibration_row['log_g0']
                                                gerr = np.sqrt(calibration_row['log_g0_err']**2 + min_xerr**2)
                                            else:
                                                g = calibration_row[f'log_g_{int(lig_conc)}_{lig}']
                                                gerr = np.sqrt(calibration_row[f'log_g_{int(lig_conc)}_{lig}_err']**2 + min_xerr**2)
                                                
                                            gerr = fitness_utils.log_plot_errorbars(log_mu=g, log_sig=gerr)
                                            g = 10**g
                                            
                                            x.append(g)
                                            xerr.append(gerr)
                                        x = np.array(x)
                                        xerr = np.array(xerr).transpose()
                                    
                                    else:# Get x values (transcription_rate_mean) from calibration data for Align-T7RNAP_1 data:
                                        x = np.array([calibration_row['transcription_rate_mean']]*len(y))
                                        xerr = np.array([calibration_row['transcription_rate_std']]*len(y))
                                    
                                elif (len(ligand_plot_list) > 1) or (plasmid == 'pCymR'):
                                    # For RamR and CymR
                                    ligand_conc_list = np.array([0, 0] + list(stan_data[f'x_{i+1}']))
                                    samples = np.array(list(stan_data[f'samp_0']) + list(stan_data[f'samp_{i+1}']))
                                    if plot_raw_fitness:
                                        y = np.array([HiSeq_row[f"fitness_S{s}_{spike_in_initial}"] for s in samples])
                                        y_err = np.array([HiSeq_row[f"fitness_S{s}_err_{spike_in_initial}"] for s in samples])
                                    else:
                                        y = np.array(list(stan_data[f'y_0']) + list(stan_data[f'y_{i+1}']))
                                        y_err = np.array(list(stan_data[f'y_0_err']) + list(stan_data[f'y_{i+1}_err']))
                                        
                                else: # For LacI
                                    # For LacI
                                    if tet == plot_antibiotic_list[0]:
                                        if f'x_{i+1}' in stan_data:
                                            ligand_conc_list = np.array([0] + list(stan_data[f'x_{i+1}']))
                                            samples = np.array([stan_data[f'samp_0_low_tet']] + list(stan_data[f'samp_{i+1}_low_tet']))
                                            y = np.array([stan_data[f'y_0_low_tet']] + list(stan_data[f'y_{i+1}_low_tet']))
                                            y_err = np.array([stan_data[f'y_0_low_tet_err']] + list(stan_data[f'y_{i+1}_low_tet_err']))
                                        else:
                                            ligand_conc_list = np.array(stan_data['x'])
                                            samples = np.array(stan_data['samp'])
                                            y = np.array(stan_data['y'])
                                            y_err = np.array(stan_data[f'y_err'])
                                    elif tet == plot_antibiotic_list[1]:
                                        ligand_conc_list = np.array(stan_data[f'x_{i+1}'])
                                        samples = np.array(stan_data[f'samp_{i+1}_high_tet'])
                                        y = np.array(stan_data[f'y_{i+1}_high_tet'])
                                        y_err = np.array(stan_data[f'y_{i+1}_high_tet_err'])
                                    
                                    if plot_raw_fitness:
                                        y = np.array([HiSeq_row[f"fitness_S{s}_{spike_in_initial}"] for s in samples])
                                        y_err = np.array([HiSeq_row[f"fitness_S{s}_err_{spike_in_initial}"] for s in samples]) 
                                
                                if plasmid not in ['Align-TF', 'Align-T7RNAP_1']:
                                    if np.any(np.isinf(calibration_params)):
                                        calibration_params = [0]*len(calibration_params)
                                    x = hill_funct(ligand_conc_list, *calibration_params)
                                
                                if color_by_ligand_conc is not None:
                                    if lig == color_by_ligand_conc:
                                        lig_color_conc = ligand_conc_list
                                    else:
                                        lig_color_conc = [0]*len(ligand_conc_list)
                                
                                # Enforce the min_err here, just before adding the values to the y_err_list and plotting
                                if type(min_err) == dict:
                                    y_err = np.sqrt(y_err**2 + min_err[tet]**2)
                                else:
                                    y_err = np.sqrt(y_err**2 + min_err**2)
                                
                                sel = RS_name in rs_exclude_list
                                if sel:
                                    fill_style = 'none'  
                                else:
                                    fill_style = None
                                    x_fit_list += list(x)
                                    y_fit_list += list(y)
                                    y_err_list += list(y_err)
                                    identifier_list += [f'{RS_name} at x = {p:.2e}' for p in x]
                                    if color_by_ligand_conc is not None:
                                        lig_color_conc_list += list(lig_color_conc)
                                    
                                    if return_resid_table:
                                        resid_frame_lists['rs_name'] += [RS_name]*len(x)
                                        if 'ligand' in resid_frame_keys:
                                            resid_frame_lists['ligand'] += [lig]*len(x)
                                            resid_frame_lists['lig_conc'] += list(ligand_conc_list)
                                        resid_frame_lists['sample'] += list(samples)
                                        
                                        if plasmid in ['Align-TF', 'Align-T7RNAP_1']:
                                            resid_frame_lists['ref_fitness'] += list(y_ref)
                                        else:
                                            y_ref = stan_data['y_ref']
                                            resid_frame_lists['ref_fitness'] += [y_ref]*len(x)
                                        
                                        resid_frame_lists['fitness'] += [HiSeq_row[f"fitness_S{smp}_{spike_in_initial}"] for smp in samples]
                                        
                                        if 'early_fitness' in resid_frame_lists:
                                            resid_frame_lists['early_fitness'] += [HiSeq_row[f"fitness_S{smp}_ea.{spike_in_initial}"] for smp in samples]
                                    
                                include_data_in_plot = show_exclude_data or (not sel)
                                if (include_data_in_plot) and (color_by_ligand_conc is None):
                                    ax.errorbar(x, y, y_err, xerr, fmt=fmt, ms=ms, color=color, 
                                                fillstyle=fill_style, label=lab, alpha=alpha)
                elif len(df) == 0:
                    if plasmid == 'Align-TF':
                        tf = align_tf_from_ligand(lig)
                        if ('norm' not in RS_name) and (tf in var):
                            print(f'No cytometry data for {RS_name}, {var} with {lig}')
                else:
                    raise ValueError('length of df > 1')
        
        
        
        x_fit_list = np.array(x_fit_list)
        y_fit_list = np.array(y_fit_list)
        y_err_list = np.array(y_err_list)
        identifier_list = np.array(identifier_list)
        
        if return_fit_data:
            df_ret = pd.DataFrame({'x':x_fit_list, 'y':y_fit_list, 'yerr':y_err_list})
            fit_data_ret = []
        
        if color_by_ligand_conc is not None:
            lig_color_conc_list = np.array(lig_color_conc_list)
            concentrations = np.unique(lig_color_conc_list)
            if return_fit_data:
                df_ret[f'{color_by_ligand_conc} conc'] = lig_color_conc_list
            for c in concentrations:
                x = x_fit_list[lig_color_conc_list==c]
                y = y_fit_list[lig_color_conc_list==c]
                yerr = y_err_list[lig_color_conc_list==c]
                ax.errorbar(x, y, yerr, fmt='o', ms=ms, label=f'[{color_by_ligand_conc}] = {c}', alpha=alpha)
        
        if return_fit_data:
            fit_data_ret.append(df_ret)
            
        ylim = ax.get_ylim()
        ax.set_xscale("symlog")
        ax.set_xlabel('Function (MEF)');
        ax.set_title(f'{tet} {manifest.get("antibiotic")}', size=14)
        if plot_raw_fitness:
            ax.set_ylabel(f'Fitness')
        else:
            ax.set_ylabel(f'Fitness Impact of {manifest.get("antibiotic")}')
        
        if show_old_fit and (old_fit_params is not None):
            x_plot_fit = np.logspace(np.log10((min(x_fit_list[x_fit_list>0]))/3), np.log10(1.2*max(x_fit_list)))
            if 0 in x_fit_list:
                x_plot_fit = np.array([0] + list(x_plot_fit))
            
            if plasmid == 'Align-T7RNAP_1':
                y_plot_fit = fit_funct(x_plot_fit, *old_fit_params[:4])
            else:
                y_plot_fit = fit_funct(x_plot_fit, *old_fit_params[:3])
            
            if run_stan_fit:
                lab = 'old fit'
            else:
                lab = 'fit'
            ax.plot(x_plot_fit, y_plot_fit, '--r', zorder=100, label=lab);
        
        
        print()
        if type(min_err) == dict:
            print(f'Plotting and fitting with minimum fitness error: {min_err[tet]} for [{manifest.get("antibiotic")}] = {tet}')
        else:
            print(f'Plotting and fitting with minimum fitness error: {min_err}')
        
        if run_stan_fit:
            
            if plasmid in ['pVER', 'pCymR', 'Align-TF']:
                key_params = ["low_level", "IC_50", "hill_n"]
            elif plasmid in ['Align-T7RNAP_1']:
                key_params = ["low_level", "high_level", "IC_50", "hill_n"]
            elif plasmid == 'pRamR':
                key_params = ["high_level", "IC_50", "hill_n"]
            
            print(f'Fitting with stan model from: {stan_model_file}')
            if turn_off_cmdstanpy_logger:
                
                cmdstanpy_logger = logging.getLogger("cmdstanpy")
                cmdstanpy_logger.disabled = True
            stan_data = dict(x=x_fit_list, y = y_fit_list, y_err = y_err_list, N=len(x_fit_list))
            if robust_error_model:
                stan_data['nu'] = robust_nu
            stan_init = init_fitness_fit(y)
            if plasmid == 'Align-T7RNAP_1':
                stan_init = dict(low_level=-0.9, high_level=-0.15, IC_50=300, hill_n=2, sigma=1)
            
            num_stan_re_runs = 3 if repeat_after_dropping_outliers else 1
            fit_data = stan_data
            num_points = len(fit_data['x'])
            drop_list = []
            for run_num in range(num_stan_re_runs):
                if (run_num == 0) or (len(drop_list) != len(old_drop_list)) or (not np.all(old_drop_list == drop_list)):
                    old_drop_list = drop_list
                    num_points = len(fit_data['x'])
                    print()
                    print(f'Fit iteration: {run_num+1}, with {num_points} data points')
                    if len(drop_list)>0:
                        print('    Dropped outliers:')
                        for d in drop_list:
                            print(f'        {d}')
                    
                    stan_fit = fitness_model.sample(data=fit_data, iter_warmup=500, iter_sampling=500, inits=stan_init, chains=4, show_progress=show_progress)
                    
                    if re_stan_on_rhat:
                        print(f'    Checking r_hat...')
                        rhat_params = stan_utils.check_rhat_by_params(stan_fit, rhat_cutoff=rhat_cutoff, stan_parameters=key_params)
                        if len(rhat_params) > 0:
                            print(f'    Re-running Stan fit becasue the following parameterrs had r_hat > {rhat_cutoff}: {rhat_params}')
                            stan_fit = fitness_model.sample(data=fit_data, iter_warmup=5000, iter_sampling=5000, inits=stan_init, chains=4, show_progress=show_progress)
                        else:
                            print(f'        ... r_hat below {rhat_cutoff} for all key parameters')
                    
                    stan_popt = [ np.mean(stan_fit.stan_variable(p))  for p in key_params]
                    stan_perr = [ np.std(stan_fit.stan_variable(p))  for p in key_params]
                
                    resid_list = y_fit_list - fit_funct(x_fit_list, *stan_popt)
                    dev_list = np.abs(resid_list)/y_err_list
                    
                    x_fit_list_2 = x_fit_list[dev_list<outlier_cutoff]
                    y_fit_list_2 = y_fit_list[dev_list<outlier_cutoff]
                    y_err_list_2 = y_err_list[dev_list<outlier_cutoff]
                    identifier_list_2 = identifier_list[dev_list<outlier_cutoff]
                    drop_list = identifier_list[dev_list>=outlier_cutoff]
                    dropped_x_list = x_fit_list[dev_list>=outlier_cutoff]
                    dropped_y_list = y_fit_list[dev_list>=outlier_cutoff]
                    
                    fit_data = dict(x=x_fit_list_2, y = y_fit_list_2, y_err = y_err_list_2, N=len(x_fit_list_2))
                    if robust_error_model:
                        fit_data['nu'] = robust_nu
            
            num_str = [ f"{x:.4}" for x in stan_popt]
            print(f"Fitness params with {tet} [{manifest.get('antibiotic')}]: {num_str}")
            num_str = [ f"{x:.4}" for x in stan_perr ]
            print(f"                 Error estimate: {num_str}")
            if plasmid in ['Align-TF', 'Align-T7RNAP_1']:
                stan_params_to_save[tet] = {'popt':list(stan_popt), 'perr':list(stan_perr)}
            else:
                stan_params_to_save.append(list(stan_popt) + list(stan_perr))
            dev = y_fit_list - fit_funct(x_fit_list, *stan_popt)
            for w_str, w in zip(['Unweighted', 'Weighted'], [None, 1/y_err_list**2]):
                rms_dev = np.sqrt(np.average(dev**2, weights=w))
                print(f"           {w_str} RMS deviation: {rms_dev:.4}")
            
            dev_2 = y_fit_list_2 - fit_funct(x_fit_list_2, *stan_popt)
            rms_dev = np.sqrt(np.mean((dev_2/y_err_list_2)**2))
            print(f"           Rescaled RMS deviation: {rms_dev:.4} (after dropping outliers; should be 1 for properly calibrated uncertanties)")
                
            spear_r = stats.spearmanr(x_fit_list, y_fit_list).statistic
            print(f"           Spearman R with outliers: {spear_r:.4}")
                
            spear_r = stats.spearmanr(x_fit_list_2, y_fit_list_2).statistic
            print(f"           Spearman R without outliers: {spear_r:.4}")
            
            x_plot_fit = np.logspace(np.log10((min(x_fit_list[x_fit_list>0]))/1.5), np.log10(1.2*max(x_fit_list)))
            if 0 in x_fit_list:
                x_plot_fit = np.array([0] + list(x_plot_fit))
            y_plot_fit = fit_funct(x_plot_fit, *stan_popt)
            ax.plot(x_plot_fit, y_plot_fit, '--k', zorder=200, label='calibration fit');
            
            if len(drop_list)>0:
                ax.plot(dropped_x_list, dropped_y_list, 'o', color='k', fillstyle='none', ms=ms+3, label='dropped data')
            
            if turn_off_cmdstanpy_logger:
                cmdstanpy_logger.disabled = False    
        
        if return_resid_table:
            if run_stan_fit:
                popt = stan_popt
            else:
                popt = old_fit_params[:3]
            
            resid_list = y_fit_list - fit_funct(x_fit_list, *popt)
            
            resid_frame = pd.DataFrame(resid_frame_lists)
            resid_frame['gene_expression'] = x_fit_list
            resid_frame['fitness_effect'] = y_fit_list
            resid_frame['fitness_effect_err'] = y_err_list
            resid_frame['resid'] = resid_list
            resid_frame['resid_err'] = y_err_list
            resid_table_list.append(resid_frame)
            
    if return_resid_table:
        resid_frame = pd.concat(resid_table_list, ignore_index=True)
    
    if color_by_ligand_conc is not None:
        ncol = 1
    elif plasmid == 'pRamR':
        ncol = int(len(RS_list)*len(lig_list)/40)
    elif plasmid in ['Align-TF', 'Align-T7RNAP_1']:
        ncol = int(np.round(len(RS_list)/12))
    else:
        ncol = int(len(RS_list)*len(lig_list)/40)
    if ncol == 0:
        ncol = 1
    
    if len(plot_antibiotic_list)<=2:
        axs[-1].legend(loc='upper left', bbox_to_anchor= (1.03, 0.97), ncol=ncol, borderaxespad=0, frameon=True);
    else:
        for ax in axs:
            ax.legend(loc='upper left', bbox_to_anchor= (1.03, 0.97), ncol=ncol, borderaxespad=0, frameon=True);
        
    if run_stan_fit and save_fitness_difference_params:
        if plasmid in ['Align-TF', 'Align-T7RNAP_1']:
            if type(manifest.get('fit_fitness_difference_params')) is dict:
                manifest['fit_fitness_difference_params'][spike_in_initial] = stan_params_to_save
            else:
                manifest['fit_fitness_difference_params'] = {spike_in_initial: stan_params_to_save}
        else:
            manifest['fit_fitness_difference_params'] = stan_params_to_save
    
    if return_fig and return_resid_table:
        return fig, axs, resid_frame
    elif return_fig:
        return fig, axs
    elif return_resid_table:
        return resid_frame
    elif return_fit_data:
        return fit_data_ret


def set_ramr_fitness_correction(
    manifest,
    data,
    resid_frame,
    auto_save=True,
    overwrite=False,
    plot_all_dates=True,
    return_plot=False,
    params=None,
    *,
    state_dir=None,
    state_tag=None,
):

    if params is None:
        params = ['lig_conc', 'fitness_effect', 'ref_fitness', 'early_fitness']

    min_samples_split = 5

    def model_plots(model, params, print_stats=True):
        plt.rcParams["figure.figsize"] = [7, 3]
        fig, axs = plt.subplots(1, 2, layout='tight')

        ax = axs[1]
        if plot_all_dates:
            df_label_list = (
                [[resid_frame, 'All data']]
                + [[df, label] for label, df in resid_frame.groupby('exp_date')]
            )
        else:
            df_label_list = [[resid_frame, 'All data']]

        for df, label in df_label_list:
            ms = 8 if label == 'All data' else 6
            alpha = 1 if label == 'All data' else 0.5

            y = df['resid']
            w = 1 / df['resid_err']**2
            X = df[params]

            predicted = model.predict(X)
            actual = y
            ax.plot(actual, predicted, 'o', ms=ms, alpha=alpha)

            if print_stats:
                resid_test = y - predicted
                rms_resid = np.sqrt(np.mean(resid_test**2))
                rms_pre = np.sqrt(np.mean(y**2))
                print(label)
                print(f'    RMS residual before model: {rms_pre:.4f}')
                print(f'    correction model RMS residual: {rms_resid:.4f}')
                print(f'    correction model R2: {model.score(X, y, sample_weight=w):.2f}')
                print()

        xlim = ax.get_xlim()
        ax.plot(xlim, xlim, '--k')
        ax.set_xlabel('actual deviation', size=16)
        ax.set_ylabel('predicted deviation', size=16)

        ax = axs[0]
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)
        pos = np.arange(len(sorted_idx)) + 0.5

        ax.barh(pos, importance[sorted_idx], align="center")
        ax.set_yticks(pos, np.array(params)[sorted_idx])
        ax.set_xlabel("Feature Importance (MDI)", size=14)

        return fig, axs

    # ----------------------------
    # Train model
    # ----------------------------
    y_train = resid_frame['resid']
    w_train = 1 / resid_frame['resid_err']**2
    X_train = resid_frame[params]

    ramr_model = GradientBoostingRegressor(
        loss="squared_error",
        min_samples_split=min_samples_split,
    )
    ramr_model.fit(X_train, y_train, sample_weight=w_train)

    fig, axs = model_plots(ramr_model, params, print_stats=True)

    # ----------------------------
    # Extract serializable artifacts
    # ----------------------------
    model_state = {
        "model_type": "GradientBoostingRegressor",
        "loss": "squared_error",
        "min_samples_split": min_samples_split,
        "features": list(params),
        "feature_importances": ramr_model.feature_importances_.tolist(),
        "n_train": int(len(X_train)),
        "training_r2": float(ramr_model.score(X_train, y_train, sample_weight=w_train)),
    }

    # Optional: store training predictions for audit/debug
    train_predictions = resid_frame[['resid', 'resid_err'] + params].copy()
    train_predictions['predicted_resid'] = ramr_model.predict(X_train)

    # ----------------------------
    # Update manifest (NO model object)
    # ----------------------------
    manifest['ramr_fitness_correction'] = {
        "params": list(params),
        "model_state": model_state,
    }

    # ----------------------------
    # Persist state
    # ----------------------------
    if auto_save:
        if state_dir is None:
            raise ValueError("state_dir must be provided when auto_save=True")

        tag = state_tag or "ramr_fitness_correction"

        save_state_v1(
            manifest=manifest,
            state_dir=state_dir,
            tables={
                f"{tag}_training_predictions": train_predictions,
            },
        )

    if return_plot:
        return fig, axs


def plot_count_ratios_vs_time(manifest, data, plot_range=None,
                                with_tet=None,
                                mark_samples=[],
                                show_spike_ins=None,
                                plot_samples=None):
    
    return plot_or_fit_barcode_ratios(manifest, data, plots_not_fits=True,
                                            plot_range=plot_range,
                                            show_spike_ins=show_spike_ins,
                                            plot_samples=plot_samples)
    

''' The plot_counts_vs_time() function needs to be updated or deleted. Comment it out for now.
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
    for r in fitness_utils.rows():
        for c in fitness_utils.columns():
            plate_list.append( int(2+(c-1)/3) )
            with_tet.append(r in fitness_utils.rows()[1::2])

    sample_plate_map = pd.DataFrame({"well": fitness_utils.wells()})
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
    
'''
        

def plot_chimera_plot(manifest, data,
                        save_plots=False,
                        chimera_cut_line=None,
                        plot_size=6, alpha=0.1, chimera_alpha=0.3):
        
    barcode_frame = data[data["possibleChimera"]]
    
    # Turn interactive plotting on or off depending on show_plots
    plt.ion()
    
    os.chdir(manifest.get('data_directory'))
    if save_plots:
        pdf_file = 'barcode fitness plots.pdf'
        pdf = PdfPages(pdf_file)
    
    plt.rcParams["figure.figsize"] = [plot_size, plot_size]
    fig, axs = plt.subplots(1, 1)
    axs.set_ylabel('Chimera Read Count per Sample')
    axs.set_xlabel('Geometric Mean of Parental Read Counts');
    axs.tick_params(labelsize=16);

    #axs.plot(np.sqrt(for_parent_count_list_96*rev_parent_count_list_96), chimera_count_list_96, 'o', ms=5,
    #        label="Individual Sample Counts");
    x = barcode_frame["parent_geo_mean"]/96
    y = barcode_frame["total_counts"]/96
    axs.plot(x, y, 'o', ms=7, alpha=alpha, label="Possible Chimeras, Total Counts ÷ 96");
    
    if "parent_geo_mean_p2" in barcode_frame.columns:
        x = barcode_frame["parent_geo_mean_p2"]/24
        y = barcode_frame["total_counts_plate_2"]/24
        axs.plot(x, y, 'o', ms=5, alpha=alpha, label="Total from Time Point 1 ÷ 24");
    
    if "isChimera" in barcode_frame.columns:
        plot_frame = barcode_frame[barcode_frame["isChimera"]]
        x = plot_frame["parent_geo_mean"]/96
        y = plot_frame["total_counts"]/96
        axs.plot(x, y, 'o', ms=5, alpha=chimera_alpha, label="Actual Chimeras, Total Counts ÷ 96");
    
    if ("parent_geo_mean_p2" in barcode_frame.columns) or ("isChimera" in barcode_frame.columns):
        leg = axs.legend(loc='upper left', bbox_to_anchor= (1.03, 0.97), ncol=1)
        #leg.get_frame().set_edgecolor('k');
        
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
def plot_hill_params(manifest, data, input_frames, in_labels=None, in_colors=None, in_alpha=0.7,
                        error_bars=True, log_ginf_err_cutoff=0.71, legend=True,
                        everything_color=None, box_size=6, ligand=None, plot_g0_vs_ginf=True):
    
    if in_labels is None:
        in_labels = [""] * len (input_frames)
    
    if in_colors is None:
        in_colors = [fitness_utils.gray_out("indigo")] * len (input_frames)
        
    if everything_color is None:
        everything_color = fitness_utils.gray_out("xkcd:tea green", s_factor=0.7, v_factor=0.8)
        
    plt.rcParams["figure.figsize"] = [2*box_size, 2*box_size]
    fig, axs_grid = plt.subplots(2, 2)
    axs = axs_grid.flatten()

    y_label_list = ["G0", "Ginf", "Ginf/G0", "n"]
    x_label_list = [f'EC50']*len(y_label_list)
    if ligand is None:
        lig_str = ''
    else:
        lig_str = f'_{ligand}'
    if 'log_g0' in data.columns.values:
        param_names = ["log_g0", f"log_ginf{lig_str}", f"log_ginf_g0_ratio{lig_str}", f"sensor_n{lig_str}"]
        x_param_list = [f"log_ec50{lig_str}"]*len(param_names)
        x_err_label_list = [f"log_ec50{lig_str}_err"]*len(param_names)
        if plot_g0_vs_ginf:
            param_names[-1] = f"log_ginf{lig_str}"
            x_param_list[-1] = "log_g0"
            x_err_label_list[-1] = "log_g0_err"
            y_label_list[-1] = "Ginf"
            x_label_list[-1] = "G0"
    else:
        param_names = ["log_low_level", "log_high_level", "log_high_low_ratio", "log_n"]
        x_param_list = [f'log_ic50']*len(param_names)
        x_err_label_list = [f'log_ic50 error']*len(param_names)


    # This part plots the input input_frames
    for input_frame, c, lab in zip(input_frames, in_colors, in_labels):
        for ax, name, x_param, x_err_label in zip(axs, param_names, x_param_list, x_err_label_list):
            y_err_label = f"{name}_err"
    
            params_x = input_frame[x_param]
            params_y = input_frame[name]
            err_x = input_frame[x_err_label]
            err_y = input_frame[y_err_label]
            
            yerr = err_y
            xerr = err_x
            xerr = fitness_utils.log_plot_errorbars(params_x, xerr)
            x = 10**params_x
            if plot_g0_vs_ginf or (ax is not axs[-1]):
                yerr = fitness_utils.log_plot_errorbars(params_y, yerr)
                y = 10**params_y
            else:
                y = params_y
            
            if error_bars:
                ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="o", ms=4, color=c,
                            label=lab, alpha=in_alpha);
            else:
                ax.plot(x, y, "o", ms=4, color=c,
                        label=lab, alpha=in_alpha);

    # This part plots all the rest
    plot_frame = data[3:]
    plot_frame = plot_frame[plot_frame["total_counts"]>3000]
    if 'log_g0' in data.columns.values:
        plot_frame = plot_frame[plot_frame[f"log_ginf{lig_str}_err"]<log_ginf_err_cutoff]
    else:
        plot_frame = plot_frame[plot_frame["log_high_level error"]<log_ginf_err_cutoff]
    
    for ax, name, x_param, y_label, x_label in zip(axs, param_names, x_param_list, y_label_list, x_label_list):
        params_x = plot_frame[x_param]
        params_y = plot_frame[name]
        
        x = 10**params_x
        if plot_g0_vs_ginf or (ax is not axs[-1]):
            y = 10**params_y
        else:
            y = params_y
        
        ax.set_xscale("log");
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(x, y, "o", ms=3, color=everything_color, zorder=0, alpha=0.3, label="everything");
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

def save_as_pickle(manifest, data, notebook_dir=None, experiment=None, pickle_file=None, overwrite=False):
    raise RuntimeError(
        "save_as_pickle is deprecated.\n"
        "This pipeline now uses structured state persistence.\n"
        "Use save_state_v1(...) instead."
    )

    if notebook_dir is None:
        notebook_dir = manifest.get('notebook_dir')
    if experiment is None:
        experiment = manifest.get('experiment')
    if pickle_file is None:
        pickle_file = experiment + '_BarSeqFitnessFrame.pkl'
        
    os.chdir(notebook_dir)
    
    # If file already exists, default is to rename old version instead of overwriting it.
    file_exists = os.path.isfile(pickle_file)
    if file_exists and (not overwrite):
        t = os.path.getmtime(pickle_file)
        t = datetime.datetime.fromtimestamp(t)
        t = f'{t}'
        t = t.replace(' ', '.')
        t = t.replace(':', '.')
        t = t[:16]
        old_pickle_file = pickle_file.replace('_BarSeqFitnessFrame.pkl', f'_BSF_{t}.pkl')
        os.rename(pickle_file, old_pickle_file)
        print(f"Previous version of BarSeqFitnessFrame renamed as: {old_pickle_file}")
    
    with open(pickle_file, 'wb') as f:
        pickle.dump({'manifest': manifest, 'data': data}, f)
    print(f"BarSeqFitnessFrame saved as: {pickle_file}")
    now = datetime.datetime.now()
    print(now)
    
def cleaned_frame(manifest, data, count_threshold=3000, log_ginf_error_cutoff=None, num_good_hill_points=12, exclude_mut_regions=None):
    frame = data
    frame = frame[frame["total_counts"]>count_threshold]
    
    #if log_ginf_error_cutoff is None:
    #    if self.experiment == '2019-10-16_IPTG_Select-DNA-5-plates':
    #        log_ginf_error_cutoff = 0.7
    
    if log_ginf_error_cutoff is not None:
        if 'log_ginf' in frame.columns.values:
            frame = frame[frame["log_ginf error"]<log_ginf_error_cutoff]
        else:
            frame = frame[frame["log_high_level error"]<log_ginf_error_cutoff]
    
    if num_good_hill_points > 0:
        frame = frame[frame["good_hill_fit_points"]>=num_good_hill_points]
    
    if exclude_mut_regions is None:
        if "pacbio_KAN_mutations" in frame.columns:
            exclude_mut_regions = ["KAN", "Ori", "tetA", "YFP", "insulator"]
        elif "KAN_1_confident_seq" in frame.columns:
            #exclude_mut_regions = ['empty_1', 'empty_4', 'insulator', 'KAN_1', 'KAN_2', 'Ori_1', 'Ori_2', 'tetA_1', 'tetA_2', 'YFP_1', 'YFP_2']
            exclude_mut_regions = ['insulator', 'tetA_1', 'tetA_2']
        elif "ramr_promoter_confident_seq" in frame.columns:
            exclude_mut_regions = ['ramr_promoter']
        else:
            exclude_mut_regions = []
        
    
    if len(exclude_mut_regions)>0:
        print(f'excluding the following regions with mutations: {exclude_mut_regions}')
    
    if "pacbio_KAN_mutations" in frame.columns:
        #This is for the original LacI experiment; we only rejected variants with known mutations in each reagion
        #    i.e., we kept variants without a sequence assignment for a region ("pacbio_" + reg + "_mutations" == -1 indicates no sequence assignment)
        for reg in exclude_mut_regions:
            frame = frame[frame["pacbio_" + reg + "_mutations"]<=0]
    elif "KAN_1_confident_seq" in frame.columns:
        #This is for the newer LacI experiment; we want to only keep variants with non-confident sequence assignments OR zero mutations in each of the regions
        #    If there is not a confident sequence assignment, the sequence is probably correct.
        for reg in exclude_mut_regions:
            frame = frame[(~frame[f"{reg}_confident_seq"])|(frame[f'{reg}_mutations']==0)]
    elif "amp_barcode_confident_seq" in frame.columns:
        #This is for the RamR experiments
        for reg in exclude_mut_regions:
            frame = frame[(~frame[f"{reg}_confident_seq"])|(frame[f'{reg}_mutations']==0)]

    return frame
    

def plot_hill_param_density_scatter(manifest, data, plot_frame=None, log_z=True, log_g=True, ligand='IPTG', box_size=4):
    if plot_frame is None:
        plot_frame = data
    
    cmap = fitness_utils.density_scatter_cmap()
    cmap = cmocean.tools.crop_by_percent(cmap, 10, which='min', N=None)

    bins = 51

    plt.rcParams["figure.figsize"] = [2*box_size, 2*box_size]
    fig, axs_grid = plt.subplots(2, 2)
    fig.suptitle(f'Hill Fit Parameters for {ligand}', size=20, y=0.92)
    axs = axs_grid.flatten()

    cb_ax = fig.add_axes([0.95, 0.12, 0.02, 0.75])

    log_params = ["log_g0", f"log_ginf_{ligand}", f"log_ginf_g0_ratio_{ligand}", f"log_ginf_{ligand}"]
    x_params = [f"log_ec50_{ligand}"]*3 + ["log_g0"]

    if log_g:
        y_axs_labels = ["$G_{0}$ (MEF)", "$G_{∞}$ (MEF)", "$G_{∞}$/$G_{0}$", "$G_{∞}$ (MEF)"]
        x_axs_labels = ["$EC_{50}$ (µmol/L)"]*3 + ["$G_{0}$ (MEF)"]
    else:
        y_axs_labels = ["$G_{0}$ (kMEF)", "$G_{∞}$ (kMEF)", "$G_{∞}$/$G_{0}$", "$G_{∞}$ (kMEF)"]
        x_axs_labels = ["$EC_{50}$ (µmol/L)"]*3 + ["$G_{0}$ (kMEF)"]

    log_y = True

    for zip_tuple in zip(axs, x_params, log_params, x_axs_labels, y_axs_labels):
        ax, x_par, log_par, x_axs_label, y_axs_label = zip_tuple

        thresh = 1000 #threshold_dict[log_par]
        x_thresh = 1000# threshold_dict[x_par]

        params_x = 10**plot_frame[x_par]
        sel = ~params_x.isnull()
        params_x = params_x[sel]

        params_y = 10**plot_frame[log_par]
        params_y = params_y[sel]

        err = plot_frame[f"{log_par}_err"][sel]
        x_err = plot_frame[f"{x_par}_err"][sel]

        sel = (err<thresh)&(x_err<x_thresh)
        params_x = np.array(params_x[sel])
        params_y = np.array(params_y[sel])

        hist_ret = fitness_utils.density_scatter_plot(params_x, params_y, ax=ax, sort=True, bins=bins, log_y=log_y,
                                                cmap=cmap, z_cutoff=8, alpha=0.5, rasterized=True)
        if ax==axs[1]:
            cbar = fig.colorbar(hist_ret[1], cax=cb_ax)#, ticks=[])
            cbar.solids.set(alpha=1)

            cb_ax.set_axisbelow(False)
            cb_ax.set_ylabel('Relative Density', rotation=270, labelpad=22)
            cb_ax.yaxis.set_ticks_position('right')
            cb_ax.yaxis.set_label_position('right')

        ax.set_xlabel(x_axs_label)
        ax.set_ylabel(y_axs_label)
        
        if log_g | (x_par==f"log_ec50_{ligand}"):
            ax.set_xscale("log")
        if log_g | (log_par==f"log_g0_ginf_ratio_{ligand}"):
            ax.set_yscale("log")
        
    if not log_g:
        axs[0].set_yticks([i*10000 for i in range(6)])
        axs[1].set_yticks([i*10000 for i in range(6)])
        axs[3].set_yticks([i*10000 for i in range(6)])
        axs[3].set_xticks([i*10000 for i in range(6)])
        axs[0].set_yticklabels([i*10 for i in range(6)])
        axs[1].set_yticklabels([i*10 for i in range(6)])
        axs[3].set_yticklabels([i*10 for i in range(6)])
        axs[3].set_xticklabels([i*10 for i in range(6)])
        
    x_shift = 0.05
    y_shift = 0.05
    for j, ax_row in enumerate(axs_grid):
        for i, ax in enumerate(ax_row):
            box = ax.get_position()
            box.x0 = box.x0 + x_shift*i
            box.x1 = box.x1 + x_shift*i
            box.y0 = box.y0 - y_shift*j
            box.y1 = box.y1 - y_shift*j
            ax.set_position(box)
    
    box = cb_ax.get_position()
    box.x0 = box.x0 + x_shift
    box.x1 = box.x1 + x_shift
    cb_ax.set_position(box)
    
    return axs, fig

def bs_frame_stan_data(
    manifest,
    data,
    st_row,
    old_style_columns=False,
    initial=None,
    is_gp_model=False,
    min_err=0.05,
    anti_list=None,
    apply_ramr_correction=None,
):

    # -----------------------------
    # helpers (local, minimal)
    # -----------------------------
    def _ensure_list(x, name):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        raise TypeError(f"{name} must be list-like, got {type(x)}")

    if initial is None:
        initial = get_default_initial(manifest)

    plasmid = manifest.get("plasmid")

    # ============================================================
    # Align-TF
    # ============================================================
    if plasmid == "Align-TF":
        sample_map = manifest.get("sample_plate_map")
        if sample_map is None:
            raise KeyError("sample_plate_map missing from manifest")

        sample_map = sample_map[
            sample_map.growth_plate == 2
        ].sort_values(by=["antibiotic_conc"])

        anti_conc_list = _ensure_list(anti_list, "anti_list")
        init_list = _ensure_list(initial, "initial")
        min_err_list = np.array(_ensure_list(min_err, "min_err"))

        rs_name = st_row.RS_name
        if "norm" in rs_name:
            return None

        tf = st_row.transcription_factor
        if rs_name != "":
            tf = align_tf_from_RS_name(rs_name)

        if tf == "":
            return None

        lig = align_ligand_from_tf(tf)
        sample_map = sample_map[sample_map.transcription_factor == tf]

        ligand_concentrations = np.unique(sample_map[lig])
        if len(ligand_concentrations) != 2:
            raise ValueError(
                f"Unexpected number of ligand concentrations for TF {tf}"
            )

        y_arr = []
        y_err_arr = []

        for lig_conc in ligand_concentrations:
            sample_map_lig = sample_map[sample_map[lig] == lig_conc]

            samples = []
            for c in anti_conc_list:
                df = sample_map_lig[sample_map_lig.antibiotic_conc == c]
                if len(df) != 1:
                    raise ValueError(
                        "Unexpected DataFrame length in bs_frame_stan_data()"
                    )
                samples.append(df.iloc[0]["sample_id"])

            ref_sample = sample_map_lig[
                sample_map_lig.antibiotic_conc == 0
            ]["sample_id"]
            if len(ref_sample) != 1:
                raise ValueError(
                    "Unexpected number of zero-antibiotic reference samples"
                )
            ref_sample = ref_sample.iloc[0]

            y = np.array(
                [st_row[f"fitness_S{s}_{i}"] for s, i in zip(samples, init_list)]
            )
            y_err = np.array(
                [
                    st_row[f"fitness_S{s}_err_{i}"]
                    for s, i in zip(samples, init_list)
                ]
            )

            y_ref = np.array(
                [st_row[f"fitness_S{ref_sample}_{i}"] for i in init_list]
            )
            y_ref_err = np.array(
                [st_row[f"fitness_S{ref_sample}_err_{i}"] for i in init_list]
            )

            y_err = np.sqrt(
                (y_err / y_ref) ** 2
                + (y * y_ref_err / y_ref ** 2) ** 2
                + min_err_list ** 2
            )
            y = (y - y_ref) / y_ref

            y_arr.append(y)
            y_err_arr.append(y_err)

        y_arr = np.array(y_arr)
        y_err_arr = np.array(y_err_arr)

        stan_data = {
            "ligand_id": lig,
            "ligand_concentrations": ligand_concentrations,
            "y": y_arr,
            "y_err": y_err_arr,
            "N_lig": len(ligand_concentrations),
            "N_antibiotic": len(y_arr[0]),
        }

        fit_fitness_difference_params = manifest.get(
            "fit_fitness_difference_params"
        )
        if fit_fitness_difference_params is None:
            raise KeyError("fit_fitness_difference_params missing from manifest")

        stan_data["low_fitness_mu"] = np.array(
            [
                fit_fitness_difference_params[i][c]["popt"][0]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["mid_g_mu"] = np.array(
            [
                fit_fitness_difference_params[i][c]["popt"][1]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["fitness_n_mu"] = np.array(
            [
                fit_fitness_difference_params[i][c]["popt"][2]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )

        stan_data["low_fitness_std"] = np.array(
            [
                fit_fitness_difference_params[i][c]["perr"][0]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["mid_g_std"] = np.array(
            [
                fit_fitness_difference_params[i][c]["perr"][1]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["fitness_n_std"] = np.array(
            [
                fit_fitness_difference_params[i][c]["perr"][2]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )

        log_g_min, log_g_max, _, _ = fitness_utils.log_g_limits(plasmid=plasmid)
        stan_data["log_g_min"] = log_g_min
        stan_data["log_g_max"] = log_g_max

        return stan_data

    # ============================================================
    # Align-T7RNAP_1
    # ============================================================
    if plasmid == "Align-T7RNAP_1":
        rs_name = st_row.RS_name
        if "norm" in rs_name.lower():
            return None

        sample_map = manifest.get("sample_plate_map")
        if sample_map is None:
            raise KeyError("sample_plate_map missing from manifest")

        sample_map = sample_map[
            sample_map.growth_plate == 2
        ].sort_values(by=["antibiotic_conc", "sample_id"])

        df_ref = sample_map[sample_map.antibiotic_conc == 0]
        ref_samples = np.array(df_ref["sample_id"])

        anti_conc_list = _ensure_list(anti_list, "anti_list")
        init_list = _ensure_list(initial, "initial")
        min_err_list = np.array(_ensure_list(min_err, "min_err"))

        y_arr = []
        y_err_arr = []

        for ref_sample in ref_samples:
            samp_num = ref_sample + 12
            y = np.array(
                [st_row[f"fitness_S{samp_num}_{i}"] for i in init_list]
            )
            y_err = np.array(
                [st_row[f"fitness_S{samp_num}_err_{i}"] for i in init_list]
            )

            y_ref = np.array(
                [st_row[f"fitness_S{ref_sample}_{i}"] for i in init_list]
            )
            y_ref_err = np.array(
                [st_row[f"fitness_S{ref_sample}_err_{i}"] for i in init_list]
            )

            y_err = np.sqrt(
                (y_err / y_ref) ** 2
                + (y * y_ref_err / y_ref ** 2) ** 2
                + min_err_list ** 2
            )
            y = (y - y_ref) / y_ref

            y_arr.append(y)
            y_err_arr.append(y_err)

        y_arr = np.array(y_arr)
        y_err_arr = np.array(y_err_arr)

        stan_data = {
            "y": y_arr,
            "y_err": y_err_arr,
            "N_lig": 12,
            "N_antibiotic": 1,
        }

        fit_fitness_difference_params = manifest.get(
            "fit_fitness_difference_params"
        )
        if fit_fitness_difference_params is None:
            raise KeyError("fit_fitness_difference_params missing from manifest")

        stan_data["low_fitness_mu"] = np.array(
            [
                fit_fitness_difference_params[i][c]["popt"][0]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["high_fitness_mu"] = np.array(
            [
                fit_fitness_difference_params[i][c]["popt"][1]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["mid_g_mu"] = np.array(
            [
                fit_fitness_difference_params[i][c]["popt"][2]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["fitness_n_mu"] = np.array(
            [
                fit_fitness_difference_params[i][c]["popt"][3]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )

        stan_data["low_fitness_std"] = np.array(
            [
                fit_fitness_difference_params[i][c]["perr"][0]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["high_fitness_std"] = np.array(
            [
                fit_fitness_difference_params[i][c]["perr"][1]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["mid_g_std"] = np.array(
            [
                fit_fitness_difference_params[i][c]["perr"][2]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )
        stan_data["fitness_n_std"] = np.array(
            [
                fit_fitness_difference_params[i][c]["perr"][3]
                for c, i in zip(anti_conc_list, init_list)
            ]
        )

        log_g_min, log_g_max, _, _ = fitness_utils.log_g_limits(plasmid=plasmid)
        stan_data["log_g_min"] = log_g_min
        stan_data["log_g_max"] = log_g_max

        return stan_data

    # ============================================================
    # All other plasmids: unchanged
    # ============================================================
    return get_stan_data(
        st_row=st_row,
        plot_df=manifest.get("sample_plate_map"),
        antibiotic_conc_list=manifest.get("antibiotic_conc_list"),
        lig_list=list(np.unique(manifest.get("sample_plate_map").ligand)),
        fit_fitness_difference_params=manifest.get(
            "fit_fitness_difference_params"
        ),
        old_style_columns=old_style_columns,
        initial=initial,
        plasmid=plasmid,
        is_gp_model=is_gp_model,
        min_err=min_err,
        ref_samples=manifest.get("ref_samples"),
        apply_ramr_correction=apply_ramr_correction,
        ramr_fitness_correction=manifest.get("ramr_fitness_correction"),
        ramr_fitness_correction_params=manifest.get(
            "ramr_fitness_correction_params"
        ),
        ramr_resid_frame=manifest.get("ramr_resid_frame"),
    )

def bs_frame_stan_data_(manifest, data, st_row,
                        old_style_columns=False, 
                        initial=None,
                        is_gp_model=False,
                        min_err=0.05,
                        anti_list=None,
                        apply_ramr_correction=None):
    
    if initial is None:
        initial = get_default_initial(manifest)
    
    if manifest.get('plasmid') == 'Align-TF':
        sample_map = manifest.get('sample_plate_map')
        sample_map = sample_map[sample_map.growth_plate==2].sort_values(by=['antibiotic_conc'])
            
        # For Align-TF project, initial, min_err, and anti_list are synchronized lists 
        #     of normalization/fit initials, min errors, and antibiotic concentrations
        anti_conc_list = anti_list
        init_list = initial
        min_err_list = np.array(min_err)
        
        # For Align-TF project, measurements at zero ligand and one non-zero ligand per TF
        rs_name = st_row.RS_name
        if 'norm' in rs_name:
            return None
        
        tf = st_row.transcription_factor
        if rs_name != '':
            tf = align_tf_from_RS_name(rs_name)
        
        if tf == '': # If the row does not have an identifiable transcription factor
            return None
        
        lig = align_ligand_from_tf(tf)
        sample_map = sample_map[sample_map.transcription_factor==tf]
        
        ligand_concentrations = np.unique(sample_map[lig])
        
        if len(ligand_concentrations) != 2:
            raise ValueError(f'Unexpected number of ligand concentrations for transcription_factor {tf}')
        
        y_arr = []
        y_err_arr = []
        for lig_conc in ligand_concentrations:
            sample_map_lig = sample_map[sample_map[lig]==lig_conc]
            
            samples = []
            for c in anti_conc_list:
                df = sample_map_lig
                df = df[df.antibiotic_conc==c]
                if len(df) == 1:
                    samples.append(df.iloc[0]['sample_id'])
                else:
                    raise ValueError(f'Unexpected DataFrame length in bs_frame_stan_data()')
            
            # Reference sample has zero antibiotic:
            ref_sample = sample_map_lig[sample_map_lig.antibiotic_conc==0]['sample_id']
            if len(ref_sample) == 1:
                ref_sample = ref_sample.iloc[0]
            else:
                raise ValueError(f'Unexpected number of samples with zero antibiotic: len(ref_sample): {len(ref_sample)}')
            
            y = np.array([st_row[f"fitness_S{s}_{init}"] for s, init in zip(samples, init_list)])
            y_err = np.array([st_row[f"fitness_S{s}_err_{init}"] for s, init in zip(samples, init_list)])
            
            y_ref = np.array([st_row[f"fitness_S{ref_sample}_{init}"] for init in init_list])
            y_ref_err = np.array([st_row[f"fitness_S{ref_sample}_err_{init}"] for init in init_list])
            
            y_err = np.sqrt((y_err/y_ref)**2 + (y*y_ref_err/y_ref**2)**2 + min_err_list**2)
            y = (y - y_ref)/y_ref
            
            y_arr.append(y)
            y_err_arr.append(y_err)
        y_arr = np.array(y_arr)
        y_err_arr = np.array(y_err_arr)
            
        stan_data = {'ligand_id':lig, 
                        'ligand_concentrations':ligand_concentrations, 
                        'y':y_arr, 
                        'y_err':y_err_arr,
                        'N_lig':len(ligand_concentrations),
                        'N_antibiotic':len(y_arr[0])}
        
        fit_fitness_difference_params = self.fit_fitness_difference_params
        
        low_fitness_mu = np.array([fit_fitness_difference_params[init][c]['popt'][0] for c, init in zip(anti_conc_list, init_list)])
        stan_data['low_fitness_mu'] = low_fitness_mu
        
        mid_g_mu = np.array([fit_fitness_difference_params[init][c]['popt'][1] for c, init in zip(anti_conc_list, init_list)])
        stan_data['mid_g_mu'] = mid_g_mu
        
        fitness_n_mu = np.array([fit_fitness_difference_params[init][c]['popt'][2] for c, init in zip(anti_conc_list, init_list)])
        stan_data['fitness_n_mu'] = fitness_n_mu
        
        low_fitness_std = np.array([fit_fitness_difference_params[init][c]['perr'][0] for c, init in zip(anti_conc_list, init_list)])
        stan_data['low_fitness_std'] = low_fitness_std
        
        mid_g_std = np.array([fit_fitness_difference_params[init][c]['perr'][1] for c, init in zip(anti_conc_list, init_list)])
        stan_data['mid_g_std'] = mid_g_std
        
        fitness_n_std = np.array([fit_fitness_difference_params[init][c]['perr'][2] for c, init in zip(anti_conc_list, init_list)])
        stan_data['fitness_n_std'] = fitness_n_std
        
        log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf = fitness_utils.log_g_limits(plasmid=self.plasmid)
        stan_data['log_g_min'] = log_g_min
        stan_data['log_g_max'] = log_g_max
        
        return stan_data
    
    
    
    if manifest.get('plasmid') == 'Align-T7RNAP_1':
        rs_name = st_row.RS_name
        if 'norm' in rs_name.lower():
            return None
        sample_map = manifest.get('sample_plate_map')
        sample_map = sample_map[sample_map.growth_plate==2].sort_values(by=['antibiotic_conc', 'sample_id'])
        
        df_ref = sample_map[sample_map.antibiotic_conc==0]
        ref_samples = np.array(df_ref['sample_id'])
            
        # For Align-T7RNAP_1 project, initial, min_err, and anti_list are synchronized lists 
        #     of normalization/fit initials, min errors, and antibiotic concentrations
        anti_conc_list = anti_list
        init_list = initial
        min_err_list = np.array(min_err)
        
        # For Align-T7RNAP_1 project, there is no induction.
        y_arr = []
        y_err_arr = []
        # Use the N_lig parameter in the Stan model to represent the 12 different replicate measurements (all at zero ligand)
        for ref_sample in ref_samples:
            samp_num = ref_sample + 12 # Pair the samples with and without antibiotic by column in the growth plate
            
            samples = [samp_num]
            
            y = np.array([st_row[f"fitness_S{s}_{init}"] for s, init in zip(samples, init_list)])
            y_err = np.array([st_row[f"fitness_S{s}_err_{init}"] for s, init in zip(samples, init_list)])
            
            y_ref = np.array([st_row[f"fitness_S{ref_sample}_{init}"] for init in init_list])
            y_ref_err = np.array([st_row[f"fitness_S{ref_sample}_err_{init}"] for init in init_list])
            
            y_err = np.sqrt((y_err/y_ref)**2 + (y*y_ref_err/y_ref**2)**2 + min_err_list**2)
            y = (y - y_ref)/y_ref
            
            y_arr.append(y)
            y_err_arr.append(y_err)
        
        y_arr = np.array(y_arr)
        y_err_arr = np.array(y_err_arr)
            
        stan_data = {'y':y_arr, 
                        'y_err':y_err_arr,
                        'N_lig':12,
                        'N_antibiotic':1}
        
        fit_fitness_difference_params = manifest.get('fit_fitness_difference_params')
        
        low_fitness_mu = np.array([fit_fitness_difference_params[init][c]['popt'][0] for c, init in zip(anti_conc_list, init_list)])
        stan_data['low_fitness_mu'] = low_fitness_mu
        
        high_fitness_mu = np.array([fit_fitness_difference_params[init][c]['popt'][1] for c, init in zip(anti_conc_list, init_list)])
        stan_data['high_fitness_mu'] = high_fitness_mu
        
        mid_g_mu = np.array([fit_fitness_difference_params[init][c]['popt'][2] for c, init in zip(anti_conc_list, init_list)])
        stan_data['mid_g_mu'] = mid_g_mu
        
        fitness_n_mu = np.array([fit_fitness_difference_params[init][c]['popt'][3] for c, init in zip(anti_conc_list, init_list)])
        stan_data['fitness_n_mu'] = fitness_n_mu
        
        low_fitness_std = np.array([fit_fitness_difference_params[init][c]['perr'][0] for c, init in zip(anti_conc_list, init_list)])
        stan_data['low_fitness_std'] = low_fitness_std
        
        high_fitness_std = np.array([fit_fitness_difference_params[init][c]['perr'][1] for c, init in zip(anti_conc_list, init_list)])
        stan_data['high_fitness_std'] = high_fitness_std
        
        mid_g_std = np.array([fit_fitness_difference_params[init][c]['perr'][2] for c, init in zip(anti_conc_list, init_list)])
        stan_data['mid_g_std'] = mid_g_std
        
        fitness_n_std = np.array([fit_fitness_difference_params[init][c]['perr'][3] for c, init in zip(anti_conc_list, init_list)])
        stan_data['fitness_n_std'] = fitness_n_std
        
        log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf = fitness_utils.log_g_limits(plasmid=manifest.get('plasmid'))
        stan_data['log_g_min'] = log_g_min
        stan_data['log_g_max'] = log_g_max
        
        return stan_data
    
    
    if manifest.get('plasmid') == 'pRamR':
        if apply_ramr_correction is None:
            apply_ramr_correction = True
    
    sample_plate_map = manifest.get('sample_plate_map')
    lig_list = list(np.unique(sample_plate_map.ligand))
    if 'none' in lig_list:
        lig_list.remove('none')
    
    plot_df = sample_plate_map
    plot_df = plot_df[plot_df.growth_plate==2].sort_values(by=lig_list)
    
    
    plasmid = manifest.get('plasmid')
    antibiotic_conc_list = manifest.get('antibiotic_conc_list')
    fit_fitness_difference_params = manifest.get('fit_fitness_difference_params')
    
    ramr_fitness_correction = manifest.get('ramr_fitness_correction', None)
    ramr_fitness_correction_params = manifest.get('ramr_fitness_correction_params', None)
    ramr_resid_frame = manifest.get('ramr_resid_frame', None)
    
    stan_data =  get_stan_data(st_row=st_row, plot_df=plot_df, antibiotic_conc_list=antibiotic_conc_list, 
                                lig_list=lig_list, fit_fitness_difference_params=fit_fitness_difference_params, 
                                old_style_columns=old_style_columns, initial=initial, plasmid=plasmid,
                                is_gp_model=is_gp_model,
                                min_err=min_err, 
                                ref_samples=manifest.get('ref_samples'),
                                apply_ramr_correction=apply_ramr_correction,
                                ramr_fitness_correction=ramr_fitness_correction,
                                ramr_fitness_correction_params=ramr_fitness_correction_params,
                                ramr_resid_frame=ramr_resid_frame,
                                )
                    
                    
    if plasmid == 'pVER':
        if len(manifest.get('antibiotic_conc_list')) == 2:
            # Original 2019 experiment, with single ligand and single antibiotic
            x_min = stan_data['x']
            x_min = x_min[x_min>0]
            if len(x_min)>0:
                x_min = min(x_min)
                x_max = max(stan_data['x'])
            else:
                x_min = 1
                x_max = 1000
            log_x_max = 2*np.log10(x_max) - np.log10(x_min) + 1
            stan_data['log_x_max'] = np.array([log_x_max])
        elif len(manifest.get('antibiotic_conc_list')) == 3:
            # 2022 experimetn, with two ligands and two antibiotic concentrations
            log_x_max_arr = []
            for k  in ['x_1', 'x_2']:
                x_min = stan_data[k]
                x_min = x_min[x_min>0]
                if len(x_min)>0:
                    x_min = min(x_min)
                else:
                    x_min = 1
                x_max = max(stan_data[k])
                log_x_max = 2*np.log10(x_max) - np.log10(x_min) + 0.2
                log_x_max_arr.append(log_x_max)
            log_x_max_arr = np.array(log_x_max_arr)
            stan_data['log_x_max'] = log_x_max_arr
    elif plasmid in ['pRamR', 'pCymR']:
        log_x_max_arr = []
        for k  in ['x_1', 'x_2', 'x_3']:
            x_min = stan_data[k]
            x_min = x_min[x_min>0]
            if len(x_min)>0:
                x_min = min(x_min)
            else:
                x_min = 1
            x_max = max(stan_data[k])
            log_x_max = 2*np.log10(x_max) - np.log10(x_min) + 0.2
            log_x_max_arr.append(log_x_max)
        log_x_max_arr = np.array(log_x_max_arr)
        stan_data['log_x_max'] = log_x_max_arr
    
    if plasmid == 'pCymR':
        stan_data['zero_spacing_factor'] = np.array([3]*3)
    
    return stan_data


def get_default_initial(manifest):
    plasmid = manifest.get('plasmid')
    if plasmid == 'pVER':
        initial = 'b'
    elif plasmid == 'pRamR':
        initial = 'sp01'
    elif plasmid == 'pCymR':
        initial = 'sp09'
    elif plasmid == 'Align-TF':
        initial = 'laci'
    elif plasmid == 'Align-Protease':
        initial = 'nrm03'
    elif plasmid == 'Align-T7RNAP_1':
        initial = 'nrm03'
    
    return initial


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
    
    low_fitness = fit_fitness_difference_params[0][0]
    mid_g = fit_fitness_difference_params[0][1]
    fitness_n = fit_fitness_difference_params[0][2]
    
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
                
def init_stan_fit_three_ligand(stan_data, fit_fitness_difference_params, plasmid='pRamR'):
    min_ic = np.log10(min(stan_data['x_1']))
    max_ic = np.log10(max(stan_data['x_1']))
    log_ec50_1 = np.random.uniform(min_ic, max_ic)
    log_ec50_2 = np.random.uniform(min_ic, max_ic)
    log_ec50_3 = np.random.uniform(min_ic, max_ic)
    
    n_1 = np.random.uniform(1.3, 1.7)
    n_2 = np.random.uniform(1.3, 1.7)
    n_3 = np.random.uniform(1.3, 1.7)
    
    sig = np.random.uniform(1, 3)
    
    # Indices for x_y_s_list[ligand][tet][x,y,s][n]
    ret_dict = dict(log_g0=log_level(np.mean(stan_data['y_0']), plasmid=plasmid), 
                    log_ginf_1=log_level(np.mean(stan_data['y_1'][-2:]), plasmid=plasmid), 
                    log_ginf_2=log_level(np.mean(stan_data['y_2'][-2:]), plasmid=plasmid), 
                    log_ginf_3=log_level(np.mean(stan_data['y_3'][-2:]), plasmid=plasmid), 
                    log_ec50_1=log_ec50_1, 
                    log_ec50_2=log_ec50_2, 
                    log_ec50_3=log_ec50_3, 
                    sensor_n_1=n_1, 
                    sensor_n_2=n_2, 
                    sensor_n_3=n_3, 
                    sigma=sig, 
                    mid_g=fit_fitness_difference_params[0][1],
                    fitness_n=fit_fitness_difference_params[0][2],
                    )
    if plasmid == 'pRamR':
        ret_dict['high_fitness'] = fit_fitness_difference_params[0][0]
    else:
        ret_dict['low_fitness'] = fit_fitness_difference_params[0][0]
    return ret_dict

def init_stan_fit_single_point(stan_data):
    sig = np.random.uniform(1, 3)
    
    init_dict = dict(sigma=sig, 
                low_fitness=stan_data['low_fitness_mu'],
                mid_g=stan_data['mid_g_mu'],
                fitness_n=stan_data['fitness_n_mu']
                )
    
    if 'high_fitness_mu' in stan_data:
        init_dict['high_fitness'] = stan_data['high_fitness_mu']
        
    return init_dict
    
    
def init_stan_GP_fit(fit_fitness_difference_params, single_tet, single_ligand, plasmid='pVER'):
    sig = np.random.uniform(1, 3)
    rho = np.random.uniform(0.9, 1.1)
    alpha = np.random.uniform(0.009, 0.011)
    
    if plasmid == 'pVER':
        if single_tet:
            low_fitness = fit_fitness_difference_params[0][0]
            mid_g = fit_fitness_difference_params[0][1]
            fitness_n = fit_fitness_difference_params[0][2]
            
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
    elif plasmid == 'pRamR':
        return dict(sigma=sig, rho=rho, alpha=alpha,
                    high_fitness=fit_fitness_difference_params[0][0],
                    mid_g=fit_fitness_difference_params[0][1],
                    fitness_n=fit_fitness_difference_params[0][2],
                    )
    elif plasmid == 'pCymR':
        return dict(sigma=sig, rho=rho, alpha=alpha,
                    low_fitness=fit_fitness_difference_params[0][0],
                    mid_g=fit_fitness_difference_params[0][1],
                    fitness_n=fit_fitness_difference_params[0][2],
                    )
    
    
def log_level(fitness_difference, plasmid='pVER'):
    if plasmid == 'pVER':
        log_g = 1.439*fitness_difference + 3.32
        log_g = log_g*np.random.uniform(0.9,1.1)
        if log_g<1.5:
            log_g = 1.5
        if log_g>4:
            log_g = 4
        return log_g
    elif plasmid == 'pRamR':
        log_g = -2.1*fitness_difference/1.5 + 2
        log_g = log_g*np.random.uniform(0.9,1.1)
        if log_g<2:
            log_g = 2
        if log_g>4.5:
            log_g = 4.5
        return log_g
    elif plasmid == 'pCymR':
        log_g = np.log10(200)*(1 + fitness_difference)
        log_g = log_g*np.random.uniform(0.9,1.1)
        if log_g<0:
            log_g = 0
        if log_g>np.log10(300):
            log_g = np.log10(300)
        return log_g
        
def get_stan_data(st_row, plot_df, antibiotic_conc_list, 
                  lig_list, fit_fitness_difference_params, 
                  old_style_columns=False, initial="b", plasmid="pVER",
                  is_gp_model=False,
                  min_err=0.05,
                  ref_samples=None,
                  apply_ramr_correction=True,
                  ramr_fitness_correction=None,
                  ramr_fitness_correction_params=None,
                  ramr_resid_frame=None,
                  ):
    
    log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf = fitness_utils.log_g_limits(plasmid=plasmid)
    
    antibiotic_conc_list = np.array(antibiotic_conc_list)
    
    spike_in = fitness_utils.get_spike_in_name_from_inital(plasmid, initial)
    
    if old_style_columns:
        high_tet = antibiotic_conc_list[1]
        
        y_zero = st_row[f"fitness_{0}_estimate_{initial}"]
        s_zero = st_row[f"fitness_{0}_err_{initial}"]
        y_high = st_row[f"fitness_{high_tet}_estimate_{initial}"]
        s_high = st_row[f"fitness_{high_tet}_err_{initial}"]
        
        y = (y_high - y_zero)/y_zero
        s = np.sqrt( s_high**2 + s_zero**2 )/y_zero
        
        x_fit = x
        x_y_s_list = [[x_fit, y, s]]
    else:
        y = [st_row[f"fitness_S{i}_{initial}"] for i in ref_samples]
        s = [st_row[f"fitness_S{i}_err_{initial}"] for i in ref_samples]
        y_ref_list = np.array(y)
        s_ref_list = np.array(s)
        
        sel = ~np.isnan(y_ref_list)
        if len(sel[sel])>0:
            y_ref_list = y_ref_list[sel]
            s_ref_list = s_ref_list[sel]
        
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
                df = df.sort_values(by=lig)
                x = np.array(df[lig])
                samples = np.array(df.sample_id)
                # Correction factor for non-constant ref fitness (i.e., fitness decreases with [ligand]
                ref_correction = np.array([fitness_utils.ref_fit_correction(z, plasmid, ligand=lig, spike_in=spike_in) for z in x])
                y = np.array([st_row[f"fitness_S{i}_{initial}"] for i in df.sample_id])
                raw_fitness = y.copy()
                s = np.array([st_row[f"fitness_S{i}_err_{initial}"] for i in df.sample_id])
                
                if plasmid in ['pVER', 'pCymR']:
                    y = (y - y_ref*ref_correction)/(y_ref*ref_correction)
                    s = np.sqrt(s**2 + (s_ref*ref_correction)**2)/(y_ref*ref_correction)
                if plasmid == 'pRamR':
                    y = (y - y_ref)/(y_ref*ref_correction)
                    s = np.sqrt(s**2 + s_ref**2)/(y_ref*ref_correction)
                    early_fitness = np.array([st_row[f"fitness_S{i}_ea.{initial}"] for i in df.sample_id])
                    
                    # Ligand effects on fitness make the measurements at the highest concentration less reliable
                    s[x>=500] *= 2
                    
                    if apply_ramr_correction:
                        # Calibration correction for RamR system
                        ramr_model = ramr_fitness_correction
                        if ramr_model is None:
                            raise Exception('RamR calibration correction model (ramr_fitness_correction) is None')
                            
                        params = ramr_fitness_correction_params
                        if ramr_model is None:
                            raise Exception('RamR calibration correction parameters (ramr_fitness_correction_params) is None')
                        
                        df = df.copy()
                        df['lig_conc'] = x
                        df['fitness_effect'] = y
                        df['ref_fitness'] = [y_ref]*len(x)
                        df['early_fitness'] = np.array([st_row[f"fitness_S{i}_ea.{initial}"] for i in df.sample_id])

                        X_test = df[params]
                        X_test = X_test.dropna()
                        
                        y_corr = ramr_model.predict(X_test)
                        y = y - y_corr
                
                s = np.sqrt(s**2 + min_err**2)
                
                if is_gp_model:
                    # For GP model, can't have missing data. So, if either y or s is nan, replace with values that won't affect GP model results (i.e. s=100)
                    invalid = (np.isnan(y) | np.isnan(s))
                    if len(tet_list) == 1:
                        middle_fitness = fit_fitness_difference_params[0][0]/2
                    elif len(tet_list) == 2:
                        middle_fitness = (fit_fitness_difference_params[0][0] + fit_fitness_difference_params[1][0])/4
                    y[invalid] = middle_fitness
                    s[invalid] = 100
                else:
                    valid = ~(np.isnan(y) | np.isnan(s))
                    x = x[valid]
                    y = y[valid]
                    s = s[valid]
                    samples = samples[valid]
                
                sub_list.append([x, y, s, samples])
            x_y_s_list.append(sub_list)
            
        if len(lig_list) == 1:
            # Case for single ligand and single antibiotic concentration
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = np.full((1, 6), np.nan)
    
            low_fitness = fit_fitness_difference_params[0][0]
            mid_g = fit_fitness_difference_params[0][1]
            fitness_n = fit_fitness_difference_params[0][2]
            
            x = x_y_s_list[0][0][0]
            y = x_y_s_list[0][0][1]
            y_err = x_y_s_list[0][0][2]
            samp = x_y_s_list[0][0][3]
            
            stan_data = dict(x=x, y=y, N=len(y), y_err=y_err,
                             low_fitness_mu=low_fitness, mid_g_mu=mid_g, fitness_n_mu=fitness_n,
                             log_g_min=log_g_min, log_g_max=log_g_max, log_g_prior_scale=log_g_prior_scale,
                             y_ref=y_ref, samp=samp)
        
        elif (len(lig_list) == 2) and (len(tet_list) == 2):
            # Case for two-tet, two-ligand (e.g., LacI with high and low tet)
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = np.full((2, 6), np.nan)
                
            y_0_med = x_y_s_list[0][0][1][0]
            s_0_med = x_y_s_list[0][0][2][0]
            samp_0_med = x_y_s_list[0][0][3][0]
            
            x_1 = x_y_s_list[0][0][0]
            y_1_med = x_y_s_list[0][0][1]
            s_1_med = x_y_s_list[0][0][2]
            samp_1_med = x_y_s_list[0][0][3]
            y_1_med = y_1_med[x_1>0]
            s_1_med = s_1_med[x_1>0]
            samp_1_med = samp_1_med[x_1>0]
            x_1 = x_1[x_1>0]
            
            x_1_high = x_y_s_list[0][1][0]
            y_1_high = x_y_s_list[0][1][1]
            s_1_high = x_y_s_list[0][1][2]
            samp_1_high = x_y_s_list[0][1][3]
            y_1_high = y_1_high[x_1_high>0]
            s_1_high = s_1_high[x_1_high>0]
            samp_1_high = samp_1_high[x_1_high>0]
            x_1_high = x_1_high[x_1_high>0]
            
            x_2 = x_y_s_list[1][0][0]
            y_2_med = x_y_s_list[1][0][1]
            s_2_med = x_y_s_list[1][0][2]
            samp_2_med = x_y_s_list[1][0][3]
            y_2_med = y_2_med[x_2>0]
            s_2_med = s_2_med[x_2>0]
            samp_2_med = samp_2_med[x_2>0]
            x_2 = x_2[x_2>0]
            
            x_2_high = x_y_s_list[1][1][0]
            y_2_high = x_y_s_list[1][1][1]
            s_2_high = x_y_s_list[1][1][2]
            samp_2_high = x_y_s_list[1][1][3]
            y_2_high = y_2_high[x_2_high>0]
            s_2_high = s_2_high[x_2_high>0]
            samp_2_high = samp_2_high[x_2_high>0]
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
                             y_ref=y_ref,
                             samp_0_low_tet=samp_0_med, 
                             samp_1_low_tet=samp_1_med,
                             samp_2_low_tet=samp_2_med,
                             samp_1_high_tet=samp_1_high,
                             samp_2_high_tet=samp_2_high,
                             )
        
        elif (len(lig_list) == 3) and (len(tet_list) == 1):
            # Case for three-ligand experiment (e.g., RamR)
            # x_y_s_list: 1st index is the ligand (0, 1, or 2)
            #             2nd index is the antibiotic concentration (always 0 here)
            #             3rd index is 0 for x, 1 for y, 2 for s
            #             4th index is for individual data points
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = np.full((1, 6), np.nan)
            
            x_1, y_1, s_1, samp_1 = tuple(x_y_s_list[0][0][n] for n in range(4))
            y_0 = y_1[x_1==0]
            s_0 = s_1[x_1==0]
            samp_0 = samp_1[x_1==0]
            
            y_1 = y_1[x_1>0]
            s_1 = s_1[x_1>0]
            samp_1 = samp_1[x_1>0]
            x_1 = x_1[x_1>0]
            
            x_2, y_2, s_2, samp_2 = tuple(x_y_s_list[1][0][n] for n in range(4))
            y_2 = y_2[x_2>0]
            s_2 = s_2[x_2>0]
            samp_2 = samp_2[x_2>0]
            x_2 = x_2[x_2>0]
            
            x_3, y_3, s_3, samp_3 = tuple(x_y_s_list[2][0][n] for n in range(4))
            y_3 = y_3[x_3>0]
            s_3 = s_3[x_3>0]
            samp_3 = samp_3[x_3>0]
            x_3 = x_3[x_3>0]
            
            stan_data = dict(N_lig=len(x_1),
                             y_0=y_0, y_0_err=s_0,
                             x_1=x_1, y_1=y_1, y_1_err=s_1,
                             x_2=x_2, y_2=y_2, y_2_err=s_2,
                             x_3=x_3, y_3=y_3, y_3_err=s_3,
                             samp_0=samp_0, samp_1=samp_1, samp_2=samp_2, samp_3=samp_3,
                             log_g_min=log_g_min, log_g_max=log_g_max, log_g_prior_scale=log_g_prior_scale,
                             mid_g_mu=fit_fitness_difference_params[0][1],
                             fitness_n_mu=fit_fitness_difference_params[0][2],
                             mid_g_std=fit_fitness_difference_params[0][4],
                             fitness_n_std=fit_fitness_difference_params[0][5],
                             y_ref=y_ref,
                             )
            if plasmid == 'pRamR':
                stan_data['high_fitness_mu'] = fit_fitness_difference_params[0][0]
                stan_data['high_fitness_std'] = fit_fitness_difference_params[0][3]
            else:
                stan_data['low_fitness_mu'] = fit_fitness_difference_params[0][0]
                stan_data['low_fitness_std'] = fit_fitness_difference_params[0][3]
                             
    return stan_data

def align_tf_from_ligand(lig):
    # TODO: edit this to use sample_plate_map
    if lig == 'IPTG':
        return 'LacI'
    if lig == '1S-TIQ':
        return 'RamR'
    if lig == 'Van':
        return 'VanR'

def align_ligand_from_tf(tf):
    # TODO: edit this to use sample_plate_map
    if tf == 'LacI':
        return 'IPTG'
    if tf == 'RamR':
        return '1S-TIQ'
    if tf == 'VanR':
        return 'Van'

def align_tf_from_RS_name(rs):
    if 'LacI' in rs:
        return 'LacI'
    if 'RamR' in rs:
        return 'RamR'
    if 'VanR' in rs:
        return 'VanR'
    
