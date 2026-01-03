"""
Example Usage of Refactored BarSeqFitnessFrame Functions

This demonstrates how to use the new functional API with manifest-based state management.
"""

from gsf_ims_fitness.BarSeqFitnessFrame import (
    # Manifest I/O
    load_manifest,
    save_manifest,
    initialize_manifest_and_load_data,
    load_barcode_frame,
    save_barcode_frame,
    load_sample_plate_map,
    save_sample_plate_map,
    
    # Data Processing
    trim_and_sum_barcodes_func,
    label_reference_sequences_func,
    flag_possible_chimeras_func,
    mark_actual_chimeras_func,
    merge_barcodes_func,
    set_sample_plate_map_func,
    
    # Fitness Calculation
    add_fitness_from_slopes_func,
    set_fit_fitness_difference_params_func,
    
    # Utilities
    get_default_initial_func,
    cleaned_frame_func,
)

# ========================================================================================
# EXAMPLE 1: Initialize New Experiment
# ========================================================================================

def example_initialize():
    """Initialize a new experiment from scratch."""
    
    # Initialize manifest and load data
    manifest, barcode_frame = initialize_manifest_and_load_data(
        notebook_dir='/path/to/experiment/data',
        experiment='Align-TF_GBA_1',
        plasmid='Align-TF',
        antibiotic_conc_list=[0, 0.3, 3.0, 6.0],
        inducer_conc_lists=[[0, 2000], [0, 250], [0, 100]],
        ligand_list=['IPTG', '1S-TIQ', 'Van'],
        ref_samples=[1, 9, 13, 17, 21],
        min_read_count=200,
        single_barcode=True,
        merge_dist_cutoff=3
    )
    
    # Save initial state
    save_manifest(manifest, 'manifest.yaml')
    save_barcode_frame(manifest, barcode_frame)
    
    print(f"Initialized experiment: {manifest['metadata']['experiment_id']}")
    print(f"Number of barcodes: {len(barcode_frame)}")
    
    return manifest, barcode_frame


# ========================================================================================
# EXAMPLE 2: Load Existing Experiment and Process
# ========================================================================================

def example_process_existing():
    """Load an existing experiment and process it."""
    
    # Load manifest and data
    manifest = load_manifest('manifest.yaml')
    barcode_frame = load_barcode_frame(manifest)
    
    print(f"Loaded experiment: {manifest['metadata']['experiment_id']}")
    print(f"Number of barcodes: {len(barcode_frame)}")
    
    # Trim and sum barcodes
    manifest, barcode_frame = trim_and_sum_barcodes_func(
        manifest, barcode_frame, 
        cutoff=1000, 
        auto_save=True
    )
    print(f"After trimming: {len(barcode_frame)} barcodes")
    
    # Label reference sequences
    manifest, barcode_frame = label_reference_sequences_func(
        manifest, barcode_frame,
        show_output=False,
        auto_save=True
    )
    
    # Flag possible chimeras
    barcode_frame = flag_possible_chimeras_func(
        manifest, barcode_frame,
        use_faster_search=True
    )
    
    # Mark actual chimeras (using a cutoff function)
    def chimera_cutoff(geo_mean):
        return 0.8 * geo_mean
    
    manifest, barcode_frame = mark_actual_chimeras_func(
        manifest, barcode_frame,
        chimera_cut_line=chimera_cutoff,
        auto_save=True
    )
    
    return manifest, barcode_frame


# ========================================================================================
# EXAMPLE 3: Fitness Calculation Workflow
# ========================================================================================

def example_fitness_workflow():
    """Complete fitness calculation workflow."""
    
    # Load experiment
    manifest = load_manifest('manifest.yaml')
    barcode_frame = load_barcode_frame(manifest)
    
    # Set fitness difference parameters
    manifest = set_fit_fitness_difference_params_func(
        manifest,
        auto_save=True
    )
    
    # Add fitness from slopes
    # (Assumes slopes have been calculated previously)
    manifest, barcode_frame = add_fitness_from_slopes_func(
        manifest, barcode_frame,
        initial=None,  # Will use default
        auto_save=True
    )
    
    print("Fitness values added to barcode frame")
    
    return manifest, barcode_frame


# ========================================================================================
# EXAMPLE 4: Get Cleaned Data Frame
# ========================================================================================

def example_cleaned_frame():
    """Get a cleaned version of the barcode frame."""
    
    # Load experiment
    manifest = load_manifest('manifest.yaml')
    barcode_frame = load_barcode_frame(manifest)
    
    # Get cleaned frame with quality filters
    cleaned = cleaned_frame_func(
        manifest, barcode_frame,
        count_threshold=3000,
        log_ginf_error_cutoff=0.7,
        num_good_hill_points=12
    )
    
    print(f"Original barcodes: {len(barcode_frame)}")
    print(f"After cleaning: {len(cleaned)}")
    
    return cleaned


# ========================================================================================
# EXAMPLE 5: Merge Barcodes
# ========================================================================================

def example_merge_barcodes():
    """Merge barcodes that should be combined."""
    
    # Load experiment
    manifest = load_manifest('manifest.yaml')
    barcode_frame = load_barcode_frame(manifest)
    
    # Merge small barcodes into larger one
    # (Example: merge indices 100, 101, 102 into index 50)
    manifest, barcode_frame = merge_barcodes_func(
        manifest, barcode_frame,
        small_bc_index_list=[100, 101, 102],
        big_bc_index=50,
        auto_save=True
    )
    
    # After merging, you should re-run trim_and_sum_barcodes
    manifest, barcode_frame = trim_and_sum_barcodes_func(
        manifest, barcode_frame,
        auto_save=True
    )
    
    return manifest, barcode_frame


# ========================================================================================
# EXAMPLE 6: Work with Manifest Directly
# ========================================================================================

def example_manifest_operations():
    """Examples of working with manifest directly."""
    
    # Load manifest
    manifest = load_manifest('manifest.yaml')
    
    # Access configuration
    plasmid = manifest['system']['plasmid']
    ref_samples = manifest['experiment']['ref_samples']
    antibiotic_concs = manifest['experiment']['antibiotics'][0]['concentrations']
    
    print(f"Plasmid: {plasmid}")
    print(f"Reference samples: {ref_samples}")
    print(f"Antibiotic concentrations: {antibiotic_concs}")
    
    # Check processing status
    if manifest['processing'].get('clustering', {}).get('completed'):
        print("Clustering completed")
    
    # Get QC metrics
    total_barcodes = manifest['qc'].get('total_raw_barcodes')
    after_qc = manifest['qc'].get('barcodes_after_qc')
    
    print(f"Total barcodes: {total_barcodes}")
    print(f"After QC: {after_qc}")
    
    # Modify manifest and save
    manifest['metadata']['notes'] = 'Updated processing parameters'
    save_manifest(manifest, 'manifest.yaml')


# ========================================================================================
# EXAMPLE 7: Template-Based Initialization
# ========================================================================================

def example_template_initialization():
    """Initialize using a template manifest."""
    
    # Load template
    template = load_manifest('manifest_example.yaml')
    
    # Initialize with template (overrides specific values)
    manifest, barcode_frame = initialize_manifest_and_load_data(
        notebook_dir='/path/to/new/experiment',
        experiment='New_Experiment_ID',
        manifest_template=template
    )
    
    # Template provides structure, but notebook_dir and experiment are overridden
    print(f"Initialized from template: {manifest['metadata']['experiment_id']}")
    
    return manifest, barcode_frame


# ========================================================================================
# EXAMPLE 8: Parallel Processing (AWS-Ready)
# ========================================================================================

def example_parallel_ready():
    """
    Example showing how the functional approach enables parallel processing.
    Each function is stateless and can run independently.
    """
    
    # In a parallel processing environment (AWS Batch, etc.):
    # 1. Load manifest from S3
    # 2. Load only the data needed
    # 3. Process
    # 4. Save results back
    
    # This could be a Lambda function or Batch job
    def process_batch(manifest_s3_path, barcode_indices):
        # Load manifest
        manifest = load_manifest(manifest_s3_path)
        barcode_frame = load_barcode_frame(manifest)
        
        # Process only specified indices
        batch_data = barcode_frame.loc[barcode_indices]
        
        # ... perform processing ...
        
        # Save results
        # Results could be saved to separate files and merged later
        batch_data.to_parquet(f'results_batch_{barcode_indices[0]}.parquet')
        
        return batch_data
    
    print("Functional approach enables easy parallelization")


if __name__ == '__main__':
    print("BarSeqFitnessFrame Functional API Usage Examples")
    print("=" * 70)
    print()
    print("This file contains examples of using the new functional API.")
    print("Each function demonstrates a different use case.")
    print()
    print("Key Benefits:")
    print("  - State stored in YAML manifest (version controllable)")
    print("  - DataFrames in parquet (efficient, columnar)")
    print("  - Pure functions (testable, composable)")
    print("  - AWS-ready (stateless, parallelizable)")
    print()
    print("See individual functions for usage examples.")
