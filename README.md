# gsf_ims_fitness

The repository contains a Python package and Jupyter notebooks used for analysis of genotype-phenotype landscape data. As input, gsf_ims_fitness takes several files produced by software in the bartender-1.1, NISTBartender, and engineering-bio-lacI-landscape repositories (see below). It produces output data in the form of a Pandas DataFrame that can be exported in various formats for additional analysis, if necessary.

# Prerequisites

gsf_ims_fitness was written in Python3, version 3.11.4

It was written with the following python packages:

- matplotlib

- numpy

- pandas

- scipy

- cmdstanpy

- seaborn

- biopython

- scikit-learn

- palettable

- cmocean

- ipython

- pyyaml

To install the correct package versions, use one of the .yml files included in this repository. `gsf_ims_env.yml` is from a Windows installation. `gsf_ims_env.AWS.yml` is from a Linux installation (on an AWS EC2 instance). They are very similar, but they were set up independently, so we reccomend using the environment file spoecific to the computer OS.

After installing conda (miniconda reccomended: https://docs.conda.io/en/latest/miniconda.html), create the gsf_ims environment with:

Windows:
```
conda env create -f gsf_ims_env.yml
```
Linux:
```
conda env create -f gsf_ims_env.AWS.yml
```

Then install the gsf_ims_fitness package from source:

For Windows:
```
cd gsf_ims_fitness
pip install -e .
```
For Linux:
```
cd gsf_ims_fitness
python3 -m pip install -e .
```
  

# Data Requirements

gsf_ims_fitness requires the following input data files:

- Output from the NISTBartender software: "output_file_label.trimmed_sorted_counts.csv"

- Output from bartender-1.1 software (run as stand-alone or automatically from NISTBartender): "output_file_label_forward_cluster.csv", "output_file_label_forward_barcode.csv", "output_file_label_reverse_cluster.csv", and "output_file_label_reverse_barcode.csv".

- Output from the engineering-bio-lacI-landscape long-read sequencing data analysis pipeline (one file for each plasmid region):

  - "barcode_1.tsv.gz"

  - "barcode_2.tsv.gz"

  - "empty_1.tsv.gz"

  - "empty_2.tsv.gz"

  - "empty_3.tsv.gz"

  - "empty_4.tsv.gz"

  - "insulator.tsv.gz"

  - "KAN.tsv.gz"

  - "lacI.tsv.gz"

  - "Ori.tsv.gz"

  - "primers.tsv.gz"

  - "tetA.tsv.gz"

  - "YFP.tsv.gz"

    

# Usage

After generating the required input files listed above, run the following Jupyter notebooks (in order):

#### "BarSeq DataFrame.ipynb"

This notebook initializes the Pandas Dataframe that will contain the genotype-phenotype landscape data. It also has a step to mark actual chimera reads (dual barcode pairs that are improperly matched) for later removal. Finally, it runs the least-squares fitting routine to covert from barcode read counts to fitness for each sample and runs the Bayesian inference methods to estimate the dose-response curves from the fitness results. The Stan code for the Bayesian inference models is located in the "\gsf_ims_fitness\Stan models" folder in this repository.

### "BarSeq post sensor fit data cleanup.ipynb"

This notebook adds columns to the DataFrame to indicate the number of points of agreement between the Hill model and GP model fits, the LacI amino acid sequence for each variant, and the mutation codes indicating the amino acid changes relative to the wild-type sequence (e.g. "A106D")

### "Attach cds from long-read PacBio.ipynb"

This notebook uses the output from from bartender-1.1 and engineering-bio-lacI-landscape to add the lacI CDS sequences to the DataFrame.

### Notebooks for other plasmid regions

The following notebooks use the output from bartender-1.1 and engineering-bio-lacI-landscape to add the sequence information for other plasmid regions to the DataFrame:

"PacBio YFP.ipynb", "PacBio tetA.ipynb", "PacBio empty_1.ipynb", "PacBio empty_2.ipynb", "PacBio empty_3.ipynb", "PacBio insulator.ipynb", "PacBio KAN.ipynb", "PacBio Ori.ipynb".

### DataFrame Columns

After running the Jupyter notebooks listed above the Pandas DataFrame referenced by the BarSeqFitnessFrame.barcode variable will contain a row for each LacI variant in the library and the following columns of data for each variant:

| column                      | contents                                                     |
| --------------------------- | ------------------------------------------------------------ |
| forward_BC                  | variant barcode sequence from the forward read               |
| reverse_BC                  | variant barcode sequence from the reverse read               |
| A1 - H12                    | dual barcode read count for the variant from each sample in the 96-well  library prep plate |
| total_counts                | total dual barcode read count                                |
| possibleChimera             | Boolean value indicating if the dual barcode is a possible chimera |
| parent_geo_mean             | Geometric mean of the total_counts of the possible parent barcodes, used  to assess chimeric barcodes |
| isChimera                   | Boolean value indicating whether or not the dual barcode is chimeric  (i.e. not representative of a real LacI variant) |
| fraction_total              | total_counts divided by the total dual barcode count for the entire  experiment |
| read_count_0_2              | array of dual barcode read counts for time point 1 (the 2nd growth plate)  without tetracycline, the array gives the values for the 12 different IPTG  concentrations used |
| read_count_20_2             | array of dual barcode read counts for time point 1 (the 2nd growth plate)  with tetracycline, the array gives the values for the 12 different IPTG  concentrations used |
| read_count_0_3              | array of dual barcode read counts for time point 2 (the 3rd growth plate)  without tetracycline, the array gives the values for the 12 different IPTG  concentrations used |
| read_count_20_3             | array of dual barcode read counts for time point 2 (the 3rd growth plate)  with tetracycline, the array gives the values for the 12 different IPTG  concentrations used |
| read_count_0_4              | array of dual barcode read counts for time point 3 (the 4th growth plate)  without tetracycline, the array gives the values for the 12 different IPTG  concentrations used |
| read_count_20_4             | array of dual barcode read counts for time point 3 (the 4th growth plate)  with tetracycline, the array gives the values for the 12 different IPTG  concentrations used |
| read_count_0_5              | array of dual barcode read counts for time point 4 (the 5th growth plate)  without tetracycline, the array gives the values for the 12 different IPTG  concentrations used |
| read_count_20_5             | array of dual barcode read counts for time point 4 (the 5th growth plate)  with tetracycline, the array gives the values for the 12 different IPTG  concentrations used |
| fitness_0_estimate_b        | estimated fitness without tetracycline, normalized by the plate-to-plate  dilution factor and the growth plate time (i.e. fitness=1 indicates 10-fold  growth during one plate cycle, 160 minutes) |
| fitness_0_err_b             | uncertainty estimate for fitness without tetracycline, from the  non-linear least-squares fit |
| fitness_20_estimate_b       | estimated fitness with tetracycline, normalized by the plate-to-plate  dilution factor and the growth plate time (i.e. fitness=1 indicates 10-fold  growth during one plate cycle, 160 minutes) |
| fitness_20_err_b            | uncertainty estimate for fitness with tetracycline, from the non-linear  least-squares fit |
| sensor_params               | posterior medians for fit parameters from Baysian Hill equation model |
| sensor_params_cov           | posterior covariance matrix for fit parameters from Baysian Hill equation  model |
| sensor_rms_residuals        | rms residual for Baysian Hill equation model fit             |
| sensor_stan_samples         | 32 posterior samples for each fit parameter from Baysian Hill equation  model |
| sensor_params_quantiles     | 0.05, 0.25, 0.5, 0.75, and 0.95 posterior quantiles for each fit  parameter from Baysian Hill equation model |
| sensor_GP_params            | posterior medians for fit parameters from Baysian Gaussian process (GP)  model |
| sensor_GP_cov               | posterior covariance matrix for fit parameters from Baysian Gaussian  process (GP) model |
| sensor_GP_g_quantiles       | 0.05, 0.25, 0.5, 0.75, and 0.95 posterior quantiles for the estimated  log10(gene expression output) at each IPTG concentration from Baysian GP  model |
| sensor_GP_Dg_quantiles      | 0.05, 0.25, 0.5, 0.75, and 0.95 posterior quantiles for the estimated  derivative of log10(gene expression output) vs. log10([IPTG]), at each IPTG  concentration from Baysian GP model |
| sensor_GP_Df_quantiles      | 0.05, 0.25, 0.5, 0.75, and 0.95 posterior quantiles for the estimated  fitness impact of tetracycline at each IPTG concentration from Baysian GP  model |
| sensor_GP_residuals         | rms residual for Baysian GP model fit                        |
| sensor_GP_g_var             | posterior variance for the log10(gene expression output) at each IPTG  concentration from Baysian GP model |
| sensor_GP_dg_var            | posterior variance for the derivative of log10(gene expression output)  vs. log10([IPTG]), at each IPTG concentration from Baysian GP model |
| sensor_GP_g_samples         | 32 posterior samples for the log10(gene expression output) at each IPTG  concentration from Baysian GP model |
| sensor_GP_dg_samples        | 32 posterior samples for the derivative of log10(gene expression output)  vs. log10([IPTG]), at each IPTG concentration from Baysian GP model |
| good_hill_fit_points        | number of points of agreement between the Bayesian Hill equation and GP  model fits, i.e. the number of IPTG concentrations at which the median  estimate for the Hill equation dose-response curve was within the central 90%  credible interval from the GP model |
| rev_BC_ID                   | identification number assigned to reverse barcode by bartender-1.1  algorithm |
| dual_BC_ID                  | string concatenation of for_BC_ID and rev_BC_ID, separated by underscore  character |
| concensus_cds               | the consensus LacI CDS sequence from the long-read sequencing |
| cds_error_rate              | per-base read error rate for consensus CDS, equal to zero unless  different CCS reads for the same variant gave conflicting CDS results |
| pacbio_count                | number of PacBio CCS reads with the LacI sequence in the long-read data  for the variant |
| concensus_cds_mutation_rate | per-base mutation rate of the CDS relative to the wild-type CDS |
| hasConfidentCds             | Boolean value indicating whether or not the CDS assignment is confident  (see manuscript supplementary information) |
| lacI_amino_seq              | LacI amino acid sequence for the variant                     |
| lacI_amino_mutations        | number of animo acid subsitutions relative to the wild type  |
| mutation_codes              | mutation codes indicating the amino acid substitutions relative to the  wild type |
| log_low_level               | log10(*G*~0~)                                                |
| log_low_level error         | posterior uncertainty for log10(*G*~0~), 1 standard deviation |
| log_low_level samples       | 32 posterior samples for log10(*G*~0~)                       |
| log_high_level              | log10(*G*_~∞~)                                               |
| log_high_level error        | posterior uncertainty for log10(*G*~∞~), 1 standard deviation |
| log_high_level samples      | 32 posterior samples for log10(*G*~∞~)                       |
| log_ic50                    | log10(*EC*~50~)                                              |
| log_ic50 error              | posterior uncertainty for log10(*EC*~50~), 1 standard deviation |
| log_ic50 samples            | 32 posterior samples for log10(*EC*~50~)                     |
| log_n                       | log10(*n*)                                                   |
| log_n error                 | posterior uncertainty for log10(*n*), 1 standard deviation   |
| log_n samples               | 32 posterior samples for log10(*n*)                          |
| log_high_low_ratio          | log10(*G*~∞~/*G*~0~)                                         |
| log_high_low_ratio error    | posterior uncertainty for log10(*G*~∞~/*G*~0~), 1 standard deviation |
| log_high_low_ratio samples  | 32 posterior samples for log10(*G*~∞~/*G*~0~)                |
| pacbio_KAN_count            | number of PacBio CCS reads with the sequence for the kanamycin resistance  plasmid region in the long-read data for the variant |
| pacbio_KAN_mutations        | number of DNA mutations found in the kanamycin resistance plasmid region  for the variant |
| pacbio_Ori_count            | number of PacBio CCS reads with the sequence for the origin of  replication plasmid region in the long-read data for the variant |
| pacbio_Ori_mutations        | number of DNA mutations found in the origin of replication plasmid region  for the variant |
| pacbio_YFP_count            | number of PacBio CCS reads with the sequence for the YFP plasmid region  in the long-read data for the variant |
| pacbio_YFP_mutations        | number of DNA mutations found in the YFP plasmid region for the variant |
| pacbio_empty_1_count        | number of PacBio CCS reads with the sequence for the intergenic plasmid  region between the origin and the barcodes in the long-read data for the  variant |
| pacbio_empty_1_mutations    | number of DNA mutations found in the intergenic plasmid region between  the origin and the barcodes for the variant |
| pacbio_empty_2_count        | number of PacBio CCS reads with the sequence for the intergenic plasmid  region between the LacI gene and the barcodes in the long-read data for the  variant |
| pacbio_empty_2_mutations    | number of DNA mutations found in the intergenic plasmid region between  the LacI gene and the barcodes for the variant |
| pacbio_empty_3_count        | number of PacBio CCS reads with the sequence for the intergenic plasmid  region between the origin and the barcodes in the long-read data for the  variant, split because fot the starting point for CCS reads |
| pacbio_empty_3_mutations    | number of DNA mutations found in the intergenic plasmid region between  the origin and the barcodes for the variant |
| pacbio_insulator_count      | number of PacBio CCS reads with the sequence for the regulatory plasmid  region in the long-read data for the variant |
| pacbio_insulator_mutations  | number of DNA mutations found in the regulatory plasmid region for the  variant |
| pacbio_tetA_count           | number of PacBio CCS reads with the sequence for the tetA plasmid region  in the long-read data for the variant |
| pacbio_tetA_mutations       | number of DNA mutations found in the tetA plasmid region for the variant |

