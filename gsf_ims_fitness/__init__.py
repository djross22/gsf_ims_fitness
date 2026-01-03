"""
`gsf_ims_fitness`: Python package for analyzing fitness data for GSF IMS project.

Refactored functional API with manifest-based state management for Nextflow/AWS Batch.
"""

# Versions should comply with PEP440.  For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
__version__ = '0.1'

from . import barseq_fitness
from . import fitness_utils
from . import stan_utils
from .state_io import load_state_v1, save_state_v1