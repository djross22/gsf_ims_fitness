"""
`gsf_ims_fitness`: Python package for analyzing fitness data for GSF IMS project.

"""

# Versions should comply with PEP440.  For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
__version__ = '0.1'

from .ODFitnessFrame import ODFitnessFrame
from .BarSeqFitnessFrame import BarSeqFitnessFrame
from .fitness import *