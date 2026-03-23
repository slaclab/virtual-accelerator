import os
from pathlib import Path
from lume_cheetah import LUMECheetahModel, CheetahSimulator
from virtual_accelerator.cheetah.transformer import SLACCheetahTransformer
from virtual_accelerator.cheetah.variables import get_variables_from_segment
from virtual_accelerator.utils.variables import (
    get_epics_to_name_mapping,
    split_control_and_observable,
    get_diag0_screen_variables
)
from cheetah.accelerator import Segment
from cheetah.particles import ParticleBeam
import torch
