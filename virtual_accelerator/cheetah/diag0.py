import os
import torch
from cheetah.accelerator import Segment, Screen
from cheetah.particles import ParticleBeam
from pathlib import Path

from cheetah.accelerator.patch import Patch
from cheetah.accelerator.superimposed import SuperimposedElement
from cheetah.accelerator import Dipole, Quadrupole

def get_diag0_beamline():
    # try to get LCLS_LATTICE -- returns none if not found
    lcls_lattice_location = os.getenv('LCLS_LATTICE')

    # if LCLS_LATTICE not found, use the local json model
    if lcls_lattice_location is None:
        tracking_segment = Segment.from_lattice_json(
            os.path.join(Path(__file__).parent, "sc_diag0.json")
        ).subcell(start="bpmdg000")
    else:
        tracking_segment = Segment.from_lattice_json(f"{lcls_lattice_location}/cheetah/sc_diag0.json").subcell(start="bpmdg000")
    
    dyqdg001 = Patch(name = "dyqdg001", pitch = torch.tensor((0.0,9.49758257820075558E-003)))
    dyqdg003 = Patch(name = "dyqdg003", pitch = torch.tensor((0.0,5.88487966838956720E-004)))
    
    elements = list(tracking_segment.elements)
    
    # get index of qdg001
    element_names = [ele.name for ele in tracking_segment.elements]
    
    # create SuperimposedElements for qdg001 and qdg003
    quad = tracking_segment.qdg001[0]
    # In bmad this (and other elements below) are referenced twice, for us we need to multiply the length of a single use element by 2
    quad.length = quad.length * 2 
    super_qdg001 = SuperimposedElement(
        name="qdg001",
        base_element=quad,
        superimposed_element=Segment([
            dyqdg001, tracking_segment.bpmdg001
        ])
    )
    quad = tracking_segment.qdg003[0]
    quad.length = quad.length * 2
    super_qdg003 = SuperimposedElement(
        name="qdg003",
        base_element=quad,
        superimposed_element=Segment([
            dyqdg003, tracking_segment.bpmdg003
        ])
    )
    for ele in [super_qdg001, super_qdg003]:
        idx = [ele.name for ele in elements].index(ele.name)
        elements[idx:idx+3] = [ele]
    
    
    # create superimposed elements for other quads containing bpms
    split_quads = [2,4,5,8,9,11]
    for idx in split_quads:
        quad = getattr(tracking_segment, f"qdg{idx:0>3}")[0]
        quad.length = quad.length * 2
        bpm = getattr(tracking_segment, f"bpmdg{idx:0>3}")
        super_element = SuperimposedElement(
            name=f"qdg{idx:0>3}",
            base_element=quad,
            superimposed_element=bpm
        )
        idx = [ele.name for ele in elements].index(f"qdg{idx:0>3}")
        elements[idx:idx+3] = [super_element]
    
    # create the superimposed element for the transverse deflecting cavity
    tdc_idx = [ele.name for ele in elements].index(f"tcxdg0")
    tdc = tracking_segment.tcxdg0[0]
    tdc.length = tdc.length * 2.0
    tdc.num_steps = 11
    vkick = tracking_segment.ycdgtcx
    xkick = tracking_segment.xcdgtcx
    tdc.voltage = torch.tensor(0.0)
    
    super_tdc = SuperimposedElement(
        name="tcxdg0",
        base_element=tdc,
        superimposed_element=Segment([vkick, xkick])
    )
    elements[tdc_idx:tdc_idx+4] = [super_tdc]
        
    tracking_segment = Segment(elements)
    
    # change the offset of these quads
    tracking_segment.qdg001.base_element.misalignment = torch.tensor((0.0,-4.83865231890650768E-003))
    tracking_segment.qdg001.superimposed_element.misalignment = torch.tensor((0.0,-4.83865231890650768E-003))
    
    tracking_segment.qdg003.base_element.misalignment = torch.tensor((0.0,-2.99813063290814820E-004))
    tracking_segment.qdg003.superimposed_element.misalignment = torch.tensor((0.0,-2.99813063290814820E-004))

    # set screens to use kde
    for ele in tracking_segment.elements:
        if isinstance(ele, Screen):
            ele.method = "charge_deposition"
        elif isinstance(ele, Quadrupole):
            ele.tracking_method = "second_order"
        elif isinstance(ele, SuperimposedElement):
            ele.base_element.tracking_method = "second_order"
        else:
            ele.tracking_method = "linear"

    return tracking_segment