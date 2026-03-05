from pytao import Tao
import bmad_modeling as mod
import json
import numpy as np
import yaml
from epics import caget
from lcls_live.datamaps import get_datamaps
from pytao import Tao

OPTIONS = ' -noplot' 
INIT = f'-init $LCLS_LATTICE/bmad/models/cu_hxr/tao.init {OPTIONS}'
tao = Tao(INIT)

"""
see https://docs.google.com/document/d/14d-ruEJ11zRfsiWoxtEGd3U4o-aBX6MUSZIkazlGiy8/edit?usp=sharing
"""


#hxr = mod.BmadModeling('cu_hxr', 'DES') #mdl_obj
output = mod.get_output(tao) #TODO replace with output from lume-bmad ?

def get_device_min_max(pv):
    dev_min, dev_max = None, None
    device = pv[0:4]
    if 'BACT' in pv:
        pv = pv.replace('BACT','BDES')
    if device in ['QUAD','BEND', 'XCOR', 'YCOR']:
        dev_min = caget(pv.replace('BDES','BMIN'))
        dev_max = caget(pv.replace('BDES','BMAX'))    
    if device in ['SBST']:
        bmin = -180
        bmax = 180
    return [dev_min, dev_max]


def get_control_name_to_bmad_cu(beam_path):
    """
    use lcls-live datamaps to generate dictionary with PV name as key and Bmad
    name and attribute as value
    """
    yaml_data = {}
    control_name_to_bmad = {}
    datamaps = get_datamaps(beam_path)
    datamaps.pop('beginning_WS02') #TODO support emittance meas locations
    for dm_key,  datamap in datamaps.items():
        yaml_data[dm_key] = []
        if dm_key.startswith('K'):
            datamap = datamaps[dm_key].asdict()
            control_name_to_bmad[ datamap['ampl_act_pvname']] = " ".join(
                    [datamap["name"], 'ENLD_MEV']
                )
            control_name_to_bmad[ datamap['phase_act_pvname']] = " ".join(
                    [datamap["name"][indx], 'PHASE_DEG']
                )
            control_name_to_bmad[ datamap['accelerate_pvname']] = " ".join(
                    [datamap["name"], 'IN_USE']
                )
            klys_datamap = datamaps[dm_key]
            yaml_data[dm_key].append(klys_datamap.asdict())
        else:
            for indx in range(0, len(datamap.data)):
                if 'bpms' in dm_key:
                    control_pv =  datamap.data['pvname'][indx]
                    control_name_to_bmad[control_pv] = " ".join(
                    [datamap.data["bmad_name"][indx], datamap.data["tao_datum"][indx]]
                )
                elif 'linac' in dm_key:
                    control_pv =  datamap.data['pvname'][indx]
                    control_name_to_bmad[control_pv] = " ".join(
                    [datamap.data["bmad_name"][indx], datamap.data["name"][indx]]
                )
                elif 'tao_energy_measurements' in dm_key:
                    control_pv = datamap.data['pvname'][indx]
                    control_name_to_bmad[ control_pv] = " ".join(
                    [datamap.data["name"][indx], datamap.data["tao_datum"][indx]]
                )
                else:
                    if datamap.data["bmad_name"][indx] in ['XCAPM2','YCAPM2']:
                        continue
                    control_pv = datamap.data["pvname_rbv"][indx]
                    control_name_to_bmad[control_pv] = " ".join(
                    [datamap.data["bmad_name"][indx],
                     datamap.data["bmad_attribute"][indx]]                   
                )
                sub_keys = datamap.data.keys().to_list()
                number_of_elements = len(datamap.data[sub_keys[0]])
                #for indx in range(0, number_of_elements):
                ele_dict = {}
                for sk in sub_keys:
                    val =  datamaps[dm_key].data[sk][indx]
                    if isinstance(val, (np.integer, np.floating)):
                        val = float(val)
                    ele_dict[sk] = val
                #dev_min_max = get_device_min_max(control_pv)
                #ele_dict['min_value'] = dev_min_max[0]
                #ele_dict['max_value'] = dev_min_max[1]
                yaml_data[dm_key].append(ele_dict)
    return control_name_to_bmad, yaml_data


beam_path = 'cu_hxr'
control_name_to_bmad, yaml_data = get_control_name_to_bmad_cu(beam_path)

#make YAML file
with open('hxr_input.yaml', 'w') as yaml_file:
    yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)


"""
output dictionary
"""
outkeys = ['ele.name', 'ele.ix_ele', 'ele.ix_branch', 'ele.a.beta',
               'ele.a.alpha', 'ele.a.eta', 'ele.a.etap', 'ele.a.gamma',
               'ele.a.phi', 'ele.b.beta', 'ele.b.alpha', 'ele.b.eta',
               'ele.b.etap', 'ele.b.gamma', 'ele.b.phi', 'ele.x.eta',
               'ele.x.etap', 'ele.y.eta', 'ele.y.etap', 'ele.s', 'ele.l',
               'ele.e_tot', 'ele.p0c', 'ele.mat6', 'ele.vec0']

output = mod.get_output(tao)
output_keys = output.keys()

out_data = {}
for indx in output['ele.ix_ele']:
    element = output['ele.name'][indx] 
    out_data[element] = []
    ele_dict = {}
    for out_key in output_keys:
        val = output[out_key][indx]
        if isinstance(val, (np.integer, np.floating)):
            val = float(val)
        ele_dict[out_key] = val
    out_data[element] = ele_dict

with open('hxr_output.yaml', 'w') as yaml_file:
    yaml.dump(out_data, yaml_file,
        default_flow_style=False, sort_keys=False)
