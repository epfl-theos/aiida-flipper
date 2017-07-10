

from aiida.orm.calculation.chillstep.user.dynamics.branching import BranchingCalculation
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm import DataFactory, Code, Computer
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.upf import UpfData
import os
from copy import deepcopy
# Run dynamics for silicon!

from start_replay import structure, kpoints, pseudo_Si, parameters as parameters_thermalize



def start_run():
    
    parameter_d = deepcopy(parameters_thermalize.get_dict())
    parameter_d['IONS']['ion_velocities'] = 'from_input'
    parameters_nvt = ParameterData(dict=deepcopy(parameter_d))
    parameter_d['IONS']['ion_temperature'] = 'not_controlled'
    parameters_nve=ParameterData(dict=deepcopy(parameter_d))
    moldyn_parameters_thermalize=ParameterData(dict={'nstep':3, 'max_steps_percalc':7, 'resources':{'num_machines':1}, 'max_wallclock_seconds':1000})
    moldyn_parameters_nvt=ParameterData(dict={'nstep':10, 'max_steps_percalc':7, 'resources':{'num_machines':1}, 'max_wallclock_seconds':1000})
    moldyn_parameters_nve=ParameterData(dict={'nstep':2, 'max_steps_percalc':7, 'resources':{'num_machines':1}, 'max_wallclock_seconds':1000})
    code = Code.get_from_string('flipper')
    bc = BranchingCalculation(
            structure=structure,pseudo_Si=pseudo_Si, kpoints=kpoints, # The generic stuff
            parameters_branching=ParameterData(dict=dict(nr_of_branches=2)), # branching etc!
            code=code,
            parameters_thermalize=parameters_thermalize, parameters_nvt=parameters_nvt, parameters_nve=parameters_nve,
            moldyn_parameters_thermalize=moldyn_parameters_thermalize, moldyn_parameters_nvt=moldyn_parameters_nvt, moldyn_parameters_nve=moldyn_parameters_nve,
        )
    bc.submit()



if __name__ == '__main__':
    start_run()
