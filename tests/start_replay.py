

from aiida.orm.calculation.chillstep.user.dynamics.replay import ReplayCalculation
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm import DataFactory, Code, Computer
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.upf import UpfData
import os
# Run dynamics for silicon!
def create_diamond_fcc(element):
    """
    Workfunction to create a diamond crystal structure of a given element.
    At the moment only Si and Ge are valid elements (for C we miss the pseudo)
    :param element: The element to create the structure with.
    :return: The structure.
    """
    import numpy as np

    elem_alat= {
                'Si': 5.431, # Angstrom
                "Ge": 5.658,
               }

    # Validate input element
    symbol = str(element)
    if symbol not in elem_alat.keys():
       raise ValueError("Valid elements are only Si and Ge")

    # Create cel starting from a protopype with alat=1
    alat = elem_alat[symbol]
    the_cell = np.array([[0., 0.5, 0.5],
                         [0.5, 0., 0.5],
                         [0.5, 0.5, 0.]]) * alat
 
    # Create a structure data object
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(0., 0., 0.), symbols=str(element))
    structure.append_atom(position=(0.25*alat, 0.25*alat, 0.25*alat), symbols=str(element))
    supercell = StructureData(ase=structure.get_ase().repeat([2,2,2]))
    
    return supercell

structure = create_diamond_fcc('Si')
parameters = ParameterData(dict={
                u'CONTROL': {
                    u'calculation': 'md',
                    u'restart_mode': 'from_scratch',
                    u'dt':40,
                    u'verbosity':'low',
                },
                u'ELECTRONS': {
                    u'conv_thr':1.e-10,
                },
                u'SYSTEM': {
                    u'nosym': True,
                    u'noinv': True,
                    u'ecutwfc':20,
                    u'ecutrho':160,
                },
                u'IONS': {
                    u'ion_temperature':'rescaling',
                    u'tempw':1000,
                    u'pot_extrapolation':'second-order',
                    u'wfc_extrapolation':'second-order',
                }
            })

KpointsData = DataFactory('array.kpoints')
kpoints = KpointsData()
kpoints.set_kpoints_mesh([2,2,2])

pseudo_Si, = QueryBuilder().append(UpfData, filters={'attributes.element':'Si'}).first()

def start_run():
    mc = ReplayCalculation(code=Code.get_from_string('pw-5.3.0'), structure=structure, parameters=parameters, 
            moldyn_parameters=ParameterData(dict={'nstep':10, 'max_steps_percalc':7, 'resources':{'num_machines':1}, 'max_wallclock_seconds':1000}),
            pseudo_Si=pseudo_Si, kpoints=kpoints)
    mc.submit()


if __name__ == '__main__':
    start_run()
