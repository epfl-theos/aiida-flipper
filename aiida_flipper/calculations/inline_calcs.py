from aiida.orm.calculation.inline import optional_inline, make_inline
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm.data.parameter import ParameterData

import numpy as np

SINGULAR_TRAJ_KEYS = ('symbols','atomic_species_name')


@make_inline
def get_structure_from_trajectory_inline(trajectory, parameters, structure=None, settings=None):
    """
    Get a structure from a trajectory, given the step index.
    
    :param trajectory: A trajectory data instance.
    :param parameters: An instance of parameter data. Needs to store the key step_index,
        optionally also the keys:
        * create_settings: whether to also create the settings (an instance of ParameterData) that stores the velocities.
            Of course, that assumes that the velocities are in stored in the trajectory data.
        * complete_missing: If the trajectory does not store all the information required to create the structure,
            i.e. certain atoms were excluded. This will use the structure to complete these atoms.
            An example is the optimized pinball parser, which only stores the positions/velocities of the pinballs.
        * missing_velocities: The velocities to give, if complete_missing and create_settings are both set to True. By default [0,0,0]
        * recenter: When true, set the center of mass momentum to 0 (when restarting from a trajectory that doesn't preserve the center of mass.
    :param structure: If comlete_missing is True, I need a structure
    :param settings: If create_settings is True, I can (if provided) just update the dictionary of this instance.
    """
    from aiida.common.exceptions import InputValidationError

    step_index = parameters.dict.step_index
    recenter = parameters.get_dict().get('recenter', False)
    create_settings = parameters.get_dict().get('create_settings', False)
    complete_missing = parameters.get_dict().get('complete_missing', False)
    missing_velocities = parameters.get_dict().get('missing_velocities', [0,0,0])

    pos_units = trajectory.get_attr('units|positions', 'angstrom')
    atoms = trajectory.get_step_structure(step_index).get_ase()

    if pos_units == 'angstrom':
        pass
    elif pos_units == 'atomic':
        for atom in atoms:
            atom.position *= bohr_to_ang
    else:
        raise Exception("Can't deal with units of positions {}".format(pos_units))

    if create_settings:
        vel_units = trajectory.get_attr('units|velocities', 'atomic')
        velocities = trajectory.get_step_data(step_index)[-1]
        if recenter:
            com = np.zeros(3)
            M =0.
            # Calculate the center of mass displacement:
            for atom,vel in zip(atoms, velocities):
                com = com + atom.mass*vel
                M += atom.mass
            #~ print vel, 1000*atom.mass*vel, com
            velocities[:,0:3] -= com[0:3]/M
            # CHECK:
            com = np.zeros(3)
            for atom,vel in zip(atoms, velocities):
                com = com + atom.mass*vel
            assert abs(np.linalg.norm(com)) < 1e-12, "COM did not disappear"

        velocities = velocities.tolist()
        if vel_units == 'atomic':
            pass
        else:
            raise Exception("Can't deal with units of velocities {}".format(vel_units))

    if complete_missing:
        if structure is None:
            raise InputValidationError("You need to pass a structure when completing missing atoms")
        for atom in structure.get_ase()[len(atoms):]:
            atoms.append(atom)
            if create_settings:
                velocities.append([0.,0.,0.])

    newstruc = StructureData(ase=atoms)
    newstruc.label = newstruc.get_formula(mode='count')
    return_dict=dict(structure=newstruc)

    if create_settings:
        if settings is not None:
            settings_d = settings.get_dict()
        else:
            settings_d = {}
        settings_d['ATOMIC_VELOCITIES'] = velocities
    
        return_dict['settings'] = ParameterData(dict=settings_d)

    return return_dict



@optional_inline
def concatenate_trajectory_optional_inline(**kwargs):
    for k, v in kwargs.iteritems():
        if not isinstance(v, TrajectoryData):
            raise Exception("All my inputs have to be instances of TrajectoryData")
    sorted_trajectories = zip(*sorted(kwargs.items()))[1]
    # I assume they store the same arrays!
    arraynames = sorted_trajectories[0].get_arraynames()
    traj = TrajectoryData()
    for arrname in arraynames:
        if arrname in SINGULAR_TRAJ_KEYS:
            traj.set_array(arrname, sorted_trajectories[0].get_array(arrname))
        else:
            traj.set_array(arrname, np.concatenate([t.get_array(arrname) for t in sorted_trajectories]))
    [traj._set_attr(k,v) for k,v in sorted_trajectories[0].get_attrs().items() if not k.startswith('array|')]
    return {'concatenated_trajectory':traj}

@make_inline
def concatenate_trajectory_inline(**kwargs):
    for k, v in kwargs.iteritems():
        if not isinstance(v, TrajectoryData):
            raise Exception("All my inputs have to be instances of TrajectoryData")
    sorted_trajectories = zip(*sorted(kwargs.items()))[1]
    # I assume they store the same arrays!
    arraynames = sorted_trajectories[0].get_arraynames()
    traj = TrajectoryData()
    for arrname in arraynames:
        if arrname in SINGULAR_TRAJ_KEYS:
            traj.set_array(arrname, sorted_trajectories[0].get_array(arrname))
        else:
            traj.set_array(arrname, np.concatenate([t.get_array(arrname) for t in sorted_trajectories]))
    [traj._set_attr(k,v) for k,v in sorted_trajectories[0].get_attrs().items() if not k.startswith('array|')]
    return {'concatenated_trajectory':traj}


@make_inline 
def get_diffusion_from_msd_inline(**kwargs):
    return get_diffusion_from_msd(**kwargs)

def get_diffusion_from_msd(structure, parameters, **trajectories):

    from aiida.common.constants import timeau_to_sec
    from mdtools.libs.mdlib.trajectory_analysis import TrajectoryAnalyzer

    parameters_d = parameters.get_dict()
    trajdata_list = trajectories.values()

    ####################### CHECKS ####################
    units_set = set()
    timesteps_set = set()
    units_set = set()
    vel_units_set = set()
    for t in trajdata_list:
        units_set.add(t.get_attr('units|positions'))
        vel_units_set.add(t.get_attr('units|velocities'))
    try:
        for t in trajdata_list:
            timesteps_set.add(t.get_attr('timestep_in_fs'))

    except AttributeError:
        for t in trajdata_list:
            input_dict = t.inp.output_trajectory.inp.parameters.get_dict()
            # get the timestep on the fly
            timesteps_set.add(timeau_to_sec*2*1e15*input_dict['CONTROL']['dt']*input_dict['CONTROL'].get('iprint', 1))
    # Checking if everything is consistent, 
    # Check same units:
    units_positions = units_set.pop()
    if units_set:
        raise Exception("Incommensurate units")
    units_velocities = vel_units_set.pop()
    if vel_units_set:
        raise Exception("Incommensurate units")
    # legacy:
    if units_velocities == 'atomic':
        units_velocities = 'pw'
    # Same timestep is mandatory!
    timestep_fs = timesteps_set.pop()
    if timesteps_set:
        timesteps_set.add(timestep_fs)
        raise Exception("Multiple timesteps {}".format(timesteps_set))

    # I work with fs, not QE units!
    #~ timestep_fs = timestep
    equilibration_steps = int(parameters_d.get('equilibration_time_fs', 0) / timestep_fs)

    trajectories = [t.get_positions()[equilibration_steps:] for t in trajdata_list]
    ta = TrajectoryAnalyzer(verbosity=0)
    species_of_interest = parameters_d.pop('species_of_interest', None)
    ta.set_structure(structure, species_of_interest=species_of_interest)
    ta.set_trajectories(
            trajectories, # velocities=velocities if plot else None,
            pos_units=units_positions, # vel_units=units_velocities,
            timestep_in_fs=timestep_fs, recenter=True,) # parameters_d.pop('recenter', False) Always recenter
    print parameters
    res, arr = ta.get_msd(**parameters_d)
    arr_data = ArrayData()
    arr_data.label = '{}-MSD'.format(structure.label)
    arr_data.set_array('msd', arr)
    arr_data._set_attr('species_of_interest', species_of_interest)
    for k,v in res.items():
        arr_data._set_attr(k,v)

    return {'msd_results':arr_data}






