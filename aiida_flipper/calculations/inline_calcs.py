from aiida.orm.calculation.inline import optional_inline, make_inline
from aiida.orm.data.structure import StructureData
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
