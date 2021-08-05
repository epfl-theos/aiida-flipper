# -*- coding: utf-8 -*-
"""Calculation function to extract a structure from a trajectory."""
from aiida.engine import calcfunction
bohr_to_ang = 0.52917720859
timeau_to_sec = 2.418884254E-17

from __future__ import absolute_import
from aiida import orm
from aiida.engine import calcfunction
from aiida_flipper.utils.utils import get_or_create_input_node

from math import ceil as math_ceil
import numpy as np
import six
from six.moves import map
from six.moves import zip

SINGULAR_TRAJ_KEYS = ('symbols', 'atomic_species_name')

@calcfunction
def remove_lithium_from_structure(structure, parameters):
    parameters_d = parameters.get_dict()

    element = parameters_d['element']
    atoms = structure.get_ase()
    indices_potential_removal = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == element]
    nat_rem = len(indices_potential_removal)

    if not indices_potential_removal:
        raise RuntimeError('There are no atoms to remove')

    if 'nr_removals' in parameters_d and 'fractional_removal' in parameters_d:
        raise RuntimeError('Both fraction and number specified')
    if 'nr_removals' in parameters_d:
        nr_removals = parameters_d['nr_removals']
        assert 0 < nr_removals <= nat_rem, 'Number of removals is either too small or too large (or not an integer?)'
    elif 'fractional_removal' in parameters_d:
        frac = parameters_d['fractional_removal']
        assert 0 < frac < 1.0, 'Fraction should be 0<frac<1'
        # If given a percentage, I will remove up to the next integer number
        nr_removals = int(math_ceil(frac * nat_rem))
        if nr_removals == nat_rem:
            return None  # Cannot return a structure that is fully delithaited.
    else:
        raise RuntimeError('No fraction of number of specified')

    # I use np.random.choice to get nr_removals random selections from the list, without replacement
    indices_to_remove = np.random.choice(indices_potential_removal, nr_removals, replace=False)
    # I need to reverse-order that list to move from the bottom of the list,
    # Otherwise the indices will be messed up!
    sorted_indices_to_remove = sorted(indices_to_remove, reverse=True)
    for index_to_pop in sorted_indices_to_remove:
        atoms.pop(index_to_pop)
    comp1 = structure.get_composition()
    partially_delithiated = structure = orm.StructureData(ase=atoms) # why change structure to new delithiated structure already
    partially_delithiated.set_attribute('indices_{}_removed'.format(element), sorted_indices_to_remove)
    comp2 = partially_delithiated.get_composition()
    assert comp1.pop(element) != comp2.pop(element), 'No {} was removed'.format(element)
    assert comp1 == comp2, 'composition neglecting element to remove do not match'
    return dict(structure=partially_delithiated)


@calcfunction
def get_diffusion_from_msd(structure, parameters, plot_and_exit=False, **trajectories):
    """
    Compute the Diffusion coefficient from the mean-square displacement.

    :param structure:  the StructureData node or the ASE of the trajectories
    :param parameters: a ParameterData node or a dictionary containing the parameters for MSD computation. Specifically:
        equilibration_time_fs
    and all the other samos.DynamicsAnalyzer.get_msd input parameters:
        species_of_interest
        stepsize_t
        stepsize_tau
        t_start_fs OR t_start_ps OR t_start_dt
        t_end_fs OR t_end_ps OR t_end_dt
        nr_of_blocks OR block_length_fs OR block_length_ps OR block_length_dt
        t_start_fit_fs OR t_start_fit_ps OR t_start_fit_dt
        t_end_fit_fs OR t_end_fit_ps OR t_end_fit_dt
        do_long
        t_long_end_fs OR t_long_end_ps OR t_long_end_dt OR t_long_factor
        do_com

    If t_start_fit and t_end_fit are arrays, the Diffusion coefficient will be calculated for each pair of values.
    This allows one to study its convergence as a function of the window chosen to fit the MSD.
    """

    from samos.trajectory import Trajectory
    from samos.analysis.dynamics import DynamicsAnalyzer
    from ase import Atoms

    if isinstance(structure, orm.StructureData):
        atoms = structure.get_ase()
    elif isinstance(structure, Atoms):
        atoms = structure
    else:
        raise TypeError('structure type not valid')
    if isinstance(parameters, orm.Dict):
        parameters_d = parameters.get_dict()
    elif isinstance(parameters, dict):
        parameters_d = parameters.copy()
    else:
        raise TypeError('parameters type not valid')
    if parameters_d['Decomposed']:
        decomposed = parameters_d['Decomposed']
    else:
        raise TypeError('calculating msd with decomposition not defined')
    for traj in six.itervalues(trajectories):
        if not isinstance(traj, orm.TrajectoryData):
            raise TypeError('trajectories must be TrajectoryData')
    trajdata_list = list(trajectories.values())

    ####################### CHECKS ####################
    units_set = set()
    timesteps_set = set()
    for t in trajdata_list:
        units_set.add(t.get_attr('units|positions'))
    try:
        for t in trajdata_list:
            timesteps_set.add(t.get_attr('timestep_in_fs'))
    except AttributeError:
        for t in trajdata_list:
            input_dict = t.inp.output_trajectory.inp.parameters.get_dict()
            # get the timestep on the fly
            timesteps_set.add(
                timeau_to_sec * 2 * 1e15 * input_dict['CONTROL']['dt'] * input_dict['CONTROL'].get('iprint', 1)
            )

    # Checking if everything is consistent,
    # Check same units:
    units_positions = units_set.pop()
    if units_set:
        raise Exception('Incommensurate units')
    # Same timestep is mandatory!
    timestep_fs = timesteps_set.pop()
    if timesteps_set:
        timesteps_set.add(timestep_fs)
        raise Exception('Multiple timesteps {}'.format(timesteps_set))
    equilibration_steps = int(parameters_d.pop('equilibration_time_fs', 0) / timestep_fs)
    if units_positions in ('bohr', 'atomic'):
        pos_conversion = bohr_to_ang
    elif units_positions == 'angstrom':
        pos_conversion = 1.0
    else:
        raise RuntimeError('Unknown units for positions {}'.format(units_positions))

    ####################### COMPUTE MSD ####################
    trajectories = []
    species_of_interest = parameters_d.pop('species_of_interest', None)
    # if species_of_interest is unicode:
    if isinstance(species_of_interest, (tuple, set, list)):
        species_of_interest = list(map(str, species_of_interest))
    for trajdata in trajdata_list:
        positions = pos_conversion * trajdata.get_positions()[equilibration_steps:]
        nat_in_traj = positions.shape[1]
        trajectory = Trajectory(timestep=t.get_attr('timestep_in_fs'))
        if nat_in_traj != len(atoms):
            indices = [i for i, a in enumerate(atoms.get_chemical_symbols()) if a in species_of_interest]
            if len(indices) == nat_in_traj:
                trajectory.set_atoms(atoms[indices])
            else:
                raise ValueError('number of atoms in trajectory is weird')
        else:
            trajectory.set_atoms(atoms)
        trajectory.set_positions(positions)
        trajectories.append(trajectory)

    # compute msd
    dynanalyzer = DynamicsAnalyzer(verbosity=parameters_d.pop('verbosity', 0))
    dynanalyzer.set_trajectories(trajectories)
    msd_iso = dynanalyzer.get_msd(species_of_interest=species_of_interest, decomposed=decomposed, **parameters_d)

    if plot_and_exit:
        raise NotImplementedError

    # define MSD-results array
    arr_data = orm.ArrayData()
    arr_data.label = '{}-MSD'.format(structure.label)
    for arrayname in msd_iso.get_arraynames():
        arr_data.set_array(arrayname, msd_iso.get_array(arrayname))
    for attr, val in msd_iso.get_attrs().items():
        arr_data.set_attribute(attr, val)

    return {'msd_decomposed_results': arr_data} if decomposed else {'msd_results': arr_data}


@calcfunction
def get_structure_from_trajectory(trajectory, parameters, structure=None, settings=None):
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
            Another example: if the cell trajectory is not found, the structure's cell will be used.
        * missing_velocities: The velocities to give, if complete_missing and create_settings are both set to True. By default [0,0,0]
        * recenter: When true, set the center of mass momentum to 0 (when restarting from a trajectory that doesn't preserve the center of mass.
    :param structure: If comlete_missing is True, I need a structure
    :param settings: If create_settings is True, I can (if provided) just update the dictionary of this instance.
    """
    from aiida.common.exceptions import InputValidationError

    step_index = parameters.dict.step_index
    recenter = parameters.get_attribute('recenter', False)
    create_settings = parameters.get_attribute('create_settings', False)
    complete_missing = parameters.get_attribute('complete_missing', False)
    missing_velocities = parameters.get_attribute('missing_velocities', [0, 0, 0])

    if complete_missing and structure is None:
            raise InputValidationError('You need to pass a structure when completing missing atoms.')
    if create_settings and settings is None:
            raise InputValidationError('You need to pass settings when creating settings.')

    pos_units = trajectory.get_attribute('units|positions', 'angstrom')
    atoms = trajectory.get_step_structure(step_index).get_ase()

    if ('cells' not in trajectory.get_arraynames()) and complete_missing:
        cell_units = trajectory.get_attribute('units|cells', 'angstrom')
        if cell_units == 'angstrom':
            atoms.set_cell(structure.cell)
        elif cell_units == 'atomic':
            atoms.set_cell(np.array(structure.cell) * bohr_to_ang)
        else:
            raise Exception("Can't deal with units of cells {}.".format(cell_units))

    if pos_units == 'angstrom':
        pass
    elif pos_units == 'atomic':
        for atom in atoms:
            atom.position *= bohr_to_ang
    else:
        raise Exception("Can't deal with units of positions {}".format(pos_units))

    if create_settings:
        vel_units = trajectory.get_attribute('units|velocities', 'atomic')
        velocities = trajectory.get_step_data(step_index)[-1]
        if recenter:
            com = np.zeros(3)
            M = 0.
            # Calculate the center of mass displacement:
            for atom, vel in zip(atoms, velocities):
                com = com + atom.mass * vel
                M += atom.mass
            #~ print vel, 1000*atom.mass*vel, com
            velocities[:, 0:3] -= com[0:3] / M
            # CHECK:
            com = np.zeros(3)
            for atom, vel in zip(atoms, velocities):
                com = com + atom.mass * vel
            assert abs(np.linalg.norm(com)) < 1e-12, 'COM did not disappear'

        velocities = velocities.tolist()
        if vel_units == 'atomic':
            pass
        else:
            raise Exception("Can't deal with units of velocities {}".format(vel_units))

    if complete_missing:
        for atom in structure.get_ase()[len(atoms):]:
            atoms.append(atom)
            if create_settings:
                velocities.append([0., 0., 0.])

    newstruc = orm.StructureData(ase=atoms)
    newstruc.label = newstruc.get_formula(mode='count')
    return_dict = dict(structure=newstruc)

    if create_settings:
        if settings is not None:
            settings_d = settings.get_dict()
        else:
            settings_d = {}
        settings_d['ATOMIC_VELOCITIES'] = velocities
        return_dict['settings'] = orm.Dict(dict=settings_d)

    return return_dict


@calcfunction
def concatenate_trajectory(**kwargs):
    for k, v in kwargs.items():
        if not isinstance(v, orm.TrajectoryData):
            raise Exception('All my inputs have to be instances of TrajectoryData')
    sorted_trajectories = list(zip(*sorted(kwargs.items())))[1]
    # I assume they store the same arrays!
    arraynames = sorted_trajectories[0].get_arraynames()
    traj = orm.TrajectoryData()
    for arrname in arraynames:
        if arrname in SINGULAR_TRAJ_KEYS:
            traj.set_array(arrname, sorted_trajectories[0].get_array(arrname))
        else:
            # traj.set_array(arrname, np.concatenate([t.get_array(arrname)[:-1] for t in sorted_trajectories]))
            # concatenate arrays -- remove last step that is repeated when restarting, but keep the very last
            traj.set_array(
                arrname,
                np.concatenate([
                    np.concatenate([t.get_array(arrname)[:-1] for t in sorted_trajectories[:-1]]),
                    sorted_trajectories[-1].get_array(arrname)
                ])
            )
    [traj.set_attribute(k, v) for k, v in sorted_trajectories[0].attributes_items() if not k.startswith('array|')]
    traj.set_attribute('sim_time_fs', (traj.get_array('steps').size - 1) * sorted_trajectories[0].get_attribute('timestep_in_fs'))
    return {'concatenated_trajectory': traj}


@calcfunction
def update_parameters_with_coefficients(parameters, coefficients):
    """
    Updates the ParameterData instance passed with the coefficients
    TODO: nonlocal vs local, currently only nonlocal is correclty implemented
    """
        
    coefs = coefficients.get_attr('coefs')
    parameters_main_d = parameters.get_dict()
    parameters_main_d['SYSTEM']['flipper_local_factor'] = coefs[0]
    parameters_main_d['SYSTEM']['flipper_nonlocal_correction'] = coefs[1]
    parameters_main_d['SYSTEM']['flipper_ewald_rigid_factor'] = coefs[2]
    parameters_main_d['SYSTEM']['flipper_ewald_pinball_factor'] = coefs[3]

    return {'updated_parameters': get_or_create_input_node(orm.Dict, parameters_main_d, store=True)}
