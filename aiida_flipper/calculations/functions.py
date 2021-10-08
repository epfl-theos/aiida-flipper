# -*- coding: utf-8 -*-
from aiida.engine import calcfunction
bohr_to_ang = 0.52917720859
timeau_to_sec = 2.418884254E-17

from aiida import orm
from aiida.engine import calcfunction
from aiida_flipper.utils.utils import get_or_create_input_node

import numpy as np

SINGULAR_TRAJ_KEYS = ('symbols', 'atomic_species_name')

@calcfunction
def get_diffusion_from_msd(structure, parameters, trajectory):
    """
    Compute the Diffusion coefficient from the mean-square displacement.

    :param structure: the StructureData node or the ASE of the trajectories
    :param trajectory: concatenated trajectories in the form of one TrajectoryData
    :param parameters: a ParameterData node or a dictionary containing the parameters for MSD computation. Specifically:
        equilibration_time_fs, which is the time to assumed to equilibrate the atoms
        decomposed, which if true, decomposes the MSD into contribution of each atom types
    and all the other samos.DynamicsAnalyzer.get_msd input parameters:
        do_long - doesn't exist...
        t_long_end_fs OR t_long_end_ps OR t_long_end_dt OR t_long_factor - non existent...
        :param list species_of_interest: The species to calculate.
        :param int stepsize_t: Integer value of the outer-loop stepsize.
            Setting this to higher than 1 will decrease the resolution. Defaults to 1
        :param int stepsize_tau: Integer value of the inner loop stepsize.
            If higher than 1, the sliding window will be moved more sparsely through the block. Defaults to 1.
        :param float t_start_fs: Minimum value of the sliding window in femtoseconds.
        :param float t_start_ps: Minimum value of the sliding window in picoseconds.
        :param int t_start_dt: Minimum value of the sliding window in multiples of the trajectory timestep.
        :param float t_end_fs: Maximum value of the sliding window in femtoseconds.
        :param float t_end_ps: Maximum value of the sliding window in picoseconds.
        :param int t_end_dt: Maximum value of the sliding window in multiples of the trajectory timestep.
        :param float block_length_fs: Block size for trajectory blocking in fs.
        :param float block_length_ps: Block size for trajectory blocking in picoseconds.
        :param int block_length_dt: Block size for trajectory blocking in multiples of the trajectory timestep.
        :param int nr_of_blocks: Nr of blocks that the trajectory should be split in (excludes setting of block_length). If nothing else is set, defaults to 1.
        :param float t_start_fit_fs: Time to start the fitting of the time series in femtoseconds.
        :param float t_start_fit_ps: Time to start the fitting of the time series in picoseconds.
        :param int t_start_fit_dt: Time to start the fitting of the time series in multiples of the trajectory timestep.
        :param float t_end_fit_fs: Time to end the fitting of the time series in femtoseconds.
        :param float t_end_fit_ps: Time to end the fitting of the time series in picoseconds.
        :param int t_end_fit_dt: Time to end the fitting of the time series in multiples of the trajectory timestep.
        :param bool do_com: Whether to calculate centre of mass diffusion.

        Note: assert that t_end_dt > t_end_fit_dt


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

    if not isinstance(trajectory, orm.TrajectoryData):
        raise TypeError('trajectory must be TrajectoryData')

    ####################### CHECKS ####################
    # Checking if everything is consistent

    units_positions = trajectory.get_attribute('units|positions')
    timestep_fs = trajectory.get_attribute('timestep_in_fs') 
    equilibration_steps = int(parameters_d.pop('equilibration_time_fs', 0) / timestep_fs)
    if units_positions in ('bohr', 'atomic'):
        pos_conversion = bohr_to_ang
    elif units_positions == 'angstrom':
        pos_conversion = 1.0
    else:
        raise RuntimeError('Unknown units for positions {}'.format(units_positions))

    ####################### COMPUTE MSD ####################
    species_of_interest = parameters_d.pop('species_of_interest', None)

    positions = pos_conversion * trajectory.get_positions()[equilibration_steps:]
    nat_in_traj = positions.shape[1]
    trajectory = Trajectory(timestep=trajectory.get_attribute('timestep_in_fs'))
    if nat_in_traj != len(atoms):
        indices = [i for i, a in enumerate(atoms.get_chemical_symbols()) if a in species_of_interest]
        if len(indices) == nat_in_traj:
            trajectory.set_atoms(atoms[indices])
        else:
            raise ValueError('number of atoms in trajectory is weird')
    else:
        trajectory.set_atoms(atoms)
    trajectory.set_positions(positions)

    # compute msd
    dynanalyzer = DynamicsAnalyzer(verbosity=parameters_d.pop('verbosity'))
    dynanalyzer.set_trajectories(trajectory)
    decomposed = parameters_d.pop('decomposed')
    msd_iso = dynanalyzer.get_msd(species_of_interest=species_of_interest, decomposed=decomposed, **parameters_d)

    # define MSD-results array
    arr_data = orm.ArrayData()
    arr_data.label = '{}-MSD'.format(structure.label)
    # Following are the collection of trajectories, not sure why we need this
    for arrayname in msd_iso.get_arraynames():
        arr_data.set_array(arrayname, msd_iso.get_array(arrayname))
    # Following attributes are results_dict of samos.analysis.DynamicsAnalyzer.get_msd()
    for attr, val in msd_iso.get_attrs().items():
        arr_data.set_attribute(attr, val)
    return {'msd_results': arr_data}


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
    :param  : If create_settings is True, I can (if provided) just update the dictionary of this instance.
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
    remove_repeated_last_step = kwargs.pop('remove_repeated_last_step', True)
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
            # concatenate arrays
            if len(sorted_trajectories) > 1:
                if remove_repeated_last_step:
                    # remove last step that is repeated when restarting, keep the very last
                    traj.set_array(arrname, np.concatenate([
                        np.concatenate([t.get_array(arrname)[:-1] for t in sorted_trajectories[:-1]]),
                        sorted_trajectories[-1].get_array(arrname)]))
                else:
                    # just concatenate - used e.g. for Hustler
                    traj.set_array(arrname, np.concatenate([t.get_array(arrname) for t in sorted_trajectories]))
            else:
                traj.set_array(arrname, np.concatenate([t.get_array(arrname) for t in sorted_trajectories]))
    [traj.set_attribute(k, v) for k, v in sorted_trajectories[0].attributes_items() if not k.startswith('array|')]
    if 'timestep_in_fs' in sorted_trajectories[0].attributes:
        traj.set_attribute('sim_time_fs', traj.get_array('steps').size * sorted_trajectories[0].get_attribute('timestep_in_fs'))
    return {'concatenated_trajectory': traj}


@calcfunction
def update_parameters_with_coefficients(parameters, coefficients):
    """
    Updates the ParameterData instance passed with the coefficients
    TODO: nonlocal vs local, currently only nonlocal is correclty implemented
    """
        
    coefs = coefficients.get_attribute('coefs')
    parameters_main_d = parameters.get_dict()
    parameters_main_d['SYSTEM']['flipper_local_factor'] = coefs[0]
    parameters_main_d['SYSTEM']['flipper_nonlocal_correction'] = coefs[1]
    parameters_main_d['SYSTEM']['flipper_ewald_rigid_factor'] = coefs[2]
    parameters_main_d['SYSTEM']['flipper_ewald_pinball_factor'] = coefs[3]

    return {'updated_parameters': get_or_create_input_node(orm.Dict, parameters_main_d, store=True)}


@calcfunction
def get_pinball_factors(parameters, trajectory_scf, trajectory_pb):
    from aiida_flipper.utils.utils import Force, fit_with_lin_reg, make_fitted, plot_forces
    from scipy.stats import linregress

    params_dict = parameters.get_dict()
    starting_point = params_dict['starting_point']
    stepsize = params_dict['stepsize']
    nsample = params_dict.get('nsample', None)
    signal_indices = params_dict.get('signal_indices', None)

    atom_indices_scf = [i for i, s in enumerate(trajectory_scf.get_attribute('symbols')) if s == params_dict['symbol']]
    atom_indices_pb = [i for i, s in enumerate(trajectory_pb.get_attribute('symbols')) if s == params_dict['symbol']]

    all_forces_scf = trajectory_scf.get_array('forces')[:, atom_indices_scf,:]
    all_forces_pb = trajectory_pb.get_array('forces')[:, atom_indices_pb,:]

    # You need to remove all the steps that are starting indices due to this stupid thing with the hustler first step. 
    starting_indices = set()
    for traj in (trajectory_pb, trajectory_scf):
        [starting_indices.add(_) for _ in np.where(trajectory_scf.get_array('steps') == 0)[0]]

    # You also need to remove steps for the trajectory_scf that did not SCF CONVERGE!!!!
    convergence = trajectory_scf.get_array('scf_convergence')
    [starting_indices.add(_) for _ in np.where(~convergence)[0]]

    for idx in sorted(starting_indices, reverse=True):
        all_forces_scf = np.delete(all_forces_scf, idx, axis=0)
        all_forces_pb  = np.delete(all_forces_pb,  idx, axis=0)

    if nsample == None: nsample = min((len(all_forces_scf), len(all_forces_pb)))

    forces_scf = Force(all_forces_scf[starting_point:starting_point+nsample*stepsize:stepsize])
    forces_pb = Force(all_forces_pb[starting_point:starting_point+nsample*stepsize:stepsize])

    coefs, mae = fit_with_lin_reg(forces_scf, forces_pb,
            verbosity=0, divide_r2=params_dict['divide_r2'], signal_indices=signal_indices)
    try:
        mae_f = float(mae)
    except:
        mae_f = None

    forces_fitted = make_fitted(forces_pb, coefs=coefs, signal_indices=signal_indices)
    slope_before_fit, intercept_before_fit, rvalue_before_fit, pvalue_before_fit, stderr_before_fit = linregress(
            forces_scf.get_signal(0).flatten(), forces_fitted.get_signal(0).flatten())
    slope_after_fit, intercept_after_fit, rvalue_after_fit, pvalue_after_fit, stderr_after_fit = linregress(
            forces_scf.get_signal(0).flatten(), forces_pb.get_signal(0).flatten())
    # plot_forces([forces_scf, forces_pb, forces_fitted], labels=('DFT', 'pinball', 'pinball-fitted'), format_='0:0,1:0;0:0,2:0', titles=("Before Fit", "After Fit"),savefig=None)

    coeff_params = orm.Dict(dict={
        'coefs': coefs.tolist(),
        'mae': mae_f,
        'nr_of_coefs': len(coefs),
        'indices_removed': sorted(starting_indices),
        'linreg_before_fit': {
            'slope': slope_before_fit,
            'intercept': intercept_before_fit,
            'r2value': rvalue_before_fit**2,
            'pvalue_zero_slope': pvalue_before_fit,
            'stderr': stderr_before_fit
            },
        'linreg_after_fit': {
            'slope': slope_after_fit,
            'intercept': intercept_after_fit,
            'r2value': rvalue_after_fit**2,
            'pvalue_zero_slope': pvalue_after_fit,
            'stderr': stderr_after_fit
            },
    })
    return {'coefficients': coeff_params}