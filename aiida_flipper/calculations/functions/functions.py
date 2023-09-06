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
    species_of_interest = parameters_d.pop('species_of_interest', 'Li')

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

        # calculating slope of MSD(timelag) for different fitting lengths to check if it converged
    t_end_fit_fs_length = parameters_d.pop('t_end_fit_fs_length')

    if t_end_fit_fs_length > 1:
        t_end_fit_list = np.arange(t_end_fit_fs_length, parameters_d['t_end_fit_fs'], t_end_fit_fs_length)
        slope_msd_list, diffusion_list = [], []

        for t_end_fit in t_end_fit_list:
            parameters_d['t_end_fit_fs'] = t_end_fit
            msd_iso = dynanalyzer.get_msd(species_of_interest=species_of_interest, decomposed=decomposed, **parameters_d)
            # I only care about Li now
            slope_msd_list.append(msd_iso.get_attr('Li')['slope_msd_mean'])
            diffusion_list.append(msd_iso.get_attr('Li')['diffusion_mean_cm2_s'])

        slope_std = np.std(slope_msd_list[-3:], axis=0)
        slope_sem = slope_std/np.sqrt(3-1)
        diff_std = np.std(diffusion_list[-3:], axis=0)
        diff_sem = diff_std/np.sqrt(3-1)

        atomic_species_dict_tmp = msd_iso.get_attr(atomic_species)
        atomic_species_dict_tmp.update({'slope_msd_std': slope_std, 'slope_msd_sem': slope_sem, 'diffusion_std_cm2_s': diff_std, 'diffusion_sem_cm2_s': diff_sem, })
        msd_iso.set_attr(atomic_species, atomic_species_dict_tmp)

    else:
        msd_iso = dynanalyzer.get_msd(species_of_interest=species_of_interest, decomposed=decomposed, **parameters_d)

    if parameters_d['nr_of_blocks']==1:
        # setting up std values for sem in case only 1 block is used, so that we don't have nan values that can't be stored in aiida database, we use std values instead of 0 to be consistent with output format
        for atomic_species in species_of_interest:
            msd_iso.set_array('msd_{}_{}_sem'.format('decomposed' if decomposed else 'isotropic', atomic_species), msd_iso.get_array('msd_{}_{}_std'.format('decomposed' if decomposed else 'isotropic', atomic_species)))
            
    arr_data = orm.ArrayData()
    arr_data.label = '{}-MSD'.format(structure.label)
    # Following are the collection of trajectories, not sure why we need this
    for arrayname in msd_iso.get_arraynames():
        arr_data.set_array(arrayname, msd_iso.get_array(arrayname))
    # Following attributes are results_dict of samos.analysis.DynamicsAnalyzer.get_msd()
    for attr, val in msd_iso.get_attrs().items():
        arr_data.set_attribute(attr, val)
    arr_data.set_attribute('nr_of_pinballs', nat_in_traj)
    
    return {'msd_results': arr_data}


@calcfunction
def get_last_step_from_trajectory(trajectory):

    if not isinstance(trajectory, orm.TrajectoryData):
        raise Exception('All my inputs have to be instances of TrajectoryData')
    traj = orm.TrajectoryData()
    for arrname in trajectory.get_arraynames():
        if arrname in ('symbols', 'atomic_species_name'):
            traj.set_array(arrname, trajectory.get_array(arrname))
        elif arrname in ('steps', 'times', 'walltimes'):
            traj.set_array(arrname, np.array([trajectory.get_array(arrname)[0]]))
        else:
            traj.set_array(arrname, np.array([trajectory.get_array(arrname)[-1]]))

    [traj.set_attribute(k, v) for k, v in trajectory.attributes_items() if not k.startswith('array|')]
    if 'timestep_in_fs' in trajectory.attributes:
        traj.set_attribute('sim_time_fs', traj.get_array('steps').size * trajectory.get_attribute('timestep_in_fs'))
    return {'last_step_trajectory': traj}


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
    remove_repeated_last_step = kwargs.pop('remove_repeated_last_step')
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
def get_pinball_factors(trajectory_dft, trajectory_pb):
    from scipy.stats import linregress

    traj_pb_forces = trajectory_pb.get_array('forces')
    # this is required since for dft calculations, the forces of all atoms are outputted
    atom_indices_dft = [i for i, s in enumerate(trajectory_dft.get_attribute('symbols')) if s == 'Li']
    traj_dft_forces = trajectory_dft.get_array('forces')[:,atom_indices_dft,:]

    # I need to remove all the steps that are starting indices due to this stupid thing with the hustler first step. 
    starting_indices = set()
    for traj in (trajectory_pb, trajectory_dft): [starting_indices.add(_) for _ in np.where(trajectory_dft.get_array('steps') == 0)[0]]
    # I also need to remove steps for the trajectory_dft that did not SCF CONVERGE!!!!
    convergence = trajectory_dft.get_array('scf_convergence')
    [starting_indices.add(_) for _ in np.where(~convergence)[0]]
    for idx in sorted(starting_indices, reverse=True):
        traj_dft_forces = np.delete(traj_dft_forces, idx, axis=0)
        traj_pb_forces = np.delete(traj_pb_forces,  idx, axis=0)

    # reshaping to seperate the 4 components of pinball forces in one column
    nstep, nat, ndim = traj_pb_forces.shape
    traj_pb_forces_reshaped = traj_pb_forces.reshape(int(nstep*nat), int(ndim/3), 3)
    traj_pb_forces_rereshaped = []
    for idx in range(1,5): traj_pb_forces_rereshaped.append(traj_pb_forces_reshaped[:,idx,:].flatten())
    traj_dft_forces_reshaped = traj_dft_forces.flatten()
    
    if (traj_pb_forces_rereshaped[1] == 0).all(): traj_pb_forces_rereshaped.pop(1)
    coefs, sum_res, rank, s =  np.linalg.lstsq(np.array(traj_pb_forces_rereshaped).T, traj_dft_forces_reshaped, rcond=None)
    if len(coefs) == 3: coefs = np.insert(coefs, 1, 0)
    mae = np.sqrt(sum_res / len(traj_dft_forces_reshaped))
    r2 = 1.0 - sum_res / traj_dft_forces_reshaped.var() / len(traj_dft_forces_reshaped)
    if r2.size > 0: coefs /= r2
    try: mae_f = float(mae)
    except: mae_f = None

    fitted_force = np.zeros((nstep*nat, 3))
    for i, coef in enumerate(coefs): fitted_force[:,:] += coef*traj_pb_forces_reshaped[:,i+1,:]
    
    slope_before_fit, intercept_before_fit, rvalue_before_fit, pvalue_before_fit, stderr_before_fit = linregress(traj_dft_forces_reshaped.flatten(), traj_pb_forces_reshaped[:,0,:].flatten())
    slope_after_fit, intercept_after_fit, rvalue_after_fit, pvalue_after_fit, stderr_after_fit = linregress(traj_dft_forces_reshaped.flatten(), fitted_force.flatten())
    
    coeff_params = orm.Dict(dict={
        'coefs': coefs.tolist(),
        'mae': mae_f,
        'nr_of_coefs': len(coefs),
        'indices_removed': sorted(starting_indices),
        'linreg_before_fit': {'slope': slope_before_fit,
            'intercept': intercept_before_fit,
            'r2value': rvalue_before_fit**2,
            'pvalue_zero_slope': pvalue_before_fit,
            'stderr': stderr_before_fit},
        'linreg_after_fit': {'slope': slope_after_fit,
            'intercept': intercept_after_fit,
            'r2value': rvalue_after_fit**2,
            'pvalue_zero_slope': pvalue_after_fit,
            'stderr': stderr_after_fit},})

    return {'coefficients': coeff_params}


@calcfunction
def rattle_randomly_structure(structure, parameters):

    parameters_d = parameters.get_dict()
    elements_to_rattle = parameters_d['elements']
    stdev = parameters_d['stdev']
    nr_of_configurations = parameters_d['nr_of_configurations']
    atoms = structure.get_ase()
    # I remove host lattice since it is unchanging
    del atoms[[atom.index for atom in atoms if atom.symbol!=elements_to_rattle]]
    positions = atoms.positions
    nr_of_pinballs = positions.shape[0]
    new_positions = np.repeat(np.array([positions]), nr_of_configurations, axis=0)
    # Now I rattle every atom that is left
    for idx in range(nr_of_pinballs):
        new_positions[:,idx,:] += np.random.normal(0, stdev, (nr_of_configurations, 3))

    cells = np.array([structure.cell] * nr_of_configurations)
    symbols = np.array([str(i.kind_name) for i in structure.sites[:nr_of_pinballs]])
    steps = np.arange(stop=nr_of_configurations, dtype=int)
    
    trajectory_data = orm.TrajectoryData()
    trajectory_data.set_trajectory(
            stepids=steps,
            cells=cells,  
            symbols=symbols,
            positions=new_positions,
            velocities=new_positions # to standardise with aiida, we input dummy values
        )

    trajectory_data.set_attribute('atoms', symbols)
    trajectory_data.set_attribute('timestep_in_fs', parameters_d['timestep_in_fs'])

    trajectory_data.set_attribute('units|positions', 'angstrom')
    trajectory_data.set_attribute('units|cells', 'angstrom')
    trajectory_data.set_attribute('units|velocities', 'atomic')

    return dict(rattled_snapshots=trajectory_data)

@calcfunction
def get_doped_structure(unitcell, doping_parameters):
    """
    Take the input structure and create a doped structures from it.
    :param unitcell: the StructureData node
    :param doping_parameters: DictionaryData instance that contains following keywords - 
        extra_element_concentration - percent by which to increase Li concentration, a negative value
        implies the percent of Li to remove
        element - element to dope
        r_cut - cutoff radius in Angstrom, to check how far newly added atoms should be
        distance - periodic image distance in Angstrom, used to construct supercell
        size - minimum no of atoms in the supercell
    """

    import random, math
    import numpy as np
    from pymatgen.core.periodic_table import Element, Species
    from pymatgen.core.sites import PeriodicSite
    from aiida_flipper.workflows.preprocess import make_supercell_size

    assert isinstance(unitcell, orm.StructureData), f'input structure needs to be an instance of {orm.StructureData}'

    if isinstance(doping_parameters, orm.Dict):
        doping_parameters_d = doping_parameters.get_dict()
    elif isinstance(doping_parameters, dict):
        doping_parameters_d = doping_parameters.copy()
    else:
        raise TypeError('parameters type not valid')

    extra_element_concentration = doping_parameters_d['extra_element_concentration']
    assert extra_element_concentration != 0, 'input Li concentration needs to be non zero'

    r_cut = doping_parameters_d['r_cut']
    element = doping_parameters_d['element']
    distance = doping_parameters_d['distance']
    size = doping_parameters_d['size']

    structure = make_supercell_size(unitcell, distance, size)

    pinball_kinds = [kind for kind in structure.kinds if kind.symbol == element]

    kindnames_to_delithiate = [kind.name for kind in pinball_kinds]

    non_pinball_kinds = [k for i,k in enumerate(structure.kinds) if k.symbol != element]

    non_pinball_sites = [s for s in structure.sites if s.kind_name not in kindnames_to_delithiate]

    pinball_sites = [s for s in structure.sites if s.kind_name in kindnames_to_delithiate]

    pinball_structure = orm.StructureData()

    pinball_structure.set_cell(structure.cell)
    pinball_structure.set_attribute('pinball_structure', True)
    pinball_structure.set_attribute('doped_structure', True)
    pinball_structure.set_extra('original_unitcell', unitcell.uuid)
    pinball_structure.set_attribute('original_unitcell', unitcell.uuid)

    if extra_element_concentration > 0:

        Li_atoms_to_add = math.ceil(len(pinball_sites) * extra_element_concentration)

        structure_pymatgen = structure.get_pymatgen()

        a, b, c = structure_pymatgen.lattice.lengths
        x = np.arange(0, a, 0.5)
        y = np.arange(0, b, 0.5)
        z = np.arange(0, c, 0.5)

        for i in range(Li_atoms_to_add):
            find_new_site = False
            while not find_new_site:
                site_location = np.array([random.choice(x), random.choice(y), random.choice(z)])
                new_site = PeriodicSite(Element(element), site_location, structure_pymatgen.lattice, coords_are_cartesian=True)
                count = 0
                for site in structure_pymatgen:
                    count += 1
                    if site.distance(new_site) < r_cut:
                        break
                if count == structure_pymatgen.num_sites:
                    find_new_site = True
                    structure_pymatgen.append(Element(element), site_location, coords_are_cartesian=True)

        temp_doped_structure = orm.StructureData()
        temp_doped_structure.set_pymatgen(structure_pymatgen)
        doped_pinball_sites = temp_doped_structure.sites[-Li_atoms_to_add:]

        [pinball_structure.append_kind(_) for _ in pinball_kinds]
        [pinball_structure.append_site(_) for _ in pinball_sites]
        [pinball_structure.append_site(_) for _ in doped_pinball_sites]

        [pinball_structure.append_kind(_) for _ in non_pinball_kinds]
        [pinball_structure.append_site(_) for _ in non_pinball_sites]
    
    elif extra_element_concentration < 0:

        Li_atoms_to_remove = math.ceil(len(pinball_sites) * -extra_element_concentration)

        to_delete = set(random.sample(range(len(pinball_sites)), Li_atoms_to_remove))
        doped_pinball_sites = [s for i, s in enumerate(pinball_sites) if not i in to_delete]

        [pinball_structure.append_kind(_) for _ in pinball_kinds]
        [pinball_structure.append_site(_) for _ in doped_pinball_sites]

        [pinball_structure.append_kind(_) for _ in non_pinball_kinds]
        [pinball_structure.append_site(_) for _ in non_pinball_sites]

    pinball_structure.label = pinball_structure.get_formula(mode='count')

    return dict(doped_structure=pinball_structure)