# -*- coding: utf-8 -*-
"""Workchain to run hustler level MD calculations using Pinball pw.x. based on Quantum ESPRESSO"""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.links import LinkType
from aiida.engine import ToContext, append_, while_, BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.plugins import CalculationFactory, GroupFactory, WorkflowFactory

## builder imports
from aiida.common.lang import type_check
from aiida_quantumespresso.common.types import ElectronicType, SpinType
SsspFamily = GroupFactory('pseudo.family.sssp')
PseudoDojoFamily = GroupFactory('pseudo.family.pseudo_dojo')
CutoffsPseudoPotentialFamily = GroupFactory('pseudo.family.cutoffs')

from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults
from aiida_quantumespresso.utils.mapping import update_mapping, prepare_process_inputs
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_flipper.utils.utils import get_or_create_input_node
from aiida_flipper.calculations.functions import get_structure_from_trajectory, concatenate_trajectory

PwCalculation = CalculationFactory('quantumespresso.pw')
# FlipperCalculation = CalculationFactory('quantumespresso.flipper')
HustlerCalculation = CalculationFactory('quantumespresso.hustler')


def get_completed_number_of_steps(calc):
    """Read the number of steps from the trajectory."""
    traj = calc.outputs.output_trajectory
    nstep = calc.inputs.parameters.get_attribute('CONTROL').get('iprint', 1) * \
                (traj.get_attribute('array|positions')[0])  # the zeroth step is not the starting input anymore
    return nstep

def get_total_trajectory(workchain, store=False):
    """Collect all the trajectory segment and concatenate them."""
    qb = orm.QueryBuilder()
    qb.append(orm.WorkChainNode, filters={'uuid': workchain.uuid}, tag='replay')
    # TODO: Are filters on the state of the calculation needed here?
    # TODO: add project on extras.discard_trajectory, traj_d defined to skip them
    qb.append(orm.CalcJobNode, with_incoming='replay',
            edge_filters={'type': LinkType.CALL_CALC.value,
                          'label': {'like': 'iteration_%'}},
            edge_project='label', tag='calc', edge_tag='rc')
    qb.append(orm.TrajectoryData, with_incoming='calc', edge_filters={'label': 'output_trajectory'},
            project=['*'], tag='traj')
    traj_d = {item['rc']['label'].replace('iteration_', 'trajectory_'): item['traj']['*'] for item in qb.iterdict()}  ## if not extras.discard_trajectory

    # if I have produced several trajectories, I concatenate them here: (no need to sort them)
    if (len(traj_d) > 1):
        traj_d['metadata'] = {'call_link_label': 'concatenate_trajectory', 'store_provenance': store}
        # because positions are read from hustler.pos which is prepared by hustlercalcjob, there are no repeated steps
        traj_d.update({'remove_repeated_last_step': False})
        res = concatenate_trajectory(**traj_d)
        return res['concatenated_trajectory']
    elif (len(traj_d) == 1):
        # no reason to concatenate if I have only one trajectory (saves space in repository)
        return list(traj_d.values())[0]
    else:
        return None

def get_slave_calculations(workchain):
    """
    Returns a list of the calculations that was called by the WF, ordered.
    """
    qb = orm.QueryBuilder()
    qb.append(orm.WorkChainNode, filters={'uuid': workchain.uuid}, tag='m')
    qb.append(orm.CalcJobNode, with_incoming='m',
              edge_project='label', edge_filters={'label': {'like': 'iteration_%'}},
              tag='c', edge_tag='mc', project='*')
    calc_d = {item['mc']['label']: item['c']['*'] for item in qb.iterdict()}
    sorted_calcs = sorted(calc_d.items())
    return list(zip(*sorted_calcs))[1] if sorted_calcs else None


class ReplayMDHustlerWorkChain(PwBaseWorkChain):
    """
    Workchain to run a molecular dynamics Quantum ESPRESSO pw.x calculation with automated error handling and restarts.

    `nstep` can be specified as input or in the `CONTROL.nstep` attribute of the parameters node.

    Velocities are read from the `ATOMIC_VELOCITIES` key of the settings node.
    If not found they will be initialized.
    """
    _process_class = HustlerCalculation  

    defaults = AttributeDict({
        'qe': qe_defaults,
        'delta_threshold_degauss': 30,
        'delta_factor_degauss': 0.1,
        'delta_factor_mixing_beta': 0.8,
        'delta_factor_max_seconds': 0.90,
        'delta_max_seconds': 180,
        'delta_factor_nbnd': 0.05,
        'delta_minimum_nbnd': 4,
    })

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        # NOTE: input, outputs, and exit_codes are inherited from PwBaseWorkChain
        super().define(spec)
        spec.expose_inputs(PwCalculation, namespace='pw', exclude=('kpoints',))

        # the calculation namespace is still 'pw'
        spec.inputs['pw']['parent_folder'].required = True
        #spec.inputs.pop('pw.metadata.options.without_xml')

        # this stuff is not supported by pinball:
        spec.inputs['pw'].pop('hubbard_file')
        spec.inputs['pw'].pop('vdw_table')
        spec.inputs.pop('automatic_parallelization')

        spec.input('nstep', valid_type=orm.Int, required=False,
            help='Number of MD steps it will be read from the input parameters otherwise; these many snapshots will be extracted from input trajectory.')
        spec.input('hustler_snapshots', valid_type=orm.TrajectoryData, required=False,
            help='Trajectory containing the uncorrelated configurations to be used in hustler calculation.')
        spec.outline(
            cls.setup,
            cls.validate_parameters,
            cls.validate_kpoints,
            cls.validate_pseudos,
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,
                cls.inspect_process,
                cls.update_mdsteps,
            ),
            cls.results,
        )

        #spec.expose_outputs(PwCalculation)
        spec.outputs.clear()
        spec.output('total_trajectory', valid_type=orm.TrajectoryData, required=True,
                help='The full concatenated trajectory.')
        spec.default_output_node = 'total_trajectory'

        spec.inputs['handler_overrides'].default = lambda: orm.Dict(dict={
            'sanity_check_insufficient_bands': False,
            'handle_out_of_walltime': True,
            'handle_vcrelax_converged_except_final_scf': False,
            'handle_relax_recoverable_ionic_convergence_error': False,
            'handle_relax_recoverable_electronic_convergence_error': False,
            #'handle_electronic_convergence_not_achieved': False,
            })

        # TODO: we should pop the spec.exit_codes that do not apply
        spec.exit_code(205, 'ERROR_INVALID_INPUT_MD_PARAMETERS',
            message='Input parameters for molecular dynamics are not correct.')
        spec.exit_code(206, 'ERROR_INVALID_INPUT_VELOCITIES',
            message='Velocities are not compatible with the number of atoms of the structure.')
        spec.exit_code(601, 'ERROR_TOTAL_ENERGY_FLUCTUATIONS',
            message='Fluctuations of the total energy exceeded the set threshold.')

    def setup(self):
        """Call the `setup` of the `PwBaseWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop.
        """
        super().setup()
        #self.ctx.restart_calc = None
        #self.ctx.inputs = AttributeDict(self.exposed_inputs(PwCalculation, 'pw'))
        self.ctx.inputs.pop('vdw_table', None)
        self.ctx.inputs.pop('hubbard_file', None)
        self.ctx.mdsteps_done = 0
        
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_flipper.workflows import protocols as proto
        return files(proto) / 'replayh.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        stash_directory,
        hustler_snapshots=None,
        nstep=None,
        protocol=None,
        overrides=None,
        electronic_type=ElectronicType.INSULATOR,
        spin_type=SpinType.NONE,
        initial_magnetic_moments=None,
        **_
    ):
        """
        !! This is a copy of PwBaseWorkChain get_builder_from_protocol() with a change of default
         electronic type to insulator and addition of hustler inputs and loading a trajectory for hustler calculation!!
        Return a builder prepopulated with inputs selected according to the chosen protocol.
        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param stash_directory: the location of charge densities of host lattice
        :param hustler_snapshots: a trajectory file typically the output of a/multiple flipper calculation(s) from which I shall extract `nstep` configurations
        :param nstep: the number of MD steps to perform, which in case of hustler calculation means the number of configurations on which pinball/DFT forces shall be evaluated
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param electronic_type: indicate the electronic character of the system through ``ElectronicType`` instance.
        :param spin_type: indicate the spin polarization type to use through a ``SpinType`` instance.
        :param initial_magnetic_moments: optional dictionary that maps the initial magnetic moment of each kind to a
            desired value for a spin polarized calculation. Note that for ``spin_type == SpinType.COLLINEAR`` an 
            initial guess for the magnetic moment is automatically set in case this argument is not provided.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.workflows.protocols.utils import get_starting_magnetization

        if isinstance(code, str):
            code = orm.load_code(code)

        type_check(code, orm.Code)
        type_check(electronic_type, ElectronicType)
        type_check(spin_type, SpinType)

        if electronic_type not in [ElectronicType.METAL, ElectronicType.INSULATOR]:
            raise NotImplementedError(f'electronic type `{electronic_type}` is not supported.')

        if spin_type not in [SpinType.NONE, SpinType.COLLINEAR]:
            raise NotImplementedError(f'spin type `{spin_type}` is not supported.')

        if initial_magnetic_moments is not None and spin_type is not SpinType.COLLINEAR:
            raise ValueError(f'`initial_magnetic_moments` is specified but spin type `{spin_type}` is incompatible.')

        inputs = cls.get_protocol_inputs(protocol, overrides)

        meta_parameters = inputs.pop('meta_parameters')
        pseudo_family = inputs.pop('pseudo_family')

        natoms = len(structure.sites)

        try:
            pseudo_set = (PseudoDojoFamily, SsspFamily, CutoffsPseudoPotentialFamily)
            pseudo_family = orm.QueryBuilder().append(pseudo_set, filters={'label': pseudo_family}).one()[0]
        except exceptions.NotExistent as exception:
            raise ValueError(
                f'required pseudo family `{pseudo_family}` is not installed. Please use `aiida-pseudo install` to'
                'install it.'
            ) from exception

        try:
            cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=structure, unit='Ry')
        except ValueError as exception:
            raise ValueError(
                f'failed to obtain recommended cutoffs for pseudo family `{pseudo_family}`: {exception}'
            ) from exception

        parameters = inputs['pw']['parameters']
        # parameters['CONTROL']['etot_conv_thr'] = natoms * meta_parameters['etot_conv_thr_per_atom']
        parameters['ELECTRONS']['conv_thr'] = natoms * meta_parameters['conv_thr_per_atom']
        parameters['SYSTEM']['ecutwfc'] = cutoff_wfc
        parameters['SYSTEM']['ecutrho'] = cutoff_rho

        if electronic_type is ElectronicType.METAL:
            parameters['SYSTEM']['occupations'] = 'smearing'
            parameters['SYSTEM'].update({'degauss': 0.01, 'smearing': 'cold'})

        if spin_type is SpinType.COLLINEAR:
            starting_magnetization = get_starting_magnetization(structure, pseudo_family, initial_magnetic_moments)

            parameters['SYSTEM']['nspin'] = 2
            parameters['SYSTEM']['starting_magnetization'] = starting_magnetization

        # pylint: disable=no-member
        builder = cls.get_builder()
        builder.pw['code'] = code
        builder.pw['pseudos'] = pseudo_family.get_pseudos(structure=structure)
        builder.pw['structure'] = structure
        builder.pw['parameters'] = orm.Dict(dict=parameters)
        builder.pw['metadata'] = inputs['pw']['metadata']
        if 'parallelization' in inputs['pw']:
            builder.pw['parallelization'] = orm.Dict(dict=inputs['pw']['parallelization'])
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        if hustler_snapshots: builder['hustler_snapshots'] = hustler_snapshots
        if 'settings' in inputs['pw']:
            builder['pw'].settings = orm.Dict(dict=inputs['pw']['settings'])
            if inputs['pw']['settings']['gamma_only']:
                kpoints = orm.KpointsData()
                kpoints.set_kpoints_mesh([1,1,1])
                builder.kpoints = kpoints
            else: 
                raise NotImplementedError('Only gamma k-points possible in hustler calculations.')

        builder['pw']['parent_folder'] = stash_directory
        if nstep: builder['nstep'] = nstep
        else: builder['nstep'] = orm.Int(inputs['nstep'])

        # pylint: enable=no-member
        return builder

    def validate_parameters(self):
        """Validate inputs that might depend on each other and cannot be validated by the spec.

        Also define dictionary `inputs` in the context, that will contain the inputs for the calculation that will be
        launched in the `run_calculation` step.
        """
        #super().validate_parameters()
        self.ctx.inputs.parameters = self.ctx.inputs.parameters.get_dict()

        if not self.ctx.inputs.parameters['CONTROL']['calculation'] == 'md':
            return self.exit_codes.ERROR_INVALID_INPUT_MD_PARAMETERS
        if not self.ctx.inputs.parameters['CONTROL'].get('lhustle', False):
            raise NotImplementedError('Please run flipper workchain.')

        nstep = self.ctx.inputs.parameters['CONTROL'].get('nstep', None)
        inp_nstep = self.inputs.get('nstep')
        if inp_nstep and nstep:
            self.report('You cannot specify both "nstep" and "parameters.CONTROL.nstep"')
            return self.exit_codes.ERROR_INVALID_INPUT_MD_PARAMETERS
        elif inp_nstep is None and nstep is None:
            self.report('You must specify either "nstep" or "parameters.CONTROL.nstep"')
            return self.exit_codes.ERROR_INVALID_INPUT_MD_PARAMETERS
        elif inp_nstep:
            nstep = inp_nstep.value
        self.ctx.mdsteps_todo = nstep
        self.ctx.nsteps = nstep

        self.ctx.inputs.settings = self.ctx.inputs.settings.get_dict() if 'settings' in self.ctx.inputs else {}

        # In the pinball, the parent folder contains the host-lattice charge density and is always given as input,
        # so this is done automatically during the setup:
        #   self.ctx.inputs.parent_folder = self.inputs.pw.parent_folder
        # We will not change the parent_folder after a replay

        # At each replay, a restart_calc will be set, meaning that md should be (dirty)-restarted from the trajectory
        # produced by the last calculation
        self.ctx.restart_calc = None

        # if velocities were given in the input parameters or settings, we will use them
        initial_velocities = self.ctx.inputs.settings.get('ATOMIC_VELOCITIES', None)
        params_velocities = self.ctx.inputs.parameters.pop('ATOMIC_VELOCITIES', None)
        if initial_velocities and params_velocities:
            self.report('Please specify initial ATOMIC_VELOCITIES either in parameters or in settings.')
            return self.exit_codes.ERROR_INVALID_INPUT_MD_PARAMETERS
        elif params_velocities:
            self.ctx.inputs.settings['ATOMIC_VELOCITIES'] = params_velocities
            initial_velocities = params_velocities
        if initial_velocities:
            self.ctx.has_initial_velocities = True
            if len(initial_velocities) != len(self.ctx.inputs.structure.sites):
                raise self.exit_codes.ERROR_INVALID_INPUT_VELOCITIES
        else:
            self.ctx.has_initial_velocities = False

        self.ctx.inputs.parameters.setdefault('CONTROL', {})
        self.ctx.inputs.parameters['CONTROL'].setdefault('calculation', 'md')

        # Checking if the hustler snapshot was generated with same structure that is input to me
        hustler_snapshots = self.inputs.get('hustler_snapshots')
        qb = orm.QueryBuilder()
        qb.append(orm.TrajectoryData, filters={'id':{'==':hustler_snapshots.pk}}, tag='traj')
        qb.append(CalculationFactory('quantumespresso.flipper'), with_outgoing='traj')
        if qb.count():
            cc, = qb.first()
            struct = cc.inputs['structure']
            if struct.pk != self.ctx.inputs.structure.pk: raise Exception('Structure of previous trajectory not matching with input structure, please provide right trajectory.')
        else:
            self.report('WorkChain of previous trajectory not found, trying preceding calcfunction')
            qb = orm.QueryBuilder()
            qb.append(orm.TrajectoryData, filters={'id':{'==':hustler_snapshots.pk}}, tag='traj')
            qb.append(orm.CalcFunctionNode, with_outgoing='traj', tag='calcfunc')
            qb.append(orm.TrajectoryData, with_outgoing='calcfunc', tag='old_traj')
            qb.append(WorkflowFactory('quantumespresso.flipper.replaymd'), with_outgoing='old_traj', tag='replay')
            if qb.count():
                wc, = qb.first()
                struct = wc.inputs['pw']['structure']
                if struct.pk != self.ctx.inputs.structure.pk: raise Exception('Structure of previous trajectory not matching with input structure, please provide right trajectory.')
            else:
                self.report('Provided trajectory does not match any calcfunction or workchain; continuing nonetheless')

    def set_max_seconds(self, max_wallclock_seconds):
        # called by self.validate_resources
        """Set the `max_seconds` to a fraction of `max_wallclock_seconds` option to prevent out-of-walltime problems.

        :param max_wallclock_seconds: the maximum wallclock time that will be set in the scheduler settings.
        """
        max_seconds_factor = self.defaults.delta_factor_max_seconds
        max_seconds_delta = self.defaults.delta_max_seconds
        # give the code 3 minutes to terminate gracefully, or 90% of your estimate (for very low numbers, to avoid negative)
        max_seconds = max((max_seconds_delta, max_wallclock_seconds * max_seconds_factor))
        self.ctx.inputs.parameters['CONTROL']['max_seconds'] = max_seconds
        # if needed, set the scheduler max walltime to be at least 1 minute longer than max_seconds
        self.ctx.inputs.metadata.options.update({'max_wallclock_seconds': max(max_wallclock_seconds, max_seconds + 60)})

    def should_run_process(self):
        """Return whether a new process should be run.

        This is the case as long as the last process has not finished successfully, the maximum number of restarts has
        not yet been exceeded, and the number of desired MD steps has not been reached.
        """
        return not(self.ctx.is_finished) and (self.ctx.iteration < self.inputs.max_iterations.value) and (self.ctx.mdsteps_todo > 0)

    def prepare_process(self):
        """Prepare the inputs for the next calculation.

        In the pinball, the `parent_folder` is never changed, and the `restart_mode` is not set.

        If a `restart_calc` has been set in the context, the structure & velocities will be read from its output trajectory.
        """
        self.ctx.inputs.parameters['CONTROL']['nstep'] = self.ctx.mdsteps_todo
        self.ctx.inputs.metadata['label'] = f'hustler_{self.ctx.iteration:02d}'
        self.ctx.has_initial_velocities = False

        # I extract `nsteps` configuration from the input trajectory, by dividing the trajecory in `nsteps+1` chunks
        # and skipping the 1st snapshot which is the default starting positions
        hustler_snapshots = self.inputs.get('hustler_snapshots')
        arraynames = hustler_snapshots.get_arraynames()
        traj = orm.TrajectoryData()
        for arrname in arraynames:
            if arrname in ('symbols', 'atomic_species_name'):
                traj.set_array(arrname, hustler_snapshots.get_array(arrname))
            else:
                to_skip, = hustler_snapshots.get_shape('steps')
                tmp_array = hustler_snapshots.get_array(arrname)[::int(to_skip/self.ctx.nsteps)]
                # to skip the positions calculated before
                traj.set_array(arrname, tmp_array[self.ctx.mdsteps_done+1:])
        [traj.set_attribute(k, v) for k, v in hustler_snapshots.attributes_items() if not k.startswith('array|')]
        if 'timestep_in_fs' in hustler_snapshots.attributes:
            traj.set_attribute('sim_time_fs', traj.get_array('steps').size * hustler_snapshots.get_attribute('timestep_in_fs'))

        self.ctx.inputs['hustler_snapshots'] = traj

    def update_mdsteps(self):
        """Get the number of steps of the last trajectory and update the counters. If there are more MD steps to do,
        set `restart_calc` and set the state to not finished.
        """
        # The extra 'discard_trajectory' indicates if we wish to discard the trajectory, for whatever reason (not sure it ever happens).
        # If the calculation was successfull, there will be a trajectory
        # In this case we we shall restart from this calculation, otherwise restart_calc is not modified, such that we
        # will restart from the previous one.
        node = self.ctx.children[self.ctx.iteration - 1]
        try:
            traj = node.outputs.output_trajectory
        except (KeyError, exceptions.NotExistent):
            self.report('No output_trajectory was generated by {}<{}>.'.format(node.label, node.pk))
            # restart_calc is not updated, so we will restart from the last calculation (i.e. we retry the same thing)
        else:
            nsteps_run_last_calc = get_completed_number_of_steps(node)
            if not traj.get_extra('discard_trajectory', False):
                self.ctx.mdsteps_todo -= nsteps_run_last_calc
                self.ctx.mdsteps_done += nsteps_run_last_calc
                self.report('{}<{}> ran {} steps ({} done - {} to go).'.format(node.process_label, node.pk, nsteps_run_last_calc, self.ctx.mdsteps_done, self.ctx.mdsteps_todo))

                # if there are more MD steps to do, set the restart_calc to the last calculation
                if self.ctx.mdsteps_todo > 0:
                    self.ctx.restart_calc = node
                    self.ctx.is_finished = False
            else:
                self.report('{}<{}> ran {} steps. This trajectory will be DISCARDED!'.format(node.process_label, node.pk, nsteps_run_last_calc))

    def results(self):  # pylint: disable=inconsistent-return-statements
        """Concatenate the trajectories and attach the outputs."""
        # get the concatenated trajectory, even if the max number of iterations have been reached
        traj = get_total_trajectory(self, store=True)
        if traj:
            self.out('total_trajectory', traj)
        else:
            self.report('No trajectories were produced!')
#            return self.exit_codes.ERROR_NO_TRAJECTORY_PRODUCED

        node = self.ctx.children[self.ctx.iteration - 1]

        # We check the `is_finished` attribute of the work chain and not the successfulness of the last process
        # because the error handlers in the last iteration can have qualified a "failed" process as satisfactory
        # for the outcome of the work chain and so have marked it as `is_finished=True`.
        if not self.ctx.is_finished and self.ctx.iteration >= self.inputs.max_iterations.value:
            self.report('reached the maximum number of iterations {}: last ran {}<{}>'.format(
                self.inputs.max_iterations.value, self.ctx.process_name, node.pk))
            return self.exit_codes.ERROR_MAXIMUM_ITERATIONS_EXCEEDED  # pylint: disable=no-member

        self.report('work chain completed after {} iterations'.format(self.ctx.iteration))

    def _wrap_bare_dict_inputs(self, port_namespace, inputs):
        """Wrap bare dictionaries in `inputs` in a `Dict` node if dictated by the corresponding inputs portnamespace.

        :param port_namespace: a `PortNamespace`
        :param inputs: a dictionary of inputs intended for submission of the process
        :return: an attribute dictionary with all bare dictionaries wrapped in `Dict` if dictated by the port namespace
        """
        from aiida.engine.processes import PortNamespace

        wrapped = {}

        for key, value in inputs.items():

            if key not in port_namespace:
                wrapped[key] = value
                continue

            port = port_namespace[key]

            if isinstance(port, PortNamespace):
                wrapped[key] = self._wrap_bare_dict_inputs(port, value)
            elif port.valid_type == orm.Dict and isinstance(value, dict):
                wrapped[key] = get_or_create_input_node(orm.Dict, value, store=True)
            else:
                wrapped[key] = value

        return AttributeDict(wrapped)

    @process_handler(priority=700)
    def handle_salvage_output_trajectory(self, calculation):
        """Check if an output trajectory was generated, no matter if the calculation failed, and restart."""
        try:
            traj = calculation.outputs.output_trajectory
        except exceptions.NotExistent:
            # no output_trajectory, go through the other error handlers
            return
        else:
            # restart from trajectory
            self.ctx.restart_calc = calculation
            if calculation.exit_status != 0: self.report_error_handled(calculation, 'Restarting calculation...')
            return ProcessHandlerReport(True)