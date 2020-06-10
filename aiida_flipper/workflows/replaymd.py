# -*- coding: utf-8 -*-

from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.links import LinkType
from aiida.engine import ToContext, if_, while_, BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.plugins import CalculationFactory

#from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults
from aiida_quantumespresso.utils.mapping import update_mapping, prepare_process_inputs
#from aiida_quantumespresso.utils.pseudopotential import validate_and_prepare_pseudos_inputs
#from aiida_quantumespresso.utils.resources import get_default_options, get_pw_parallelization_parameters
#from aiida_quantumespresso.utils.resources import cmdline_remove_npools, create_scheduler_resources
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_flipper.utils.utils import get_or_create_input_node
from aiida_flipper.calculations.functions.trajectory import get_structure_from_trajectory, concatenate_trajectory

#PwCalculation = CalculationFactory('quantumespresso.pw')
FlipperCalculation = CalculationFactory('quantumespresso.flipper')
#HustlerCalculation = CalculationFactory('quantumespresso.hustler')


def get_completed_number_of_steps(calc):
    """Read the number of steps from the trajectory."""
    try:
        traj = calc.outputs.output_trajectory
    except NotExistent:
        raise Exception('Output trajectory not found.')
    nstep = calc.inputs.parameters.get_attribute('CONTROL').get('iprint', 1) * \
                (traj.get_attribute('array|positions')[0] - 1)  # the zeroth step is also saved
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
        res, calc = concatenate_trajectory.run_get_node(**traj_d)
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


class ReplayMdWorkChain(PwBaseWorkChain):
    """
    Workchain to run a molecular dynamics Quantum ESPRESSO pw.x calculation with automated error handling and restarts.

    `nstep` can be specified as input or in the `CONTROL.nstep` attribute of the parameters node.

    Velocities are read from the `ATOMIC_VELOCITIES` key of the settings node.
    If not found they will be initialized.
    """
    ## NOTE ##
    # with 'replay' we indicate a molecular dynamics restart, in which positions & velocities are read from the
    # last step of the previous trajectory. In this work chain we always perform a 'dirty restart', i.e. the charge
    # densities restart from scratch (we do not use the `restart_mode='restart'` in QE). In reality, in the Pinball code the
    # host-lattice charge density is always read from file, therefore we always set a `parent_folder`, so that the
    # plugin will copy it when restarting.

    # Question:
    # probably we could define the `_process_class` at the instance level, thus allowing one to choose
    # to use either a `PwCalculation`, `FlipperCalculation`, or `HustlerCalculation` without redefining this class.
    # The drawback is probably that (I guess) the inputs will not be exposed (in the builder?)?
    # Alternatively, one could define subclasses were `_process_class` is set accordingly.
    _process_class = FlipperCalculation  # probably we can define this in a subclass

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
        super().define(spec)
        #spec.expose_inputs(PwCalculation, namespace='pw', exclude=('kpoints',))

        # the calculation namespace is still 'pw'
        spec.inputs['pw']['metadata']['options']['parser_name'].default = 'quantumespresso.flipper'
        spec.inputs['pw']['parent_folder'].required = True
        #spec.inputs.pop('pw.metadata.options.without_xml')

        # this stuff is not supported by pinball:
        spec.inputs['pw'].pop('hubbard_file')
        spec.inputs['pw'].pop('vdw_table')
        spec.inputs.pop('automatic_parallelization')

        spec.input('nstep', valid_type=orm.Int, required=False,
            help='Number of MD steps (it will be read from the input parameters otherwise).')
        #spec.input('is_hustler', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False))
        spec.input('total_energy_max_fluctuation', valid_type=orm.Float, required=False,
            help='The maximum total energy fluctuation allowed (eV). If the total energy has varied more than this '
                 'threshold, the workchain will fail.')

        spec.outline(
            cls.setup,
            cls.validate_parameters,
            cls.validate_kpoints,
            cls.validate_pseudos,
            cls.validate_resources,
            #if_(cls.should_run_init)(
            #    cls.validate_init_inputs,
            #    cls.run_init,
            #    cls.inspect_init,
            #),
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,      # run calculation
                cls.inspect_process,  # evaluate calc
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
            'handle_vcrelax_converged_except_final_scf': False,
            'handle_relax_recoverable_ionic_convergence_error': False,
            'handle_relax_recoverable_electronic_convergence_error': False,
            #'handle_electronic_convergence_not_achieved': False,
            })
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
        print(self.ctx.inputs)
        self.ctx.inputs.pop('vdw_table', None)
        self.ctx.inputs.pop('hubbard_file', None)
        self.ctx.mdsteps_done = 0

    def validate_parameters(self):
        """Validate inputs that might depend on each other and cannot be validated by the spec.

        Also define dictionary `inputs` in the context, that will contain the inputs for the calculation that will be
        launched in the `run_calculation` step.
        """
        #super().validate_parameters()
        self.ctx.inputs.parameters = self.ctx.inputs.parameters.get_dict()

        if not self.ctx.inputs.parameters['CONTROL']['calculation'] == 'md':
            return self.exit_codes.ERROR_INVALID_INPUT_MD_PARAMETERS
        if not self.ctx.inputs.parameters['CONTROL'].get('lflipper', False):
            raise NotImplementedError('Non-pinball MD is not implemented yet.')
        if self.inputs.get('is_hustler', False):
            raise NotImplementedError('Hustler not implemented.')

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
        #self.ctx.is_hustler = self.inputs.is_hustler

#    def validate_kpoints(self):
#        """Validate the inputs related to k-points.
#
#        Either an explicit `KpointsData` with given mesh/path, or a desired k-points distance should be specified. In
#        the case of the latter, the `KpointsData` will be constructed for the input `StructureData` using the
#        `create_kpoints_from_distance` calculation function.
#        """

#    def validate_pseudos(self):
#        """Validate the inputs related to pseudopotentials.
#
#        Either the pseudo potentials should be defined explicitly in the `pseudos` namespace, or alternatively, a family
#        can be defined in `pseudo_family` that will be used together with the input `StructureData` to generate the
#        required mapping.
#        """

#    def validate_resources(self):
#        """Validate the inputs related to the resources.
#
#        One can omit the normally required `options.resources` input for the `PwCalculation`, as long as the input
#        `automatic_parallelization` is specified. If this is not the case, the `metadata.options` should at least
#        contain the options `resources` and `max_wallclock_seconds`, where `resources` should define the `num_machines`.
#        """

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

        If a `restart_calc` has been set in the context, the structure & velocities will be read from its output
        trajectory.
        """
        # NOTE pinball: the parent folder (charge density) does not change, we do not need to specify the restart mode
        if self.ctx.restart_calc:
            # if it is a replay, extract structure and velocities from trajectory of restart_calc
            try:
                prev_trajectory = self.ctx.restart_calc.outputs.output_trajectory
            except (KeyError, exceptions.NotExistent):
                raise RuntimeError('Previous trajectory not found!')
            self.ctx.inputs.parameters['IONS']['ion_velocities'] = 'from_input'
            kwargs = {'trajectory': prev_trajectory,
                      'parameters': get_or_create_input_node(orm.Dict,
                          dict(step_index=-1,
                               recenter=False,
                               create_settings=True,
                               complete_missing=True),
                          store=True),
                      'structure': self.ctx.inputs.structure,
                      'metadata': {'call_link_label': 'get_structure'}}
            if self.ctx.inputs.settings:
                kwargs['settings'] = get_or_create_input_node(orm.Dict, self.ctx.inputs.settings, store=True)

            res, calc = get_structure_from_trajectory.run_get_node(**kwargs)

            self.ctx.inputs.structure = res['structure']
            self.ctx.inputs.settings = res['settings'].get_dict()
            #self.ctx.inputs.parameters['CONTROL']['restart_mode'] = 'restart'  ## NOT NEEDED IN PINBALL
        else:
            # start from scratch, eventually use `initial_velocities` if defined in input settings
            # (these were already added to `self.ctx.inputs.settings` by `validate_parameters`)
            if self.ctx.has_initial_velocities:
                self.ctx.inputs.parameters['IONS']['ion_velocities'] = 'from_input'
            #self.ctx.inputs.parameters['CONTROL']['restart_mode'] = 'from_scratch'  ## NOT NEEDED IN PINBALL
            #self.ctx.inputs.pop('parent_folder', None)

        self.ctx.inputs.parameters['CONTROL']['nstep'] = self.ctx.mdsteps_todo
        self.ctx.inputs.metadata['label'] = '{}_{:02d}'.format(self.inputs.metadata.label, self.ctx.iteration + 1)
        self.ctx.has_initial_velocities = False

        ## if this is not flipper MD
        #if not input_dict['CONTROL'].get('lflipper', False):
        #    input_dict['IONS']['wfc_extrapolation'] = 'second_order'
        #    input_dict['IONS']['pot_extrapolation'] = 'second_order'

        ## HUSTLER STUFF (not implemented)
        if self.inputs.get('is_hustler', False):
            raise NotImplementedError('Hustler not implemented.')
        #   hustler_positions = self.inputs.hustler_positions
        #   if self.ctx.steps_done:
        #       #~ self.ctx.array_splitting_indices.append(self.ctx.steps_done)
        #       inlinec, res = split_hustler_array_inline(
        #           array=hustler_positions, parameters=get_or_create_parameters(dict(index=self.ctx.steps_done))
        #       )
        #       return_d['split_hustler_array_{}'.format(
        #           str(self.ctx.iteration).rjust(len(str(self._MAX_ITERATIONS)), str(0))
        #       )] = inlinec
        #       calc.use_array(res['split_array'])
        #   else:
        #       calc.use_array(hustler_positions)

#    def run_process(self):
#        """Run the next process, taking the input dictionary from the context at `self.ctx.inputs`."""

#    def inspect_process(self):
#        """Analyse the results of the previous process and call the handlers when necessary. [...]"""
#        super().inspect_process()

    def update_mdsteps(self):
        """Get the number of steps of the last trajectory and update the counters. If there are more MD steps to do,
        set `restart_calc` and set the state to not finished.
        """
        # TODO: SHOULD THIS METHOD BE MOVED TO A @process_handler ?
        # the extra 'discard_trajectory' indicates if we wish to discard the trajectory, for whatever reason (not sure it ever happens)

        # if the calculation was successfull, there will be a trajectory
        node = self.ctx.children[self.ctx.iteration - 1]
        try:
            traj = node.outputs.output_trajectory
        except (KeyError, exceptions.NotExistent):
            self.report('No output_trajectory was generated by {}<{}>.'.format(node.label, node.pk))
        else:
            nsteps_run_last_calc = get_completed_number_of_steps(node)
            if not traj.get_extra('discard_trajectory', False):
                self.ctx.mdsteps_todo -= nsteps_run_last_calc
                self.ctx.mdsteps_done += nsteps_run_last_calc
                self.report('{}<{}> ran {} steps ({} done - {} to go).'.format(node.process_label, node.pk, nsteps_run_last_calc, self.ctx.mdsteps_done, self.ctx.mdsteps_todo))
            else:
                self.report('{}<{}> ran {} steps. This trajectory will be DISCARDED!'.format(node.process_label, node.pk, nsteps_run_last_calc))

            # if there are more MD steps to do, set the restart_calc to the last calculation
            if self.ctx.mdsteps_todo > 0:
                self.ctx.restart_calc = node
                self.ctx.is_finished = False

#    def report_error_handled(self, calculation, action):
#        """Report an action taken for a calculation that has failed.
#
#        This should be called in a registered error handler if its condition is met and an action was taken.
#
#        :param calculation: the failed calculation node
#        :param action: a string message with the action taken
#        """
#        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
#        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
#        self.report('Action taken: {}'.format(action))

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


    ### PROCESS HANDLERS ###
    # error codes > 600 are related to MD trajectories
    # Often these errors happen when the calculation is killed in the middle of writing.
    # We can define a specific handler to restart the calculation, but the same will happen
    # if no handler is called: in this case the calculation will be relaunched once, and if it fails
    # again, it the work chain will fail

#    @process_handler(priority=600)
#    def handle_unrecoverable_failure(self, calculation):
#        """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
#        if calculation.is_failed and calculation.exit_status < 400:
#            self.report_error_handled(calculation, 'unrecoverable error, aborting...')
#            return ProcessHandlerReport(True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)

    @process_handler(priority=599)
    def check_energy_fluctuations(self, calculation):
        total_energy_max_fluctuation = self.inputs.get('total_energy_max_fluctuation', None)
        if total_energy_max_fluctuation:
            # checking the fluctuations of the total energy:
            try:
                traj = calculation.outputs.output_trajectory
            except exceptions.NotExistent:
                self.report_error_handled('trajectory not found: skipping energy fluctuations check')
                return
            traj = calculation.outputs.output_trajectory
            total_energies = traj.get_array('total_energies')
            diff = total_energies.max() - total_energies.min()
            print('  Energy fluctuations:', diff, total_energy_max_fluctuation)
            if (diff > total_energy_max_fluctuation):
                self.report_error_handled('{}<{}> : fluctuations of the total energy ({}) exceeded the threshold ({}). Aborting...'.format(
                    calculation.process_label, calculation.pk, diff, total_energy_max_fluctuation))
                ProcessHandlerReport(True, self.exit_codes.ERROR_TOTAL_ENERGY_FLUCTUATIONS)
        return

#    @process_handler(priority=590, exit_codes=[
#        PwCalculation.exit_codes.ERROR_COMPUTING_CHOLESKY,
#    ])
#    def handle_known_unrecoverable_failure(self, calculation):
#        """Handle calculations with an exit status that correspond to a known failure mode that are unrecoverable.
#
#        These failures may always be unrecoverable or at some point a handler may be devised.
#        """
#        self.report_error_handled(calculation, 'known unrecoverable failure detected, aborting...')
#        return ProcessHandlerReport(True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE)

    @process_handler(priority=580, exit_codes=[
        FlipperCalculation.exit_codes.ERROR_OUT_OF_WALLTIME,
        ])
    def handle_out_of_walltime(self, calculation):
        """Handle `ERROR_OUT_OF_WALLTIME` exit code: calculation shut down neatly and we can simply restart."""
        try:
            output_trajectory = calculation.outputs.output_trajectory
        except exceptions.NotExistent:
            #self.ctx.restart_calc = calculation
            self.report_error_handled(calculation, 'out of walltime: no trajectory (should not happen): redo the last calculation')
        else:
            self.ctx.restart_calc = calculation
            self.report_error_handled(calculation, 'out of walltime: get structure & velocities from trajectory, and restart')

        return ProcessHandlerReport(True)

## TODO: check that this error is detected by the flipper parser
    @process_handler(priority=410, exit_codes=[
        FlipperCalculation.exit_codes.ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED,])
    def handle_electronic_convergence_not_achieved(self, calculation):
        """Handle `ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED`: restart from last trajectory step."""
#        factor = self.defaults.delta_factor_mixing_beta
#        mixing_beta = self.ctx.inputs.parameters.get('ELECTRONS', {}).get('mixing_beta', self.defaults.qe.mixing_beta)
#        mixing_beta_new = mixing_beta * factor

        output_trajectory = calculation.outputs.output_trajectory
        self.ctx.restart_calc = calculation
        self.report_error_handled(calculation, 'electronic convergence not reached: get structure & velocities from trajectory, and restart')
#        self.ctx.inputs.parameters.setdefault('ELECTRONS', {})['mixing_beta'] = mixing_beta_new

#        action = 'reduced beta mixing from {} to {} and restarting from the last calculation'.format(
#            mixing_beta, mixing_beta_new)
        self.report_error_handled(calculation, action)
        return ProcessHandlerReport(True)

#    ## add possible flipper-specific error handlers
