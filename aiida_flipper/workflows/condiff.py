# -*- coding: utf-8 -*-
"""Mother Workchain that calls LinDiffusionWorkChain and FittingWorkChain to run MD simulations using 
Pinball pw.x. based on Quantum ESPRESSO and fit pinball hyperparameters resepectively"""
import numpy as np
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.engine import ToContext, append_, if_, while_, WorkChain
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.plugins import WorkflowFactory

from aiida_flipper.calculations.functions.functions import get_structure_from_trajectory, rattle_randomly_structure
from aiida_flipper.utils.utils import get_or_create_input_node

LinDiffusionWorkChain = WorkflowFactory('quantumespresso.flipper.lindiffusion')
FittingWorkChain = WorkflowFactory('quantumespresso.flipper.fitting')

class ConvergeDiffusionWorkChain(ProtocolMixin, WorkChain): # maybe BaseRestartWorkChain?
    """The main Workchain of aiida_flipper that calls all other workchains to 
    run following two workchains in a self-consistent-loop
    run MD calculations untill estimated diffusion coefficient converges,
    run a fitting workflow to find pinball hyperparameters for next MD runs
    using Pinball Quantum ESPRESSO pw.x."""
    # _process_class = LinDiffusionWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(LinDiffusionWorkChain, namespace='ld',
            exclude=('clean_workdir', 'structure', 'parent_folder'),
            namespace_options={'help': 'Inputs for the `LinDiffusionWorkChain` for estimating diffusion coefficient with MD runs are called in the `ld` namespace.'})
        spec.expose_inputs(FittingWorkChain, namespace='ft',
            exclude=('clean_workdir', 'structure', 'parent_folder'),
            namespace_options={'help': 'Inputs for the `FittingWorkChain` for fitting calculations that are called in the `ft` namespace.'})
        spec.input('structure', valid_type=orm.StructureData, help='The input supercell structure.')
        spec.input('parent_folder', valid_type=orm.RemoteData, required=True,
            help='The stashed directory containing charge densities of host lattice.')
        spec.input('first_fit_with_random_rattling', valid_type=orm.Bool, required=False, help='If true I do a first fit of pinball hyperparameters using randomly rattled positions of the input structure instead of using unity as parameters.')
        spec.input('run_last_lindiffusion', valid_type=orm.Bool, required=False, help='If true I do an additional run of lindiffusion workchain by doubling the `nstep` and using the previous pinball hyperparameters.')
        spec.input('diffusion_convergence_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing the parameters used to converge diffusion coefficient.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup,
            cls.run_first_fit,
            while_(cls.should_run_process)(
            cls.run_lindiff,
            cls.run_fit,
            cls.inspect_process,
            ),
            cls.run_last_lindiff,
            cls.results,
        )

        spec.exit_code(707, 'ERROR_FITTING_FAILED',
            message='The linear regression to fit pinball and dft forces failed.')
        spec.exit_code(703, 'ERROR_CHARGE_DENSITIES_NOT_FOUND',
            message='Either the stashed charge densities or the flipper compatible supercell structure not found.')
        spec.exit_code(708, 'ERROR_LINDIFFUSION_FAILED',
            message='The LinDiffusionWorkChain sub process failed.')
        spec.exit_code(709, 'ERROR_LAST_LINDIFFUSION_FAILED',
            message='The last LinDiffusionWorkChain sub process failed.')
        spec.exit_code(704, 'ERROR_TRAJECTORY_NOT_FOUND',
            message='The output trajectory of ReplayMDWorkChain not found.')
        # spec.expose_outputs(ReplayMDWorkChain)
        spec.output('msd_results', valid_type=orm.ArrayData,
            help='The dictionary containing the results of msd calculations using the samos library of the last LinDiffusionWorkChain')
        spec.output('total_trajectory', valid_type=orm.TrajectoryData,
            help='The full concatenated trajectory of the last LinDiffusinWorkChains.')
        spec.output('coefficients', valid_type=orm.Dict,
            help='The dictionary containing the final pinball hyperparameters(keyword - `coefs`) along with linear regression values.')
        
    def setup(self):
        """Input validation and context setup."""

        self.ctx.diffusion_counter = 0
        self.ctx.converged = False
        self.ctx.last_lindiffusion_previous_trajectory = None
        self.ctx.current_structure = self.inputs.structure
        try:
            original_unitcell = orm.load_node(self.ctx.current_structure.extras['original_unitcell'])
            self.report(f'Starting workchain on Structure {original_unitcell.get_formula()} (pk: <{self.ctx.current_structure.id}>)')
        except:
            self.report(f'Starting workchain on Structure {self.ctx.current_structure.get_formula()} (pk: <{self.ctx.current_structure.id}>)')

        # I make empty lists of workchains for comparison
        self.ctx.workchains_fitting, self.ctx.workchains_lindiff = [], []

        # I store input dictionaries in context variables
        self.ctx.diffusion_convergence_parameters_d = self.inputs.diffusion_convergence_parameters.get_dict()

        self.ctx.lindiff_inputs = AttributeDict(self.exposed_inputs(LinDiffusionWorkChain, namespace='ld'))
        self.ctx.fitting_inputs = AttributeDict(self.exposed_inputs(FittingWorkChain, namespace='ft'))
        self.ctx.lindiff_inputs.md.pw.parameters = self.ctx.lindiff_inputs.md.pw.parameters.get_dict()
        # Without putting as a dict inside the namespace of lindiff, msd_parameters can't be updated later
        self.ctx.lindiff_inputs.msd_parameters = self.ctx.lindiff_inputs.msd_parameters.get_dict()
        self.ctx.lindiff_inputs.diffusion_parameters = self.ctx.lindiff_inputs.diffusion_parameters.get_dict()
        # I store this in context variable to update for every MD run after the first one
        self.ctx.max_lindiff_iterations = self.ctx.lindiff_inputs.diffusion_parameters['max_md_iterations']
        if self.ctx.lindiff_inputs.get('coefficients'):

            self.report(f'I was given pinball coefficients <{self.ctx.lindiff_inputs.coefficients.pk}>')

            # Adding the fitting and lindiff workchains that generated this pinball parameter 
            qb = orm.QueryBuilder()
            qb.append(orm.Dict, filters={'uuid':{'==':self.ctx.lindiff_inputs.coefficients.uuid}}, tag='coefs')
            qb.append(WorkflowFactory('quantumespresso.flipper.fitting'), with_outgoing='coefs', tag='fit')
            qb.append(orm.StructureData, with_outgoing='fit', tag='structure')
            # I look for all fit and lindiff wcs that led to this coefficient
            qb.append(WorkflowFactory('quantumespresso.flipper.fitting'), with_incoming='structure', filters={'and':[{'attributes.process_state':{'==':'finished'}}, {'attributes.exit_status':{'==':0}}]}, tag='fit2')
            # I sort in ascending order so that latest WorkChains are at the end like they should be
            qb.order_by({'fit2':[{'id':{'order':'asc'}}]})
            self.ctx.workchains_fitting = qb.all(flat=True)

            # I do this again otherwise the LinDiffusionWorkChains are counted multiple times
            qb = orm.QueryBuilder()
            qb.append(orm.Dict, filters={'uuid':{'==':self.ctx.lindiff_inputs.coefficients.uuid}}, tag='coefs')
            qb.append(WorkflowFactory('quantumespresso.flipper.fitting'), with_outgoing='coefs', tag='fit')
            qb.append(orm.StructureData, with_outgoing='fit', tag='structure')
            qb.append(WorkflowFactory('quantumespresso.flipper.lindiffusion'), with_incoming='structure', filters={'and':[{'attributes.process_state':{'==':'finished'}}, {'attributes.exit_status':{'==':0}}]}, tag='lin2')
            qb.order_by({'lin2':[{'id':{'order':'asc'}}]})
            self.ctx.workchains_lindiff = qb.all(flat=True)

            self.report(f'FittingWorkChains {[wc.id for wc in self.ctx.workchains_fitting]} which are ancestor to the coefficients <{self.ctx.lindiff_inputs.coefficients.pk}> will be considered in convergence logic')
            self.report(f'LinDiffusionWorkChains {[wc.id for wc in self.ctx.workchains_lindiff]} which are ancestor to the coefficients <{self.ctx.lindiff_inputs.coefficients.pk}> will be considered in convergence logic')

            self.ctx.diffusion_counter += len(self.ctx.workchains_fitting)

            # I check if convergence is already achieved, using the same logic as used in `inspect_process`
            param_d = self.ctx.diffusion_convergence_parameters_d
            # I will start checking when minimum no. of iterations are reached
            if self.ctx.diffusion_counter > param_d['min_ld_iterations']:
                # Since I am here, it means I need to check the last 3 calculations to
                # see whether I converged or need to run again:
                # Now let me see the pinball coefficients that I get and if they have converged
                # I consider it converged if either the last 3 estimates have not changed more than the threshold or the difference of the last 2 estimates is within the threshold
                self.report(f'I can check for convergence because {self.ctx.diffusion_counter} iterations have already been run.')
                coefficients = np.array([workchain_fitting.outputs.coefficients.get_dict()['coefs'] for workchain_fitting in self.ctx.workchains_fitting])
                stddev_3 = np.std(coefficients[-3:], axis=0)
                # checking the variation (relative error) in last 2 fits, if the standard deviation of last 3 fits is too high
                difference_2 = abs((coefficients[-1]-coefficients[-2])/coefficients[-1])
                # For only local interactions, 2nd coefficient is always 0
                if np.isnan(difference_2)[1]: difference_2[1] = 0

                if (stddev_3 < param_d['coefficient_threshold_std']).all() and (difference_2 < param_d['coefficient_threshold_diff']).all():
                    # I have converged, yay me!
                    self.report(f'Diffusion converged with std = {stddev_3} < threshold = {param_d["coefficient_threshold_std"]}')
                    self.ctx.converged = True
                elif (difference_2 < param_d['coefficient_threshold_diff']).all():
                    self.report(f'Last two estimates of Pinball parameters have converged with relative error = {difference_2} < threshold = {param_d["coefficient_threshold_diff"]}')
                    self.ctx.converged = True
                elif (stddev_3 < param_d['coefficient_threshold_std']).any() and (difference_2 < param_d['coefficient_threshold_diff']).any():
                    self.report(f'Not all Pinball parameters have converged with std = {stddev_3} and relative error = {difference_2}, so I start another MD iteration')
                    self.ctx.converged = False
                else:
                    self.report(f'The Pinball parameters have not converged with std = {stddev_3} and relative error = {difference_2}')
                    self.ctx.converged = False
                    
            # If I have already converged and if a previous trajectory is provided, I run the last MD with it
            if self.ctx.converged and self.ctx.lindiff_inputs.md.get('previous_trajectory') and self.inputs.get('run_last_lindiffusion'):
                self.ctx.last_lindiffusion_previous_trajectory = self.ctx.lindiff_inputs.md.pop('previous_trajectory', None)

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_flipper.workflows import protocols as proto
        return files(proto) / 'condiff.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, parent_folder, protocol=None, overrides=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol, usually takes the pseudo potential family.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        # I cannot use args = (code, structure, parent_folder, protocol) in get_builder_from_protocol(*args) since the 
        # order of the variables is different in the  of LinDiffusionWorkChain

        lindiff = LinDiffusionWorkChain.get_builder_from_protocol(code=code, structure=structure, parent_folder=parent_folder, protocol=protocol, overrides=inputs['ld'], **kwargs)

        lindiff.pop('structure', None)
        lindiff.pop('clean_workdir', None)
        lindiff.pop('parent_folder', None)

        fitting = FittingWorkChain.get_builder_from_protocol(code=code, structure=structure, parent_folder=parent_folder, protocol=protocol, overrides=inputs['ft'], **kwargs)

        fitting.pop('structure', None)
        fitting.pop('clean_workdir', None)
        fitting.pop('parent_folder', None)

        builder = cls.get_builder()
        builder.ld = lindiff
        builder.ft = fitting

        builder.structure = structure
        builder.parent_folder = parent_folder
        builder.first_fit_with_random_rattling = orm.Bool(inputs['first_fit_with_random_rattling'])
        builder.run_last_lindiffusion = orm.Bool(inputs['run_last_lindiffusion'])
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.diffusion_convergence_parameters = orm.Dict(dict=inputs['diffusion_convergence_parameters'])

        return builder

    def should_run_process(self):
        """Return whether a lindiffusion and fitting workchains should be run.

        This is the case as long as the last process has not finished successfully, and the number of maximum replays has not been reached or diffusion coefficient converged or minimum number of replays has not been reached.
        """
        if (self.ctx.diffusion_convergence_parameters_d['min_ld_iterations'] > self.ctx.diffusion_counter): return True
        elif (not(self.ctx.converged) and (self.ctx.diffusion_convergence_parameters_d['max_ld_iterations'] > self.ctx.diffusion_counter)): return True
        else: return False

    def run_first_fit(self):
        """
        Runs a fitting workflow on positions generated from random rattling of input structure
        """

        if self.inputs.get('first_fit_with_random_rattling') and self.ctx.diffusion_counter==0:

            self.ctx.diffusion_counter +=1

            inputs = self.ctx.fitting_inputs
            inputs['parent_folder'] = self.inputs.parent_folder

            inputs['structure'] = self.ctx.current_structure

            timestep_in_fs = self.ctx.lindiff_inputs.md.pw.parameters['CONTROL']['dt'] * 0.02418884254 * self.ctx.lindiff_inputs.md.pw.parameters['CONTROL']['iprint']
            forces_to_fit = inputs.fitting_parameters.get_dict()['forces_to_fit']
            stddev = inputs.fitting_parameters.get_dict()['stddev']

            pinballs = [s for s in self.ctx.current_structure.sites if s.kind_name == 'Li']
            nr_of_configurations = int(forces_to_fit/(len(pinballs)*3)+1+1)
            hustler_snapshots = rattle_randomly_structure(self.ctx.current_structure, orm.Dict(dict={'elements':'Li', 'stdev':stddev, 'nr_of_configurations':nr_of_configurations, 'timestep_in_fs':timestep_in_fs}))['rattled_snapshots']

            inputs.md['hustler_snapshots'] = hustler_snapshots
            
            # Set the `CALL` link label
            self.inputs.metadata.call_link_label = f'fitting_{self.ctx.diffusion_counter:02d}'
            inputs.metadata.label = f'fitting_{self.ctx.diffusion_counter:02d}'

            inputs = prepare_process_inputs(FittingWorkChain, inputs)
            running = self.submit(FittingWorkChain, **inputs)

            self.report(f'launching FittingWorkChain<{running.pk}>')
            
            return ToContext(workchains_fitting=append_(running))

        else: return

    def run_lindiff(self):
        """
        Runs a LinDiffusionWorkChain for an estimate of the diffusion.
        If there is a last fitting estimate, I update the parameters for the pinball.
        """
        ## At the end of each iteration there must be one extra fitting workchain
        ## So to ensure that workchains appended from input coefficients are in correct order
        ## I check their lengths
        if self.inputs.get('first_fit_with_random_rattling') and len(self.ctx.workchains_lindiff) == len(self.ctx.workchains_fitting):
            return

        inputs = self.ctx.lindiff_inputs
        inputs['parent_folder'] = self.inputs.parent_folder
        # I always use the original structure to start a new LinDiffusionWorkChain
        inputs['structure'] = self.ctx.current_structure

        if (self.ctx.diffusion_counter == 1):
            # If this is first run, then I launch an unmodified LinDiffusionWorkChain
            # Since this is a first run, I don't want to run for too long
            inputs.diffusion_parameters.update({'max_md_iterations': 1})
            if self.inputs.get('first_fit_with_random_rattling'):
                # I have yet to inspect the first fitting workchain, so I do it now
                try:
                    coefs = self.ctx.workchains_fitting[-1].outputs.coefficients
                    inputs.coefficients = coefs
                except (KeyError, exceptions.NotExistent):
                    self.report('the Fitting subworkchain failed to generate coefficients')
                    return self.exit_codes.ERROR_FITTING_FAILED

        else:
            # for every run after the first one, the pinball hyperparameters are taken from the output of the
            # last fitting workchain, which used the output trajectory of previous MD run to do fitting       
            # I need to use the input value for every run after first one
            inputs.diffusion_parameters.update({'max_md_iterations': self.ctx.max_lindiff_iterations})
            # Updating the pinball hyperparameters
            inputs.coefficients = self.ctx.workchains_fitting[-1].outputs.coefficients
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = f'lindiffusion_{self.ctx.diffusion_counter:02d}'
        inputs.metadata.label = f'lindiffusion_{self.ctx.diffusion_counter:02d}'

        inputs = prepare_process_inputs(LinDiffusionWorkChain, inputs)
        running = self.submit(LinDiffusionWorkChain, **inputs)

        self.report(f'launching LinDiffusionWorkChain<{running.pk}>')
        
        return ToContext(workchains_lindiff=append_(running))

    def run_fit(self):
        """
        Runs a fitting workflow on positions taken from the output of the previous lindiff run
        """
        inputs = self.ctx.fitting_inputs
        inputs['parent_folder'] = self.inputs.parent_folder
        # There's no difference between the first and subsequent runs so I don't change anything
        inputs['structure'] = self.ctx.current_structure
        # I have yet to inspect the previous lindiffusion workchain, so I do it now
        try:
            total_trajectory = self.ctx.workchains_lindiff[-1].outputs.total_trajectory
            inputs.md['hustler_snapshots'] = total_trajectory
        except (KeyError, exceptions.NotExistent):
            self.report('the LinearDiffusion subworkchain failed to generate a trajectory')
            return self.exit_codes.ERROR_LINDIFFUSION_FAILED
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = f'fitting_{self.ctx.diffusion_counter:02d}'
        inputs.metadata.label = f'fitting_{self.ctx.diffusion_counter:02d}'

        inputs = prepare_process_inputs(FittingWorkChain, inputs)
        running = self.submit(FittingWorkChain, **inputs)

        self.ctx.diffusion_counter +=1
        self.report(f'launching FittingWorkChain<{running.pk}>')
        
        return ToContext(workchains_fitting=append_(running))

    def inspect_process(self):
        """Inspect the results of the last `LinDiffusionWorkChain` and `FittingWorkChain`.

        I check if pinball hyperparameters converged with respect to the provided threshold, 
        both relative and absolute.
        """
        # I don't need to check much, since the daughter workchains take care of themselves
        try:
            coefs = self.ctx.workchains_fitting[-1].outputs.coefficients
        except (KeyError, exceptions.NotExistent):
            self.report('the Fitting subworkchain failed to generate coefficients')
            return self.exit_codes.ERROR_FITTING_FAILED

        param_d = self.ctx.diffusion_convergence_parameters_d
        # I check with same logic that was used in `setup`
        if self.ctx.diffusion_counter > param_d['min_ld_iterations']:

            coefficients = np.array([workchain_fitting.outputs.coefficients.get_dict()['coefs'] for workchain_fitting in self.ctx.workchains_fitting])
            stddev_3 = np.std(coefficients[-3:], axis=0)
            # checking the variation (relative error) in last 2 fits, if the standard deviation of last 3 fits is too high
            difference_2 = abs((coefficients[-1]-coefficients[-2])/coefficients[-1])
            # For only local interactions, 2nd coefficient is always 0
            if np.isnan(difference_2)[1]: difference_2[1] = 0

            if (stddev_3 < param_d['coefficient_threshold_std']).all() and (difference_2 < param_d['coefficient_threshold_diff']).all():
                # I have converged, yay me!
                self.report(f'Diffusion converged with std = {stddev_3} < threshold = {param_d["coefficient_threshold_std"]}')
                self.ctx.converged = True
            elif (difference_2 < param_d['coefficient_threshold_diff']).all():
                self.report(f'Last two estimates of Pinball parameters have converged with relative error = {difference_2} < threshold = {param_d["coefficient_threshold_diff"]}')
                self.ctx.converged = True
            elif (stddev_3 < param_d['coefficient_threshold_std']).any() and (difference_2 < param_d['coefficient_threshold_diff']).any():
                self.report(f'Not all Pinball parameters have converged with std = {stddev_3} and relative error = {difference_2}, so I start another MD iteration')
                self.ctx.converged = False
            else:
                self.report(f'The Pinball parameters have not converged with std = {stddev_3} and relative error = {difference_2}')
                self.ctx.converged = False

        return

    def run_last_lindiff(self):
        """
        Runs a final LinDiffusionWorkChain after converging Pinball parameters, starting from the previous trajectory.
        This is the MD run that to be used for all post processing.
        """
        if self.inputs.get('run_last_lindiffusion'):

            inputs = self.ctx.lindiff_inputs
            inputs['parent_folder'] = self.inputs.parent_folder

            # I use the last estimated Pinball parameters along with the last output trajectory
            last_traj = self.ctx.workchains_lindiff[-1].outputs.total_trajectory
            inputs['structure'] = self.ctx.current_structure
            # I increase the length of previous run by 50%
            inputs.md['nstep'] = orm.Int(2 * (last_traj.get_array('steps').size - 1) * self.ctx.lindiff_inputs.md.pw.parameters['CONTROL']['iprint'])
            inputs.msd_parameters['t_end_fit_fs_length'] *= 1.5
            inputs.msd_parameters['t_fit_fraction'] /= 1.5
            # No need to change other parameters, as they are still the same as the previous LinDiffusinWorkChain 
            
            # Updating the pinball hyperparameters
            inputs.coefficients = self.ctx.workchains_fitting[-2].outputs.coefficients
            # if following exist I use it to start the final MD run from 
            if self.ctx.last_lindiffusion_previous_trajectory:
                previous_traj = self.ctx.last_lindiffusion_previous_trajectory
                if last_traj.get_array('steps').size >= previous_traj.get_array('steps').size:
                    self.report(f'given trajectory <{previous_traj.id}> is shorter than the output trajectory <{last_traj.id}> of last LinDiffusionWorkChain <{self.ctx.workchains_lindiff[-1].id}>')
                    return self.exit_codes.ERROR_LAST_LINDIFFUSION_FAILED
                self.report(f'using the given trajectory <{previous_traj.id}> to extend the last LinDiffusionWorkChain')
                inputs.md.previous_trajectory = previous_traj
            else:
            # Otherwise I start from trajectory generated by the last lindiffusion workchain
                inputs.md.previous_trajectory = last_traj
            
            # Set the `CALL` link label
            self.inputs.metadata.call_link_label = f'lindiffusion_{self.ctx.diffusion_counter:02d}'
            inputs.metadata.label = f'lindiffusion_{self.ctx.diffusion_counter:02d}'

            inputs = prepare_process_inputs(LinDiffusionWorkChain, inputs)
            running = self.submit(LinDiffusionWorkChain, **inputs)

            self.report(f'launching LinDiffusionWorkChain<{running.pk}> to extend last MD run')
            
            return ToContext(workchains_lindiff=append_(running))
            
        else: return

    def results(self):
        """Retrun the output trajectory and diffusion coefficients generated in the last MD run."""
        if self.ctx.converged and self.ctx.diffusion_counter <= self.ctx.diffusion_convergence_parameters_d['max_ld_iterations']:
            self.report(f'workchain completed after {self.ctx.diffusion_counter} iterations')
        else:
            self.report('maximum number of LinDiffusion convergence iterations exceeded')
        try:
            self.out('msd_results', self.ctx.workchains_lindiff[-1].outputs.msd_results)
            self.out('total_trajectory', self.ctx.workchains_lindiff[-1].outputs.total_trajectory)
            self.out('coefficients', self.ctx.workchains_fitting[-1].outputs.coefficients)
        except (KeyError, exceptions.NotExistent):
            if self.inputs.get('run_last_lindiffusion'):
                self.report('the LinearDiffusion subworkchain that was going to extend the last MD run failed to generate msd results')
                self.out('msd_results', self.ctx.workchains_lindiff[-2].outputs.msd_results)
                self.out('total_trajectory', self.ctx.workchains_lindiff[-2].outputs.total_trajectory)
                self.out('coefficients', self.ctx.workchains_fitting[-1].outputs.coefficients)
                self.report('the last LinearDiffusion subworkchain failed to generate a trajectory')
                return self.exit_codes.ERROR_LAST_LINDIFFUSION_FAILED
