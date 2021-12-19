# -*- coding: utf-8 -*-
"""Workchain to call MD workchains using Pinball pw.x. based on Quantum ESPRESSO"""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.engine import BaseRestartWorkChain, ToContext, if_, while_, append_
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_flipper.calculations.functions import get_diffusion_from_msd, get_structure_from_trajectory, concatenate_trajectory
from aiida_flipper.utils.utils import get_or_create_input_node

ReplayMDWorkChain = WorkflowFactory('quantumespresso.flipper.replaymd')

def get_trajectories_dict(pk_list):
    """
    Returns a dictionary of the output trajectories with the calling ReplayMDWorkChain's label as the key.
    """
    qb = orm.QueryBuilder()
    qb.append(ReplayMDWorkChain, filters={'id':{'in':pk_list}}, tag='replay', project='label')
    qb.append(orm.TrajectoryData, with_incoming='replay', project='*', tag='traj')
    return {'{}'.format(item['replay']['label']):item['traj']['*'] for item in qb.iterdict()}

class LinDiffusionWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """Workchain to run multiple MD calculations till the diffusion coefficient is 
    converged, using Pinball Quantum ESPRESSO pw.x."""
    _process_class = ReplayMDWorkChain
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(ReplayMDWorkChain, namespace='md',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={'help': 'Inputs for the `ReplayMDWorkChain` for MD runs are called in the `md` namespace.'})
        spec.input('structure', valid_type=orm.StructureData, help='The input supercell structure.')
        spec.input('parent_folder', valid_type=orm.RemoteData, required=True,
            help='The stashed directory containing charge densities of host lattice.')
        # spec.input('code', valid_type=orm.Code, help='The code used to run the calculations.')
        spec.input('diffusion_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing all the threshold values for diffusion convergence.')
        spec.input('msd_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing all the parameters required for MSD computation by Samos.')
        spec.input('coefficients', valid_type=orm.Dict, required=False, help='The dictionary containing the pinball hyperparameters generated after fitting.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.exit_code(701, 'ERROR_MAXIMUM_STEPS_NOT_REACHED',
            message='The calculation failed before reaching the maximum no. of MD steps.')
        spec.exit_code(702, 'ERROR_DIFFUSION_NOT_CONVERGED',
            message='The calculation reached the maximum no. of MD steps, but the diffusion coefficient stil did not converge.')
        spec.exit_code(703, 'ERROR_CHARGE_DENSITIES_NOT_FOUND',
            message='Either the stashed charge densities or the flipper compatible supercell structure not found.')
        spec.exit_code(704, 'ERROR_SUB_PROCESS_FAILED_MD',
            message='The ReplayMDWorkChain sub process failed.')
        spec.exit_code(705, 'ERROR_TRAJECTORY_NOT_FOUND',
            message='The output trajectory of ReplayMDWorkChain not found.')
        # spec.expose_outputs(ReplayMDWorkChain)
        spec.output('total_trajectory', valid_type=orm.TrajectoryData,
            help='The full concatenated trajectory of all called ReplayMDWorkChains.')
        spec.output('msd_results', valid_type=orm.ArrayData,
            help='The dictionary containing the results of msd calculations using the samos library.')

    def setup(self):
        """Input validation and context setup."""

        self.ctx.converged = False
        self.ctx.replay_counter = 0

        # I store the flipper/pinball compatible structure as current_structure
        self.ctx.current_structure = self.inputs.structure

        # I store all the input dictionaries in context variables
        self.ctx.replay_inputs = AttributeDict(self.exposed_inputs(ReplayMDWorkChain, namespace='md'))
        self.ctx.replay_inputs.pw.parameters = self.ctx.replay_inputs.pw.parameters.get_dict()
        self.ctx.replay_inputs.pw.settings = self.ctx.replay_inputs.pw.settings.get_dict()
        self.ctx.msd_parameters_d = self.inputs.msd_parameters.get_dict()
        self.ctx.diffusion_parameters_d = self.inputs.diffusion_parameters.get_dict()

        # MSD dict cannot contain items not recognised by SAMOS
        self.ctx.t_fit_fraction = self.ctx.msd_parameters_d.pop('t_fit_fraction')
        
        # I load the pinball hyper parameters here
        # change how its loaded depending on 3 coefficients or 4
        if self.inputs.get('coefficients'):
            coefs = self.inputs.coefficients.get_attribute('coefs')
            self.ctx.replay_inputs.pw.parameters['SYSTEM']['flipper_local_factor'] = coefs[0]
            if self.ctx.replay_inputs.pw.parameters['CONTROL']['flipper_do_nonloc']: 
                # no need to add the non local factor if it is disabled
                self.ctx.replay_inputs.pw.parameters['SYSTEM']['flipper_nonlocal_correction'] = coefs[1]
            self.ctx.replay_inputs.pw.parameters['SYSTEM']['flipper_ewald_rigid_factor'] = coefs[2]
            self.ctx.replay_inputs.pw.parameters['SYSTEM']['flipper_ewald_pinball_factor'] = coefs[3]
            self.report(f'launching WorkChain with pinball coefficients defined by <{self.inputs.coefficients.pk}>')
        else: self.report(f'launching WorkChain without any pinball hyperparameters')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_flipper.workflows import protocols as proto
        return files(proto) / 'lindiff.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, parent_folder, coefficients=None, protocol=None, overrides=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param parent_folder: the location of charge densities of host lattice
        :param coefficients: optional dictionary containing pinball hyperparameters, if not provided the pinball parameters are assumed to be 1
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol, usually takes the pseudo potential family.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.common.types import ElectronicType
        inputs = cls.get_protocol_inputs(protocol, overrides)

        # validating whether the charge density is correct, better I validate here before the workchain is submitted
        qb = orm.QueryBuilder()
        # querying the original unitcell
        qb.append(orm.StructureData, filters={'uuid':{'==':structure.extras['original_unitcell']}}, tag='struct')
        qb.append(WorkflowFactory('quantumespresso.flipper.preprocess'), with_incoming='struct', tag='prepro')
        qb.append(orm.RemoteData, with_incoming='prepro', project='id')
        parent_folders = qb.all(flat=True)
        if not parent_folder.pk in parent_folders: 
            print(f'the charge densities <{parent_folder.pk}> do not match with structure <{structure.pk}>')
            print('Proceed at your own risk')

        args = (code, structure, parent_folder, protocol)
        replay = ReplayMDWorkChain.get_builder_from_protocol(*args, electronic_type=ElectronicType.INSULATOR, overrides=inputs['md'], **kwargs)

        replay['pw'].pop('structure', None)
        replay.pop('clean_workdir', None)
        replay['pw'].pop('parent_folder', None)

        # For fireworks scheduler, setting up the required resources options
        if 'fw' in code.get_computer_label(): 
            replay['pw']['metadata']['options']['resources'].pop('num_machines')
            replay['pw']['metadata']['options']['resources']['tot_num_mpiprocs'] = 2

        builder = cls.get_builder()
        builder.md = replay

        builder.structure = structure
        builder.parent_folder = parent_folder
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.diffusion_parameters = orm.Dict(dict=inputs['diffusion_parameters'])
        builder.msd_parameters = orm.Dict(dict=inputs['msd_parameters'])
        if coefficients: builder.coefficients = coefficients

        # builder.msd_parameters['t_end_fit_fs'] = builder.md['nstep'].value * 0.7 * builder.md['pw']['parameters']['CONTROL']['dt'] * 4.8378 * 10**-2

        return builder

    def should_run_process(self):
        """Return whether an MD workchain should be run.

        This is the case as long as the last process has not finished successfully, and the number of maximum replays has not been reached or diffusion coefficient converged or minimum number of replays has not been reached.
        """
        if (self.ctx.diffusion_parameters_d['min_md_iterations'] > self.ctx.replay_counter): return True
        elif (not(self.ctx.converged) and (self.ctx.diffusion_parameters_d['max_md_iterations'] > self.ctx.replay_counter)): return True
        else: return False

    def run_process(self):
        """Run the `ReplayMDWorkChain` to launch a `FlipperCalculation`."""

        inputs = self.ctx.replay_inputs
        inputs.pw['parent_folder'] = self.inputs.parent_folder

        if (self.ctx.replay_counter == 0):
            # if this is first run, then I launch an unmodified ReplayMDWorkChain
            inputs.pw['structure'] = self.ctx.current_structure

        else:
            # for every run after the first one, the velocities and positions (which make the new structure) are read from the trajectory of previous MD run
            # if trajectory and structure have different shapes (i.e. for pinball level 1 calculation only pinball positions are printed:)

            # This input can only be used by the first MD run that I call.
            try: inputs.pop('previous_trajectory', None)
            except: pass

            workchain = self.ctx.workchains[-1]
            create_missing = len(self.ctx.current_structure.sites) != workchain.outputs.total_trajectory.get_attribute('array|positions')[1]
            # create missing tells inline function to append additional sites from the structure that needs to be passed in such case
            kwargs = dict(trajectory=workchain.outputs.total_trajectory, 
                        parameters=get_or_create_input_node(orm.Dict, dict(
                            step_index=-1,
                            recenter=False,
                            create_settings=True,
                            complete_missing=create_missing), store=False),)
            if create_missing:
                kwargs['structure'] = self.ctx.current_structure
                kwargs['settings'] = get_or_create_input_node(orm.Dict, self.ctx.replay_inputs.pw.settings, store=False)

            res = get_structure_from_trajectory(**kwargs)
            
            inputs.pw['structure'] = res['structure']
            self.ctx.current_structure = res['structure']
            inputs.pw['parameters']['IONS'].update({'ion_velocities': 'from_input'})
            inputs.pw['settings'] = res['settings'].get_dict()
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = f'replay_{self.ctx.replay_counter:02d}'
        inputs.metadata.label = f'replay_{self.ctx.replay_counter:02d}'

        inputs = prepare_process_inputs(ReplayMDWorkChain, inputs)
        running = self.submit(ReplayMDWorkChain, **inputs)

        self.ctx.replay_counter +=1
        self.report(f'launching ReplayMDWorkChain<{running.pk}>')
        
        return ToContext(workchains=append_(running))

    def inspect_process(self):
        """Inspect the results of the last `ReplayMDWorkChain`.

        I compute the MSD from the previous trajectory and check if it converged with respect to the provided threshold, both relative and absolute.
        """
        workchain = self.ctx.workchains[-1]

        # Maybe add some acceptable failed status in future?
        # acceptable_statuses = ['ERROR_IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF']

        try:
            trajectory = workchain.outputs.total_trajectory
            # setting up the fitting window 
            self.ctx.msd_parameters_d['t_end_fit_fs'] = round(trajectory.attributes['sim_time_fs'] * self.ctx.t_fit_fraction)
        except (KeyError, exceptions.NotExistent):
            self.report('the Md run with ReplayMDWorkChain did not generate output trajectory')
            
            if workchain.is_excepted or workchain.is_killed:
                self.report('called ReplayMDWorkChain was excepted or killed')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MD
            
            if workchain.is_failed: # and workchain.exit_status not in ReplayMDWorkChain.get_exit_statuses(acceptable_statuses):
                self.report(f'called ReplayMDWorkChain failed with exit status {workchain.exit_status}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MD

            return self.exit_codes.ERROR_TRAJECTORY_NOT_FOUND

        # Calculate MSD and check if it converged
        pk_list = []
        for calc in self.ctx.workchains: pk_list.append(calc.pk)

        if len(pk_list) > 1:
            concat_input_d = get_trajectories_dict(pk_list)
            concat_input_d.update({'remove_repeated_last_step': True})
            concatenated_trajectory = concatenate_trajectory(**concat_input_d)['concatenated_trajectory']
        elif len(pk_list) == 1:
            concatenated_trajectory = trajectory
        else:
            return self.exit_codes.ERROR_TRAJECTORY_NOT_FOUND
        
        msd_results = get_diffusion_from_msd(
                structure=self.ctx.current_structure,
                parameters=get_or_create_input_node(orm.Dict, self.ctx.msd_parameters_d, store=False),
                trajectory=concatenated_trajectory)['msd_results']
        sem = msd_results.attributes['{}'.format(self.ctx.msd_parameters_d['species_of_interest'][0])]['diffusion_sem_cm2_s']
        mean_d = msd_results.attributes['{}'.format(self.ctx.msd_parameters_d['species_of_interest'][0])]['diffusion_mean_cm2_s']
        sem_relative = sem / mean_d
        sem_target = self.ctx.diffusion_parameters_d['sem_threshold']
        sem_relative_target = self.ctx.diffusion_parameters_d['sem_relative_threshold']

        self.report(f'after iteration {self.ctx.replay_counter} mean msd is {mean_d}')

        if (mean_d < 0.):
            # the diffusion is negative: means that the value is not converged enough yet
            self.report(f'The Diffusion coefficient ( {mean_d} +/- {sem} ) is negative, i.e. not converged.')
            self.ctx.converged = False
        elif (sem < sem_target):
            # This means that the  standard error of the mean in my diffusion coefficient is below the target accuracy
            self.report(f'The error ( {sem} ) is below the target value ( {sem_target} ).') 
            self.ctx.converged = True
        elif (sem_relative < sem_relative_target):
            # the relative error is below my targe value
            self.report(f'The relative error ( {sem_relative} ) is below the target value ( {sem_relative_target} ).')
            self.ctx.converged = True
        else:
            self.report('The error has not converged')
            self.report('absolute sem: {:.5e}  Target: {:.5e}'.format(sem, sem_target))
            self.report('relative sem: {:.5e}  Target: {:.5e}'.format(sem_relative, sem_relative_target))
            self.ctx.converged = False

        #storing following as context variables to be later used in results, without recalculation
        self.ctx.msd_results = msd_results
        self.ctx.concatenated_trajectory = concatenated_trajectory
        
        return

    def results(self):
        """Attach the output parameters and combined trajectories of all called ReplayMDWorkChains to the outputs."""
        if self.ctx.converged and self.ctx.replay_counter <= self.ctx.diffusion_parameters_d['max_md_iterations']:
            self.report(f'workchain completed after {self.ctx.replay_counter} iterations')
        else:
            self.report('maximum number of MD convergence iterations exceeded')

        self.out('msd_results', self.ctx.msd_results)
        self.out('total_trajectory', self.ctx.concatenated_trajectory)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

