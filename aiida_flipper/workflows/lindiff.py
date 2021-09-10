# -*- coding: utf-8 -*-
"""Workchain to call MD workchains using Quantum ESPRESSO pw.x."""
from aiida_flipper.workflows.replaymd import ReplayMDWorkChain
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.engine import BaseRestartWorkChain, WorkChain, ToContext, if_, while_, append_
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_flipper.calculations.functions import get_diffusion_from_msd, get_structure_from_trajectory, concatenate_trajectory, update_parameters_with_coefficients
from aiida_flipper.utils.utils import get_or_create_input_node

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

def get_trajectories_dict(calc_list):
    """
    Returns a dictionary of the output trajectories with the calling ReplayMDWorkChain's label as the key.
    """
    qb = orm.QueryBuilder()
    qb.append(ReplayMDWorkChain, filters={'id':{'in':calc_list}}, tag='replay', project='label')
    qb.append(orm.TrajectoryData, with_incoming='replay', project='*', tag='traj')
    return {'{}'.format(item['replay']['label']):item['traj']['*'] for item in qb.iterdict()}

class LinDiffusionWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """Workchain to run multiple MD calculations till the diffusion coefficient is 
    converged, using Quantum ESPRESSO pw.x."""
    _process_class = ReplayMDWorkChain
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(ReplayMDWorkChain, namespace='md',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={'help': 'Inputs for the `ReplayMDWorkChain` for MD runs are called in the `md` namespace.'})
        spec.input('structure', valid_type=orm.StructureData, help='The input unitcell structure and not the supercell.')
        # spec.input('code', valid_type=orm.Code, help='The code used to run the calculations.')
        spec.input('max_md_convergence_iterations', valid_type=orm.Int, default=lambda: orm.Int(6),
            help='The maximum number of MD runs that will be called by this workchain.')
        spec.input('min_md_convergence_iterations', valid_type=orm.Int, default=lambda: orm.Int(3),
            help='The minimum number of MD runs that will be called by this workchain even if diffusion coefficient has converged.')
        spec.input('diffusion_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing all the threshold values for diffusion convergence.')
        spec.inputs['diffusion_parameters'].default = lambda: orm.Dict(dict={
            'sem_threshold': 1.e-5,
            'sem_relative_threshold': 1.e-2})
        spec.input('msd_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing all the parameters required for MSD computation by Samos.')
        spec.inputs['msd_parameters'].default = lambda: orm.Dict(dict={
            'equilibration_time_fs': 1.e6,
            'species_of_interest': 'Li',
            'stepsize_t' : 1,
            'stepsize_tau' : 1,
            'nr_of_blocks' : 1,
            't_start_fit_dt' : 10,
            't_end_fit_dt' : 1.e9,
            't_long_factor' : 1,
            'do_com' : False,
            'decomposed' : False,
            'verbosity' : 0})
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup,
            while_(cls.should_run_replay)(
                cls.run_replay,
                cls.inspect_replay,
            ),
            cls.results,
        )
        spec.exit_code(701, 'ERROR_MAXIMUM_STEPS_NOT_REACHED',
            message='The calculation failed before reaching the maximum no. of MD steps.')
        spec.exit_code(702, 'ERROR_DIFFUSION_NOT_CONVERGED',
            message='The calculation reached the maximum no. of MD steps, but the diffusion coefficient stil did not converge.')
        spec.exit_code(703, 'ERROR_CHARGE_DENSITIES_NOT_FOUND',
            message='Either the stashed charge densities or the flipper compatible supercell structure not found.') 
        # spec.expose_outputs(ReplayMDWorkChain)
        spec.output('output_parameters', valid_type=orm.Dict,
            help='The `output_parameters` output node of the successful calculation.')

    def setup(self):
        """Input validation and context setup."""

        self.ctx.converged = False
        self.ctx.replay_counter = 0
        self.ctx.is_finished = False

        # Querying the charge densities and flipper compatible structure
        qb = orm.QueryBuilder()
        qb.append(orm.StructureData, filters={'id':{'==':self.inputs.structure.pk}}, tag='struct')
        qb.append(WorkflowFactory('quantumespresso.flipper.preprocess'), with_incoming='struct', tag='prepro')
        try:
            qb.append(orm.RemoteData, with_incoming='prepro')
            self.ctx.stashed_folder, = qb.first()
            qb.append(orm.StructureData, with_incoming='prepro')
            self.ctx.current_structure, = qb.first()
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_CHARGE_DENSITIES_NOT_FOUND

        # I can add the builder options here in ctx form, if need be

        self.ctx.replay_inputs = AttributeDict(self.exposed_inputs(ReplayMDWorkChain, namespace='md'))
        self.ctx.replay_inputs.pw.parameters = self.ctx.replay_inputs.pw.parameters.get_dict()
        self.ctx.replay_inputs.pw['parent_folder'] = self.ctx.stashed_folder
        self.ctx.replay_inputs.pw.setdefault('settings', {})
        # to store the pk of all ReplayMDWorkChains called by this workchain, to help in querying of all TrajectoryData
        self.ctx.last_calc_list = [] 
   
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_flipper.workflows import protocols as proto
        return files(proto) / 'lindiff.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, nstep=None, protocol=None, overrides=None, **kwargs
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
        from aiida_quantumespresso.common.types import ElectronicType
        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (code, structure, protocol)
        replay = ReplayMDWorkChain.get_builder_from_protocol(*args, electronic_type=ElectronicType.INSULATOR, overrides=inputs['md'], **kwargs)

        replay['pw'].pop('structure', None)
        replay.pop('clean_workdir', None)
        if nstep: replay['nstep'] = nstep
        # setting 0 k-points for gamma option
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh([1,1,1])
        replay.kpoints = kpoints

        builder = cls.get_builder()
        builder.md = replay

        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.max_md_convergence_iterations = orm.Int(inputs['max_md_convergence_iterations'])

        return builder

    def should_run_replay(self):
        """Return whether a relaxation workchain should be run.

        This is the case as long as the last process has not finished successfully, and the number of maximum replays has not been reached or diffusion coefficient converged or minimum number of replays has not been reached.
        """
        if (self.inputs.min_md_convergence_iterations >= self.ctx.replay_counter): return True
        elif (not(self.ctx.converged) or (self.inputs.max_md_convergence_iterations <= self.ctx.replay_counter)): return True
        else: return False

    def run_replay(self):
        """Run the `ReplayMDWorkChain` to run a `FlipperCalculation`."""

        #Maybe get_builder_from_protocol() is not callable here??

        if (self.ctx.replay_counter == 0):
            # if this is first run, then we do a simple thermalising run which is essentially launching the same ReplayMDWorkChain
            builder = LinDiffusionWorkChain.get_builder_from_protocol(code=self.inputs.code, structure=self.ctx.current_structure, nstep=100000)

        else:
            # for every run after thermalisation, the velocities are read from the trajectory of previous MD run
            # if trajectory and structure have different shapes (i.e. for pinball level 1 calculation only pinball positions are printed:)
            create_missing = len(self.ctx.current_structure.sites) != self.ctx.last_calc.outputs.total_trajectory.get_attribute('array|positions')[1]
            # create missing tells inline function to append additional sites from the structure that needs to be passed in such case
            kwargs = dict(trajectory=self.ctx.last_calc.outputs.total_trajectory, 
                        parameters=get_or_create_input_node(orm.Dict, dict(
                            step_index=-1,
                            recenter=False,
                            create_settings=True,
                            complete_missing=create_missing), store=True),)
            if create_missing:
                kwargs['structure'] = self.ctx.current_structure
                kwargs['settings'] = get_or_create_input_node(orm.Dict, dict(gamma_only=True), store=True)


            res = get_structure_from_trajectory(**kwargs)
            
            builder = LinDiffusionWorkChain.get_builder_from_protocol(code=self.inputs.code, structure=res['structure'], stashed_folder=self.ctx.stashed_folder, nstep=1000000)
            builder.pw['parameters']['IONS'].update({'ion_velocities': 'from_input'})
            builder.pw['settings'] = res['settings']
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = f'replay_{self.ctx.replay_counter:02d}'
        builder['md']['pw']['parent_folder'] = self.ctx.stashed_folder
        builder['metadata']['label']=f'replay_{self.ctx.replay_counter}'
        running = self.submit(builder)
        self.ctx.last_calc = running
        self.ctx.last_calc_list.append(running.pk)
        self.ctx.replay_counter +=1
        self.report(f'launching ReplayMDWorkChain<{running.pk}>')
        
        return ToContext(workchains=append_(running))

    def inspect_replay(self):
        """Inspect the results of the last `PwBaseWorkChain`.

        Compare the cell volume of the relaxed structure of the last completed workchain with the previous. If the
        difference ratio is less than the volume convergence threshold we consider the cell relaxation converged.
        """
        concatenated_trajectory = concatenate_trajectory(**get_trajectories_dict(self.ctx.last_calc_list), remove_repeated_last_step=True)['concatenated_trajectory']

        
        workchain = self.ctx.workchains[-1]

        acceptable_statuses = ['ERROR_IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF']

        if workchain.is_excepted or workchain.is_killed:
            self.report('relax PwBaseWorkChain was excepted or killed')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        if workchain.is_failed and workchain.exit_status not in ReplayMDWorkChain.get_exit_statuses(acceptable_statuses):
            self.report(f'relax PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        try:
            structure = workchain.outputs.output_structure
        except exceptions.NotExistent:
            # If the calculation is set to 'scf', this is expected, so we are done
            if self.ctx.relax_inputs.pw.parameters['CONTROL']['calculation'] == 'scf':
                self.ctx.is_converged = True
                return

            self.report('`vc-relax` or `relax` PwBaseWorkChain finished successfully but without output structure')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        prev_cell_volume = self.ctx.current_cell_volume
        curr_cell_volume = structure.get_cell_volume()

        # Set relaxed structure as input structure for next iteration
        self.ctx.current_structure = structure
        self.ctx.current_number_of_bands = workchain.outputs.output_parameters.get_dict()['number_of_bands']
        self.report(f'after iteration {self.ctx.replay_counter} cell volume of relaxed structure is {curr_cell_volume}')

        # After first iteration, simply set the cell volume and restart the next base workchain
        if not prev_cell_volume:
            self.ctx.current_cell_volume = curr_cell_volume

            # If meta convergence is switched off we are done
            if not self.ctx.meta_convergence:
                self.ctx.is_converged = True
            return

        # !! somewhere here check if msd converged
        self.ctx.is_finished = True

        concat_input_d = get_trajectories_dict(last_calc_list)
        concat_input_d.update({'remove_repeated_last_step': True})
        concatenated_trajectory = concatenate_trajectory(**concat_input_d)['concatenated_trajectory']

        #store following as context variables to be later used in results

        msd_results = get_diffusion_from_msd(
                structure=pinball_struct,
                parameters=get_or_create_input_node(orm.Dict, msd_parameters, store=False),
                trajectory=concatenated_trajectory)
        sem = msd_results.attributes['{}'.format(msd_parameters['species_of_interest'][0])]['diffusion_sem_cm2_s']
        mean_d = msd_results.attributes['{}'.format(msd_parameters['species_of_interest'][0])]['diffusion_mean_cm2_s']
        sem_relative = sem / mean_d
        sem_target = diffusion_parameters['sem_threshold']
        sem_relative_target = diffusion_parameters['sem_relative_threshold']

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


        # Check whether the cell volume is converged
        volume_threshold = self.inputs.diffusion_parameters.value
        volume_difference = abs(prev_cell_volume - curr_cell_volume) / prev_cell_volume

        if volume_difference < volume_threshold:
            self.ctx.is_converged = True
            self.report(
                'relative cell volume difference {} smaller than convergence threshold {}'.format(
                    volume_difference, volume_threshold
                )
            )
        else:
            self.report(
                'current relative cell volume difference {} larger than convergence threshold {}'.format(
                    volume_difference, volume_threshold
                )
            )

        self.ctx.current_cell_volume = curr_cell_volume

        return

    def results(self):
        """Attach the output parameters and structure of the last workchain to the outputs."""
        if self.ctx.is_converged and self.ctx.replay_counter <= self.inputs.max_md_convergence_iterations.value:
            self.report(f'workchain completed after {self.ctx.replay_counter} iterations')
        else:
            self.report('maximum number of meta convergence iterations exceeded')

        # Get the latest relax workchain and pass the outputs
        final_relax_workchain = self.ctx.workchains[-1]

        if self.ctx.relax_inputs.pw.parameters['CONTROL']['calculation'] != 'scf':
            self.out('output_structure', final_relax_workchain.outputs.output_structure)

        try:
            self.out_many(self.exposed_outputs(self.ctx.workchain_scf, ReplayMDWorkChain))
        except AttributeError:
            self.out_many(self.exposed_outputs(final_relax_workchain, ReplayMDWorkChain))

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

