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

def get_slave_calculations(workchain):
    """
    Returns a list of the all the calculations that were called by the WF, ordered.
    """
    qb = orm.QueryBuilder()
    qb.append(orm.WorkChainNode, filters={'uuid': workchain.uuid}, tag='m')
    qb.append(orm.CalcJobNode, with_incoming='m',
              edge_project='label', edge_filters={'label': {'like': 'replay_%'}},
              tag='c', edge_tag='mc', project='*')
    calc_d = {item['mc']['label']: item['c']['*'] for item in qb.iterdict()}
    sorted_calcs = sorted(calc_d.items())
    return list(zip(*sorted_calcs))[1::2] if sorted_calcs else None

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
        spec.input('max_md_convergence_iterations', valid_type=orm.Int, default=lambda: orm.Int(5),
            help='The maximum number of MD calculations that will be called by this workchain.')
        spec.input('diffusion_parameters', valid_type=orm.Dict, required=True, help='The dictionary containing all the main parameters.')
        spec.inputs['diffusion_parameters'].default = lambda: orm.Dict(dict={
            'distance': 8,
            'element_to_remove': 'Li'})
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup,
            while_(cls.should_run_replay)(
                cls.run_replay,
                cls.inspect_replay,
            ),
            # cls.results,
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
   
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_flipper.workflows import protocols as lindiff_proto
        return files(lindiff_proto) / 'lindiff.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, protocol=None, overrides=None, **kwargs
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

        # Should implement a default protocol in the future 
        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (code, structure, protocol)
        replay = ReplayMDWorkChain.get_builder_from_protocol(*args, electronic_type=ElectronicType.INSULATOR, overrides=inputs['md'], **kwargs)

        replay['pw'].pop('structure', None)
        replay.pop('clean_workdir', None)
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

        This is the case as long as the last process has not finished successfully, the maximum number of restarts has
        not yet been exceeded, and the number of maximum replays has not been reached or diffusion coefficient converged.
        """
        return not(self.ctx.is_finished) and (
            not(self.ctx.converged) or (self.inputs.max_md_convergence_iterations.value <= self.ctx.replay_counter))

    def run_replay(self):
        """Run the `ReplayMDWorkChain` to run a `FlipperCalculation`."""

        self.ctx.replay_counter +=1
        inp_d = self.ctx.replay_inputs.pw.parameters
        print('/n', self.ctx.replay_inputs.pw.settings)

        if (self.ctx.replay_counter == 0):
            recenter = inp_d['moldyn_parameters'].get_attribute('recenter_before_main', False)
            inp_d['settings'] = self.ctx.replay_inputs.pw.settings
        else:
            recenter = False
            # if trajectory and structure have different shapes (i.e. for pinball level 1 calculation only pinball positions are printed:)
            create_missing = len(self.ctx.current_structure.sites
                                ) != last_calc.outputs.total_trajectory.get_attribute('array|positions')[1]
            # create missing tells inline function to append additional sites from the structure that needs to be passed in such case
            kwargs = dict(
                trajectory=last_calc.outputs.total_trajectory,
                parameters=get_or_create_input_node(orm.Dict,
                    dict(step_index=-1, recenter=recenter, create_settings=True, complete_missing=create_missing), store=True),)
            if create_missing:
                kwargs['structure'] = self.ctx.current_structure
            try:
                kwargs['settings'] = self.ctx.replay_inputs.pw.settings
            except:
                pass  # settings will be None

            res = get_structure_from_trajectory(**kwargs)
            inp_d['settings'] = res['settings']
            inp_d['structure'] = res['structure']
        
        inputs = self.ctx.replay_inputs
        inputs.pw.structure = self.ctx.current_structure

        # Set the `CALL` link label
        inputs.metadata.call_link_label = f'replay_{self.ctx.replay_counter:02d}'
        
        # For 2nd and onward calculations 
        if (self.ctx.replay_counter > 1): inputs.pw.parameters['IONS']['ion_velocities']='from_input'


        inputs = prepare_process_inputs(ReplayMDWorkChain, inp_d)
        running = self.submit(ReplayMDWorkChain, **inputs)

        self.report(f'launching ReplayMDWorkChain<{running.pk}>')
        
        return ToContext(workchains=append_(running))

    def inspect_replay(self):
        """Inspect the results of the last `PwBaseWorkChain`.

        Compare the cell volume of the relaxed structure of the last completed workchain with the previous. If the
        difference ratio is less than the volume convergence threshold we consider the cell relaxation converged.
        """
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

    @staticmethod
    def _fix_atomic_positions(structure, settings):
        """Fix the atomic positions, by setting the `FIXED_COORDS` key in the `settings` input node."""
        if settings is not None:
            settings = settings.get_dict()
        else:
            settings = {}

        settings['FIXED_COORDS'] = [[True, True, True]] * len(structure.sites)

        return settings
