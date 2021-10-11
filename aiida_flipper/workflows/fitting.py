# -*- coding: utf-8 -*-
"""Workchain to generate pinball hyperparameters"""
from aiida.engine import calcfunction
from aiida.engine.processes import workchains
from aiida_quantumespresso.utils.defaults import calculation
from samos.trajectory import Trajectory
from aiida import orm
from aiida.common import AttributeDict, exceptions  
from aiida.engine import BaseRestartWorkChain, WorkChain, ToContext, if_, while_, append_
from aiida.plugins import CalculationFactory, WorkflowFactory
import numpy as np

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_flipper.calculations.functions import update_parameters_with_coefficients, get_pinball_factors
from aiida_flipper.utils.utils import get_or_create_input_node

ReplayMDHWorkChain = WorkflowFactory('quantumespresso.flipper.replaymdhustler')

class FittingWorkChain(ProtocolMixin, WorkChain):
    """Workchain to run hustler level `pinball` and `DFT` calculations to fit forces and
    generate pinball hyperparameters, using Pinball Quantum ESPRESSO pw.x."""
    _process_class = ReplayMDHWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(ReplayMDHWorkChain, namespace='md',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={'help': 'Inputs for the `ReplayMDWorkChain` for MD runs are called in the `md` namespace.'})
        spec.input('structure', valid_type=orm.StructureData, help='The input unitcell structure and not the supercell.')
        spec.input('parent_folder', valid_type=orm.RemoteData, required=True,
            help='The stashed directory containing charge densities of host lattice.')
        spec.input('fitting_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing the fitting parameters.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup,
            cls.run_process_pb,
            cls.run_process_dft,
            cls.inspect_process,
            cls.results,
        )

        spec.exit_code(702, 'ERROR_FITTING_FAILED',
            message='The linear regression to fit pinball and dft forces failed.')
        spec.exit_code(703, 'ERROR_CHARGE_DENSITIES_NOT_FOUND',
            message='Either the stashed charge densities or the flipper compatible supercell structure not found.')
        spec.exit_code(704, 'ERROR_SUB_PROCESS_FAILED_MD',
            message='The ReplayMDHustlerWorkChain sub process failed.')
        spec.exit_code(705, 'ERROR_TRAJECTORY_NOT_FOUND',
            message='The output trajectory of ReplayMDWorkChain not found.')
        # spec.expose_outputs(ReplayMDWorkChain)
        spec.output('coefficients', valid_type=orm.Dict,
            help='The dictionary containing the newly fitted pinball hyperparameters(keyword - `coefs`) along with linear regression values.')
        spec.output('trajectory_pb', valid_type=orm.TrajectoryData,
            help='The output trajectory of pinball Hustler calculation for easy manual fitting/post-processing if needed.')
        spec.output('trajectory_dft', valid_type=orm.TrajectoryData,
            help='The output trajectory of DFT Hustler calculation for easy manual fitting/post-processing if needed.')

    def setup(self):
        """Input validation and context setup."""

        # I store the flipper/pinball compatible structure as current_structure
        qb = orm.QueryBuilder()
        qb.append(orm.StructureData, filters={'id':{'==':self.inputs.structure.pk}}, tag='struct')
        qb.append(WorkflowFactory('quantumespresso.flipper.preprocess'), with_incoming='struct', tag='prepro')
        # no need to check if supercell structure exists, already checked by builder
        qb.append(orm.StructureData, with_incoming='prepro')
        if qb.count() > 1: self.report('Multiple charge densities found for structure <{}>; using the last one to start MD runs'.format(self.inputs.structure.pk))
        self.ctx.current_structure = qb.all(flat=True)[-1]

        # I store all the input dictionaries in context variables
        self.ctx.replay_inputs = AttributeDict(self.exposed_inputs(ReplayMDHWorkChain, namespace='md'))
        self.ctx.replay_inputs.pw.parameters = self.ctx.replay_inputs.pw.parameters.get_dict()
        self.ctx.replay_inputs.pw.settings = self.ctx.replay_inputs.pw.settings.get_dict()

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_flipper.workflows import protocols as proto
        return files(proto) / 'fitting.yaml'

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
        inputs = cls.get_protocol_inputs(protocol, overrides)

        # I query the charge densities and supercell
        qb = orm.QueryBuilder()
        qb.append(orm.StructureData, filters={'id':{'==':structure.pk}}, tag='struct')
        qb.append(WorkflowFactory('quantumespresso.flipper.preprocess'), with_incoming='struct', tag='prepro')
        qb.append(orm.RemoteData, with_incoming='prepro')
        if qb.count(): stashed_folder = qb.all(flat=True)[-1]
        else: raise RuntimeError(f'charge densities and/or flipper compatible supercell not found for {structure.pk}')
        qb.append(orm.StructureData, with_incoming='prepro')
        if qb.count(): current_structure = qb.all(flat=True)[-1]
        else: raise RuntimeError(f'charge densities and/or flipper compatible supercell not found for {structure.pk}')
        # I query the trajectory from which I extract snapshots for force fitting
        qb = orm.QueryBuilder()
        qb.append(orm.StructureData, filters={'id':{'==':structure.pk}}, tag='struct')
        qb.append(WorkflowFactory('quantumespresso.flipper.lindiffusion'), with_incoming='struct', tag='lindiff', filters={'and':[{'attributes.process_state':{'==':'finished'}}, {'attributes.exit_status':{'==':0}}]})
        qb.append(orm.TrajectoryData, with_incoming='lindiff')
        if qb.count(): out_traj = qb.all(flat=True)[-1]
        else: raise RuntimeError(f'output trajectories not found for {structure.pk}, please run LinDiffusionWorkChain before force fitting')

        args = (code, structure, stashed_folder, out_traj, protocol)
        replay = ReplayMDHWorkChain.get_builder_from_protocol(*args, electronic_type=ElectronicType.INSULATOR, overrides=inputs['md'], **kwargs)

        replay['pw'].pop('structure', None)
        replay.pop('clean_workdir', None)
        replay['pw'].pop('parent_folder', None)

        builder = cls.get_builder()
        builder.md = replay

        builder.structure = structure
        builder.parent_folder = stashed_folder
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.fitting_parameters = orm.Dict(dict=inputs['fitting_parameters'])

        return builder

    def run_process_pb(self):
        """Run the `ReplayMDHustlerWorkChain` to launch a `HustlerCalculation`."""

        inputs = self.ctx.replay_inputs
        inputs.pw['parent_folder'] = self.inputs.parent_folder
        inputs.pw['structure'] = self.ctx.current_structure
        inputs.pw['parameters']['CONTROL']['lflipper'] = True
        inputs.pw['parameters']['CONTROL']['ldecompose_forces'] = True
        inputs.pw['parameters']['CONTROL']['ldecompose_ewald'] = True
        inputs.pw['parameters']['CONTROL']['flipper_do_nonloc'] = True
                    
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = 'replayh_pb'
        inputs.metadata.label = 'replayh_pb'

        inputs = prepare_process_inputs(ReplayMDHWorkChain, inputs)
        running = self.submit(ReplayMDHWorkChain, **inputs)

        self.report(f'launching ReplayMDHustlerWorkChain<{running.pk}>')
        
        return ToContext(workchains=append_(running))

    def run_process_dft(self):
        """Run the `ReplayMDHustlerWorkChain` to launch a `HustlerCalculation`."""

        inputs = self.ctx.replay_inputs
        inputs.pw['parent_folder'] = self.inputs.parent_folder
        inputs.pw['structure'] = self.ctx.current_structure
        inputs.pw['parameters']['CONTROL'].pop('lflipper')
        inputs.pw['parameters']['CONTROL'].pop('ldecompose_forces')
        inputs.pw['parameters']['CONTROL'].pop('ldecompose_ewald')
        inputs.pw['parameters']['CONTROL'].pop('flipper_do_nonloc')

        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = 'replayh_dft'
        inputs.metadata.label = 'replayh_dft'

        inputs = prepare_process_inputs(ReplayMDHWorkChain, inputs)
        running = self.submit(ReplayMDHWorkChain, **inputs)

        self.report(f'launching ReplayMDHustlerWorkChain<{running.pk}>')
        
        return ToContext(workchains=append_(running))

    def inspect_process(self):
        """Inspect the results of the last `ReplayMDHustlerWorkChain`.

        I compute the MSD from the previous trajectory and check if it converged with respect to the provided threshold, both relative and absolute.
        """
        workchain_pb = self.ctx.workchains[0]
        workchain_dft = self.ctx.workchains[1]

        for workchain in [workchain_pb, workchain_dft]:
            if workchain.is_excepted or workchain.is_killed:
                self.report('called ReplayMDHustlerWorkChain was excepted or killed')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MD

            if workchain.is_failed: # and workchain.exit_status not in ReplayMDHustlerWorkChain.get_exit_statuses(acceptable_statuses):
                self.report(f'called ReplayMDHustlerWorkChain failed with exit status {workchain.exit_status}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MD

            try:
                trajectory = workchain.outputs.total_trajectory
            except Exception:
                self.report('the Md run with ReplayMDHustlerWorkChain finished successfully but without output trajectory')
                return self.exit_codes.ERROR_TRAJECTORY_NOT_FOUND

        # Start fitting and compute pinball hyperparameters
        trajectory_pb = workchain_pb.outputs.total_trajectory
        trajectory_dft = workchain_dft.outputs.total_trajectory
        nstep = self.ctx.replay_inputs.nstep.value

        for traj in (trajectory_pb, trajectory_dft):
            shape = traj.get_positions().shape
            if shape[0] != nstep:
                self.report('Wrong shape of array returned by {} ({} vs {})'.format(traj.pk, shape, nstep))
                self.exit_codes.ERROR_FITTING_FAILED

        self.ctx.coefficients = get_pinball_factors(trajectory_dft, trajectory_pb)['coefficients']
        self.ctx.trajectory_pb = trajectory_pb
        self.ctx.trajectory_dft = trajectory_dft
        
        return

    def results(self):
        """Output the pinball hyperparameter and results of the fit along with the trajectories."""
        self.out('coefficients', self.ctx.coefficients)
        self.out('trajectory_pb', self.ctx.trajectory_pb)
        self.out('trajectory_dft', self.ctx.trajectory_dft)
