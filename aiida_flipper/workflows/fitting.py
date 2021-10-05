# -*- coding: utf-8 -*-
"""Workchain to generate pinball hyperparameters"""
from aiida.engine import calcfunction
from aiida.engine.processes import workchains
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


class FittingWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """Workchain to run hustler level `pinball` and `DFT` calculations to fit forces and
    generate pinball hyperparameters, using Pinball Quantum ESPRESSO pw.x."""
    _process_class = ReplayMDHWorkChain


    def positions(self):
        return self.inp.positions

    def start(self):
        """
        Method to start (i.e. how to create the positions) 
        has to be defined by the subclass
        """
        self.inp.structure
        self.inp.remote_folder_flipper
        # self.inp.electron_parameters
        self.inp.parameters
        # self.inp.flipper_code
        # self.inp.pseudo_Li
        # an array that I will calculate the forces on!
        self.ctx.nstep = self.inp.positions.get_array('positions').shape[0]

        self.inp.parameters
        self.goto(self.launch_replays)

    def launch_calculations(self):
        #~ rattled_positions = self.out.rattled_positions
        #################################
        # raise DeprecationWarning()
        #################################
        rattled_positions = self.start()['rattled_positions']
        nstep = self.ctx.nstep
        remote_folder = self.inp.remote_folder
        try:
            chargecalc, = remote_folder.get_inputs(node_type=CalculationFactory('quantumespresso.pw'))
        except Exception as e:
            # This must have been a copied remote folder
            chargecalc = remote_folder.inp.copied_remote_folder.inp.remote_folder.inp.remote_folder

        structure = self.inp.structure
        pseudofamily = self.inp.parameters.dict.pseudofamily
        pseudos=pseudofamily.get_pseudos(structure=structure)
        ecutwfc, ecutrho = pseudofamily.get_recommended_cutoffs(structure=structure, unit='Ry')

        flipper_calc = self.inp.flipper_code.new_calc()
        flipper_calc._set_parent_remotedata(remote_folder)
        flipper_calc.use_structure(structure)
    
        flipper_calc.use_array(rattled_positions)
        flipper_calc.use_kpoints(chargecalc.inp.kpoints)
        flipper_calc.use_settings(chargecalc.inp.settings)
        parameters_input_flipper = chargecalc.inp.parameters.get_dict()
        for card, key in (
                ('SYSTEM', 'tot_charge'),
                ('CONTROL', 'max_seconds'),
                ('ELECTRONS', 'conv_thr'),
                ('ELECTRONS', 'electron_maxstep'),
                ('ELECTRONS', 'mixing_beta'),
                ('ELECTRONS', 'diagonalization')
            ):
            try:
                del parameters_input_flipper[card][key]
            except KeyError:
                pass
        parameters_input_flipper['CONTROL']['lhustle'] = True
        parameters_input_flipper['CONTROL']['verbosity'] = 'low'
        parameters_input_flipper['CONTROL']['lflipper'] = True
        parameters_input_flipper['CONTROL']['calculation'] = 'md'
        parameters_input_flipper['CONTROL']['ldecompose_ewald'] = True
        parameters_input_flipper['CONTROL']['nstep'] = nstep
        parameters_input_flipper['IONS'] = {}

        flipper_calc.use_parameters(get_or_create_input_node(orm.Dict, parameters_input_flipper, store=False))
        flipper_resources = {'num_machines': chargecalc.get_resources()['num_machines']}
        if chargecalc.get_resources().get('num_mpiprocs_per_machine', 0):
            flipper_resources['num_mpiprocs_per_machine'] = chargecalc.get_resources().get('num_mpiprocs_per_machine')
        flipper_calc.set_resources(flipper_resources)
        flipper_calc.set_max_wallclock_seconds(self.inp.parameters.dict.flipper_walltime_seconds)
        flipper_calc._set_attr('is_flipper', True)
        flipper_calc._set_attr('is_hustler', True)
        flipper_calc.label = '%s-hustler-flipper'  % structure.label

        if self.inp.parameters.dict.use_same_code:
            dft_calc = self.inp.flipper_code.new_calc()
        else:
            dft_calc = self.inp.dft_code.new_calc()

        dft_calc.use_structure(structure)
    
        dft_calc.use_array(rattled_positions)
        # I can use different kpoints and settings than charge calc, relay from input
        dft_calc.use_kpoints(self.inp.kpoints)
        dft_calc.use_settings(self.inp.settings)

        parameters_input_dft = {
                                    u'CONTROL': {
                                        u'calculation': 'md',
                                        u'restart_mode': 'from_scratch',
                                        u'dt':40,
                                        u'iprint':1,
                                        u'verbosity':'low',
                                        u'ldecompose_forces':True,
                                        u'lhustle':True,
                                    },
                                    u'SYSTEM': {
                                        u'nosym': True,
                                    },
                                    u'IONS':{}
                                }
        
        parameters_input_dft['SYSTEM']['ecutwfc'] = ecutwfc
        parameters_input_dft['SYSTEM']['ecutrho'] = ecutrho
        parameters_input_dft['CONTROL']['nstep'] = nstep
        parameters_input_dft['ELECTRONS'] = self.inp.electron_parameters.get_dict()

        dft_calc.use_parameters(get_or_create_input_node(orm.Dict, parameters_input_dft, store=False))
        dft_resources = {"num_machines": self.inp.parameters.dict.dft_num_machines}
        if self.inp.parameters.get_attr("dft_num_mpiprocs_per_machine", 0):
            dft_resources["num_mpiprocs_per_machine"] = self.inp.parameters.get_attr("dft_num_mpiprocs_per_machine")
        dft_calc.set_resources(dft_resources)
        dft_calc.set_max_wallclock_seconds(self.inp.parameters.dict.dft_walltime_seconds)
        for k,v in pseudos.iteritems():
            dft_calc.use_pseudo(v, k)
        # overwriting pseudo for lithium calculation

        pseudos['Li'] = self.inp.li_pseudo

        for k,v in pseudos.iteritems():
            flipper_calc.use_pseudo(v,k)

        dft_calc._set_attr('is_hustler', True)
        dft_calc.label = '%s-hustler-dft'  % structure.label
        self.goto(self.fit)
        return {'hustler_flipper':flipper_calc, 'hustler_dft':dft_calc}


    def launch_replays(self):
        positions = self.positions
        own_inputs = self.get_inputs_dict()
        own_parameters = own_inputs['parameters'].get_dict()
        # pseudofamily = own_parameters['pseudofamily']
        # pseudos=get_pseudos(structure=structure,pseudo_family_name=pseudofamily)
        # building parameters for DFT Replay!

        dft_resources = {'num_machines': own_parameters['dft_num_machines']}
        if own_parameters.get('dft_num_mpiprocs_per_machine', 0):
            dft_resources['num_mpiprocs_per_machine'] = own_parameters.get('dft_num_mpiprocs_per_machine')
        inputs_dft = dict(
            moldyn_parameters=get_or_create_input_node(orm.Dict, dict(
                    nstep=self.ctx.nstep,
                    max_wallclock_seconds=own_parameters['dft_walltime_seconds'],
                    resources=dft_resources,
                    is_hustler=True,
                ), store=True),
            structure=own_inputs['structure'],
            hustler_positions=positions,
            parameters=own_inputs['parameters_dft'],
        )
        flipper_resources = {'num_machines': own_parameters['flipper_num_machines']}
        if own_parameters.get('flipper_num_mpiprocs_per_machine', 0):
            flipper_resources['num_mpiprocs_per_machine'] = own_parameters.get('flipper_num_mpiprocs_per_machine')
        inputs_flipper = dict(
            moldyn_parameters=get_or_create_input_node(orm.Dict, dict(
                    nstep=self.ctx.nstep,
                    max_wallclock_seconds=own_parameters['flipper_walltime_seconds'],
                    resources=flipper_resources,
                    is_hustler=True,
                ), store=True),
            structure=own_inputs['structure'],
            hustler_positions=positions,
            parameters=own_inputs['parameters_flipper'],
            remote_folder=self.inp.remote_folder_flipper,
        )

        pseudos_dft = {}
        pseudos_flipper = {}

        for s in own_inputs['structure'].get_site_kindnames():
            # Logic: If I specified a pseudo specifically for the use in only DFT or only flipper part, I pass
            # it with _dft or _flipper. That will be taken by default
            pseudos_dft['pseudo_{}'.format(s)] = own_inputs.get('pseudo_{}_dft'.format(s), None) or own_inputs['pseudo_{}'.format(s)]
            pseudos_flipper['pseudo_{}'.format(s)] = own_inputs.get('pseudo_{}_flipper'.format(s), None) or own_inputs['pseudo_{}'.format(s)]

        inputs_dft.update(pseudos_dft)
        inputs_flipper.update(pseudos_flipper)

        if own_parameters['use_same_settings']:
            inputs_dft['settings'] = own_inputs['settings']
            inputs_flipper['settings'] = own_inputs['settings']
        else:
            inputs_dft['settings'] = own_inputs['settings_dft']
            inputs_flipper['settings'] = own_inputs['settings_flipper']

        if own_parameters['use_same_kpoints']:
            inputs_dft['kpoints'] = own_inputs['kpoints']
            inputs_flipper['kpoints'] = own_inputs['kpoints']
        else:
            inputs_dft['kpoints'] = own_inputs['kpoints_dft']
            inputs_flipper['kpoints'] = own_inputs['kpoints_flipper']

        if own_parameters['use_same_code']:
            inputs_dft['code'] = own_inputs['code']
            inputs_flipper['code'] = own_inputs['code']
        else:
            inputs_dft['code'] = own_inputs['dft_code']
            inputs_flipper['code'] = own_inputs['flipper_code']

        self.goto(self.analyze)

        # Launch replays
        ret = {'hustler_flipper':ReplayMDHWorkChain(**inputs_flipper), 'hustler_dft':ReplayMDHWorkChain(**inputs_dft)}
        ret['hustler_flipper'].label = own_inputs['structure'].label + '_hustler_flipper'
        ret['hustler_dft'].label = own_inputs['structure'].label + '_hustler_DFT'
        return ret

    def analyze(self):
        # TODO: Implement the analysis of how far the hustler reached! and relaunch if necessary!
        # TODO: Remove non-converged steps from the analysis.
        self.goto(self.fit)

    def fit(self):
        parameters_d = self.inp.parameters.get_dict()
        nstep = self.ctx.nstep
        # trajectory_scf = self.out.hustler_dft.out.output_trajectory
        trajectory_scf = self.out.hustler_dft.out.total_trajectory
        # trajectory_pb = self.out.hustler_flipper.out.output_trajectory
        trajectory_pb = self.out.hustler_flipper.out.total_trajectory
        
        for traj in (trajectory_scf, trajectory_pb):
            shape = traj.get_positions().shape
            if shape[0] != nstep:
                #~ raise Exception("Wrong shape of array returned by {} ({} vs {})".format(traj.inp.output_trajectory.id, shape, nstep))
                raise Exception("Wrong shape of array returned by {} ({} vs {})".format(traj.inp.total_trajectory.id, shape, nstep))

        # IMPORTANT TODO: Exclude forces where scf failed! The hustler (maybe?) doesn't fail if SCF doesn't converge...

        params_d = get_or_create_input_node(orm.Dict, dict(
            signal_indices = (1,3,4) if parameters_d['is_local'] else (1,2,3,4),
            symbol=parameters_d['pinball_kind_symbol'],
            stepsize=1,
            nsample=nstep-1, # starting at 1!
            starting_point=0, # The first step is cut within the function!
            divide_r2=parameters_d['divide_r2']
            ), store=True)
        calc, res = get_pinball_factors(
                parameters=params_d,
                trajectory_scf=trajectory_scf,
                trajectory_pb=trajectory_pb)
        coefficients = res['coefficients']
        coefficients.label = '{}-PBcoeffs'.format(self.inp.structure.label)
        try:
            # Maybe I'm supposed to store the result?
            g,_ = orm.Group.get_or_create(name=self.inp.parameters.dict.results_group_name)
            g.add_nodes(coefficients)
        except Exception as e:
            pass
        # TODO: CALL linkv
        self.goto(self.exit)
        return {'get_pinball_factors':calc, 'coefficients':coefficients}

