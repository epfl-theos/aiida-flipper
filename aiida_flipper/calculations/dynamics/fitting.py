from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm import Data, load_node, Calculation, CalculationFactory, Group
from aiida.orm.calculation.inline import optional_inline, make_inline
from aiida.orm.data.array import ArrayData
from aiida_scripts.database_utils.reuse import get_or_create_parameters
from aiida_scripts.upf_utils.get_pseudos import get_pseudos, get_suggested_cutoff
import numpy as np, copy

from replay import ReplayCalculation

HUSTLER_DFT_PARAMS_DICT = {
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

@make_inline
def rattle_randomly_structure_inline(structure, parameters):
    #~ from ase.constraints import FixAtoms
    #~ from random import randint
    elements_to_rattle = parameters.dict.elements
    stdev = parameters.dict.stdev
    nr_of_configurations = parameters.dict.nr_of_configurations
    indices_to_rattle = [i for i,k in enumerate(structure.get_site_kindnames()) if k in elements_to_rattle]
    positions = structure.get_ase().positions
    new_positions = np.repeat(np.array([positions]), nr_of_configurations, axis=0)


    for idx in indices_to_rattle:
        new_positions[:,idx,:] += np.random.normal(0, stdev, (nr_of_configurations, 3))


    # final_positions = np.concatenate(([positions], new_positions))
    
    array = ArrayData()
    array.set_array('symbols', np.array(structure.get_site_kindnames()))
    array.set_array('positions', new_positions)
    array._set_attr('units|positions', 'angstrom')
    return dict(rattled_positions=array)

@make_inline
def get_pinball_factors_inline(parameters, trajectory_scf, trajectory_pb):
    from aiida_scripts.fitting.fit_forces import Force, fit_with_lin_reg, make_fitted, plot_forces
    params_dict = parameters.get_dict()
    starting_point = params_dict['starting_point']
    stepsize = params_dict['stepsize']
    nsample = params_dict.get('nsample', None)
    signal_indices = params_dict.get('signal_indices', None)

    atom_indices_scf = [i for i, s in enumerate(trajectory_scf.get_symbols()) if s == params_dict['symbol']]
    atom_indices_pb = [i for i, s in enumerate(trajectory_pb.get_symbols()) if s == params_dict['symbol']]

    all_forces_scf = trajectory_scf.get_array('forces')[:, atom_indices_scf,:]
    all_forces_pb = trajectory_pb.get_array('forces')[:, atom_indices_pb,:]


    # You need to remove tall the steps that are starting indices due to this stupid thing with the hustler first
    # step. 
    starting_indices = set()
    for traj in (trajectory_pb, trajectory_scf):
        [starting_indices.add(_) for _ in np.where(trajectory_scf.get_array('steps') == 0)[0]]

    for idx in sorted(starting_indices, reverse=True):
        all_forces_scf = np.delete(all_forces_scf, idx, axis=0)
        all_forces_pb  = np.delete(all_forces_pb,  idx, axis=0)

    if nsample == None:
        nsample = min((len(all_forces_scf), len(all_forces_pb)))

    #~ print (all_forces_scf[starting_point:starting_point+nsample*stepsize:stepsize]).shape
    forces_scf = Force(all_forces_scf[starting_point:starting_point+nsample*stepsize:stepsize])
    forces_pb = Force(all_forces_pb[starting_point:starting_point+nsample*stepsize:stepsize])

    coefs, mae = fit_with_lin_reg(forces_scf, forces_pb,
            fit_forces=True, verbosity=0, divide_r2=params_dict['divide_r2'], signal_indices=signal_indices)

    #~ pb_fitted = make_fitted(forces_pb, coefs=coefs, signal_indices=signal_indices)
    #~ plot_forces((forces_scf, pb_fitted))
    try:
        mae_f = float(mae)
    except:
        mae_f = None

    return {'coefficients':ParameterData(dict=dict(coefs=coefs.tolist(), mae=mae_f, nr_of_coefs=len(coefs), indices_removed=sorted(starting_indices)))}

class FittingFlipper1RandomlyDisplacedPosCalculation(ChillstepCalculation):
    def start(self):
        # So, I have a structure that should be at the energetic minumum.
        # I will produce a trajectory that comes from randomly displacing
        # the pinball atoms.

        self.inp.structure
        self.inp.remote_folder_flipper
        # self.inp.electron_parameters
        self.inp.parameters
        # self.inp.flipper_code
        self.inp.pseudo_Li

        parameters_d = self.inp.parameters.get_dict()
        pks = parameters_d['pinball_kind_symbol']
        nr_of_pinballs = self.inp.structure.get_site_kindnames().count(pks)
        # Nr of configurations: How many configuration do I need to achieve the data points I want?
        nr_of_configurations = int(float(parameters_d['nr_of_force_components']) / nr_of_pinballs / 3) + 1 # Every pinball atoms has 3 force components
        rattling_parameters_d = {
            'elements':[pks],
            'nr_of_configurations': nr_of_configurations,
            'stdev':parameters_d['stdev']
        }
        # TODO: CALL link
        c, res = rattled_positions = rattle_randomly_structure_inline(
                structure=self.inp.structure,
                parameters=get_or_create_parameters(rattling_parameters_d))
        
        self.ctx.nstep = res['rattled_positions'].get_attr('array|positions')[0]
        res['rattle_structure'] = c
        self.goto(self.launch_replays)
        return res


    def launch_calculations(self):
        #~ rattled_positions = self.out.rattled_positions
        rattled_positions = self.start()['rattled_positions']
        nstep = self.ctx.nstep
        remote_folder = self.inp.remote_folder
        try:
            chargecalc, = remote_folder.get_inputs(node_type=CalculationFactory('quantumespresso.pw'))
        except Exception as e:
            print e
            # This must have been a copied remote folder
            chargecalc = remote_folder.inp.copied_remote_folder.inp.remote_folder.inp.remote_folder
        print chargecalc

        structure = self.inp.structure
        pseudofamily = self.inp.parameters.dict.pseudofamily
        pseudos=get_pseudos(structure=structure,pseudo_family_name=pseudofamily)
        ecutwfc, ecutrho = get_suggested_cutoff(pseudofamily, pseudos.values())

        flipper_calc = self.inp.flipper_code.new_calc()
        flipper_calc._set_parent_remotedata(remote_folder)
        flipper_calc.use_structure(structure)
    
        flipper_calc.use_array(rattled_positions)
        flipper_calc.use_kpoints(chargecalc.inp.kpoints)
        flipper_calc.use_settings(chargecalc.inp.settings)
        paramaters_input_flipper = chargecalc.inp.parameters.get_dict()
        for card, key in (
                ('SYSTEM', 'tot_charge'),
                ('CONTROL', 'max_seconds'),
                ('ELECTRONS', 'conv_thr'),
                ('ELECTRONS', 'electron_maxstep'),
                ('ELECTRONS', 'mixing_beta'),
                ('ELECTRONS', 'diagonalization')
            ):
            try:
                del paramaters_input_flipper[card][key]
            except KeyError:
                pass
        paramaters_input_flipper['CONTROL']['lhustle'] = True
        paramaters_input_flipper['CONTROL']['verbosity'] = 'low'
        paramaters_input_flipper['CONTROL']['lflipper'] = True
        paramaters_input_flipper['CONTROL']['calculation'] = 'md'
        paramaters_input_flipper['CONTROL']['ldecompose_ewald'] = True
        paramaters_input_flipper['CONTROL']['nstep'] = nstep
        paramaters_input_flipper['IONS'] = {}

        flipper_calc.use_parameters(get_or_create_parameters(paramaters_input_flipper))
        flipper_calc.set_resources(dict(num_machines=chargecalc.get_resources()["num_machines"]))
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
        
        paramaters_input_dft = copy.deepcopy(HUSTLER_DFT_PARAMS_DICT)
        paramaters_input_dft['SYSTEM']['ecutwfc'] = ecutwfc
        paramaters_input_dft['SYSTEM']['ecutrho'] = ecutrho
        paramaters_input_dft['CONTROL']['nstep'] = nstep
        paramaters_input_dft['ELECTRONS'] = self.inp.electron_parameters.get_dict()

        dft_calc.use_parameters(get_or_create_parameters(paramaters_input_dft))
        dft_calc.set_resources(dict(num_machines=self.inp.parameters.dict.dft_num_machines))
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

        # start was before this step, so I have rattled_positions in my outputs!
        rattled_positions = self.out.rattled_positions


        own_inputs = self.get_inputs_dict()
        own_parameters = own_inputs['parameters'].get_dict()
        # pseudofamily = own_parameters['pseudofamily']
        # pseudos=get_pseudos(structure=structure,pseudo_family_name=pseudofamily)
        # building parameters for DFT Replay!
        

        inputs_dft = dict(
            moldyn_parameters=get_or_create_parameters(dict(
                    nstep=self.ctx.nstep,
                    max_wallclock_seconds=self.inp.parameters.dict.dft_walltime_seconds,
                    resources=dict(num_machines=self.inp.parameters.dict.dft_num_machines),
                    is_hustler=True,
                ), store=True),
            structure=own_inputs['structure'],
            hustler_positions=rattled_positions,
            parameters=own_inputs['parameters_dft'],
        )
        inputs_flipper = dict(
            moldyn_parameters=get_or_create_parameters(dict(
                    nstep=self.ctx.nstep,
                    max_wallclock_seconds=own_parameters['flipper_walltime_seconds'],
                    resources=dict(num_machines=own_parameters['flipper_num_machines']),
                    is_hustler=True,
                ), store=True),
            structure=own_inputs['structure'],
            hustler_positions=rattled_positions,
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
        
        ret = {'hustler_flipper':ReplayCalculation(**inputs_flipper), 'hustler_dft':ReplayCalculation(**inputs_dft)}
        ret['hustler_flipper'].label = own_inputs['structure'].label+'_hustler_flipper'
        ret['hustler_dft'].label = own_inputs['structure'].label+'_hustler_DFT'
        return ret

    def analyze(self):
        # TODO: Implement the analysis of how far the hustler reached! and relaunch if necessary!
        # TODO: Remove non-converged steps from the analysis.
        self.goto(self.fit)
    def fit(self):
        parameters_d = self.inp.parameters.get_dict()
        nstep =self.ctx.nstep
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

        params_d = dict(
            signal_indices = (1,3,4) if parameters_d['is_local'] else (1,2,3,4),
            symbol=parameters_d['pinball_kind_symbol'],
            stepsize=1,
            nsample=self.ctx.nstep-1, # starting at 1!
            starting_point=0, # The first step is cut within the function!
            divide_r2=parameters_d['divide_r2']
        )
        calc, res = get_pinball_factors_inline(
                parameters=get_or_create_parameters(params_d),
                trajectory_scf=trajectory_scf,
                trajectory_pb=trajectory_pb)
        coefficients = res['coefficients']
        try:
            # Maybe I'm supposed to store the result?
            g = Group.get_from_string(self.inp.parameters.dict.results_group_name)
            g.add_nodes(coefficients)
        except Exception as e:
            print '!!!!!!!!!', e
            pass
        # TODO: CALL linkv
        self.goto(self.exit)
        return {'get_pinball_factors':calc, 'coefficients':coefficients}

