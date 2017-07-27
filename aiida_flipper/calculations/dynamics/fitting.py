from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm import Data, load_node, Calculation, CalculationFactory
from aiida.orm.calculation.inline import optional_inline
from aiida.orm.data.array import ArrayData
from aiida_scripts.database_utils.reuse import get_or_create_parameters
from aiida_scripts.upf_utils.get_pseudos import get_pseudos, get_suggested_cutoff
import numpy as np, copy

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

@optional_inline
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

@optional_inline
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
    print coefs
    return {'coefficients':ParameterData(dict=dict(coefs=coefs.tolist(), mae=mae_f, nr_of_coefs=len(coefs)))}

class FittingFlipper1RandomlyDisplacedPosCalculation(ChillstepCalculation):
    def start(self):
        # So, I have a structure that should be at the energetic minumum.
        # I will produce a trajectory that comes from randomly displacing
        # the pinball atoms.
        self.inp.structure
        self.inp.remote_folder
        self.inp.electron_parameters
        self.inp.parameters
        self.inp.flipper_code

        self.goto(self.launch_calculations)
        parameters_d = self.inp.parameters.get_dict()
        pks= parameters_d['pinball_kind_symbol']
        nr_of_pinballs = self.inp.structure.get_site_kindnames().count(pks)
        # Nr of configurations: How many configuration do I need to achieve the data points I want?
        nr_of_configurations = int(float(parameters_d['nr_of_force_components']) / nr_of_pinballs / 3) + 1 # Every pinball atoms has 3 force components
        rattling_parameters_d = {
            'elements':[pks],
            'nr_of_configurations': nr_of_configurations,
            'stdev':parameters_d['stdev']
        }
        # TODO: CALL link
        rattled_positions = rattle_randomly_structure_inline(
                structure=self.inp.structure,
                parameters=get_or_create_parameters(rattling_parameters_d), store=True)['rattled_positions']
        self.ctx.nstep = rattled_positions.get_attr('array|positions')[0]
        return {'rattled_positions':rattled_positions}

    def launch_calculations(self):
        #~ rattled_positions = self.out.rattled_positions
        rattled_positions = self.start()['rattled_positions']
        nstep = self.ctx.nstep
        remote_folder = self.inp.remote_folder
        chargecalc, = remote_folder.get_inputs(node_type=CalculationFactory('quantumespresso.pw'))
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

    def fit(self):
        parameters_d = self.inp.parameters.get_dict()
        nstep =self.ctx.nstep
        trajectory_scf = self.out.hustler_dft.out.output_trajectory
        trajectory_pb = self.out.hustler_flipper.out.output_trajectory
        
        for traj in (trajectory_scf, trajectory_pb):
            shape = traj.get_positions().shape
            if shape[0] != nstep:
                raise Exception("Wrong shape of array returned by {} ({} vs {})".format(traj.inp.output_trajectory.id, shape, nstep))

        params_d = dict(
            signal_indices = (1,3,4) if parameters_d['is_local'] else (1,2,3,4),
            symbol=parameters_d['pinball_kind_symbol'],
            stepsize=1,
            nsample=self.ctx.nstep-1, # starting at 1!
            starting_point=1,
            divide_r2=parameters_d['divide_r2']
        )
        # TODO: CALL linkv
        self.goto(self.exit)
        return get_pinball_factors_inline(
                parameters=get_or_create_parameters(params_d),
                trajectory_scf=trajectory_scf,
                trajectory_pb=trajectory_pb, store=True)

