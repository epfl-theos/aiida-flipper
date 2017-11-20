import numpy as np
from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.calculation.inline import make_inline
from aiida_scripts.database_utils.reuse import get_or_create_parameters
from aiida.orm.data.upf import UpfData
from aiida.orm.data.structure import StructureData, Site
from aiida.common.datastructures import calc_states
from aiida.orm import CalculationFactory
PwCalculation = CalculationFactory('quantumespresso.pw')

@make_inline
def make_pes_structure_inline(structure, parameters):
    # I'm reading the element that is supposed to be removed except for one occurence
    element=parameters.dict.element
    first_pos=parameters.dict.first_pos
    pes_structure=StructureData()
    pes_structure.cell = structure.cell
    pes_structure.pbc = structure.pbc
    [pes_structure.append_kind(_) for _ in structure.kinds]
    #~ for s in structure.sites:
        #~ if s.kind_name == element:
            #~ pos = s.position
    pes_structure.append_site(Site(kind_name=element, position=first_pos))
    [pes_structure.append_site(_) for _ in structure.sites if _.kind_name != element]

    return {'pes_structure':pes_structure}




class PesCalculation(ChillstepCalculation):
    def start(self):
        self.inp.structure
        self.inp.parameters
        self.inp.code
        self.inp.settings
        self.inp.remote_folder
        ## Optional
        self.inp.coefficients
        # can't check pseudos
        own_parametes_d = self.inp.parameters.get_dict()
        pm = get_or_create_parameters(dict(first_pos=[0,0,0], element=own_parametes_d['element']))
        inlinec, res = make_pes_structure_inline(structure=self.inp.structure, parameters=pm)
        self.goto(self.run)
        res['get_pes_structure'] = inlinec
        return res


    def run(self):
        calc = self.inp.code.new_calc()
        pes_structure =self.out.pes_structure
        own_parametes_d = self.inp.parameters.get_dict()
        pes_density = own_parametes_d['pes_density']
        calc.use_structure(pes_structure)
        calc.use_settings(self.inp.settings)
        charge_calc, = self.inp.remote_folder.get_inputs(node_type=PwCalculation)
        charge_params = charge_calc.inp.parameters.get_dict()

        parameters_d = {
            'CONTROL':{
                'calculation':'md',
                'max_seconds': own_parametes_d['max_seconds']-360,
                'lpes': True,
                'verbosity': 'low',
                'lflipper': True,
                'flipper_do_nonloc': not(own_parametes_d['local']),
            },
            'SYSTEM':charge_params['SYSTEM'],
            'ELECTRONS':{},
            'IONS':{},
        }
        coefficients_d = self.inp.coefficients.get_dict()
        if coefficients_d['nr_of_coefs'] == 3:
            parameters_d['SYSTEM']['flipper_local_factor'] = coefficients_d['coefs'][0]
            parameters_d['SYSTEM']['flipper_ewald_rigid_factor'] = coefficients_d['coefs'][1]
            parameters_d['SYSTEM']['flipper_ewald_pinball_factor'] = coefficients_d['coefs'][2]
            parameters_d['CONTROL']['flipper_do_nonloc'] = False
        elif coefficients_d['nr_of_coefs'] == 4:
            parameters_d['SYSTEM']['flipper_local_factor'] = coefficients_d['coefs'][0]
            parameters_d['SYSTEM']['flipper_nonlocal_correction'] = coefficients_d['coefs'][1]
            parameters_d['SYSTEM']['flipper_ewald_rigid_factor'] = coefficients_d['coefs'][2]
            parameters_d['SYSTEM']['flipper_ewald_pinball_factor'] = coefficients_d['coefs'][3]
        else:
            raise Exception("Don't know what to do with {} coefficeints".format(coefficients_d['nr_of_coefs']))

        veca, vecb, vecc = [np.linalg.norm(vec) for vec in pes_structure.get_ase().cell]
        parameters_d['CONTROL']['npoints_a'] = int(veca / pes_density) 
        parameters_d['CONTROL']['npoints_b'] = int(vecb / pes_density)
        parameters_d['CONTROL']['npoints_c'] = int(vecc / pes_density)


        for k, v in self.get_inputs_dict().items():
            if isinstance(v, UpfData):
                calc.use_pseudo(v, k.replace('pseudo_',''))

        calc.set_resources(dict(num_machines=charge_calc.get_resources()['num_machines']))
        calc.add_link_from(self.inp.remote_folder, label='remote_folder')
        calc.set_max_wallclock_seconds(own_parametes_d['max_seconds'])
        calc.use_kpoints(self.inp.kpoints)
        calc.use_parameters(get_or_create_parameters(parameters_d, store=True))
        self.goto(self.analyze)
        calc.submit_test()
        return {'pes_calculation':calc}


    def analyze(self):
        calc = self.out.pes_calculation
        if calc.get_state() != calc_states.FINISHED:
            raise Exception("My PesCalculation ( {} ) did end up in state {}".format(calc.pk, calc.get_state()))
        own_parametes_d = self.inp.parameters.get_dict()
        retrieved = calc.out.retrieved
        try:
            results_group_name = own_parametes_d['results_group_name']
            g = Group.get_from_string(results_group_name)
            g.add_nodes(retrieved)
        except:
            pass

        self.goto(self.exit)
        return {'retrieved_potential':retrieved}
