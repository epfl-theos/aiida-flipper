import copy

from aiida.orm.calculation.chillstep import ChillstepCalculation

from aiida.orm.data.parameter import ParameterData
from aiida.orm import Data, load_node, Calculation
from aiida_scripts.database_utils.reuse import get_or_create_parameters
from aiida.common.datastructures import calc_states
from aiida_scripts.upf_utils.get_pseudos import get_pseudos, get_suggested_cutoff
from aiida.orm.group import Group

CHARGE_PARAMS_DICT = {
    u'CONTROL': {
        u'calculation': 'scf',
        u'verbosity': 'high',
    },
    u'SYSTEM': {
        u'nosym': True,
    }
}


class SinglescfCalculation(ChillstepCalculation):
    def start(self):
        # I need 2 structures
        self.inp.pinball_structure
        self.inp.delithiated_structure
        # I need a code
        self.inp.code
        # I need kpoints
        self.inp.kpoints
        # I need settings
        self.inp.settings
        # I need some general parameters to define pseudofamily, parser_name
        self.inp.parameters
        self.inp.electron_parameters
        self.goto(self.launch)

    def launch(self):
        # I launch the calculation
        
        charge_calc_param_dict = copy.deepcopy(CHARGE_PARAMS_DICT)
        pseudofamily = self.inp.parameters.dict.pseudofamily

        nr_of_atoms_removed = len(self.inp.pinball_structure.sites) -  len(self.inp.delithiated_structure.sites)
        pseudos=get_pseudos(
                        structure=self.inp.pinball_structure,
                        pseudo_family_name=pseudofamily,
                )
        ecutwfc, ecutrho = get_suggested_cutoff(pseudofamily, pseudos.values())
        #~ except Exception as e:
            #~ print "WARNING: defaulting to default cutoffs" 
            #~ ecutwfc, ecutrho = ECUTWFC_DEFAULT, ECUTRHO_DEFAULT
        #~ for kind in set(pinball_structure.get_site_kindnames()).difference(delithiated_structure.get_site_kindnames()):
        # Remove Lithium from the pseudos
        pseudos.pop('Li')
        charge_calc_param_dict['ELECTRONS'] = self.inp.electron_parameters.get_dict()
        charge_calc_param_dict['SYSTEM']['tot_charge'] = -nr_of_atoms_removed
        charge_calc_param_dict['SYSTEM']['ecutwfc'] = ecutwfc
        charge_calc_param_dict['SYSTEM']['ecutrho'] = ecutrho
        charge_calc_param_dict['CONTROL']['max_seconds'] = self.inp.parameters.dict.walltime_seconds - 120

        params = get_or_create_parameters(charge_calc_param_dict)
        calc = self.inp.code.new_calc()

        calc.set_resources({"num_machines": self.inp.parameters.dict.num_machines})
        calc.set_max_wallclock_seconds(self.inp.parameters.dict.walltime_seconds)
        calc.use_parameters(params)
        calc.use_structure(self.inp.delithiated_structure)
        calc.use_settings(self.inp.settings)
        calc.use_kpoints(self.inp.kpoints)
        
        for k,v in pseudos.iteritems():
            calc.use_pseudo(v, k)
        calc.label = "charge-{}".format(self.inp.delithiated_structure.label)
        self.goto(self.check)
        return {'scf_calculation':calc}

    def check(self):
        subc = self.out.scf_calculation
        if subc.get_state() != calc_states.FINISHED:
            raise Exception("My SCF-calculation {} failed".format(subc.uuid))
        else:
            try:
                # Maybe I'm supposed to store the result?
                g = Group.get_from_string(self.inp.parameters.dict.results_group_name)
                g.add_nodes(subc.out.remote_folder)
            except Exception as e:
                print '!!!!!!!!!', e
                pass
            self.goto(self.exit)
            return {
                'remote_folder':subc.out.remote_folder # for the restart
            }

