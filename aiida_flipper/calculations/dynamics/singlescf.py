import copy, os

from aiida.backends.utils import get_authinfo
from aiida.common.datastructures import calc_states
from aiida.orm import Data, load_node, Calculation, DataFactory, Computer
from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.calculation.inline import make_inline
from aiida.orm.data.parameter import ParameterData
from aiida.orm.group import Group

from aiida_flipper.utils import get_pseudos, get_suggested_cutoff
from aiida_flipper.utils import get_or_create_parameters

CHARGE_PARAMS_DICT = {
    u'CONTROL': {
        u'calculation': 'scf',
        u'verbosity': 'high',
    },
    u'SYSTEM': {
        u'nosym': True,
    }
}


def copy_directory(remote_folder, parameters):
    
    #~ print remote_folder, parameters
    RemoteData = DataFactory('remote')
    params_dict = parameters.get_dict()
    computer_dest_name = params_dict.get('destination_computer_name',None)
    if computer_dest_name:
        computer_dest = Computer.get(computer_dest_name)
    else:
        computer_dest = remote_folder.get_computer()
    t_dest = get_authinfo(computer=computer_dest,
                          aiidauser=remote_folder.get_user()).get_transport()
    dest_dir = params_dict['destination_directory']
    # get the uuid of the parent calculation
    calc = remote_folder.inp.remote_folder
    calcuuid = calc.uuid
    t_source = get_authinfo(computer=remote_folder.get_computer(),
                            aiidauser=remote_folder.get_user()).get_transport()
    source_dir = os.path.join(remote_folder.get_remote_path(), calc._OUTPUT_SUBFOLDER)

    with t_dest, t_source:
        # build the destination folder
        t_dest.chdir(dest_dir)
        # we do the same logic as in the repository and in the working directory,
        # i.e. we create the final directory where to put the file splitting the
        # uuid of the calculation
        t_dest.mkdir(calcuuid[:2], ignore_existing=True)
        t_dest.chdir(calcuuid[:2])
        t_dest.mkdir(calcuuid[2:4], ignore_existing=True)
        t_dest.chdir(calcuuid[2:4])
        t_dest.mkdir(calcuuid[4:])
        t_dest.chdir(calcuuid[4:])
        final_dest_dir = t_dest.getcwd()
        print 'Copying directory "{}" to "{}"'.format(source_dir, final_dest_dir)
        # copying files!
        t_source.copy(source_dir, final_dest_dir)
    return {'copied_remote_folder': RemoteData(computer=computer_dest,
                                       remote_path=final_dest_dir)}
@make_inline
def copy_directory_inline(**kwargs):
    """
    Inline calculation to copy the charge density and spin polarization files
    from a pw calculation
    :param parameters: ParameterData object with a dictionary of the form
        {'destination_directory': absolute path of directory where to put the files,
         'destination_computer_name': name of the computer where to put the file
                                      (if absent or None, we take the same
                                      computer as that of remote_folder)
         }
    :param remote_folder: the remote folder of the pw calculation
    :return: a dictionary of the form
        {'density_remote_folder': RemoteData_object}
    """
    return copy_directory(**kwargs)



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
        own_parameters_d = self.inp.parameters.get_dict()
        if subc.get_state() != calc_states.FINISHED:
            raise Exception("My SCF-calculation {} failed".format(subc.uuid))

        returnd = {}
        calc_remote_folder = subc.out.remote_folder
        if own_parameters_d.get('copy_remote', False):
            copy_parameters = self.inp.copy_parameters
            inlinec, res = copy_directory_inline(remote_folder=calc_remote_folder, parameters=self.inp.copy_parameters)
            returnd['copy_directory'] = inlinec
            returnd['remote_folder'] = res['copied_remote_folder']
        else:
            returnd['remote_folder'] = calc_remote_folder

        if 'results_group_name' in own_parameters_d:
            g = Group.get_from_string(own_parameters_d['results_group_name'])
            g.add_nodes(returnd['remote_folder'])

        self.goto(self.exit)
        return returnd
