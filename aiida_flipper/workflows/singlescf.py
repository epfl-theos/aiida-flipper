
from aiida.orm import Code
from aiida.orm.data.base import Str, Float, Bool
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.calculation.inline import optional_inline
from aiida.common.exceptions import AiidaException, NotExistent
from aiida.common.datastructures import calc_states
from aiida.work.run import submit
from aiida.work.workchain import WorkChain, ToContext, if_, while_, append_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.workflows.user.epfl_theos.quantumespresso.pw import get_bands_and_occupations_inline


# I say that a band is empty if it has less than a thousands of an electron:
# Corresponds roughly to a 0.2 eV bandgap at 300K
_OCCUPATION_THRESHOLD = 1e-3

@optional_inline
def check_if_insulator_inline(pw_output_parameters, occupations):
    
    # TODO you need to check whether nelec = 2*nbands
    # Not the case for some calculations??
    # First I check whether I have an even number of electrons:
    nelec = pw_output_parameters.get_attr('number_of_electrons')
    results = {}
    if ( nelec % 2):
        # I have an uneven number of electrons
        results['msg'] = 'uneven number of electrons'
        results['is_insulator'] = False
    else:
        occup_arr = occupations.get_array('occupations')
        # nkp is the number of kpoints
        # nbnd the number of bands
        nkp, nbnd = occup_arr.shape
        # I sum over the kpoints of each band times 2 
        occupation_p_band = occup_arr.sum(axis=0) / (2*nkp)
        # Now I see what the occupation of the conduction band is:
        cond_band_idx = int(nelec) / 2 + 1
        lumo_occ = occupation_p_band[cond_band_idx]
        if lumo_occ > _OCCUPATION_THRESHOLD:
            results['is_insulator'] = False
            results['msg'] = 'Occupation of LUMO is above threshold'
        else:
            results['is_insulator'] =  True
            # results['msg'] = 'Occupation of LUMO below threshold'
        results['threshold'] = _OCCUPATION_THRESHOLD
        results['lumo_occ'] = lumo_occ
    return {'output_parameters':ParameterData(dict=results)}
    
    
class SingleScfWorkChain(WorkChain):
    """
    Workchain to launch a Quantum Espresso pw.x to relax a structure
    """

    @classmethod
    def define(cls, spec):
        super(SingleScfWorkChain, cls).define(spec)
        spec.input('code', valid_type=Code)
        spec.input('structure', valid_type=StructureData)
        spec.input_group('pseudos', required=False)
        spec.input('pseudo_family', valid_type=Str, required=False)
        spec.input('kpoints', valid_type=KpointsData, required=False)
        spec.input('kpoints_distance', valid_type=Float, default=Float(0.2))
        # spec.input('vdw_table', valid_type=SinglefileData, required=False)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('settings', valid_type=ParameterData)
        spec.input('options', valid_type=ParameterData)
        spec.outline(
            cls.setup,
            while_(cls.should_run)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.results,
        )

        spec.output('output_parameters', valid_type=ParameterData)
        spec.output('remote_folder', valid_type=RemoteData)
        spec.output('occupations', valid_type=FolderData)


    def setup(self):
        """
        Input validation and context setup
        """
        self.ctx.current_parent_folder = None
        self.ctx.current_cell_volume = None
        self.ctx.is_converged = False
        self.ctx.iteration = 0

        self.ctx.inputs = {
            'code': self.inputs.code,
            'structure': self.inputs.structure,
            'parameters': self.inputs.parameters.get_dict(),
            'settings': self.inputs.settings,
            'options': self.inputs.options,
        }

        # We expect either a KpointsData with given mesh or a desired distance between k-points
        if all([key not in self.inputs for key in ['kpoints', 'kpoints_distance']]):
            self.abort_nowait('neither the kpoints nor a kpoints_distance was specified in the inputs')
            return

        # We expect either a pseudo family string or an explicit list of pseudos
        if self.inputs.pseudo_family:
            self.ctx.inputs['pseudo_family'] = self.inputs.pseudo_family
        elif self.inputs.pseudos:
            self.ctx.inputs['pseudos'] = self.inputs.pseudos
        else:
            self.abort_nowait('neither explicit pseudos nor a pseudo_family was specified in the inputs')
            return

        # Add the van der Waals kernel table file if specified
        #~ if 'vdw_table' in self.inputs:
            #~ self.ctx.inputs['vdw_table'] = self.inputs.vdw_table

        # Set the correct relaxation scheme in the input parameters
        if 'CONTROL' not in self.ctx.inputs['parameters']:
            self.ctx.inputs['parameters']['CONTROL'] = {}

        #~ self.ctx.inputs['parameters']['CONTROL']['calculation'] = self.inputs.relaxation_scheme
        # Construct a kpoint mesh on the current structure or pass the static mesh
        if 'kpoints' not in self.inputs or self.inputs.kpoints == None:
            kpoints = KpointsData()
            kpoints.set_cell_from_structure(self.ctx.inputs['structure'])
            kpoints.set_kpoints_mesh_from_density(self.inputs.kpoints_distance.value, force_parity=True)
            self.ctx.inputs['kpoints'] = kpoints
        else:
            self.ctx.inputs['kpoints'] = self.inputs.kpoints


        return

    def should_run(self):
        """
        Return whether a relaxation workchain should be run, which is the case as long as the volume
        change between two consecutive relaxation runs is larger than the specified volumen convergence
        threshold value.
        """
        return not(self.ctx.is_converged)


    def run_scf(self):
        """
        Run the PwBaseWorkChain
        """
        self.ctx.iteration += 1

        inputs = dict(self.ctx.inputs)
        inputs['parameters'] = ParameterData(dict=inputs['parameters'])

        running = submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}>'.format(running.pid))

        return ToContext(workchains=append_(running))

    def inspect_scf(self):
        """
        Compare the cell volume of the relaxed structure of the last completed workchain with the previous.
        If the difference ratio is less than the volume convergence threshold we consider the cell relaxation
        converged and can quit the workchain. If the 
        """
        # Since I am not able to change anything, let's stop here!
        self.ctx.is_converged = True
        try:
            workchain = self.ctx.workchains[-1]
        except IndexError:
            # I have the suspicion that this this doesn't work! self.abort_nowait
            self.abort_nowait('the first iteration finished without returning a PwBaseWorkChain')
            return


        try:
            remote_folder = workchain.out.remote_folder
            output_parameters = workchain.out.output_parameters
        except AttributeError as exception:
            self.abort_nowait('the workchain {} probably failed'.format(workchain))
            return

    def results(self):
        self.report('I have reached the results')
        workchain = self.ctx.workchains[-1]
        try:
            remote_folder = workchain.out.remote_folder
            output_parameters = workchain.out.output_parameters
            self.report('{} returned remote_folder and output_parameters'.format(workchain))
        except AttributeError as exception:
            self.abort_nowait('the workchain {} probably failed'.format(workchain))
            return
        c, res = get_bands_and_occupations_inline(remote_folder=remote_folder, pw_output_parameters=output_parameters)
        occupations = res['output_band']
        res = check_if_insulator_inline(pw_output_parameters=output_parameters, occupations=occupations, store=True)
        self.report('Based output parameters {} and occupations {} I judge this to be{} an insulator'.
                format(output_parameters, occupations, ' NOT' if not(res['output_parameters'].get_attr('is_insulator')) else ''))
        self.out('output_occupations', occupations)
        self.out('insulator_judgement', res['output_parameters'])
        self.out('remote_folder', remote_folder)



if __name__ == '__main__':
    from aiida.orm import load_node
    p = load_node(398054)
    b = load_node(398058)
    print check_if_insulator_inline(pw_output_parameters=p, occupations=b)['output_parameters'].get_dict()
