import copy

from aiida.common.datastructures import calc_states
from aiida.common.exceptions import AiidaException, NotExistent
from aiida.common.datastructures import calc_states

from aiida.orm import Data, load_node, Calculation, Code
from aiida.orm.calculation.chillstep import ChillstepCalculation

from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.base import Str, Float, Bool
from aiida.orm.data.folder import FolderData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.structure import StructureData
from aiida.orm.group import Group
from aiida.workflows.user.epfl_theos.quantumespresso.pw import get_bands_and_occupations_inline

from aiida_flipper.utils import (get_or_create_parameters, get_pseudos, get_suggested_cutoff)
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain


class ScfwfCalculation(WorkChain):
    """
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
        spec.output('output_structure', valid_type=StructureData)
        spec.output('output_parameters', valid_type=ParameterData)
        spec.output('remote_folder', valid_type=RemoteData)
        spec.output('retrieved', valid_type=FolderData)


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
        return not self.ctx.is_converged


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
        try:
            workchain = self.ctx.workchains[-1]
        except IndexError:
            self.abort_nowait('the first iteration finished without returning a PwBaseWorkChain')
            return


        try:
            remote_folder = workchain.out.remote_folder
            output_parameters = workchain.out.output_parameters
        except AttributeError as exception:
            self.abort_nowait('the workchain {} probably failed'.format(workchain))
            return

        self.ctx.is_converged = True
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

        print remote_folder, output_parameters
        c, res = get_bands_and_occupations_inline(remote_folder=remote_folder, pw_output_parameters=output_parameters)
        print c
        print res
        occupations = res['output_band']

        self.report('I have output parameters {} and occupations {}'.format(output_parameters, occupations))
        return
