# -*- coding: utf-8 -*-

from aiida import orm
from aiida.engine.processes.workchains.workchain import WorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.common import AttributeDict, exceptions

from aiida.plugins import WorkflowFactory
from aiida.engine import ToContext, if_, ExitCode, append_
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida_quantumespresso.common.types import ElectronicType
from aiida.common.datastructures import StashMode
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

def make_supercell(structure, distance):
    from supercellor import supercell as sc
    pym_sc_struct = sc.make_supercell(structure.get_pymatgen_structure(), distance, verbosity=0, do_niggli_first=False)[0]
    sc_struct = orm.StructureData()
    sc_struct.set_extra('original_unitcell', structure.uuid)
    sc_struct.set_pymatgen(pym_sc_struct)
    return sc_struct

def delithiate_structure(structure, element_to_remove):
    """
    Take the input structure and create two structures from it.
    One structure is flipper_compatible/pinball_structure which is essentially
    the same structure, just that Li is on first places both in kinds and 
    sites as required for the flipper; the other structure has no Lithium
    """
    
    assert isinstance(structure, orm.StructureData), f'input structure needs to be an instance of {orm.StructureData}'

    pinball_kinds = [kind for kind in structure.kinds if kind.symbol == element_to_remove]

    kindnames_to_delithiate = [kind.name for kind in pinball_kinds]

    non_pinball_kinds = [k for i,k in enumerate(structure.kinds) if k.symbol != element_to_remove]

    non_pinball_sites = [s for s in structure.sites if s.kind_name not in kindnames_to_delithiate]

    pinball_sites = [s for s in structure.sites if s.kind_name in kindnames_to_delithiate]

    delithiated_structure = orm.StructureData()
    pinball_structure = orm.StructureData()

    delithiated_structure.set_cell(structure.cell)
    delithiated_structure.set_attribute('delithiated_structure', True)
    delithiated_structure.set_attribute('missing_Li', len(pinball_sites))
    pinball_structure.set_cell(structure.cell)
    pinball_structure.set_attribute('pinball_structure', True)
    pinball_structure.set_extra('original_unitcell', structure.extras['original_unitcell'])
    pinball_structure.set_attribute('original_unitcell', structure.extras['original_unitcell'])

    [pinball_structure.append_kind(_) for _ in pinball_kinds]
    [pinball_structure.append_site(_) for _ in pinball_sites]
    [pinball_structure.append_kind(_) for _ in non_pinball_kinds]
    [pinball_structure.append_site(_) for _ in non_pinball_sites]

    [delithiated_structure.append_kind(_) for _ in non_pinball_kinds]
    [delithiated_structure.append_site(_) for _ in non_pinball_sites]

    delithiated_structure.label = delithiated_structure.get_formula(mode='count')
    pinball_structure.label = pinball_structure.get_formula(mode='count')

    return dict(pinball_structure=pinball_structure, delithiated_structure=delithiated_structure)

class PreProcessWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain that takes a primitive structure as its input and makes supercell using Supercellor class,
    makes the pinball and delithiated structures and then performs an scf calculation on the host lattice,
    stashes the charge densities and wavefunctions. It outputs the pinball supercell and RemoteData containing 
    charge densities to be used in all future workchains for performing MD.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(PwBaseWorkChain, namespace='prepro',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for running the scf on host lattice.'})
        
        spec.input('clean_workdir', valid_type=orm.Bool,
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.input('distance', valid_type=orm.Float,
            help='The minimum image distance as a float, the cell created will not have any periodic image below this distance.')
        spec.input('element_to_remove', valid_type=orm.Str,
            help='The element that will become the pinball, typically Lithium.')
        spec.input('stash_directory', valid_type=orm.Str, required=False,
            help='The location where host lattice charge denisites will be stored.')
        spec.input('structure', valid_type=orm.StructureData, required=True,
        help='The primitive structure that is used to build the supercell for MD simulations.')

        spec.outline(
            cls.supercell, cls.setup, cls.run_scf,
            cls.inspect_scf, cls.result)

        spec.output('pinball_supercell', valid_type=orm.StructureData,
        help='The Pinball/Flipper compatible structure onto which MD will be run.')
        spec.output('host_lattice_scf_output', valid_type=orm.RemoteData,
        help='The node containing the symbolic link to the stashed charged densities.')

        spec.exit_code(611, 'ERROR_SCF_FINISHED_WITH_ERROR',
            message='Host Lattice pw scf calculation finished but with some error code.')
        spec.exit_code(612, 'ERROR_SCF_FAILED', 
            message='Host Lattice pw scf calculation did not finish.')
        spec.exit_code(613, 'ERROR_KPOINTS_NOT_SPECIFIED', 
            message='Only gamma or automatic kpoints argument is allowed.')

    def supercell(self):
        # Create the supercells and store the pinball/flipper structure and delithiated structure in a dictionary
        if self.inputs.distance == 0: sc_struct = self.inputs.structure
        else: sc_struct = make_supercell(self.inputs.structure, self.inputs.distance)
        self.ctx.supercell = delithiate_structure(sc_struct, self.inputs.element_to_remove)

    def setup(self):
        """Input validation and context setup."""

        # I store all the input dictionaries in context variables
        self.ctx.preprocess_inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='prepro'))
        self.ctx.preprocess_inputs.pw.parameters = self.ctx.preprocess_inputs.pw.parameters.get_dict()
        self.ctx.preprocess_inputs.pw.settings = self.ctx.preprocess_inputs.pw.settings.get_dict()
        if not self.ctx.preprocess_inputs.pw.settings['gamma_only']: 
            return self.exit_codes.ERROR_KPOINTS_NOT_SPECIFIED
        self.ctx.max_wallclock_seconds = self.ctx.preprocess_inputs.pw['metadata']['options']['max_wallclock_seconds']
        
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_flipper.workflows import protocols as proto
        return files(proto) / 'preprocess.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, distance, element_to_remove=None, stash_directory=None, protocol=None, overrides=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param distance: the ``distance`` used to make supercells, if distance is 0 I assume to take it as supercell and do not generate another supercell, do NOT change it after calling the builder
        :param elemet_to_remove: the ``element`` treated as pinball in the model, do NOT change it after calling the builder
        :param stash_directory: the ``path`` where the charge densities of host lattice are stored
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        if element_to_remove: element = element_to_remove
        else: element = inputs['element_to_remove']

        if stash_directory: stash = stash_directory
        else: stash = orm.Str(inputs['stash_directory'])

        if distance == 0: sc_struct = structure
        else: sc_struct = make_supercell(structure, distance)
        supercell = delithiate_structure(sc_struct, element)

        args = (code, structure, protocol)
        PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
        prepro = PwBaseWorkChain.get_builder_from_protocol(*args, 
        electronic_type=ElectronicType.INSULATOR, overrides=inputs['prepro'], **kwargs)

        prepro['pw'].pop('structure', None)
        prepro.pop('clean_workdir', None)

        prepro['pw']['metadata']['options'].update({'stash': {'source_list': ['out', 'aiida.in', 'aiida.out'], 
                                                        'target_base': stash.value, 
                                                        'stash_mode': StashMode.COPY.value}})
        prepro['pw']['parameters']['SYSTEM']['tot_charge'] = float(-supercell['delithiated_structure'].attributes['missing_Li'])

        # removing the Li upf data because the input structure of this builder is unitcell with Li, while the input structure of PwBaseWorkChain is delithiated supercell
        prepro['pw']['pseudos'].pop('Li', None)

        if 'settings' in inputs['prepro']['pw']:
            prepro['pw'].settings = orm.Dict(dict=inputs['prepro']['pw']['settings'])
            if inputs['prepro']['pw']['settings']['gamma_only']:
                kpoints = orm.KpointsData()
                kpoints.set_kpoints_mesh([1,1,1])
                prepro.kpoints = kpoints
            else: 
                raise NotImplementedError('Only gamma k-points possible in flipper calculations, so it is recommended to use the same in host lattice calculation.')
        
        # For hyperqueue scheduler, setting up the required resources options
        if 'hq' in code.get_computer_label(): 
            prepro['pw']['metadata']['options']['resources'].pop('num_cores_per_mpiproc')
            prepro['pw']['metadata']['options']['resources'].pop('num_mpiprocs_per_machine')
            prepro['pw']['metadata']['options']['resources']['num_cores'] = 32
            prepro['pw']['metadata']['options']['resources']['memory_Mb'] = 50000

        builder = cls.get_builder()
        builder.prepro = prepro
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.distance = orm.Float(distance)
        builder.element_to_remove = orm.Str(element)

        return builder

    def run_scf(self):

        inputs = self.ctx.preprocess_inputs
        inputs.pw.structure = self.ctx.supercell['delithiated_structure']
        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'Launching PwBaseWorkChain<{running.pk}>')

        return ToContext(add_node=running)
    
    def inspect_scf(self):
        # Check if the scf finished properly, and stash the charge densities

        workchain = self.ctx.add_node

        if workchain.is_excepted or workchain.is_killed:
            self.report('Host Lattice scf was excepted or killed')
            return self.exit_codes.ERROR_SCF_FAILED

        if workchain.is_failed:
            self.report(f'Host Lattice scf failed with exit status {workchain.exit_status}')
            # If the given time was too short I start another pwbase workchain with longer walltime
            if workchain.exit_status == 401:
                inputs = self.ctx.preprocess_inputs
                inputs.pw['metadata']['options']['max_wallclock_seconds'] = int(self.ctx.max_wallclock_seconds * 10)
                inputs.pw['metadata']['options'].pop('queue_name')
                inputs.pw.structure = self.ctx.supercell['delithiated_structure']
                inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
                running = self.submit(PwBaseWorkChain, **inputs)
                self.report(f'Launching PwBaseWorkChain{running.pk}> with longer walltime')
                return ToContext(add_node=running)
            else:
                return self.exit_codes.ERROR_SCF_FAILED

    def result(self):
        
        workchain = self.ctx.add_node
        try:
            stashed_folder_data = workchain.outputs.remote_stash
            self.ctx.stashed_data = orm.RemoteData(remote_path=stashed_folder_data.attributes['target_basepath'], computer=stashed_folder_data.computer)
        except Exception:
            self.report(f'Host Lattice scf finished with exit status {workchain.exit_status}, but stashed directories not found.')
            return self.exit_codes.ERROR_SCF_FINISHED_WITH_ERROR

        if self.inputs.distance == 0: 
            self.out('pinball_supercell', self.inputs.structure)
        else: self.out('pinball_supercell', self.ctx.supercell['pinball_structure'])
        self.out('host_lattice_scf_output', self.ctx.stashed_data)
