# -*- coding: utf-8 -*-

from aiida import orm
from aiida.engine.processes.functions import calcfunction
from aiida.engine.processes.workchains.workchain import WorkChain

from aiida.plugins import WorkflowFactory, CalculationFactory
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
from aiida.engine import ToContext, if_, ExitCode

@calcfunction
def make_supercell(structure, distance):
    from supercellor import supercell as sc
    pym_sc_struct = sc.make_supercell(structure.get_pymatgen_structure(), distance, verbosity=0)[0]
    sc_struct = orm.StructureData()
    sc_struct.set_pymatgen(pym_sc_struct)
    return sc_struct

@calcfunction
def delithiate_structure(structure, element_to_remove):
    """
    Take the input structure and create two structures from it.
    One structure is flipper_compatible/pinball_structure which is essentially
    the same structure, just that Li is on first places both in kinds and 
    sites as required for the flipper; the other structure has no Lithium
    """
    
    assert isinstance(structure, orm.StructureData), "input structure needs to be an instance of {}".format(orm.StructureData)

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

    [pinball_structure.append_kind(_) for _ in pinball_kinds]
    [pinball_structure.append_site(_) for _ in pinball_sites]
    [pinball_structure.append_kind(_) for _ in non_pinball_kinds]
    [pinball_structure.append_site(_) for _ in non_pinball_sites]

    [delithiated_structure.append_kind(_) for _ in non_pinball_kinds]
    [delithiated_structure.append_site(_) for _ in non_pinball_sites]

    delithiated_structure.label = delithiated_structure.get_formula(mode='count')
    pinball_structure.label = pinball_structure.get_formula(mode='count')

    return dict(pinball_structure=pinball_structure, delithiated_structure=delithiated_structure)


class PreProcessWorkChain(WorkChain):
    """
    WorkChain that takes a primitive structure as its input and makes supercell using Supercellor class,
    makes the pinball and delithiated structures and then performs an scf calculation on the host lattice,
    stashes the charge densities and wavefunctions. It outputs the pinball supercell and RemoteData containing 
    charge densities to be used in all future workchains for performing MD.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData, required=True,
        help='The primitive structure that is used to build the supercell for MD simulations.')
        spec.input('PwBaseWorkChain_builder_parameters', valid_type=orm.Dict, required=False,
        help='This is a dictionary containing the builder parameters to be used in the host lattice scf run')
        spec.inputs['PwBaseWorkChain_builder_parameters'].default = lambda: orm.Dict(dict={
            'distance': 8,
            'element_to_remove': 'Li',
            'code': 'pw_pinball_5.2',
            'walltime': 14400,
            'num_cores_per_mpiproc': 8,
            'num_machines': 1, 
            'stash_directory':'/home/tthakur/git/diffusion/pinball_example/charge_densities/'})

        spec.outline(
            cls.supercell, cls.scf_run,
            if_(cls.inspect_scf_run)(cls.stash_output), 
            cls.result)

        spec.output('pinball_supercell', valid_type=orm.StructureData)
        spec.output('host_lattice_scf_output', valid_type=orm.RemoteData,
        help='The node containing the symbolic link to the stashed charged densities.')

        spec.exit_code(611, 'ERROR_SCF_FINISHED_WITH_ERROR',
            message='Host Lattice pw scf calculation finished but with some error code.')
        spec.exit_code(612, 'ERROR_SCF_DID_NOT_FINISH', 
            message='Host Lattice pw scf calculation did not finish.')

    def supercell(self):
        ## Create the supercells and store the pinball/flipper structure and delithiated structure in a dictionary
        supercell_parameters_d = self.inputs['PwBaseWorkChain_builder_parameters'].get_dict()
        sc_struct = make_supercell(self.inputs.structure, orm.Int(supercell_parameters_d['distance']))
        self.ctx.supercell = delithiate_structure(sc_struct, orm.Str(supercell_parameters_d['element_to_remove']))

    def scf_run(self):
        ## Run an scf calculation by launching the PwBaseWorkChain

        from aiida_quantumespresso.common.types import ElectronicType
        from aiida.common.datastructures import StashMode
        
        builder_parameters_d = self.inputs['PwBaseWorkChain_builder_parameters'].get_dict()
        PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
        builder = PwBaseWorkChain.get_builder_from_protocol(code=orm.load_code(builder_parameters_d['code']),
            structure=self.ctx.supercell['delithiated_structure'],
            electronic_type=ElectronicType.INSULATOR,
            # could change the pseudo family here or make this part of the input dictionary
            overrides={'pseudo_family': 'SSSP/1.1.2/PBEsol/efficiency'})

        ## the prefix has to be same for all calculations
        # builder['pw']['parameters']['CONTROL'].update({'prefix': 'aiida'})
        # builder.pw.metadata['options']['account'] = 's1073'
        builder['pw']['metadata']['options']['max_wallclock_seconds'] = builder_parameters_d['walltime']
        builder['pw']['parameters']['SYSTEM']['tot_charge'] = float(-self.ctx.supercell['delithiated_structure'].attributes['missing_Li'])
        builder['pw']['parameters']['SYSTEM']['nosym'] = True

        # following is equivalent to the command #SBATCH --ntasks-per-node=
        builder['pw']['metadata']['options']['resources']['num_mpiprocs_per_machine'] = 1
        # following is equivalent to the command #SBATCH --cpus-per-task=
        builder.pw.metadata['options']['resources']['num_cores_per_mpiproc'] = builder_parameters_d['num_cores_per_mpiproc']
        builder.pw.metadata['options']['resources']['num_machines'] = builder_parameters_d['num_machines']
        # to save the output folder on a non-scratch partition
        builder['pw']['metadata']['options'] = {'stash': {'source_list': ['out', 'aiida.in', 'aiida.out'], 
                                                        'target_base': builder_parameters_d['stash_directory'], 
                                                        'stash_mode': StashMode.COPY.value}}
        # no. of cores used = tot_num_mpi_procs*num_cores_per_mpiproc, which is independent of the no. of cores requested
        builder.clean_workdir = orm.Bool(False)
        builder.max_iterations = orm.Int(1)
        
        scf_workchain_node = self.submit(builder)
        return ToContext(add_node=scf_workchain_node)

    def inspect_scf_run(self):
        ## Check if the scf finished properly, and return True if yes

        status = self.ctx.add_node.attributes['process_state']
        exit_status = self.ctx.add_node.attributes['exit_status']

        if status=='finished' and exit_status==0:
            self.ctx.scf_run_success = True
            return self.ctx.scf_run_success
        elif status=='finished' and exit_status!=0:
            self.ctx.scf_run_success = False
            self.report('Host Lattice scf finished but with some error code.')
            return self.exit_codes.ERROR_SCF_FINISHED_WITH_ERROR
        else:
            self.ctx.scf_run_success = False
            self.report('Host Lattice scf did not finish.')
            return self.exit_codes.ERROR_SCF_DID_NOT_FINISH

    def stash_output(self):
        ## Converting the stashed folder to RemoteData for future use
        qb = orm.QueryBuilder()
        qb.append(WorkflowFactory('quantumespresso.pw.base'), filters={'id': {'==':self.ctx.add_node.pk}}, tag='pw')
        qb.append(orm.RemoteStashFolderData, with_incoming='pw')
        stashed_folder_data, = qb.first()
        self.ctx.stashed_output_folder = orm.RemoteData(remote_path=stashed_folder_data.attributes['target_basepath'], computer=stashed_folder_data.computer)

    def result(self):
        if self.ctx.scf_run_success:
            self.out('pinball_supercell', self.ctx.supercell['pinball_structure'])
            self.out('host_lattice_scf_output', self.ctx.stashed_output_folder)

