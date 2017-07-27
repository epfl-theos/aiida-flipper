from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.calculation.chillstep.user.dynamics.replay import ReplayCalculation
from aiida_flipper.calculations.inline_calcs import get_structure_from_trajectory_inline
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm import Data, load_node, Calculation
from aiida.common.constants import bohr_to_ang
from aiida.orm.querybuilder import QueryBuilder
from aiida.common.links import LinkType




class BranchingCalculation(ChillstepCalculation):
    """
    Run a Molecular Dynamics calculations
    """
    def _validate(self):
        inp_d = self.get_inputs_dict()
        # Also my caller appears here:
        for k,v in inp_d.items():
            if isinstance(v, Calculation):
                inp_d.pop(k)
        parameters_branching_d = inp_d.pop('parameters_branching').get_dict()
        assert isinstance(parameters_branching_d['nr_of_branches'], int)
        parameters_nvt_d = inp_d.pop('parameters_nvt').get_dict()
        assert parameters_nvt_d['IONS']['ion_velocities'] == 'from_input'
        parameters_nve_d = inp_d.pop('parameters_nve').get_dict()
        assert parameters_nve_d['IONS']['ion_velocities'] == 'from_input'
        
        inp_d.pop('moldyn_parameters_nvt')
        inp_d.pop('moldyn_parameters_nve')
        structure = inp_d.pop('structure')
        for kind in structure.kinds:
            # Correct pseudos?
            inp_d.pop('pseudo_{}'.format(kind.name))
        inp_d.pop('kpoints')
        inp_d.pop('code')


        try:
            inp_d.pop('moldyn_parameters_thermalize')
            inp_d.pop('parameters_thermalize')
            self.ctx.thermalize = True
        except KeyError:
            self.ctx.thermalize = False

        # Optional settings:
        inp_d.pop('settings', None)
        # Optional remote filder
        inp_d.pop('remote_folder', None)
        if inp_d:
            raise Exception("More keywords provided than needed: {}".format(inp_d.keys()))

    def start(self):
        print "starting"
        # Get the parameters
        params_d = self.inputs.parameters_branching.get_dict()
        self.ctx.nr_of_branches = params_d['nr_of_branches']
        if self.ctx.thermalize:
            self.goto(self.thermalize)
        else:
            self.goto(self.run_NVT)
        

    def thermalize(self):
        """
        Thermalize a run! This is the first set of calculations, I thermalize with the criterion
        being the number of steps set in moldyn_parameters_thermalize.dict.nstep
        """
        # all the settings are the same for thermalization, NVE and NVT
        inp_d = {k:v for k,v in self.get_inputs_dict().items() if not 'parameters_' in k}
        inp_d['moldyn_parameters'] = self.inp.moldyn_parameters_thermalize
        inp_d['parameters'] = self.inp.parameters_thermalize
        self.goto(self.run_NVT)
        return {'thermalizer':ReplayCalculation(**inp_d)}

    def run_NVT(self):
        """
        Here I restart from the the thermalized run! I run NVT until I have reached the
        number of steps specified in self.inp.moldyn_parameters_NVT.dict.nstep
        """
        # Transfer all the inputs to the subworkflow, without stuff that is paramaters-annotated:
        inp_d = {k:v for k,v in self.get_inputs_dict().items() if not 'parameters_' in k}
        # These are the right parameters:
        inp_d['moldyn_parameters'] = self.inp.moldyn_parameters_nvt
        inp_d['parameters'] = self.inp.parameters_nvt
        returnval = {}
        if self.ctx.thermalize:
            traj = self.out.thermalizer.out.total_trajectory

            kwargs = dict(trajectory=traj, parameters=ParameterData(dict=dict(
                            step_index=-1,
                            recenter=self.inputs.parameters_branching.dict.recenter_before_nvt,
                            create_settings=True,
                            complete_missing=True)),
                    structure=self.inp.structure)

            try:
                kwargs['settings'] = self.inp.settings
            except:
                pass # settings will be None

            inlinec, res = get_structure_from_trajectory_inline(**kwargs)
            returnval['get_structure'] = inlinec
            inp_d['settings']=res['settings']
            inp_d['structure']=res['structure']

        returnval['slave_NVT'] = ReplayCalculation(**inp_d)
        self.goto(self.run_NVE)
        return returnval

    def run_NVE(self):
        inp_d = {k:v for k,v in self.get_inputs_dict().items() if not 'parameters_' in k}
        inp_d['moldyn_parameters'] = self.inp.moldyn_parameters_nve
        inp_d['parameters'] = self.inp.parameters_nve

        traj = self.out.slave_NVT.out.total_trajectory

        trajlen = traj.get_positions().shape[0]
        block_length =  1.0*trajlen / self.ctx.nr_of_branches
        
        indices = [int(i*block_length)-1 for i in range(1, self.ctx.nr_of_branches+1)]
        try:
            settings = self.inp.settings
        except:
            settings = ParameterData().store()
        slaves = {}
        for count, idx in enumerate(indices):
            kwargs = dict(
                    structure=self.inp.structure, trajectory=traj, settings=settings,
                    parameters=ParameterData(dict=dict(
                            step_index=idx,
                            recenter=self.parameters_branching.dict.recenter_before_nve,
                            create_settings=True,
                            complete_missing=True)))
            inlinec, res = get_structure_from_trajectory_inline(**kwargs)
            inp_d['settings']=res['settings']
            inp_d['structure']=res['structure']
            slaves['slave_NVE_{}'.format(str(count).rjust(len(str(len(indices))),str(0)))]  = ReplayCalculation(**inp_d)
            slaves['get_step_{}'.format(str(idx).rjust(len(str(len(indices))),str(0)))] = inlinec
        self.goto(self.collect_trajectories)
        return slaves

    def collect_trajectories(self):
        qb = QueryBuilder()
        qb.append(BranchingCalculation, filters={'id':self.id}, tag='b')
        qb.append(ReplayCalculation, output_of='b', edge_project='label', edge_filters={'type':LinkType.CALL.value, 'label':{'like':'slave_NVE_%'}}, tag='c', edge_tag='mb')
        qb.append(TrajectoryData, output_of='c', edge_filters={'type':LinkType.RETURN.value, 'label':'total_trajectory'}, project='*', tag='t')
        d = {item['mb']['label'].replace('slave_NVE_', 'branch_'):item['t']['*'] for item in qb.iterdict()}
        self.goto(self.exit)
        return d

    def get_output_trajectories(self, store=False):
        # I don't even have to be finished,  for this
        qb = QueryBuilder()
        qb.append(BranchingCalculation, filters={'id':self.id}, tag='b')
        qb.append(ReplayCalculation, output_of='b', edge_project='label', edge_filters={'type':LinkType.CALL.value, 'label':{'like':'slave_NVE_%'}}, tag='c', edge_tag='mb', project='*')
        d = {item['mb']['label']:item['c']['*'].get_output_trajectory() for item in qb.iterdict()}
        return zip(*sorted(d.items()))[1]

