from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.calculation.chillstep.user.dynamics.replay import ReplayCalculation
from aiida_flipper.calculations.inline_calcs import get_structure_from_trajectory_inline
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm import Data, load_node
from aiida.common.constants import bohr_to_ang
from aiida.orm.querybuilder import QueryBuilder
from aiida.common.links import LinkType




class BranchingCalculation(ChillstepCalculation):
    """
    Run a Molecular Dynamics calculations
    """
    def start(self):
        print "starting"
        # Get the parameters
        self.ctx.nr_of_branches = self.inputs.parameters_branching.dict.nr_of_branches
        self.goto(self.thermalize)
        assert self.inp.parameters_nvt.dict['IONS']['ion_velocities'] == 'from_input'
        assert self.inp.parameters_nve.dict['IONS']['ion_velocities'] == 'from_input'


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
        inp_d = {k:v for k,v in self.get_inputs_dict().items() if not 'parameters_' in k}
        inp_d['moldyn_parameters'] = self.inp.moldyn_parameters_nvt
        inp_d['parameters'] = self.inp.parameters_nvt
        

        traj = self.out.thermalizer.out.total_trajectory

        kwargs = dict(trajectory=traj, parameters=ParameterData(dict=dict(
                        step_index=-1,
                        recenter=True,
                        create_settings=True,
                        complete_missing=True)),
                structure=self.inp.structure)
        try:
            kwargs['settings'] = self.inp.settings
        except:
            pass # settings will be None
        


        inlinec, res = get_structure_from_trajectory_inline(**kwargs)
        inp_d['settings']=res['settings']
        inp_d['structure']=res['structure']

        self.goto(self.run_NVE)
        return {'slave_NVT':ReplayCalculation(**inp_d), 'get_structure':inlinec}

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
                            recenter=True,
                            create_settings=True,
                            complete_missing=True)))
            inlinec, res = get_structure_from_trajectory_inline(**kwargs)
            inp_d['settings']=res['settings']
            inp_d['structure']=res['structure']
            slaves['slave_NVE_{}'.format(str(count).rjust(len(str(len(indices))),str(0)))] = ReplayCalculation(**inp_d)
            slaves['get_step_{}'.format(idx)] = inlinec
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


        
        

