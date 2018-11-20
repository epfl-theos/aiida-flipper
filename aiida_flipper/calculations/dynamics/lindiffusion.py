from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.calculation.chillstep.user.dynamics.replay import ReplayCalculation
from aiida_flipper.calculations.inline_calcs import (
        get_diffusion_from_msd_inline, get_diffusion_from_msd, 
        get_structure_from_trajectory_inline, concatenate_trajectory, concatenate_trajectory_inline)
from aiida.orm import load_node, Group, Calculation, Code
from aiida.orm.querybuilder import QueryBuilder
from aiida.common.links import LinkType
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida_scripts.database_utils.reuse import get_or_create_parameters

from aiida.backends.utils import get_automatic_user
USER = get_automatic_user()

class LindiffusionCalculation(ChillstepCalculation):

    def _get_last_calc(self, diffusion_parameters_d=None):
        if diffusion_parameters_d is None:
            diffusion_parameters_d = self.inputs.diffusion_parameters.get_dict()
        return getattr(self.out, 
            'replay_{}'.format(str(self.ctx.replay_counter-1).rjust(len(str(diffusion_parameters_d['max_nr_of_replays'])),str(0))))

    def start(self):
        # Now, I start by checking that I have all the parameters I need
        # Don't need to check to much because the BranchingCalculations will validate
        # most of the parameters!
        inp_d = self.get_inputs_dict()
        
        for k,v in inp_d.items():
            # This is a top level workflow, but if it is was called by another, I remove the calls:
            if isinstance(v, Calculation):
                inp_d.pop(k)
        # parameters_branching_d = inp_d.pop('parameters_branching').get_dict()
        # assert isinstance(parameters_branching_d['nr_of_branches'], int)
        inp_d.pop('moldyn_parameters_main')
        parameters_main = inp_d.pop('parameters_main').get_dict()

        try:
            inp_d.pop('moldyn_parameters_thermalize')
            inp_d.pop('parameters_thermalize')
            self.ctx.thermalize = True
            self.goto(self.thermalize)
        except KeyError:
            self.ctx.thermalize = False
            self.goto(self.run_replays)

        structure = inp_d.pop('structure')
        for kind in structure.kinds:
            inp_d.pop('pseudo_{}'.format(kind.name))
        inp_d.pop('kpoints')

        # The code label has to be set as an attribute, and can be changed during the dynamics
        Code.get_from_string(self.ctx.code_string)
        self.ctx.num_machines
        self.ctx.walltime_seconds

        # attribute during the dynamics!
        inp_d.pop('settings', None)
        # Optional remote filder
        ## inp_d.pop('remote_folder', None)

        inp_d.pop('msd_parameters')
        inp_d.pop('diffusion_parameters')
        # diffusion_parameters_d = inp_d.pop('diffusion_parameters').get_dict()

        if inp_d:
            raise Exception("More keywords provided than needed: {}".format(inp_d.keys()))

        # The replay Counter counts how many REPLAYS I launched
        self.ctx.replay_counter = 0



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
        c = ReplayCalculation(**inp_d)
        # This will crash because of code etc not being set
        c.label = '{}{}thermalize'.format(self.label, '-' if self.label else '')
        return {'thermalizer':c}


    def run_replays(self):
        """
        Here I restart from the the thermalized run! I run NVT until I have reached the
        number of steps specified in self.inp.moldyn_parameters_NVT.dict.nstep
        """
        # Transfer all the inputs to the subworkflow, without stuff that is paramaters-annotated:
        diffusion_parameters_d = self.inp.diffusion_parameters.get_dict()
        inp_d = {k:v for k,v in self.get_inputs_dict().items() if not 'parameters' in k}
        # These are the right parameters:

        #~ moldyn_params_d = self.inp.moldyn_parameters_main.get_dict()
        #~ moldyn_params_d['max_wallclock_seconds'] = self.ctx.walltime_seconds
        #~ moldyn_params_d['resources'] = {'num_machines':self.ctx.num_machines}
        inp_d['moldyn_parameters'] = self.inp.moldyn_parameters_main # get_or_create_parameters(moldyn_params_d, store=True)
        #~ inp_d['code'] = Code.get_from_string(self.ctx.code_string)
        inp_d['parameters'] = self.inp.parameters_main
        returnval = {}

        # Now I'm checking whether I am starting this
        if self.ctx.replay_counter == 0:
            if self.ctx.thermalize:
                last_calc = self.out.thermalizer
            else:
                last_calc = None
        else:
            last_calc = self._get_last_calc(diffusion_parameters_d)
            if last_calc.get_state() == 'FAILED':
                raise Exception("Last calculation {} failed".format(last_calc))
            elif last_calc.get_state() != 'FINISHED':
                raise Exception("Last calculation {} is in state {}".format(last_calc.get_state()))
                traj = self.out.thermalizer.out.total_trajectory


        if last_calc is None:
            inp_d['structure'] = self.inputs.structure
            inp_d['settings'] = self.inp.settings
        else:
            kwargs = dict(trajectory=last_calc.out.total_trajectory, parameters=get_or_create_parameters(dict(
                            step_index=-1,
                            recenter=False, #self.inputs.parameters_branching.dict.recenter_before_nvt,
                            create_settings=True,
                            complete_missing=False), store=True),
                    # structure=self.inp.structure
                )

            try:
                kwargs['settings'] = self.inp.settings
            except:
                pass # settings will be None

            inlinec, res = get_structure_from_trajectory_inline(**kwargs)
            returnval['get_structure'] = inlinec
            inp_d['settings']=res['settings']
            inp_d['structure']=res['structure']
            # I have to set the parameters so that they read from input!
            params_for_calculation_d = inp_d['parameters'].get_dict()
            params_for_calculation_d['IONS']['ion_velocities'] = 'from_input'
            inp_d['parameters'] = get_or_create_parameters(params_for_calculation_d, store=True)
        repl = ReplayCalculation(**inp_d)
        repl.label = '{}{}replay-{}'.format(self.label, '-' if self.label else '', self.ctx.replay_counter)
        for attr_key in ('num_machines', 'walltime_seconds', 'code_string'):
            repl._set_attr(attr_key , self.get_attr(attr_key))
        #~ moldyn_params_d['resources'] = {'num_machines':self.ctx.num_machines
        returnval = {'replay_{}'.format(str(self.ctx.replay_counter).rjust(len(str(diffusion_parameters_d['max_nr_of_replays'])),str(0))):repl}
        # Last thing I do is set up the counter:
        self.goto(self.check)
        self.ctx.replay_counter += 1
        return returnval


    def check(self):
        diffusion_parameters_d =  self.inputs.diffusion_parameters.get_dict()
        msd_parameters =  self.inputs.msd_parameters

        minimum_nr_of_replays = diffusion_parameters_d.get('min_nr_of_replays', 0)
        lastcalc = self._get_last_calc(diffusion_parameters_d=diffusion_parameters_d)
        if lastcalc.get_state() == 'FAILED':
            raise Exception("Last replay {} failed with message:\n{}".format(lastcalc, lastcalc.get_attr('fail_msg')))

        try:
            user_wants_termination = self.ctx.force_termination
        except AttributeError:
            user_wants_termination = False

        if user_wants_termination:
            print 'User wishes termination'
            self.goto(self.collect)
        elif minimum_nr_of_replays > self.ctx.replay_counter:
            # I don't even care, I just launch the next!
            print 'Did not run enough'
            self.goto(self.run_replays)
        elif diffusion_parameters_d['max_nr_of_replays'] <= self.ctx.replay_counter:
            print 'Cannot run more'
            self.goto(self.collect)
        else:
            # Now let me calculate the diffusion coefficient that I get:
            #~ print len(self._get_trajectories())
            concatenated_trajectory = concatenate_trajectory(**self._get_trajectories())['concatenated_trajectory']
            # I estimate the diffusion coefficients: without storing
            msd_results = get_diffusion_from_msd(
                    structure=self.inputs.structure,
                    parameters=msd_parameters,
                    trajectory=concatenated_trajectory)['msd_results']
            sem = msd_results.get_attr('{}'.format(msd_parameters.dict.species_of_interest[0]))['diffusion_sem_cm2_s']
            mean_d = msd_results.get_attr('{}'.format(msd_parameters.dict.species_of_interest[0]))['diffusion_mean_cm2_s']
            sem_relative = sem / mean_d
            sem_target = diffusion_parameters_d['sem_threshold']
            sem_relative_target = diffusion_parameters_d['sem_relative_threshold']
            print mean_d, sem/mean_d, sem
            #~ return
            if sem < sem_target:
                # This means that the  standard error of the mean in my diffusion coefficient is below the target accuracy
                print "The error ( {} ) is below the target value ( {} )".format(sem, sem_target)
                self.goto(self.collect)
            elif sem_relative < sem_relative_target:
                # the relative error is below my targe value
                print "The relative error ( {} ) is below the target value ( {} )".format(sem_relative, sem_relative_target)
                self.goto(self.collect)

            else:
                print "The error has not converged"
                print "absolute sem: {:.5e}  Target: {:.5e}".format(sem, sem_target)
                print "relative sem: {:.5e}  Target: {:.5e}".format(sem_relative, sem_relative_target)
                self.goto(self.run_replays)


    def collect(self):
        msd_parameters =  self.inputs.msd_parameters
        c1, res1 = concatenate_trajectory_inline(**self._get_trajectories())
        concatenated_trajectory = res1['concatenated_trajectory']

        c2, res2 = get_diffusion_from_msd_inline(
                    structure=self.inputs.structure,
                    parameters=msd_parameters,
                    trajectory=concatenated_trajectory)

        try:
            # Maybe I'm supposed to store the result?
            g = Group.get_from_string(self.inputs.diffusion_parameters.dict.results_group_name)
            g.add_nodes(res2['msd_results'])
        except Exception as e:
            pass


        res2['get_diffusion'] = c2
        res2['concatenate_trajectory'] = c1
        res2.update(res1)
        self.goto(self.exit)
        return res2

    def show_msd_now_nosave(self, **kwargs):
        from aiida.orm.data.parameter import ParameterData
        msd_parameters_d =  self.inputs.msd_parameters.get_dict()
        msd_parameters_d.update(kwargs)
        concatenated_trajectory = concatenate_trajectory(**self._get_trajectories())['concatenated_trajectory']
        # I estimate the diffusion coefficients: without storing
        get_diffusion_from_msd(
                structure=self.inputs.structure,
                parameters=ParameterData(dict=msd_parameters_d),
                trajectory=concatenated_trajectory, plot_and_exit=True)

    def _get_trajectories(self):
        qb = QueryBuilder()
        qb.append(LindiffusionCalculation, filters={'id':self.id}, tag='ldc')
        qb.append(ReplayCalculation, 
                output_of='ldc',
                edge_project='label',
                edge_filters={'type':LinkType.CALL.value, 'label':{'like':'replay_%'}}, 
                tag='repl', edge_tag='mb')
        qb.append(
                TrajectoryData, output_of='repl',
                edge_filters={'type':LinkType.RETURN.value, 'label':'total_trajectory'},
                project='*', tag='t')
        return {'{}'.format(item['mb']['label']):item['t']['*'] for item in qb.iterdict()}

