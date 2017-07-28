from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.calculation.chillstep.user.dynamics.branching import BranchingCalculation
from aiida_flipper.calculations.inline_calcs import get_diffusion_from_msd_inline, get_diffusion_from_msd, get_structure_from_trajectory_inline
from aiida.orm import load_node
from aiida.orm.querybuilder import QueryBuilder
from aiida.common.links import LinkType
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida_scripts.database_utils.reuse import get_or_create_parameters
class DiffusionCalculation(ChillstepCalculation):
    def start(self):
        # Now, I start by checking that I have all the parameters I need
        # Don't need to check to much because the BranchingCalculations will validate
        # most of the parameters!
        inp_d = self.get_inputs_dict()
        diffusion_parameters_d = inp_d.pop('diffusion_parameters').get_dict()
        inp_d.pop('msd_parameters')

        self.ctx.branching_counter = 0
        calculation = BranchingCalculation(**inp_d)
        calculation.label = '{}-branching-{}'.format(self.label, self.ctx.branching_counter)
        self.ctx.lastcalculation_uuid = calculation.uuid
        self.goto(self.iterate)
        return {'branching_{}'.format(str(self.ctx.branching_counter).rjust(len(str(diffusion_parameters_d['max_nr_of_branching'])),str(0))) : calculation}

    def iterate(self):
        diffusion_parameters_d =  self.inputs.diffusion_parameters.get_dict()
        msd_parameters =  self.inputs.msd_parameters
        #~ branches = self.get_branches()
        minimum_nr_of_branching = diffusion_parameters_d.get('min_nr_of_branching', 0)
        
        if minimum_nr_of_branching > self.ctx.branching_counter:
            # I don't even care, I just launch the next!

            self.goto(self.launch_branching)
            
        elif diffusion_parameters_d['max_nr_of_branching'] < self.ctx.branching_counter:

            self.goto(self.collect)
        else:
            # Now let me calculate the diffusion coefficient that I get:
            
            branches = self._get_branches()
            # I estimate the diffusion coefficients: without storing
            msd_results = get_diffusion_from_msd(
                    structure=self.inputs.structure,
                    parameters=msd_parameters,
                    **branches)['msd_results']
            print msd_results.get_attr('Li').keys()
            if msd_results.get_attr('{}'.format(msd_parameters.dict.species_of_interest[0]))['diffusion_sem_cm2_s'] < diffusion_parameters_d['sem_threshold']:
                self.goto(self.collect)
            else:
                # Not converged, launch more!
                self.goto(self.launch_branching)
    def launch_branching(self):
        # Get the last calculation!
        # Make a new settings object to restart from the last configurations
        # getting the last trajectory of the NVT run!

        inp_d = self.get_inputs_dict()
        diffusion_parameters_d = inp_d.pop('diffusion_parameters').get_dict()
        print inp_d.keys()

        nvt_replay = getattr(self.out, 'branching_{}'.format(str(self.ctx.branching_counter).rjust(len(str(diffusion_parameters_d['max_nr_of_branching'])),str(0)))).out.slave_NVT

        
        
        kwargs = dict(trajectory=nvt_replay.out.total_trajectory,
                    parameters=get_or_create_parameters(dict(
                        step_index=-1,
                        recenter=False,
                        create_settings=True,
                        complete_missing=True), store=True),
                structure=self.inp.structure)
        try:
            kwargs['settings'] = self.inp.settings
        except:
            pass # settings will be None

        inlinec, res = get_structure_from_trajectory_inline(**kwargs)
        # Now, let me get t
        
        
        inp_d.pop('msd_parameters')
        inp_d.pop('moldyn_parameters_thermalize')
        inp_d.pop('parameters_thermalize')
        inp_d['structure'] = res['structure']
        inp_d['settings'] = res['settings']

        calculation = BranchingCalculation(**inp_d)
        self.ctx.lastcalculation_uuid = calculation.uuid
        returnvals = {}
        # return the inlinecalc to mark it as a slave. Give it the counter value before incrementing
        returnvals['get_structure_{}'.format(str(self.ctx.branching_counter).rjust(len(str(diffusion_parameters_d['max_nr_of_branching'])),str(0)))] = inlinec
        self.ctx.branching_counter +=1
        calculation.label = '{}-branching-{}'.format(self.label, self.ctx.branching_counter)
        # Give it the value after incrementing!
        returnvals['branching_{}'.format(str(self.ctx.branching_counter).rjust(len(str(diffusion_parameters_d['max_nr_of_branching'])),str(0)))] = calculation
        self.goto(self.iterate)
        return returnvals


    def collect(self):
        msd_parameters =  self.inputs.msd_parameters
        branches = self._get_branches()
        c, res = get_diffusion_from_msd_inline(
                    structure=self.inputs.structure,
                    parameters=msd_parameters,
                    **branches)
        self.goto(self.exit)
        res['get_diffusion'] = c
        return res

    def _get_branches(self):
        qb = QueryBuilder()
        qb.append(DiffusionCalculation, filters={'id':self.id}, tag='b')
        qb.append(BranchingCalculation, 
                output_of='b',
                edge_project='label',
                edge_filters={'type':LinkType.CALL.value, 'label':{'like':'branching_%'}}, 
                tag='c', edge_tag='mb' )
        qb.append(
                TrajectoryData, output_of='c',
                edge_filters={'type':LinkType.RETURN.value, 'label':{'like':'branch_%'}},
                project='*', tag='t', edge_project='label', edge_tag='ct')
        return {'{}-{}'.format(item['mb']['label'], item['ct']['label']):item['t']['*'] for item in qb.iterdict()}

