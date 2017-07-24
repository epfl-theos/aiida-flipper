from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm import Data, load_node, Calculation
#~ from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm.querybuilder import QueryBuilder
from aiida.common.datastructures import calc_states
from aiida.common.links import LinkType
from aiida_flipper.calculations.inline_calcs import concatenate_trajectory_inline, get_structure_from_trajectory_inline

import numpy as np




def get_completed_number_of_steps(calc):
    try:
        nstep = calc.res.nstep
    except AttributeError:
        nstep = calc.out.output_trajectory.get_attr('array|positions.0')
    return nstep



class ReplayCalculation(ChillstepCalculation):
    """
    Run a Molecular Dynamics calculations. Run it until the end! Don't stop!!!
    """
    _MAX_ITERATIONS = 999
    def start(self):
        print "starting"
        self.ctx.steps_todo = self.inputs.moldyn_parameters.dict.nstep # Number of steps I have to do:
        self.ctx.max_steps_percalc = self.inputs.moldyn_parameters.dict.max_steps_percalc # Number of steps I have to do:
        self.ctx.steps_done = 0 # Number of steps done, obviously 0 in the beginning
        self.goto(self.run_calculation)
        self.ctx.iteration = 0


    def run_calculation(self):
        # create a calculation:
        calc = self.inp.code.new_calc()
        max_wallclock_seconds = self.inputs.moldyn_parameters.dict.max_wallclock_seconds
        for linkname, input_node in self.get_inputs_dict().iteritems():
            if linkname.startswith('moldyn_'): # stuff only for the moldyn workflow has this prefix!
                continue
            if isinstance(input_node, Data):
                calc.add_link_from(input_node, label=linkname)
        input_dict = self.inp.parameters.get_dict()
        input_dict['CONTROL']['nstep'] = min((self.ctx.steps_todo, self.ctx.max_steps_percalc))
        input_dict['CONTROL']['max_seconds'] = max_wallclock_seconds - 180 # Give the code 3 minnutes to terminate gracefully
        # set the resources:
        calc.set_resources(self.inputs.moldyn_parameters.dict.resources)
        calc.set_max_wallclock_seconds(max_wallclock_seconds)

        # Now, if I am restarting from a previous calculation, I will use the inline calculation
        # to give me a new structure and new settings!
        return_d = {'calc_{}'.format(str(self.ctx.iteration).rjust(len(str(self._MAX_ITERATIONS)),str(0))):calc}
        if self.ctx.iteration > 0:
            # This is a restart from the previous calculation!
            lastcalc = load_node(self.ctx.lastcalc_uuid)
            # The old way:
            #~ newcalc = lastcalc.create_restart(parent_folder_symlink=True) # I don't really care how this is achieved, the plugin needs to create a valid restart!

            input_dict['IONS']['ion_velocities'] = 'from_input'
            kwargs = dict(trajectory=lastcalc.out.output_trajectory,
                    parameters=ParameterData(dict=dict(
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
            calc.use_structure(res['structure'])
            calc.use_settings(res['settings'])
            return_d['get_structure'] = inlinec

        calc.use_parameters(ParameterData(dict=input_dict))
        self.ctx.lastcalc_uuid = calc.uuid
        self.goto(self.evaluate_calc)
        return return_d

    def evaluate_calc(self):
        lastcalc = load_node(self.ctx.lastcalc_uuid)
        if lastcalc.get_state() != calc_states.FINISHED: # Check what todo to improve
            raise Exception("My last calculation {} did not finish".format(lastcalc))
        nsteps_run_last_calc = get_completed_number_of_steps(lastcalc)
        self.ctx.steps_todo -= nsteps_run_last_calc
        self.ctx.steps_done += nsteps_run_last_calc
        if self.ctx.steps_todo > 0:
            # I have to run another calculation
            self.ctx.iteration += 1
            self.goto(self.run_calculation)
        else:
            self.goto(self.produce_output_trajectory)


    def firstcalc(self):
        calc = self.inp.code.new_calc()
        for linkname, input_node in self.get_inputs_dict().iteritems():
            if linkname.startswith('moldyn_'): # stuff only for the moldyn workflow has this prefix!
                continue
            if isinstance(input_node, Data):
                calc.add_link_from(input_node, label=linkname)
        # set the resources:
        input_dict = self.inp.parameters.get_dict()
        input_dict['CONTROL']['nstep'] = min((self.ctx.steps_todo, self.ctx.max_steps_percalc))
        calc.use_parameters(ParameterData(dict=input_dict))
        calc.set_resources(self.inputs.moldyn_parameters.dict.resources)
        calc.set_max_wallclock_seconds(self.inputs.moldyn_parameters.dict.max_wallclock_seconds)
        self.ctx.lastcalc_uuid = calc.uuid
        self.goto(self.iterate)
        return {'calc_{}'.format(str(self.ctx.iteration).rjust(len(str(self._MAX_ITERATIONS)),str(0))):calc}

        # I simply us all my inputs here as input for the calculation!
    
    def iterate(self):
        """
        Check if I have to run again. If not, I exit
        """
        lastcalc = load_node(self.ctx.lastcalc_uuid)
        if lastcalc.get_state() != calc_states.FINISHED:
            raise Exception("My last calculation {} did not finish".format(lastcalc))
        # get the steps I run
        if self.ctx.steps_todo > 0:
            # I have to run another calculation
            self.ctx.iteration += 1
            input_dict = newcalc.inp.parameters.get_dict()
            input_dict['CONTROL']['nstep'] = min((self.ctx.steps_todo, self.ctx.max_steps_percalc))
            newcalc.use_parameters(ParameterData(dict=input_dict))
            self.goto(self.iterate)
            self.ctx.lastcalc_uuid = newcalc.uuid
            return {'calc_{}'.format(str(self.ctx.iteration).rjust(len(str(self._MAX_ITERATIONS)),str(0))):newcalc}
        else:
            self.goto(self.produce_output_trajectory)


    def produce_output_trajectory(self):
        qb = QueryBuilder()
        qb.append(ReplayCalculation, filters={'id':self.id}, tag='m')
        qb.append(Calculation, output_of='m', edge_project='label', edge_filters={'type':LinkType.CALL.value, 'label':{'like':'calc_%'}}, tag='c', edge_tag='mc')
        qb.append(TrajectoryData, output_of='c', project='*', tag='t')
        d = {item['mc']['label'].replace('calc_', 'trajectory_'):item['t']['*'] for item in qb.iterdict()}
        calc, res = concatenate_trajectory_inline(**d)
        self.goto(self.exit)
        return {'concatenate':calc, 'total_trajectory':res['concatenated_trajectory']}

    def get_slave_calculations(self):
        """
        Returns a list of the calculations that was called by the WF, ordered.
        """
        qb = QueryBuilder()
        qb.append(ReplayCalculation, filters={'id':self.id}, tag='m')
        qb.append(Calculation, output_of='m', edge_project='label', edge_filters={'type':LinkType.CALL.value, 'label':{'like':'calc_%'}}, tag='c', edge_tag='mc', project='*')
        d = {item['mc']['label']:item['c']['*'] for item in qb.iterdict()}
        sorted_calcs = sorted(d.items())
        return zip(*sorted_calcs)[1]

    def get_output_trajectory(self, store=False):
        # I don't even have to be finished,  for this
        qb = QueryBuilder()
        qb.append(ReplayCalculation, filters={'id':self.id}, tag='m')
        qb.append(Calculation, output_of='m', edge_project='label', edge_filters={'type':LinkType.CALL.value, 'label':{'like':'calc_%'}}, tag='c', edge_tag='mc')
        qb.append(TrajectoryData, output_of='c', project='*', tag='t')
        d = {item['mc']['label'].replace('calc_', 'trajectory_'):item['t']['*'] for item in qb.iterdict()}
        return concatenate_trajectory_inline(store=store, **d)['concatenated_trajectory']

        
        
        
        
        
