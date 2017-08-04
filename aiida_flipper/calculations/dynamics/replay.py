from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm import Data, load_node, Calculation
#~ from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm.querybuilder import QueryBuilder
from aiida.common.datastructures import calc_states
from aiida.common.links import LinkType
from aiida_flipper.calculations.inline_calcs import concatenate_trajectory_inline, get_structure_from_trajectory_inline
from aiida_scripts.database_utils.reuse import get_or_create_parameters

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
        moldyn_parameters_d = self.inputs.moldyn_parameters.get_dict()
        self.ctx.steps_todo = moldyn_parameters_d['nstep'] # Number of steps I have to do:
        self.ctx.max_steps_percalc = moldyn_parameters_d.get('max_steps_percalc', None) # Number of steps I have to do:
        self.ctx.steps_done = 0 # Number of steps done, obviously 0 in the beginning
        self.goto(self.run_calculation)
        self.ctx.iteration = 0
        # TODO, you could potentially change that in the future
        self.ctx.restart_from = None


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
        if self.ctx.max_steps_percalc is not None:
            input_dict['CONTROL']['nstep'] = min((self.ctx.steps_todo, self.ctx.max_steps_percalc))
        else:
            input_dict['CONTROL']['nstep'] = self.ctx.steps_todo
        
        input_dict['CONTROL']['max_seconds'] = max((max_wallclock_seconds-180, max_wallclock_seconds*0.9)) # Give the code 3 minnutes to terminate gracefully, or 90% of your estimate (for very low numbers, to avoid negative)
        # set the resources:
        calc.set_resources(self.inputs.moldyn_parameters.dict.resources)
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        try:
            calc._set_parent_remotedata(self.inp.remote_folder)
        except:
            # No remote folder!
            pass
        # Now, if I am restarting from a previous calculation, I will use the inline calculation
        # to give me a new structure and new settings!
        return_d = {'calc_{}'.format(str(self.ctx.iteration).rjust(len(str(self._MAX_ITERATIONS)),str(0))):calc}
        if self.ctx.restart_from:
            # This is a restart from the previous calculation!
            lastcalc = load_node(self.ctx.restart_from)


            input_dict['IONS']['ion_velocities'] = 'from_input'
            kwargs = dict(trajectory=lastcalc.out.output_trajectory,
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
            calc.use_structure(res['structure'])
            calc.use_settings(res['settings'])
            return_d['get_structure'] = inlinec

        calc.use_parameters(get_or_create_parameters(input_dict, store=True))
        calc.label = '{}-{}'.format(self.label or 'Replay', self.ctx.iteration)
        self.ctx.lastcalc_uuid = calc.uuid
        self.goto(self.evaluate_calc)
        return return_d

    def evaluate_calc(self):
        lastcalc = load_node(self.ctx.lastcalc_uuid)
        if lastcalc.get_state() != calc_states.FINISHED:
            # This is temporary solution implementing an exponential backoff mechanism.
            # I raise the iterations by 1
            # and I raise a flag that tells me that there are problems...
            # I call this the back-off counter, and the prob that this chillstep is ticked
            # decreases expontntially with the value of the backoff-counter.
            # There is a limit to how many times I can do this, obviously, and I'm gonna set this to 5
            # for now
            try:
                backoff_counter = self.ctx.backoff_counter
            except AttributeError:
                backoff_counter = 0
            if backoff_counter > 5:
                raise Exception("My last calculation {} did not finish, and I used my 5 trials up!".format(lastcalc))
            try:
                self.ctx.restart_from
            except:
                self.ctx.restart_from = None
            self.ctx.iteration += 1
            self.ctx.backoff_counter = backoff_counter + 1
            self.goto(self.run_calculation)
        else:
            nsteps_run_last_calc = get_completed_number_of_steps(lastcalc)
            self.ctx.steps_todo -= nsteps_run_last_calc
            self.ctx.steps_done += nsteps_run_last_calc
            # I set the calculation to restart from as the last one!
            
            if self.ctx.steps_todo > 0:
                # I have to run another calculation
                self.ctx.restart_from = self.ctx.lastcalc_uuid
                self.ctx.iteration += 1
                self.goto(self.run_calculation)
            else:
                self.goto(self.produce_output_trajectory)



    def produce_output_trajectory(self):
        qb = QueryBuilder()
        qb.append(ReplayCalculation, filters={'id':self.id}, tag='m')
        qb.append(Calculation, output_of='m', edge_project='label', edge_filters={'type':LinkType.CALL.value, 'label':{'like':'calc_%'}}, tag='c', edge_tag='mc')
        qb.append(TrajectoryData, output_of='c', project='*', tag='t')
        d = {item['mc']['label'].replace('calc_', 'trajectory_'):item['t']['*'] for item in qb.iterdict()}
        # If I have produced several trajectories, I concatenate them here:
        if len(d) > 1:
            calc, res = concatenate_trajectory_inline(**d)
            returnval = {'concatenate':calc, 'total_trajectory':res['concatenated_trajectory']}
        elif len(d) == 1:
            # No reason to concatenate if I have only one trajectory (saves space in repository)
            returnval = {'total_trajectory':d.values()[0]}
        else:
            raise Exception("I found no trajectories produced")

        self.goto(self.exit)
        return returnval

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

        
        
        
        
        
