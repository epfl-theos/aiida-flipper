
from aiida.common.datastructures import calc_states
from aiida.common.links import LinkType
from aiida.orm import Data, load_node, Calculation, Code
from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.calculation.inline import InlineCalculation, make_inline
from aiida.orm.data.array import ArrayData
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.querybuilder import QueryBuilder

from aiida_flipper.calculations.inline_calcs import concatenate_trajectory_inline, concatenate_trajectory_new_inline, get_structure_from_trajectory_inline
from aiida_flipper.utils import get_or_create_parameters

import numpy as np

NTRIALS = 8


def get_completed_number_of_steps(calc):
    #~ try:
        #~ nstep = calc.res.nstep
    #~ except AttributeError:
    # Reading the number of steps from the trajectory!
    nstep = calc.inp.parameters.dict.CONTROL.get('iprint', 1)*calc.out.output_trajectory.get_attr('array|positions.0')
    return nstep

@make_inline
def split_hustler_array_inline(array, parameters):
    assert isinstance(parameters, ParameterData), "parametes is not ParameterData"
    assert isinstance(array, ArrayData)

    newarray = ArrayData()
    newarray.set_array('symbols', array.get_array('symbols'))
    newarray.set_array('positions', array.get_array('positions')[parameters.dict.index:])
    newarray._set_attr('units|positions', array.get_attr('units|positions'))
    return dict(split_array=newarray)


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
        #~ if moldyn_parameters_d.get('is_hustler', False):
            #~ self.ctx.array_splitting_indices = []


    def run_calculation(self):
        # create a calculation:

        try:
            code = self.inp.code
        except AttributeError:
            code = Code.get_from_string(self.ctx.code_string)
        calc = code.new_calc()
        moldyn_parameters_d = self.inputs.moldyn_parameters.get_dict()
        try:
            max_wallclock_seconds = self.ctx.walltime_seconds
        except (AttributeError, KeyError):
            max_wallclock_seconds = moldyn_parameters_d['max_wallclock_seconds']
        queue_name = moldyn_parameters_d.get('queue_name', None)
        custom_scheduler_commands = moldyn_parameters_d.get('custom_scheduler_commands', None)

        for linkname, input_node in self.get_inputs_dict().iteritems():
            if linkname.startswith('moldyn_'): # stuff only for the moldyn workflow has this prefix!
                continue
            if linkname in ('hustler_positions', ):
                continue
            if isinstance(input_node, Data):
                calc.add_link_from(input_node, label=linkname)
        input_dict = self.inp.parameters.get_dict()
        if self.ctx.max_steps_percalc is not None:
            input_dict['CONTROL']['nstep'] = min((self.ctx.steps_todo, self.ctx.max_steps_percalc))
        else:
            input_dict['CONTROL']['nstep'] = self.ctx.steps_todo
        # Give the code 3 minnutes to terminate gracefully, or 90% of your estimate (for very low numbers, to avoid negative)
        input_dict['CONTROL']['max_seconds'] = max((max_wallclock_seconds-180, max_wallclock_seconds*0.9))
        if not input_dict['CONTROL'].get('lflipper', False) and not input_dict['CONTROL'].get('lhustle', False):
            input_dict['IONS']['wfc_extrapolation'] = 'second_order'
            input_dict['IONS']['pot_extrapolation'] = 'second_order'
        # set the resources:
        try:
            resources = {"num_machines": self.ctx.num_machines}
        except (KeyError, AttributeError):
            resources = self.inputs.moldyn_parameters.dict.resources
        calc.set_resources(resources)
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_queue_name(queue_name)
        if custom_scheduler_commands is not None:
            calc.set_custom_scheduler_commands(custom_scheduler_commands)
        try:
            # There's something very strange going on: This works only for flipper, not for Hustler! Why???
            calc._set_parent_remotedata(self.inp.remote_folder)
        except Exception as e:
            # No remote folder!
            print e
            pass
        # Now, if I am restarting from a previous calculation, I will use the inline calculation
        # to give me a new structure and new settings!
        return_d = {'calc_{}'.format(str(self.ctx.iteration).rjust(len(str(self._MAX_ITERATIONS)),str(0))):calc}
        if moldyn_parameters_d.get('is_hustler', False):
            hustler_positions = self.inputs.hustler_positions
            if self.ctx.steps_done:
                #~ self.ctx.array_splitting_indices.append(self.ctx.steps_done)
                inlinec, res = split_hustler_array_inline(
                        array=hustler_positions,
                        parameters=get_or_create_parameters(dict(index=self.ctx.steps_done)))
                return_d['split_hustler_array_{}'.format(str(self.ctx.iteration).rjust(len(str(self._MAX_ITERATIONS)),str(0)))] = inlinec
                calc.use_array(res['split_array'])
            else:
                calc.use_array(hustler_positions)
        elif self.ctx.restart_from:
            # This is a restart from the previous calculation!
            lastcalc = load_node(self.ctx.restart_from)
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

        # if settings contain velocities, use them
        if 'settings' in calc.get_inputs_dict() and calc.inp.settings.get_attr('ATOMIC_VELOCITIES', None):
            input_dict['IONS']['ion_velocities'] = 'from_input'
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
            if backoff_counter > NTRIALS:
                raise Exception("My last calculation {} did not finish, and I used my {} trials up!".format(lastcalc, NTRIALS))
            try:
                self.ctx.restart_from
            except:
                self.ctx.restart_from = None
            self.ctx.iteration += 1
            self.ctx.backoff_counter = backoff_counter + 1
            self.goto(self.run_calculation)
        else:
            # Set backoff counter back to 0 since the calculation now finished:
            try:
                if self.ctx.backoff_counter > 0:
                    self.ctx.backoff_counter = 0
            except AttributeError:
                pass
            total_energy_max_fluctuation = self.inputs.moldyn_parameters.get_dict().get('total_energy_max_fluctuation', None)
            if total_energy_max_fluctuation is not None:
                # Checking the fluctuations of the total energy:
                t = lastcalc.out.output_trajectory
                total_energies = t.get_array('total_energies')
                diff = total_energies.max() - total_energies.min()
                print diff, total_energy_max_fluctuation
                if diff > total_energy_max_fluctuation:
                    print "NOOO"
                    raise Exception("Fluctuations of the total energy ( {} ) exceeded threshold ( {} ) !".format(diff, total_energy_max_fluctuation))
                else:
                    print "ALL GOOD"

            nsteps_run_last_calc = get_completed_number_of_steps(lastcalc)
            self.ctx.steps_todo -= nsteps_run_last_calc
            self.ctx.steps_done += nsteps_run_last_calc
            #~ print nsteps_run_last_calc, self.ctx.steps_todo,self.ctx.steps_done
            #~ return
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
        qb.append(ReplayCalculation, filters={'id': self.id}, tag='m')
        # TODO: Are filters on the state of the calculation needed here?
        qb.append(Calculation, output_of='m', edge_project='label', edge_filters={'type':LinkType.CALL.value, 'label':{'like':'calc_%'}}, tag='c', edge_tag='mc')
        qb.append(TrajectoryData, output_of='c', project='*', tag='t')
        d = {item['mc']['label'].replace('calc_', 'trajectory_'):item['t']['*'] for item in qb.iterdict()}
        # If I have produced several trajectories, I concatenate them here:
        if len(d) > 1:
            if (self.inputs.moldyn_parameters.get_attr('is_hustler', False) or
                    not(self.inputs.moldyn_parameters.get_attr('remove_repeated_last_step', True))):
                calc, res = concatenate_trajectory_new_inline(**d)
            else:
                calc, res = concatenate_trajectory_inline(**d)
            returnval = {'concatenate':calc, 'total_trajectory':res['concatenated_trajectory']}
        elif len(d) == 1:
            # No reason to concatenate if I have only one trajectory (saves space in repository)
            returnval = {'total_trajectory':d.values()[0]}
        else:
            raise Exception("I found no trajectories produced")

        moldyn_parameters_d = self.inputs.moldyn_parameters.get_dict()

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
        qb.append(Calculation, output_of='m', edge_project='label', filters={'state':calc_states.FINISHED}, edge_filters={'type':LinkType.CALL.value, 'label':{'like':'calc_%'}}, tag='c', edge_tag='mc')
        qb.append(TrajectoryData, output_of='c', project='*', tag='t')
        d = {item['mc']['label'].replace('calc_', 'trajectory_'):item['t']['*'] for item in qb.iterdict()}
        if self.inputs.moldyn_parameters.get_attr('is_hustler', False):
            return concatenate_trajectory_new_inline(store=store, **d)['concatenated_trajectory']
        else:
            return concatenate_trajectory_inline(store=store, **d)['concatenated_trajectory']
