from aiida import orm
import functools
from aiida.common.links import LinkType
from aiida.common import AttributeDict
from aiida.engine import ToContext, append_, if_, while_, BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.engine.processes.workchains.restart import validate_handler_overrides
from aiida_flipper.workflows.replaymd import ReplayMDWorkChain
from aiida_flipper.calculations.functions import get_diffusion_from_msd, get_structure_from_trajectory, concatenate_trajectory, update_parameters_with_coefficients
from aiida_flipper.utils.utils import get_or_create_input_node
import six
from six.moves import range

class LinDiffusionWorkChain(BaseRestartWorkChain):

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        
        spec.input('diffusion_parameters',
            valid_type=orm.Dict, required=True, validator=functools.partial(validate_handler_overrides, cls),
            help='The dictionary containing all the main parameters.')
        
        spec.outline(
            cls.setup,
            cls.validate_parameters,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,)
        
        # When or how these error codes would show since I didn't define the error condition
        spec.exit_code(701, 'ERROR_MAXIMUM_STEPS_NOT_REACHED',
            message='The calculation failed before reaching the maximum no. of MD steps.')
        spec.exit_code(702, 'ERROR_DIFFUSION_NOT_CONVERGED',
            message='The calculation reached the maximum no. of MD steps, but the diffusion coefficient stil did not converge.')
        
    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.
        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(ReplayMDWorkChain, 'md'))
        # The counter counts how many REPLAYS I launched
        self.ctx.converged = False
        self.ctx.replay_counter = 0

    def validate_parameters(self):
        """Validate inputs that might depend on each other and cannot be validated by the spec.

        Also define dictionary `inputs` in the context, that will contain the inputs for the calculation that will be
        launched in the `run_calculation` step.
        """
        parameters_main = self.ctx.inputs.pop('parameters_main').get_dict()
        assert parameters_main['IONS']['ion_velocities'] == 'from_input'

        for kw in ('kpoints', 'msd_parameters', 'diffusion_parameters'):
            try:
                self.ctx.inputs.pop(kw)
            except KeyError:
                raise KeyError('Key {} is not present in inputs')
        for optkw in ('settings', 'remote_folder'):
            self.ctx.inputs.pop(optkw, None)

        if self.ctx.inputs:
            raise Exception('More keywords provided than needed: {}'.format(list(self.ctx.inputs.keys())))

    def should_run_process(self):
        """Return whether a new process should be run.

        This is the case as long as the last process has not finished successfully, the maximum number of restarts has
        not yet been exceeded, and the number of maximum replays has not been reached or diffusion coefficient converged.
        """
        return not(self.ctx.is_finished) and (self.ctx.iteration < self.inputs.max_iterations.value) and (
            (self.ctx.converged) or (self.inp.diffusion_parameters.get_dict()['max_nr_of_replays'] <= self.ctx.replay_counter))

    def _get_last_calc(self, diffusion_parameters_d=None):
        # This function needs to be defined again, maybe as a query? Probably don't even need this now 
        if diffusion_parameters_d is None:
            diffusion_parameters_d = self.ctx.inputs.diffusion_parameters.get_dict()
        return get_attribute(self.out, 'replay_{}'.format(str(self.ctx.replay_counter - 1).rjust(len(str(diffusion_parameters_d['max_nr_of_replays'])), str(0))))

    def run_process(self):
        """
        Here I restart from the thermalized run! I run NVT until I have reached the
        number of steps specified in self.inp.moldyn_parameters_NVT.dict.nstep
        First run is always run with thermalized label.
        """
        super().run_process()
        ## dont think this block is useful
        '''
        returnval = {}
        if (self.ctx.replay_counter == 0):
            recenter = inp_d['moldyn_parameters'].get_attr('recenter_before_main', False)
        else:
            recenter = False

        last_calc = self._get_last_calc(diffusion_parameters_d)
        if last_calc is None:
            inp_d['structure'] = self.inp.structure
            inp_d['settings'] = self.inp.settings
        else:
            # if trajectory and structure have different shapes (i.e. for pinball level 1 calculation on pinball positions are printed:)
            create_missing = len(self.inp.structure.sites
                                ) != last_calc.out.total_trajectory.get_attribute('array|positions')[1]
            # create missing tells inline function to append additional sites from the structure that needs to be passed in such case
            kwargs = dict(
                trajectory=last_calc.out.total_trajectory,
                parameters=get_or_create_input_node(orm.Dict,
                    dict(step_index=-1, recenter=recenter, create_settings=True, complete_missing=create_missing), store=True),)
            if create_missing:
                kwargs['structure'] = self.inp.structure
            try:
                kwargs['settings'] = self.inp.settings
            except:
                pass  # settings will be None

            inlinec, res = get_structure_from_trajectory(**kwargs)
            returnval['get_structure'] = inlinec
            inp_d['settings'] = res['settings']
            inp_d['structure'] = res['structure']
        '''
        diffusion_parameters_d = self.inp.diffusion_parameters.get_dict()
        repl = ReplayMDWorkChain(**diffusion_parameters_d)
        if (self.ctx.replay_counter == 0): 
            repl.label = '{}{}thermalize'.format(self.label, '-' if self.label else '')
        else:
            repl.label = '{}{}replay-{}'.format(self.label, '-' if self.label else '', self.ctx.replay_counter)
        
        # returnval['replay_{}'.format(str(self.ctx.replay_counter).rjust(len(str(diffusion_parameters_d['max_nr_of_replays'])), str(0)))] = repl

        self.ctx.is_finished = False
        self.ctx.replay_counter += 1

        return ToContext(children=append_(node))

    def inspect_process(self):

        super().inspect_process()

        diffusion_parameters_d = self.inp.diffusion_parameters.get_dict()
        msd_parameters = self.inp.msd_parameters

        concatenated_trajectory = concatenate_trajectory(**self._get_trajectories())['concatenated_trajectory']
        # Estimate the diffusion coefficients without storing, to check convergence
        msd_results = get_diffusion_from_msd(structure=self.inputs.structure, parameters=msd_parameters, trajectory=concatenated_trajectory)
        # sem is standard error = sigma/sq_root(no.)
        sem = msd_results.get_attribute('{}'.format(msd_parameters.dict.species_of_interest[0]))['diffusion_sem_cm2_s']
        mean_d = msd_results.get_attribute('{}'.format(msd_parameters.dict.species_of_interest[0]))['diffusion_mean_cm2_s']
        sem_relative = sem / mean_d
        sem_target = diffusion_parameters_d['sem_threshold']
        sem_relative_target = diffusion_parameters_d['sem_relative_threshold']
        if (mean_d < 0.):
            # the diffusion is negative: means that the value is not converged enough yet
            self.report('The Diffusion coefficient ( {} +/- {} ) is negative, i.e. not converged.'.format(mean_d, sem))
            self.goto(self.run_replays)
        elif (sem < sem_target):
            # This means that the  standard error of the mean in my diffusion coefficient is below the target accuracy
            self.report('Converged! The error ( {} ) is below the target value ( {} )'.format(sem, sem_target))
            self.ctx.converged = True
        elif (sem_relative < sem_relative_target):
            # the relative error is below my targe value
            self.report('Converged! The relative error ( {} ) is below the target value ( {} )'.format(sem_relative, sem_relative_target))
            self.ctx.converged = True
        else:
            self.report('The error has not converged')
            self.report('absolute sem: {:.5e}  Target: {:.5e}'.format(sem, sem_target))
            self.report('relative sem: {:.5e}  Target: {:.5e}'.format(sem_relative, sem_relative_target))

        return None

    def results(self):
        super().results()
        msd_parameters = self.inputs.msd_parameters
        concatenated_trajectory = concatenate_trajectory(**self._get_trajectories())['concatenated_trajectory']
        res = get_diffusion_from_msd(structure=self.inputs.structure, parameters=msd_parameters, trajectory=concatenated_trajectory)
        self.out('msd_results', res)
        return None

    def _get_trajectories(self):
        qb = orm.QueryBuilder()
        qb.append(LinDiffusionWorkChain, filters={'id': self.id}, tag='ldc')
        qb.append(ReplayMDWorkChain, output_of='ldc', edge_project='label', edge_filters={'type': LinkType.CALL.value,
                'label': {'like': 'replay_%'}}, tag='repl', edge_tag='mb')
        qb.append(orm.TrajectoryData, output_of='repl', edge_filters={'type': LinkType.RETURN.value, 'label': 'total_trajectory'}, project='*', tag='t')
        return {'{}'.format(item['mb']['label']): item['t']['*'] for item in qb.iterdict()}


class ConvergeDiffusionWorkChain(BaseRestartWorkChain):
    _diff_name = 'diff'
    _fit_name = 'fit'

    def _get_last_calcs(self, link_name, nr_of_calcs=None, diffusion_convergence_parameters_d=None):
        """
        Get the N last diffusion calculations, where N is given by the integer nr_of_calcs:
        """
        if diffusion_convergence_parameters_d is None:
            diffusion_convergence_parameters_d = self.inputs.diffusion_convergence_parameters.get_dict()
        current_counter = self.ctx.diff_counter
        if nr_of_calcs is None:
            start = 1
        else:
            start = current_counter - nr_of_calcs + 1
            if start < 1:
                raise ValueError('You asked for more calculations than there are')
        res = []
        for idx in range(start, current_counter + 1):
            res.append(get_attribute(self.out, '{}_{}'.format(link_name, str(idx).rjust(len(str(diffusion_convergence_parameters_d['max_iterations'])), str(0)))))
        return res

    def _get_last_diffs(self, nr_of_calcs=None, diffusion_convergence_parameters_d=None):
        return self._get_last_calcs(self._diff_name, nr_of_calcs=nr_of_calcs,
            diffusion_convergence_parameters_d=diffusion_convergence_parameters_d)

    def _get_last_fits(self, nr_of_calcs=None, diffusion_convergence_parameters_d=None):
        return self._get_last_calcs(self._fit_name, nr_of_calcs=nr_of_calcs,
            diffusion_convergence_parameters_d=diffusion_convergence_parameters_d)

    def start(self):
        # Now, I start by checking that I have all the parameters I need
        # Don't need to check to much because the BranchingCalculations will validate
        # most of the parameters!
        inp_d = self.get_inputs_dict()
        for k, v in inp_d.items():
            # This is a top level workflow, but if it is was called by another, I remove the calls:
            if isinstance(v, orm.CalculationNode):
                inp_d.pop(k)

        structure = inp_d.pop('structure')
        for kind in structure.kinds:
            inp_d.pop('pseudo_{}'.format(kind.name))
            inp_d.pop('pseudo_{}_flipper'.format(kind.name), None)

        # The code label has to be set as an attribute, and can be changed during the dynamics, but not really required
        # Code.get_from_string(self.ctx.code_string)
        self.ctx.num_machines
        self.ctx.walltime_seconds

        for required_kw in ('moldyn_parameters_main', 'parameters_main', 'msd_parameters', 'diffusion_parameters', 'parameters_fitting',
            'parameters_fitting_dft', 'parameters_fitting_flipper', 'structure_fitting_parameters', 'hustler_code', 'kpoints'):
            if required_kw not in inp_d:
                raise KeyError('Input requires value with keyword {}'.format(required_kw))
            inp_d.pop(required_kw)

        for optional_kw in ('remote_folder', 'settings'): ## these are not supported yet
            inp_d.pop(optional_kw, None)

        diffusion_convergence_parameters_d = inp_d.pop('diffusion_convergence_parameters').get_dict()
        try:
            maxiter = diffusion_convergence_parameters_d['max_iterations']
            if not isinstance(maxiter, int):
                raise TypeError('max_iterations needs to be an integer')
            if maxiter < 1:
                raise ValueError('max_iterations needs to be a positive integer')
        except KeyError:
            raise KeyError('Keyword max_iterations not included in diffusion parameters')
        try:
            miniter = diffusion_convergence_parameters_d['min_iterations']
            if not isinstance(miniter, int):
                raise TypeError('min_iterations needs to be an integer')
            if miniter < 3:
                raise ValueError('min_iterations needs to be larger than 2')
            if miniter > maxiter:
                raise ValueError('max_iterations has to be larger or equal to min_iterations')
        except KeyError:
            raise KeyError('Keyword min_iterations not included in diffusion convergence parameters')

        for key, typ in (('species', six.string_types),):
            if key not in diffusion_convergence_parameters_d:
                raise KeyError('Key {} has to be in diffusion convergence parameters')
            if not isinstance(diffusion_convergence_parameters_d[key], typ):
                raise TypeError('Key {} has the wrong type ({} {}) as value'.format(key, diffusion_convergence_parameters_d[key], type(diffusion_convergence_parameters_d[key])))

        if inp_d:
            raise Exception('More keywords provided than needed: {}'.format(list(inp_d.keys())))

        # The replay Counter counts how many REPLAYS I launched
        self.ctx.diff_counter = 0
        self.ctx.converged = False
        self.goto(self.run_estimates)
        # check if we should start by performing a fit over an old trajectory
        try:
            first_fit_trajectory = self.inp.first_fit_trajectory
        except AttributeError:
            first_fit_trajectory = None
        if first_fit_trajectory:
            self.goto(self.run_fit)
        else:
            self.goto(self.run_estimates)
        
    def run_preprocess(self):
        """
        Runs the PreProcessStructureWorkChain to stash the charge densities of host lattice.
        This is the first workchain that this class must run if charge densities are not found
        """
        returndict = {}
        return returndict

    def run_estimates(self):
        """
        Runs a LinDiffusionWorkChain for an estimate of the diffusion.
        If there is a last fitting estimate, I update the parameters for the pinball.
        """
        inp_d = self.get_inputs_dict()
        for k, v in inp_d.items():
            # This is a top level workflow, but if it is was called by another, I remove the calls:
            if isinstance(v, orm.CalculationNode):
                inp_d.pop(k)
        diffusion_convergence_parameters_d = inp_d.pop('diffusion_convergence_parameters').get_dict()

        # the dictionary for the inputs to the diffusion workflow, first UPF:
        lindiff_inp = {k: v for k, v in inp_d.items() if isinstance(v, orm.UpfData)}
        lindiff_inp['pseudo_Li'] = lindiff_inp.pop('pseudo_Li_flipper')

        # now all the required keywords that have same name as for self:
        for required_kw in ('structure', 'kpoints', 'moldyn_parameters_main', 'diffusion_parameters', 'msd_parameters'):
            lindiff_inp[required_kw] = inp_d[required_kw]
        for optional_kw in ('settings', 'remote_folder'):
            if optional_kw in inp_d:
                lindiff_inp[optional_kw] = inp_d[optional_kw]

        returndict = {}
        if self.ctx.diff_counter:
            coefs = self._get_last_fits(
                nr_of_calcs=1, diffusion_convergence_parameters_d=diffusion_convergence_parameters_d
            )[0].out.coefficients
            c, res = update_parameters_with_coefficients(parameters=inp_d['parameters_main'], coefficients=coefs)
            returndict['update_parameters_{}'.format(
                str(self.ctx.diff_counter
                   ).rjust(len(str(diffusion_convergence_parameters_d['max_iterations'])), str(0))
            )] = c
            lindiff_inp['parameters_main'] = res['updated_parameters']
        else:
            # In case there was no fitting done. I don't set anything in case
            # the user already gave a good guess of what the parameters are
            lindiff_inp['parameters_main'] = inp_d['parameters_main']

        if self.ctx.diff_counter < 3:
            diffusion_parameters_d = lindiff_inp['diffusion_parameters'].get_dict()
            diffusion_parameters_d['max_nr_of_replays'
                                  ] = 1  # setting just one replay calculation in the first 2 iterations
            # to reduce total simulation time.
            lindiff_inp['diffusion_parameters'] = get_or_create_input_node(diffusion_parameters_d, store=True)

        diff = LinDiffusionWorkChain(**lindiff_inp)
        diff.label = '{}{}diff-{}'.format(self.label, '-' if self.label else '', self.ctx.diff_counter)
        for attr_key in ('num_machines', 'walltime_seconds', 'code_string'):
            diff.set_attribute(attr_key, self.get_attribute(attr_key))
        if self.get_attr('num_mpiprocs_per_machine', 0):
            diff._set_attr('num_mpiprocs_per_machine', self.get_attr('num_mpiprocs_per_machine'))

        self.goto(self.check)
        self.ctx.diff_counter += 1
        returndict['{}_{}'.format(
            self._diff_name,
            str(self.ctx.diff_counter).rjust(len(str(diffusion_convergence_parameters_d['max_iterations'])), str(0))
        )] = diff
        return returndict

    def run_fit(self):
        """
        Runs a fitting workflow on positions taken from the last trajectory
        """
        from .fitting import get_configurations_from_trajectories_inline, FittingFromTrajectoryCalculation

        diffusion_convergence_parameters_d = self.inputs.diffusion_convergence_parameters.get_dict()
        
        # if first_fit_trajectory was specified, use it to perform an initial fit
        try:
            first_fit_trajectory = self.inp.first_fit_trajectory
        except AttributeError:
            first_fit_trajectory = None
        if self.ctx.diff_counter == 0 and first_fit_trajectory:
            trajectory = first_fit_trajectory
        else:
            lastcalc = self._get_last_diffs(diffusion_convergence_parameters_d=diffusion_convergence_parameters_d, nr_of_calcs=1)[0]
            trajectory = lastcalc.out.concatenated_trajectory

        # Since I have not converged I need to run another fit calculation:
        returndict = {}
        # I need to launch a fitting calculation based on positions from the last diffusion estimate trajectory:
        calc, res = get_configurations_from_trajectories_inline(
            parameters=self.inp.structure_fitting_parameters,
            structure=self.inp.structure,
            trajectory=lastcalc.out.concatenated_trajectory)
        returndict['get_configurations_{}'.format(str(self.ctx.diff_counter).rjust(len(str(diffusion_convergence_parameters_d['max_iterations'])), str(0)))] = calc
        pseudos = {k: v for k, v in self.get_inputs_dict().items() if isinstance(v, orm.UpfData)}
        fit = FittingFromTrajectoryCalculation(
            structure=self.inp.structure,
            remote_folder_flipper=self.inp.remote_folder,
            positions=res['positions'],
            parameters=self.inp.parameters_fitting,
            parameters_dft=self.inp.parameters_fitting_dft,
            parameters_flipper=self.inp.parameters_fitting_flipper,
            code=self.inp.hustler_code,
            kpoints=self.inp.kpoints,
            settings=self.inp.settings,
            **pseudos
        )
        returndict['{}_{}'.format(self._fit_name, str(self.ctx.diff_counter).rjust(len(str(diffusion_convergence_parameters_d['max_iterations'])), str(0)))] = fit
        self.goto(self.run_estimates)
        return returndict

    def check(self):
        import numpy as np
        diffusion_convergence_parameters_d = self.inputs.diffusion_convergence_parameters.get_dict()
        lastcalc = self._get_last_diffs(diffusion_convergence_parameters_d=diffusion_convergence_parameters_d, nr_of_calcs=1)[0]

        if diffusion_convergence_parameters_d['max_iterations'] <= self.ctx.diff_counter:
            print('Cannot run more')
            self.goto(self.collect)
        elif lastcalc.get_state() == 'FAILED':
            raise Exception(
                'Last diffusion {} failed with message:\n{}'.format(lastcalc, lastcalc.get_attribute('fail_msg'))
            )

        elif diffusion_convergence_parameters_d['min_iterations'] > self.ctx.diff_counter:
            # just launch the next!
            print('Did not run enough')
            self.goto(self.run_fit)
        else:
            # Since I am here, it means I need to check the last 3 calculations to
            # see whether I converged or need to run again:
            # Now let me see the diffusion coefficient that I get and if it's converged
            # I consider it converged if the last 3 estimates have not changed more than the threshold
            # In case min_iterations == 2, I just use the last 2 calculations
            if diffusion_convergence_parameters_d['min_iterations'] == 2:
                last_diff_calculations = self._get_last_diffs(diffusion_convergence_parameters_d=diffusion_convergence_parameters_d, nr_of_calcs=2)
            else:
                last_diff_calculations = self._get_last_diffs(diffusion_convergence_parameters_d=diffusion_convergence_parameters_d, nr_of_calcs=3)

            diffusions = np.array([
                calc.out.msd_results.get_attribute(diffusion_convergence_parameters_d['species'])['diffusion_mean_cm2_s']
                for calc in last_diff_calculations
            ])
            if diffusions.std() < diffusion_convergence_parameters_d['diffusion_thr_cm2_s']:
                # all good, I have converged!
                print('Diffusion converged (std = {} < threshold = {})'.format(diffusions.std(), diffusion_convergence_parameters_d['diffusion_thr_cm2_s']))
                self.ctx.converged = True
                self.goto(self.collect)
            elif (
                'diffusion_thr_cm2_s_rel' in diffusion_convergence_parameters_d and
                abs(diffusions.mean()) > 1e-12 and  # avoid division by 0
                abs(diffusions.std() / diffusions.mean()
                   ) < diffusion_convergence_parameters_d['diffusion_thr_cm2_s_rel']
            ):
                # Checked relative convergence by dividing the standard deviation by the mean
                print('Diffusion converged (std = {} < threshold = {})'.format(diffusions.std(), diffusion_convergence_parameters_d['diffusion_thr_cm2_s']))
                self.ctx.converged = True
                self.got(self.collect)
            else:
                self.goto(self.run_fit)

    def collect(self):
        last_calc = self._get_last_diffs(nr_of_calcs=1)[-1]
        self.goto(self.exit)
        return {'converged_msd_results': last_calc.out.msd_results}
