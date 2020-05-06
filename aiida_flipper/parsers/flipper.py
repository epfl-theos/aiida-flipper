# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from aiida.orm import CalculationFactory
from six.moves import map
from six.moves import range
PwCalculation = CalculationFactory('quantumespresso.pw')
from aiida_quantumespresso.parsers.basicpw import BasicpwParser
from aiida_quantumespresso.parsers.basic_raw_parser_pw import convert_qe_time_to_sec

from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.folder import FolderData
from aiida.parsers.parser import Parser

from aiida.common.datastructures import calc_states
from aiida.common.exceptions import UniquenessError
from aiida.orm.data.array import ArrayData
from aiida.orm.data.array.kpoints import KpointsData

from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm.data.array import ArrayData
from aiida.common.utils import xyz_parser_iterator
from aiida.common.constants import bohr_to_ang, timeau_to_sec

from aiida.parsers.exceptions import OutputParsingError
from aiida.parsers.parser import Parser

#~ from aiida.parsers.plugins.quantumespresso.pw_warnings import get_warnings
import os, numpy as np, re

#~ NSTEP_REGEX = re.compile('^[ \t]*Entering Dynamics:')
NSTEP_REGEX = re.compile('Entering[ \t]+Dynamics')

WALLTIME_REGEX = re.compile(
    """
    PWSCF [ \t]* [:] [ \t]* (?P<cputime>([ 0-9]+[dhm])+) [ \t]+CPU
    [ \t]+(?P<walltime>([ 0-9]+[dhm])+) [ \t]+ WALL
    """, re.X
)

TIME_USED_REGEX = re.compile(
    'PWSCF[ \t]*\:[ \t]+(?P<cputime>[A-Za-z0-9\. ]+)[ ]+CPU[ ]+(?P<walltime>[A-Za-z0-9\. ]+)[ ]+WALL'
)
#~ POS_REGEX = re.compile("""
#~ ^                                                                             # Linestart
#~ [ \t]*                                                                        # Optional white space
#~ (?P<sym>[A-Za-z]+[A-Za-z0-9]*)\s+                                             # get the symbol
#~ (?P<x> [\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)? ) [ \t]+  # Get x
#~ (?P<y> [\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)? ) [ \t]+  # Get y
#~ (?P<z> [\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)? )         # Get z
#~ """, re.X | re.M)

POS_REGEX_3 = re.compile(
    """
^                                                                             # Linestart
[ \t]*                                                                        # Optional white space
(?P<sym>[A-Za-z]+[A-Za-z0-9]*)                                             # get the symbol
(?P<vals>(\s+ ([\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)? )){3})
""", re.X | re.M
)

POS_REGEX_12 = re.compile(
    """
^                                                                             # Linestart
[ \t]*                                                                        # Optional white space
(?P<sym>[A-Za-z]+[A-Za-z0-9]*)                                             # get the symbol
(?P<vals>(\s+ ([\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)? )){12})
""", re.X | re.M
)

POS_REGEX_15 = re.compile(
    """
^                                                                             # Linestart
[ \t]*                                                                        # Optional white space
(?P<sym>[A-Za-z]+[A-Za-z0-9]*)                                             # get the symbol
(?P<vals>(\s+ ([\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)? )){15})
""", re.X | re.M
)

#~ POS_BLOCK_REGEX = re.compile("""
#~ ([A-Za-z]+[A-Za-z0-9]*\s+([ \t]+ [\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)?){3}\s*)+
#~ """, re.X | re.M)
POS_BLOCK_REGEX = re.compile(
    """
([A-Za-z]+[A-Za-z0-9]*\s+([ \t]+ [\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)?)+\s*)+
""", re.X | re.M
)

STRESS_REGEX = re.compile(
    """
    total\s+stress.*[\n]
    (?P<vals>((\s+ ([\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)? ) ){6}[\n]){3})
    """, re.X | re.M
)

#~ total   stress  (Ry/bohr**3)                   (kbar)     P=    0.06
#~ 0.00000164   0.00000126  -0.00000083          0.24      0.19     -0.12
#~ 0.00000126  -0.00000004  -0.00000103          0.19     -0.01     -0.15
#~ -0.00000083  -0.00000103  -0.00000034         -0.12     -0.15     -0.05

F90_BOOL_DICT = {'T': True, 'F': False}


class FlipperParser(Parser):

    def parse_with_retrieved(self, retrieved):

        # from carlo.codes.kohn.pw.pwimmigrant import get_nstep, get_warnings
        successful = True

        calc_input = self._calc.inp.parameters

        # look for eventual flags of the parser
        try:
            parser_opts = self._calc.inp.settings.get_dict()[self.get_parser_settings_key()]
        except (AttributeError, KeyError):
            parser_opts = {}

        # load the input dictionary
        # TODO: pass this input_dict to the parser. It might need it.
        input_dict = self._calc.inp.parameters.get_dict()
        try:
            timestep_in_fs = 2 * timeau_to_sec * 1.0e15 * input_dict['CONTROL']['dt'] * input_dict['CONTROL'].get(
                'iprint', 1
            )
        except KeyError:
            timestep_in_fs = None
        try:
            nstep_thermo = input_dict['IONS']['nstep_thermo'] * input_dict['CONTROL'].get('iprint', 1)
        except KeyError:
            nstep_thermo = None
        try:
            temperature_thermostat = input_dict['IONS'].get('tempw', None)
        except KeyError:
            temperature_thermostat = None

        # Check that the retrieved folder is there
        try:
            out_folder = retrieved[self._calc._get_linkname_retrieved()]
        except KeyError:
            self.logger.error('No retrieved folder found')
            return False, ()

        # check what is inside the folder
        in_struc = self._calc.get_inputs_dict()['structure']
        list_of_files = out_folder.get_folder_list()

        # at least the stdout should exist
        if not self._calc._OUTPUT_FILE_NAME in list_of_files:
            self.logger.error('Standard output not found')
            successful = False
            return successful, ()
        list_of_files.remove(self._calc._OUTPUT_FILE_NAME)
        for f in ('data-file.xml', '_scheduler-stdout.txt', '_scheduler-stderr.txt'):
            try:
                list_of_files.remove(f)
            except Exception as e:
                print(e)
        evp_file = os.path.join(out_folder.get_abs_path('.'), self._calc._EVP_FILE)
        pos_file = os.path.join(out_folder.get_abs_path('.'), self._calc._POS_FILE)
        for_file = os.path.join(out_folder.get_abs_path('.'), self._calc._FOR_FILE)
        vel_file = os.path.join(out_folder.get_abs_path('.'), self._calc._VEL_FILE)

        new_nodes_list = []

        ########################## OUTPUT FILE ##################################

        out_file = os.path.join(out_folder.get_abs_path('.'), self._calc._OUTPUT_FILE_NAME)
        with open(out_file) as f:
            txt = f.read()
            #~ warnings, fatality = get_warnings(txt)

            match = TIME_USED_REGEX.search(txt)
            try:
                cputime = convert_qe_time_to_sec(match.group('cputime'))
            except:
                cputime = -1.
            try:
                walltime = convert_qe_time_to_sec(match.group('walltime'))
            except:
                walltime = -1.
            try:
                # TODO: This does not work!!
                nstep = get_nstep_from_outputf(txt)
            except:
                nstep = -1
            if input_dict['CONTROL'].get('tstress', False):
                print('reading stresses')
                stresses = list()
                iprint = input_dict['CONTROL'].get('iprint', 1)
                # I implement reading every iprint value only, to be consistent with
                # everything else being printed
                for imatch, match in enumerate(STRESS_REGEX.finditer(txt)):
                    if imatch % iprint:
                        continue
                    stress_vals = match.group('vals').split('\n')
                    stress = np.empty((3, 3))
                    for i in range(3):
                        stress[i, :] = stress_vals[i].split()[:3]
                    stresses.append(stress)
                stresses = np.array(stresses)
                print(stresses.shape)
            else:
                stresses = None

            del txt

        #~ if nstep==0:
        #~ successful = False
        #~ if warnings['MAX_CPU_TIME']:
        #~ self.logger.error("No MD steps were done in the given CPU TIME")
        #~ elif warnings['SCF_NOT_CONVERGED']:
        #~ self.logger.error("SCF did not converge")
        #~ else:
        #~ self.logger.error("No MD step for unknown reason")
        #~ return False, ()

        output_params = ParameterData(
            dict=dict(
                #~ warnings=warnings,
                cputime=cputime,
                walltime=walltime,
                nstep=nstep
            )
        )
        new_nodes_list.append((self.get_linkname_outparams(), output_params))

        ########################## TRAJECTORY #################################

        with open(evp_file) as f:
            try:
                # Using np.genfromtxt instead of np.loadtxt, because this function
                # gives NaN to a value it cannot read, and doesn't throw an error!
                scalar_quantities = np.genfromtxt(f, usecols=list(range(1, 8)))
                if scalar_quantities.shape[1] != 7:
                    raise ValueError('Bad shape detected {}'.format(scalar_quantities.shape))
                f.seek(0)
                convergence = np.genfromtxt(f, dtype='S1', usecols=(8), converters={8: (lambda s: F90_BOOL_DICT[s])})
            except (ValueError, IndexError) as e:
                # There was an error conversion, it has happened
                # that '************' appears in an evp file....
                print('There was an error: {}\nwhen reading {}'.format(e, evp_file))
                f.seek(0)
                # Getting the length of the file

                for idx, line in enumerate(f.readlines()):
                    if len(line.split()) != 9:
                        break
                # This is much slower, but the only way to things properly
                try:
                    scalar_quantities = np.empty((idx, 8))
                    convergence = np.empty(idx)
                except NameError:
                    raise OutputParsingError('Empty file {}'.format(evp_file))
                f.seek(0)
                for iline, line in enumerate(f.readlines()):
                    if (iline == idx):
                        break
                    for ival, val in enumerate(line.split()[1:8]):
                        try:
                            scalar_quantities[iline, ival] = float(val)
                        except ValueError:
                            # print line
                            scalar_quantities[iline, ival] = np.nan
                    try:
                        convergence[iline] = F90_BOOL_DICT[line.split()[8]]
                    except KeyError:
                        convergence[iline] = np.nan
                # print scalar_quantities[-1,:]
                # np.save(evp_file.replace('evp', 'npy'),scalar_quantities)

        if len(scalar_quantities) == 0:
            raise OutputParsingError('No scalar quantities in output')

        stepids = np.array(scalar_quantities[:, 0], dtype=int)
        times = scalar_quantities[:, 1]
        kinetic_energies = scalar_quantities[:, 2]
        potential_energies = scalar_quantities[:, 3]
        total_energies = scalar_quantities[:, 4]
        temperatures = scalar_quantities[:, 5]
        walltimes = scalar_quantities[:, 6]

        if input_dict['CONTROL'].get('ldecompose_ewald', False):
            pos_regex_forces = POS_REGEX_15
            ncol = 15
        elif input_dict['CONTROL'].get('ldecompose_forces', False):
            pos_regex_forces = POS_REGEX_12
            ncol = 12
        else:
            pos_regex_forces = POS_REGEX_3
            ncol = 3

        try:
            forces = get_coords_from_file(for_file, POS_BLOCK_REGEX, pos_regex_forces)
        except ValueError:
            # Had this before, that a file could not be read using the fast regular expressions
            # We therefore have to use a different function to do that:
            forces = get_coords_from_file_slow_and_steady(for_file, ncol)
        try:
            positions = get_coords_from_file(pos_file, POS_BLOCK_REGEX, POS_REGEX_3)
        except ValueError:
            positions = get_coords_from_file_slow_and_steady(pos_file, 3)

        try:
            velocities = get_coords_from_file(vel_file, POS_BLOCK_REGEX, POS_REGEX_3)
        except ValueError:
            velocities = get_coords_from_file_slow_and_steady(vel_file, 3)

        trajectory_data = TrajectoryData()

        nstep_set = set()
        nat_set = set()
        # Removed check on dimenstions( 18.10.16 since forces can now be printed with decomposition
        #~ dimensions_set = set()
        for arr in (stepids, times, kinetic_energies, potential_energies, total_energies, temperatures, walltimes):
            nstep_set.add(len(arr))
        for arr in (positions, velocities, forces):
            n1, n2, n3 = arr.shape
            nstep_set.add(n1)
            nat_set.add(n2)
            #~ dimensions_set.add(n3)
        nat = nat_set.pop()
        #~ dimensions = dimensions_set.pop()
        if nat_set:
            raise OutputParsingError(
                'Incommensurate array shapes\n'
                'read from \n{}\n{}\n{}\n'
                'forces:            {}\n'
                'positions:         {}\n'
                'velocities:        {}\n'.format(
                    pos_file, vel_file, for_file, positions.shape, velocities.shape, forces.shape
                )
            )

        # Now what if something was fucked up with the steps?
        # I think I can fix it by reducing every thing to the minumum number of states

        if len(nstep_set) > 1:
            print((
                'Warning for  {} Incommensurate array shapes\n'
                'read from \n{}\n{}\n{}\n{}\n'
                'forces:            {}\n'
                'positions:         {}\n'
                'velocities:        {}\n'
                'stepids:           {}\n'
                'times:             {}\n'
                'kinetic_energies:  {}\n'
                'potential_energies:{}\n'
                'total_energies:    {}\n'
                'temperatures:      {}\n'
                'walltimes:         {}\n'.format(
                    self._calc, evp_file, pos_file, for_file, vel_file, forces.shape, positions.shape, velocities.shape,
                    stepids.shape, times.shape, kinetic_energies.shape, potential_energies.shape, total_energies.shape,
                    temperatures.shape, walltimes.shape
                )
            ))
            nstep = min(nstep_set)
            print('Trying to fix that by setting back every array to length = {}'.format(nstep))
            positions = positions[:nstep]
            velocities = velocities[:nstep]
            forces = forces[:nstep]
            stepids = stepids[:nstep]
            times = times[:nstep]
            kinetic_energies = kinetic_energies[:nstep]
            potential_energies = potential_energies[:nstep]
            total_energies = total_energies[:nstep]
            temperatures = temperatures[:nstep]
            walltimes = walltimes[:nstep]

        else:
            nstep = nstep_set.pop()

        cells = np.array([in_struc.cell] * nstep)
        symbols = np.array([str(i.kind_name) for i in in_struc.sites[:nat]])

        # Transforming bohr to angstrom, because this is mostly what's being used in AiiDA.
        # For now, I will not transform the velocities, because there's no real concept for them in AiiDA.
        # Maybe in the future, we should have Angstrom/fs ?
        positions *= bohr_to_ang

        trajectory_data.set_trajectory(
            stepids=stepids,
            cells=cells,
            symbols=symbols,
            positions=positions,
            velocities=velocities,
        )

        trajectory_data._set_attr('atoms', in_struc.get_site_kindnames())

        if timestep_in_fs is not None:
            trajectory_data._set_attr('timestep_in_fs', timestep_in_fs)
            trajectory_data._set_attr('sim_time_fs', nstep * timestep_in_fs)
        if nstep_thermo is not None:
            trajectory_data._set_attr('nstep_thermo', nstep_thermo)
        if temperature_thermostat is not None:
            trajectory_data._set_attr('temperature_thermostat', temperature_thermostat)

        # Old: positions were stored in atomic coordinates, made conversions a bit messy,
        # and makes it hard to use some functions that suppose angstroms as units
        # trajectory_data._set_attr('units|positions','atomic')
        trajectory_data._set_attr('units|positions', 'angstrom')
        trajectory_data._set_attr('units|cells', 'angstrom')
        trajectory_data._set_attr('units|velocities', 'atomic')

        # FORCES:
        trajectory_data.set_array('forces', forces)
        trajectory_data._set_attr('units|forces', 'atomic')

        # TIMES
        trajectory_data.set_array('times', times)
        trajectory_data._set_attr('units|times', 'ps')

        # ENERGIES
        trajectory_data.set_array('kinetic_energies', kinetic_energies)
        trajectory_data._set_attr('units|kinetic_energies', 'Ry')

        trajectory_data.set_array('potential_energies', potential_energies)
        trajectory_data._set_attr('units|potential_energies', 'Ry')

        trajectory_data.set_array('total_energies', total_energies)
        trajectory_data._set_attr('units|total_energies', 'Ry')

        # TEMPERATURES
        trajectory_data.set_array('temperatures', temperatures)
        trajectory_data._set_attr('units|temperatures', 'K')

        # WALLTIMES
        trajectory_data.set_array('walltimes', walltimes)
        trajectory_data._set_attr('units|walltimes', 's')

        # SCF convergence
        trajectory_data.set_array('scf_convergence', convergence)

        # STRESSES
        if stresses is not None:
            trajectory_data._set_attr('units|stresses', 'atomic')
            trajectory_data.set_array('stresses', stresses)
        ## DONE

        new_nodes_list.append((self.get_linkname_outtrajectory(), trajectory_data))

        # comment the following if you want this check.
        # For the hustler I don't want it
        if not calc_input.dict.CONTROL.get('lhustle', False):
            for idx, arr in enumerate(
                (forces, positions, velocities, kinetic_energies, potential_energies, total_energies, temperatures)
            ):
                if np.isnan(arr).any():
                    print(('Array {} contains NAN'.format(idx)))
                    successful = False

        return successful, new_nodes_list

    def get_parser_settings_key(self):
        """
        Return the name of the key to be used in the calculation settings, that
        contains the dictionary with the parser_options
        """
        return 'parser_options'

    def get_linkname_outstructure(self):
        """
        Returns the name of the link to the output_structure
        Node exists if positions or cell changed.
        """
        return 'output_structure'

    def get_linkname_outtrajectory(self):
        """
        Returns the name of the link to the output_trajectory.
        Node exists in case of calculation='md', 'vc-md', 'relax', 'vc-relax'
        """
        return 'output_trajectory'

    def get_linkname_outarray(self):
        """
        Returns the name of the link to the output_array
        Node may exist in case of calculation='scf'
        """
        return 'output_array'

    def get_linkname_out_kpoints(self):
        """
        Returns the name of the link to the output_kpoints
        Node exists if cell has changed and no bands are stored.
        """
        return 'output_kpoints'


def get_nstep_from_outputf(text):
    return len(list(NSTEP_REGEX.finditer(text)))


def get_coords_from_file(filename, pos_block_regex, pos_regex):
    with open(filename) as f:
        coords = np.array([[
            list(map(float,
                     coord_match.group('vals').split())) for coord_match in pos_regex.finditer(match.group(0))
        ] for match in pos_block_regex.finditer(f.read())],
                          dtype=np.float64)
    return coords


def get_coords_from_file_slow_and_steady(filename, ncol):
    trajectory = []
    with open(filename) as f:
        for iline, line in enumerate(f.readlines()):
            line = line.strip()
            if line.startswith('>'):
                try:
                    trajectory.append(timestep)
                except UnboundLocalError:
                    # First step, no timestep yet
                    pass
                # We have a new timestep
                timestep = list()
            else:
                vals = line.split()[1:]  # Ignoring the first column, which is the symbol
                if len(vals) != ncol:
                    raise Exception('This line ({}) linenr {} has the wrong numbner of columns'.format(line, iline))
                else:
                    try:
                        npvals = np.empty(ncol)
                        for ival, val in enumerate(vals):
                            try:
                                npvals[ival] = float(val)
                            except ValueError:
                                npvals[ival] = np.nan
                        timestep.append(npvals)  # TODO: Exceptions...
                    except Exception as e:
                        raise Exception(
                            'An exception \n{}\noccured when '
                            'parsing line {} of \n{}'
                            ''.format(e, iline, filename)
                        )
        # Append the last timestep
        trajectory.append(timestep)
    return np.array(trajectory, dtype=np.float64)
