# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

from six.moves import map
from six.moves import range

import os, numpy as np, re

from aiida import orm

# ~ from aiida_quantumespresso.parsers.basic_raw_parser_pw import convert_qe_time_to_sec

# ~ from aiida.orm.data.parameter import ParameterData
# ~ from aiida.orm.data.folder import FolderData
# ~ from aiida.parsers.parser import Parser

# ~ from aiida.common.datastructures import calc_states
# ~ from aiida.common.exceptions import UniquenessError
# ~ from aiida.orm.data.array import ArrayData
# ~ from aiida.orm.data.array.kpoints import KpointsData

# ~ from aiida.orm.data.array.trajectory import TrajectoryData
# ~ from aiida.orm.data.array import ArrayData
# ~ from aiida.common.utils import xyz_parser_iterator

# ~ from aiida.parsers.exceptions import OutputParsingError
# ~ from aiida.parsers.parser import Parser

#~ from aiida.parsers.plugins.quantumespresso.pw_warnings import get_warnings
from aiida_quantumespresso.parsers.pw import PwParser
from aiida_flipper.calculations.flipper import FlipperCalculation
from qe_tools.constants import bohr_to_ang, timeau_to_sec


WALLTIME_REGEX = re.compile(
    """
    PWSCF [ \t]* [:] [ \t]* (?P<cputime>([ 0-9]+[dhm])+) [ \t]+CPU
    [ \t]+(?P<walltime>([ 0-9]+[dhm])+) [ \t]+ WALL
    """, re.X
)

TIME_USED_REGEX = re.compile(
    'PWSCF[ \t]*\:[ \t]+(?P<cputime>[A-Za-z0-9\. ]+)[ ]+CPU[ ]+(?P<walltime>[A-Za-z0-9\. ]+)[ ]+WALL'
)


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

# T, F for python2, b'T', b'F' for python3
F90_BOOL_DICT = {'T': True, 'F': False, b'T': True, b'F': False}


class FlipperParser(PwParser):

    def parse(self, **kwargs):


        # Check that the retrieved folder is there
        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit(self.exit(self.exit_codes.ERROR_NO_RETRIEVED_FOLDER))


        calc_input_dict = self.node.inputs.parameters.get_dict()
        input_dict = self.node.inputs.parameters.get_dict()
        try:
            # This referes to the SAMPLING timestep for the trajectory:
            timestep_in_fs = timeau_to_sec * 2e15 * input_dict['CONTROL']['dt'] * input_dict['CONTROL'].get(
                'iprint', 1)
        except KeyError:
            return self.exit(self.exit_codes.ERROR_UNKNOWN_TIMESTEP)

        in_struc = self.node.inputs.structure
        print(self.node.get_retrieve_temporary_list())

        list_of_files = self.retrieved.list_object_names()
        # the stdout should exist
        filename_stdout = self.node.get_attribute('output_filename')
        if filename_stdout not in list_of_files:
             return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING


        for f in ('data-file.xml', '_scheduler-stdout.txt', '_scheduler-stderr.txt'):
            if f in list_of_files:
                list_of_files.remove(f)

        try:
            temp_folder = kwargs['retrieved_temporary_folder']
        except KeyError:
            return self.exit(self.exit_codes.ERROR_NO_RETRIEVED_TEMPORARY_FOLDER)
        print(self.node)
        evp_file = os.path.join(temp_folder, FlipperCalculation._EVP_FILE)
        pos_file = os.path.join(temp_folder, FlipperCalculation._POS_FILE)
        for_file = os.path.join(temp_folder, FlipperCalculation._FOR_FILE)
        vel_file = os.path.join(temp_folder, FlipperCalculation._VEL_FILE)

        new_nodes_list = []
        ########################## OUTPUT FILE ##################################
        stdout_txt = self.retrieved.get_object_content(filename_stdout)

        match = TIME_USED_REGEX.search(stdout_txt)
        try:
            cputime = convert_qe_time_to_sec(match.group('cputime'))
        except:
            cputime = -1.
        try:
            walltime = convert_qe_time_to_sec(match.group('walltime'))
        except:
            walltime = -1.
        if input_dict['CONTROL'].get('tstress', False):
            stresses = list()
            iprint = input_dict['CONTROL'].get('iprint', 1)
            # I implement reading every iprint value only, to be consistent with
            # everything else being printed
            for imatch, match in enumerate(STRESS_REGEX.finditer(stdout_txt)):
                if imatch % iprint:
                    continue
                stress_vals = match.group('vals').split('\n')
                stress = np.empty((3, 3))
                for i in range(3):
                    stress[i, :] = stress_vals[i].split()[:3]
                stresses.append(stress)
            stresses = np.array(stresses)
        else:
            stresses = None
        del stdout_txt

        self.out('output_parameters', orm.Dict(
            dict=dict(cputime=cputime, walltime=walltime,)))
        
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
                f.seek(0)
                # Getting the length of the file

                for idx, line in enumerate(f.readlines()):
                    if len(line.split()) != 9:
                        break
                # This is much slower, but the only way to do things properly
                try:
                    scalar_quantities = np.empty((idx, 8))
                    convergence = np.empty(idx)
                except NameError:
                    return self.exit_codes.ERROR_EMPTY_TRAJECTORY_FILES
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

        if len(scalar_quantities) == 0:
            return self.exit_codes.ERROR_CORRUPTED_TRAJECTORY_FILES

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


        nstep_set = set()
        nat_set = set()

        for arr in (stepids, times, kinetic_energies, potential_energies, total_energies, temperatures, walltimes):
            nstep_set.add(len(arr))

        for arr in (positions, velocities, forces):
            n1, n2, _ = arr.shape
            nstep_set.add(n1)
            nat_set.add(n2)
        nat = nat_set.pop()
        nstep = nstep_set.pop()
        #~ dimensions = dimensions_set.pop()
        if nat_set or nstep_set:
            return self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSIONS

        # cells = np.array([in_struc.cell] * nstep)
        symbols = np.array([str(i.kind_name) for i in in_struc.sites[:nat]])

        # Transforming bohr to angstrom, because this is mostly what's being used in AiiDA.
        # For now, I will not transform the velocities, because there's no real concept for them in AiiDA.
        # Maybe in the future, we should have Angstrom/fs ?
        positions *= bohr_to_ang

        trajectory_data = orm.TrajectoryData()

        trajectory_data.set_trajectory(
            stepids=stepids,
            # cells=cells,
            symbols=symbols,
            positions=positions,
            velocities=velocities,
        )

        trajectory_data.set_attribute('atoms', symbols)
        trajectory_data.set_attribute('timestep_in_fs', timestep_in_fs)

        # Old: positions were stored in atomic coordinates, made conversions a bit messy,
        # and makes it hard to use some functions that suppose angstroms as units
        # trajectory_data._set_attr('units|positions','atomic')
        trajectory_data.set_attribute('units|positions', 'angstrom')
        # trajectory_data.set_attribute('units|cells', 'angstrom')
        trajectory_data.set_attribute('units|velocities', 'atomic')

        # FORCES:
        trajectory_data.set_array('forces', forces)
        trajectory_data.set_attribute('units|forces', 'atomic')

        # TIMES
        trajectory_data.set_array('times', times)
        trajectory_data.set_attribute('units|times', 'ps')

        # ENERGIES
        trajectory_data.set_array('kinetic_energies', kinetic_energies)
        trajectory_data.set_attribute('units|kinetic_energies', 'Ry')

        trajectory_data.set_array('potential_energies', potential_energies)
        trajectory_data.set_attribute('units|potential_energies', 'Ry')

        trajectory_data.set_array('total_energies', total_energies)
        trajectory_data.set_attribute('units|total_energies', 'Ry')

        # TEMPERATURES
        trajectory_data.set_array('temperatures', temperatures)
        trajectory_data.set_attribute('units|temperatures', 'K')

        # WALLTIMES
        trajectory_data.set_array('walltimes', walltimes)
        trajectory_data.set_attribute('units|walltimes', 's')

        # SCF convergence
        trajectory_data.set_array('scf_convergence', convergence)

        # STRESSES
        if stresses is not None:
            trajectory_data.set_array('stresses', stresses)
            trajectory_data.set_attribute('units|stresses', 'atomic')

        if not calc_input_dict['CONTROL'].get('lhustle', False):
            for idx, arr in enumerate(
                (forces, positions, velocities, kinetic_energies, potential_energies, total_energies, temperatures)):
                if np.isnan(arr).any():
                    return self.exit_codes.ERROR_TRAJECTORY_WITH_NAN
        self.out('output_trajectory', trajectory_data)



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
