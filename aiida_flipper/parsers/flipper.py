# -*- coding: utf-8 -*-

import os, re
import numpy as np

from aiida import orm
from aiida.common import exceptions
from aiida_quantumespresso.parsers.pw import PwParser
from aiida_flipper.calculations.flipper import FlipperCalculation
bohr_to_ang = 0.52917720859
timeau_to_sec = 2.418884254E-17


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

        self.exit_code_stdout = None

        # Check that the retrieved folder is there
        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit(self.exit(self.exit_codes.ERROR_NO_RETRIEVED_FOLDER))

        calc_input_dict = self.node.inputs.parameters.get_dict()
        input_dict = self.node.inputs.parameters.get_dict()

        raise_if_nan_in_values = not (calc_input_dict['CONTROL'].get('lhustle', False))
        # If ******* occurs (i.e. value not printed), I will raise immediately if this
        # flag is set to True. Before, this was checked at the very end, which wastes computer time.
        if raise_if_nan_in_values:
            try:
                # This refers to the SAMPLING timestep for the trajectory:
                timestep_in_fs = timeau_to_sec * 2e15 * input_dict['CONTROL']['dt'] * input_dict['CONTROL'].get('iprint', 1)
            except KeyError:
                return self.exit(self.exit_codes.ERROR_UNKNOWN_TIMESTEP)
        else: timestep_in_fs = 1

        in_struc = self.node.inputs.structure

        list_of_files = self.retrieved.list_object_names()
        # the stdout should exist
        filename_stdout = self.node.get_attribute('output_filename') #this is aiida.out folder
        if filename_stdout not in list_of_files:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        for f in ('data-file.xml', '_scheduler-stdout.txt', '_scheduler-stderr.txt'):
            if f in list_of_files:
                list_of_files.remove(f)

        try:
            temp_folder = kwargs['retrieved_temporary_folder']
        except KeyError:
            return self.exit(self.exit_codes.ERROR_NO_RETRIEVED_TEMPORARY_FOLDER)

        evp_file = os.path.join(temp_folder, FlipperCalculation._EVP_FILE)
        pos_file = os.path.join(temp_folder, FlipperCalculation._POS_FILE)
        for_file = os.path.join(temp_folder, FlipperCalculation._FOR_FILE)
        vel_file = os.path.join(temp_folder, FlipperCalculation._VEL_FILE)
        for filename in (evp_file, pos_file, for_file, vel_file):
            # Checking if files exists here.
            # Check whether files have meaningful information is done later...
            if not os.path.exists(filename):
                return self.exit_codes.ERROR_MISSING_TRAJECTORY_FILES

        ########################## OUTPUT FILE ##################################
        if input_dict['CONTROL'].get('tstress', False):
            stdout_txt = self.retrieved.get_object_content(filename_stdout)
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
            del stdout_txt
        else:
            stresses = None

        parsed_stdout, logs_stdout = self.parse_stdout(input_dict, parser_options=None, parsed_xml=None)
        parsed_stdout.pop('trajectory', None)
        parsed_stdout.pop('structure', None)

        self.out('output_parameters', orm.Dict(dict=parsed_stdout))

        ignore = ['Error while parsing ethr.', 'DEPRECATED: symmetry with ibrav=0, use correct ibrav instead']
        self.emit_logs(logs_stdout, ignore=ignore)

        ########################## EVP FILE #################################
        with open(evp_file) as f:
            try:
                # I put this variable here to check later if any indices need to be deleted
                indices_to_delete = []
                # Using np.genfromtxt instead of np.loadtxt, because this function
                # gives NaN to a value it cannot read, and doesn't throw an error!
                # reading the first 8 columns for verlet.evp
                # It also jumps over empty lines, which is good.
                scalar_quantities = np.genfromtxt(f, usecols=list(range(1, 8)))
                if scalar_quantities.shape[1] != 7:
                    return self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSIONS_2
                if raise_if_nan_in_values and np.isnan(scalar_quantities).any():
                    for index, nan_rows in enumerate(np.isnan(scalar_quantities)):
                        if nan_rows.any():
                            indices_to_delete.append(index)
                f.seek(0)
                # reading the last column of the verlet.evp
                try:
                    convergence = np.genfromtxt(
                        f, dtype='S1', usecols=(8), converters={8: (lambda s: F90_BOOL_DICT[s])}
                    )  # Raise KeyError if neither F nor T
                    # in 8th column
                    # raise IndexError if column does not exist
                except IndexError:
                    return self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSIONS_2
                # TODO is there a function that can do this in one go?
            except (ValueError, IndexError, KeyError) as e:
                # There was an error conversion
                # that '************' appears in an evp file is NOT the problem, since that
                # would have been dealt by np.genfromtext

                # I first check number of lines in text file:
                f.seek(0)  # back to line 0
                # Getting the length of the file
                evp_data = []
                for idx, line in enumerate(f.readlines()):
                    data_this_line = line.split()[1:]
                    if len(data_this_line) == 0:
                        pass  # skipping empty lines
                    elif len(data_this_line) == 8:
                        evp_data.append(data_this_line)
                    else:
                        return self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSIONS_2

                # Check if this is an empty file,
                # Maybe check before if file is empty?
                if len(evp_data) == 0:
                    return self.exit_codes.ERROR_EMPTY_TRAJECTORY_FILES

                # I go through the data line by line, which is much slower, but the only way to do things properly.
                # Since the only way I can get here (I think) is if the last column of EVP file is not readable,
                # this might be replaced with a simple check if raise_if_nan_in_values before
                # going through it. For now better safe than sorry
                scalar_quantities = np.empty((len(evp_data), 8))
                convergence = np.empty(len(evp_data), dtype='S1')
                for iline, data_this_line in enumerate(evp_data):
                    for ival, val in enumerate(data_this_line[:7]):
                        try:
                            scalar_quantities[iline, ival] = float(val)
                        except ValueError:
                            if raise_if_nan_in_values:
                                return self.exit_codes.ERROR_TRAJECTORY_WITH_NAN
                            else:
                                scalar_quantities[iline, ival] = np.nan
                    try:
                        convergence[iline] = F90_BOOL_DICT[line.split()[7]]
                    except KeyError:
                        if raise_if_nan_in_values:
                            return self.exit_codes.ERROR_TRAJECTORY_WITH_NAN
                        else:
                            convergence[iline] = np.nan

        if len(scalar_quantities) == 0 or len(convergence) == 0:
            return self.exit_codes.ERROR_CORRUPTED_TRAJECTORY_FILES

        #################################### FORCES ###########################################
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
            # A file could not be read using the fast regular expressions, because it contains
            # non-numerical values or because there too many or too few values
            # in line.
            forces, exit_code = get_coords_from_file_slow_and_steady(
                for_file,
                ncol,
                raise_if_nan_in_values,
                self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_1,
                self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_2,
                self.exit_codes.ERROR_TRAJECTORY_WITH_NAN,
            )
            if exit_code:
                return exit_code

        #################################### POSITIONS ###########################################
        try:
            positions = get_coords_from_file(pos_file, POS_BLOCK_REGEX, POS_REGEX_3)
        except ValueError:
            positions, exit_code = get_coords_from_file_slow_and_steady(
                pos_file,
                3,
                raise_if_nan_in_values,
                self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_1,
                self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_2,
                self.exit_codes.ERROR_TRAJECTORY_WITH_NAN,
            )
            if exit_code:
                return exit_code

        #################################### VELOCITIES ###########################################
        try:
            velocities = get_coords_from_file(vel_file, POS_BLOCK_REGEX, POS_REGEX_3)
        except ValueError:
            velocities, exit_code = get_coords_from_file_slow_and_steady(
                vel_file,
                3,
                raise_if_nan_in_values,
                self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_1,
                self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_2,
                self.exit_codes.ERROR_TRAJECTORY_WITH_NAN,
            )
            if exit_code:
                return exit_code

        #################################### CHECK ARRAYS ###########################################
        # If there are nan values in evp files I drop them from all arrays
        if indices_to_delete:
            scalar_quantities = np.delete(scalar_quantities, indices_to_delete, axis=0)
            convergence = np.delete(convergence, indices_to_delete, axis=0)
            positions = np.delete(positions, indices_to_delete, axis=0)
            velocities = np.delete(velocities, indices_to_delete, axis=0)
            forces = np.delete(forces, indices_to_delete, axis=0)

        # I take an inelegant/hacky approach to make eveything the same size
        n = min(len(scalar_quantities), len(convergence), len(positions), len(velocities), len(forces))
        scalar_quantities = scalar_quantities[:n]
        convergence = convergence[:n]
        positions = positions[:n]
        velocities = velocities[:n]
        forces = forces[:n]
        
        nstep_set = set()
        nat_set = set()

        # All arrays have to have the same length:
        for arr in (scalar_quantities, convergence):
            nstep_set.add(len(arr))
        # In addition, positions, velocities and forces have to have the same number of atoms
        for arr in (positions, velocities, forces):
            n1, n2, _ = arr.shape
            nstep_set.add(n1)
            nat_set.add(n2)
        # These sets should contain one element therefore be empty when I pop that element:
        nstep = nstep_set.pop()
        if nstep_set:
            return self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_0

        nat = nat_set.pop()
        if nat_set:
            return self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_1

        cells = np.array([in_struc.cell] * nstep)
        symbols = np.array([str(i.kind_name) for i in in_struc.sites[:nat]])

        # Transforming bohr to angstrom, because this is mostly what's being used in AiiDA.
        # For now, I will not transform the velocities, because there's no real concept for them in AiiDA.
        # Maybe in the future, we should have Angstrom/fs ?
        positions *= bohr_to_ang

        # Produce trajectory
        trajectory_data = orm.TrajectoryData()
        trajectory_data.set_trajectory(
            stepids=np.array(scalar_quantities[:, 0], dtype=int),
            cells=cells,  # flipper always runs NVT, but need to standardise with aiida
            symbols=symbols,
            positions=positions,
            velocities=velocities
        )

        trajectory_data.set_attribute('atoms', symbols)
        trajectory_data.set_attribute('timestep_in_fs', timestep_in_fs)

        # Old: positions were stored in atomic coordinates, made conversions a bit messy,
        # and makes it hard to use some functions that suppose angstroms as units
        trajectory_data.set_attribute('units|positions', 'angstrom')
        trajectory_data.set_attribute('units|cells', 'angstrom')    
        trajectory_data.set_attribute('units|velocities', 'atomic')

        # FORCES:
        trajectory_data.set_array('forces', forces)
        trajectory_data.set_attribute('units|forces', 'atomic')

        # TIMES # is this necessary, these are easy to recreate?
        trajectory_data.set_array('times', scalar_quantities[:, 1])
        # do we want to change the time to fs to make it consistent with samos?
        # trajectory_data.set_array('times', 1000 * scalar_quantities[:, 1])
        # trajectory_data.set_attribute('units|times', 'fs')
        trajectory_data.set_attribute('units|times', 'ps')
        trajectory_data.set_attribute('sim_time_fs', 1000 * (scalar_quantities[-1, 1] - scalar_quantities[0, 1]))

        # ENERGIES
        trajectory_data.set_array('kinetic_energies', scalar_quantities[:, 2])
        trajectory_data.set_attribute('units|kinetic_energies', 'Ry')

        trajectory_data.set_array('potential_energies', scalar_quantities[:, 3])
        trajectory_data.set_attribute('units|potential_energies', 'Ry')

        trajectory_data.set_array('total_energies', scalar_quantities[:, 4])
        trajectory_data.set_attribute('units|total_energies', 'Ry')

        # TEMPERATURES
        trajectory_data.set_array('temperatures', scalar_quantities[:, 5])
        trajectory_data.set_attribute('units|temperatures', 'K')

        # WALLTIMES
        trajectory_data.set_array('walltimes', scalar_quantities[:, 6])
        trajectory_data.set_attribute('units|walltimes', 's')

        # SCF convergence
        trajectory_data.set_array('scf_convergence', convergence)

        # STRESSES
        if stresses is not None:
            if len(stresses) != nstep:
                return self.exit_codes.ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_0
            trajectory_data.set_array('stresses', stresses)
            trajectory_data.set_attribute('units|stresses', 'atomic')

        self.out('output_trajectory', trajectory_data)

        ######## At last, check output for problems to be reported (e.g. timeout, ...)

        # First check for specific known problems that can cause a pre-mature termination of the calculation
        exit_code = self.validate_premature_exit(logs_stdout)
        if exit_code:
            return self.exit(exit_code)

        ## If the both stdout and xml exit codes are set, there was a basic problem with both output files and there
        ## is no need to investigate any further.
        #if self.exit_code_stdout and self.exit_code_xml:
        #    return self.exit(self.exit_codes.ERROR_OUTPUT_FILES)

        if self.exit_code_stdout:
            return self.exit(self.exit_code_stdout)

        #if self.exit_code_xml:
        #    return self.exit(self.exit_code_xml)

        ## TODO: ADD A VALIDATOR IF THE OUTPUT PRODUCES 'md'-SPECIFIC MESSAGES
        ## First determine issues that can occurr for all calculation types. Note that the generic errors, that are
        ## common to all types are done first. If a problem is found there, we return the exit code and don't continue
        #for validator in [self.validate_electronic, self.validate_dynamics, self.validate_ionic]:
        #    exit_code = validator(trajectory, parsed_parameters, logs_stdout)
        #    if exit_code:
        #        return self.exit(exit_code)

        # TODO: not sure this can actually happen
        if 'ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED' in logs_stdout.error:
            return self.exit_codes.ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED


def get_coords_from_file(filename, pos_block_regex, pos_regex):
    with open(filename) as f:
        coords = np.array([[
            list(map(float,
                     coord_match.group('vals').split())) for coord_match in pos_regex.finditer(match.group(0))
        ] for match in pos_block_regex.finditer(f.read())],
                          dtype=np.float64)
    return coords


def get_coords_from_file_slow_and_steady(
    filename, ncol, raise_if_nan, exit_code_dimension_1, exit_code_dimension_2, exit_code_nan
):
    trajectory = []
    timestep = None
    with open(filename) as f:
        for iline, line in enumerate(f.readlines()):
            line = line.strip()  # If there is space in front of '>'
            if not line:  # skip empty lines:
                continue
            if line.startswith('>'):
                if timestep is None:
                    pass
                    # This is the first step and there is no old timestep
                elif len(timestep) == 0:
                    # The timestep has length 0, something is wrong
                    return None, exit_code_dimension_1
                else:
                    trajectory.append(timestep)
                # We have a new timestep:
                timestep = list()
            else:
                vals = line.split()[1:]  # Ignoring the first column, which is the symbol
                if len(vals) != ncol:
                    return None, exit_code_dimension_2
                else:
                    npvals = np.empty(ncol)
                    for ival, val in enumerate(vals):
                        try:
                            npvals[ival] = float(val)
                        except ValueError:
                            if raise_if_nan:
                                return None, exit_code_nan
                            else:
                                npvals[ival] = np.nan
                    timestep.append(npvals)
        # Append the last timestep
        trajectory.append(timestep)
    timestep_length_set = set([len(timestep) for timestep in trajectory])
    if len(timestep_length_set) > 1:
        del trajectory[-1]
        timestep_length_set = set([len(timestep) for timestep in trajectory])
        if len(timestep_length_set) > 1:
            return None, exit_code_dimension_1
    return np.array(trajectory, dtype=np.float64), False
