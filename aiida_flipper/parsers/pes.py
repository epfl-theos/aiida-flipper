# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os, numpy as np, re
from aiida.orm import CalculationFactory
# from basicpw import BasicpwParser
from aiida.parsers.parser import Parser
from six.moves import map

JOB_DONE_REGEX = re.compile('JOB\s+DONE')


class PesParser(Parser):

    def parse_with_retrieved(self, retrieved):
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
        # Check that the retrieved folder is there
        try:
            out_folder = retrieved[self._calc._get_linkname_retrieved()]
        except KeyError:
            self.logger.error('No retrieved folder found')
            return False, ()

        # check what is inside the folder

        list_of_files = out_folder.get_folder_list()

        # at least the stdout should exist
        if not self._calc._OUTPUT_FILE_NAME in list_of_files:
            self.logger.error('Standard output not found')
            successful = False
            return successful, ()
        if not self._calc._PES_FILE in list_of_files:
            self.logger.error('Potential file not found')
            successful = False
            return successful, ()
        list_of_files.remove(self._calc._OUTPUT_FILE_NAME)
        for f in ('data-file.xml', '_scheduler-stdout.txt', '_scheduler-stderr.txt'):
            try:
                list_of_files.remove(f)
            except Exception as e:
                print(e)

        new_nodes_list = []

        ########################## OUTPUT FILE ##################################

        out_file = os.path.join(out_folder.get_abs_path('.'), self._calc._OUTPUT_FILE_NAME)

        with open(out_file) as f:
            txt = f.read()
            successful = bool(JOB_DONE_REGEX.search(txt))
        return successful, new_nodes_list


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
