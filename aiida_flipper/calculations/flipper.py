from __future__ import absolute_import
import os

import six
from six.moves import zip

from aiida import orm
from aiida.common import exceptions
from aiida.common.lang import classproperty

from aiida_quantumespresso.calculations import BasePwCpInputGenerator, _lowercase_dict, _uppercase_dict
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry

class FlipperCalculation(BasePwCpInputGenerator):

    _automatic_namelists = {
        'scf': ['CONTROL', 'SYSTEM', 'ELECTRONS'],
        'nscf': ['CONTROL', 'SYSTEM', 'ELECTRONS'],
        'bands': ['CONTROL', 'SYSTEM', 'ELECTRONS'],
        'relax': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS'],
        'md': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS'],
        'vc-md': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL'],
        'vc-relax': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL'],
    }

    # Keywords that cannot be set by the user but will be set by the plugin
    _blocked_keywords = [
        ('CONTROL', 'pseudo_dir'),
        ('CONTROL', 'outdir'),
        ('CONTROL', 'prefix'),
        ('SYSTEM', 'ibrav'),
        ('SYSTEM', 'celldm'),
        ('SYSTEM', 'nat'),
        ('SYSTEM', 'ntyp'),
        ('SYSTEM', 'a'),
        ('SYSTEM', 'b'),
        ('SYSTEM', 'c'),
        ('SYSTEM', 'cosab'),
        ('SYSTEM', 'cosac'),
        ('SYSTEM', 'cosbc'),
    ]

    _use_kpoints = True

    # Not using symlink in pw to allow multiple nscf to run on top of the same scf
    _default_symlink_usage = False


    _EVP_FILE = 'verlet.evp'
    _FOR_FILE = 'verlet.for'
    _VEL_FILE = 'verlet.vel'
    _POS_FILE = 'verlet.pos'


    @classproperty
    def xml_filepaths(cls):
        """Return a list of XML output filepaths relative to the remote working directory that should be retrieved."""
        # pylint: disable=no-self-argument,not-an-iterable
        filepaths = []

        for filename in cls.xml_filenames:
            filepath = os.path.join(cls._OUTPUT_SUBFOLDER, '{}.save'.format(cls._PREFIX), filename)
            filepaths.append(filepath)

        return filepaths

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super(FlipperCalculation, cls).define(spec)
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='quantumespresso.flipper')
        #spec.input('metadata.options.without_xml', valid_type=bool, required=False, help='If set to `True` the parser '
        #    'will not fail if the XML file is missing in the retrieved folder.')
        spec.input('kpoints', valid_type=orm.KpointsData,
            help='kpoint mesh or kpoint path')
        #spec.input('hubbard_file', valid_type=orm.SinglefileData, required=False,
        #    help='SinglefileData node containing the output Hubbard parameters from a HpCalculation')
        spec.output('output_parameters', valid_type=orm.Dict,
            help='The `output_parameters` output node of the successful calculation.')
        #spec.output('output_structure', valid_type=orm.StructureData, required=False,
        #    help='The `output_structure` output node of the successful calculation if present.')
        spec.output('output_trajectory', valid_type=orm.TrajectoryData, required=False)
        #spec.output('output_band', valid_type=orm.BandsData, required=False,
        #    help='The `output_band` output node of the successful calculation if present.')
        #spec.output('output_kpoints', valid_type=orm.KpointsData, required=False)
        #spec.output('output_atomic_occupations', valid_type=orm.Dict, required=False)
        spec.default_output_node = 'output_parameters'

        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete
        spec.exit_code(300, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.')
        spec.exit_code(301, 'ERROR_NO_RETRIEVED_TEMPORARY_FOLDER',
            message='The retrieved temporary folder could not be accessed.')
        spec.exit_code(302, 'ERROR_OUTPUT_STDOUT_MISSING',
            message='The retrieved folder did not contain the required stdout output file.')
        # ~ spec.exit_code(303, 'ERROR_OUTPUT_XML_MISSING',
            # ~ message='The retrieved folder did not contain the required required XML file.')
        # ~ spec.exit_code(304, 'ERROR_OUTPUT_XML_MULTIPLE',
            # ~ message='The retrieved folder contained multiple XML files.')
        # ~ spec.exit_code(305, 'ERROR_OUTPUT_FILES',
            # ~ message='Both the stdout and XML output files could not be read or parsed.')
        spec.exit_code(310, 'ERROR_OUTPUT_STDOUT_READ',
            message='The stdout output file could not be read.')
        spec.exit_code(311, 'ERROR_OUTPUT_STDOUT_PARSE',
            message='The stdout output file could not be parsed.')
        spec.exit_code(312, 'ERROR_OUTPUT_STDOUT_INCOMPLETE',
            message='The stdout output file was incomplete probably because the calculation got interrupted.')
        # ~ spec.exit_code(320, 'ERROR_OUTPUT_XML_READ',
            # ~ message='The XML output file could not be read.')
        # ~ spec.exit_code(321, 'ERROR_OUTPUT_XML_PARSE',
            # ~ message='The XML output file could not be parsed.')
        # ~ spec.exit_code(322, 'ERROR_OUTPUT_XML_FORMAT',
            # ~ message='The XML output file has an unsupported format.')
        spec.exit_code(340, 'ERROR_OUT_OF_WALLTIME_INTERRUPTED',
            message='The calculation stopped prematurely because it ran out of walltime but the job was killed by the '
                    'scheduler before the files were safely written to disk for a potential restart.')
        spec.exit_code(350, 'ERROR_UNEXPECTED_PARSER_EXCEPTION',
            message='The parser raised an unexpected exception.')

        # Significant errors but calculation can be used to restart
        # ~ spec.exit_code(400, 'ERROR_OUT_OF_WALLTIME',
            # ~ message='The calculation stopped prematurely because it ran out of walltime.')
        # ~ spec.exit_code(410, 'ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED',
            # ~ message='The electronic minimization cycle did not reach self-consistency.')

        # ~ spec.exit_code(461, 'ERROR_DEXX_IS_NEGATIVE',
            # ~ message='The code failed with negative dexx in the exchange calculation.')
        # ~ spec.exit_code(462, 'ERROR_COMPUTING_CHOLESKY',
            # ~ message='The code failed during the cholesky factorization.')

        # ~ spec.exit_code(481, 'ERROR_NPOOLS_TOO_HIGH',
            # ~ message='The k-point parallelization "npools" is too high, some nodes have no k-points.')

        # ~ spec.exit_code(500, 'ERROR_IONIC_CONVERGENCE_NOT_REACHED',
            # ~ message='The ionic minimization cycle did not converge for the given thresholds.')
        # ~ spec.exit_code(501, 'ERROR_IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF',
            # ~ message='Then ionic minimization cycle converged but the thresholds are exceeded in the final SCF.')
        # ~ spec.exit_code(502, 'ERROR_IONIC_CYCLE_EXCEEDED_NSTEP',
            # ~ message='The ionic minimization cycle did not converge after the maximum number of steps.')
        # ~ spec.exit_code(510, 'ERROR_IONIC_CYCLE_ELECTRONIC_CONVERGENCE_NOT_REACHED',
            # ~ message='The electronic minimization cycle failed during an ionic minimization cycle.')
        # ~ spec.exit_code(511, 'ERROR_IONIC_CONVERGENCE_REACHED_FINAL_SCF_FAILED',
            # ~ message='The ionic minimization cycle converged, but electronic convergence was not reached in the '
                    # ~ 'final SCF.')
        # ~ spec.exit_code(520, 'ERROR_IONIC_CYCLE_BFGS_HISTORY_FAILURE',
            # ~ message='The ionic minimization cycle terminated prematurely because of two consecutive failures in the '
                    # ~ 'BFGS algorithm.')
        # ~ spec.exit_code(521, 'ERROR_IONIC_CYCLE_BFGS_HISTORY_AND_FINAL_SCF_FAILURE',
            # ~ message='The ionic minimization cycle terminated prematurely because of two consecutive failures in the '
                    # ~ 'BFGS algorithm and electronic convergence failed in the final SCF.')

        # ~ spec.exit_code(531, 'ERROR_CHARGE_IS_WRONG',
            # ~ message='The electronic minimization cycle did not reach self-consistency.')
        # ~ spec.exit_code(541, 'ERROR_SYMMETRY_NON_ORTHOGONAL_OPERATION',
            # ~ message='The variable cell optimization broke the symmetry of the k-points.')


    def prepare_for_submission(self, folder):
        """
        Create the input files from the input nodes passed to this instance of the `CalcJob`.
        Calls the parent's prepare_for_submission, and adds files to calcinfo instance.

        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        calcinfo = super(FlipperCalculation, self).prepare_for_submission(folder)
        verlet_files = [self._EVP_FILE, self._FOR_FILE, self._VEL_FILE, self._POS_FILE]
        try:
            calcinfo.retrieve_temporary_list += verlet_files
        except AttributeError:
            calcinfo.retrieve_temporary_list = verlet_files
        return calcinfo

    @classmethod
    def _generate_PWCPinputdata(cls, parameters, settings, pseudos, structure, kpoints=None, use_fractional=False):  # pylint: disable=invalid-name
        """
        Copied from original _general_PWCPinputdata,
        with one key change, namely that kind list is NOT sorted alphabetically.
        """
        # pylint: disable=too-many-branches,too-many-statements
        from aiida.common.utils import get_unique_filename
        import re
        local_copy_list_to_append = []

        # I put the first-level keys as uppercase (i.e., namelist and card names)
        # and the second-level keys as lowercase
        # (deeper levels are unchanged)
        input_params = _uppercase_dict(parameters.get_dict(), dict_name='parameters')
        input_params = {k: _lowercase_dict(v, dict_name=k) for k, v in six.iteritems(input_params)}

        # I remove unwanted elements (for the moment, instead, I stop; to change when we setup a reasonable logging)
        for blocked in cls._blocked_keywords:
            namelist = blocked[0].upper()
            flag = blocked[1].lower()
            defaultvalue = None
            if len(blocked) >= 3:
                defaultvalue = blocked[2]
            if namelist in input_params:
                # The following lines is meant to avoid putting in input the
                # parameters like celldm(*)
                stripped_inparams = [re.sub('[(0-9)]', '', _) for _ in input_params[namelist].keys()]
                if flag in stripped_inparams:
                    raise exceptions.InputValidationError(
                        "You cannot specify explicitly the '{}' flag in the '{}' "
                        'namelist or card.'.format(flag, namelist)
                    )
                if defaultvalue is not None:
                    if namelist not in input_params:
                        input_params[namelist] = {}
                    input_params[namelist][flag] = defaultvalue

        # Set some variables (look out at the case! NAMELISTS should be uppercase,
        # internal flag names must be lowercase)
        input_params.setdefault('CONTROL', {})
        input_params['CONTROL']['pseudo_dir'] = cls._PSEUDO_SUBFOLDER
        input_params['CONTROL']['outdir'] = cls._OUTPUT_SUBFOLDER
        input_params['CONTROL']['prefix'] = cls._PREFIX
        input_params['CONTROL']['verbosity'] = input_params['CONTROL'].get('verbosity', cls._default_verbosity)

        # ============ I prepare the input site data =============
        # ------------ CELL_PARAMETERS -----------
        cell_parameters_card = 'CELL_PARAMETERS angstrom\n'
        for vector in structure.cell:
            cell_parameters_card += ('{0:18.10f} {1:18.10f} {2:18.10f}' '\n'.format(*vector))

        # ------------- ATOMIC_SPECIES ------------
        atomic_species_card_list = []

        # Keep track of the filenames to avoid to overwrite files
        # I use a dictionary where the key is the pseudo PK and the value
        # is the filename I used. In this way, I also use the same filename
        # if more than one kind uses the same pseudo.
        pseudo_filenames = {}

        # I keep track of the order of species
        kind_names = []
        # I add the pseudopotential files to the list of files to be copied
        for kind in structure.kinds:
            # This should not give errors, I already checked before that
            # the list of keys of pseudos and kinds coincides
            pseudo = pseudos[kind.name]
            if kind.is_alloy or kind.has_vacancies:
                raise exceptions.InputValidationError(
                    "Kind '{}' is an alloy or has "
                    'vacancies. This is not allowed for pw.x input structures.'
                    ''.format(kind.name)
                )

            try:
                # If it is the same pseudopotential file, use the same filename
                filename = pseudo_filenames[pseudo.pk]
            except KeyError:
                # The pseudo was not encountered yet; use a new name and also add it to the local copy list
                filename = get_unique_filename(pseudo.filename, list(pseudo_filenames.values()))
                pseudo_filenames[pseudo.pk] = filename
                local_copy_list_to_append.append(
                    (pseudo.uuid, pseudo.filename, os.path.join(cls._PSEUDO_SUBFOLDER, filename))
                )

            kind_names.append(kind.name)
            atomic_species_card_list.append('{} {} {}\n'.format(kind.name.ljust(6), kind.mass, filename))

        # Below is a change with respect to the original PwCalculation plugin.
        # In that one, for unknown reasons, the atomic_species card is sorted alphabetically.
        # The pinball code requires the pinball species to be the first species!
        atomic_species_card = ''.join(['ATOMIC_SPECIES\n'] + list(atomic_species_card_list))

        # The format of mapping_species required later is a dictionary, whose
        # values are the indices, so I convert to this format
        # Note the (idx+1) to convert to fortran 1-based lists
        mapping_species = {sp_name: (idx + 1) for idx, sp_name in enumerate(kind_names)}

        # Free memory
        del atomic_species_card_list

        # ------------ ATOMIC_POSITIONS -----------
        # Check on validity of FIXED_COORDS
        fixed_coords_strings = []
        fixed_coords = settings.pop('FIXED_COORDS', None)
        if fixed_coords is None:
            # No fixed_coords specified: I store a list of empty strings
            fixed_coords_strings = [''] * len(structure.sites)
        else:
            if len(fixed_coords) != len(structure.sites):
                raise exceptions.InputValidationError(
                    'Input structure contains {:d} sites, but '
                    'fixed_coords has length {:d}'.format(len(structure.sites), len(fixed_coords))
                )

            for i, this_atom_fix in enumerate(fixed_coords):
                if len(this_atom_fix) != 3:
                    raise exceptions.InputValidationError('fixed_coords({:d}) has not length three' ''.format(i + 1))
                for fixed_c in this_atom_fix:
                    if not isinstance(fixed_c, bool):
                        raise exceptions.InputValidationError(
                            'fixed_coords({:d}) has non-boolean '
                            'elements'.format(i + 1)
                        )

                if_pos_values = [cls._if_pos(_) for _ in this_atom_fix]
                fixed_coords_strings.append('  {:d} {:d} {:d}'.format(*if_pos_values))

        abs_pos = [_.position for _ in structure.sites]
        if use_fractional:
            import numpy as np
            atomic_positions_card_list = ['ATOMIC_POSITIONS crystal\n']
            coordinates = np.dot(np.array(abs_pos), np.linalg.inv(np.array(structure.cell)))
        else:
            atomic_positions_card_list = ['ATOMIC_POSITIONS angstrom\n']
            coordinates = abs_pos

        for site, site_coords, fixed_coords_string in zip(structure.sites, coordinates, fixed_coords_strings):
            atomic_positions_card_list.append(
                '{0} {1:18.10f} {2:18.10f} {3:18.10f} {4}\n'.format(
                    site.kind_name.ljust(6), site_coords[0], site_coords[1], site_coords[2], fixed_coords_string
                )
            )

        atomic_positions_card = ''.join(atomic_positions_card_list)
        del atomic_positions_card_list

        # Optional ATOMIC_FORCES card
        atomic_forces = settings.pop('ATOMIC_FORCES', None)
        if atomic_forces is not None:

            # Checking that there are as many forces defined as there are sites in the structure
            if len(atomic_forces) != len(structure.sites):
                raise exceptions.InputValidationError(
                    'Input structure contains {:d} sites, but atomic forces has length {:d}'.format(
                        len(structure.sites), len(atomic_forces)
                    )
                )

            lines = ['ATOMIC_FORCES\n']
            for site, vector in zip(structure.sites, atomic_forces):

                # Checking that all 3 dimensions are specified:
                if len(vector) != 3:
                    raise exceptions.InputValidationError('Forces({}) for {} has not length three'.format(vector, site))

                lines.append('{0} {1:18.10f} {2:18.10f} {3:18.10f}\n'.format(site.kind_name.ljust(6), *vector))

            # Append to atomic_positions_card so that this card will be printed directly after
            atomic_positions_card += ''.join(lines)
            del lines

        # Optional ATOMIC_VELOCITIES card
        atomic_velocities = settings.pop('ATOMIC_VELOCITIES', None)
        if atomic_velocities is not None:

            # Checking that there are as many velocities defined as there are sites in the structure
            if len(atomic_velocities) != len(structure.sites):
                raise exceptions.InputValidationError(
                    'Input structure contains {:d} sites, but atomic velocities has length {:d}'.format(
                        len(structure.sites), len(atomic_velocities)
                    )
                )

            lines = ['ATOMIC_VELOCITIES\n']
            for site, vector in zip(structure.sites, atomic_velocities):

                # Checking that all 3 dimensions are specified:
                if len(vector) != 3:
                    raise exceptions.InputValidationError(
                        'Velocities({}) for {} has not length three'.format(vector, site)
                    )

                lines.append('{0} {1:18.10f} {2:18.10f} {3:18.10f}\n'.format(site.kind_name.ljust(6), *vector))

            # Append to atomic_positions_card so that this card will be printed directly after
            atomic_positions_card += ''.join(lines)
            del lines

        # I set the variables that must be specified, related to the system
        # Set some variables (look out at the case! NAMELISTS should be
        # uppercase, internal flag names must be lowercase)
        input_params.setdefault('SYSTEM', {})
        input_params['SYSTEM']['ibrav'] = 0
        input_params['SYSTEM']['nat'] = len(structure.sites)
        input_params['SYSTEM']['ntyp'] = len(structure.kinds)

        # ============ I prepare the k-points =============
        if cls._use_kpoints:
            try:
                mesh, offset = kpoints.get_kpoints_mesh()
                has_mesh = True
                force_kpoints_list = settings.pop('FORCE_KPOINTS_LIST', False)
                if force_kpoints_list:
                    kpoints_list = kpoints.get_kpoints_mesh(print_list=True)
                    num_kpoints = len(kpoints_list)
                    has_mesh = False
                    weights = [1.] * num_kpoints

            except AttributeError:

                try:
                    kpoints_list = kpoints.get_kpoints()
                    num_kpoints = len(kpoints_list)
                    has_mesh = False
                    if num_kpoints == 0:
                        raise exceptions.InputValidationError(
                            'At least one k point must be '
                            'provided for non-gamma calculations'
                        )
                except AttributeError:
                    raise exceptions.InputValidationError('No valid kpoints have been found')

                try:
                    _, weights = kpoints.get_kpoints(also_weights=True)
                except AttributeError:
                    weights = [1.] * num_kpoints

            gamma_only = settings.pop('GAMMA_ONLY', False)

            if gamma_only:
                if has_mesh:
                    if tuple(mesh) != (1, 1, 1) or tuple(offset) != (0., 0., 0.):
                        raise exceptions.InputValidationError(
                            'If a gamma_only calculation is requested, the '
                            'kpoint mesh must be (1,1,1),offset=(0.,0.,0.)'
                        )

                else:
                    if (len(kpoints_list) != 1 or tuple(kpoints_list[0]) != tuple(0., 0., 0.)):
                        raise exceptions.InputValidationError(
                            'If a gamma_only calculation is requested, the '
                            'kpoints coordinates must only be (0.,0.,0.)'
                        )

                kpoints_type = 'gamma'

            elif has_mesh:
                kpoints_type = 'automatic'

            else:
                kpoints_type = 'crystal'

            kpoints_card_list = ['K_POINTS {}\n'.format(kpoints_type)]

            if kpoints_type == 'automatic':
                if any([i not in [0, 0.5] for i in offset]):
                    raise exceptions.InputValidationError('offset list must only be made of 0 or 0.5 floats')
                the_offset = [0 if i == 0. else 1 for i in offset]
                the_6_integers = list(mesh) + the_offset
                kpoints_card_list.append('{:d} {:d} {:d} {:d} {:d} {:d}\n' ''.format(*the_6_integers))

            elif kpoints_type == 'gamma':
                # nothing to be written in this case
                pass
            else:
                kpoints_card_list.append('{:d}\n'.format(num_kpoints))
                for kpoint, weight in zip(kpoints_list, weights):
                    kpoints_card_list.append(
                        '  {:18.10f} {:18.10f} {:18.10f} {:18.10f}'
                        '\n'.format(kpoint[0], kpoint[1], kpoint[2], weight)
                    )

            kpoints_card = ''.join(kpoints_card_list)
            del kpoints_card_list

        # =================== NAMELISTS AND CARDS ========================
        try:
            namelists_toprint = settings.pop('NAMELISTS')
            if not isinstance(namelists_toprint, list):
                raise exceptions.InputValidationError(
                    "The 'NAMELISTS' value, if specified in the settings input "
                    'node, must be a list of strings'
                )
        except KeyError:  # list of namelists not specified; do automatic detection
            try:
                control_nl = input_params['CONTROL']
                calculation_type = control_nl['calculation']
            except KeyError:
                raise exceptions.InputValidationError(
                    "No 'calculation' in CONTROL namelist."
                    'It is required for automatic detection of the valid list '
                    'of namelists. Otherwise, specify the list of namelists '
                    "using the NAMELISTS key inside the 'settings' input node."
                )

            try:
                namelists_toprint = cls._automatic_namelists[calculation_type]
            except KeyError:
                raise exceptions.InputValidationError(
                    "Unknown 'calculation' value in "
                    'CONTROL namelist {}. Otherwise, specify the list of '
                    "namelists using the NAMELISTS inside the 'settings' input "
                    'node'.format(calculation_type)
                )

        inputfile = u''
        for namelist_name in namelists_toprint:
            inputfile += u'&{0}\n'.format(namelist_name)
            # namelist content; set to {} if not present, so that we leave an empty namelist
            namelist = input_params.pop(namelist_name, {})
            for key, value in sorted(namelist.items()):
                inputfile += convert_input_to_namelist_entry(key, value, mapping=mapping_species)
            inputfile += u'/\n'

        # Write cards now
        inputfile += atomic_species_card
        inputfile += atomic_positions_card
        if cls._use_kpoints:
            inputfile += kpoints_card
        inputfile += cell_parameters_card

        if input_params:
            raise exceptions.InputValidationError(
                'The following namelists are specified in input_params, but are '
                'not valid namelists for the current type of calculation: '
                '{}'.format(','.join(list(input_params.keys())))
            )

        return inputfile, local_copy_list_to_append
