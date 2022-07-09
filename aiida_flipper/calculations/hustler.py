import os
import numpy as np

from aiida import orm
from aiida.common import exceptions, datastructures
from aiida.common.lang import classproperty

from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations import _lowercase_dict, _uppercase_dict, _pop_parser_options
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry

from aiida.plugins import CalculationFactory
FlipperCalculation = CalculationFactory('quantumespresso.flipper')

class HustlerCalculation(PwCalculation):
    
    _EVP_FILE = 'verlet.evp'
    _FOR_FILE = 'verlet.for'
    _VEL_FILE = 'verlet.vel'
    _POS_FILE = 'verlet.pos'
    ## verlet.evp columns - timestep, time(ps), kinetic energy, potential energy, total energy, temperature, walltimes, scf_convergence

    @classmethod
    def define(cls, spec):
        super(HustlerCalculation, cls).define(spec)
        """
        Following errors are taken from FlipperCalculation class
        """
        # Input TrajectoryData containing the snapshots that will be used to generate hustler.pos file
        spec.input('hustler_snapshots', valid_type=orm.TrajectoryData, required=False,
            help='The trajectory containing the uncorrelated configurations, that shall be used to calculate DFT and pinball forces')

        # Unrecoverable errors
        spec.exit_code(360, 'ERROR_UNKNOWN_TIMESTEP',
            message='The parser could not get the timestep in the calculation.')
        spec.exit_code(370, 'ERROR_MISSING_TRAJECTORY_FILES',
            message='At least one of trajectory files is missing')

        # MD errors, it is worth trying restarting another time
        spec.exit_code(601, 'ERROR_EMPTY_TRAJECTORY_FILES',
            message='The trajectory files are empty (do not contain values)')
        spec.exit_code(602, 'ERROR_TRAJECTORY_WITH_NAN',
            message='The trajectory files contains non-numeric entries')
        spec.exit_code(603, 'ERROR_CORRUPTED_TRAJECTORY_FILES',
            message='The trajectory files seem corrupted and cannot be read')
        spec.exit_code(604, 'ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_0',
            message='The trajectory files contain arrays of different lengths')
        spec.exit_code(605, 'ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_1',
            message='The trajectory files contain arrays with a different number of entries for each timestep')
        spec.exit_code(606, 'ERROR_INCOMMENSURATE_TRAJECTORY_DIMENSION_2',
            message='The trajectory files contain files with an unexpect number of entries for each line')
        
    def prepare_for_submission(self, folder):
        """
        Create the input files from the input nodes passed to this instance of the `CalcJob`.
        Copy of the original function with one change - hustler_snapshots which is the trajectory containing
        snapshots to generate hustler.pos file is passed as an additional input. 

        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        # pylint: disable=too-many-branches,too-many-statements
        if 'settings' in self.inputs:
            settings = _uppercase_dict(self.inputs.settings.get_dict(), dict_name='settings')
        else:
            settings = {}

        # Check that a pseudo potential was specified for each kind present in the `StructureData`
        kinds = [kind.name for kind in self.inputs.structure.kinds]
        if set(kinds) != set(self.inputs.pseudos.keys()):
            raise exceptions.InputValidationError(
                'Mismatch between the defined pseudos and the list of kinds of the structure.\n'
                'Pseudos: {};\nKinds: {}'.format(', '.join(list(self.inputs.pseudos.keys())), ', '.join(list(kinds)))
            )

        local_copy_list = []
        remote_copy_list = []
        remote_symlink_list = []

        # Create the subfolder that will contain the pseudopotentials
        folder.get_subfolder(self._PSEUDO_SUBFOLDER, create=True)
        # Create the subfolder for the output data (sometimes Quantum ESPRESSO codes crash if the folder does not exist)
        folder.get_subfolder(self._OUTPUT_SUBFOLDER, create=True)

        # If present, add also the Van der Waals table to the pseudo dir. Note that the name of the table is not checked
        # but should be the one expected by Quantum ESPRESSO.
        if 'vdw_table' in self.inputs:
            uuid = self.inputs.vdw_table.uuid
            src_path = self.inputs.vdw_table.filename
            dst_path = os.path.join(self._PSEUDO_SUBFOLDER, self.inputs.vdw_table.filename)
            local_copy_list.append((uuid, src_path, dst_path))

        if 'hubbard_file' in self.inputs:
            uuid = self.inputs.hubbard_file.uuid
            src_path = self.inputs.hubbard_file.filename
            dst_path = self.filename_input_hubbard_parameters
            local_copy_list.append((uuid, src_path, dst_path))

        arguments = [
            self.inputs.parameters,
            settings,
            self.inputs.pseudos,
            self.inputs.structure,
            self.inputs.hustler_snapshots,
        ]
        if self._use_kpoints:
            arguments.append(self.inputs.kpoints)
        input_filecontent, hustler_filecontent, local_copy_pseudo_list = self._generate_PWCPinputdata(*arguments)
        local_copy_list += local_copy_pseudo_list

        # generating the hustler.pos file, no need to link it to `local_copy_list`
        with folder.open(self.inputs.parameters.get_dict()['CONTROL'].get('hustlerfile'), 'w') as hustler:
            hustler.write(hustler_filecontent)

        with folder.open(self.metadata.options.input_filename, 'w') as handle:
            handle.write(input_filecontent)

        # operations for restart
        symlink = settings.pop('PARENT_FOLDER_SYMLINK', self._default_symlink_usage)  # a boolean
        if symlink:
            if 'parent_folder' in self.inputs:
                # I put the symlink to the old parent ./out folder
                remote_symlink_list.append((
                    self.inputs.parent_folder.computer.uuid,
                    os.path.join(self.inputs.parent_folder.get_remote_path(),
                                 self._restart_copy_from), self._restart_copy_to
                ))
        else:
            # copy remote output dir, if specified
            if 'parent_folder' in self.inputs:
                remote_copy_list.append((
                    self.inputs.parent_folder.computer.uuid,
                    os.path.join(self.inputs.parent_folder.get_remote_path(),
                                 self._restart_copy_from), self._restart_copy_to
                ))

        # Create an `.EXIT` file if `only_initialization` flag in `settings` is set to `True`
        if settings.pop('ONLY_INITIALIZATION', False):
            with folder.open(f'{self._PREFIX}.EXIT', 'w') as handle:
                handle.write('\n')

        # Check if specific inputs for the ENVIRON module where specified
        environ_namelist = settings.pop('ENVIRON', None)
        if environ_namelist is not None:
            if not isinstance(environ_namelist, dict):
                raise exceptions.InputValidationError('ENVIRON namelist should be specified as a dictionary')
            # We first add the environ flag to the command-line options (if not already present)
            try:
                if '-environ' not in settings['CMDLINE']:
                    settings['CMDLINE'].append('-environ')
            except KeyError:
                settings['CMDLINE'] = ['-environ']
            # To create a mapping from the species to an incremental fortran 1-based index
            # we use the alphabetical order as in the inputdata generation
            kind_names = sorted([kind.name for kind in self.inputs.structure.kinds])
            mapping_species = {kind_name: (index + 1) for index, kind_name in enumerate(kind_names)}

            with folder.open(self._ENVIRON_INPUT_FILE_NAME, 'w') as handle:
                handle.write('&ENVIRON\n')
                for key, value in sorted(environ_namelist.items()):
                    handle.write(convert_input_to_namelist_entry(key, value, mapping=mapping_species))
                handle.write('/\n')

        # Check for the deprecated 'ALSO_BANDS' setting and if present fire a deprecation log message
        also_bands = settings.pop('ALSO_BANDS', None)
        if also_bands:
            self.node.logger.warning(
                "The '{}' setting is deprecated as bands are now parsed by default. "
                "If you do not want the bands to be parsed set the '{}' to True {}. "
                'Note that the eigenvalue.xml files are also no longer stored in the repository'.format(
                    'also_bands', 'no_bands', type(self)
                )
            )

        calcinfo = datastructures.CalcInfo()

        calcinfo.uuid = str(self.uuid)
        # Start from an empty command line by default
        cmdline_params = self._add_parallelization_flags_to_cmdline_params(cmdline_params=settings.pop('CMDLINE', []))

        # we commented calcinfo.stin_name and added it here in cmdline_params
        # in this way the mpirun ... pw.x ... < aiida.in
        # is replaced by mpirun ... pw.x ... -in aiida.in
        # in the scheduler, _get_run_line, if cmdline_params is empty, it
        # simply uses < calcinfo.stin_name
        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = (list(cmdline_params) + ['-in', self.metadata.options.input_filename])
        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.code_uuid = self.inputs.code.uuid
        calcinfo.codes_info = [codeinfo]

        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list

        # Retrieve by default the output file and the xml file
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self.metadata.options.output_filename)
        calcinfo.retrieve_list.extend(self.xml_filepaths)
        calcinfo.retrieve_list += settings.pop('ADDITIONAL_RETRIEVE_LIST', [])
        calcinfo.retrieve_list += self._internal_retrieve_list

        # Retrieve the k-point directories with the xml files to the temporary folder
        # to parse the band eigenvalues and occupations but not to have to save the raw files
        # if and only if the 'no_bands' key was not set to true in the settings
        no_bands = settings.pop('NO_BANDS', False)
        if no_bands is False:
            xmlpaths = os.path.join(self._OUTPUT_SUBFOLDER, self._PREFIX + '.save', 'K*[0-9]', 'eigenval*.xml')
            calcinfo.retrieve_temporary_list = [[xmlpaths, '.', 2]]

        # We might still have parser options in the settings dictionary: pop them.
        _pop_parser_options(self, settings)

        if settings:
            unknown_keys = ', '.join(list(settings.keys()))
            raise exceptions.InputValidationError(f'`settings` contained unexpected keys: {unknown_keys}')

        calcinfo.retrieve_temporary_list = [self._EVP_FILE, self._FOR_FILE, self._VEL_FILE, self._POS_FILE]
        return calcinfo
        
    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        Do we really need this?
        """
        retdict = PwCalculation._use_methods

        retdict['array'] = {
            'valid_types': orm.ArrayData,
            'additional_parameter': None,
            'linkname': 'array',
            'docstring': 'Use the node defining the trajectory to sample over',
        }
        return retdict

    @classmethod
    def _generate_PWCPinputdata(cls, parameters, settings, pseudos, structure, hustler_snapshots, kpoints=None, use_fractional=False):  # pylint: disable=invalid-name
        """
        Copied from original _general_PWCPinputdata, with two key changes - 
        kind list is NOT sorted alphabetically, and an extra input of hustler-positions.
        """

        ##### THE hustler file:
        hustler_positions = hustler_snapshots.get_array('positions')
        pos_units = hustler_snapshots.get_attribute('units|positions', 'angstrom')

        if pos_units in ('atomic', 'bohr'):
            pass
        elif pos_units == 'angstrom':
            hustler_positions /= 0.529177249
        else:
            raise exceptions.InputValidationError("Unknown position units {}".format(pos_units))
        # I need to check what units the positions were given
        try:
            symbols = hustler_snapshots.get_attribute('symbols')
        except Exception:
            symbols = hustler_snapshots.get_array('symbols')
        nstep, nhustled, _ = hustler_positions.shape
        if len(symbols) != nhustled:
            raise exceptions.InputValidationError(
                "Length of symbols does not match array dimensions"
            )

        hustlerfile = u''
        for istep, tau_of_t in enumerate(hustler_positions):
            hustlerfile += u'> {}\n'.format(istep)
            for symbol, pos in zip(symbols, tau_of_t):
                hustlerfile += u'{:<3}   {}\n'.format(symbol, '   '.join(['{:16.10f}'.format(f) for f in pos]))
                            
        # pylint: disable=too-many-branches,too-many-statements
        from aiida.common.utils import get_unique_filename
        import re
        local_copy_list_to_append = []
        
        # I put the first-level keys as uppercase (i.e., namelist and card names)
        # and the second-level keys as lowercase
        # (deeper levels are unchanged)
        input_params = _uppercase_dict(parameters.get_dict(), dict_name='parameters')
        input_params = {k: _lowercase_dict(v, dict_name=k) for k, v in input_params.items()}

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
        input_params['CONTROL']['lhustle'] = True
        input_params['CONTROL']['hustler_nat'] = nhustled
        input_params['CONTROL']['hustlerfile'] = input_params['CONTROL'].get('hustlerfile')
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

        return inputfile, hustlerfile, local_copy_list_to_append
