from __future__ import absolute_import
import os, numpy as np

from aiida.orm import CalculationFactory
import six
from six.moves import zip
PwCalculation = CalculationFactory('quantumespresso.pw')
from aiida_quantumespresso.calculations import get_input_data_text, _lowercase_dict, _uppercase_dict

from aiida.common.utils import classproperty
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.upf import UpfData
from aiida.orm.data.remote import RemoteData
from aiida.common.datastructures import CodeInfo, CalcInfo
from aiida.common.exceptions import InputValidationError
from aiida.common.constants import bohr_to_ang


class HustlerCalculation(PwCalculation):

    def _init_internal_params(self):
        super(HustlerCalculation, self)._init_internal_params()
        self._default_parser = 'quantumespresso.flipper'
        self._EVP_FILE = 'verlet.evp'
        self._FOR_FILE = 'verlet.for'
        self._VEL_FILE = 'verlet.vel'
        self._POS_FILE = 'verlet.pos'
        self._HUSTLER_FILE = 'hustler.pos'

    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        """
        retdict = PwCalculation._use_methods

        retdict['array'] = {
            'valid_types': ArrayData,
            'additional_parameter': None,
            'linkname': 'array',
            'docstring': 'Use the node defining the trajectory to sample over',
        }
        return retdict

    def _prepare_for_submission(self, tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputs_dict (without the Code!)


        ..note:: Basically everything was copied here from
            quantumespresso.__init__
            The difference is that the keycard atomic_species_card_list
            has to remain unsorted!
        """
        from aiida.common.utils import get_unique_filename, get_suggestion
        import re

        local_copy_list = []
        remote_copy_list = []
        remote_symlink_list = []

        try:
            parameters = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            raise InputValidationError('No parameters specified for this calculation')
        if not isinstance(parameters, ParameterData):
            raise InputValidationError('parameters is not of type ParameterData')

        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise InputValidationError('No structure specified for this calculation')
        if not isinstance(structure, StructureData):
            raise InputValidationError('structure is not of type StructureData')

        if self._use_kpoints:
            try:
                kpoints = inputdict.pop(self.get_linkname('kpoints'))
            except KeyError:
                raise InputValidationError('No kpoints specified for this calculation')
            if not isinstance(kpoints, KpointsData):
                raise InputValidationError('kpoints is not of type KpointsData')

        # Settings can be undefined, and defaults to an empty dictionary
        settings = inputdict.pop(self.get_linkname('settings'), None)
        if settings is None:
            settings_dict = {}
        else:
            if not isinstance(settings, ParameterData):
                raise InputValidationError('settings, if specified, must be of ' 'type ParameterData')
            # Settings converted to uppercase
            settings_dict = _uppercase_dict(settings.get_dict(), dict_name='settings')

        pseudos = {}
        # I create here a dictionary that associates each kind name to a pseudo
        for link in inputdict.keys():
            if link.startswith(self._get_linkname_pseudo_prefix()):
                kindstring = link[len(self._get_linkname_pseudo_prefix()):]
                kinds = kindstring.split('_')
                the_pseudo = inputdict.pop(link)
                if not isinstance(the_pseudo, UpfData):
                    raise InputValidationError(
                        'Pseudo for kind(s) {} is not of '
                        'type UpfData'.format(','.join(kinds))
                    )
                for kind in kinds:
                    if kind in pseudos:
                        raise InputValidationError('Pseudo for kind {} passed ' 'more than one time'.format(kind))
                    pseudos[kind] = the_pseudo

        #~ parent_calc_folder = inputdict.pop(self.get_linkname('parent_folder'), None)
        # parent_calc_folder = inputdict.pop(self.get_linkname('remote_folder'), None)
        parent_calc_folder = inputdict.pop('remote_folder', None)
        if parent_calc_folder is not None:
            if not isinstance(parent_calc_folder, RemoteData):
                raise InputValidationError('parent_calc_folder, if specified, ' 'must be of type RemoteData')

        vdw_table = inputdict.pop(self.get_linkname('vdw_table'), None)
        if vdw_table is not None:
            if not isinstance(vdw_table, SinglefileData):
                raise InputValidationError('vdw_table, if specified, ' 'must be of type SinglefileData')

        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError('No code specified for this calculation')

        try:
            hustler_arr = inputdict.pop(self.get_linkname('array'))
        except KeyError:
            raise InputValidationError('No array specified for this calculation')

        # Here, there should be no more parameters...
        if inputdict:
            raise InputValidationError(
                'The following input data nodes are '
                'unrecognized: {}'.format(list(inputdict.keys()))
            )

        # Check structure, get species, check peudos
        kindnames = [k.name for k in structure.kinds]
        if set(kindnames) != set(pseudos.keys()):
            err_msg = (
                'Mismatch between the defined pseudos and the list of '
                'kinds of the structure. Pseudos: {}; kinds: {}'.format(
                    ','.join(list(pseudos.keys())), ','.join(list(kindnames))
                )
            )
            raise InputValidationError(err_msg)

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        ##### THE hustler file:
        hustler_positions = hustler_arr.get_array('positions')
        pos_units = hustler_arr.get_attr('units|positions', 'angstrom')

        if pos_units in ('atomic', 'bohr'):
            pass
        elif pos_units == 'angstrom':
            hustler_positions /= bohr_to_ang
        else:
            raise InputValidationError('Unknown position units {}'.format(pos_units))
        # I need to check what units the positions were given
        try:
            symbols = hustler_arr.get_attr('symbols')
        except Exception:
            symbols = hustler_arr.get_array('symbols')
        nstep, nhustled, _ = hustler_positions.shape
        if len(symbols) != nhustled:
            raise InputValidationError('Length of symbols does not match array dimensions')

        with open(tempfolder.get_abs_path(self._HUSTLER_FILE), 'w') as hustlerfile:
            for istep, tau_of_t in enumerate(hustler_positions):
                hustlerfile.write('> {}\n'.format(istep))
                for symbol, pos in zip(symbols, tau_of_t):
                    hustlerfile.write('{:<3}   {}\n'.format(symbol, '   '.join(['{:16.10f}'.format(f) for f in pos])))

        # I put the first-level keys as uppercase (i.e., namelist and card names)
        # and the second-level keys as lowercase
        # (deeper levels are unchanged)
        input_params = _uppercase_dict(parameters.get_dict(), dict_name='parameters')
        input_params = {k: _lowercase_dict(v, dict_name=k) for k, v in six.iteritems(input_params)}

        # I remove unwanted elements (for the moment, instead, I stop; to change when
        # we setup a reasonable logging)
        for blocked in self._blocked_keywords:
            nl = blocked[0].upper()
            flag = blocked[1].lower()
            defaultvalue = None
            if len(blocked) >= 3:
                defaultvalue = blocked[2]
            if nl in input_params:
                # The following lines is meant to avoid putting in input the
                # parameters like celldm(*)
                stripped_inparams = [re.sub('[(0-9)]', '', _) for _ in input_params[nl].keys()]
                if flag in stripped_inparams:
                    raise InputValidationError(
                        "You cannot specify explicitly the '{}' flag in the '{}' "
                        'namelist or card.'.format(flag, nl)
                    )
                if defaultvalue is not None:
                    if nl not in input_params:
                        input_params[nl] = {}
                    input_params[nl][flag] = defaultvalue

        # Set some variables (look out at the case! NAMELISTS should be uppercase,
        # internal flag names must be lowercase)
        if 'CONTROL' not in input_params:
            input_params['CONTROL'] = {}
        input_params['CONTROL']['pseudo_dir'] = self._PSEUDO_SUBFOLDER
        input_params['CONTROL']['outdir'] = self._OUTPUT_SUBFOLDER
        input_params['CONTROL']['prefix'] = self._PREFIX
        input_params['CONTROL']['lhustle'] = True
        input_params['CONTROL']['hustler_nat'] = nhustled
        input_params['CONTROL']['hustlerfile'] = self._HUSTLER_FILE

        input_params['CONTROL']['verbosity'] = input_params['CONTROL'].get(
            'verbosity', self._default_verbosity
        )  # Set to high if not specified

        # ============ I prepare the input site data =============
        # ------------ CELL_PARAMETERS -----------
        cell_parameters_card = 'CELL_PARAMETERS angstrom\n'
        for vector in structure.cell:
            cell_parameters_card += ('{0:18.10f} {1:18.10f} {2:18.10f}' '\n'.format(*vector))

        # ------------- ATOMIC_SPECIES ------------
        # I create the subfolder that will contain the pseudopotentials
        tempfolder.get_subfolder(self._PSEUDO_SUBFOLDER, create=True)
        # I create the subfolder with the output data (sometimes Quantum
        # Espresso codes crash if an empty folder is not already there
        tempfolder.get_subfolder(self._OUTPUT_SUBFOLDER, create=True)

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
            ps = pseudos[kind.name]
            if kind.is_alloy() or kind.has_vacancies():
                raise InputValidationError(
                    "Kind '{}' is an alloy or has "
                    'vacancies. This is not allowed for pw.x input structures.'
                    ''.format(kind.name)
                )

            try:
                # It it is the same pseudopotential file, use the same filename
                filename = pseudo_filenames[ps.pk]
            except KeyError:
                # The pseudo was not encountered yet; use a new name and
                # also add it to the local copy list
                filename = get_unique_filename(ps.filename, list(pseudo_filenames.values()))
                pseudo_filenames[ps.pk] = filename
                # I add this pseudo file to the list of files to copy
                local_copy_list.append((ps.get_file_abs_path(), os.path.join(self._PSEUDO_SUBFOLDER, filename)))
            kind_names.append(kind.name)
            atomic_species_card_list.append('{} {} {}\n'.format(kind.name.ljust(6), kind.mass, filename))

        # If present, add also the Van der Waals table to the pseudo dir
        # Note that the name of the table is not checked but should be the
        # one expected by QE.
        if vdw_table:
            local_copy_list.append((
                vdw_table.get_file_abs_path(),
                os.path.join(self._PSEUDO_SUBFOLDER,
                             os.path.split(vdw_table.get_file_abs_path())[1])
            ))

        # I join the lines, but I resort them using the alphabetical order of
        # species, given by the kind_names list. I also store the mapping_species
        # list, with the order of species used in the file

        # THE FLIPPER NEEDS UNSORTED KEYCARD
        mapping_species, sorted_atomic_species_card_list = kind_names, atomic_species_card_list
        #~ mapping_species, sorted_atomic_species_card_list = zip(
        #~ *sorted(zip(kind_names, atomic_species_card_list)))

        # The format of mapping_species required later is a dictionary, whose
        # values are the indices, so I convert to this format
        # Note the (idx+1) to convert to fortran 1-based lists
        mapping_species = {sp_name: (idx + 1) for idx, sp_name in enumerate(mapping_species)}
        # I add the first line
        sorted_atomic_species_card_list = (['ATOMIC_SPECIES\n'] + list(sorted_atomic_species_card_list))
        atomic_species_card = ''.join(sorted_atomic_species_card_list)
        # Free memory
        del sorted_atomic_species_card_list
        del atomic_species_card_list

        # ------------ ATOMIC_POSITIONS -----------
        atomic_positions_card_list = ['ATOMIC_POSITIONS angstrom\n']

        # Check on validity of FIXED_COORDS
        fixed_coords_strings = []
        fixed_coords = settings_dict.pop('FIXED_COORDS', None)
        if fixed_coords is None:
            # No fixed_coords specified: I store a list of empty strings
            fixed_coords_strings = [''] * len(structure.sites)
        else:
            if len(fixed_coords) != len(structure.sites):
                raise InputValidationError(
                    'Input structure contains {:d} sites, but '
                    'fixed_coords has length {:d}'.format(len(structure.sites), len(fixed_coords))
                )

            for i, this_atom_fix in enumerate(fixed_coords):
                if len(this_atom_fix) != 3:
                    raise InputValidationError('fixed_coords({:d}) has not length three' ''.format(i + 1))
                for fixed_c in this_atom_fix:
                    if not isinstance(fixed_c, bool):
                        raise InputValidationError('fixed_coords({:d}) has non-boolean ' 'elements'.format(i + 1))

                if_pos_values = [self._if_pos(_) for _ in this_atom_fix]
                fixed_coords_strings.append('  {:d} {:d} {:d}'.format(*if_pos_values))

        for site, fixed_coords_string in zip(structure.sites, fixed_coords_strings):
            atomic_positions_card_list.append(
                '{0} {1:18.10f} {2:18.10f} {3:18.10f} {4}\n'.format(
                    site.kind_name.ljust(6), site.position[0], site.position[1], site.position[2], fixed_coords_string
                )
            )
        atomic_positions_card = ''.join(atomic_positions_card_list)
        del atomic_positions_card_list  # Free memory
        atomic_velocities = settings_dict.pop('ATOMIC_VELOCITIES', None)
        if atomic_velocities is not None:
            # Checking if as many velocities are set as structures:
            if len(atomic_velocities) != len(structure.sites):
                raise InputValidationError(
                    'Input structure contains {:d} sites, but '
                    'atomic velocities has length {:d}'.format(len(structure.sites), len(atomic_velocities))
                )
            atomic_velocities_strings = ['ATOMIC_VELOCITIES\n']
            for site, vel in zip(structure.sites, atomic_velocities):
                # Checking that all 3 dimension are specified:
                if len(vel) != 3:
                    raise InputValidationError('Velocities({}) for {} has not length three' ''.format(vel, site))
                # Appending to atomic_velocities_strings
                atomic_velocities_strings.append(
                    '{0} {1:18.10f} {2:18.10f} {3:18.10f}\n'.format(site.kind_name.ljust(6), vel[0], vel[1], vel[2])
                )
            # I append to atomic_positions_card  so that velocities
            # are defined right after positions:
            atomic_positions_card = atomic_positions_card + ''.join(atomic_velocities_strings)
            # Freeing the memory
            del atomic_velocities_strings

        # I set the variables that must be specified, related to the system
        # Set some variables (look out at the case! NAMELISTS should be
        # uppercase, internal flag names must be lowercase)
        if 'SYSTEM' not in input_params:
            input_params['SYSTEM'] = {}
        input_params['SYSTEM']['ibrav'] = 0
        input_params['SYSTEM']['nat'] = len(structure.sites)
        input_params['SYSTEM']['ntyp'] = len(kindnames)

        # ============ I prepare the k-points =============
        if self._use_kpoints:
            try:
                mesh, offset = kpoints.get_kpoints_mesh()
                has_mesh = True
                force_kpoints_list = settings_dict.pop('FORCE_KPOINTS_LIST', False)
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
                        raise InputValidationError(
                            'At least one k point must be '
                            'provided for non-gamma calculations'
                        )
                except AttributeError:
                    raise InputValidationError('No valid kpoints have been found')

                try:
                    _, weights = kpoints.get_kpoints(also_weights=True)
                except AttributeError:
                    weights = [1.] * num_kpoints

            gamma_only = settings_dict.pop('GAMMA_ONLY', False)

            if gamma_only:
                if has_mesh:
                    if tuple(mesh) != (1, 1, 1) or tuple(offset) != (0., 0., 0.):
                        raise InputValidationError(
                            'If a gamma_only calculation is requested, the '
                            'kpoint mesh must be (1,1,1),offset=(0.,0.,0.)'
                        )

                else:
                    if (len(kpoints_list) != 1 or tuple(kpoints_list[0]) != tuple(0., 0., 0.)):
                        raise InputValidationError(
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
                if any([(i != 0. and i != 0.5) for i in offset]):
                    raise InputValidationError('offset list must only be made ' 'of 0 or 0.5 floats')
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
            namelists_toprint = settings_dict.pop('NAMELISTS')
            if not isinstance(namelists_toprint, list):
                raise InputValidationError(
                    "The 'NAMELISTS' value, if specified in the settings input "
                    'node, must be a list of strings'
                )
        except KeyError:  # list of namelists not specified; do automatic detection
            try:
                control_nl = input_params['CONTROL']
                calculation_type = control_nl['calculation']
            except KeyError:
                raise InputValidationError(
                    "No 'calculation' in CONTROL namelist."
                    'It is required for automatic detection of the valid list '
                    'of namelists. Otherwise, specify the list of namelists '
                    "using the NAMELISTS key inside the 'settings' input node"
                )

            try:
                namelists_toprint = self._automatic_namelists[calculation_type]
            except KeyError:
                sugg_string = get_suggestion(calculation_type, list(self._automatic_namelists.keys()))
                raise InputValidationError(
                    "Unknown 'calculation' value in "
                    'CONTROL namelist {}. Otherwise, specify the list of '
                    "namelists using the NAMELISTS inside the 'settings' input "
                    'node'.format(sugg_string)
                )

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)

        with open(input_filename, 'w') as infile:
            for namelist_name in namelists_toprint:
                infile.write('&{0}\n'.format(namelist_name))
                # namelist content; set to {} if not present, so that we leave an
                # empty namelist
                namelist = input_params.pop(namelist_name, {})
                for k, v in sorted(six.iteritems(namelist)):
                    infile.write(get_input_data_text(k, v, mapping=mapping_species))
                infile.write('/\n')

            # Write cards now
            infile.write(atomic_species_card)
            infile.write(atomic_positions_card)
            if self._use_kpoints:
                infile.write(kpoints_card)
            infile.write(cell_parameters_card)
            #TODO: write CONSTRAINTS
            #TODO: write OCCUPATIONS

        if input_params:
            raise InputValidationError(
                'The following namelists are specified in input_params, but are '
                'not valid namelists for the current type of calculation: '
                '{}'.format(','.join(list(input_params.keys())))
            )

        # operations for restart
        symlink = settings_dict.pop('PARENT_FOLDER_SYMLINK', self._default_symlink_usage)  # a boolean
        if symlink:
            if parent_calc_folder is not None:
                # I put the symlink to the old parent ./out folder
                remote_symlink_list.append((
                    parent_calc_folder.get_computer().uuid,
                    os.path.join(parent_calc_folder.get_remote_path(), self._restart_copy_from), self._restart_copy_to
                ))
        else:
            # copy remote output dir, if specified
            if parent_calc_folder is not None:
                remote_copy_list.append((
                    parent_calc_folder.get_computer().uuid,
                    os.path.join(parent_calc_folder.get_remote_path(), self._restart_copy_from), self._restart_copy_to
                ))

        # here we may create an aiida.EXIT file
        create_exit_file = settings_dict.pop('ONLY_INITIALIZATION', False)
        if create_exit_file:
            exit_filename = tempfolder.get_abs_path('{}.EXIT'.format(self._PREFIX))
            with open(exit_filename, 'w') as f:
                f.write('\n')

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        # Empty command line by default
        cmdline_params = settings_dict.pop('CMDLINE', [])
        #we commented calcinfo.stin_name and added it here in cmdline_params
        #in this way the mpirun ... pw.x ... < aiida.in
        #is replaced by mpirun ... pw.x ... -in aiida.in
        # in the scheduler, _get_run_line, if cmdline_params is empty, it
        # simply uses < calcinfo.stin_name
        calcinfo.cmdline_params = (list(cmdline_params) + ['-in', self._INPUT_FILE_NAME])
        #calcinfo.stdin_name = self._INPUT_FILE_NAME
        #calcinfo.stdout_name = self._OUTPUT_FILE_NAME

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = (list(cmdline_params) + ['-in', self._INPUT_FILE_NAME])
        #calcinfo.stdin_name = self._INPUT_FILE_NAME
        codeinfo.stdout_name = self._OUTPUT_FILE_NAME
        codeinfo.code_uuid = code.uuid
        calcinfo.codes_info = [codeinfo]

        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list

        # Retrieve by default the output file and the xml file
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self._OUTPUT_FILE_NAME)
        calcinfo.retrieve_list.append(self._VEL_FILE)
        calcinfo.retrieve_list.append(self._POS_FILE)
        calcinfo.retrieve_list.append(self._FOR_FILE)
        calcinfo.retrieve_list.append(self._EVP_FILE)
        calcinfo.retrieve_list.append(self._DATAFILE_XML)

        settings_retrieve_list = settings_dict.pop('ADDITIONAL_RETRIEVE_LIST', [])

        if settings_dict.pop('ALSO_BANDS', False):
            # To retrieve also the bands data
            settings_retrieve_list.append([
                os.path.join(self._OUTPUT_SUBFOLDER, self._PREFIX + '.save', 'K*[0-9]', 'eigenval*.xml'), '.', 2
            ])

        calcinfo.retrieve_list += settings_retrieve_list
        calcinfo.retrieve_list += self._internal_retrieve_list

        if settings_dict:
            try:
                Parserclass = self.get_parserclass()
                parser = Parserclass(self)
                parser_opts = parser.get_parser_settings_key()
                settings_dict.pop(parser_opts)
            except (KeyError, AttributeError):  # the key parser_opts isn't inside the dictionary
                raise InputValidationError(
                    'The following keys have been found in '
                    'the settings input node, but were not understood: {}'.format(','.join(list(settings_dict.keys())))
                )

        return calcinfo
