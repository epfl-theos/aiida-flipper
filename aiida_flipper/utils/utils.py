from aiida.common.hashing import make_hash
from aiida.orm import Calculation
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.upf import UpfData
from aiida.orm.group import Group
from aiida.orm.querybuilder import QueryBuilder

def get_pseudos(
        calc=None, pseudo_family_name=None, structure=None, with_pseudo_in_label=False,
        return_cutoffs=False, min_ecutwfc=0,**kwargs):
    """
    Pseudo family is stored in the input parameters
    """
    if pseudo_family_name is None:
        raise Exception("Provide  a pseudo family name")
    pseudo_dict = {}

    if isinstance(calc, Calculation):
        structure = calc.inp.structure
    elif isinstance(structure, StructureData):
        structure = structure
    else:
        raise Exception('Specify calculation  or input structure')

    # Finding a pseudopotential for every kind:
    for kind in structure.kinds:
        element = kind.symbols[0]
        qb = QueryBuilder()
        qb.append(Group, filters={'name':pseudo_family_name})
        qb.append(UpfData, member_of=Group, filters={'attributes.element':element})
        try:
            pseudo, = qb.first()
        except TypeError:
            raise Exception ("Pseudo for  {} not found".format(element))
        if with_pseudo_in_label:
            pseudo_dict['pseudo_{}'.format(kind.name)] = pseudo
        else:
            pseudo_dict[kind.name] = pseudo
    # I can rewrite the choice of some pseudopotentials a keyword arguments:
    # eg. Li=MyPseudo
    for kn, upfdata in kwargs.iteritems():
        assert isinstance(upfdata, UpfData), '{} is not an instance of {}'.format(upfdata, UpfData)
        if kn not in pseudo_dict.keys():
            print 'WARNING: {} is not in the structure'.format(kn)
        pseudo_dict[kn] = upfdata
    if return_cutoffs:
        ecutwfc, ecutrho = get_suggested_cutoff(pseudo_family_name, pseudo_dict.values(), min_ecutwfc=min_ecutwfc)
        pseudo_dict["ecutwfc"] = ecutwfc
        pseudo_dict["ecutrho"] = ecutrho
    return pseudo_dict


def attach_pseudos(calc, pseudo_family_name, return_cutoffs=True):
    pseudos=get_pseudos(
            calc=calc,
            pseudo_family_name=pseudo_family_name,
            return_cutoffs=return_cutoffs,
        )
    ecutwfc = pseudos.pop('ecutwfc', None)
    ecutrho = pseudos.pop('ecutrho', None)
    for k,v in pseudos.iteritems():
        calc.use_pseudo(v, k)
    return ecutwfc, ecutrho


def get_suggested_cutoff(pseudo_family_name, pseudos, min_ecutwfc=0):
    ecutwfc = min_ecutwfc
    ecutrho = 0.
    try:
        for p in pseudos:
            suggestion_for_this_group = p.get_extra(pseudo_family_name)
            ecutwfc = max([ecutwfc, suggestion_for_this_group['cutoff']])
            ecutrho = max([ecutrho, ecutwfc*suggestion_for_this_group['duality']])

    except AttributeError as e:
        raise AttributeError(
                "raised when asking for cutoff and duality\n"
                "for pseudos {}\n"
                "{}".format(pseudos, e)
            )
    return ecutwfc, ecutrho


def get_or_create_kpoints_from_mesh(mesh):
    raise Exception("No excplicit check")
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh(mesh)
    existing = QueryBuilder().append(KpointsData, filters={'extras.hash':make_hash(kpoints.get_attrs())}).first()
    if existing:
        return existing[0]
    else:
        return kpoints #.store()


def get_or_create_parameters(parameters, store=False):
    hash_ = make_hash(parameters)
    same_hash_results = QueryBuilder().append(ParameterData, filters={'extras.hash':hash_}).all()
    for res, in same_hash_results:
        if res.get_attrs() == parameters:
            return res
    params = ParameterData(dict=parameters)
    if store:
        params.store()
        params.set_extra('hash', hash_)
    return params

