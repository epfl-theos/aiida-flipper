# -*- coding: utf-8 -*-
"""Common utilities."""
from aiida import orm

def get_or_create_input_node(cls, value, store=True):
    """Return a `Node` of a given class and given value.

    If a `Node` of the given type and value already exists, that will be returned, otherwise a new one will be created,
    stored and returned.

    :param cls: the `Node` class
    :param value: the value of the `Node`
    :param store: whether to store the new node
    
    check if we need other datatypes like arraydata
    """

    if cls in (orm.Bool, orm.Float, orm.Int, orm.Str):

        result = orm.QueryBuilder().append(cls, filters={'attributes.value': value}).first()

        if result is None:
            node = cls(value)
            if store:
                node = node.store()
        else:
            node = result[0]

    elif cls is orm.Dict:
        result = orm.QueryBuilder().append(cls, filters={'attributes': {'==': value}}).first()

        if result is None:
            node = cls(dict=value)
            if store:
                node = node.store()
        else:
            node = result[0]

    else:
        raise NotImplementedError

    return node

# May or may not be required
def get_or_compute_diffusion_from_msd(calc, msd_parameters, add_to_group_name=None):
    """
    Computes the diffusion from the mean-square displacement of a ReplayCalculation or DiffusionCalculation
    using msd_parameters, or retrieves results array, if it was already stored.
    """
    from aiida.orm.calculation.chillstep.user.dynamics import DiffusionCalculation, ReplayCalculation
    from aiida_flipper.calculations.functions import get_diffusion_from_msd

    if isinstance(calc, DiffusionCalculation):
        branches = calc._get_branches()
    elif isinstance(calc, ReplayCalculation):
        try:
            branches = {'traj_0': calc.get_outputs_dict()['total_trajectory']}
        except KeyError:
            return None
    else:
        raise TypeError('calc must be a DiffusionCalculation or ReplayCalculation')

    structure = calc.inp.structure
    if isinstance(msd_parameters, dict):
        parameters = get_or_create_input_node(orm.Dict, msd_parameters, store=True)
    elif isinstance(msd_parameters, orm.Dict):
        parameters = msd_parameters

    # find if msd was already computed with the same parameters
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={'id': calc.id})
    qb.append(orm.TrajectoryData)
    qb.append(orm.CalcFunctionNode, tag='msd_calc')
    qb.append(orm.Dict, with_incoming='msd_calc', project=['*'])
    qb.append(orm.ArrayData, edge_filters={'label': {'like': 'msd_results%'}}, with_outgoing='msd_calc', project=['*'])
    for n in qb.iterall():
        # n[0]: input msd parameters;  n[1]: output msd results
        if (n[0].get_attribute() == parameters.get_attribute()):
            return n[1]  # msd_results was already computed --> return results node

    # msd_results was not computed --> compute it, store and return results node
    if (len(branches) > 0):
        inlinecalc, res = get_diffusion_from_msd(structure=structure, parameters=parameters, **branches)
        inlinecalc.label = calc.label + '-get_diffusion'
        if add_to_group_name is not None:
            group, _ = Group.get_or_create(name=add_to_group_name)
            group.add_nodes(res['msd_results'])
        return res['msd_results']
    else:
        return None
    
def delithiate_structure(structure, element_to_remove='Li'):
    """
    Take the input structure and create two structures from it.
    One structure is "flipper_compatible" whic is essentially the same 
    structure, just that Li is on first places both in kinds and sites
    as required for the flipper; the other structure has no Lithium
    """
    
    assert isinstance(structure, orm.StructureData), "input structure needs to be an instance of {}".format(orm.StructureData)

    pinball_kinds = [kind for kind in structure.kinds if kind.symbol == element_to_remove]

    kindnames_to_delithiate = [kind.name for kind in pinball_kinds]

    non_pinball_kinds = [k for i,k in enumerate(structure.kinds) if k.symbol != element_to_remove]

    non_pinball_sites = [s for s in structure.sites if s.kind_name not in kindnames_to_delithiate]

    pinball_sites = [s for s in structure.sites if s.kind_name in kindnames_to_delithiate]

    delithiated_structure = orm.StructureData()
    pinball_structure = orm.StructureData()

    delithiated_structure.set_cell(structure.cell)
    delithiated_structure.set_attribute('delithiated', True)
    pinball_structure.set_cell(structure.cell)
    pinball_structure.set_attribute('flipper_compatible', True)

    [pinball_structure.append_kind(_) for _ in pinball_kinds]
    [pinball_structure.append_site(_) for _ in pinball_sites]
    [pinball_structure.append_kind(_) for _ in non_pinball_kinds]
    [pinball_structure.append_site(_) for _ in non_pinball_sites]

    [delithiated_structure.append_kind(_) for _ in non_pinball_kinds]
    [delithiated_structure.append_site(_) for _ in non_pinball_sites]

    delithiated_structure.label = delithiated_structure.get_formula(mode='count')
    pinball_structure.label = pinball_structure.get_formula(mode='count')

    return pinball_structure, delithiated_structure

def displace_atoms(structure, sigma):
    """
    :param structure: A valid StructureData instance
    :param sigma: The input parameters, an instance of Float, defining the sigma value
        of the displacement.

    :returns: A structure data instance with shaken positions
    """
    
    import numpy as np
    assert isinstance(sigma, orm.Float), "input sigma needs to be an instance of {}".format(orm.Float)
    assert isinstance(structure, orm.StructureData), "input structure needs to be an instance of {}".format(orm.StructureData)
    atoms = structure.get_ase()
    atoms.rattle(stdev=sigma.value, seed=np.random.randint(65536))
    shaken_structure = orm.StructureData(ase=atoms)
    return shaken_structure

