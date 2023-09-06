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

# This is NOT supposed to be a calcfunction
def get_total_trajectory(workchain, previous_trajectory=None, store=False):
    """Collect all the trajectory segment and concatenate them."""
    qb = orm.QueryBuilder()
    qb.append(orm.WorkChainNode, filters={'uuid': workchain.uuid}, tag='replay')
    # TODO: Are filters on the state of the calculation needed here?
    # TODO: add project on extras.discard_trajectory, traj_d defined to skip them
    qb.append(orm.CalcJobNode, with_incoming='replay',
            edge_filters={'type': LinkType.CALL_CALC.value,
                          'label': {'like': 'iteration_%'}},
            edge_project='label', tag='calc', edge_tag='rc')
    qb.append(orm.TrajectoryData, with_incoming='calc', edge_filters={'label': 'output_trajectory'},
            project=['*'], tag='traj')
    traj_d = {item['rc']['label'].replace('iteration_', 'trajectory_'): item['traj']['*'] for item in qb.iterdict()}  ## if not extras.discard_trajectory

    # adding the trajectory of previous MD run, if it exists
    if previous_trajectory:
        traj_d.update({'trajectory_00': previous_trajectory})

    # if I have produced several trajectories, I concatenate them here: (no need to sort them)
    if (len(traj_d) > 1):
        traj_d['metadata'] = {'call_link_label': 'concatenate_trajectory', 'store_provenance': store}
        traj_d.update({'remove_repeated_last_step': True})
        res = concatenate_trajectory(**traj_d)
        return res['concatenated_trajectory']
    elif (len(traj_d) == 1):
        # no reason to concatenate if I have only one trajectory (saves space in repository)
        return list(traj_d.values())[0]
    else:
        return None
