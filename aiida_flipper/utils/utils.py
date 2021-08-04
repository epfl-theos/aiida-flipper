# -*- coding: utf-8 -*-
"""Common utilities."""

def get_or_create_input_node(cls, value, store=True):
    """Return a `Node` of a given class and given value.

    If a `Node` of the given type and value already exists, that will be returned, otherwise a new one will be created,
    stored and returned.

    :param cls: the `Node` class
    :param value: the value of the `Node`
    :param store: whether to store the new node
    
    check if we need other datatypes like arraydata
    """
    from aiida import orm

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
