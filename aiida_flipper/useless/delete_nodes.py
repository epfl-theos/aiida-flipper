#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from six.moves import map
from six.moves import input


def delete_nodes(pks_to_delete, ask_confirmation=True):
    """
    Delete a set of nodes.

    :note: The script will also delete
    all children calculations generated from the specified nodes.

    :param pks_to_delete: a list of the PKs of the nodes to delete
    """
    raise Exception("HAven't implemented things torun without the DbPath")
    from django.db import transaction
    from django.db.models import Q
    from aiida.backends.djsite.db import models
    from aiida.orm import load_node

    # Delete also all children of the given calculations
    # Here I get a set of all pks to actually delete, including
    # all children nodes.
    all_pks_to_delete = set(pks_to_delete)
    for pk in pks_to_delete:
        print(pk)
        all_pks_to_delete.update(models.DbNode.objects.filter(parents__in=pks_to_delete).values_list('pk', flat=True))

    print('I am going to delete {} nodes, including ALL THE CHILDREN'.format(len(all_pks_to_delete)))
    print('of the nodes you specified. Do you want to continue? [y/N]')
    if ask_confirmation:
        answer = input()
    else:
        answer = 'y'

    if answer.strip().lower() == 'y':
        # Recover the list of folders to delete before actually deleting
        # the nodes.  I will delete the folders only later, so that if
        # there is a problem during the deletion of the nodes in
        # the DB, I don't delete the folders
        folders = [load_node(pk).folder for pk in all_pks_to_delete]

        with transaction.atomic():
            # Delete all links pointing to or from a given node
            models.DbLink.objects.filter(Q(input__in=all_pks_to_delete) | Q(output__in=all_pks_to_delete)).delete()
            # now delete nodes
            models.DbNode.objects.filter(pk__in=all_pks_to_delete).delete()

        # If we are here, we managed to delete the entries from the DB.
        # I can now delete the folders
        for f in folders:
            f.erase()


def get_all_children(node_pks, return_values=['id'], follow_calls=False, follow_returns=False):
    """
    Get all the children of given nodes
    :param nodes: one node or an iterable of nodes
    :param args & kwargs: additional query parameters
    :return: a list of aiida objects with all the children of the nodes
    """
    from aiida.backends.djsite.db import models
    from aiida.common.links import LinkType
    try:
        the_node_pks = list(node_pks)
    except TypeError:
        the_node_pks = [node_pks]

    link_types_to_follow = [LinkType.CREATE.value, LinkType.INPUT.value]
    if follow_calls:
        link_types_to_follow.append(LinkType.CALL.value)
    if follow_returns:
        link_types_to_follow.append(LinkType.RETURN.value)
    children = models.DbNode.objects.none()
    q_outputs = models.DbNode.aiidaobjects.filter(
        inputs__pk__in=the_node_pks, input_links__type__in=link_types_to_follow
    ).distinct()
    #~ count = 0
    while q_outputs.count() > 0:
        #~ count += 1
        #~ print count
        outputs = list(q_outputs)
        children = q_outputs | children.all()
        q_outputs = models.DbNode.aiidaobjects.filter(inputs__in=outputs,
                                                      input_links__type__in=link_types_to_follow).distinct()

    #~ return children.filter(*args,**kwargs).distinct()
    return children.values_list(*return_values)


def delete_nodes_serial(pks, follow_calls=False, follow_returns=False, dry_run=False, force=False, time_order=True):
    """
    Delete a set of nodes in a serialway, if deleting a bunch of nodes together
    as above functions fails due to memory issues.

    :note: The script will also delete
    all children calculations generated from the specified nodes.

    :param pks_to_delete: a list of the PKs of the nodes to delete
    """
    from django.db import transaction
    from django.db.models import Q
    from aiida.backends.djsite.db import models
    from aiida.orm import load_node, Node
    from aiida.common.exceptions import NotExistent
    from aiida.orm.querybuilder import QueryBuilder
    # Delete also all children of the given calculations
    # Here I get a set of all pks to actually delete, including
    # all children nodes.
    if not pks:
        print('Nothing to delete')
        return

    if time_order:
        pks_to_delete = QueryBuilder().append(Node, project='id', filters={
            'id': {
                'in': pks
            }
        }, tag='n').order_by({
            'n': {
                'ctime': {
                    'order': 'desc'
                }
            }
        }).all()
    else:
        pks_to_delete = pks
    #~ print "I was given {} valid pks".format(len(pks_to_delete))
    for pk, in pks_to_delete:

        #~ pks_set_to_delete_1 = set([pk])
        pks_set_to_delete = set([pk])
        try:
            #~ pks_set_to_delete.update(zip(*QueryBuilder().append(Node, filters={'id':pk}, tag='parent').append(Node, descendant_of='parent',project='id').all())[0])
            #~ pks_set_to_delete_1.update(zip(*QueryBuilder().append(Node, filters={'id':pk}, tag='parent').append(Node, descendant_of='parent',project='id').all())[0])
            pks_set_to_delete.update(
                _ for _, in get_all_children(pk, follow_calls=follow_calls, follow_returns=follow_returns)
            )
        except IndexError:
            pass
        #~ print pks_set_to_delete
        #~ return
        #~ pks_set_to_delete.update(models.DbNode.objects.filter(
        #~ parents__in=pks_to_delete).values_list('pk', flat=True)
        #~ parents = pk).values_list('pk', flat=True))

        if dry_run:
            print('I would have deleted', ' '.join(map(str, sorted(pks_set_to_delete))))
            continue
        print("I'm deleting", ' '.join(map(str, sorted(pks_set_to_delete))))
        if not (force) and input('Continue?').lower() != 'y':
            continue

        #~ print "Deleting",pk,"and parents", pks_set_to_delete_2
        #~ print pks_set_to_delete_1 == pks_set_to_delete_2
        #~ continue
        # Recover the list of folders to delete before actually deleting
        # the nodes.  I will delete the folders only later, so that if
        # there is a problem during the deletion of the nodes in
        # the DB, I don't delete the folders
        folders = [load_node(_).folder for _ in pks_set_to_delete]

        with transaction.atomic():
            # Delete all links pointing to or from a given node
            models.DbLink.objects.filter(Q(input__in=pks_set_to_delete) | Q(output__in=pks_set_to_delete)).delete()
            # now delete nodes
            models.DbNode.objects.filter(pk__in=pks_set_to_delete).delete()

        # If we are here, we managed to delete the entries from the DB.
        # I can now delete the folders
        for f in folders:
            f.erase()


if __name__ == '__main__':
    from aiida.backends.utils import load_dbenv, is_dbenv_loaded
    if not is_dbenv_loaded():
        load_dbenv()
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser()
    parser.add_argument('pks', nargs='+', type=int)
    # parser.add_argument('--serial', action='store_true')
    parser.add_argument('-c', '--follow-calls', help='follow also the calls', action='store_true')
    parser.add_argument('-n', '--dry-run', help='dry run, does not delete', action='store_true')
    parser.add_argument('-f', '--force', help='force deletion', action='store_true')
    parsed_args = parser.parse_args(sys.argv[1:])
    #~ if parsed_args.serial:
    delete_nodes_serial(**vars(parsed_args))
    #~ else:
    #~ delete_nodes(parsed_args.pks, ask_confirmation=False)
