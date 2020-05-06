#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from .delete_nodes import delete_nodes_serial
from aiida.common.links import LinkType
from aiida.orm import load_node, Node
from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm.calculation import Calculation
import sys


def delete_calcs_rec(master_pk, dry_run, force):
    """
    First I delete everything I called, ordered by time.
    Then I delete myself"""
    qb = QueryBuilder().append(ChillstepCalculation, filters={
        'id': master_pk
    }).append(Calculation, tag='c', project='id').order_by({'c': {
        'ctime': {
            'order': 'desc'
        }
    }})
    # By deleting newest first, I avoid going nuts on the transitive closure!
    for cpk, in qb.all():
        print('First deleteing', cpk)
        delete_calcs_rec(cpk, dry_run, force)
    delete_nodes_serial([master_pk], dry_run=dry_run, force=force)
    #~ sys.exit(0)


def main_delete_chillsteps(pks, dry_run=False, force=False):
    """
    Delete a set of nodes in a serialway, if deleting a bunch of nodes together
    as above functions fails due to memory issues.

    :note: The script will also delete
    all children calculations generated from the specified nodes.

    :param pks_to_delete: a list of the PKs of the nodes to delete
    """
    #~ from django.db import transaction
    #~ from django.db.models import Q
    #~ from aiida.backends.djsite.db import models
    # Delete also all children of the given calculations
    # Here I get a set of all pks to actually delete, including
    # all children nodes.
    if not pks:
        print('Nothing to delete')
        return

    pks_to_delete = QueryBuilder().append(ChillstepCalculation, project='id', filters={
        'id': {
            'in': pks
        }
    }, tag='cs').all()
    print('I was given {} valid pks'.format(len(pks_to_delete)))
    for pk, in pks_to_delete:
        delete_calcs_rec(pk, dry_run, force)


if __name__ == '__main__':
    from aiida.backends.utils import load_dbenv, is_dbenv_loaded
    if not is_dbenv_loaded():
        load_dbenv()
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser()
    parser.add_argument('pks', nargs='+', type=int)
    # parser.add_argument('--serial', action='store_true')
    #~ parser.add_argument('-c', '--follow-calls', help='follow also the calls', action='store_true')
    parser.add_argument('-n', '--dry-run', help='dry run, does not delete', action='store_true')
    parser.add_argument('-f', '--force', help='force deletion', action='store_true')
    parsed_args = parser.parse_args(sys.argv[1:])
    #~ if parsed_args.serial:
    main_delete_chillsteps(**vars(parsed_args))
    #~ else:
    #~ delete_nodes(parsed_args.pks, ask_confirmation=False)
