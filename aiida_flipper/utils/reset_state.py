#!/usr/bin/env runaiida

from __future__ import absolute_import
from __future__ import print_function
from .delete_nodes import delete_nodes_serial
from aiida.common.datastructures import calc_states
from sqlalchemy.sql import text
from aiida.orm import load_node
from aiida.orm.querybuilder import QueryBuilder
import six


def set_states(pks, newstate, job_id=None):
    from aiida.orm import JobCalculation
    from aiida.orm.data.folder import FolderData
    from aiida.orm.data.remote import RemoteData
    #~ from aiida.orm import JobCalculation
    for pk in pks:
        calc = load_node(pk)
        assert isinstance(calc, JobCalculation)
        oldstate = calc.get_state()
        if oldstate in ('FAILED', 'PARSINGFAILED') and newstate == 'WITHSCHEDULER':
            outputs_to_del = [n.pk for n in calc.get_outputs() if not isinstance(n, RemoteData)]
            states_to_del = ('PARSING', 'RETRIEVING', 'COMPUTED', 'PARSINGFAILED', 'FAILED')
        elif oldstate in ('FAILED', 'PARSINGFAILED') and newstate == 'COMPUTED':
            outputs_to_del = [n.pk for n in calc.get_outputs() if not isinstance(n, RemoteData)]
            states_to_del = ('PARSING', 'RETRIEVING', 'PARSINGFAILED', 'FAILED')

        elif oldstate in ('SUBMITTING', 'SUBMISSIONFAILED') and newstate == 'TOSUBMIT':
            outputs_to_del = [n.pk for n in calc.get_outputs()]
            states_to_del = ('SUBMITTING', 'SUBMISSIONFAILED')
        elif oldstate in ('SUBMITTING') and newstate == 'WITHSCHEDULER':  # Loris
            outputs_to_del = []
            states_to_del = []
        elif oldstate in ('FAILED', 'RETRIEVALFAILED', 'PARSINGFAILED') and newstate == 'TOSUBMIT':
            outputs_to_del = [n.pk for n in calc.get_outputs()]
            states_to_del = (
                'PARSING', 'RETRIEVING', 'COMPUTED', 'PARSINGFAILED', 'FAILED', 'WITHSCHEDULER', 'SUBMITTING',
                'SUBMISSIONFAILED', 'RETRIEVALFAILED'
            )
            try:
                delete_remote(calc)
            except Exception as e:
                print(e)
        # YOU'RE ALSO DELEATING REMOTEDATA, which is wrong
        elif oldstate in ('RETRIEVALFAILED', 'RETRIEVING') and newstate in ('WITHSCHEDULER', 'COMPUTED'):
            outputs_to_del = []
            states_to_del = ('RETRIEVING', 'COMPUTED', 'RETRIEVALFAILED')
        #~ elif oldstate in ("PARSING", "PARSINGFAILED") and newstate == 'COMPUTED':
        #~ outputs_to_del = [n.pk for n in calc.get_outputs()]
        #~ states_to_del = ('RETRIEVING', 'COMPUTED', 'PARSING', 'PARSINGFAILED')
        elif oldstate in ('FINISHED', 'COMPUTED') and newstate == 'COMPUTED':
            outputs_to_del = [n.pk for n in calc.get_outputs() if not isinstance(n, RemoteData)]
            states_to_del = ('RETRIEVING', 'PARSING', 'FINISHED')
        elif oldstate == newstate:
            states_to_del = None
            outputs_to_del = []
        else:
            print('Cannot deal with oldstate=', oldstate, 'and newstate=', newstate)
            continue

        calc._set_attr('state', newstate)

        if states_to_del is not None:
            sql_commd = '' +\
                'DELETE from db_dbcalcstate '+\
                " WHERE dbnode_id = {} AND state in ('{}')".format(pk, "', '".join(states_to_del))
            QueryBuilder()._impl.get_session().get_bind().execute(text(sql_commd))

        if job_id is not None:
            calc._set_attr('job_id', six.text_type(job_id))

        delete_nodes_serial(outputs_to_del)


def delete_remote(calc):
    from aiida.backends.utils import get_authinfo
    authinfo = get_authinfo(calc.get_computer(), calc.get_user())
    t = authinfo.get_transport()
    t.open()
    remote_user = t.whoami()

    remote_working_directory = authinfo.get_workdir().format(username=remote_user)
    t.chdir(remote_working_directory)
    t.chdir(calc.uuid[:2])
    t.chdir(calc.uuid[2:4])
    #t.mkdir(calcinfo.uuid[4:])
    # LEONID: Here, to be able to restart fAILED calculations, I tell him to ignore existing
    t.rmtree(calc.uuid[4:])
    t.close()


def reset_chillers(pks, newstate, output_links=[], step=None):
    from aiida.orm.calculation.chillstep import ChillstepCalculation

    for pk in pks:
        calc = load_node(pk)
        assert isinstance(calc, ChillstepCalculation)
        oldstate = calc.get_state()
        if oldstate == newstate:
            states_to_del = None
        elif oldstate == 'FAILED' and newstate == 'WITHSCHEDULER':
            states_to_del = ('FAILED',)
        elif oldstate == 'FINISHED' and newstate == 'WITHSCHEDULER':
            states_to_del = ('FINISHED',)
        else:
            print('Cannot deal with oldstate=', oldstate, 'and newstate=', newstate)
            continue
        outputs_to_del = [v.pk for k, v in calc.get_outputs_dict().items() if k in output_links]
        calc._set_attr('state', newstate)

        if states_to_del is not None:
            sql_commd = '' +\
                'DELETE from db_dbcalcstate '+\
                " WHERE dbnode_id = {} AND state in ('{}')".format(pk, "', '".join(states_to_del))
            QueryBuilder()._impl.get_session().get_bind().execute(text(sql_commd))

        #~ print outputs_to_del
        if step:
            print('resetting state from', calc.get_attr('_next'), 'to', step)
            calc._set_attr('_next', step)
        delete_nodes_serial(outputs_to_del)


if __name__ == '__main__':
    from aiida.backends.utils import load_dbenv, is_dbenv_loaded
    if not is_dbenv_loaded():
        load_dbenv()
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser()
    parser.add_argument('pks', nargs='+', type=int)
    parser.add_argument('-s', '--state', choices=list(calc_states), default='WITHSCHEDULER')
    parser.add_argument('-c', '--chiller', action='store_true')
    parser.add_argument('--step', help='set next step to that')
    parser.add_argument('--outputs', nargs='+', help='the linknames of the outputs the you need to remove', default=[])
    parser.add_argument('--job-id', type=int, help='give that job ID')
    parsed_args = parser.parse_args(sys.argv[1:])
    if parsed_args.chiller:
        reset_chillers(parsed_args.pks, parsed_args.state, output_links=parsed_args.outputs, step=parsed_args.step)
    else:
        set_states(
            parsed_args.pks,
            parsed_args.state,
            job_id=parsed_args.job_id,
        )
