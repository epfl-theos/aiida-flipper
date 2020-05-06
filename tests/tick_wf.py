from __future__ import absolute_import
from aiida.orm.calculation.chillstep import tick_all
from aiida_scripts.job_utils.submit_now import submit_jobs
from aiida.daemon.execmanager import update_jobs, retrieve_jobs

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()
    parser.add_argument('-n', '--dry-run', action='store_true')
    parsed = parser.parse_args(sys.argv[1:])

    update_jobs()
    retrieve_jobs()
    tick_all(dry_run=parsed.dry_run)
    #~ submit_jobs(maxcalc_running=1)
    submit_jobs()
