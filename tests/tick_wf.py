from aiida.orm.calculation.chillstep import tick_all
from aiida_scripts.job_utils.submit_now import submit_jobs
from aiida.daemon.execmanager import update_jobs, retrieve_jobs


update_jobs()
retrieve_jobs()
#~ tick_all(dry_run=False)
tick_all(dry_run=True)
submit_jobs()

