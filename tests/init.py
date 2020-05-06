from __future__ import absolute_import
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm import DataFactory, Code, Computer
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.upf import UpfData
import os

c = Computer(
    name='localhost',
    hostname='localhost',
    transport_type='local',
    scheduler_type='pbspro',
    workdir='/scratch/{username}/aiida_run_test/',
    mpirun_command=['mpirun', '-np', '{tot_num_mpiprocs}'],
    default_mpiprocs_per_machine=12
)
c.store()

code = Code()
code.set_computer(c)
code.set_remote_computer_exec((c, '/home/kahle/bin/pw.x'))
code.set_input_plugin_name('quantumespresso.pw')
code.label = 'pw-5.3.0'
code.store()

code = Code()
code.set_computer(c)
code.set_remote_computer_exec((c, '/home/kahle/git/pinball/espresso-5.2.0/PW/src/pw.x'))
code.set_input_plugin_name('quantumespresso.flipper')
code.label = 'flipper'
code.store()
