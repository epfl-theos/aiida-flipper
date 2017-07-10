#!/bin/bash

verdi devel tests db.query
verdi import ~/repositories/aiida/sssp_eff.aiida


verdi run init.py

# verdi run start_wf.py

verdi computer configure localhost
