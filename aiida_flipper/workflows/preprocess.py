# -*- coding: utf-8 -*-
from aiida import orm
from aiida.engine.processes.workchains.workchain import WorkChain
from aiida_flipper.utils import utils
from aiida_flipper.calculations import functions

from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.common import AttributeDict, exceptions
from aiida.common.links import LinkType
from aiida.engine import ToContext, if_, while_, BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode

class PreProcessWorkChain(WorkChain):
    """
    WorkChain that takes a primitive structure as its input and makes supercell using Supercellor class,
    makes the pinball and delithiated structures and then performs an scf calculation on the host lattice,
    stashes the charge densities and wavefunctions. It outputs the pinball supercell and RemoteData to be 
    used in all future workchains for performing MD.
    """

    @classmethod
    def define(cls, spec):