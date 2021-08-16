from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.links import LinkType
from aiida.engine import ToContext, if_, while_, BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.plugins import CalculationFactory

from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults
from aiida_quantumespresso.utils.mapping import update_mapping, prepare_process_inputs
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_flipper.utils.utils import get_or_create_input_node
from aiida_flipper.calculations import functions

