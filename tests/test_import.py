#CalculationFactory('quantumespresso.pw')
from __future__ import absolute_import
from __future__ import print_function
from aiida.parsers import ParserFactory
FlipperCalculation = CalculationFactory('quantumespresso.flipper')
CalculationFactory('quantumespresso.hustler')
ParserFactory('quantumespresso.flipper')
calc = FlipperCalculation()
print(calc._plugin_type_string)

#~ from aiida.orm.calculation.job.quantumespresso import flipper
#~ from aiida.orm.calculation.job.quantumespresso.flipper import FlipperCalculation
#~ from aiida.orm.calculation.job.quantumespresso.hustler import HustlerCalculation
#~ from aiida.orm.parsers.plugins.quantumespresso.flipper import FlipperParser
