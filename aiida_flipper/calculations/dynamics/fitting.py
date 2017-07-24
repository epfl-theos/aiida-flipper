from aiida.orm.calculation.chillstep import ChillstepCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm import Data, load_node, Calculation
from aiida.orm.calculation.inline import optional_inline

@optional_inline
def rattle_randomly_structure_inline(structure, params):
    #~ from ase.constraints import FixAtoms
    #~ from random import randint
    elements_to_rattle = params.dict.elements
    stdev = params.dict.stdev
    nr_of_configurations = params.dict.nr_of_configurations
    indices_to_rattle = [i for i,k in enumerate(structure.get_site_kindnames()) if k in elements_to_rattle]
    positions = structure.get_ase().positions
    new_positions = np.repeat(np.array([positions]), nr_of_configurations, axis=0)
    print new_positions.shape

    for idx in indices_to_rattle:
        new_positions[:,idx,:] += np.random.normal(0, stdev, (nr_of_configurations, 3))


    # final_positions = np.concatenate(([positions], new_positions))
    array = ArrayData()
    array.set_array('symbols', np.array(structure.get_site_kindnames()))
    array.set_array('positions', new_positions)
    array._set_attr('units|positions', 'angstrom')
    return dict(rattled_positions=array)
    

class FittingFlipper1RandomlyDisplacedPosCalculation(ChillstepCalculation):
    def start(self):
        # So, I have a structure that should be at the energetic minumum.
        # I will produce a trajectory that comes from randomly displacing
        # the pinball atoms.
        self.goto(self.launch_calculations)
        return {'rattled_position':rattle_randomly_structure_inline(structure=self.inp.structure, params=self.inp.rattling_parameters, store=False)['rattled_positions')

    def launch_calculations(self):
        flipper_calc = self.inp.flipper_code.new_calc()
        
        
