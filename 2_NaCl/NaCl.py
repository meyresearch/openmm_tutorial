from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout


ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
top = Topology()
pos = []
mod = Modeller(top, pos)
mod.addSolvent(ff, model='tip3p', boxSize=Vec3(3, 3, 3)*nanometers,
               positiveIon='Na+', negativeIon='Cl-',
               ionicStrength=9.23*molar)

# Create OpenMM System
system = ff.createSystem(mod.topology, nonbondedMethod=PME,
                        nonbondedCutoff=1*nanometers,
                        constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picoseconds,
                               1*femtoseconds)

system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))

simulation = Simulation(mod.topology, system, integrator)
simulation.context.setPositions(mod.positions)

# Minimize System
print "Minimizing Energy...."
simulation.minimizeEnergy(maxIterations=1000)

# Equilibrate System
print "Doing an equilibration"
simulation.reporters.append(
   PDBReporter('NaCl_traj.pdb', 500))
simulation.context.setVelocitiesToTemperature(300)
simulation.reporters.append(StateDataReporter(stdout, 500, step=True, potentialEnergy=True, temperature=True))
simulation.step(5000)
file = open('NaCl.pdb', 'w')
PDBFile.writeModel(mod.topology,mod.positions, file=file)
