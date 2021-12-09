#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import numpy

import simtk
import simtk.unit as units
import simtk.openmm as openmm
from simtk.openmm.app import *


#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# DEFAULT PARAMETERS
#=============================================================================================

natoms = 216                             # number of particles

# WCA fluid parameters (argon).
mass     = 39.9 * units.amu              # reference mass
sigma    = 3.4 * units.angstrom          # reference lengthscale
epsilon  = 120.0 * units.kelvin * kB     # reference energy

r_WCA    = 2.**(1./6.) * sigma           # interaction truncation range
tau = units.sqrt(sigma**2 * mass / epsilon) # characteristic timescale

# Simulation conditions.
temperature = 0.824 / (kB / epsilon)     # temperature
kT = kB * temperature                    # thermal energy
beta = 1.0 / kT                          # inverse temperature
density  = 0.96 / sigma**3               # default density
stable_timestep = 0.001 * tau            # stable timestep
collision_rate = 1 / tau                 # collision rate for Langevin interator

# Dimer potential parameters.
h = 9.0 * kT                             # barrier height
#h = 5.0 * kT                             # barrier height (DEBUG)
r0 = r_WCA                               # compact state distance
w = 0.5 * r_WCA                          # extended state distance is r0 + 2*w

#=============================================================================================
# WCA Fluid
#=============================================================================================

def WCAFluid(N=natoms, density=density, mm=None, mass=mass, epsilon=epsilon, sigma=sigma):
    """
    Create a Weeks-Chandler-Andersen system.
    OPTIONAL ARGUMENTS
    N (int) - total number of atoms (default: 150)
    density (float) - N sigma^3 / V (default: 0.96)
    sigma
    epsilon
    """

    # Choose OpenMM package.
    if mm is None:
        mm = openmm

    # Create system
    system = mm.System()

    # Compute total system volume.
    volume = N / density

    # Make system cubic in dimension.
    length = volume**(1.0/3.0)
    # TODO: Can we change this to use tuples or 3x3 array?
    a = units.Quantity(numpy.array([1.0, 0.0, 0.0], numpy.float32), units.nanometer) * length/units.nanometer
    b = units.Quantity(numpy.array([0.0, 1.0, 0.0], numpy.float32), units.nanometer) * length/units.nanometer
    c = units.Quantity(numpy.array([0.0, 0.0, 1.0], numpy.float32), units.nanometer) * length/units.nanometer
    print "box edge length = %s" % str(length)
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # Add particles to system.
    for n in range(N):
        system.addParticle(mass)

    # Create nonbonded force term implementing Kob-Andersen two-component Lennard-Jones interaction.
    energy_expression = '4.0*epsilon*((sigma/r)^12 - (sigma/r)^6) + epsilon'

    # Create force.
    force = mm.CustomNonbondedForce(energy_expression)

    # Set epsilon and sigma global parameters.
    force.addGlobalParameter('epsilon', epsilon)
    force.addGlobalParameter('sigma', sigma)

    # Add particles
    for n in range(N):
        force.addParticle([])

    # Set periodic boundary conditions with cutoff.
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    print "setting cutoff distance to %s" % str(r_WCA)
    force.setCutoffDistance(r_WCA)

    # Add nonbonded force term to the system.
    system.addForce(force)

    # Create initial coordinates using random positions.
    coordinates = units.Quantity(numpy.random.rand(N,3), units.nanometer) * (length / units.nanometer)

    # Return system and coordinates.
    return [system, coordinates]

#=============================================================================================
# Functions for simulations
#=============================================================================================
def minimize(platform, system, positions):
    # Create a Context.
    timestep = 1.0 * units.femtoseconds
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator, platform)
    # Set coordinates.
    context.setPositions(positions)
    # Compute initial energy.
    state = context.getState(getEnergy=True)
    initial_potential = state.getPotentialEnergy()
    print "initial potential: %12.3f kcal/mol" % (initial_potential / units.kilocalories_per_mole)
    # Minimize.
    openmm.LocalEnergyMinimizer.minimize(context)
    # Compute final energy.
    state = context.getState(getEnergy=True, getPositions=True)
    final_potential = state.getPotentialEnergy()
    positions = state.getPositions(asNumpy=True)
    # Report
    print "final potential  : %12.3f kcal/mol" % (final_potential / units.kilocalories_per_mole)

    return positions

def equilibrate_langevin(system, timestep, collision_rate, temperature, sqrt_kT_over_m, coordinates, platform):
    nsteps = 5000

    print "Equilibrating for %.3f ps..." % ((nsteps * timestep) / units.picoseconds)

    # Create integrator and context.
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)

    # Set coordinates.
    context.setPositions(coordinates)

    # Set Maxwell-Boltzmann velocities
    velocities = sqrt_kT_over_m * numpy.random.standard_normal(size=sqrt_kT_over_m.shape)
    context.setVelocities(velocities)

    # Equilibrate.
    integrator.step(nsteps)

    # Compute energy
    print "Computing energy."
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    print "potential energy: %.3f kcal/mol" % (potential_energy / units.kilocalories_per_mole)

    # Get coordinates.
    state = context.getState(getPositions=True, getVelocities=True)
    coordinates = state.getPositions(asNumpy=True)
    velocities = state.getVelocities(asNumpy=True)
    box_vectors = state.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box_vectors)

    print "Computing energy again."
    context.setPositions(coordinates)
    context.setVelocities(velocities)
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    print "potential energy: %.3f kcal/mol" % (potential_energy / units.kilocalories_per_mole)

    total_energy = compute_energy(context, coordinates, velocities)

    return [coordinates, velocities]

def compute_forces(context, positions):
    """
    Compute forces for given positions.
    """

    context.setPositions(positions)
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True)
    return forces

def compute_energy(context, positions, velocities):
    """
    Compute total energy for positions and velocities.
    """
    # Set positions and velocities.
    context.setPositions(positions)
    context.setVelocities(velocities)
    # Compute total energy.
    state = context.getState(getEnergy=True)
    total_energy = state.getPotentialEnergy() + state.getKineticEnergy()

    #print "potential energy: %.3f kcal/mol" % (state.getPotentialEnergy() / units.kilocalories_per_mole)
    #print "kinetic   energy: %.3f kcal/mol" % (state.getKineticEnergy() / units.kilocalories_per_mole)

    return total_energy

def compute_forces_and_energy(context, positions, velocities):
    """
    Compute total energy for positions and velocities.
    """
    # Set positions and velocities.
    context.setPositions(positions)
    context.setVelocities(velocities)
    # Compute total energy.
    state = context.getState(getForces=True, getEnergy=True)
    forces = state.getForces(asNumpy=True)
    total_energy = state.getPotentialEnergy() + state.getKineticEnergy()

    #print "potential energy: %.3f kcal/mol" % (state.getPotentialEnergy() / units.kilocalories_per_mole)
    #print "kinetic   energy: %.3f kcal/mol" % (state.getKineticEnergy() / units.kilocalories_per_mole)

    return [forces, total_energy]

def compute_potential(context, positions):
    """
    Compute potential energy for positions.
    """
    # Set positions and velocities.
    context.setPositions(positions)
    # Compute total energy.
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    return potential_energy

if __name__ == '__main__':
    print "tau = %.3f ps" % (tau / units.picoseconds)
        # Compute timestep.
    equilibrate_timestep = 2 * stable_timestep
    timestep = 5 * stable_timestep
    print "equilibrate timestep = %.1f fs, switch timestep = %.1f fs" % (equilibrate_timestep / units.femtoseconds, timestep / units.femtoseconds)
    print "temperature = %.1f K" % (temperature / units.kelvin)
    beta = 1.0 / kT
    collision_rate = 1.0 / tau
    print 'collision_rate: ', collision_rate

    #create the WCA dimer system
    [system, coordinates] = WCAFluid()
    print "Creating masses array..."
    nparticles = system.getNumParticles()
    masses = units.Quantity(numpy.zeros([nparticles,3], numpy.float64), units.amu)
    for particle_index in range(nparticles):
        masses[particle_index,:] = system.getParticleMass(particle_index)
    sqrt_kT_over_m = units.Quantity(numpy.zeros([nparticles,3], numpy.float64), units.nanometers / units.picosecond)
    for particle_index in range(nparticles):
        sqrt_kT_over_m[particle_index,:] = units.sqrt(kT / masses[particle_index,0]) # standard deviation of velocity distribution for each coordinate for this atom

    platform_name = "OpenCL"
    precision_model = "single"

    platform = openmm.Platform.getPlatformByName(platform_name)
    if platform_name == "OpenCL":
        platform.setPropertyDefaultValue('OpenCLPrecision', precision_model)

    print "Minimizing energy..."
    coordinates = minimize(platform, system, coordinates)

    # Equilibrate.
    print "Equilibrating..."
    [coordinates, velocities] = equilibrate_langevin(system, equilibrate_timestep, collision_rate, temperature, sqrt_kT_over_m, coordinates, platform)
