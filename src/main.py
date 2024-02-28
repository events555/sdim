import cProfile
import pstats
from chp_parser import read_circuit
from program import Program

def main():
    circuit = read_circuit("profiling/random_circuit.chp")
    program = Program(circuit)
    program.simulate()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs()  # Remove the extraneous path from all module names
    stats.sort_stats(pstats.SortKey.TIME)  # Sort the statistics by the cumulative time spent in the function
    stats.print_stats("tableau_simulator")  # Only print statistics for your modules