from .circuit import Circuit
from .program import Program
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from pathlib import Path
import os
import re
import shlex
import numpy as np

class DEMInstructionType(Enum):
    """
    Enum that lists the valid lines types in a DEM.  Repeat instructions envelop their block with curly braces {...}.
    """
    DIMENSION = 0
    ERROR = 1
    DETECTOR = 2
    LOGICAL_OBSERVABLE = 3
    SHIFT_DETECTORS = 4
    REPEAT = 5

class DEMTargetType(Enum):
    """
    Enum that lists the valid types of positional targets and symbols in a DEM line.
    """
    DETECTOR = 1
    LOGICAL_OBSERVABLE = 2
    SEPARATOR = 3
    

@dataclass
class DEMInstruction:
        """
        Stores the information in a single line or block of a DEM.

        Attributes:
            instruction_type (DEMInstructionType): Indicates the type of entry (e.g. an error mechanism)
            id_num  (int | None): Sequential or positional data identifying a target (e.g. a logical observable or shift declaration) type.  The id is absolute compared to the relative indices and shift instructions within an .sdem file.  This entry is None by default for error mechanisms.
            argument (float | Tuple[int, int, int, int] | None): 

        """
        instruction_type : DEMInstructionType
        id_num : int | None = None      #Error mechanisms have no id numbers, only detectors and logicals do
        argument : float | Tuple[int, int, int, int] | None = None      #Coordinates for detection events, probabilities for errors
        detector_packed_target_flip_pairs : set | None = None
        logical_observable_packed_target_flip_pairs : set | None = None
        repeat_count : int = 0
        #TODO: Build REPEAT blocks
        # repeat_block : DEMInstruction = None


        def add_packed_pair(self, target_type: DEMTargetType, target : int, flip : int):

            if target_type not in (DEMTargetType.DETECTOR, DEMTargetType.LOGICAL_OBSERVABLE):
                raise ValueError("The pair must be either a detector or logical observable.")

            target_set = self.detector_packed_target_flip_pairs if target_type == DEMTargetType.DETECTOR else self.logical_observable_packed_target_flip_pairs
            
            # Mask the inputs and pack them into a single 64-bit integer
            target = target & 0xFFFFFFFF
            flip = flip & 0xFFFFFFFF     
            pair = (flip << 32) | target

            target_set.add(pair)

            return
        

        def get_unpacked_pairs(self, target_type: DEMTargetType):

            if target_type not in (DEMTargetType.DETECTOR, DEMTargetType.LOGICAL_OBSERVABLE):
                raise ValueError("The pair must be either a detector or logical observable.")
            
            target_set = self.detector_packed_target_flip_pairs if target_type == DEMTargetType.DETECTOR else self.logical_observable_packed_target_flip_pairs

            pairs = []

            for packed_pair in target_set:

                target = packed_pair & 0xFFFFFFFF
                flip = packed_pair >> 32
                pairs.append((target, flip))

            return pairs
        

        def __iter__(self):

            if self.instruction_type == DEMInstructionType.ERROR:
                yield self.get_unpacked_pairs(DEMTargetType.DETECTOR)
                yield self.get_unpacked_pairs(DEMTargetType.LOGICAL_OBSERVABLE)
            else:
                yield NotImplemented
        

        def __str__(self):

            if self.instruction_type == DEMInstructionType.ERROR:

                detector_flips = self.get_unpacked_pairs(DEMTargetType.DETECTOR)
                logical_flips = self.get_unpacked_pairs(DEMTargetType.LOGICAL_OBSERVABLE)

                flip_string = ""

                for target, flip in detector_flips:
                    flip_string = flip_string+ f"D{target}={flip} "

                for target, flip in logical_flips:
                    flip_string = flip_string + f"L{target}={flip} "


                return f"ERROR prob={self.argument} {flip_string}"
            
            elif self.instruction_type in (DEMInstructionType.DETECTOR, DEMInstructionType.LOGICAL_OBSERVABLE):
                #TODO
                name= "DETECTOR" if self.instruction_type == DEMInstructionType.DETECTOR else "LOGICAL_OBSERVABLE"
                coord = "" if self.argument == None else f"coord={self.argument}"

                return f"{name} {coord} {name[:1]}{self.id_num}"
            
            elif self.instruction_type == "REPEAT":
                raise NotImplementedError
            
            else:
                raise NotImplementedError

            return
        

        def __eq__(self, other):

            if not isinstance(other, DEMInstruction):
                return False

            if self.instruction_type != other.instruction_type:
                return False
            
            if self.instruction_type == DEMInstructionType.ERROR:
                return (self.detector_packed_target_flip_pairs == other.detector_packed_target_flip_pairs and
                        self.logical_observable_packed_target_flip_pairs == other.logical_observable_packed_target_flip_pairs)

            elif self.instruction_type in (DEMInstructionType.DETECTOR, DEMInstructionType.LOGICAL_OBSERVABLE):
                return NotImplemented

            else:
                return NotImplemented
        




@dataclass
class DetectorErrorModel:
    dimension : int = None
    instructions : list[DEMInstruction] = field(default_factory=list)
    shift_list : list[Tuple[float, list, list]] = field(default_factory=list)
    num_detectors : int = 0
    num_logicals : int = 0

    def add_error_mechanism(self, probability : float, detector_event_pairs : list, logical_observable_event_pairs : list):

        #TODO: Input sanitization (check that event pairs are correctly formatted)

        error = DEMInstruction(
                    instruction_type=DEMInstructionType.ERROR,
                    argument=probability,
                    detector_packed_target_flip_pairs=set(),
                    logical_observable_packed_target_flip_pairs=set()
                    )
        
        for target, flip in detector_event_pairs:
            error.add_packed_pair(DEMTargetType.DETECTOR, target, flip)
            #TODO: Make the following variables resistant to breaking under relative numbering (i.e. REPEAT blocks)
            if target >= self.num_detectors:
                self.num_detectors = target + 1

        for target, flip in logical_observable_event_pairs:
            error.add_packed_pair(DEMTargetType.LOGICAL_OBSERVABLE, target, flip)
            #TODO: Make the following variables resistant to breaking under relative numbering (i.e. REPEAT blocks)
            if target >= self.num_logicals:
                self.num_logicals = target + 1

        self.instructions.append(error)

        return error
    

    def add_detector_label(self, target_type : DEMTargetType, coord : Tuple[int, int, int, int] | None = None, id_num : int = 0):

        event = DEMInstruction(
            instruction_type=target_type, 
            argument=coord,
            id_num=id_num
        )

        self.instructions.append(event)

        return event


    def read_from_file(self, filepath : str, overwrite : bool = False):
        """
        Parser that reads a .sdem file into a DetectorErrorModel object.
        """

        # Check that list is empty first
        if self.instructions:
            if (overwrite):
                self.dimension = None
                self.instructions.clear()
            else:
                raise ValueError("The list of instructions is non-empty.  Use the overwrite parameter if you are sure you want to replace it with the contents of the file.")


        # # Get the directory of the current script
        # script_dir = os.path.dirname(os.path.realpath(__file__))
        # parent_dir = os.path.join(script_dir, '..')

        # # Construct the absolute path to the file
        # abs_file_path = os.path.join(parent_dir, filename)

        with open(filepath, 'r') as file:
            lines = file.readlines()

        # Find the line with only '#'
        start_index = next(i for i, line in enumerate(lines) if line.strip() == '#')

        # Extract the lines after '#'
        entries = lines[start_index + 1:]

        # Break up lines for their parameters in unix shell convention and append them accordingly
        for j, line in enumerate(entries):
            if not line.strip():
                continue

            parts = shlex.split(line)

            if (len(parts) < 2):
                raise ValueError(f"Not enough parameters specified on line {start_index + j + 1}, which reads:\n {line}")

            instr_name = parts[0].upper()
            params = [text for text in parts[1:] if '=' in text]
            argument = None

            # Instruction string to type dictionary
            

            # Used to calculate absolute detector indices and coordinates
            relative_offset = 0

            # Cases for DEM instructions
            if instr_name == "ERROR":
                error = DEMInstruction(
                    instruction_type=DEMInstructionType.ERROR,
                    argument=0,
                    detector_packed_target_flip_pairs=set(),
                    logical_observable_packed_target_flip_pairs=set(),
                    repeat_count=0
                    )

                #TODO: Add in support for suggested separations.  Right now, shlex-style parameter processing ignores separator ^
                for arg in params:
                    arg_parts = arg.split('=')

                    if arg_parts[0] == "prob":
                        error.argument = float(arg_parts[1])

                    # Checking if the entry matches a detector or logical to a shift, e.g. D0=2, L5=7
                    elif match := re.fullmatch(r"^([LD])(\d+)", arg_parts[0]):
                        event = DEMTargetType.DETECTOR if match.group(1) == "D" else DEMTargetType.LOGICAL_OBSERVABLE
                        # Mask the values as 32 bit integers and pack them into a single pair
                        target = int(match.group(2))
                        flip = int(arg_parts[1])
                        error.add_packed_pair(event, target, flip)

                # Append error mechanism here
                self.instructions.append(error)



            elif instr_name in ("DETECTOR", "LOGICAL_OBSERVABLE", "SHIFT_DETECTORS"):

                for arg in params:
                    arg_parts = arg.split('=')

                    if arg_parts[0] == "coord":
                        #Check that the coordinate matches (x, y, z, t) format
                        if match := re.fullmatch(r"^\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", arg_parts[1]):
                            x, y, z, t = map(int, match.groups())
                            argument = (x, y, z, t)
                        else:
                            raise ValueError(f"The string f{arg_parts[1]} on line {start_index + j + 1} is an invalid spacetime coordinate, which must take the form (x, y, z, t).")

                    # Checking if the entry matches a detector or logical to a flip, e.g. D0=2, L5=7
                    elif match := re.fullmatch(r"^[LD](\d+)", arg_parts[0]):
                        target = int(match.group(1))

                    # Checking if the entry matches a shift in detector indices
                    elif match := re.fullmatch(r"(\d+)", arg_parts[0]):
                        target = int(match.group(1))
                        relative_offset += target


                    else:
                        raise ValueError(f"The string f{arg} on line {start_index + j + 1} is not a valid parameter line of paramters for a detector or logical observable.")

                self.add_detector_label(target_type=DEMInstructionType[instr_name], coord=argument, id_num=target)
            
            elif instr_name == "DIMENSION":
                self.dimension = int(parts[1])

            else:
                raise ValueError(f"Instruction {instr_name} on line {start_index + j + 1} not recognized.")


            if self.dimension is None:
                raise ValueError("Dimension was left unset.  A valid .sdem file must include a line listing the dimension of the corresponding circuit as `DIMENSION Q`, where Q >= 2 is an integer.")
            
        self.merge_errors()

        return
    

    def merge_errors(self):

        new_instructions = []
        processed_mechanisms = [0 for _ in range(len(self.instructions))]

        for j, instruction in enumerate(self.instructions):

            if processed_mechanisms[j] == 0 and instruction.instruction_type == DEMInstructionType.ERROR:

                prob = instruction.argument

                for k in range(j + 1, len(self.instructions)):
                    if instruction == self.instructions[k]:
                        additional_prob = self.instructions[k].argument
                        # New total probability
                        prob = prob * (1 - additional_prob) + (1 - prob) * additional_prob
                        # Mark this mechanism processed
                        processed_mechanisms[k] = 1

                instruction.argument = prob
                new_instructions.append(instruction)
                processed_mechanisms[j] = 1


        self.instructions = new_instructions

        return
    
    @classmethod
    def from_circuit(cls, circuit : Circuit, merge_initial_errors : bool = True):
        dimension = circuit.dimension
        noise_probabilities = []

        dem_from_circuit = cls()
        dem_from_circuit.dimension = dimension

        # Linearly process the circuit instructions for any noise events, mark down the probabilities
        # TODO: Support for 2-qudit noise channels
        print(f"Sweeping for error instructions...")
        for instr in circuit.operations:
            if instr.gate_name == "N1":
                channel_prob = float(instr.params['prob']) 
                if instr.params['noise_channel'] == 'd':
                    for _ in range(dimension**2 - 1):
                        noise_probabilities.append(channel_prob / (dimension**2 - 1))
                if instr.params['noise_channel'] in ('f', 'p'):
                    for _ in range(dimension - 1):
                        noise_probabilities.append(channel_prob / (dimension - 1))

            if instr.gate_name == "N2":
                channel_probs = instr.params['prob_dist']
                noise_probabilities.extend(channel_probs[1:])
                
        # Use the Pauli frame sampler to generate detector and logical flip events
        shots = len(noise_probabilities)
        print(f"Sampling {shots} shots to build the DEM...")
        error_enumerator = Program(circuit)
        _, detection_events = error_enumerator.simulate(shots=shots, building_error_mechanism=True)

        # Read out the target index and flip pairs into the circuit
        print(f"Building DEM...")
        for mechanism in range(shots):
            detector_pairs = []
            logical_pairs = []

            for target, d in enumerate(detection_events['detectors']):
                if d['data'][mechanism] != 0:
                    detector_pairs.append((target, d['data'][mechanism]))

            for target, d in enumerate(detection_events['logicals']):
                if d['data'][mechanism] != 0:
                    logical_pairs.append((target, d['data'][mechanism]))

            if len(detector_pairs) > 0 or len(logical_pairs) > 0:
                dem_from_circuit.add_error_mechanism(noise_probabilities[mechanism], detector_pairs, logical_pairs)

        if merge_initial_errors:
            dem_from_circuit.merge_errors()

        return dem_from_circuit
    

    def update_sampler(self):

        self.merge_errors()
        self.shift_list = []

        for instr in self.instructions:
            if instr.instruction_type == DEMInstructionType.ERROR:
                detector_pairs, logical_pairs = list(instr)
                detector_shift = np.zeros(self.num_detectors, dtype=np.int64)
                logical_shift = np.zeros(self.num_logicals, dtype=np.int64)

                for target, flip in detector_pairs:
                   detector_shift[target] = flip

                for target, flip in logical_pairs:
                   logical_shift[target] = flip

                #TODO: Broadcast that the data is laid out like this a bit more obviously
                self.shift_list.append((instr.argument, detector_shift, logical_shift))

        return
    

    def sample(self, shots : int):
        # TODO: Multithread or parallelize this
        detector_samples = []
        logical_samples = []
        
        if not self.shift_list:
            self.update_sampler()

        for _ in range(shots):
            detector_shift = np.zeros(self.num_detectors, dtype=np.int64)
            logical_shift = np.zeros(self.num_logicals, dtype=np.int64)

            for e in self.shift_list:
                # print(f"The error mechanism has with probability {e[0]} to fire")
                select = np.random.choice([0, 1], p=[1 - e[0], e[0]])
                detector_shift = (detector_shift + select * e[1]) % self.dimension
                # print(f"The select was {select} with probability {e[0]}")
                logical_shift = (logical_shift + select * e[2]) % self.dimension

            # print(f"Sample reads with detector shifts: {detector_shift} and logical shifts : {logical_shift}")
            detector_samples.append(detector_shift)
            logical_samples.append(logical_shift)

        return detector_samples, logical_samples

    def __str__(self):
        lines = f"DIMENSION {self.dimension}\n"

        for instr in self.instructions:
            lines = lines + str(instr) + "\n"

        return lines
   
   
    def write_to_file(self, path : str = "./", filename : str = "default.sdem", comment : str = ""):

        full_path = Path(path) / filename
        absolute_path = full_path.resolve()

        with open(absolute_path, 'w') as file:
            file.write(comment + "\n#\n" + str(self))

        return


    def flatten(self):

        return NotImplemented


