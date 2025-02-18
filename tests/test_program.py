import numpy as np
import pytest
from sdim import Program, Circuit
from sdim import MeasurementResult

MEASUREMENT_DTYPE = np.dtype([
    ('qudit_index', np.int64),
    ('meas_round', np.int64),
    ('shot', np.int64),
    ('deterministic', np.bool_),
    ('measurement_value', np.int64)
])

def test_measurement_format():
    circuit = Circuit(dimension=3, num_qudits=2)
    circuit.append("M", [0,1])
    program = Program(circuit)
    result = program.simulate(shots=1)
    assert result == [MeasurementResult(0, True, 0), MeasurementResult(1, True, 0)] # flattened list
    assert program.measurement_results == [[[MeasurementResult(0, True, 0)]], [[MeasurementResult(1, True, 0)]]] # list of list of list

    circuit = Circuit(dimension=3, num_qudits=2)
    circuit.append("M", [0,1])
    circuit.append("M", [0,1])
    program = Program(circuit)
    result = program.simulate(shots=1)
    assert result == [MeasurementResult(0, True, 0), MeasurementResult(0, True, 0), MeasurementResult(1, True, 0), MeasurementResult(1, True, 0)] #still flattened list
    assert program.measurement_results == [[[MeasurementResult(0, True, 0)],[MeasurementResult(0, True, 0)]], [[MeasurementResult(1, True, 0)], [MeasurementResult(1, True, 0)]]]

    circuit = Circuit(dimension=3, num_qudits=2)
    circuit.append("M", [0,1])
    program = Program(circuit)
    result = program.simulate(shots=2, force_tableau=True)
    assert result == [[[MeasurementResult(0, True, 0), MeasurementResult(0, True, 0)]], [[MeasurementResult(1, True, 0), MeasurementResult(1, True, 0)]]]
    assert program.measurement_results == [[[MeasurementResult(0, True, 0), MeasurementResult(0, True, 0)]], [[MeasurementResult(1, True, 0), MeasurementResult(1, True, 0)]]]
    assert result == program.measurement_results

    circuit = Circuit(dimension=3, num_qudits=2)
    circuit.append("M", [0,1])
    circuit.append("M", [0,1])
    program = Program(circuit)
    result = program.simulate(shots=2, force_tableau=True)
    assert result == [[[MeasurementResult(0, True, 0), MeasurementResult(0, True, 0)], [MeasurementResult(0, True, 0), MeasurementResult(0, True, 0)]], 
                      [[MeasurementResult(1, True, 0), MeasurementResult(1, True, 0)], [MeasurementResult(1, True, 0), MeasurementResult(1, True, 0)]]] # no longer flattened
    assert program.measurement_results == [[[MeasurementResult(0, True, 0), MeasurementResult(0, True, 0)], [MeasurementResult(0, True, 0), MeasurementResult(0, True, 0)]], 
                                           [[MeasurementResult(1, True, 0), MeasurementResult(1, True, 0)], [MeasurementResult(1, True, 0), MeasurementResult(1, True, 0)]]]
    assert result == program.measurement_results


def test_results_to_array():
    # Test 2D list of MeasurementResult objects
    results = [[MeasurementResult(0, True, 1), MeasurementResult(0, True, 2)],
               [MeasurementResult(2, True, 3), MeasurementResult(2, True, 4)]]
    expected = np.array([[(0, 0, 0, True, 1), (0, 1, 0, True, 2)],
                            [(2, 0, 0, True, 3), (2, 1, 0, True, 4)]],
                            dtype=MEASUREMENT_DTYPE)
    converted = Program._results_to_array(results)
    np.testing.assert_array_equal(converted, expected)

    results = [[MeasurementResult(0, True, 1)], [MeasurementResult(1, True, 2)],
               [MeasurementResult(2, True, 3)], [MeasurementResult(3, True, 4)]]
    expected = np.array([[(0, 0, 0, True, 1)], [(1, 0, 0, True, 2)],
                            [(2, 0, 0, True, 3)], [(3, 0, 0, True, 4)]],
                            dtype=MEASUREMENT_DTYPE)
    converted = Program._results_to_array(results)
    np.testing.assert_array_equal(converted, expected)

    results = [[[MeasurementResult(0, True, 1), MeasurementResult(0, True, 2)],
               [MeasurementResult(2, True, 3), MeasurementResult(2, True, 4)]],
               [[MeasurementResult(0, True, 1), MeasurementResult(0, True, 2)],
               [MeasurementResult(2, True, 3), MeasurementResult(2, True, 4)]]]
    expected = np.array([[[[0, 0, 0, True, 1], [0, 1, 0, True, 2]],
                            [[2, 0, 0, True, 3], [2, 1, 0, True, 4]]],
                        [[[0, 0, 1, True, 1], [0, 1, 1, True, 2]],
                            [[2, 0, 1, True, 3], [2, 1, 1, True, 4]]]],
                            dtype=MEASUREMENT_DTYPE)
    
    circuit = Circuit(dimension=3, num_qudits=2)
    circuit.append("M", [0,1])
    program = Program(circuit)
    program.simulate()
    results = program.measurement_results
    expected = np.array([[(0, 0, 0, True, 0)], [(1, 0, 0, True, 0)]],
                            dtype=MEASUREMENT_DTYPE)
    converted = Program._results_to_array(results)
    np.testing.assert_array_equal(converted, expected)

def test_combine_results():
    # Test case 1: Single shot measurement
    MEASUREMENT_DTYPE = np.dtype([
        ('qudit_index', np.int64),
        ('meas_round', np.int64),
        ('shot', np.int64),
        ('deterministic', np.bool_),
        ('measurement_value', np.int64)
    ])

    circuit = Circuit(dimension=3, num_qudits=2)
    circuit.append("M", [0,1])
    program = Program(circuit)
    program.simulate()
    # reference_results = program._results_to_array(program.measurement_results)
    # empty_test = np.empty((2, reference_results.shape[1], 1), 
    #                        dtype=MEASUREMENT_DTYPE)    
    test_measurements = np.array([
        [[(0, 0, 0, True, 0)]],  # Measurement for qudit 0 (1 round, 1 shot)
        [[(1, 0, 0, True, 0)]]   # Measurement for qudit 1 (1 round, 1 shot)
    ], dtype=MEASUREMENT_DTYPE)
    program._combine_results(test_measurements)
    assert program.measurement_results == [[[MeasurementResult(0, True, 0), MeasurementResult(0, True, 0)]], [[MeasurementResult(1, True, 0), MeasurementResult(1, True, 0)]]]

def test_build_ir():
    ir_dtype = np.dtype([
        ('gate_id', np.int64),
        ('qudit_index', np.int64),
        ('target_index', np.int64)
    ])
    c = Circuit(dimension=3, num_qudits=2)
    c.append("H", 0)
    c.append("CNOT", 0, 1)
    c.append("M", [0, 1])
    p = Program(c)
    extra_shots = 1
    ir, noise = p._build_ir(p.circuits, extra_shots)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)

    ir, noise = p._build_ir(p.circuits, extra_shots+1)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)

    c.append("N1", 0, prob=1.0, noise_channel='f')
    ir, noise = p._build_ir(p.circuits, extra_shots)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1), (17, 0, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)
    assert np.any(noise[0][0][0])

    c.append("N1", 1, prob=1.0, noise_channel='p')
    ir, noise = p._build_ir(p.circuits, extra_shots)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1), (17, 0, -1), (17, 1, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)
    assert np.any(noise[1][0][1])

    c.append("N1", 1, prob=1.0, noise_channel='d')
    ir, noise = p._build_ir(p.circuits, extra_shots)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1), (17, 0, -1), (17, 1, -1), (17, 1, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)
    assert np.any(noise[2][0])

   

    
