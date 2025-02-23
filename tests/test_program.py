import numpy as np
import pytest
from sdim.circuit import Circuit

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
    extra_shots = 1
    ir, noise = c._build_ir(p.circuits, extra_shots)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)

    ir, noise = c._build_ir(p.circuits, extra_shots+1)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)

    c.append("N1", 0, prob=1.0, noise_channel='f')
    ir, noise = c._build_ir(p.circuits, extra_shots)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1), (17, 0, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)
    assert np.any(noise[0][0][0])

    c.append("N1", 1, prob=1.0, noise_channel='p')
    ir, noise = c._build_ir(p.circuits, extra_shots)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1), (17, 0, -1), (17, 1, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)
    assert np.any(noise[1][0][1])

    c.append("N1", 1, prob=1.0, noise_channel='d')
    ir, noise = c._build_ir(p.circuits, extra_shots)
    test_ir = np.array([(5, 0, -1), (9, 0, 1), (14, 0, -1), (14, 1, -1), (17, 0, -1), (17, 1, -1), (17, 1, -1)], dtype=ir_dtype)
    np.testing.assert_array_equal(ir, test_ir)
    assert np.any(noise[2][0])

   

    
