from numba import njit, prange
import numpy as np

@njit(parallel=True)
def hadamard_optimized(x_block, z_block, phase_vector,
                       destab_x_block, destab_z_block,
                       destab_phase_vector, qudit_index, 
                       num_qudits, phase_order):
    
    # Get array slices for better cache locality
    x_row = x_block[qudit_index]
    z_row = z_block[qudit_index]
    dx_row = destab_x_block[qudit_index]
    dz_row = destab_z_block[qudit_index]

    # Handle stabilizer blocks
    for i in prange(num_qudits):
        new_x = -z_row[i]
        new_z = x_row[i]
        phase_vector[i] += phase_order * new_x * new_z
        x_row[i] = new_x
        z_row[i] = new_z

    # Handle destabilizer blocks
    for i in prange(num_qudits):
        new_x = -dz_row[i]
        new_z = dx_row[i]
        destab_phase_vector[i] += phase_order * new_x * new_z
        dx_row[i] = new_x
        dz_row[i] = new_z

@njit(parallel=True)
def hadamard_inv_optimized(x_block, z_block, phase_vector,
                          destab_x_block, destab_z_block,
                          destab_phase_vector, qudit_index, 
                          num_qudits, phase_order):
    
    # Get array slices for better cache locality
    x_row = x_block[qudit_index]
    z_row = z_block[qudit_index]
    dx_row = destab_x_block[qudit_index]
    dz_row = destab_z_block[qudit_index]

    # Handle stabilizer blocks
    for i in prange(num_qudits):
        new_x = z_row[i]
        new_z = -x_row[i]
        phase_vector[i] += phase_order * new_x * new_z
        x_row[i] = new_x
        z_row[i] = new_z

    # Handle destabilizer blocks
    for i in prange(num_qudits):
        new_x = dz_row[i]
        new_z = -dx_row[i]
        destab_phase_vector[i] += phase_order * new_x * new_z
        dx_row[i] = new_x
        dz_row[i] = new_z


@njit(parallel=True)
def phase_optimized(x_block, z_block, phase_vector,
                   destab_x_block, destab_z_block, 
                   destab_phase_vector,
                   qudit_index: int,
                   num_qudits: int,
                   phase_order: int,
                   even: int):
    # Parallelize loop over qudits
    for i in prange(num_qudits):
        if even:
            phase_vector[i] += x_block[qudit_index, i] ** 2
            destab_phase_vector[i] += destab_x_block[qudit_index, i] ** 2
        else:
            phase_vector[i] += (phase_order * x_block[qudit_index, i] * 
                            (x_block[qudit_index, i] - 1)) // 2
            destab_phase_vector[i] += (phase_order * destab_x_block[qudit_index, i] * 
                                    (destab_x_block[qudit_index, i] - 1)) // 2
        z_block[qudit_index, i] += x_block[qudit_index, i]
        destab_z_block[qudit_index, i] += destab_x_block[qudit_index, i]


@njit(parallel=True)
def phase_inv_optimized(x_block, z_block, phase_vector,
                       destab_x_block, destab_z_block, 
                       destab_phase_vector,
                       qudit_index: int,
                       num_qudits: int,
                       phase_order: int,
                       even: bool):
    for i in prange(num_qudits):
        if even:
            phase_vector[i] -= x_block[qudit_index, i] ** 2
            destab_phase_vector[i] -= destab_x_block[qudit_index, i] ** 2
        else:
            phase_vector[i] -= (phase_order * x_block[qudit_index, i] * 
                             (x_block[qudit_index, i]-1)) // 2
            destab_phase_vector[i] -= (phase_order * destab_x_block[qudit_index, i] * 
                                    (destab_x_block[qudit_index, i]-1)) // 2
            
        z_block[qudit_index, i] = (z_block[qudit_index, i] - 
                                  x_block[qudit_index, i])
        destab_z_block[qudit_index, i] = (destab_z_block[qudit_index, i] - 
                                         destab_x_block[qudit_index, i])
