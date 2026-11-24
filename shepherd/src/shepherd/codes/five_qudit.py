from shepherd.qecc import QECC, PauliString

class FiveQuditCode(QECC, name="five_qudit"):
    """[[5,1,3]]_d: cyclic shifts of X Z Z^-1 X^-1 I."""
    def stabilizers(self):
        d = self.dimension
        xs, zs = [1, 0, 0, d - 1, 0], [0, 1, d - 1, 0, 0]
        return [PauliString(d, 0, xs[-k:] + xs[:-k], zs[-k:] + zs[:-k]) for k in range(4)]
    # no encoding() override -> inherits stabilizer projection