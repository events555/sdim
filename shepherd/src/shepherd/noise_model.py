"""
noise_model.py -- noise models for qudit simulations.

Reads, validates, generates, writes and applies noise models. A NoiseModel records,
for each gate name, which Pauli channel ("flip", "phase" or "dep") is associated with
it, with what probability, and whether it acts before or after the gate; and which
channel applies to idling qudits. `NoiseModel.apply(circuit)` turns a noiseless sdim
circuit into a noisy one (see "Applying a model" below).

JSON format (schema_version 1)
------------------------------
{
  "schema_version": 1,
  "name": "example",
  "description": "optional free text",
  "dimension": 3,
  "gates": {
    "H":    {"channel": "dep", "when": "after", "p": 0.0001},
    "CNOT": {"channel": "dep", "when": "after", "two_qudit_channel": false,
             "single_qudit_noise_gate_target": "target", "p": 0.001},
    "M":    {"channel": "flip", "when": "before", "p": 0.002},
    "RESET": {"channel": "flip", "when": "after", "p": 0.001}
  },
  "idle": {
    "default":     {"channel": "phase", "p": 5e-05},
    "measurement": {"channel": "dep", "p": 2e-04}
  }
}

Gates
-----
* "channel" is "flip", "phase" or "dep"; "p" is that channel's probability.
* "when" is "before" or "after" the gate. Measurement errors belong "before" the
  measurement: "flip" for a Z-basis measurement, "phase" for an X-basis one.
* A two-qudit gate (CNOT, ...) must say "two_qudit_channel": true or false.
    true  -> the channel acts on both qudits of the gate (sdim's N2, which is two-qudit
             depolarizing, so the channel must be "dep" when the model is applied).
    false -> a single-qudit channel on one qudit of the gate, named by
             "single_qudit_noise_gate_target": "control" or "target", or an independent copy
             on each of them: "both".
* Single-qudit gates take neither key.
* "I" (identity) is an ordinary gate: a circuit can place I where qudits wait, and the model
  gives that wait its noise. The memory experiment (benchmarking/ler.py) puts I on every data
  qudit once per round, so code-capacity and phenomenological noise are models too.

Idling
------
The keys under "idle" name idling contexts (for example "measurement" for qudits
idling while others are measured); "default" is the fallback for a context with no
entry. Idle channels are single-qudit.

Applying a model
----------------
`apply` copies the circuit instruction by instruction. Each gate with an entry gets its
channel immediately before or after it (N1 with 'f', 'p' or 'd'; N2 for a two-qudit
channel); gates without an entry, annotations (DETECTOR, LOGICAL_OBSERVABLE) and noise
already in the circuit are copied unchanged. Idle noise needs layers: if the circuit has
TICKs, the instructions between two TICKs (and before the first and after the last) are a
layer, and every qudit no gate in a layer touches gets the idle channel for that layer, at
its end: "measurement" if the layer contains a measurement or reset, else "default". A
circuit without TICKs gets no idle noise. Zero-probability channels are left out.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from sdim import Circuit
from sdim.gatedata import GateData

SCHEMA_VERSION = 1

CHANNEL_KINDS = ("flip", "phase", "dep")
WHEN_VALUES = ("before", "after")
TWO_QUDIT_KEY = "two_qudit_channel"
TARGET_KEY = "single_qudit_noise_gate_target"
TARGET_VALUES = ("control", "target", "both")

# Arity of sdim's gates (canonical names), taken from sdim's gate table so the two stay in
# sync. Noise gates and zero-qudit annotations (DETECTOR, TICK, ...) are left out. Used for
# validation. A gate not listed here is treated as two-qudit if its entry has
# "two_qudit_channel", and as single-qudit otherwise.
KNOWN_GATE_ARITY: dict[str, int] = {
    g.name: g.arg_count for g in GateData().gateMap.values()
    if g.arg_count > 0 and g.name not in ("N1", "N2")
}


class NoiseModelError(ValueError):
    """Raised for a malformed or invalid noise model description."""


def _check_keys(obj: Any, *, allowed: Iterable[str], required: Iterable[str] = (), where: str) -> None:
    if not isinstance(obj, Mapping):
        raise NoiseModelError(f"{where}: expected an object, got {type(obj).__name__}")
    unknown = set(obj) - set(allowed)
    if unknown:
        raise NoiseModelError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed keys are {sorted(allowed)}")
    missing = set(required) - set(obj)
    if missing:
        raise NoiseModelError(f"{where}: missing required key(s) {sorted(missing)}")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_probability(p: Any, where: str) -> float:
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise NoiseModelError(f"{where}: 'p' must be a number, got {p!r}")
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise NoiseModelError(f"{where}: 'p' must lie between 0 and 1, got {p}")
    return float(p)


# ---------------------------------------------------------------------------
# Building blocks of a noise model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Channel:
    """A named noise channel ("flip", "phase" or "dep") on one qudit, or on two (two_qudit)."""
    kind: str
    p: float
    two_qudit: bool = False

    def __post_init__(self) -> None:
        if self.kind not in CHANNEL_KINDS:
            raise NoiseModelError(f"unknown channel {self.kind!r}; known channels are {list(CHANNEL_KINDS)}")
        if not isinstance(self.two_qudit, bool):
            raise NoiseModelError(f"two_qudit must be true or false, got {self.two_qudit!r}")
        _check_probability(self.p, f"channel {self.kind}")

    @classmethod
    def from_dict(cls, obj: Any, where: str, *, extra_keys: Iterable[str] = (),
                  two_qudit: bool = False) -> Channel:
        _check_keys(obj, allowed={"channel", "p", *extra_keys},
                    required={"channel", "p"}, where=where)
        kind = obj["channel"]
        if kind not in CHANNEL_KINDS:
            raise NoiseModelError(f"{where}: unknown channel {kind!r}; "
                                  f"known channels are {list(CHANNEL_KINDS)}")
        return cls(kind, _check_probability(obj["p"], where), two_qudit)

    def to_dict(self) -> dict:
        return {"channel": self.kind, "p": self.p}


@dataclass(frozen=True)
class GateNoise:
    """The noise attached to one gate.

    when:   "before" or "after" the gate.
    target: for a single-qudit channel on a two-qudit gate, the qudit it acts on
            ("control" or "target", or "both" for an independent copy on each); None otherwise.
    The entry describes a two-qudit gate when its channel is two-qudit or it names
    a target (see is_two_qudit_gate).
    """
    channel: Channel
    when: str = "after"
    target: str | None = None

    @property
    def is_two_qudit_gate(self) -> bool:
        return self.channel.two_qudit or self.target is not None

    @classmethod
    def from_dict(cls, obj: Any, gate_name: str, where: str) -> GateNoise:
        _check_keys(obj, allowed={"channel", "p", "when", TWO_QUDIT_KEY, TARGET_KEY},
                    required={"channel", "p"}, where=where)
        gate_arity = KNOWN_GATE_ARITY.get(gate_name)
        has_flag = TWO_QUDIT_KEY in obj
        if gate_arity == 2 and not has_flag:
            raise NoiseModelError(f"{where}: two-qudit gate {gate_name} needs "
                                  f"'{TWO_QUDIT_KEY}': true or false")
        if gate_arity == 1 and (has_flag or TARGET_KEY in obj):
            bad = TWO_QUDIT_KEY if has_flag else TARGET_KEY
            raise NoiseModelError(f"{where}: '{bad}' only applies to two-qudit gates")
        two_qudit = obj.get(TWO_QUDIT_KEY, False)
        if not isinstance(two_qudit, bool):
            raise NoiseModelError(f"{where}: '{TWO_QUDIT_KEY}' must be true or false, got {two_qudit!r}")
        target = obj.get(TARGET_KEY)
        if has_flag and not two_qudit and target is None:
            raise NoiseModelError(f"{where}: with '{TWO_QUDIT_KEY}': false, give "
                                  f"'{TARGET_KEY}': \"control\", \"target\" or \"both\"")
        channel = Channel.from_dict(obj, where, extra_keys=("when", TWO_QUDIT_KEY, TARGET_KEY),
                                    two_qudit=two_qudit)
        return cls(channel, obj.get("when", "after"), target)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"channel": self.channel.kind, "when": self.when}
        if self.is_two_qudit_gate:
            out[TWO_QUDIT_KEY] = self.channel.two_qudit
        if self.target is not None:
            out[TARGET_KEY] = self.target
        out["p"] = self.channel.p
        return out

    def validate(self, gate_name: str, where: str) -> None:
        if self.when not in WHEN_VALUES:
            raise NoiseModelError(f"{where}: 'when' must be one of {WHEN_VALUES}, got {self.when!r}")
        if self.target is not None and self.target not in TARGET_VALUES:
            raise NoiseModelError(f"{where}: '{TARGET_KEY}' must be one of {TARGET_VALUES}, "
                                  f"got {self.target!r}")
        if self.channel.two_qudit and self.target is not None:
            raise NoiseModelError(f"{where}: with '{TWO_QUDIT_KEY}': true the channel acts on "
                                  f"both qudits; remove '{TARGET_KEY}'")
        gate_arity = KNOWN_GATE_ARITY.get(gate_name)
        if gate_arity == 1 and self.is_two_qudit_gate:
            raise NoiseModelError(f"{where}: {gate_name} acts on one qudit; remove "
                                  f"'{TWO_QUDIT_KEY}' / '{TARGET_KEY}'")
        if gate_arity == 2 and not self.is_two_qudit_gate:
            raise NoiseModelError(f"{where}: two-qudit gate {gate_name} needs "
                                  f"'{TWO_QUDIT_KEY}': true, or false with '{TARGET_KEY}'")


@dataclass
class IdleNoise:
    """Single-qudit noise on idling qudits, keyed by idling context.

    "default" applies to any context without its own entry. With no "default",
    idle qudits in unlisted contexts get no noise.
    """
    contexts: dict[str, Channel] = field(default_factory=dict)

    def channel_for(self, context: str = "default") -> Channel | None:
        return self.contexts.get(context, self.contexts.get("default"))


# ---------------------------------------------------------------------------
# Emitting noise into sdim circuits
# ---------------------------------------------------------------------------

_N1_CODE = {"flip": "f", "phase": "p", "dep": "d"}
_MEASUREMENTS = frozenset({"M", "M_X", "RESET"})


def _emit(out: Circuit, gate_name: str, noise: GateNoise, qudits: list[int]) -> None:
    """Appends the noise of one gate entry acting on `qudits` to `out`."""
    ch = noise.channel
    if ch.p == 0:
        return
    if len(qudits) == 1:
        out.add_gate("N1", qudits[0], noise_channel=_N1_CODE[ch.kind], prob=ch.p)
        return
    if ch.two_qudit:
        if ch.kind != "dep":
            raise NoiseModelError(f"gates.{gate_name}: a two-qudit channel becomes sdim's N2, which is "
                                  f"two-qudit depolarizing; {ch.kind!r} is not available")
        out.add_gate("N2", qudits[0], qudits[1], prob=ch.p)
        return
    picked = {"control": qudits[:1], "target": qudits[1:], "both": qudits}[noise.target]
    for qd in picked:
        out.add_gate("N1", qd, noise_channel=_N1_CODE[ch.kind], prob=ch.p)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class NoiseModel:
    dimension: int
    gates: dict[str, GateNoise] = field(default_factory=dict)
    idle: IdleNoise = field(default_factory=IdleNoise)
    name: str = "unnamed"
    description: str = ""

    def __post_init__(self) -> None:
        self.validate()

    # -- queries ------------------------------------------------------------

    def gate(self, name: str) -> GateNoise | None:
        """Noise for a gate, or None if the model has no entry for it."""
        return self.gates.get(name)

    def idle_channel(self, context: str = "default") -> Channel | None:
        return self.idle.channel_for(context)

    # -- validation -------------------------------------------------------------

    def validate(self) -> None:
        if not _is_int(self.dimension) or self.dimension < 2:
            raise NoiseModelError(f"dimension must be an integer >= 2, got {self.dimension!r}")
        for name, g in self.gates.items():
            g.validate(name, f"gates.{name}")
        for context, ch in self.idle.contexts.items():
            if ch.two_qudit:
                raise NoiseModelError(f"idle.{context}: idle noise must be a single-qudit channel")

    # -- JSON ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, obj: Any) -> NoiseModel:
        _check_keys(obj, allowed={"schema_version", "name", "description", "dimension",
                                  "gates", "idle"},
                    required={"dimension"}, where="noise model")
        version = obj.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise NoiseModelError(f"unsupported schema_version {version!r} "
                                  f"(this library reads version {SCHEMA_VERSION})")
        gates_obj = obj.get("gates", {})
        idle_obj = obj.get("idle", {})
        for key, val in (("gates", gates_obj), ("idle", idle_obj)):
            if not isinstance(val, Mapping):
                raise NoiseModelError(f"{key}: expected an object, got {type(val).__name__}")
        return cls(
            dimension=obj["dimension"],
            gates={n: GateNoise.from_dict(g, n, f"gates.{n}") for n, g in gates_obj.items()},
            idle=IdleNoise({c: Channel.from_dict(ch, f"idle.{c}") for c, ch in idle_obj.items()}),
            name=obj.get("name", "unnamed"),
            description=obj.get("description", ""),
        )

    def to_dict(self) -> dict:
        self.validate()
        out: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "name": self.name}
        if self.description:
            out["description"] = self.description
        out["dimension"] = self.dimension
        out["gates"] = {n: g.to_dict() for n, g in self.gates.items()}
        out["idle"] = {c: ch.to_dict() for c, ch in self.idle.contexts.items()}
        return out

    @classmethod
    def loads(cls, text: str) -> NoiseModel:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise NoiseModelError(f"invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}") from None
        return cls.from_dict(obj)

    def dumps(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def load(cls, path: str | Path) -> NoiseModel:
        path = Path(path)
        try:
            return cls.loads(path.read_text())
        except NoiseModelError as e:
            raise NoiseModelError(f"{path}: {e}") from None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps() + "\n")

    # -- identity and metadata -----------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON; equal models give equal fingerprints."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def metadata(self) -> dict:
        """Compact summary for a task's json_metadata."""
        return {"noise": self.name, "noise_dimension": self.dimension,
                "noise_fingerprint": self.fingerprint()[:16]}

    # -- applying the model -----------------------------------------------------

    def apply(self, circuit: Circuit) -> Circuit:
        """
        A noisy copy of `circuit` (see "Applying a model" in the module docstring). The input
        is not changed. Measurement records are untouched, so DETECTOR and LOGICAL_OBSERVABLE
        expressions keep their meaning.
        """
        if circuit.dimension != self.dimension:
            raise NoiseModelError(f"model is for dimension {self.dimension}, circuit has "
                                  f"dimension {circuit.dimension}")
        n = circuit.num_qudits
        out = Circuit(n, circuit.dimension)
        use_idle = bool(self.idle.contexts) and any(op.name == "TICK" for op in circuit.operations)
        touched: set[int] = set()
        measures = False

        def end_layer() -> None:
            nonlocal measures
            if use_idle:
                ch = self.idle_channel("measurement" if measures else "default")
                idle = [i for i in range(n) if i not in touched]
                if ch is not None and ch.p > 0 and idle:
                    out.add_gate("N1", idle, noise_channel=_N1_CODE[ch.kind], prob=ch.p)
            touched.clear()
            measures = False

        for op in circuit.operations:
            name, params = op.name, dict(op.params)
            if name == "TICK":
                end_layer()
                out.add_gate("TICK")
                continue
            if name in ("DETECTOR", "LOGICAL_OBSERVABLE"):
                out.add_gate(name, **params)
                continue
            qudits = [op.qudit_index] if op.target_index is None else [op.qudit_index, op.target_index]
            if name in ("N1", "N2"):                       # noise already in the circuit
                out.add_gate(name, *qudits, **params)
                continue
            touched.update(qudits)
            measures = measures or name in _MEASUREMENTS
            noise = self.gate(name)
            if noise is not None and noise.when == "before":
                _emit(out, name, noise, qudits)
            out.add_gate(name, *qudits, **params)
            if noise is not None and noise.when == "after":
                _emit(out, name, noise, qudits)
        end_layer()
        return out

    # -- standard models used by the memory experiments ----------------------------

    @classmethod
    def code_capacity(cls, dimension: int, p: float) -> NoiseModel:
        """A uniformly random non-zero shift with probability p wherever a circuit places I."""
        return cls(dimension=dimension, gates={"I": GateNoise(Channel("flip", p), "after")},
                   name="code_capacity",
                   description=f"shift noise p={p} on I (waiting data); everything else perfect")

    @classmethod
    def phenomenological(cls, dimension: int, p: float, p_meas: float | None = None) -> NoiseModel:
        """Shift noise p on I (data, once per round), and outcome flips p_meas (default p) on M."""
        p_meas = p if p_meas is None else p_meas
        return cls(dimension=dimension,
                   gates={"I": GateNoise(Channel("flip", p), "after"),
                          "M": GateNoise(Channel("flip", p_meas), "before")},
                   name="phenomenological",
                   description=f"shift noise p={p} on I, outcome flips p_meas={p_meas} before M; "
                               "perfect gates")

    @classmethod
    def circuit_level(cls, dimension: int, p: float, *, two_qudit: str = "independent") -> NoiseModel:
        """
        Every operation noisy at rate p: depolarizing after I (data, once per round) and after
        each SUM (CNOT / CNOT_INV), a shift before M and after RESET.

        two_qudit: "independent" for single-qudit depolarizing on each qudit of a SUM (graph-like
            DEMs), or "N2" for two-qudit depolarizing (hyperedges and correlated faults).
        """
        if two_qudit == "independent":
            sum_noise = GateNoise(Channel("dep", p, False), "after", "both")
        elif two_qudit == "N2":
            sum_noise = GateNoise(Channel("dep", p, True), "after")
        else:
            raise NoiseModelError(f"two_qudit must be 'independent' or 'N2', got {two_qudit!r}")
        return cls(dimension=dimension,
                   gates={"I": GateNoise(Channel("dep", p), "after"),
                          "CNOT": sum_noise, "CNOT_INV": sum_noise,
                          "M": GateNoise(Channel("flip", p), "before"),
                          "RESET": GateNoise(Channel("flip", p), "after")},
                   name=f"circuit_{two_qudit}",
                   description=f"p={p} on every I, SUM, M and RESET; SUM noise {two_qudit}")

    # -- generators ------------------------------------------------------------

    @classmethod
    def standard(
        cls,
        dimension: int,
        *,
        p_1q: float,
        p_2q: float,
        p_meas: float,
        p_reset: float,
        p_idle: float,
        p_idle_measurement: float | None = None,
        single_qudit_channel: str = "dep",
        two_qudit_gate_channel: str = "dep",
        two_qudit_channel: bool = True,
        single_qudit_noise_gate_target: str | None = None,
        idle_channel: str = "dep",
        single_qudit_gates: Iterable[str] = ("X", "Z", "H", "P"),
        two_qudit_gates: Iterable[str] = ("CNOT",),
        measurements: Mapping[str, str] | None = None,
        resets: Mapping[str, str] | None = None,
        name: str = "standard",
        description: str = "",
    ) -> NoiseModel:
        """Build a complete model from a handful of probabilities.

        single_qudit_channel / two_qudit_gate_channel / idle_channel: "flip", "phase" or "dep".
        two_qudit_channel: True for a channel on both qudits of each two-qudit gate; False for
            a single-qudit channel on the qudit named by single_qudit_noise_gate_target.
        measurements / resets map gate name -> basis ("Z" or "X"); defaults {"M": "Z"},
            {"RESET": "Z"}. A Z-basis gate gets "flip" noise, an X-basis gate "phase" noise.
        """
        measurements = {"M": "Z"} if measurements is None else dict(measurements)
        resets = {"RESET": "Z"} if resets is None else dict(resets)
        basis_channel = {"Z": "flip", "X": "phase"}
        if not two_qudit_channel and single_qudit_noise_gate_target is None:
            raise NoiseModelError("two_qudit_channel=False needs single_qudit_noise_gate_target "
                                  "('control', 'target' or 'both')")

        def outcome_channel(gate: str, basis: str, p: float) -> Channel:
            if basis not in basis_channel:
                raise NoiseModelError(f"{gate}: basis must be 'Z' or 'X', got {basis!r}")
            return Channel(basis_channel[basis], p)

        gates: dict[str, GateNoise] = {}
        for g in single_qudit_gates:
            gates[g] = GateNoise(Channel(single_qudit_channel, p_1q), "after")
        for g in two_qudit_gates:
            gates[g] = GateNoise(Channel(two_qudit_gate_channel, p_2q, two_qudit_channel), "after",
                                 None if two_qudit_channel else single_qudit_noise_gate_target)
        for g, basis in measurements.items():
            gates[g] = GateNoise(outcome_channel(g, basis, p_meas), "before")
        for g, basis in resets.items():
            gates[g] = GateNoise(outcome_channel(g, basis, p_reset), "after")
        idle = {"default": Channel(idle_channel, p_idle)}
        if p_idle_measurement is not None:
            idle["measurement"] = Channel(idle_channel, p_idle_measurement)
        return cls(dimension=dimension, gates=gates, idle=IdleNoise(idle),
                   name=name, description=description)


if __name__ == "__main__":
    # Validate one or more JSON files:  python noise_model.py model1.json model2.json
    import sys

    status = 0
    for arg in sys.argv[1:]:
        try:
            m = NoiseModel.load(arg)
        except NoiseModelError as e:
            print(f"INVALID  {e}")
            status = 1
        else:
            print(f"OK       {arg}: '{m.name}', dimension={m.dimension}, {len(m.gates)} gates, "
                  f"idle contexts {sorted(m.idle.contexts)}")
    sys.exit(status)