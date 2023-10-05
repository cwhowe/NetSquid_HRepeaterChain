import pandas
import pydynaa
import numpy as np
import random

import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel, QuantumErrorModel
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z
from netsquid.nodes import Node, Network
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.util.datacollector import DataCollector
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
__all__ = [
    "SwapProtocol",
    "SwapCorrectProgram",
    "CorrectProtocol",
    "FibreDepolarizeModel",
    "create_qprocessor",
    "setup_network",
    "setup_repeater_protocol",
    "setup_datacollector",
    "run_simulation",
    "create_plot",
]



class SwapProtocol(NodeProtocol):
    """Perform Swap on a repeater node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    name : str
        Name of this protocol.

    """

    def __init__(self, node, name):
        super().__init__(node, name)
        self._qmem_input_port_l = self.node.qmemory.ports["qin1"]
        self._qmem_input_port_r = self.node.qmemory.ports["qin0"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):
        while True:
            yield (self.await_port_input(self._qmem_input_port_l) &
                   self.await_port_input(self._qmem_input_port_r))
            # Perform Bell measurement
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1, 0])
            m, = self._program.output["m"]
            # Send result to right node on end
            self.node.ports["ccon_R"].tx_output(Message(m))


class SwapCorrectProgram(QuantumProgram):
    """Quantum processor program that applies all swap corrections."""
    default_num_qubits = 1

    def set_corrections(self, x_corr, z_corr):
        self.x_corr = x_corr % 2
        self.z_corr = z_corr % 2

    def program(self):
        q1, = self.get_qubit_indices(1)
        if self.x_corr == 1:
            self.apply(INSTR_X, q1)
        if self.z_corr == 1:
            self.apply(INSTR_Z, q1)
        yield self.run()


class CorrectProtocol(NodeProtocol):
    """Perform corrections for a swap on an end-node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    num_nodes : int
        Number of nodes in the repeater chain network.

    """

    def __init__(self, node, num_nodes):
        super().__init__(node, "CorrectProtocol")
        self.num_nodes = num_nodes
        self._x_corr = 0
        self._z_corr = 0
        self._program = SwapCorrectProgram()
        self._counter = 0

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["ccon_L"])
            message = self.node.ports["ccon_L"].rx_input()
            if message is None or len(message.items) != 1:
                continue
            m = message.items[0]
            if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                self._x_corr += 1
            if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                self._z_corr += 1
            self._counter += 1
            if self._counter == self.num_nodes - 2:
                if self._x_corr or self._z_corr:
                    self._program.set_corrections(self._x_corr, self._z_corr)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1])
                self.send_signal(Signals.SUCCESS)
                self._x_corr = 0
                self._z_corr = 0
                self._counter = 0


def create_qprocessor(name):
    """Factory to create a quantum processor for each node in the repeater chain network.

    Has two memory positions and the physical instructions necessary for teleportation.

    Parameters
    ----------
    name : str
        Name of the quantum processor.

    Returns
    -------
    :class:`~netsquid.components.qprocessor.QuantumProcessor`
        A quantum processor to specification.

    """
    noise_rate = 200
    gate_duration = 1
    gate_noise_model = DephaseNoiseModel(noise_rate)
    mem_noise_model = DepolarNoiseModel(noise_rate)
    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration=gate_duration,
                            quantum_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_Z, duration=gate_duration,
                            quantum_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=gate_duration),
    ]
    qproc = QuantumProcessor(name, num_positions=2, fallback_to_nonphysical=False,
                             mem_noise_models=[mem_noise_model] * 2,
                             phys_instructions=physical_instructions)
    return qproc

def setup_network(num_nodes, node_distance, source_frequency):
    if num_nodes < 3:
        raise ValueError(f"Can't create repeater chain with {num_nodes} nodes.")
    
    network = Network("Repeater_star_network")
    nodes = []

    # Create the central node
    central_node = Node("Central_Node", qmemory=create_qprocessor("qproc_Central"))
    network.add_node(central_node)

    # Create and add other nodes to the network
    for i in range(num_nodes):
        # Prepend leading zeros to the node index for consistent naming
        num_zeros = int(np.log10(num_nodes)) + 1
        node = Node(f"Node_{i:0{num_zeros}d}", qmemory=create_qprocessor(f"qproc_{i}"))
        nodes.append(node)
    network.add_nodes(nodes)

    # Create connections from the central node to all other nodes
    for node in nodes:
        # Create a quantum connection from central_node to node
        qconn = EntanglingConnection(name=f"qconn_Central-{node.name}", length=node_distance, source_frequency=source_frequency)
        
        # Add the quantum connection to the network
        #network.add_connection(central_node, node, connection=qconn, label="quantum")
        # Forward qconn directly to quantum memories for node's input
        #central_node.ports["qin0"].forward_input(qconn.ports["qA"])
        #node.ports["qin1"].forward_input(qconn.ports["qB"])
        
        port_name, port_r_name = network.add_connection(
            node, central_node, connection=qconn, label="quantum")
        
        node.ports[port_name].forward_input(node.qmemory.ports["qin0"])  # R input
        central_node.ports[port_r_name].forward_input(
            central_node.qmemory.ports["qin1"])  # L input

       # Create classical connection
        cconn = ClassicalConnection(name=f"cconn_{i}-{i+1}", length=node_distance)
        port_name, port_r_name = network.add_connection(
            node, central_node, connection=cconn, label="classical",
            port_name_node1="ccon_R", port_name_node2="ccon_L")
        # Forward cconn to right most node
        if "ccon_L" in node.ports:
            node.ports["ccon_L"].bind_input_handler(
                lambda message, _node=node: _node.ports["ccon_R"].tx_output(message))
        print("connection 1")
    # Add CorrectProtocol to the central node
    correct_protocol = CorrectProtocol(central_node, num_nodes)
    central_node.add_subcomponent(correct_protocol)

    return network


def setup_repeater_protocol(network):
    """Setup repeater protocol on repeater chain network.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        Repeater chain network to put protocols on.

    Returns
    -------
    :class:`~netsquid.protocols.protocol.Protocol`
        Protocol holding all subprotocols used in the network.

    """
    protocol = LocalProtocol(nodes=network.nodes)
    # Add SwapProtocol to all repeater nodes. Note: we use unique names,
    # since the subprotocols would otherwise overwrite each other in the main protocol.
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
    for node in nodes[1:-1]:
        subprotocol = SwapProtocol(node=node, name=f"Swap_{node.name}")
        protocol.add_subprotocol(subprotocol)
    # Add CorrectProtocol to Bob
    subprotocol = CorrectProtocol(nodes[-1], len(nodes))
    protocol.add_subprotocol(subprotocol)
    return protocol


def setup_datacollector(network, protocol):
    """Setup the datacollector to calculate the fidelity
    when the CorrectionProtocol has finished.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        Repeater chain network to put protocols on.

    protocol : :class:`~netsquid.protocols.protocol.Protocol`
        Protocol holding all subprotocols used in the network.

    Returns
    -------
    :class:`~netsquid.util.datacollector.DataCollector`
        Datacollector recording fidelity data.

    """
    # Ensure nodes are ordered in the chain:
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]

    def calc_fidelity(evexpr):
        qubit_a, = nodes[0].qmemory.peek([0])
        qubit_b, = nodes[-1].qmemory.peek([1])
        fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True)
        return {"fidelity": fidelity}

    dc = DataCollector(calc_fidelity, include_entity_name=False)
    dc.collect_on(pydynaa.EventExpression(source=protocol.subprotocols['CorrectProtocol'],
                                          event_type=Signals.SUCCESS.value))
    return dc


def run_simulation(num_nodes=4, node_distance=20, num_iters=100):
    ns.sim_reset()
    est_runtime = (0.5 + num_nodes - 1) * node_distance * 5e3
    network = setup_network(num_nodes, node_distance=node_distance, source_frequency=1e9 / est_runtime)
    dc = setup_datacollector(network)

    protocol = setup_repeater_protocol(network)
    
    # Create CorrectProtocol for the central node and add it here
    central_node = network.nodes["Central_Node"]
    correct_protocol = CorrectProtocol(central_node, num_nodes)
    central_node.add_subcomponent(correct_protocol)

    protocol.start()
    
    ns.sim_run(est_runtime * num_iters)
    return dc.dataframe



def calculate_fidelity_between_nodes(node1, node2, node3, node4, num_iters=100):
    """
    Calculate the fidelity observed by communicating between two nodes in the network.

    Parameters
    ----------
    node1 : :class:`~netsquid.nodes.node.Node`
        The first node for communication.
    node2 : :class:`~netsquid.nodes.node.Node`
        The second node for communication.
    num_iters : int, optional
        Number of simulation runs. Default 100.

    Returns
    -------
    float
        The average fidelity observed between the two nodes.

    """
    
    ns.sim_reset()
    
    # Create the network with the two specified nodes
    network = Network("Repeater_chain_network")
    network.add_node(node1)
    network.add_node(node2)
    network.add_node(node3)
    network.add_node(node4)
    
    # Setup the protocol for communication between the nodes
    protocol = LocalProtocol(nodes=[node1, node2, node3, node4])
    
    # Run the simulation
    est_runtime = 1e6  # Adjust this as needed
    protocol.start()
    ns.sim_run(est_runtime * num_iters)
    
    # Calculate and return the average fidelity
    qubit_a, = node1.qmemory.peek([0])
    qubit_b, = node4.qmemory.peek([0])
    fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ns.qubits.ketstates.b00, squared=True)
    
    return fidelity

if __name__ == "__main__":
    run_simulation()
    """
    ns.set_qstate_formalism(ns.QFormalism.DM)
    node1 = Node("Node_1", qmemory=create_qprocessor("qproc_1"))
    node2 = Node("Node_2", qmemory=create_qprocessor("qproc_2"))
    node3 = Node("Node_3", qmemory=create_qprocessor("qproc_3"))
    node4 = Node("Node_4", qmemory=create_qprocessor("qproc_4"))
    fidelity = calculate_fidelity_between_nodes(node1, node2, node3, node4)
    print(f"Fidelity between Node 1 and Node 2: {fidelity}")
    """


class FibreDepolarizeModel(QuantumErrorModel):
    """Custom non-physical error model used to show the effectiveness
    of repeater chains.

    The default values are chosen to make a nice figure,
    and don't represent any physical system.

    Parameters
    ----------
    p_depol_init : float, optional
        Probability of depolarization on entering a fibre.
        Must be between 0 and 1. Default 0.009
    p_depol_length : float, optional
        Probability of depolarization per km of fibre.
        Must be between 0 and 1. Default 0.025

    """

    def __init__(self, p_depol_init=0.009, p_depol_length=0.025):
        super().__init__()
        self.properties['p_depol_init'] = p_depol_init
        self.properties['p_depol_length'] = p_depol_length
        self.required_properties = ['length']

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Uses the length property to calculate a depolarization probability,
        and applies it to the qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Not used.

        """
        for qubit in qubits:
            prob = 1 - (1 - self.properties['p_depol_init']) * np.power(
                10, - kwargs['length']**2 * self.properties['p_depol_length'] / 10)
            ns.qubits.depolarize(qubit, prob=prob)


if __name__ == "__main__":
    ns.set_qstate_formalism(ns.QFormalism.DM)
    create_plot(20)
