from conex import *
import torch
from pymonntorch import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class PoissionGenerator(Behavior) :

    def initialize(self, ng) : 
        self.offset = ng.network.iteration
        self.T = self.parameter("T", 50)
        self.lamda = self.parameter("lamda", 20)
        self.target = self.parameter("target", required = True)
        self.poisson = 0

    def forward(self, ng) : 
        pat = self.pattern(ng)
        if(pat == -1) : 
            ng.spikes[self.target] = torch.rand(len(self.target)) < 0
        else :
            self.poisson = (np.exp(-self.lamda) * (self.lamda ** (ng.network.iteration - self.offset))) / (np.math.factorial(ng.network.iteration - self.offset)) / ((np.exp(-self.lamda) * (self.lamda ** self.lamda)) / (np.math.factorial(self.lamda))*2)
            ng.spikes[self.target] = torch.rand(len(self.target)) <= self.poisson

    def pattern(self, ng) : 
        if(ng.network.iteration - self.offset > self.T) : 
            return -1   
        return 1


net = Network(behavior = prioritize_behaviors([
    TimeResolution(),
]))

ng1 = NeuronGroup(size = 10, net = net, behavior = prioritize_behaviors([
    SimpleDendriteStructure(),
    SimpleDendriteComputation(),
    LIF(
        tau = 7,
        R = 10,
        threshold = -13,
        v_rest = -65,
        v_reset = -70
    ),
    Fire(),
    SpikeTrace(tau_s = 15),
    NeuronAxon(),
]) | ({
    # 341 : PoissionGenerator(T = 50, lamda = 20, target = [0, 1, 2, 3, 4]),
    800 : Recorder(['v', "I", 'trace']),
    801 : EventRecorder(['spikes'])
}))

ng2 = NeuronGroup(size = 2, net = net, behavior = prioritize_behaviors([
    SimpleDendriteStructure(),
    SimpleDendriteComputation(),
    LIF(
        tau = 10,
        R = 8,
        threshold = -13,
        v_rest = -65,
        v_reset = -70
    ),
    Fire(),
    SpikeTrace(tau_s = 15),
    NeuronAxon(),
]) | ({
    800 : Recorder(['v', "I", "trace"]),
    801 : EventRecorder(['spikes'])
}))

sg = SynapseGroup(net = net, src = ng1, dst = ng2, tag = "Proximal,exi", behavior = prioritize_behaviors([
    SynapseInit(),
    WeightInitializer(mode = "normal(6, 4)"),
    SimpleDendriticInput(),
    SimpleSTDP(
        w_min = 0,
        w_max = 25,
        a_plus = 1.2,
        a_minus = 1,
    ),
    # WeightNormalization(norm = 50)
]) | ({
    800 : Recorder(["I", "weights"]),
}))

weights = torch.Tensor([1, 0, 1])
weight_shape = (1, 1, 1, 1, 3)
weights = weights.view(weight_shape)


sg_lateral = SynapseGroup(net = net, src = ng2, dst = ng2, tag = "Proximal, inh", behavior= prioritize_behaviors([
    SynapseInit(),
    WeightInitializer(weights = weights, weight_shape = weight_shape),
    LateralDendriticInput(inhibitory = True, current_coef = 25)
])| ({
    800 : Recorder(["I"]),
}))



net.initialize()

for i in range(10) : 
    if(i < 5) : 
        ng1.add_behavior(341, PoissionGenerator(T = 50, lamda = 20, target = [0, 1, 2, 3, 4]))
    else :
        ng1.add_behavior(341, PoissionGenerator(T = 50, lamda = 20, target = [5, 6, 7, 8, 9]))

    net.simulate_iterations(100)
    ng1.remove_behavior(341)

# Resting phase
net.simulate_iterations(200)

# test-phase
ng2.remove_behavior(400) # Remove STDP
for i in range(4) : 
    if i < 2 : 
        ng1.add_behavior(341, PoissionGenerator(T = 50, lamda = 20, target = [0, 1, 2, 3, 4]))
    else :
        ng1.add_behavior(341, PoissionGenerator(T = 50, lamda = 20, target = [5, 6, 7, 8, 9]))

    net.simulate_iterations(100)
    ng1.remove_behavior(341)



plt.figure(figsize = (6, 2))
plt.plot(sg_lateral['I', 0])
plt.legend(["Lateral on (A)", "Lateral on (B)"])
plt.show()

plt.plot(ng1['spikes.t', 0], ng1['spikes.i', 0], '.', color = "blue")
plt.title("Input spike raster plot")
plt.show()

plt.figure(figsize = (6, 2))
plt.plot(ng2['spikes.t', 0], ng2['spikes.i', 0], '.', color = "blue", marker = "|", markersize = 20, markeredgewidth=1.4)
plt.title("output spikes raster plot")
plt.yticks([0, 1])
plt.show()

A = sg['weights', 0][:,:,0]
plt.plot(A[:,:5], color = "blue")
plt.plot(A[:,5:], color = "orange")
plt.title("Synaptic Weights connceted to (A)")
plt.show()

A = sg['weights', 0][:,:,1]
plt.plot(A[:,:5], color = "blue")
plt.plot(A[:,5:], color = "orange")
plt.title("Synaptic Weights connceted to (B)")
# # plt.legend(["neuron{} -> A".format(i + 1) for i in range(10)])
plt.show()