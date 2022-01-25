# Bi-VQRAE

The source code for paper “Anomaly Detection in Time Series with Robust
Variational Quasi-Recurrent Autoencoders”

## Abstract

We propose variational quasi-recurrent autoencoders (VQRAEs) to enable
robust and efficient anomaly detection in time series in unsupervised
settings. The proposed VQRAEs employs a judiciously designed objective
function based on robust divergences including alpha, beta, and
gamma-divergence, making it possible to separate anomalies from normal
data without the reliance on anomaly labels, thus achieving robustness
and fully unsupervised training. To better capture temporal dependencies
in time series data, VQRAEs are built upon quasi-recurrent neural
networks, which employ convolution and gating mechanisms to avoid the
inefficient recursive computations used by classic recurrent neural
networks. Further, VQRAEs can be extended to bi-directional BiVQRAEs
that utilize bi-directional information to further improve the accuracy.
The above design choices make VQRAEs not only robust and thus accurate,
but also efficient at detecting anomalies in streaming settings.
Experiments on five real-world time series offer insight into the design
properties of VQRAEs and demonstrate that VQRAEs are capable of
outperforming state-of-the-art methods.

## VQRAE

### QRNN

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{i}_{t} &= \mathsf{tanh}(\mathbf{W}^{1}_{\mathbf{i}} \cdot \mathbf{s}_{t-1} + \mathbf{W}^{2}_{\mathbf{i}} \cdot \mathbf{s}_{t} + \mathbf{b}_\mathbf{i})">

\\begin{align}
            \\mathbf{i}\_{t} &= \\mathsf{tanh}(\\mathbf{W}^{1}\_{\\mathbf{i}} \\cdot \\mathbf{s}\_{t-1} + \\mathbf{W}^{2}\_{\\mathbf{i}} \\cdot \\mathbf{s}\_{t} + \\mathbf{b}\_\\mathbf{i}) \\label{eqn:qrnn_1} \\\\
            \\mathbf{f}\_{t} &= \\sigma(\\mathbf{W}^{1}\_{\\mathbf{f}} \\cdot \\mathbf{s}\_{t-1} + \\mathbf{W}^{2}\_{\\mathbf{f}} \\cdot \\mathbf{s}\_{t} + \\mathbf{b}\_\\mathbf{f}) \\label{eqn:qrnn_2}\\\\
            \\mathbf{o}\_{t} &= \\sigma(\\mathbf{W}^{1}\_{\\mathbf{o}} \\cdot \\mathbf{s}\_{t-1} + \\mathbf{W}^{2}\_{\\mathbf{o}} \\cdot \\mathbf{s}\_{t} + \\mathbf{b}\_\\mathbf{o}) \\label{eqn:qrnn_3}\\\\
            \\mathbf{c}\_{t} &= \\mathbf{f}\_{t} \\odot \\mathbf{c}\_{t-1} + (1 - \\mathbf{f}\_{t}) \\odot \\mathbf{i}\_{t} \\label{eqn:qrnn_4}\\\\
            \\mathbf{h}\_{t} &= \\mathbf{o}\_{t} \\odot \\mathbf{c}\_{t} \\label{eqn:qrnn_5}
        \\end{align}
$$

### qnet

$$
\\begin{align}
        \\mathbf{h}\_{t} &= \\mathsf{QRNN}(\\mathbf{s}\_{t-1}, \\mathbf{s}\_{t}) \\label{eqn:qnet_1}\\\\
        \\mathbf{a}\_{t} &= \\mathsf{QRNN}(\[\\mathbf{s}\_{t+1}, \\mathbf{h}\_{t+1}\], \[\\mathbf{s}\_{t}, \\mathbf{h}\_{t}\]) \\label{eqn:qnet_2} \\\\
        \\Phi\_{\\mathbf{z}\_{t}} &= f(\\mathbf{W}\_{\\Phi\_{\\mathbf{z}}} \\cdot \\mathbf{a}\_{t} + \\mathbf{b}\_{\\Phi\_{\\mathbf{z}}}) \\label{eqn:qnet_3} \\\\
        \\mu\_{\\mathbf{z}\_{t}} &= \\mathbf{W}\_{\\mu\_{\\mathbf{z}}} \\cdot \\Phi\_{\\mathbf{z}\_{t}} + \\mathbf{b}\_{\\mu\_{\\mathbf{z}}} \\label{eqn:qnet_4} \\\\
        \\sigma\_{\\mathbf{z}\_{t}} &= \\mathsf{softplus}(\\mathbf{W}\_{\\sigma\_{\\mathbf{z}}} \\cdot \\Phi\_{\\mathbf{z}\_{t}} + \\mathbf{b}\_{\\sigma\_{\\mathbf{z}}}) \\label{eqn:qnet_5}
        \\end{align}
$$

<img src="D:\Source Code\(Bi)-VQRAE\q_net.png" alt="q_net" style="zoom:25%;" />

### pnet

$$ \\begin{align} *{*{t}} &= f(*{*{}} \[
