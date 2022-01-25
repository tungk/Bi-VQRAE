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

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{i}_{t} = \mathsf{tanh}(\mathbf{W}^{1}_{\mathbf{i}} \cdot \mathbf{s}_{t-1} + \mathbf{W}^{2}_{\mathbf{i}} \cdot \mathbf{s}_{t} + \mathbf{b}_\mathbf{i})">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{f}_{t} = \sigma(\mathbf{W}^{1}_{\mathbf{f}} \cdot \mathbf{s}_{t-1} + \mathbf{W}^{2}_{\mathbf{f}} \cdot \mathbf{s}_{t} + \mathbf{b}_\mathbf{f})">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{o}_{t} = \sigma(\mathbf{W}^{1}_{\mathbf{o}} \cdot \mathbf{s}_{t-1} + \mathbf{W}^{2}_{\mathbf{o}} \cdot \mathbf{s}_{t} + \mathbf{b}_\mathbf{o})">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{c}_{t} = \mathbf{f}_{t} \odot \mathbf{c}_{t-1} + (1 - \mathbf{f}_{t}) \odot \mathbf{i}_{t}">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{h}_{t} = \mathbf{o}_{t} \odot \mathbf{c}_{t}">


### qnet

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{h}_{t} = \mathsf{QRNN}(\mathbf{s}_{t-1}, \mathbf{s}_{t})">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{a}_{t} = \mathsf{QRNN}([\mathbf{s}_{t+1}, \mathbf{h}_{t+1}], [\mathbf{s}_{t}, \mathbf{h}_{t}])">
<img src="https://render.githubusercontent.com/render/math?math=\Phi_{\mathbf{z}_{t}} = f(\mathbf{W}_{\Phi_{\mathbf{z}}} \cdot \mathbf{a}_{t} + \mathbf{b}_{\Phi_{\mathbf{z}}})">
<img src="https://render.githubusercontent.com/render/math?math=\mu_{\mathbf{z}_{t}} = \mathbf{W}_{\mu_{\mathbf{z}}} \cdot \Phi_{\mathbf{z}_{t}} + \mathbf{b}_{\mu_{\mathbf{z}}}">
<img src="https://render.githubusercontent.com/render/math?math=\sigma_{\mathbf{z}_{t}} = \mathsf{softplus}(\mathbf{W}_{\sigma_{\mathbf{z}}} \cdot \Phi_{\mathbf{z}_{t}} + \mathbf{b}_{\sigma_{\mathbf{z}}})">

<img src="q_net.png" alt="q_net" style="zoom:5%;" />

### qnet

<img src="https://render.githubusercontent.com/render/math?math=\Phi_{\mathbf{s}_{t}} = f(\mathbf{W}_{\Phi_{\mathbf{s}}} \cdot [\mathbf{h}_{t}, \mathbf{z}_{t}] + \mathbf{b}_{\Phi_{\mathbf{s}}})">
<img src="https://render.githubusercontent.com/render/math?math=\mu_{\mathbf{s}_{t}} = \mathbf{W}_{\mu_{\mathbf{s}}} \cdot \Phi_{\mathbf{s}_{t}} + \mathbf{b}_{\mu_{\mathbf{s}}}">
<img src="https://render.githubusercontent.com/render/math?math=\sigma_{\mathbf{s}_{t}} = \mathsf{softplus}(\mathbf{W}_{\sigma_{\mathbf{s}}} \cdot \Phi_{\mathbf{s}_{t}} + \mathbf{b}_{\sigma_{\mathbf{s}}})">

<img src="p_net.png" alt="q_net" style="zoom:5%;" />

### Objective Function

<img src="https://render.githubusercontent.com/render/math?math=argmax_{\phi, \theta}\mathcal{L}(\mathbf{s}_{t} = - \mathbb{E}_{q_{\phi}(\mathbf{z}_{t}|\mathbf{s}_{t})}[\mathsf{D}_{\alpha,\beta,\gamma}(\hat{p}(\mathbf{s}_{t})||p_{\theta}(\mathbf{s}_{t}|\mathbf{z}_{t}))] - \mathsf{D_{KL}}[q_{\phi}(\mathbf{z}_{t}|\mathbf{s}_{t})||p_{\theta}(\mathbf{z}_{t})]">

## Citation

If you use the code, please cite the following paper:

```latex
@inproceedings{DBLP:conf/icde/KieuYGCZSJ22,
	author     = {Tung Kieu and Bin Yang and Chenjuan Guo and Razvan-Gabriel Cirstea and Yan Zhao and Yale Song and Christian S. Jensen},
	title      = {Anomaly Detection in Time Series with Robust Variational Quasi-Recurrent Autoencoders},
	booktitle  = {{ICDE}},
	pages      = {1--13},
	year       = {2022}
}
```
