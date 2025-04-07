# NoProp: Training Neural Networks Without Forward/Backward Propagation 🚀

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2503.24322)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

Official implementation of **NoProp**, a novel neural network training method that eliminates both forward and backward propagation through denoising diffusion. Achieves competitive performance on MNIST/CIFAR while enabling parallel layer training.


## 🔍 Table of Contents
- [Key Innovations](#-key-innovations)
- [Mathematical Foundations](#-mathematical-foundations)
- [Implementation](#-implementation)
- [Results](#-results)
  
## 🚀 Key Innovations
NoProp introduces three paradigm shifts:

1. **Propagation-Free Training**  
   - No sequential forward/backward passes
   - Layers train **independently** via denoising

2. **Diffusion-Based Learning**  
   - Corrupts labels progressively (forward diffusion)
   - Each layer learns to denoise a specific noise level

3. **Biological Plausibility**  
   - Avoids weight transport problem
   - Local learning only (no global gradients)

## 📜 Mathematical Foundations

### Forward Diffusion Process
Gradually corrupts clean labels $u_y$ with noise:

$$
z_t = \sqrt{\alpha_t} z_{t-1} + \sqrt{1-\alpha_t} \epsilon_t
$$

- $z_t$: Noisy label at step $t$
- $\alpha_t$: Noise schedule (e.g., linear/cosine)
- $\epsilon_t \sim \mathcal{N}(0,I)$: Gaussian noise

*Example*: For MNIST with $\alpha_t=0.9$:  
`[0,0,1,0] → [0.1,0.05,0.8,0.05] → ... → pure noise`

### Reverse Process (Training)
Each MLP layer $t$ predicts clean labels from noisy inputs:

$$
mathcal{L}_t = \mathbb{E} \| \hat{u}_\theta(z_t,x) - u_y \|^2
$$

- $\hat{u}_\theta$: MLP prediction
- $x$: Input image features
- $u_y$: Ground truth one-hot labe

## ⚙️ Implementation

### Core Components
```python
# 1. Diffusion Noise Scheduler
alpha = torch.linspace(1.0, 0.1, T)  # T diffusion steps

# 2. Denoising MLP (per layer)
class DenoisingMLP(nn.Module):
    def forward(self, x_features, z_t):
        return self.mlp(torch.cat([x_features, z_t], dim=1))

# 3. Training Loop
for t in range(T):
    u_hat = mlps[t](x_features, z[t+1].detach())
    loss = F.mse_loss(u_hat, u_y)
```

### Key Features
- **Parallel Training**: All `T` MLPs update simultaneously
- **Memory Efficient**: No activation storage
- **Flexible**: Works with CNNs/Transformers

## 📊 Results

| Method       | MNIST Acc | CIFAR-10 Acc | GPU Mem (GB) |
|--------------|-----------|--------------|--------------|
| NoProp (Ours)| 99.5%     | 80.5%        | 0.64         |
| Backprop     | 99.4%     | 79.9%        | 1.17         |
| Forward-Forward| 98.6%    | -            | 1.05         |

*Training 3x faster than backprop on multi-GPU setups*


## 👤 Author

For any questions or issues, please open an issue on GitHub: [@Siddharth Mishra](https://github.com/Sid3503)

---

<p align="center">
  Made with ❤️ and lots of ☕
</p>
