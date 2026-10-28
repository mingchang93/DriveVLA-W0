# Diffusion models: why the network predicts noise

A self-contained primer. No prior diffusion background assumed; the only
prerequisites are "a neural network can regress a tensor" and "MSE prefers the
average".

---

## 1. The problem: sampling, not fitting

We want a model that can produce a sample from a complicated distribution —
images, or in the case that motivates this note, the next video frame. Two
things are being asked at once:

1. represent the *distribution* of plausible outputs,
2. produce one *particular* member of it on demand.

A regression network trained with MSE does neither well. If the future is
multi-modal (the truck ahead either turns or stops), the MSE-optimal prediction
is the average of the two futures — an image that is neither, and that has
never existed.

Diffusion solves this by never asking the network to produce a sample in one
shot. Instead it asks a much easier question, over and over.

---

## 2. The forward process: destroy the data in a known way

Take a clean sample $x_0$ and a variance schedule $\beta_1 \dots \beta_T$ with
$\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$. The
forward (noising) process adds Gaussian noise in $T$ small steps,

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{\alpha_t}\,x_{t-1},\ \beta_t \mathbf{I}\right),
$$

which composes in closed form:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\mathbf{I}\right)
\quad\Longleftrightarrow\quad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0,\mathbf{I}).
$$

That second form is the whole trick. For any $t$ you can jump straight to a
noisy sample with **one** line of code, and you know the exact noise you added.

- small $t$: $x_t$ is almost $x_0$ — a slightly damaged image;
- large $t$: $x_t$ is almost pure noise — all structure gone.

---

## 3. The training objective: recover the noise

Train a network $\epsilon_\theta(x_t, t)$ to answer: *given this noisy tensor
and how much noise was added, what was the noise?*

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_0,\,\epsilon,\,t}
\left[ \left\| \epsilon - \epsilon_\theta\!\left(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t\right)\right\|_2^2 \right].
$$

Three reasons this is a good learning problem:

- **The label is free and exact.** We generated $\epsilon$, so the target is
  perfect at every sample, for every $t$, with no annotation.
- **One network covers all noise levels.** $t$ is sampled uniformly, so the
  network learns fine detail at small $t$ and global layout at large $t$.
- **It teaches the geometry of the data.** To say *which direction is less
  noisy* from an arbitrary point in space, the network must learn where the
  data actually lives.

### Why not predict $x_0$ directly?

You can. Given $x_t$ and $t$, predicting $\epsilon$ and predicting $x_0$ are
linear reparameterisations of each other:

$$
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

$\epsilon$-prediction is the common choice because it is better scaled at high
noise (where $x_0$ is nearly unrecoverable but $\epsilon$ is exactly what was
added) and it is directly the quantity the sampler consumes.

### What the network is really learning

For a Gaussian perturbation kernel, the optimal denoiser is proportional to the
**score** of the noised distribution:

$$
\nabla_{x}\log q_t(x) \;\approx\; -\,\frac{\epsilon_\theta(x,t)}{\sqrt{1-\bar\alpha_t}}.
$$

So "predict the noise" and "estimate the gradient of the log-density" are the
same function up to a known constant. Denoising is score matching in disguise.

---

## 4. The reverse process: turn the noise back into data

At sampling time, start from pure noise and walk *down* the noise levels:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}
\left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t) \right)
+ \sigma_t z,
\qquad z \sim \mathcal{N}(0,\mathbf{I}).
$$

Repeat for $T$ steps; each step removes a little noise. The injected
$\sigma_t z$ is what keeps the trajectory stochastic — that randomness is why
different seeds produce different, individually plausible samples instead of
the blurry mean. (DDIM and other samplers drop or reshape that term to trade
steps for diversity.)

**Nothing about the loss changes between training and sampling.** The network
is the same; sampling is just a loop that calls it many times.

---

## 5. Why a UNet

The denoiser maps a tensor to a same-shaped tensor with spatial structure at
several scales simultaneously: a pixel-level artifact is a local problem, a
wrong horizon is a global one. A UNet — downsampling blocks, a bottleneck,
upsampling blocks with skip connections — handles exactly that shape of problem
cheaply, and pretrained ones are widely available. There is nothing
diffusion-specific about the choice; any architecture that preserves shape and
sees multiple scales would work.

---

## 6. Latent diffusion: do it in a compressed space

Running all of this on megapixel RGB is wasteful. Latent diffusion first trains
(or reuses) an autoencoder, then runs the entire noising/denoising process in
its bottleneck:

$$
x \ \xrightarrow{\ \mathcal{E}\ }\ z \ \xrightarrow{\ \text{diffuse / denoise}\ }\ \hat z \ \xrightarrow{\ \mathcal{D}\ }\ \hat x .
$$

The VAE compresses spatially by $8\times$ and to few channels. A worked example
from the model this note was written for:

| space | tensor | scalars |
|---|---|---|
| pixels | $3 \times 280 \times 504$ | 423,360 |
| latent | $4 \times 35 \times 63$ | 8,820 |

≈48× fewer numbers for the network to denoise, and the encoder/decoder are
trained once and then frozen — the diffusion model only ever touches latents.

---

## 7. Conditioning, and what "guidance" means

To generate *a specific* thing rather than a random one, the denoiser takes an
extra input $c$:

$$
\epsilon_\theta(x_t, t, c).
$$

How $c$ enters is an architecture choice: additive feature maps at the input,
cross-attention, adaptive normalisation, or a separate control branch.

**Classifier-free guidance** trains the same network both with and without $c$
by randomly dropping the condition, then at sampling time extrapolates away
from the unconditional prediction:

$$
\hat\epsilon = \epsilon_\theta(x_t,t,\varnothing)
+ s\left(\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\varnothing)\right),
\qquad s > 1 .
$$

That is what a "negative prompt" is: the unconditional branch, whose embedding
is whatever the network was trained to associate with *no condition*.

---

## 8. Summary

| question | answer |
|---|---|
| What does the network output? | the noise that was added, not the data |
| Why is that trainable? | the noise is generated by us, so the label is exact and free |
| Why does it sample? | removing predicted noise is a step toward the data manifold; loop it |
| Why does it not blur? | the sampler injects fresh randomness, so each run commits to one mode |
| Where does the data live? | in the latent space of a frozen autoencoder |
| What is guidance? | interpolation between conditional and unconditional noise predictions |

One sentence: **diffusion trains a network to remove noise of every severity,
and generation is that removal run in reverse from pure noise.**
