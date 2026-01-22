# Exoplanet System Generator

AI-driven stellar system generation & N-body physics simulation with 3D visualization. The model can generate systems with planetary masses all the way down to tiny asteroid-like objects (0.0001) up to gas giants and brown dwarfs (100,000 Earth masses). That huge range is because real exoplanet data spans everything from sub-Earth planets to super-Jupiters, so the model learned that full spectrum of possibilities.  

## TL;DR [**Demo**](https://adamfehse.github.io/someoneWondered/)

**Tech**: Transformer (RoPE, AdamW, cosine annealing) (PyTorch) + FastAPI + Three.js

**Backend**: Render
- **Data**: ~6,080 real exoplanets from NASA Exoplanet Archive
- **Model**: Decoder-only transformer (6 layers, 4 heads, RoPE embeddings)
- **Task**: Autoregressive generation of exoplanet orbital parameters
- **Application**: Generate planetary systems for stars like Vega

---

## Full ML Training Pipeline

### 1. Data Pipeline (Preprocessing)

**Source**: NASA Exoplanet Archive API (~6,080 confirmed exoplanets)

**EDA & Feature Engineering**:
```
Fetch ~6,080 exoplanet records from NASA TAP query
Filter for multi-planet systems (2+ planets per system)
Extract orbital elements (mass, semi-major axis, eccentricity, inclination, arg_periapsis)
Normalize to [0,1] range (log scale for mass/orbit, linear for angles)
Quantize to 8-bit tokens [0-255]
Augment 10x (add random angle noise ±5 tokens)
```

**Key Parameters from Data**:
- **Mass**: 0.0001 - 100,000 Earth masses (log-normal distribution)
- **Semi-major axis**: 0.001 - 100 AU (log-normal, peaks near star)
- **Eccentricity**: 0 - 0.93 (peaked at 0, long tail of eccentric orbits)
- **Inclination**: 0 - 3.07 radians (concentrated near 90° for transiting systems)

**Data Augmentation**: Add Gaussian noise to angular tokens (±5) for 10x expansion

---

### 2. Training Loop

**Architecture**: Decoder-only transformer (GPT-style)
```
Input tokens: (batch, 64)
Token embedding: (batch, 64, 128)
Rotary Positional Embeddings (RoPE)
6 Transformer Blocks
  - Multi-head self-attention (4 heads, causal masking)
  - Feed-forward network (2-layer MLP)
  - Layer normalization + residual connections
Layer norm + output projection: (batch, 64, 256)
```

**Loss Function**: Cross-entropy on next-token prediction

**Optimization**:
- Optimizer: AdamW (lr=1e-4, weight decay=0.01)
- Scheduler: CosineAnnealingLR (cosine decay over epochs)
- Gradient clipping: 1.0
- Batch size: 32 (configurable)

**Validation**: 90/10 train/val split, checkpoint best model on val loss

### 3. Colab Setup

Complete training pipeline on Google Colab (free T4 GPU).

### 4. EDA & Validation

Exploratory Data Analysis on a broader dataset of 6080 confirmed exoplanet records.

**Data Completeness**: Significant sparsity in many key parameters (like pl_masse, pl_rade, pl_eqt, pl_orbsmax, pl_orbeccen, pl_orbincl, pl_orblper, pl_bmasse), indicating that not all data is available for every exoplanet. This highlightes the importance of preprocessing steps where we filtered for complete records to train the generative model.

**Key Planetary Parameter Distributions**:
- **Orbital Period (pl_orbper)**: Showed a broad, log-normal distribution, with a peak for periods ranging from a few days to a few tens of days.
- **Planet Mass (pl_masse, pl_bmasse) and Radius (pl_rade)**: Both also exhibited log-normal distributions, with a prevalence of smaller planets but a long tail towards gas giants.
- **Semi-major Axis (pl_orbsmax)**: Displayed a log-normal distribution, with many planets orbiting very close to their stars.
- **Orbital Eccentricity (pl_orbeccen)**: Peaked near zero (circular orbits) but showed a significant number of eccentric orbits.
- **Orbital Inclination (pl_orbincl)**: Most inclinations were around 90 degrees, which is expected for transiting planets.
- **Longitude of Periastron (pl_orblper)**: Showed a more uniform distribution, but with many missing values.

**Stellar Parameter Distributions**:
- **Effective Temperature (st_teff)**: Peaked around 5000-6000 K (Sun-like stars), with a considerable number of cooler stars.
- **Stellar Mass (st_mass) and Radius (st_rad)**: Skewed towards smaller stars, consistent with the abundance of M-dwarfs.

**System Demographics**:
- **Planets per System (sy_pnum)**: Single-planet systems were by far the most common, followed by systems with two, three, or more planets. This clarified that 6000+ planets do not mean 6000+ unique systems.
- **Discovery Methods**: The Transit method was shown to be the dominant discovery technique, followed by the Radial Velocity method.

**Model Validation**:
- Generates physically plausible orbital parameters
- Learned log-normal distributions from real data
- Can condition on central mass and number of planets
- Diverse generation with temperature/top-k sampling