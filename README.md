# Spatial eco-evolutionary community assembly

## Overview

This repository implements a spatially explicit eco-evolutionary community assembly model.

The model tracks individuals across habitat patches under environmental filtering, trait-mediated competition, dispersal, mutation, and stochastic disturbance. Community structure emerges from local demographic processes coupled to landscape-level environmental heterogeneity.

## Motivation

Community assembly is shaped by multiple interacting processes, including dispersal, local environmental selection, biotic interactions, and stochastic turnover. In many systems, these processes also feed back on trait distributions through time.

This model is designed to explore how biodiversity patterns emerge when ecological and evolutionary dynamics operate simultaneously in a spatially structured landscape.

## Model structure

Each individual is characterised by:
- species identity
- trait value
- patch location

Each habitat patch is characterised by:
- environmental optimum
- carrying capacity
- disturbance probability

At each time step the model updates:

1. patch environments
2. individual survival
3. local reproduction
4. trait inheritance and mutation
5. dispersal among patches
6. density-dependent recruitment
7. stochastic disturbance

## Key mechanisms

### Environmental filtering
Individual performance declines with increasing mismatch between trait value and local environment.

### Trait-mediated competition
Competitive effects are stronger among phenotypically similar individuals.

### Dispersal limitation
Recruitment depends on spatial connectivity among patches.

### Eco-evolutionary feedback
Trait distributions shift through inheritance and mutation, altering later assembly dynamics.

## Outputs

The simulation records:
- local and regional species richness
- beta diversity among patches
- patch-level mean trait values
- trait variance through time
- occupancy distributions
- turnover following disturbance

## Repository contents

- `src/main.py` — run script
- `src/model.py` — model classes and state
- `src/processes.py` — ecological and evolutionary updates
- `src/landscape.py` — patch network and environment generation
- `src/analysis.py` — summaries and plots

## Future extensions

- trophic interactions
- phylogenetic structure
- adaptive dispersal
- temporal autocorrelation in disturbance
- multi-trait assembly
