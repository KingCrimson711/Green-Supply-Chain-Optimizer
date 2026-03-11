

# Adaptive Carbon-Aware Routing

## Description
Adaptive Carbon-Aware Routing is a machine learning-driven framework for modeling context-dependent carbon emissions in urban road networks. Traditional carbon-aware routing assumes static emission rates for each road segment, which can misrepresent real-world conditions. This project uses a lightweight neural network to predict edge-level emission costs based on contextual features like traffic, road characteristics, and environmental factors. These predicted costs are then integrated into routing pipelines to enable data-driven, emission-aware route optimization. 

The goal is to provide more realistic and sustainable routing decisions for planners, policy-makers, and researchers, while remaining compatible with classical shortest-path algorithms.

---

## Features
- Learn context-dependent emission costs using a neural network.
- Predict dynamic edge-level emission costs for any road network.
- Reweight graph edges to enable emission-aware routing.
- Generate alternative routes using iterative shortest-path with stochastic perturbations.
- Analyze routing outcomes under static vs. learned emission costs.
- Modular and scalable architecture for integration into existing routing pipelines.

---

## System Architecture
1. **Road Network Graph (G = (V, E))**  
   Nodes represent intersections; edges represent road segments.
2. **Feature Extraction**  
   Each edge is represented by an 11-dimensional feature vector capturing structural, operational, and environmental attributes.
3. **Neural Network Model (ˆfθ)**  
   - Input layer: 11 features  
   - Hidden layer: 64 neurons (ReLU activation)  
   - Output: scalar emission cost per edge
4. **Training**  
   Mean Squared Error (MSE) loss minimized using gradient-based optimization.
5. **Graph Reweighting**  
   Replace static emission coefficients with predicted costs.
6. **Emission-Aware Routing**  
   Standard shortest-path algorithms applied on the reweighted graph.

---
