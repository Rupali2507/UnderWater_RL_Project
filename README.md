# AUV-SIMULATION

### Multi-Agent Maritime Reinforcement Learning Environment for Tactical Naval Operations

AUV-SIMULATION is a large-scale multi-agent maritime combat and convoy-protection simulator built for reinforcement learning research.

The project combines:

* Geographic mission generation
* Real-world maritime regions
* Multi-agent PPO training
* Surface and underwater vehicle dynamics
* Sensor-based perception
* Convoy protection strategies
* Threat interception
* Dynamic scenario generation
* Interactive mission visualization

The simulator models a realistic naval task force attempting to escort high-value cargo through contested waters while hostile fleets attempt to intercept and destroy it.

---

## Mission Scenario

A convoy consisting of cargo ships, naval escorts, and submarines must safely navigate from a starting position to a destination waypoint.

Enemy vessels are spawned outside the defensive perimeter and attempt to locate, intercept, and destroy the convoy before it reaches its objective.

The simulation evolves into a continuous interaction between:

* Navigation
* Formation control
* Target detection
* Threat interception
* Convoy protection
* Tactical decision making

---

## Core Architecture

### Blue Team

#### Cargo Ships

Primary mission objective.

Responsibilities:

* Navigate toward destination
* Survive hostile encounters
* Maintain mission continuity

#### Naval Escorts

Responsibilities:

* Maintain defensive formation
* Protect cargo vessels
* Intercept hostile contacts
* Engage enemy vessels entering defensive zones

#### Submarines

Responsibilities:

* Operate covertly
* Detect approaching threats
* Launch surprise attacks
* Extend convoy defense perimeter

---

### Red Team

Enemy naval assets.

Responsibilities:

* Detect convoy
* Close distance
* Attack cargo vessels
* Break convoy protection

---

## Observation Space

Each agent receives a tactical observation vector containing:

### Self State

* Normalized X Position
* Normalized Y Position
* Velocity
* Heading Sin
* Heading Cos
* Health

### Navigation Information

* Distance to Destination
* Direction to Destination

### Threat Awareness

* Distance to Nearest Enemy
* Direction to Nearest Enemy

### Convoy Awareness

* Distance to Nearest Cargo Ship
* Direction to Nearest Cargo Ship

These observations allow agents to learn coordinated behavior rather than relying on scripted rules.

---

## Reinforcement Learning

The simulator uses a shared-policy PPO architecture.

Each vehicle class learns a specialized policy:

* CargoShip Policy
* NavalShip Policy
* Submarine Policy

Policies are trained through self-play across thousands of procedurally generated missions.

---

## Domain Randomization

Every episode generates a new mission:

* New start location
* New destination
* New convoy geometry
* New enemy formation
* New interception opportunities

This prevents overfitting and encourages robust tactical behavior.

---

## Geographic Simulation

Supported operational regions include:

* Strait of Hormuz
* South China Sea
* Arabian Sea
* Indian Ocean
* Custom Mission Regions

Each region is converted into a local metric coordinate frame for high-performance reinforcement learning while preserving geographic realism.

---

## Physics Engine

The environment uses batched PyTorch physics for large-scale simulation.

### Surface Vessels

State Vector:

```text
[X, Y, Z, Roll, Pitch, Yaw,
 u, v, w, p, q, r,
 Health, Fuel]
```

Features:

* Throttle control
* Rudder steering
* Drag modeling
* Fuel consumption
* Heading dynamics

### Underwater Vehicles

Features:

* Surge motion
* Dive control
* Depth management
* Underwater maneuvering

All physics updates are vectorized for efficient training.

---

## Sensor Framework

Vehicles are equipped with modular sensing systems.

Supported sensors include:

* Radar
* Sonar
* Visual Detection
* AIS Receiver
* Communication Modules

The SensorSuite architecture allows new sensing technologies to be integrated without modifying vehicle logic.

---

## Tactical Zones

The convoy protection system is organized into concentric defensive layers.

### Zone C

Inner cargo protection zone.

### Zone B

Escort operating zone.

### Zone A

Outer detection and interception zone.

Enemy vessels are spawned beyond Zone A and must penetrate multiple defensive layers before reaching cargo assets.

---

## Visualization

### Real-Time Tactical Display

* Live fleet positions
* Heading visualization
* Sensor overlays
* Agent inspection tools
* Pygame-based battlefield rendering

### After Action Reports

Interactive Folium-based mission replay including:

* Fleet trajectories
* Start positions
* Destination markers
* Waypoints
* Geographic mission replay

---

## Training Pipeline

```text
Scenario Generator
        │
        ▼
PettingZoo Environment
        │
        ▼
Physics Engine
        │
        ▼
Observation Builder
        │
        ▼
PPO Policies
        │
        ▼
Reward Calculation
        │
        ▼
Policy Update
```

---

## Research Objectives

This project investigates emergent behaviors in autonomous maritime systems including:

* Convoy escort formation
* Multi-agent cooperation
* Threat interception
* Autonomous maritime defense
* Underwater warfare tactics
* Distributed sensing
* Communication-aware decision making

---

## Technology Stack

### Reinforcement Learning

* PyTorch
* PPO (Proximal Policy Optimization)
* PettingZoo

### Simulation

* Custom Maritime Physics Engine
* Multi-Agent Environment
* Domain Randomization

### Visualization

* Pygame
* Folium
* Streamlit

### Geographic Processing

* Geographic Coordinate Frames
* Local Metric Projection
* Real-World Maritime Regions

---

## Future Roadmap

* Multi-Agent Communication Learning
* Hierarchical Fleet Command
* Aircraft Integration
* Electronic Warfare Systems
* Missile Engagement Models
* Swarm Coordination
* Graph Neural Network Policies
* Transformer-Based Tactical Memory
* Human-in-the-Loop Command Systems

---

## Vision

AUV-SIMULATION aims to become a realistic research platform for studying autonomous maritime operations where fleets learn to coordinate, defend, navigate, and fight in dynamic and geographically realistic environments.
