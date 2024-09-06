# Dynamic Cutting Approach for Simulating Some Lévy Processes

This repository contains a collection of Python scripts designed to simulate **symmetric** α-stable and tempered-stable processes. The implementation is based on the numerical examples (4.1 and 4.2) provided in the paper [...].

![example_plot_alpha_stable](https://github.com/d-platonov/levy-processes-dynamic-cutting/assets/173836765/07c2ddfe-eb25-4bb5-a246-0bdbdfa67117)

**Note:** This is a simplified version, focused solely on simulating symmetric α-stable processes for simplicity, as the goal was to provide simulations specifically for the paper, which only considered the symmetric case. If you need to extend the simulations to include asymmetric cases, additional modifications will be required.

**Simulation Details:** The Python script uses a relatively small number of simulations for the Monte Carlo method to quickly display results. To obtain the same results as shown in the paper, you need to increase the number of simulations as described in the paper. However, note that this will take several hours to complete.


## Files Description

- `alpha_stable_process.py`: Defines the `AlphaStableProcess` class, which includes methods for simulating alpha-stable Levy processes.
  
- `approximation_type.py`: Contains the `ApproximationType` enumeration, which specifies the method for simulating alpha-stable processes.

- `monte_carlo.py`: Implements Monte Carlo methods for estimating moments and cumulants of the alpha-stable and tempered-stable processes, leveraging the simulations from `alpha_stable_process.py`.

- `run_simulation.py`: A script that coordinates the running of Monte Carlo simulations for both alpha-stable and tempered-stable processes.

## Usage

To run the simulation and view the results, execute the following command:

```bash
python run_simulation.py
