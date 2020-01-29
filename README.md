# Unit commitment with Kazarlis et al. (1996) genetic algorithm

A solution the unit commitment problem based on the genetic algorithm described in "A genetic algorithm solution to the unit commitment problem" by Kazarlis et al. (1996). 

This algorithm will be used as a benchmark for reinforcement learning solutions to the UC problem.

Features of the Kazarlis paper which are not yet included in this implementation include: 

- Adaptation of operator probabilities: varying the operator probabilities over time may improve performance. This may include being able to determine premature convergence and correspondingly adjusting the mutation (promoting diversity) and crossover (promoting convergence). 
- Multi-point crossover
