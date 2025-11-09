# Applied Optimization and Probabilistic Modeling in Data Science

**Note:** This repository is currently being organized and cleaned. Some files and structure may change as documentation and code are updated.

## Overview
This project demonstrates key analytical and optimization techniques used in data science, focusing on both deterministic and stochastic problem solving methods. It integrates data preprocessing, optimization algorithms, and probabilistic reasoning through practical implementations and comparative analysis.

## Objectives
- Evaluate the impact of data preprocessing on model performance and runtime  
- Compare Linear Programming (LP) and Particle Swarm Optimization (PSO) for solving constrained optimization problems  
- Design and implement a Bayesian Network for probabilistic inference in medical diagnosis and treatment decisions  
- Analyze model efficiency, accuracy, and scalability across classical and modern approaches  

## Key Topics
- Data cleaning, feature scaling, and outlier handling  
- Linear programming (scipy.optimize.linprog)  
- Particle Swarm Optimization (pyswarms)  
- Probabilistic graphical models (Bayesian Networks)  
- Comparative performance analysis and complexity evaluation  

## Technical Stack
- Languages and Libraries: Python, NumPy, Pandas, Matplotlib, PySwarms, SciPy, pgmpy  
- Concepts Demonstrated:  
  - Optimization under constraints  
  - Stochastic vs deterministic search  
  - Probabilistic inference (exact and approximate)  
  - Runtime and computational complexity analysis  

## Highlights
- Implemented and compared LP and PSO on the same objective function, validating convergence and precision differences  
- PSO achieved an approximate optimum of f(x*) ≈ -16.99999 at (x1, x2) = (2, 3), closely matching the LP exact solution  
- Constructed a Bayesian Network to perform exact and approximate inference, analyzing how graph structure affects runtime  
- Evaluated trade offs between model transparency, computational cost, and flexibility  

## Insights
- Preprocessing had minimal impact on runtime due to limited outlier presence but highlighted the importance of scalable cleaning pipelines  
- Linear programming provided deterministic precision and efficiency for convex problems  
- Particle Swarm Optimization excelled in nonlinear or derivative free search spaces, at higher computational cost  
- Bayesian inference demonstrated how graph topology and conditional independence shape decision quality in uncertain systems  

## Potential Extensions
- Extend PSO testing to high dimensional or multimodal objective functions  
- Integrate Markov Chain Monte Carlo (MCMC) for Bayesian inference  
- Apply hybrid optimization frameworks combining LP initialization with PSO refinement  
- Explore real world applications such as clinical decision support or resource allocation  

## Author
**Addison DeSalvo**  
Johns Hopkins University — M.S. Data Science  
Focus Areas: Swarm AI, Machine Learning, and Public Health Applications
