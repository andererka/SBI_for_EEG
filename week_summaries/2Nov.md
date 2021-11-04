What I did last days:

- write_to_file class: saves posterior, prior, meta-data etc.
- paper to normalizing flows

What I need to get done?

- parameter histogram check: Can I easily do something similar as presented by Michael yesterday in order to check summary stats? Steps that I understood: synthesize data several times and then sample --> 



**What are normalizing flows?**

Construction method for approximating posterior distributions. It uses a sequence of invertible transformations, starting with a simple density, until reaching a desired level of complexity.

- allow for extremely rich posterior approximations; in the asypmtotic regime, space of solutions is large enough to contain posterior

<u>Paper:</u> Variational Inference with Normalizing Flows (Rezende, Mohamed)

Finite Flows versus Infinitesimal Flows: The latter describes how the inital density evolves over time with continous-time dynamics

<u>Langevian Flow:</u> with drift vector and diffusion matrix. Dependent on the choice of drift vector and diffusion matrix, solution is often given by Boltzman distr. in ML

<u>Planar Flow:</u> Series of contractions and expansions in the direction to the hyperplane

<u>Radial Flow:</u> Series of contractions and expansions around reference point

<u>General normalizing flow:</u> linear time computation of the Jacobian

<u>Volume-preserving flow</u>: Jacobian-determinant is equal to one

-> Can these last two be mixed?

**What advantages/disadvantaged?** **What is usually used in the lab/ considered as gold-standard?** -> NSF

