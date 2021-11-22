Idea of implementing SBI for ERP as a sequential estimation problem

- doing inference in 'time steps'
- filter out simulations that do not fit in as time goes on
- Literatur zu Hierarchical Models: Bishop 372 (Hierarchical Bayesian models), 605 (Sequential data):
   - In the stationary case, the data evolves in time, but the distribution from
     which it is generated remains the same
   - 'An alternative approach is to use a parametric model for
     p(x n |x n−M , . . . , x n− 1 ) such as a neural network. This technique is sometimes
     called a tapped delay line because it corresponds to storing (delaying) the previous
     M values of the observed variable in order to predict the next value. The number
     of parameters can then be much smaller than in a completely general model (for ex-
     ample it may grow linearly with M ), although this is achieved at the expense of a
     restricted family of conditional distributions' (p.609)
   - 'If both the latent and the observed variables are Gaussian (with
     a linear-Gaussian dependence of the conditional distributions on their parents), then
     we obtain the linear dynamical system.'
- würden wir die Parameter und summary statistics in Form einer bestimmten Verteilung an den nächsten 'Step' übergeben können? Posterior von ersten Parametern sollte 'enger' sein und damit Simulation beschleunigen. 
- can we use network again or do we have to retrain?
- Idee: in zweitem Schritt 'sampled' man von posterior der schon geschätzten Parameter?
