# Classical Fokker-Planck Equation using a DG Method with AMR in Dealii

This is a modified and combined version of the deal.ii tutorial [step 12](https://www.dealii.org/current/doxygen/deal.II/step_12.html) and [step 26](https://www.dealii.org/current/doxygen/deal.II/step_26.html). After cloning the repo it can be executed like any other deal.ii tutorial. For example by

```
cmake -s source -b build
cd build/
make run
```

## Strong and Weak Formulation
The classical Fokker-Planck equation describes the evolution of a distribution $w(t, x, p)$ in phase space with position $x$ and momentum $p$ due to a Hamiltonian flow and diffusion caused by coupling to an environment. Its strong formulation is (see [[1]](#1))

$$
  \frac{\partial w}{\partial t} + \nabla (X_H w) - D\frac{\partial^2 w}{\partial p^2}=0 \quad \text{in} \quad \Omega\times (0,T),
$$

$$
  w = 0 \quad \text{on} \quad \partial\Omega\times (0, T),
$$

$$
  w(\cdot, \cdot, t=0) = w_0 \quad \text{in} \Omega,
$$

where $X_H=(p/m, -\partial V /\partial x)$ is the Hamiltonian vector field, $D$ is the diffusion constant and $w$ is the phase space probability density.

[[2]](#2) shows how to construct a weak formulation for diffusion-advection-reaction problems with all mathematical details. Using the same function spaces and notation I modified the weak form therein to treat the time dependent problem above. In short the weak form  now reads: Find $w \in V_{*h}$ such that

$$
  a^\text{FP}(w, v_h) = \int_\Omega \frac{\partial w}{\partial t} v_h + a^\text{swip} (w, v_h) + a^\text{upw} = 0\quad \forall \quad v_h \in V_h,
$$

where $a^\text{swip}$ refers to the symmetric weighted interior penalty method, which handles the diffusive part of the equation. It is given by

$$
  a^\text{swip} (w_h, v_h) = \int_\Omega D \hspace{0.1cm} \frac{\partial w_h}{\partial p} \frac{\partial v_h}{\partial p} + \sum_{F \in \mathcal{F}_h^i} (n_F)_p \eta \hspace{0.1cm}\frac{D}{h_f}\int_F [[w_h]][[v_h]]
$$

$$
    - \sum_{F \in \mathcal{F}_h^i} \int_F (n_F)_p \left(\lbrace\lbrace D\hspace{0.1cm} \frac{\partial w_h}{\partial p}\rbrace\rbrace[[v_h]] + [[w_h]]\lbrace\lbrace\hspace{0.1cm}D \hspace{0.1cm}\frac{\partial v_h}{\partial p}\rbrace\rbrace\right),
$$

where $[[\cdot]]$ and $\lbrace\lbrace\cdot\rbrace\rbrace$ refer to the usual jump and average operations respecively and $\mathcal{F}_h^i$ is the set of interior faces. $\eta$ is a user desfined parameter, that enforces coercivity. Note that the momentum components of the normal vectors $(n_F)_p$ were taken in order to let the diffusion act in the momentum direction exclusively.
Upwinding has already been implemented for the transport equation by the deal.ii tutorial. The respective weak form is given by


$$
  a^\text{upw} = -\int_\Omega (\nabla w_h)\cdot X_H v_h + \sum_{F \in \mathcal{F}_h^i} \int_F [[w_h]] v_h^{upwind} X_h \cdot n_F
$$

$$
  +\int_{\Gamma^+} w_h v_h X_h\cdot n,
$$

where $\Gamma^+$ refers to the inflow part of the boundary.

Time discretization is simply done by a implicit Euler time stepping scheme. I may come back to this in the future to implent Crank-Nicolson, etc.


## Results
For the particular case of a driven double well potential $V(x, t) = B x^4 - A x^2 + \Lambda x \cos(\omega t)$ with $m=1$, $B=0.5$, $A=10$, $\Lambda=10$ and $\omega=6.07$, as found in [[1]](#1), and initial condition

$$
  w_0(x, p) = \frac{1}{2\pi}\exp(-(x^2 + p^2)/2)
$$

we obtain the following result over 8 periods of the driving force:



https://github.com/user-attachments/assets/e9ac8f40-bb84-4689-b7ac-c31d013e8a34




## References
<a id="1">[1]</a> 
Habib, Salman & Shizume, Kosuke & Zurek, Wojciech Hubert. (1998). Decoherence, Chaos, and the Correspondence Principle. 10.1103/PhysRevLett.80.4361

<a id="2">[2]</a> 
Di Pietro, Daniele & Ern, Alexandre. (2012). Mathematical Aspects of Discontinuous Galerkin Methods. 10.1007/978-3-642-22980-0.
