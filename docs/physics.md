# Methods, Conventions, and Polarization Topology

This page details the core numerical methods, conventions, and topological concepts utilized in VectorWaves.

## 1. Discrete Plane-Wave Expansions

VectorWaves constructs arbitrary non-paraxial fields through the superposition of plane waves. The continuous angular spectrum is discretized over the source's solid angle using a **Fibonacci-sphere quadrature**.

The Fibonacci construction provides near-uniform angular sampling over the spherical cap, avoiding the polar clustering associated with conventional latitude/longitude grids. Each sampled direction $\hat{\mathbf{k}}_n$ is associated with a solid-angle element $\Delta\Omega_n$, so the discrete expansion approximates the corresponding angular-spectrum integral as the number of modes is increased.

The number of modes is controlled by `config.source.num_modes`, while the sampled solid angle is controlled by `config.source.theta_max`. Increasing the number of modes improves the angular discretization, with the required resolution depending on the angular spectrum and the spatial structure being computed.

### Angular-spectrum and power normalization

For a monochromatic wavelength, the wavevectors are

$$
\mathbf{k}_n = \frac{2\pi}{\lambda}\hat{\mathbf{k}}_n.
$$

The user-specified k-space profile produces a complex amplitude $A_n$ for each sampled direction. Before the discrete modes are constructed, the raw angular-spectrum power is evaluated as

$$
P_{\mathrm{raw}}
=
\sum_n |A_n|^2\,\Delta\Omega_n.
$$

The spectrum is then normalized by $1/\sqrt{P_{\mathrm{raw}}}$, with the solid-angle factor $\Delta\Omega_n$ included in the resulting discrete mode coefficient. Thus the discrete expansion retains the weighting associated with the underlying solid-angle integral rather than treating the sampled directions as an unweighted list of plane waves.

The source `intensity_scale` enters through the overall amplitude normalization as

$$
E \propto \sqrt{\text{intensity\_scale}},
$$

so field intensities and power-like quantities scale linearly with `intensity_scale`.

For polychromatic fields, the spectral-line weights are normalized in $L^2$ before the overall source scaling is applied. This keeps the total spectral weighting normalized independently of the particular shape of the spectral profile.

When `randomize.amplitude` is enabled, each mode amplitude is multiplied by

$$
\frac{1}{\sqrt{2}}
\left(X+iY\right),
\qquad
X,Y\sim\mathcal N(0,1),
$$

giving a zero-mean complex Gaussian random variable with unit variance. This preserves the expected ensemble power of the normalized spectrum while producing fully developed speckle statistics.

`intensity_E` is a real-space scalar representing the point-by-point squared electric-field magnitude,

$$
\text{intensity\_E}(\mathbf r) = |\mathbf E(\mathbf r)|^2.
$$

### Polarization Transport

Each plane-wave electric field must satisfy Maxwell's transversality condition,

$$
\mathbf{k}_n\cdot\mathbf{E}_n=0.
$$

The user specifies a base Jones vector

```text
config.source.pol_vect = (px, py)
```

relative to the macroscopic propagation direction `config.source.beam_axis`.

For each sampled direction $\hat{\mathbf{k}}_n$, VectorWaves constructs a local transverse basis using a Rodrigues rotation that transports the reference transverse frame from the beam axis to $\hat{\mathbf{k}}_n$. The Jones components are then expressed in this local basis, ensuring that the resulting polarization vector is transverse to the corresponding wavevector.

In the absence of polarization randomization, the mode polarization has the form

$$
\mathbf P_n = p_x\mathbf e_{1,n}+p_y\mathbf e_{2,n},
$$

where $\mathbf e_{1,n}$ and $\mathbf e_{2,n}$ span the plane perpendicular to $\hat{\mathbf{k}}_n$.

Optional polarization-state and polarization-angle randomization are applied independently to these local transverse frames.

---

## 2. Conventions and Units

### Units and phase

VectorWaves uses natural units with

$$
c=1.
$$

The phase of the $n$-th plane-wave mode is

$$
\phi_n(\mathbf r,t)
=
\mathbf k_n\cdot\mathbf r-\omega_n t.
$$

Consequently, time is expressed in units of length, equivalent to $ct$ in SI units.

The stored wavenumber has units

$$
k\;[\mathrm{rad}/\mathrm{spatial\_unit}],
$$

where the spatial unit is determined by `op.size` and `op.spacing`.

### Coordinate conventions

The default macroscopic propagation axis is $+z$.

`compute_on_op` evaluates the field on a transverse $(x,y)$ plane at a specified $z$.

`compute_point` evaluates the field at a single arbitrary 3D spatial point.

`compute_cloud` evaluates the field at an arbitrary one-dimensional collection of 3D points. No spatial connectivity or regular-grid structure is assumed; the points constitute an unstructured point cloud.

### Jacobian convention

Spatial derivatives use the tensor convention

$$
\texttt{jacobian\_E}[i,j,\ldots]
=
\frac{\partial E_i}{\partial x_j}.
$$

The first index identifies the field component $(x,y,z)$, while the second identifies the spatial derivative direction $(x,y,z)$.

The divergence is the trace,

$$
\nabla\cdot\mathbf E
=
\operatorname{tr}(\nabla\mathbf E),
$$

while the curl is obtained from the antisymmetric part of the spatial Jacobian.

### Stokes convention

The polarization state can be characterized using the Stokes parameters. In particular, $S_3$ determines the circular-polarization component according to the convention used by VectorWaves.

The sign convention is exposed directly through $S_3$, so the absolute naming of $S_3=\pm1$ as LCP or RCP is less important than maintaining the same convention consistently throughout the polarization and singularity calculations.

---

## 3. Polarization Singularities

In a general 3D electromagnetic field, the polarization ellipse need not lie in the transverse $xy$-plane.

For a complex electric field

$$
\mathbf E
=
\operatorname{Re}(\mathbf E)
+
i\,\operatorname{Im}(\mathbf E),
$$

the real and imaginary parts span the plane of polarization. Its oriented area vector is

$$
\mathbf N
=
\operatorname{Re}(\mathbf E)
\times
\operatorname{Im}(\mathbf E).
$$

### $C^T$-points: true 3D circular polarization

A $C^T$-point satisfies

$$
\mathbf E\cdot\mathbf E=0 
\implies \operatorname{Re}(\mathbf E\cdot\mathbf E)=0,
\operatorname{Im}(\mathbf E\cdot\mathbf E)=0.

$$

where the dot product is the complex bilinear dot product, with no complex conjugation. At such a point, the two semi-axes of the 3D polarization ellipse have equal magnitude, so the ellipse becomes a circle in its own polarization plane.

$C^T$-points are located using `find_C_T_points`.

### $L^T$-points: true 3D linear polarization

An $L^T$-point satisfies

$$
\operatorname{Re}(\mathbf E)
\times
\operatorname{Im}(\mathbf E)
=
\mathbf 0.
$$

The area vector of the polarization ellipse therefore vanishes, and the ellipse collapses to a line.

$L^T$-points are located using `find_L_T_points`.

Although this condition contains three scalar components, they are not independent. The vectors $\operatorname{Re}(\mathbf E)$, $\operatorname{Im}(\mathbf E)$, and their cross product are mutually orthogonal, so the three components of $\mathbf N$ do not represent three independent constraints for the purpose of locating a zero in a two-dimensional plane. VectorWaves therefore treats $L^T$-point refinement as a nonlinear least-squares problem and minimizes $\|\mathbf N\|^2$ using Gauss-Newton.

### Stokes C-points : transverse polarization

A conventional Stokes C-point is defined from the transverse polarization state. VectorWaves locates these points using

$$
s_1=s_2=0.
$$

For the transverse field $(E_x,E_y)$, this is equivalently expressed as

$$
E_x^2+E_y^2=0 \implies
\operatorname{Re}(E_x^2+E_y^2)=0,
\operatorname{Im}(E_x^2+E_y^2)=0.
$$

The longitudinal component $E_z$ is not included in this condition. Stokes C-points are located using `find_stokes_C_points`.

#### Morphological Classification

Stokes C-points possess a local morphology determined by the orientation field of the polarization ellipse's major axis.

The Poincaré index of a C-point is

$$
I_C
=
\frac12\,\operatorname{sgn}(D_I),
$$

where

$$
D_I
=
\frac{\partial S_1}{\partial x}
\frac{\partial S_2}{\partial y}
-
\frac{\partial S_1}{\partial y}
\frac{\partial S_2}{\partial x}.
$$

This separates C-points into the standard Star, Lemon, and Monstar morphologies:

* **Star:** $I_C=-\frac12$, corresponding to $D_I<0$. The orientation field contains three hyperbolic sectors.
* **Lemon:** $I_C=+\frac12$, with $D_I>0$ and $NL_{\mathrm{disc}}<0$. The pattern contains one elliptic sector and one radial line.
* **Monstar:** $I_C=+\frac12$, with $D_I>0$ and $NL_{\mathrm{disc}}>0$. The pattern contains three radial lines and mixed parabolic sectors.

These classifications concern the transverse Stokes C-point morphology and should not be conflated with the existence or morphology of a full 3D $C^T$-point.

---

## 4. Numerical Detection of Singularities

Singularity detection uses a two-stage process: discrete candidate extraction followed by continuous numerical refinement.

### Candidate extraction

The initial search operates on the discrete field grid.

For scalar zero conditions, `_zero_cross_mask` identifies $2\times2$ grid cells satisfying the strict sign-change condition

$$
f_{\max}>0
\quad\land\quad
f_{\min}<0.
$$

For the two-component singularity conditions, `_batched_plane_fit` locally approximates the relevant fields by planes and estimates a sub-pixel candidate position

$$
(\delta x,\delta y)\in[0,1)^2.
$$

These candidate positions provide initial guesses for the continuous refinement stage.

### Batched Newton refinement

Candidate points are refined simultaneously using the field-evaluation machinery of `compute_cloud`.

For a two-dimensional system

$$
\mathbf f(x,y)=
\begin{pmatrix}
f_1(x,y)\\
f_2(x,y)
\end{pmatrix}
=
\mathbf0,
$$

Newton-Raphson updates the position according to

$$
\mathbf r_{k+1}
=
\mathbf r_k
-
J^{-1}\mathbf f,
$$

where $J$ is the $2\times2$ Jacobian.

Converged candidates are deactivated dynamically during the batched iterations.

The refinement conditions are:

| Singularity        | Mathematical condition                                                                          | Refinement method | Residual                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------- |
| **Stokes C-point** | $s_1=0,\;s_2=0$                                                                               | 2D Newton-Raphson | $\sqrt{s_1^2+s_2^2}$                                                 |
| **$C^T$-point**  | $\operatorname{Re}(\mathbf E\cdot\mathbf E)=0,\;\operatorname{Im}(\mathbf E\cdot\mathbf E)=0$ | 2D Newton-Raphson | $\|\mathbf E\cdot\mathbf E\|$                                        |
| **$L^T$-point**  | $\operatorname{Re}(\mathbf E)\times\operatorname{Im}(\mathbf E)=0$                            | 2D Gauss-Newton   | $\|\operatorname{Re}(\mathbf E)\times\operatorname{Im}(\mathbf E)\|$ |

For $L^T$-points, the three components of the area vector provide an overdetermined nonlinear residual. The solver therefore uses a Gauss-Newton least-squares formulation, solved through the corresponding pseudoinverse/least-squares system rather than attempting to invert a square Jacobian.

### Spatial deduplication

The same physical singularity could be detected from multiple neighboring grid cells. After refinement, converged roots lying within the spatial tolerance `tol` are merged so that the same topological singularity is not counted multiple times.

### Dimensional and Dimensionless Tolerances

The Stokes C-point residual

$$
R_C=\sqrt{s_1^2+s_2^2}
$$

is dimensionless because the normalized Stokes parameters remove the local intensity scale $S_0$. Consequently, the default

```text
value_tol = 1e-6
```

is approximately invariant under an overall rescaling of the field amplitude.

The $C^T$ and $L^T$ residuals are different. Since they are constructed directly from products of electric-field components,

$$
\mathbf E\cdot\mathbf E
\sim |\mathbf E|^2,
$$

and

$$
\operatorname{Re}(\mathbf E)
\times
\operatorname{Im}(\mathbf E)
\sim |\mathbf E|^2,
$$

their numerical magnitudes depend on the overall field-intensity scale.

This creates an important distinction between the singularity finders. Stokes C-point tolerances are naturally normalized by the local intensity through the Stokes parameters, whereas $C^T$ and $L^T$ refinement uses dimensional residuals.

In low-intensity regions, such as dark spots, the dimensional residuals can become numerically small even when the field is not sufficiently well-conditioned for reliable refinement. Conversely, a tolerance appropriate for a high-intensity region may be too strict in a weak-field region.

In practice, the numerical scale can therefore be controlled either by adjusting `config.source.intensity_scale` or by choosing appropriate absolute tolerances for the $C^T$ and $L^T$ residuals.

---

## 5. 3D Singularity-Line Tracing

In three dimensions, a polarization singularity is generally not an isolated point but a continuous curve.

The functions

```text
trace_stokes_C_lines
trace_C_T_lines
trace_L_lines
```

perform curve continuation from refined singularity seeds using a predictor-corrector scheme.

### Predictor step

A singularity line is represented locally as the intersection of two zero-level surfaces,

$$
f_1(\mathbf r)=0,
\qquad
f_2(\mathbf r)=0.
$$

The tangent to their intersection is therefore

$$
\mathbf t
=
\frac{
\nabla f_1\times\nabla f_2
}{
\|\nabla f_1\times\nabla f_2\|
}.
$$

A step of size $ds$ predicts the next point along the curve,

$$
\mathbf r_{\mathrm{pred}}
=
\mathbf r_k+ds\,\mathbf t_k.
$$

The tangent orientation is guarded by requiring

$$
\mathbf t_k\cdot\mathbf t_{k-1}>0,
$$

which prevents numerical sign changes in the tangent from producing artificial $180^\circ$ reversals.

### Corrector step

The predicted point generally does not lie exactly on the singularity line. The corrector therefore solves

$$
\mathbf f(\mathbf r)=0
$$

locally.

Because there are two scalar constraints but three spatial coordinates, the spatial Jacobian has dimensions

$$
J\in\mathbb R^{2\times3}.
$$

The correction is therefore obtained using the minimum-norm solution of the linearized system,

$$
\Delta\mathbf r
=
-J^T(JJ^T)^{-1}\mathbf f.
$$

This is the Moore-Penrose pseudoinverse solution for the underdetermined correction problem. The predicted point is iteratively projected back onto the singularity curve.

### $L^T$-line tracing

For $L^T$-lines, the singular condition is

$$
\mathbf N=
(N_x,N_y,N_z)=0.
$$

Any two independent components can locally define the curve, but a poorly chosen pair can become nearly linearly dependent and make the correction ill-conditioned.

VectorWaves therefore dynamically selects the pair $(N_i,N_j)$ whose gradients have the largest cross-product magnitude,

$$
\left\|
\nabla N_i\times\nabla N_j
\right\|,
$$

providing a locally well-conditioned pair of constraints. This makes $L^T$-line tracing less sensitive to the orientation of the polarization ellipse or the choice of coordinate axes.
