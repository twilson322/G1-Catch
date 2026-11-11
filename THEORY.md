# Theory behind G1 Catch

This is the theory the project uses, one subject at a time, in the order the signal flows from photons to motor torque. Each section says what the idea is, gives the equations the code implements, and explains why that method was chosen over the obvious alternative. File names point to where each idea lives.

## 0. The problem sets the requirements

A ball thrown from 3 m at 6 m/s is in the air for about 0.5 s. A 60 Hz camera delivers roughly 30 frames in that time, each one 20 to 40 ms old when it arrives. The arm needs about 0.3 s to move 40 cm under its torque limits. So the robot must commit to a motion after about ten frames, when the estimate of where the ball will be is still several centimetres uncertain, and then correct the motion as later frames tighten it.

Every design decision below follows from that budget. The estimator must give an uncertainty along with a prediction. The planner must be cheap enough to rerun every frame. The controller must track a target that keeps moving.

## 1. Camera geometry and stereo vision (`g1catch/vision.py`)

A pinhole camera maps a point $(X, Y, Z)$ in the camera frame, with $Z$ along the optical axis, to a pixel

$$u = f\,\frac{X}{Z} + c_x, \qquad v = f\,\frac{Y}{Z} + c_y .$$

The focal length in pixels comes from the vertical field of view and the image height $H$ by $f = (H/2)/\tan(\mathrm{fovy}/2)$. Calibration is the job of finding $f$, $c_x$, $c_y$ and the pose of each camera. In simulation these are known exactly, and the tests check that the projection code agrees with the renderer to a fraction of a pixel.

Two cameras a baseline $b$ apart see the same point at horizontal pixel positions that differ by the disparity $d = u_L - u_R$. Depth follows from similar triangles.

$$Z = \frac{f\,b}{d}$$

Differentiating gives the result that shapes the whole project.

$$\sigma_Z = \frac{Z^2}{f\,b}\,\sigma_d$$

Depth error grows with the square of distance. With a short head-mounted baseline and a modest focal length, a quarter-pixel disparity error at 3 m is tens of centimetres of depth error, while lateral error at the same range is a few millimetres. At 1 m the depth error is nine times smaller. The ball's position is therefore known well sideways and badly in range early in the flight, and the range estimate improves fast as the ball approaches.

Triangulation turns two pixel observations into a 3D point. Each camera has a $3\times4$ projection matrix $P$, and each pixel gives two linear equations in the homogeneous point $\mathbf{X}$, namely $u\,(P_3\mathbf{X}) - P_1\mathbf{X} = 0$ and $v\,(P_3\mathbf{X}) - P_2\mathbf{X} = 0$ with $P_i$ the rows of $P$. Stacking four equations and taking the smallest singular vector gives the least-squares point. This is the direct linear transform.

The filter does not consume triangulated points. It consumes the four pixel coordinates directly. Pixel noise is close to Gaussian and the same size everywhere. The triangulated point's noise is stretched along the viewing ray by the $Z^2$ law and is not Gaussian. Feeding pixels to a filter that knows the projection equations lets the filter weight range and bearing correctly at every distance. Triangulation is used only to initialise the track.

## 2. Deep learning for detection (`g1catch/detector.py`)

The classical detector thresholds the image on the ball's colour and takes the centroid of the pixels that pass. It is fast and exact when lighting is fixed. It fails when the lighting changes the apparent colour, when another object has a similar colour, and when motion blur smears the ball into a streak whose centroid lags the true position.

The learned detector is a small fully convolutional network (a two-level U-Net with about 55 thousand weights). A convolution layer applies the same filter at every image location, so the network is translation equivariant. A ball learned in one corner is recognised in every corner, and a network trained on 96 pixel crops runs unchanged on the full frame. The network outputs a heatmap $h(u,v)$ with one value per pixel and is trained to produce a Gaussian bump centred on the ball.

Taking the argmax of the heatmap would limit accuracy to one pixel. Sub-pixel position comes from the centroid of the heatmap in a small window $W$ around the peak,

$$\hat u = \frac{\sum_{W} u\,h(u,v)}{\sum_{W} h(u,v)}, \qquad \hat v = \frac{\sum_{W} v\,h(u,v)}{\sum_{W} h(u,v)} .$$

The peak value serves as a confidence. A low peak means no ball is visible and the filter skips the update.

A detector trained on small yellow-green discs fires on any small yellow-green disc. So before a track exists the network reports up to three peaks per image, with non-maximum suppression between them, and the tracker tries every left-right pair. A wrong pairing reveals itself geometrically. Its two rays do not meet (large reprojection error after triangulation), or they meet somewhere a throw cannot come from. Once a track exists, the detector only searches a window around the predicted position, and the gate in section 4 rejects what is left.

Training data is synthetic. Domain randomization renders the ball under random lighting, colour shifts, backgrounds, blur and sensor noise, with the robot's own arm sometimes occluding it. The principle is that if the training distribution is wider than the test distribution, the test condition looks like one more sample from it. The label is free because the simulator knows where the ball is.

The detector's output is a measurement with noise. Its error distribution on held-out images sets the measurement covariance $R$ used by the filter in the next section. That link between a learned component and a model-based one is the main reason to measure the detector carefully.

## 3. Ball flight physics (`g1catch/ballistics.py`)

A ball in air feels gravity, drag and, if it spins, a Magnus force.

$$\dot{\mathbf v} = \mathbf g - k_d\,\lVert\mathbf v\rVert\,\mathbf v + k_m\,\boldsymbol\omega\times\mathbf v, \qquad k_d = \frac{\rho\,C_d\,A}{2m}$$

Here $\rho$ is air density, $C_d \approx 0.47$ for a smooth sphere, $A$ the cross-section and $m$ the mass. Drag matters more for light, large balls because $k_d$ scales with area over mass. For a 70 mm foam ball it shortens a 3 m throw by several centimetres, which is more than the catcher's radius. So a gravity-only model misses.

The filter's model includes gravity and a nominal drag term. The true drag coefficient and the spin are unknown and vary per throw. Section 5 asks a different question of the same physics. If the drag term had been left out, could a model learned from data put it back?

## 4. Stochastic estimation (`g1catch/ukf.py`, `g1catch/imm.py`)

### Bayes filtering

The state $\mathbf x = (\mathbf p, \mathbf v)$ evolves by $\mathbf x_{k+1} = f(\mathbf x_k) + \mathbf w_k$ and is observed through $\mathbf z_k = h(\mathbf x_k) + \mathbf n_k$ with $\mathbf w \sim \mathcal N(0,Q)$ and $\mathbf n \sim \mathcal N(0,R)$. A filter alternates two steps. Prediction pushes the belief through the dynamics and grows the covariance. Update compares the predicted measurement with the actual one and shrinks the covariance in the directions the measurement informs.

When $f$ and $h$ are linear the Kalman filter does this exactly. Here $h$ is the stereo projection, which divides by depth, and $f$ contains the quadratic drag term.

### Why an unscented filter

An extended Kalman filter linearises $h$ at the mean. Early in the flight the depth uncertainty is large, and $1/Z$ is far from linear over a 30 cm depth spread, so the linearisation is poor exactly when it matters. The unscented Kalman filter avoids derivatives. It picks $2n+1$ sigma points that carry the mean and covariance exactly,

$$\mathcal X_0 = \hat{\mathbf x}, \qquad \mathcal X_{\pm i} = \hat{\mathbf x} \pm \left(\sqrt{(n+\lambda)P}\right)_i ,$$

pushes each through the nonlinear function, and recovers the mean and covariance of the result as weighted sums. The update then uses

$$K = P_{xz}\,S^{-1}, \qquad \hat{\mathbf x} \leftarrow \hat{\mathbf x} + K(\mathbf z - \hat{\mathbf z}), \qquad P \leftarrow P - K S K^\top ,$$

where $S$ is the predicted measurement covariance (including $R$) and $P_{xz}$ the cross-covariance, both computed from the sigma points.

### Checking that the filter is honest

A filter can have small error and still be wrong about its own uncertainty, and the planner relies on that uncertainty. Two statistics test it. The normalised innovation squared $\nu^\top S^{-1}\nu$ should average the measurement dimension (4 here). The normalised estimation error squared $\mathbf e^\top P^{-1}\mathbf e$ should average the state dimension (6). Values far above mean the filter is overconfident. The same innovation statistic gates outliers. A measurement whose NIS exceeds a chi-squared threshold is rejected as a false detection.

### Interacting multiple models

One motion model cannot describe both free flight and the instant of a bounce. Free flight needs small process noise so the prediction is sharp. A bounce changes velocity abruptly, and a filter with small $Q$ takes many frames to believe it. The IMM runs one filter per mode and keeps a probability $\mu_j$ for each. Every step it

1. mixes the filters' estimates according to a Markov transition matrix $\Pi$ (the prior chance of switching modes),
2. runs each filter's predict and update,
3. reweights the modes by how well each explained the measurement, $\mu_j \propto \Lambda_j \sum_i \Pi_{ij}\mu_i$ with $\Lambda_j = \mathcal N(\nu_j; 0, S_j)$,
4. outputs the probability-weighted combination.

In flight the low-noise mode dominates. At a bounce its innovation spikes, its likelihood collapses, and the high-noise mode takes over for a few frames.

One detail matters in practice. An impact needs something to hit, so the transition matrix depends on the estimated height. Near the floor the chance of entering the impact mode is 10 % per frame. In open air it is 0.2 %. With a single matrix, a ball that leaves the cameras' view for a few frames has its covariance inflated every frame by an impact that cannot have happened.

## 5. Machine learning on top of the physics (`g1catch/residual.py`, `g1catch/catchability.py`)

### Gaussian process regression

Take gravity as the only known physics. The residual acceleration $\mathbf a_{res} = \dot{\mathbf v}_{true} - \mathbf g$ is then a smooth unknown function of velocity, and calibration throws give noisy samples of it. A Gaussian process places a prior over functions through a kernel $k(\mathbf x, \mathbf x')$ that says how similar outputs are for nearby inputs. Given training inputs $X$ and targets $\mathbf y$ with noise variance $\sigma_n^2$, the prediction at a new input $\mathbf x_*$ is Gaussian with

$$\mu_* = \mathbf k_*^\top (K + \sigma_n^2 I)^{-1}\mathbf y, \qquad \sigma_*^2 = k(\mathbf x_*, \mathbf x_*) - \mathbf k_*^\top (K + \sigma_n^2 I)^{-1}\mathbf k_* .$$

Two properties make it the right tool here. It works with a few hundred data points, and it reports its own uncertainty. Far from the training data $\sigma_*^2$ returns to the prior variance and the mean returns to zero, so the correction switches itself off where it has no evidence. Kernel length scales and noise are fitted by maximising the marginal likelihood of the data.

### Gradient-boosted trees

Boosting builds a sum of small regression trees. Each new tree is fitted to the negative gradient of the loss with respect to the current prediction, which for squared error is the current residual. Trees are piecewise constant, need no feature scaling and are fast to evaluate. They extrapolate as a constant and give no uncertainty. The project fits both models to the same data and compares them, because the comparison shows what the GP's smoothness and uncertainty are worth on this problem.

### Logistic regression for catchability

Some throws cannot be caught. They land outside the arm's reach or arrive too soon. Committing to them wastes motion and, on a real robot, risks a collision. A logistic model maps features of the early prediction (intercept distance from the shoulder, time to arrival, required hand speed, prediction covariance) to a probability

$$P(\text{catch}) = \sigma(\mathbf w^\top \boldsymbol\phi + b), \qquad \sigma(s) = \frac{1}{1+e^{-s}},$$

trained by minimising cross-entropy on logged outcomes. Logistic regression is chosen over a larger model because its output is a calibrated probability when the features are reasonable, the weights can be read, and a few hundred throws are enough to fit it. Calibration is checked with a reliability curve.

## 6. Rigid-body dynamics and nonlinear control (`g1catch/control.py`)

### Equations of motion and computed torque

The arm obeys

$$M(\mathbf q)\,\ddot{\mathbf q} + \mathbf h(\mathbf q, \dot{\mathbf q}) = \boldsymbol\tau ,$$

with $M$ the mass matrix and $\mathbf h$ the Coriolis, centrifugal and gravity torques. The dynamics are nonlinear and coupled. Choosing $\boldsymbol\tau = M\,\ddot{\mathbf q}^{*} + \mathbf h$ cancels the nonlinearity and leaves $\ddot{\mathbf q} = \ddot{\mathbf q}^{*}$, a set of independent double integrators. This is feedback linearization. Everything upstream (MPC, LQR) can then be designed for a linear plant. It is exact only if the model is exact, which is why the adaptive estimator in section 10 exists.

### Operational-space control

The task is defined at the hand. Its velocity is $\dot{\mathbf x} = J\dot{\mathbf q}$. The hand feels an effective inertia

$$\Lambda = (J M^{-1} J^\top)^{-1},$$

which is what the ball "hits" at impact. A task-space force $\mathbf F$ is produced by joint torques $J^\top\mathbf F$, so the controller applies

$$\boldsymbol\tau = J^\top \mathbf F + \mathbf h + N^\top \boldsymbol\tau_0, \qquad N = I - \bar J J, \quad \bar J = M^{-1}J^\top\Lambda .$$

The robot has ten joints and the task uses five (three for position, two to point the catcher along the incoming velocity). The remaining freedom is used by $\boldsymbol\tau_0$ to hold a comfortable posture. The projector $N^\top$ with the inertia-weighted pseudoinverse $\bar J$ guarantees the posture torque produces no acceleration at the hand.

### Impedance control at contact

Impedance control makes the hand behave like a spring and damper about a moving reference,

$$\mathbf F = \Lambda\,(\ddot{\mathbf x}_d - \dot J\dot{\mathbf q}) + K(\mathbf x_d - \mathbf x) + D(\dot{\mathbf x}_d - \dot{\mathbf x}),$$

with $K$ in N/m. The first term is feedforward that keeps the cup on its reference. Stiffness is low along the approach direction and high across it, so the cup stays on the ball's path. The approach direction is the reverse of the ball's velocity, tilted up if necessary so the cup never opens below 20° above horizontal. A cup aimed straight against a rising ball opens downward and cannot keep it. Damping is set per direction as $2\zeta\sqrt{k\,\lambda}$ from the task inertia $\lambda$ in that direction.

How much does compliance help a catch? Less than intuition suggests. A collision lasts about 5 ms. In that time the outcome is set by the two inertias and the restitution $e$ of the surfaces. The impulse is $(1+e)\,m_{red}\,\Delta v$ with reduced mass $m_{red} = m_b\,m_h/(m_b+m_h)$. The cup's effective mass along its axis is 2.7 kg and the ball's is 0.057 kg, so $m_{red}$ is the ball's mass to within 2 %, and no stiffness setting changes that. The ball leaves the base at $e\,\Delta v$ relative to the cup. What control can change is $\Delta v$. Moving the cup along the ball's velocity at contact lowers the relative speed, and choosing the cup speed $c = e\,v/(1+e)$ leaves the ball at rest in the world after the impact. The experiments confirm this ranking. Velocity matching decides whether the ball stays, and the impedance switch has no measurable effect.

## 7. Optimal control (`g1catch/control.py`, `g1catch/ilqr.py`, `g1catch/intercept.py`)

### LQR

For a linear system $\mathbf x_{k+1} = A\mathbf x_k + B\mathbf u_k$ and cost $\sum \mathbf x^\top Q\mathbf x + \mathbf u^\top R\mathbf u$, the optimal controller is $\mathbf u = -K\mathbf x$ with

$$K = (R + B^\top P B)^{-1} B^\top P A,$$

where $P$ solves the discrete algebraic Riccati equation. $Q$ and $R$ express the trade between error and effort. Computed torque turns every joint into the same double integrator, so one LQR design on that plant, discretised at 1 kHz, gives the position and velocity gains of the joint loop for all ten joints. Section 9 analyses that loop in the frequency domain.

### iLQR

For nonlinear dynamics and a finite horizon, iterative LQR repeats two passes. The backward pass linearises the dynamics and quadratises the cost around the current trajectory, then solves the resulting time-varying LQR problem by a Riccati recursion, giving a feedforward term $\mathbf k_t$ and a gain $K_t$ per step from

$$Q_{uu}\,\mathbf k = -Q_u, \qquad Q_{uu}\,K = -Q_{ux} .$$

The forward pass rolls out the real nonlinear dynamics with $\mathbf u_t + \alpha\mathbf k_t + K_t(\mathbf x_t - \bar{\mathbf x}_t)$ and a line search on $\alpha$. Regularisation on $Q_{uu}$ keeps the step well defined far from the optimum. iLQR uses the full arm model, so its trajectory respects torque limits that the double-integrator MPC only approximates with acceleration bounds. Each iteration linearises MuJoCo at every knot, which takes about 3 s per solve in this implementation. It is therefore used offline, to check how well the MPC's acceleration limits stand in for torque limits (`scripts/ilqr_check.py`). The torque bounds are handled inside the backward pass by clamping the controls that would leave the box and solving for the rest.

### Where and when to intercept

The predicted ball path crosses the arm's workspace for a short interval. Any point on that interval is a candidate. For each candidate time $t$ the planner needs to know whether the hand can get there. A rest-to-rest move of distance $d$ under an acceleration bound takes at least

$$t_{min} = 2\sqrt{d / a_{max}}$$

(accelerate for half, brake for half, the bang-bang solution of the minimum-time problem). Candidates with $t < t_{min}$ are penalised. Among the rest the planner prefers the least hand travel, then low forecast uncertainty, then distance from the workspace boundary. Each candidate is checked with inverse kinematics before it is accepted. Once chosen, the catch time is kept, and only the catch position is refreshed from each new forecast. Re-choosing the time every frame lets it creep along the ball's path.

### Inverse kinematics with priorities

The catch task has two parts of unequal importance. The cup centre must be on the ball's path (3 constraints). The cup axis should point along the incoming velocity (2 constraints). The solver iterates

$$\Delta\mathbf q = J_p^{+}\mathbf e_p + N_p\,(J_a N_p)^{+}(\mathbf e_a - J_a J_p^{+}\mathbf e_p), \qquad N_p = I - J_p^{+}J_p ,$$

so aiming uses only motion that leaves the position alone. A cup in the right place and 30° off still catches. A well-aimed cup 5 cm away does not.

## 8. Model predictive control (`g1catch/mpc.py`)

MPC solves a finite-horizon optimal control problem at every sample, applies the first input, and solves again with new information. With computed torque underneath, the plant is $\ddot{\mathbf q} = \mathbf u$ per joint, and the problem is a quadratic program,

$$\min_{\mathbf u_{0..N-1}} \; \sum_{k} \lVert \mathbf u_k\rVert_R^2 + \lVert \mathbf u_k - \mathbf u_{k-1}\rVert_S^2 \;+\; \lVert \mathbf x_N - \mathbf x_{goal}\rVert_{Q_f}^2$$

subject to the discrete double-integrator dynamics and box limits on position, velocity and acceleration. The horizon shrinks as the intercept time approaches, so the terminal state always sits at the predicted catch.

Re-planning matters here for a specific reason. The goal moves. Each camera frame shifts the predicted intercept point by an amount comparable to the prediction's standard deviation, which starts near 10 cm and ends under 1 cm. A fixed plan made at commit time would arrive at the wrong place. MPC folds each correction into the remaining motion at minimum effort and never violates a limit doing so.

OSQP solves the QP with the alternating direction method of multipliers. The problem has 140 variables and 420 inequality rows at the longest horizon.

## 9. Linear analysis, observers and robustness of the joint loop (`g1catch/joints.py`, `scripts/margins.py`)

### Margins

Simulation shows that a controller works on the cases you ran. Margins say how far it is from not working. For the loop transfer function $L(j\omega)$ of plant times controller,

- the gain margin is how much the loop gain can change before the loop is unstable, read where the phase is $-180°$,
- the phase margin is how much extra phase lag the loop tolerates at the crossover frequency $\omega_c$ where $\lvert L\rvert = 1$,
- the sensitivity peak $M_s = \max\lvert 1/(1+L)\rvert$ is the reciprocal of the closest approach of $L$ to the point $-1$, one number that bounds both.

A pure delay $T_d$ has unit gain and phase $-\omega T_d$. So the delay margin is $\mathrm{PM}/\omega_c$ (with PM in radians), and it bounds how much sensing and actuation latency the torque loop survives.

Gain margin is also robustness to a wrong inertia. Computed torque commands $\boldsymbol\tau = \hat M \mathbf u$. If the true inertia is $M$, the joint accelerates at $M^{-1}\hat M\,\mathbf u$, so for one joint the loop gain is multiplied by $\hat m/m$. A gain margin of 9 means the model's inertia can be up to 9 times the true one before the loop goes unstable. This is why link masses wrong by 50 % changed nothing in the experiments.

### Observers and the separation principle

A real joint reports a quantized angle and no velocity. After computed torque each joint is $\ddot q = u + d$, where $d$ lumps whatever the model got wrong. An observer runs a copy of the model and corrects it with the measurement,

$$\hat{\mathbf x}_{k+1} = A\hat{\mathbf x}_k + B u_k + L\,(y_k - C\hat{\mathbf x}_k).$$

The Kalman filter chooses $L$ to minimise the estimate's variance for given process and measurement noise, and at steady state it comes from a Riccati equation that is the dual of the LQR one. Quantization to a step $\Delta$ behaves like noise of variance $\Delta^2/12$.

LQR gains on a Kalman estimate is LQG. The separation principle says the two can be designed independently and the closed loop is stable. It does not say the margins survive. LQR by state feedback has guaranteed margins (at least 60° of phase in continuous time). LQG has none. Here the phase margin falls from 64° to 56° and the gain margin from 17 to 9 when the observer goes in. The numbers are computed from the loop transfer with the observer inside the loop, and checked in simulation.

### Disturbance observers

Add the disturbance to the observer's state, $\mathbf x = (q, \dot q, d)$ with $\dot d = 0$ plus noise, and the same Kalman machinery estimates it. Subtracting $\hat d$ from the command cancels a constant model error exactly and a slow one approximately. This is an extended-state observer, the core of active disturbance rejection control, and it is equivalent to adding integral action through the observer. Integral action costs phase: another 5° here, and the loop becomes conditionally stable, meaning it is unstable if the gain falls too far as well as if it rises.

The design knob is how fast $d$ is allowed to change. Fast means quick rejection and a noisy estimate, because the encoder's quantization is amplified through two derivatives. With 15-bit encoders, a 28 ms response costs 0.7 rad/s² of noise on $\hat d$.

### Sliding mode

Sliding-mode control picks a surface in the error space, $s = \dot e + \lambda e$, on which the error decays on its own, and pushes the state toward it with a term that depends only on the sign of $s$,

$$u = u_{nominal} + K\,\mathrm{sat}(s/\phi).$$

If $K$ exceeds the largest disturbance, $s$ reaches zero in finite time whatever the disturbance is, which is a robustness guarantee that needs no model of it. A pure sign function chatters at the sampling rate, so it is replaced by a saturation with boundary layer $\phi$, inside which the term is a high-gain proportional one.

Both robust terms were measured against plain LQG under link-mass error. Both halve the tracking error at a soft loop. Neither changes the catch rate, because tracking error was not what limited it.

## 10. Adaptive control (`g1catch/adaptive.py`)

After the catch the hand carries an unknown mass. Gravity compensation is then wrong by $J_v^\top m\,g$, where $J_v$ is the linear Jacobian at the cup, and a compliant hold sags by $m g / k$. A wrist force sensor reads $\mathbf f = (m_{cup} + m)(\mathbf a - \mathbf g)$ in the world frame, with $\mathbf a$ the cup's acceleration. The cup's mass is known, so

$$\underbrace{\mathbf f - m_{cup}(\mathbf a - \mathbf g)}_{\mathbf y} = \underbrace{(\mathbf a - \mathbf g)}_{\boldsymbol\varphi}\; m$$

is linear in the one unknown, and recursive least squares identifies it online.

$$K_k = \frac{P_{k-1}\boldsymbol\varphi_k}{\lambda + \boldsymbol\varphi_k^\top P_{k-1}\boldsymbol\varphi_k}, \qquad \hat m_k = \hat m_{k-1} + K_k^\top(\mathbf y_k - \boldsymbol\varphi_k \hat m_{k-1}), \qquad P_k = \frac{(1 - K_k^\top\boldsymbol\varphi_k)P_{k-1}}{\lambda}$$

The forgetting factor $\lambda < 1$ discounts old data so the estimate can follow a change. The estimate goes into the arm model, where it corrects both the mass matrix and the gravity torque. That closes the adaptive loop (an indirect, or self-tuning, adaptive controller). RLS only converges when the regressor is exciting. Here gravity is always present in $\boldsymbol\varphi$, so the condition holds even at rest.

Joint torques could in principle replace the force sensor. In this model they cannot. Each joint has 0.3 N m of Coulomb friction, and a 57 g ball at a 0.3 m lever arm adds 0.17 N m.

## 11. Digital and multirate control (`g1catch/runtime.py`)

Three clocks run at once. Cameras at 60 Hz, the estimator and planner at camera rate, and the torque loop at 1 kHz. Computation takes time too. A plan made from a frame takes effect 10 ms after the frame arrives, so the planner starts from where the reference will be then, and plans wait in a queue until their start time. The queue matters as soon as the delay exceeds one frame period: without it each new plan replaces the previous one before it has taken effect. Three effects of sampling drive the design.

A zero-order hold delays the signal by half a sample on average. At 1 kHz that is 0.5 ms and costs little phase at a 10 Hz crossover. At 60 Hz it would be 8 ms, which is why the planner outputs a trajectory for the fast loop to interpolate. The fast loop never holds a 60 Hz command constant.

Measurements arrive late. A frame captured at time $t_c$ reaches the filter at $t_c + T_{lat}$. The filter processes it as a measurement at $t_c$, then predicts forward to the present. The planner works with the present-time state. Ignoring latency would place the ball $v\,T_{lat}$ behind its true position, which is 18 cm at 6 m/s and 30 ms.

The intercept is specified in absolute time. All three loops share one clock, and the trajectory handed to the torque loop is indexed by that clock, so a late planner cycle shortens the remaining horizon instead of shifting the catch.

## 12. Learning a controller (`g1catch/policy.py`, `g1catch/ppo.py`)

### Imitation learning and DAgger

A network can be trained to copy the MPC by supervised learning on (state, MPC action) pairs. The MPC's cost and constraints separate by joint, so the network is small. It sees one joint's error, velocity, goal velocity, time to go and distance to its limits, returns that joint's next acceleration, and is applied to all ten joints. Plain behaviour cloning fails in closed loop because of distribution shift. Small errors take the policy to states the expert never visited, where it has no training data, and errors compound. DAgger fixes the data distribution. It runs the learner in the loop, records the states the learner visits, labels those states with what the expert would have done, adds them to the dataset and retrains. After a few rounds the training states match the states the policy really sees.

The distilled policy plans a whole reach in about a millisecond with no solver, at the price of losing hard constraint guarantees. Section 13 gives some of them back. The experiments measure both sides of that trade.

### Reinforcement learning with PPO

Policy gradient methods improve a stochastic policy $\pi_\theta$ by raising the probability of actions that did better than expected. PPO limits each update with a clipped objective,

$$L(\theta) = \mathbb E\left[\min\big(r_t A_t,\; \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\right], \qquad r_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)},$$

where the advantage $A_t$ is the return minus a learned value baseline. Clipping keeps the new policy close to the one that collected the data.

RL is used here as a residual. The model-based pipeline chooses the intercept, and the learned policy adds a small bounded correction to its timing and position. A zero residual reproduces the baseline, so learning starts from a working system and can only be credited for what it adds. Each throw is one decision, so the problem is a contextual bandit and needs thousands of throws, not millions of steps.

## 13. Safety filters and control barrier functions (`g1catch/safety.py`)

A learned policy has no constraints unless something gives it some. A safety filter sits between the policy and the plant, passes commands that are safe, and minimally changes the ones that are not.

The definition of safe is a barrier function $h(\mathbf x) \ge 0$ whose zero level set the system must not cross. For a joint with limit $q_{max}$ that can brake at $a_b$, position alone is the wrong barrier, because a joint at speed just inside the limit cannot stop. The right one includes the braking distance,

$$h(q, \dot q) = q_{max} - q - \frac{\max(\dot q, 0)^2}{2 a_b} \ge 0 .$$

A state with $h \ge 0$ can always be kept there by braking, so the set is forward invariant under the acceleration limit. The filter asks which accelerations keep $h \ge 0$ one step ahead. With next velocity $w$ the condition is

$$\frac{\Delta t\,(\dot q + w)}{2} + \frac{w^2}{2 a_b} \le q_{max} - q ,$$

a quadratic in $w$ with a closed-form largest root, which gives the largest admissible acceleration. The filter clips the policy's command to that bound and to its mirror image for the lower limit. In general a control barrier function filter is a small QP per step. For a double integrator with box limits it reduces to this clip.

Behind the distilled network, the filter takes plans that leave the joint range from 51 % to 5 %, the level of the MPC it was distilled from, and costs 0.3 ms.

## What each method is for, in one table

| Subject | Method | Job in the pipeline |
|---|---|---|
| Computer vision | Pinhole model, stereo triangulation | Pixels to 3D, track initialisation |
| Deep learning | Convolutional heatmap detector, windowed centroid | Ball centre in each image under varied lighting and blur |
| Stochastic estimation | UKF, IMM, NIS and NEES tests | Ball state and its uncertainty, bounce handling, outlier rejection |
| Machine learning | GP, boosted trees, logistic regression | Unmodelled drag and spin, catchability decision |
| Optimal control | LQR, iLQR, minimum-time bound | Joint-loop gains, offline torque-feasibility check, intercept selection |
| Model predictive control | Shrinking-horizon QP | Re-plan the reach every frame within limits |
| Nonlinear control | Computed torque, operational-space control, impedance | Linearise the arm, control the cup around the catch |
| Linear control | Loop margins, sensitivity peak, delay margin | How much latency and inertia error the torque loop tolerates |
| Observers | Steady-state Kalman filter, LQG | Joint velocity from a quantized encoder |
| Robust control | Disturbance observer, sliding mode | Tracking under model error |
| Safety | Control barrier function filter | Joint limits behind a learned policy |
| Adaptive control | RLS with forgetting | Caught mass, fed back into the model |
| Digital control | ZOH, multirate scheduling, latency compensation | Three rates on one clock |
| Imitation and reinforcement learning | DAgger, PPO residual | Solver-free policy, learned correction to the intercept |