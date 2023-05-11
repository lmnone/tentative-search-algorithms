# Fixed point prune-and-search & tentative search algorithms to solve simple geometry tasks

#### Definitions

Given n continuous(or piecewise) monotonic functions, defined on the interval [0,1] and takes values also in [0,1]. Let's start with n = 2:

$$
y = f(x), x \subseteq [0,1]
;
x = g(y), y \subseteq [0,1]
$$

#### Additional condition 1

Functions are set only “speculatively”. Those, tabulation and interpolation to obtain an arbitrary $y=f(x)$ is not expected. This condition is due to the fact that an algorithm based on interpolation and tabulation will require guaranteed >N steps to calculate. Where N is the number of discretization points.

Functions can be defined as space elements:

$$
\Im^{\Lambda }(x,y) \overset{def}{=} \Lambda(x,y,\lambda)
$$

where $\lambda$ is some logical expression on the elements

