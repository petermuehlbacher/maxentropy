# Max-entropy distributions

## Usage examples
Examples can be found in [this Colab notebook](https://colab.research.google.com/drive/1tf-Ywh5HZIXxKhXCTkhveJRV-n_9YoXm?usp=sharing).

## About the implementation
Amongst those distributions that satisfy some given constraints on moments and/or other expectations, we want to find the distribution that maximises entropy. (If you are not sure why, check out [Wikipedia](https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution) or [Brian Keng](https://bjlkeng.github.io/posts/maximum-entropy-distributions/)'s introduction.)

That is, for (not necessarily finite \(a<b\)) we want to maximise the entropy
$$S(̱̱p)=-\int_a^b p(x)\ln(p(x))dx,$$
subject to $\int_a^b p(x)dx=1$ and $\int_a^b f_i(x)p(x)dx=c_i$ for some finite number of constraints $(f_i,c_i)$.
This is accomplished by pretending that $dx>0$ is some very small number, approximating the integral with a finite sum, ending up with only a finite number of variables $p(x)$, and solving the resulting problem with Lagrange multipliers, i.e.:

* We have to find $(p,\lambda)$ (a tuple of real vectors) such that the gradient of the Lagrangian $$\mathcal L(p,\lambda)=S(p)-\lambda_0\left(1-\int_a^bp(x)dx\right)-\sum_i \lambda_i\left(c_i-\int_a^b f_i(x)p(x)dx\right)$$
vanishes. 
  * (The one condition that's always here is that $\int_a^bp(x)dx=1$, giving the multiplier corresponding to $\lambda_0$.)
* Taking partial derivatives w.r.t. $p(y)$ (for any $y\in[a,b]$) and setting them to $0$ formally gives $$\ln(p(y))=-1-\lambda_0-\sum_i \lambda_if_i(y).$$
* Hence we may simply assume that $$p(y)={1\over Z_\lambda}e^{-\sum_i\lambda_if_i(y)},$$ with $Z_\lambda$ such that $p$ induces a probability measure, i.e. $$Z_\lambda=\int_a^b e^{-\sum_i\lambda_if_i(x)}dx.$$
* It remains to make the gradient with respect to $\lambda$ equal to $0$; note that the partial derivatives are given by $${\partial\over\partial\lambda_i}\mathcal L(p,\lambda)=c_i-\int_a^bf_i(x)p(x)dx,$$
leaving us with the requirement that $\lambda$ must satisfy the system of equations
$$c_i={1\over Z_\lambda}\int_a^b f_i(x)e^{-\sum_i\lambda_if_i(x)}dx,$$
which are solved with SciPy's `fsolve` here.
Given such a $\lambda$, we have found the max-entropy distribution.

## To do
Roughly in order of importance (i.e. my personal interest):
* Implement a _Minimal Cross-Entropy Distribution_ class,
  * see e.g. [The Maximum Entropy Distribution of an Asset Inferred from Option Prices](https://www.jstor.org/stable/2331391).
* Find & fix some (presumably) numerical issues arising with the log-normal distribution.
* Allow for inequality constraints, i.e. not only $\int f_i(x)p(x)dx=c_i$, but also $\int f_i(x)p(x)dx\leq c_i$.
* Use [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) to speed up the minimisation process.
* Try to integrate this with Metaculus/Guesstimate/Elicit/…

## Further reading
* https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution
* https://bjlkeng.github.io/posts/maximum-entropy-distributions/ : A great exposition for those with no statistical mechanics background.
* https://fragile-credences.github.io/quantiles/ (app: https://makedistribution.com/). Lists alternatives to this approach, mostly by prescribing quantiles and finding the closest (in some sense) member of a given family of distributions such as normal/log-normal/metalog to the given requirements.

