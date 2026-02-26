# rust_micrograd
A micrograd engine insipired by Karpathy's micrograd, implemented in rust.

## Forward vs Reverse Automatic Differentiation

I wasn't too familiar with the importance of direction on the actual automatic differentiation (AD) engine until now, and the topic of forward vs reverse differentiation is quite an interesting one.  I am a big fan of [this](https://math.stackexchange.com/a/3119199) explanation if you are also not very familiar.  

The implementation that I have here will focus on forward AD, making note that the most efficient approach, specifically for the task of integration with any sort of neural network architecture, would be reverse AD which allows less operations to be performed on higher order inputs.
