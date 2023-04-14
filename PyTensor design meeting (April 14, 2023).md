# PyTensor design meeting (April 14, 2022)

## Backend specific types (RNGs and SparseTensors)
### The problem
 1. JAX does not accept Numpy Generators or Scipy sparse matrices
 1. Current hacky-work around for shared Generators:
     1. Replace shared RNGs by a copy, then convert the value to a jax PRNG
     1. Users can't access this RNG easily, and their original one is not connected to the compiled graph ([code](https://github.com/ricardoV94/pytensor/blob/19e1a982e5a2e9ba2bd9d708133abf102c6fb835/pytensor/link/jax/linker.py#L17-L50) - [sections](https://github.com/ricardoV94/pytensor/blob/19e1a982e5a2e9ba2bd9d708133abf102c6fb835/pytensor/link/jax/linker.py#L66-L79)).
     1. Probably fine, most users don't even know about RNGs
     1. This works for shared variables that we can assume the user doesn't want to manually update... but it doesn't work for other types nor non-shared inputs!!!
 1. This is the case with SparseTensors. The user could create a graph with Constant SparseTensor, Shared SparseTensors or Input SparseTensors. 
     1. For JAX in https://github.com/pymc-devs/pytensor/pull/273, we converted constant SparseTensors to BCOO, the format that JAX better supports
     1. We couldn't do this for Input (we can't for RNGs either) or Shared Sparse Tensors (unlike RNGs, users definitely care about controlling the values/updates of these).


### Half-assed solutions

1. Create PRNG and BCOO PyTensor variables that the users would use when creating their own graphs. These would only work for the JAX backend, and not for a Numba or C or Python backend. 
1. Use PRNG and BCOO as the common type for all backends (or a supertype that somehow works with all). We would be hurting the other backends in favor of JAX!

### Possible solution
1. Consider our input types to be more generic when we compile a function. This is in fact what we are doing in our hacky way for RNG and constant sp_matrix. The problem is that the `node` no longer provides exact type information about the inputs and outputs (if you are running in C, CSC is a sp_matrix, but in JAX it is a BCOO).


1. Replace the graphs using unsupported types with specialized, but exact types.
    *  When we have something like `structured_dot(csc, csc) -> csc` we would replace it by `structured_dot(bcoo_from_csc(csc), bcoo_from_csc(csc)) -> csc_from_bcoo(bcoo)`.
    * Note that we still need a `csc_from_bcoo` if the output was sparse again. The downstream nodes wouldn't know the input type has changed. This is not too bad, we would next replace any downstream node like this:
    * `another_sparse_op(csc_from_bcoo(bcoo)) -> dense`  by `another_sparse_op(bcoo) -> dense`.
    * For root inputs we will end up with a `bcoo_from_csc(csc(None))` which **must not be part of the JITTED function**. So should it be part of the `PyTensor Type`? 
    * Can we replace root inputs `bcoo_from_csc(csc(None)) -> bcoo(None)` and final function outputs `csc_from_bcoo(bcoo) -> bcoo`

1. The difference between 1. and 2. is that we have exact type information in the final graph. If a user calls dprint on the compiled function they will see the right types, and the dispatched functions can also trust them when deciding about their impl.

1. Either way JAX will still fail if the user passes a sp_matrix or numpy Generator. 
    * **We would need to tell users that we expect bcoo as root inputs and no longer csc! even though they created the graph with those types**.
    * Option: When we convert the user provided variable to the new input type, we raise an informative warning.
    ```python
    class BCOOType:

        def filter_variable(self, value, warn_backend=False):
            # This method is what is disabled when you set function.trust_input = True
            if isinstance(value, self):
                return value

            if isinstance(value, sp_matrix):
                value = BCOO.from_scipy_sparse_matrix(sp_matrix)

            elif isinstance(value, (np.ndarray, jax.Array):
                value = BCOO.from_dense(value)

            if warn_backend:
                warns.warn(
                    f"Input type {type(value)} had to be converted to {type(self)}" 
                    f"This can be slow. Make sure to pass {type(self)} variables next time".
                    "Set `backend_conversion_warning='ignore'` to avoid this warning."
                )
            return value
    ```

2. We still didn't solve shared variables! These can in theory be used among multiple compiled functions, including different backends. 
    * That's why in JAX we first replace the shared RNGs by new ones, and only alter those. Otherwise if a user tried to compile a random function with C backend it would fail because the underlying data is no longer a numpy Generator!
    * Idea: Could our shared have multiple "containers" for each backend? So a JAX compiled function would not simply read/write from the "global" container but the "JAX container". 
    * Could work, we would just need to make sure we synchronize the "global" and "specialized" containers, and we don't want to do this all the time, only when we are requesting a value from an "outdated container"
    ```python
    class SharedRNGVariable:
        default_backend = GeneratorType

        def __init__(self, value):
             self.container = {type(value): value}
             self.last_type_access = type(value)

        def get_value(self, type_=self.default_backend, warn_backend=True):
            if not isinstance(type_, last_type_access):
                # Do costly conversion of types among containers
                last_updated_value = self.backend[self.last_type_access]
                    value = type.filter(last_updated_value, warn_backend=warn_backend)
                    self.backend[type_] = value
                    # Next call will probably be from the same backend!
                    self.last_type_access = type_
            return self.backend[type_]

        def set_value(self, value):
            self.backend_container[type(value)] = value
            self.last_type_access = type(value)
     ```
     
## Gradient optimizations

### Problem
1. Optimization / stabilization rewrites can be more easily done on the "graph" that is being differentatioted than on the differentiated graph (sometimes you can only do them if you know what was the original graph)
1. Users are penalized for not using specialized Ops (i.e., `log1mexp` vs `log(1 - exp(x)`) for the gradients. We have rewrites that replace the latter by the former, but not for the gradients. Again it could be hard or not possible to do it on the differentiated graph
1. We probably could get rid of many Ops, whose only purpose is to return a more efficient stable gradients:
    1. Softmax and LogSoftmax and SoftMaxGrad aren't needed! 
    1. I think (didn't have time to check) that this is the difference between `sparse.dot` and `sparse._structured_dot`
1. Easier to replace Ops by `value_and_grad` specialized Ops

### Solution
1. Make gradients lazy (i.e., a dummy Op) that is unrolled at a specific time in the compilation phase, after we had the opportunity to optimize the original graph
2. This would require making more rewrite passes, but maybe we can also have less rewrites if we are not worried about identifying what the gradients of what we are seeing?
3. This applies to Scan as well. We have rewrites that can easily optimize/remove "vanilla" scans but not the gradients of such scans. What a shame! :)
4. Some pseudo-code written by Brandon: https://github.com/aesara-devs/aesara/issues/1275

### Intermediate solution
1. Do this at the PyMC level: some rewrites on the logp graph, before we call the dlogp. This is already useful work for PyTensor in that we figure out what rewrites we want!

## Scan
1. Current design is strongly influenced by the gradient graph
    1. We pass as input `set_subtensor(empty(lags + n_steps, *x0.shape[1:])[:lags], x0)`, and scan fills in the empty slots. This is so that the output can be reversed for the gradient without requiring an explicit concatenation of the output and the initial state.
    1. Problems:
        - For while scan it doesn't make sense to preallocate the output buffers, as the n_steps are usually the worst case scenario
        - Graph manipulation / rewrites are cumbersome. A lot of manipulation is needed just to figure out what is x0 
        - Also cumbersome to just figure out we are seing the output of a scan, since what is returned to the user is `trace[lags:]` (i.e, the trace without the x0)!
        - See how hard, a straightforward optimization was for while Scans https://github.com/pymc-devs/pytensor/pull/216

    1. Alternative design:
        - Simplify user-facing Scan and then specialize to fancy internal form we have now
        - See https://github.com/pymc-devs/pytensor/pull/191 approach:
        - Scan has two sets of outputs: the last state and the whole trace.
        - Most of the times we are interested in either of these, and only rarely on partical traces, so I think it's fine if we loose some of the more fine-grained optimization when say a user requests trace[-5:] or whatever. 
        - Specially if we can simplify this monster rewrite: https://github.com/pymc-devs/pytensor/blob/fda240fd4355ecb6bc28fac50daaba2ede8fc4ce/pytensor/scan/rewriting.py#L1183-L1759
        - Last state optimization is trivial: just swap the view from the trace output to the respective last state output

    1. Alternative design:
        -  https://github.com/pymc-devs/pytensor/discussions/189

1. Rewrites are not applied internally
    1. We have some rewrites that completely remove a Scan if it's just doing Elemwise operation. 
    1. However, if you have two nested Scans doing Elemwise operations over a matrix, the inner Scan is never optimized in time for the outer one to also be removed.
    1. The argument here is that optimizing the inner scan could be wasteful, if the outer scan is not even needed in the end (e.g., because we only need it's shape). Still... this should be possible to sort out

1. Scan is deemed slow, and so we try to use it as little as possible
    1. Many rewrites try to remove elemwise computations from the header or footer of the scan inner graph, arguing that "elemwise" operations on small tensors are slower than on large tensors... (https://pytensor.readthedocs.io/en/latest/extending/scan.html#rewrites)
    1. This is the case for the C-backend, because the Scan inner function is a black-box from the outside. 
    1. Is it also true for Numba/ JAX? I suspect it is for JAX, because they state that the inner function is jitted automatically.
    1. Otherwise, the Fusion rewrites are just doing the same at a lower level of abstraction (elemwise is just a loop obviously...)

1. Scan don't allow scalars in the inner graph.
    1. Some inneficiences caused by this (see https://github.com/pymc-devs/pytensor/issues/233) 
    1.  This tounches on the broader issue that our rewrites are written for Tensors only, even when many graphs are composed of scalar Tensors (https://github.com/pymc-devs/pytensor/pull/107)
    1. We should just allow it...

1. Scalar Scan
    1. Many of Scipy math functions (betainc, gammainc) gradients require iterative algorithms for efficient approximation
    2. We can't simply return a Scan as the gradient graph
    3. Solution 1: Implement Scalar Scan Op (a lot of boiler-plate code needed, and it doesn't even seem that efficient)
        * Started exploring in https://github.com/ricardoV94/pytensor/tree/scalar_scan_pure
    5. Solution 2: Use a dummy Op and later unroll as a fully fledged Scan
        * Approach taken in https://github.com/pymc-devs/pytensor/pull/174


## RandomVariable updates

1. Read: https://gist.github.com/ricardoV94/7aa6c23726f20fbec95a1cda449c15f5
1. Problem of `default_update` - graph rewrites
1. Scan limitation workaround hacks - Use TypedList?
1. FunctionGraph doesn't have an updates functionality but I don't think this should be needed. I dont' think the Scan updates are even needed. It's a limitation of ^ I think.
1. Aesara discussions: https://github.com/aesara-devs/aesara/discussions/1479


## Broadcasting
1. Still in a limbo, have to fix it!
1. https://github.com/pymc-devs/pytensor/discussions/100
1. Max started here https://github.com/pymc-devs/pytensor/pull/149, we need someone to take the next shot!