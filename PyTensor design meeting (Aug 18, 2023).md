## Runtime broadcasting in Alloc and non-gradient inputs of SetSubtensor / RandomVariables:
https://github.com/pymc-devs/pytensor/pull/390 (Adrian will check again before merge)

## Other small PRS
https://github.com/pymc-devs/pytensor/pull/389
https://github.com/pymc-devs/pytensor/pull/406


## Blockwise --Needs review!!!
1. https://github.com/pymc-devs/pytensor/pull/306 (NEEDS REVIEW)
2. Should our rewrites support both forms, or just Blockwise?
3. My approach was to support only Blockwise, and have a late rewrite that removes useless Blockwise (0 batch dims)
4. Many blockwise even with batch dims are useless in some backends (almost everything I looked at was naturally batched in JAX, and in numpy linalg, but not scipy linalg). What about Numba?
5. Easy to include a simple rewrite that removes backend-specific uselesses.
8. We do we want to do a C-implementation
10. Who wants to write a Numba impl?
11. Who wants to write a JAX impl?
12. Do we want to fuse BatchedOps on Numba backend?


### Vmap construct
1. We already do something like this for the gradient of Blockwise, where we start from a "core" gradient and vectorize it via  a dispatch.
2. Dispatch fallsback to Blockwise for Ops, but there are special cases like Elemwise/CAReduce/Dimshuffle/RandomVariables that are "natively" batcheable with little-no logic
3. This doesn't have all the bells and whistles of JAX vmap (axes and stuff) but I feel those are not really important? Batching everything to the left sounds easy enough and covers many cases. **Objections?**
4. Should be easy to support in_axes and out_axes, the dispatch functions become a bit more complex, you have to transpose inputs and outputs sometimes. Not sure it's worth it, Adrian thinks it's neat.
5. Can be done in a follow up PR to Blockwise, even if we change the signature of the dispatch function. I (Ricardo) will take the anger for breaking the "public API".


https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#batching

## Link to old pytensor dims support design doc

https://hackmd.io/O6skfEDuQbua8y0H58SkLQ


## Make shared variables context specific

This is more a rough idea than anything concrete. Right now shared variables are just global variables, but for several reansons it would be nice if there was a controlled way to have different values for each shared variable. For instance when we want to run something in parallel, or if functions in different backends could have separate contexts. An API for this could maybe look something like this:

```python!

# A shared variable context is pretty much
# just a glorified dict, that contains the
# values for different shared variables
class SharedVarContext:
    _values: Dict[SharedVar, Any]

# There is a global default context
default_ctx = pt.default_var_contexts

# This would add an entry in the default context
shared1 = pt.shared(somevalue)

# But we can create a new context if we want
ctx = pt.SharedVarContext(backend="JAX")
# This would not change the value in the
# default context, but only in `ctx`
shared1.set(new_value, ctx=ctx)

shared1.get(ctx=ctx)  # return new_value

shared1.get()  # still returns the old value

# Similarly calls to compiled functions could
# get a ctx argument, and updates are applied
# to the variable copy in that context.
```


## Updates on previous conversations
1. Gradient optimizations
    1. We experimented with running canonicalize/stabilizy in PyMC logp graph because taking the grad.
    2. Still considering the idea of a lazy dummy Grad Op. I think reasoning in terms of gradient operations could be very interesting. 
    3. Still think it achieves the same thing as `value_and_grad` optimizations easily with a very simple kind of PyTensor rewrite. 

2. More ergonomic scan
    1. Stale PR: https://github.com/pymc-devs/pytensor/pull/191
    2. Goal: Make it easier to manipulate / rewrite Scans. It's an incredibly complex Op at the moment.
    3. Idea: Simplify by:
        3. Not representing input/output storage early on
        4. Two output pairs: last state and traces (most optimization care about one of those)
        5. No magic shared updates for non tensor types. RNG and other funky types can be traced via TypedList, or the last state retrieved just like with other Ops
        6. C peculiarities (storage alloc, inplace, shared RNGs) can be introduced during compilation
    7. Numba: Allocate output of while scans dynamically? We are not benefitting from storage persistance anyway AFAICT?

3. True IfElse for non VM backends (ie JAX and Numba)
    1. To be lazy on those backends we need to wrap the minimum-independent graph of each branch in a JITTED function (like an OpFromGraph)
    2. We could change the API to use OpFromGraph from the get go (or any inner compiled function, like Scan does)
    4. Current API does facilitate rewrites...

4. Gradients cleanup
    1. Still don't know if the double Lop vs Rop thing is true. 
    2. Should still remove Lop vs Grad distinction
    3. Consider other names 



***

## For future meetings

### Type compatibility across backends:
Relevant JAX PR on Sparse / RNGs inputs: https://github.com/pymc-devs/pytensor/pull/278 (Mixed reviews so far)

### Type compatibilty across backends for Shared variables:
AFAICT can't be (easily) solved by the multiple container idea. We could implement a subclass of list that syncronizes on right, but I am afraid this would break it on the C-backend.
Solution: Copy shared variables that have incompatabile types and tell users how they can be retrieved from the compiled function
Problem: I couldn'n figure out how to do this.


### Rewrites, ordering and eggs
1. Question of rewrite ordering and worries about duplicating costly operations arise almost in every rewrite PR.
2. For instance we could replace Switch(cond, a, b) -> (empty(), set_subtensor(cond, a[cond]), set_subtensor(~cond, b[~cond])) after broadcasting everything. Indexing operations can then be lifted closer to the inputs, making the switch in fact "lazy". But we don't know when is this useful if we can't know how much lifting can be done (as it might otherwise break Elemwise fusion)
3. Eggs and some meta-optimizer sound like the right solution for this. Is it? Can it actually reason about e.g., different permutations of index lifting and fusion rewrites?
4. Do we want to consider it seriously?
5. What are the biggest obstacles? 
    1. Complexly parametrized Ops jump to mind (Good luck representing an SetSubtensor symbolically in any useful way)
    2. Do we need immutable graphs?
7. Worth doing a POC and if it looks promising trying to get some GSOC / Numfocus project for it?


