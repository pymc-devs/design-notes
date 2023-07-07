
## MLIR backend
JAX / Torch could be converted to MLIR, and nutpie could then sample graphs defined on those packages

## Numba
1. JAX seems to be winning by a far margin in our use-base and even developers.
2. Bill has been thinking about writing C/C++ implementations to be used in JAX!!! Which seems like a full-circle but this time more constrained than our C backend.
3. Numba is in theory much more flexible and a better default replacement for the C-backend. But:
1. Compilation times suck?
2. NoPython Op coverage is still lacking?

1. Function signatures could improve compilation time

# Inplace optimizations when and where do we want them?
1. The Split Op was changed to a `view` Op recently. Numba was not respecting this flag and didn't explicitly copy the outputs.
2. However, do we want this to always be the case? If something has to allocate an array downstream anyway, and could otherwise operate in-place (e.g., Elemwise) then it would be probably better if the Split already allocated new arrays? https://hackmd.io/ZjrB5liiTburJiVkudkL_A?edit
3. BroadcastTo was implement explicitly to avoid re-allocating new arrays. Otherwise it behaves exactly as Alloc does. I don't think we need 2 Ops (3 if you count Second, see https://github.com/pymc-devs/pytensor/issues/367) for this. We could just change the `view_flag` and merge the C implementations. This again raises the question of when do we want one vs the other!.
4. Are we exploiting inplace flags properly in Numba?
5. If we implement C/Numba versions of RandomVariables we could probably benefit from inplace optimizations on the inputs


# Static vs runtime broadcasting
1. Did some progress for Elemwise in https://github.com/pymc-devs/pytensor/pull/372
2. Questions about shape? Should we do like Theano and assume graphs are valid shape-wise? This allows us to simplify many cases, where otherwise we would need asserts all over the place.
3. Issue is that the static broadcasting may be surprising for users and as such they think they are writing a valid graph without knowing they are not. This seems bad.
4. This sometimes can completely mask issues. For example (this is completely made up!) you may have an Elemwise that would raise if evaluated but then take the gradient, and it's only zeros or whatever, so PyTensor rewrites it away. Now the final graph has no Op that will explicitly fail about runtime broadcasting, and the output will be wrong.


## What to do about those pesky scalars?
 1. Graphs with scalars are considerably more efficient than 0d arrays.
 2. Problem is graphs built by users and all of our rewrite machinery are geared towards 0d arrays.
 3. In JAX some shape related functions fail with 0d arrays. Right now we implicitly cast 0d constants to floats/integers!!! This can lead to weird bugs like in https://github.com/pymc-devs/pytensor/issues/373
 4. Proposals (not mutually exclusive):
     4.1 Convert 0d arrays to scalars near the end of compilation. Convert to arrays during canonicalization
     4.2 Inline constants inside Composites as in https://github.com/pymc-devs/pytensor/pull/361
     4.3 Fuse sequences of 0d arrays in a **Elemwise** Composite (right now it's disabled for those), which is pretty similar to using Scalars, with an iterator over-head for the inputs
     4.4 Fuse sequences of 0d arrays in a **Scalar** Composite with appropriate `ScalarFromTensor` and `TensorFromScalar` calls.
     4.5 Bother/ don't bother with standalone (non-fused) 0d arrays?


## Blockwise
1. Similar questions of 0d array vs scalar arise in the Blockwise PR https://github.com/pymc-devs/pytensor/pull/306
2. Should our rewrites support both forms, or just Blockwise?
3. My approach was to support only Blockwise, and have a late rewrite that removes useless Blockwise (0 batch dims)
4. Many blockwise even with batch dims are useless in some backends (almost everything I looked at was naturally batched in JAX, and in numpy linalg, but not scipy linalg). What about Numba?
5. Easy to include a simple rewrite that removes backend-specific uselesses.
6. Really in need of reviewers! Max already helped a ton.
7. Would like not to wait another year, to get this functionality in PyMC (arbitrarily batched MvNormals and alike!)
8. ~~Do we want to do a C-implementation?~~
9. Do we want to fuse BatchedOps?
10. Who wants to write a Numba impl?
11. Who wants to write a JAX impl?

https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#batching

## Vmap construct
1. We already do something like this for the gradient of Blockwise, where we start from a "core" gradient and vectorize it via  a dispatch.
2. Dispatch fallsback to Blockwise for Ops, but there are special cases like Elemwise/CAReduce/Dimshuffle/RandomVariables that are "natively" batcheable with little-no logic
3. This doesn't have all the bells and whistles of JAX vmap (axes and stuff) but I feel those are not really important? Batching everything to the left sounds easy enough and covers many cases. **Objections?**
4. Should be easy to support in_axes and out_axes, the dispatch functions become a bit more complex, you have to transpose inputs and outputs sometimes. Not sure it's worth it, Adrian thinks it's neat.
5. Can be done in a follow up PR to Blockwise, even if we change the signature of the dispatch function. I (Ricardo) will take the anger for breaking the "public API".

# Dynamic broadcasting
1. Nobody asked but we could add a dynamic broadcasting Op with all the overhad at runtime/gradients using IfElse

## Type compatibility across backends:
All backends support something like Numpy Arrays, integers / floats... But compatibility for RandomGenerator / Sparse Arrays are not 1-to-1.

How to harmonize types across backends:
1. Don't rewrite graph explicitly, and implicitly assume the inputs will be of a specific type. For instance Scipy CSC and CSR in JAX become BCOO, RandomGenerator becose PRNGKey 

    1. Explicit inputs can be converted whenever a function is called. Approach taken in https://github.com/pymc-devs/pytensor/pull/278

    2. Implicit shared inputs are tricky because they could be reused across different calls / backends and they simply store their state in a mutable list. This means that they must be mutated by PyTensor at some point (right now we mutate RNGs in JAX). After mutation they are no longer safe to use in the original backends!!!
    
    3. Proposed solution for shared inputs: Replace shared variables by copies and tell users about this and how to retrieve them once the function is compiled. **Still need to check if this is easily done in the current API**.

2. Rewrite explicitly so graph is represented correctly. Until now I couldn't see any advantage, but maybe there is?

# Rewrites, ordering and eggs
1. Question of rewrite ordering and worries about duplicating costly operations arise almost in every rewrite PR.
2. For instance we could replace Switch(cond, a, b) -> (empty(), set_subtensor(cond, a[cond]), set_subtensor(~cond, b[~cond])) after broadcasting everything. Indexing operations can then be lifted closer to the inputs, making the switch in fact "lazy". But we don't know when is this useful if we can't know how much lifting can be done (as it might otherwise break Elemwise fusion)
3. Eggs and some meta-optimizer sound like the right solution for this. Is it? Can it actually reason about e.g., different permutations of index lifting and fusion rewrites?
4. Do we want to consider it seriously?
5. What are the biggest obstacles? 
    1. Complexly parametrized Ops jump to mind (Good luck representing an SetSubtensor symbolically in any useful way)
    2. Do we need immutable graphs?
7. Worth doing a POC and if it looks promising trying to get some GSOC / Numfocus project for it?

# Other backends
1. PyTorch? Yes, no?
2. XLA directly? Yes, no?
3. MLIR?


## Updates on previous conversations
1. Gradient optimizations
    1. We experimented with running canonicalize/stabilizy in PyMC logp graph because taking the grad.
    2. Still considering the idea of a lazy dummy Grad Op. I think reasoning in terms of gradient operations could be very interesting. 
    3. Still think it achieves the same thing as `value_and_grad` optimizations easily with a very simple kind of PyTensor rewrite. 

2. More ergonomic scan
    1. Mode issue!!!
    1. No updates otherwise

3. True IfElse (in JAX and Numba)
    1. No updates 

3. RandomVariable updates logic
    1. No updates (no pun intended)

4. Gradient logic
    1. Still don't know if the double Lop vs Rop thing is true. 
    2. Should still remove Lop vs Grad distinction
    3. Consider other names 