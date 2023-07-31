
## Do we allow `TensorType(shape=(1,), broadcastable=(False,))`?

If so, what is the broadcastable flag of an Elemwise output?

```python=
x = pt.vector("x", shape=(1,), broadcastable=(False,))
y = x + x
assert y.type == ?
```

What about clone_replace? It seems like it should fail in `strict=False` (default)

```python=
import pytensor.tensor as pt
x = pt.vector("x")
y = x + 5
pytensor.dprint(y, print_type=True)

new_x = pt.vector("new_x", shape=(1,))
new_y = pytensor.clone_replace(y, {x: new_x})
pytensor.dprint(new_y, print_type=True)
```
```
Add [id A] <Vector(float64, shape=(?,), broadcastable=(False,))>
 ├─ x [id B] <Vector(float64, shape=(?,), broadcastable=(False,))>
 └─ ExpandDims{axis=0} [id C] <Vector(int8, shape=(1,), broadcastable=(True,))>
    └─ 5 [id D] <Scalar(int8, shape=(), broadcastable=())>

Add [id A] <Vector(float64, shape=(1,), broadcastable=(True,))>
 ├─ Unbroadcast{0} [id B] <Vector(float64, shape=(1,), broadcastable=(False,))>
 │  └─ new_x [id C] <Vector(float64, shape=(1,), broadcastable=(True,))>
 └─ ExpandDims{axis=0} [id D] <Vector(int8, shape=(1,), broadcastable=(True,))>
    └─ 5 [id E] <Scalar(int8, shape=(), broadcastable=())>
```

## Runtime broadcasting in Alloc and non-gradient inputs of SetSubtensor / RandomVariables:
https://github.com/pymc-devs/pytensor/pull/390 (NEEDS REVIEW)
https://github.com/pymc-devs/pytensor/pull/329#discussion_r1262320680
1. Should we allow / be as strict as in other Ops?


## Rewrite / shape safety:
 1. https://github.com/pymc-devs/pytensor/pull/381 (NEEDS REVIEW)
 2. PR cleans up many rewrites logic by not worrying about shape safety. Rewrites are tagged so users can exclude them. Seems reasonable?

## Scalar vs 0d tensors?
 1. Elemwise for non-fuseable 0d arrays doesn't seem to have that much of a drawback (at least in C / JAX backends): https://github.com/pymc-devs/pytensor/issues/349#issuecomment-1635526597
 2. We already fuse chains of 0d elemwise, which take care of boxing/unboking at the edges, but are otherwise pure scalar operations in between 
 3. We can avoid some more cases by "unbroadcasting" constant arrays that go into Elemwise (already done for fusion graphs): https://github.com/pymc-devs/pytensor/pull/361
 4. Can we consider this solved?


## Type compatibility across backends:
Relevant JAX PR on Sparse / RNGs inputs: https://github.com/pymc-devs/pytensor/pull/278 (Mixed reviews so far)

## Type compatibilty across backends for Shared variables:
AFAICT can't be (easily) solved by the multiple container idea. We could implement a subclass of list that syncronizes on right, but I am afraid this would break it on the C-backend.
Solution: Copy shared variables that have incompatabile types and tell users how they can be retrieved from the compiled function
Problem: I couldn'n figure out how to do this.


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

https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#batching

## Vmap construct
1. We already do something like this for the gradient of Blockwise, where we start from a "core" gradient and vectorize it via  a dispatch.
2. Dispatch fallsback to Blockwise for Ops, but there are special cases like Elemwise/CAReduce/Dimshuffle/RandomVariables that are "natively" batcheable with little-no logic
3. This doesn't have all the bells and whistles of JAX vmap (axes and stuff) but I feel those are not really important? Batching everything to the left sounds easy enough and covers many cases. **Objections?**
4. Should be easy to support in_axes and out_axes, the dispatch functions become a bit more complex, you have to transpose inputs and outputs sometimes. Not sure it's worth it, Adrian thinks it's neat.
5. Can be done in a follow up PR to Blockwise, even if we change the signature of the dispatch function. I (Ricardo) will take the anger for breaking the "public API".


## Rewrites, ordering and eggs
1. Question of rewrite ordering and worries about duplicating costly operations arise almost in every rewrite PR.
2. For instance we could replace Switch(cond, a, b) -> (empty(), set_subtensor(cond, a[cond]), set_subtensor(~cond, b[~cond])) after broadcasting everything. Indexing operations can then be lifted closer to the inputs, making the switch in fact "lazy". But we don't know when is this useful if we can't know how much lifting can be done (as it might otherwise break Elemwise fusion)
3. Eggs and some meta-optimizer sound like the right solution for this. Is it? Can it actually reason about e.g., different permutations of index lifting and fusion rewrites?
4. Do we want to consider it seriously?
5. What are the biggest obstacles? 
    1. Complexly parametrized Ops jump to mind (Good luck representing an SetSubtensor symbolically in any useful way)
    2. Do we need immutable graphs?
7. Worth doing a POC and if it looks promising trying to get some GSOC / Numfocus project for it?



## Updates on previous conversations
1. Gradient optimizations
    1. We experimented with running canonicalize/stabilizy in PyMC logp graph because taking the grad.
    2. Still considering the idea of a lazy dummy Grad Op. I think reasoning in terms of gradient operations could be very interesting. 
    3. Still think it achieves the same thing as `value_and_grad` optimizations easily with a very simple kind of PyTensor rewrite. 

2. More ergonomic scan
    1. No updates

3. True IfElse (in JAX and Numba)
    1. No updates 

3. RandomVariable updates logic
    1. No updates (no pun intended)

4. Gradient logic
    1. Still don't know if the double Lop vs Rop thing is true. 
    2. Should still remove Lop vs Grad distinction
    3. Consider other names 