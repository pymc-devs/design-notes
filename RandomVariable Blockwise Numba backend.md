# RandomVariables/Blockwise in Numba backend


## Problem

Allocating the output arrays requires knowing `size`/`batch_shape` (easy) and `core_shape`.

## Options

1. Add an argument `core_shape` to the Op itself

    *Downside:*
    1. Verbose, it's not a "true input" to most RVs, in that it can't be changed and is mostly not checked. It is a true input for timeseries RVs (nsteps), but we don't use RandomVariables for those these days anyway
    1. Makes graph representation more complicated 
    1. Useless for backends that don't need it (i.e., all but Numba)  

    *Upside:*
    1. It's part of the graph, can be inferred/constant folded if more complicated. Can be merged if shape graph shows up elsewhere. 
    1. Uses the same code that is already needed to infer the static_shape/output shape (DRY).
    1. Works for Blockwise. 

    Implemented in https://github.com/pymc-devs/pytensor/pull/691
    
1. Replace `size` by `shape`

    *Downside:*
    1. Same as with `core_shape` input. 
    1. Does not allow `size=None` (implied size). I am not sure what this is good for though. 
    1. Not a (great?) solution for Blockwise

    *Upside:*
    1. No extra number of inputs 
    1. PyMC can pass shape directly  
    
1.  Use a specialized Op that's introduced later and only for the backends that need it (i.e., Numba)

    *Downside:*
    1. May make rewrite ordering cumbersome
    1. Graph is not executable without rewrites (not a biggie for me)
    1. Works for Blockwise 

    *Upside:*
    1. Doesn't clutter main IR
    1. Doesn't clutter backends where it is not needed
    1. Can be made arbitrarily complex without worries. Perhaps pre-allocating the output buffers at the PyTensor level like we do for Scan 
    
1. Wait for first eval to find out `core_shape` and only then allocate. This is what the Numba impl of Scan does for outputs without taps (nit-sot). 

    *Downside:*
    1. Potentially very inneficient?

    *Upside:*
    1. No extra care needed at the graph representation
    1. Works for Blockwise 


1. Compile `core_shape` graph function at dispatch and use that. 

    *Downside:*
    1. Avoids computation merging if shape graph was already present for something else or same graph applies to multiple Ops
    1. Makes dispatch impl more complicated 

    *Upside:*
    1. No extra care needed at the graph representation
    1. Still uses same machinery (DRY) 
    1. Works for Blockwise 


1. Don't use PyTensor machinery at all. Implement a Numba dispatch that takes inputs are arguments and returns core shape

    *Downside:*
    1. Avoids computation merging if shape graph was already present for something else or same graph can be used for multiple Ops
    1. Makes dispatch impl more complicated
    1. Does not provide an automatic solution for Blockwise 

    *Upside:*
    1. No extra care needed at the graph representation


## What does Numba do?
At the moment it doesn't allow guvectorize signatures with constant shapes (literal ints), or output symbols that are not present in the inputs
1. https://github.com/numba/numba/issues/6690
1. https://github.com/numba/numba/issues/2797


## What others have been thinking about?
1. Make signature a more powerful DSL or allow callables for core_shapes: 
    1. https://github.com/numpy/numpy/issues/18151
    1. https://github.com/WarrenWeckesser/numpy-notes/blob/main/enhancements/gufunc-size-expressions.md
    1. https://github.com/WarrenWeckesser/numpy-notes/blob/main/enhancements/gufunc-shape-only-params.md
    1. https://github.com/pymc-devs/pytensor/pull/143

