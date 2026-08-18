# Installation

QMagma.jl is not registered, so it has to be installed from its repository:

```julia
using Pkg
Pkg.add(url="https://github.com/aelligp/QMagma.jl")
```

or, for a checkout you can edit:

```julia
using Pkg
Pkg.develop(url="https://github.com/aelligp/QMagma.jl")
```

The package requires Julia 1.10 or newer.

```julia
using QMagma
```

!!! info "The GUI lives in a package extension"
    Makie is a weak dependency and the app is built from backend-agnostic widgets, but
    window placement and the window icon go through GLFW, which only the GLMakie backend
    provides. [`sill_intrusion_1D`](@ref) errors until `GLMakie` is loaded as well:

    ```julia
    using QMagma, GLMakie
    ```

    The solver, tracer, and export functions need none of it, so headless runs neither
    precompile GLMakie nor require an OpenGL context. Loading GLMakie does require one;
    on a headless machine, run Julia under `xvfb-run`.

## Threads

Zircon ages are computed independently per tracer and the loop runs on all available
threads. Start Julia accordingly if you plan to use
[`compute_zircon_ages`](@ref) on a large tracer population:

```shell
julia --threads auto
```

## Testing

```julia
using Pkg
Pkg.test("QMagma")
```

!!! warning "Known flake"
    The test process occasionally segfaults after reporting that the tests passed. This is
    a GLMakie/GPU teardown failure at process exit.
