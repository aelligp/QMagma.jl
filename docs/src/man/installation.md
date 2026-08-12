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

The package requires Julia 1.10 or newer. Loading it takes a while the first time, since
GLMakie is a regular dependency and must be precompiled:

```julia
using QMagma
```

!!! info "The GUI needs no extra packages"
    GLMakie is a direct dependency rather than a package extension, so
    [`sill_intrusion_1D`](@ref) is available as soon as `QMagma` is loaded. This also
    means a working OpenGL context is needed to load the package. On a headless machine,
    run Julia under `xvfb-run`.

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
