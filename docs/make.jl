using Documenter
using DocumenterVitepress

using QMagma

# Generate the documentation page from the repository license.
open(joinpath(@__DIR__, "src", "man", "license.md"), "w") do io
    println(
        io, """
        ```@meta
        EditURL = "https://github.com/aelligp/QMagma.jl/blob/main/LICENSE"
        ```
        """
    )
    println(io, "# [License](@id license)")
    println(io, "")
    for line in eachline(joinpath(dirname(@__DIR__), "LICENSE"))
        println(io, "> ", line)
    end
end

@info "Making documentation..."

makedocs(;
    sitename = "QMagma.jl",
    authors = "Pascal Aellig and contributors",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/aelligp/QMagma.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    modules = [QMagma],
    pages = [
        "Home" => "index.md",
        "User guide" => Any[
            "Installation" => "man/installation.md",
            "The GUI" => "man/gui.md",
            "Model formulation" => "man/model.md",
            "Eruptions" => "man/eruptions.md",
            "Tracers and zircon ages" => "man/tracers.md",
            "Headless runs and export" => "man/scripting.md",
        ],
        "List of functions" => "man/listfunctions.md",
        "License" => "man/license.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/aelligp/QMagma.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
