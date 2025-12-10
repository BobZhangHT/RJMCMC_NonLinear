#!/usr/bin/env julia

# Install all required Julia packages for RJMCMC simulation
# Run with: julia install_packages.jl

using Pkg

packages = [
    "DataFrames",
    "CSV",
    "Distributions",
    "ProgressMeter",
    "StatsBase",
    "VectorizedStatistics",
    "MLDataUtils",
    "Plots",
    "StatsPlots",
    "CategoricalArrays",
    "SpecialFunctions",
    "LaTeXStrings"
]

println("Installing required Julia packages...")
println("Packages to install: ", join(packages, ", "))

try
    Pkg.add(packages)
    println("\n✓ All packages installed successfully!")
catch e
    println("\n✗ Error installing packages: ", e)
    rethrow(e)
end

