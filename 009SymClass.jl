using MLJ  # for fit/predict
using SymbolicRegression  # for SRRegressor
using Zygote  # For `enable_autodiff=true`
using SymbolicUtils
using DelimitedFiles
using LinearAlgebra # for normalize
cd(@__DIR__)

X=readdlm("X.out", ',', Float64)
Y=((readdlm("Y.out", ',', Float64)[:,1]).*2).-1
Y
X


model = SRRegressor(;
    binary_operators=[+, -, *, /],
    unary_operators=[sin, exp],
    complexity_of_constants=1,
    complexity_of_operators=[exp => 5, sin => 5],
    elementwise_loss=L1HingeLoss(),
    should_simplify = true,
    should_optimize_constants = true,
    enable_autodiff=true,
    batching=true,
    batch_size=25,
    niterations=200,#100
    #early_stop_condition=1e-10,
    maxsize=30  # for gravity force eq
    )
mach = machine(model, X, Y)

fit!(mach)

r = report(mach)
eq = r.equations[r.best_idx]

symbolic_eq = node_to_symbolic(eq, model)


r