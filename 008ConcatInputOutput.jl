using MLJ  # for fit/predict
using SymbolicRegression  # for SRRegressor
using Zygote  # For `enable_autodiff=true`
using SymbolicUtils
using DelimitedFiles
using LinearAlgebra # for normalize
cd(@__DIR__)

myarrayX=readdlm("circle_X.out", ',', Float64)
myarrayGradients=readdlm("circle_gradients.out", ',', Float64)
myinputsize=size(myarrayX)[2]

X_concat=cat(myarrayX,myarrayGradients,dims=2) # smuggle in the labels through data X

# fake labels
f(x) = 0
y = f.(myarrayX[:, 1]) 

function derivative_loss(tree, dataset::Dataset{T,L}, options, idx) where {T,L}

    # Return infinite loss for any violated assumptions, does not seem necessary, but sometimes better
    tree.degree != 2 && return L(Inf)
    #tree.l.degree != 2 && return L(Inf)

    # Prevent nodes corresponding to invalid features that arise from smuggling in the target y through x    
    is_invalid_feature(node) = node.degree == 0 && !node.constant && (node.feature > myinputsize)
    any(is_invalid_feature, tree) && return L(Inf)

    # Select from the batch indices, if given and extract normalized gradients from X
    X = idx === nothing ? dataset.X : view(dataset.X, :, idx)
    ∂y = X[myinputsize+1:2*myinputsize,:]

    # Evaluate both f(x) and f'(x), where f is defined by `tree`
    ŷ, ∂ŷ, completed = eval_grad_tree_array(tree, X, options; variable=true)
    !completed && return L(Inf)

    # Only use gradients wrt to real features
    ∂ŷ=∂ŷ[1:myinputsize,:]

    # Normalize gradients with Euclidean norm
    normalize!.(eachcol(∂ŷ))

    # Calculate mean square error loss function on normalized gradients
    mse_grad = sum(i -> (∂ŷ[i] - ∂y[i])^2, eachindex(∂y)) / length(∂y)

    return  mse_grad
end

model = SRRegressor(;
    binary_operators=[+, -, *, /],
    unary_operators=[sin, exp],
    complexity_of_constants=3,
    complexity_of_operators=[exp => 5, sin => 5],
    loss_function=derivative_loss,
    should_simplify = true,
    should_optimize_constants = true,
    enable_autodiff=true,
    batching=true,
    batch_size=25,
    niterations=200,#100
    early_stop_condition=1e-10,
    maxsize=30  # for gravity force eq
    )
mach = machine(model, X_concat, y)

fit!(mach)

r = report(mach)
eq = r.equations[r.best_idx]

symbolic_eq = node_to_symbolic(eq, model)


r




