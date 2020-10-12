### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 6e577570-0c29-11eb-11a3-91fc867a908e
begin 
	using Random
	using Flux # for deep learning and autodiff
	using Flux.Data: DataLoader
	using Flux: params, update!
	using MLJ: make_blobs, make_circles # utilities for data set creation and others 
	using MLJBase: matrix
	using Plots
	plotly()
	hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]));
	using Printf
	using InteractiveUtils
	using PlutoUI
end

# ╔═╡ 088beae0-0b62-11eb-0f0e-9f29aa060ed2
md""" 
# Machine Learning Tutorial with Julia 

In this simple tutorial we will go through the standard Machine Learning workflow of choosing and tuning a model. 

The tutorial will use synthetic data. 

For the installation instruction go to: 
"""

# ╔═╡ 55023282-0b62-11eb-1c12-4942101aece9
md"""
## 1. Load the datasets and explore 

The first part of a machine learning project is to get familiar with the dataset and formulate clearly the problem that we want to solve. In other words, answering the question what do we want our model to do?
"""

# ╔═╡ 9f1937ee-0b63-11eb-154d-0f51431234ec
md"""
In this tutorial we will use synthetic data. The first dataset consists of 2 blobs drawn from two Gaussian distributions. The second dataset consists of concentric circles. 

You can play around with the number of data points using the variables below. 
"""

# ╔═╡ f5ac26e2-0b63-11eb-12e0-7d668db8003a
n_pts = 100;

# ╔═╡ ffc6c7d0-0b62-11eb-1a2b-53fa23f8ac25
md""" 
Generate Gaussian blobs
"""

# ╔═╡ 7ed6a250-0b65-11eb-1fe4-3db1ebb1884f
X_blob, Y_blob = make_blobs(n_pts, 2, centers=2, rng=MersenneTwister(11));

# ╔═╡ 24abdc10-0b64-11eb-0125-8ba3e1a096aa
scatter(X_blob.x1, X_blob.x2, color=convert.(Int64, Y_blob), legend=false)

# ╔═╡ 549d27c0-0b65-11eb-0246-754f34082820
md"""
Generate concentric circles
"""

# ╔═╡ 468dd460-0b63-11eb-3b18-db9930297f4a
X_circle, Y_circle = make_circles(n_pts, factor=0.7);

# ╔═╡ 5ed99e00-0b63-11eb-1dac-b5d788680860
scatter(X_circle.x1, X_circle.x2, color=convert.(Int64, Y_circle), aspect_ratio=:equal, legend=false)

# ╔═╡ 6d5a3452-0b66-11eb-1617-77d692d84564
md"""
## 2. Problem formulation 

In this tutorial, the goal is to build a model, that can automatically classify whether a point in a 2D space belongs to one class or the other. 


The model we are looking for is a function $f: \mathbb{R}^2 \rightarrow \{0, 1\}$ parameterized by $\theta$.

For our problem to be well formulated we need to find a metric to optimize. Do we want to maximize the average performance, the number of false positive, the number of false negative? 

It is important to understand that there are often trade offs between the different objectives. Domain knowledge often dictates whether one is more important than the other. 

In this tutorial we will choose to find a function $f$ that maximizes the likelihood of the dataset. A common assumption is to assume that each datapoint is drawn independently from the same distribution. With that assumption and objective in mind, we can show that maximizing the log likelihood of our data is equivalent to solving: 

$$\min_{\theta} \frac{1}{|\mathcal{D}|}\sum_{(x,y)\in\mathcal{D}} L(x, y; \theta)$$

where 

$$L(x, y; \theta) = y \log(f_{\theta}(x)) + (1 - y)\log(1 - f_{\theta}(x))$$


"""

# ╔═╡ 1519a2b0-0be5-11eb-00ad-099ce6a6a134
md"""
Before we choose the model class and train it, let's decide on an evaluation metrics. As discussed above there are several candidates for classification problems: 
- accuracy 
- false positive rate 
- false negative rate 
- area under the curve 
- f1 score 

In this tutorial we will choose accuracy. Accuracy is the ratio of correctly classified examples over the total number of examples. 
"""

# ╔═╡ 2e16ec92-0be6-11eb-30a5-a7701a02a792
function evaluate(ŷ, y) 
	return sum(y .== ŷ) / length(y)
end

# ╔═╡ bd7fb920-0be6-11eb-36c0-afd7d1e5979b
md"""

## 3. Train, Evaluate, Repeat

Now that we have the dataset, and formulated our problem it is time to choose a model and train it. The training process involve the following tasks: 

- Feature selection: choosing the input to our model 
- Model selection: choosing the class of functions that we want to use to fit our data 
- Hyperparameter tuning: choosing the hyperparameters
- Train: run an optimization algorithm to find the parameters $\theta$ of your model
- Evaluate: evaluate the performance of the trained model. 
- Repeat: this is an iterative procedure, it is very likely that your first choice of model and hyperparameters will not yield good results. 

The next section will show how we can automate this whole procedure. 


"""

# ╔═╡ 7c62e920-0be7-11eb-0f99-1b7eaad426ec
md"""

### Model Selection 

The model class that we choose can affect different aspects: 
- expressivity of the function: to what extent can we represent complex patterns) 
- training speed: how fast it takes to train the model 
- inference speed: how fast it takes to run the model on a data point

In this tutorial we will use [Flux.jl](https://github.com/FluxML/Flux.jl) to construct the model. Flux is a deep learning library that allows you to construct neural networks in a very easy way. 

It relies on an autodifferentiation library, [Zygote.jl](https://github.com/FluxML/Zygote.jl) that can take gradients of functions automatically and efficiently.

Let's construct our model. The simplest model we can think of is a logistic regression:

$$f(x; \theta) = \sigma(\theta^\top x)$$

where 

$$\sigma(x) = \frac{1}{1 + \exp(-x)}$$

In logistic regression we assume a linear combination of model weights and input, but it can be generalized to non-linear functions. We could use a deep neural network for example with the last activation layer being a sigmoid function.

"""

# ╔═╡ 9f6a4740-0be9-11eb-20f7-affaacd9a460
"""
	initialize_model(n_inputs, hiddens) 

Initialize a multi layer perceptron. 

Examples: initialize_model(2, [32,32]) # 2 hidden layers with 32 nodes each
"""
function initialize_model(n_inputs, hiddens)
	# the chain operator allows us to call functions after another f(g(h(x))) = Chain(f, g, h)
	if isempty(hiddens) 
		return Dense(n_inputs, 1, sigmoid) 
	end 
	
	
	layers = []
	# add layers one at a time
	push!(layers, Dense(n_inputs, hiddens[1], relu)) 
	for i=1:length(hiddens)-1
		push!(layers, Dense(hiddens[i], hiddens[i+1], relu))
	end
	push!(layers, Dense(hiddens[end], 1, sigmoid))

		
	return Chain(layers...)
end

# ╔═╡ 83b2a310-0c0e-11eb-1a3a-f1d266f05a81
md"""
**Note that the model outputs a number between 0 and 1, how can we make prediction, output either 0 or 1 with the model?**
"""

# ╔═╡ c6244c50-0c1b-11eb-0f17-0d84adbe2f73
hint(md"""
	
To make a prediction with our model we will compare the output with 0.5. 

$\hat{y} = 1 \text{ if } f(x;\theta) > 0.5 \text{ else } 0$
""")

# ╔═╡ 2bbbfd10-0c1c-11eb-3bd7-2370e3cdc74d
md"""

Implement the predict function below:

"""

# ╔═╡ 65e0e760-0c05-11eb-1226-81a5e9d5b658
md""""
### Feature selection and Pre-Processing

Before training the model we are going to format our dataset such that it is suitable for training. 

The steps are the following: 

- Make the y labels between 0 and 1 
- Split the dataset into train and test 
- Normalize the dataset (we will skip this part in the example but it usually is very important!)
- Split the training data into mini-batches
"""

# ╔═╡ 9a458832-0c05-11eb-3fe5-b190450b9d11
function process_data(X, Y)
	x = matrix(X, transpose=true) 
	y = convert(Vector{Float64}, Y) .- 1.0 # make between 0 and 1 
	y = reshape(y, (1,:))
	
	
	# split in train and test 
	train, test = partition(eachindex(y), 0.7, shuffle=true);
	
	return x[:, train], y[:,train], x[:, test], y[:,test] 
end

# ╔═╡ b1303b30-0c05-11eb-1629-033109ccf3c4
xtrain, ytrain, xtest, ytest = process_data(X_blob, Y_blob)

# ╔═╡ d0fc070e-0beb-11eb-16ee-697d37d49d4f
md"""
### Training

Now let's train the model using Stochastic Gradient descent

We have a few hyperparameters to define for the training. 

- Learning rate $\eta$: dictate the step size of each parameter update in the direction in the descending gradient

- Batch size: how many data points to use to compute each gradient update. 

- Number of epochs: how many times to go through the whole dataset. 

Note that the model parameters can also be considered as hyperparameters, the number of nodes, number of layers, activation functions (for NNs).

"""

# ╔═╡ f24ad132-0c18-11eb-2a00-b53f3d4fbfc3
md"""

Let's now construct the inner loop of our training procedure. 


**We are given a minibatch of data and a model, what are the main steps of the training procedure?**


"""

# ╔═╡ 656d4dee-0c19-11eb-0a5a-ebc625112762
hint(md"""
	
It consists of the following: 

- compute loss function on mini batch 
- take gradients of the loss with respect to model parameters
- update model parameters 
- evaluate the model to track progress
	""")

# ╔═╡ 51c52db0-0c17-11eb-06f4-738684d02128
md"""
We are then wrapping the whole pipeline into one whole function. 

**What are the inputs to that function?**

"""

# ╔═╡ 0b00fff0-0c1a-11eb-181e-f774368ddf23
hint(md"""This function takes as input, the processed data, the hyperparameters and the model, and perform training and evaluation.""")

# ╔═╡ 80484ac0-0c15-11eb-1f65-e370ce717e03
md"""
We define a few plotting functions below, click on the eye to see the code 
"""

# ╔═╡ 529fda72-0c10-11eb-0654-df3883c69244
function plot_training_summary(n_epochs, train_loss, test_loss, train_accuracy, test_accuracy)
	p1 = plot(1:n_epochs, train_loss, color=:blue, label="train loss") 
	plot!(p1, 1:n_epochs, test_loss, color=:orange, label="test loss")
	p2 = plot(1:n_epochs, train_accuracy, color=:red, label="train accuracy") 
	plot!(p2, 1:n_epochs, test_accuracy, color=:green, label="test accuracy")
	return plot(p1, p2, layout=(2,1))
end

# ╔═╡ 23df9b10-0c17-11eb-3249-2906e0c79309
n_inputs = 2

# ╔═╡ 2162d3e0-0c1a-11eb-2ae1-37197227b4f2
begin 
	
	
	md"""
	
	Let's build intuition on the effect of each hyperparameter 
	
	Learning rate: $(@bind η NumberField(0.00 : 0.0001 : 0.30, default=0.01))
	
	Batch size: $(@bind bs Slider(1:1:n_pts, show_value=true))
	
	Model structure: $(@bind hiddens MultiSelect(["linear"=> Float64[], 
												 "[2]" => [2], 
												 "[4]" => [4], 
												 "[32]"=> [32], 
												 "[4,4]"=>[4,4]]))
	
	"""
end

# ╔═╡ 26ae2460-0c17-11eb-3bf7-0f17423bb6a0
model = initialize_model(n_inputs, hiddens)

# ╔═╡ bd9655c0-0be3-11eb-00a9-ddf68bdb4673
function loss(x, y) 
	return -mean(y.*log.(model(x)) .+ (1.0 .- y).*log.(1.0 .- model(x)))
end

# ╔═╡ c1541280-0c0e-11eb-3e99-d59f27ae2581
function predict(x) 
	# todo: implement the prediction function (1 line)
	return model(x) .> 0.5 
end

# ╔═╡ 77831d20-0c15-11eb-0665-15190accb287
function plot_decision_boundary(xtrain, ytrain, xtest, ytest, model, boundary=true)
	X = hcat(xtrain, xtest) 
	xlim = minimum(X[1,:]):0.01:maximum(X[1,:])
	ylim = minimum(X[2,:]):0.01:maximum(X[2,:])
	fun = boundary ? predict : model
	contour(xlim, ylim, (x,y) -> fun([x,y])[1], fill=true, color=[0,1], alpha=0.5, colorbar=false)
	scatter!(xtrain[1,:], xtrain[2,:], color=convert.(Int64, ytrain[:]), label="training set")
	scatter!(xtest[1,:], xtest[2,:], color=convert.(Int64, ytest[:]), label="test set", markerstrokecolor=:red)
end

# ╔═╡ 87254020-0c04-11eb-13a2-03b0711e2f83
function train!(train_data, test_data, opt)

	θ = params(model)
	
	train_loss = 0.0
	
	for (x,y) in train_data 
		# compute gradients and update parameters
		local minibatch_loss
		
		∇θ = gradient(θ) do 
			minibatch_loss = loss(x, y) 
		end
		
		train_loss += minibatch_loss
		
		update!(opt, θ, ∇θ)

	end
	
	train_loss /= length(train_data)
	
	test_loss = loss(test_data...)
	train_accuracy = evaluate(predict(xtrain), ytrain)
	test_accuracy = evaluate(predict(xtest), ytest)
	
	training_summary = @sprintf("Train loss: %4.3f | Test loss: %4.3f | Train accuracy: %4.3f | Test accuracy %4.3f", train_loss, test_loss, train_accuracy, test_accuracy)

	return train_loss, test_loss, train_accuracy, test_accuracy, training_summary
end
	

# ╔═╡ e599ea70-0c18-11eb-14ca-b541ef9de76e
function train_many_epochs!(n_epochs, train_data, test_data, opt, model)
	train_loss = zeros(n_epochs)
	test_loss = zeros(n_epochs)
	test_accuracy = zeros(n_epochs) 
	train_accuracy = zeros(n_epochs)
	
	for i=1:n_epochs
		# run training
		train_loss[i], test_loss[i], train_accuracy[i], test_accuracy[i], training_summary = train!(train_data, test_data, opt)
		training_summary 
	end

	p1 = plot_training_summary(n_epochs, train_loss, test_loss, train_accuracy ,test_accuracy)
	p2 = plot_decision_boundary(xtrain, ytrain, xtest, ytest, model)
	return plot(p1, p2, layout=(2, 1))
end

# ╔═╡ cbfd84f2-0c18-11eb-2307-55f2f1646fe1
function train_and_evaluate!((xtrain,ytrain), (xtest, ytest), η, bs, model)
	
	# collect the dataset in minibatch
	train_data = DataLoader((xtrain, ytrain), batchsize=bs, shuffle=true)
	test_data = (xtest, ytest)
	
	# initialize the optimizer (the choice of the optimizer can be a hyper parameter as well)
	opt = Descent(η) 
	
	return train_many_epochs!(n_epochs, train_data, test_data, opt, model)
end

# ╔═╡ 45886582-0c17-11eb-0eab-df58c9a59f19
train_and_evaluate!((xtrain, ytrain), (xtest, ytest), η, bs, model) 

# ╔═╡ fc6a2d10-0c17-11eb-0c5d-efd95fac8bf9


# ╔═╡ Cell order:
# ╠═088beae0-0b62-11eb-0f0e-9f29aa060ed2
# ╠═6e577570-0c29-11eb-11a3-91fc867a908e
# ╟─55023282-0b62-11eb-1c12-4942101aece9
# ╟─9f1937ee-0b63-11eb-154d-0f51431234ec
# ╠═f5ac26e2-0b63-11eb-12e0-7d668db8003a
# ╟─ffc6c7d0-0b62-11eb-1a2b-53fa23f8ac25
# ╠═7ed6a250-0b65-11eb-1fe4-3db1ebb1884f
# ╠═24abdc10-0b64-11eb-0125-8ba3e1a096aa
# ╟─549d27c0-0b65-11eb-0246-754f34082820
# ╠═468dd460-0b63-11eb-3b18-db9930297f4a
# ╠═5ed99e00-0b63-11eb-1dac-b5d788680860
# ╟─6d5a3452-0b66-11eb-1617-77d692d84564
# ╠═bd9655c0-0be3-11eb-00a9-ddf68bdb4673
# ╟─1519a2b0-0be5-11eb-00ad-099ce6a6a134
# ╠═2e16ec92-0be6-11eb-30a5-a7701a02a792
# ╟─bd7fb920-0be6-11eb-36c0-afd7d1e5979b
# ╟─7c62e920-0be7-11eb-0f99-1b7eaad426ec
# ╠═9f6a4740-0be9-11eb-20f7-affaacd9a460
# ╟─83b2a310-0c0e-11eb-1a3a-f1d266f05a81
# ╟─c6244c50-0c1b-11eb-0f17-0d84adbe2f73
# ╟─2bbbfd10-0c1c-11eb-3bd7-2370e3cdc74d
# ╠═c1541280-0c0e-11eb-3e99-d59f27ae2581
# ╟─65e0e760-0c05-11eb-1226-81a5e9d5b658
# ╠═9a458832-0c05-11eb-3fe5-b190450b9d11
# ╠═b1303b30-0c05-11eb-1629-033109ccf3c4
# ╟─d0fc070e-0beb-11eb-16ee-697d37d49d4f
# ╠═cbfd84f2-0c18-11eb-2307-55f2f1646fe1
# ╠═e599ea70-0c18-11eb-14ca-b541ef9de76e
# ╟─f24ad132-0c18-11eb-2a00-b53f3d4fbfc3
# ╟─656d4dee-0c19-11eb-0a5a-ebc625112762
# ╠═87254020-0c04-11eb-13a2-03b0711e2f83
# ╟─51c52db0-0c17-11eb-06f4-738684d02128
# ╟─0b00fff0-0c1a-11eb-181e-f774368ddf23
# ╟─80484ac0-0c15-11eb-1f65-e370ce717e03
# ╟─529fda72-0c10-11eb-0654-df3883c69244
# ╟─77831d20-0c15-11eb-0665-15190accb287
# ╠═23df9b10-0c17-11eb-3249-2906e0c79309
# ╟─2162d3e0-0c1a-11eb-2ae1-37197227b4f2
# ╠═26ae2460-0c17-11eb-3bf7-0f17423bb6a0
# ╠═45886582-0c17-11eb-0eab-df58c9a59f19
# ╠═fc6a2d10-0c17-11eb-0c5d-efd95fac8bf9
