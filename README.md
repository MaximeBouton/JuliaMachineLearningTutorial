# Machine Learning Tutorial with Julia 

This is a simple tutorial illustrating the workflow of a machine learning project. 
It consists of fitting a model on a synthetic dataset. It provides an interactive way to explore the effect of hyperparameters on model performance. 

## Installation 

- Install julia 1.5, download it from here: https://julialang.org/downloads/ 
- Add julia to your path
- clone this project: `git clone `
- go to the directory of the project and run julia: 
```bash 
cd MLTutorial 
julia --project 
```
- install the dependencies by running the following in the Julia REPL: 
  ```julia 
  julia> import Pkg; Pkg.instantiate() 
  ``` 
- close julia, you are done installing everything. 

For a more detailed explanation, you can watch this excellent video: https://www.youtube.com/watch?v=OOjKEgbt8AI&list=PLP8iPy9hna6Q2Kr16aWPOKE0dz9OnsnIJ&index=30

## Runing the notebook 

Open julia and run the following: 

```julia 
julia> using Pluto
julia> Pluto.run()
```

A window should pop up in your browser with a screen that looks like this: 

![pluto_welcome](pluto_welcome.png)

Enter the name of the notebook in the field: `machine_learning_tutorial_notebook.jl`
