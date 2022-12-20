Mix.install([
  {:nx, "~> 0.1.0"}
])

defmodule NN do
  import Nx.Defn
  defn relu(x) do
    Nx.max(x,0)
  end
  def runLayer(input,[l|tl]) do
    o =  relu(Nx.dot(input,l))

  end
  def runLayer(input,[l]) do
  end
  defn relu2deriv(output) do
    Nx.greater(output,0)
  end

  def newDenseLayer(x,y,type) do
   fitWeights(Nx.random_uniform({x, y}, 0.0, 1.0, type: {:f, 64}))
  end

  defn fitWeights(w) do
    2*w-1
  end
end

t = Nx.tensor([[99, -1], [3, -7]])

#IO.puts NN.relu(t)

inputSize = 3
hiddenSize = 4
outputSize = 1
weights_0_1 = NN.newDenseLayer(inputSize,hiddenSize,:relu)
weights_1_2 = NN.newDenseLayer(hiddenSize,outputSize,:relu)

nn = [NN.newDenseLayer(inputSize,hiddenSize,:relu), NN.newDenseLayer(hiddenSize,outputSize,:relu)]








IO.puts weights_0_1
