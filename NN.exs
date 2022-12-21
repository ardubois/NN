Mix.install([
  {:nx, "~> 0.1.0"}
])

defmodule NN do
  import Nx.Defn
  defn relu(x) do
    Nx.max(x,0)
  end
  defn relu2deriv(output) do
    Nx.greater(output,0)
  end
  def runNet(input,[weights_l],target,lr) do
    o = Nx.dot(input,weights_l)
    finalDerivative = finalD(o,target)
    newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    nextLayerD = Nx.dot(finalDerivative,Nx.transpose(weights_l))
    {newWeights,nextLayerD}
  end
  def runNet(input,[w|tl],target,lr) do
    o =  relu(Nx.dot(input,w))
    {net,wD} = runNet(o,tl,target,lr)
    myDeriv = layerD(wD,relu2deriv(o))
    newWeights = genNewWeights(w,lr,input,myDeriv)
    nextLayerD = Nx.dot(myDeriv,Nx.transpose(w))
    {[newWeights|net],nextLayerD}
  end

  defn layerD(wD,output) do
        wD*output
  end
  defn genNewWeights(weights,lr,layer,der) do
      weights - (lr*Nx.dot(Nx.transpose(layer),der))
  end
  defn finalD(output,target) do
     output - target
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
alpha = 0.2
weights_0_1 = Nx.tensor ( [[-0.16595599,  0.40763847, -0.99977125],
                            [-0.39533485, -0.70648822, -0.81532281],
                            [-0.62747958 ,-0.34188906 ,-0.20646505]]) #NN.newDenseLayer(inputSize,hiddenSize,:relu)


weights_1_2 = Nx.tensor([[ 0.07763347],
                          [-0.16161097],
                          [ 0.370439  ]])#NN.newDenseLayer(hiddenSize,outputSize,:relu)

#nn = [NN.newDenseLayer(inputSize,hiddenSize,:relu),
#      NN.newDenseLayer(hiddenSize,outputSize,:relu)]


nn = [weights_0_1,weights_1_2]

sl_input = Nx.tensor([  [ 1, 0, 1],
                        [ 0, 1, 1],
                        [ 0, 0, 1],
                        [ 1, 1, 1] ])

sl_target = Nx.transpose(Nx.tensor([[1, 1, 0, 0]]))



input1 = sl_input[0..0]

target = sl_target[0..0]


#IO.puts weights_0_1
o = Nx.dot(input1,weights_0_1)

#IO.puts o
{newNet,d} = NN.runNet(sl_input,nn,sl_target,alpha)

[head|tail] = newNet

IO.puts head
