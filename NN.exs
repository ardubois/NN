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
    error =  Nx.sum(Nx.power(finalDerivative,2))
    IO.puts("error")
    IO.inspect error
    newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    nextLayerD = Nx.dot(finalDerivative,Nx.transpose(weights_l))
    {[newWeights],nextLayerD,error}
  end
  def runNet(input,[w|tl],target,lr) do
    o =  relu(Nx.dot(input,w))
    {net,wD,error} = runNet(o,tl,target,lr)
    myDeriv = layerD(wD,relu2deriv(o))
    newWeights = genNewWeights(w,lr,input,myDeriv)
    nextLayerD = Nx.dot(myDeriv,Nx.transpose(w))
    {[newWeights|net],nextLayerD,error}
  end
  def trainNN(1, input, nn,target,lr) do
    input1 = input[0..0]
    target1 = target[0..0]
    {net,wD,error} = runNet(input1,nn,target1,lr)
    {net,error}
  end
  def trainNN(n,input,nn,target,lr) do
    input1 = input[0..0]
    target1 = target[0..0]
    {net,wD,newError} = runNet(input1,nn,target1,lr)
    tinput  = input[1..-1//1] # Drop the first "row"
    ttarget = target[1..-1//1] # Drop the first "row"
    {finalNet,errorsofar} = trainNN(n-1,tinput,net,ttarget,lr)
    myError = sumTensors(newError,errorsofar)
    {finalNet,myError}
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
  defn sumTensors(t1,t2) do
    t1 + t2
  end
  def newDenseLayer(x,y,type) do
   fitWeights(Nx.random_uniform({x, y}, 0.0, 1.0, type: {:f, 64}))
  end

  defn fitWeights(w) do
    2*w-1
  end
end

#t = Nx.tensor([[99, -1], [3, -7]])

#IO.puts NN.relu(t)

#inputSize = 3
#hiddenSize = 4
#outputSize = 1
alpha = 0.2
weights_0_1 = Nx.tensor ( [[-0.16595599,  0.40763847, -0.99977125],
                            [-0.39533485, -0.70648822, -0.81532281],
                            [-0.62747958 ,-0.34188906 ,-0.20646505]]) #NN.newDenseLayer(inputSize,hiddenSize,:relu)


weights_1_2 = Nx.tensor([[ 0.07763347],
                          [-0.16161097],
                          [ 0.370439  ]])#NN.newDenseLayer(hiddenSize,outputSize,:relu)


w0_ = Nx.tensor( [[-0.16595599,  0.44064899, -0.99977125, -0.39533485],
[-0.70648822, -0.81532281, -0.62747958 ,-0.30887855],
[-0.20646505 , 0.07763347 ,-0.16161097 , 0.370439  ]])
w1_ = Nx.tensor([[-0.5910955 ],
 [ 0.75623487],
[-0.94522481],
[ 0.34093502]])
#nn = [NN.newDenseLayer(inputSize,hiddenSize,:relu),
#      NN.newDenseLayer(hiddenSize,outputSize,:relu)]


nn = [weights_0_1,weights_1_2]

nn2_ = [w0_,w1_]

sl_input = Nx.tensor([  [ 1, 0, 1],
                        [ 0, 1, 1],
                        [ 0, 0, 1],
                        [ 1, 1, 1] ])

sl_target = Nx.transpose(Nx.tensor([[1, 1, 0, 0]]))



input1 = sl_input[0..0]

target1 = sl_target[0..0]

#{newNet,d,errorFinal} = NN.runNet(input1,nn,target1,alpha)


{newNet,errorFinal} = NN.trainNN(4,sl_input,nn2_,sl_target,alpha)

IO.puts "Error final:"
IO.inspect errorFinal

[head|tail_] = newNet

IO.inspect head
