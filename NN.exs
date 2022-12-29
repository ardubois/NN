Mix.install([
  {:nx, "~> 0.4.0"},
  {:torchx, "~> 0.4.0"},
  {:exla, "~> 0.4"}
])

Nx.default_backend(Torchx.Backend)
#Nx.default_backend(EXLA.Backend)
# Sets the global compilation options
#Nx.Defn.global_default_options(compiler: EXLA)
# Sets the process-level compilation options
#Nx.Defn.default_options(compiler: EXLA)
defmodule NN do
  import Nx.Defn
 # Nx.default_backend(Torchx)
  defn relu(x) do
    Nx.max(x,0)
  end
  defn relu2deriv(output) do
    Nx.greater(output,0)
  end
  def runNet(input,[weights_l],target,lr) do
    o = Nx.dot(input,weights_l)
    finalDerivative = sub(target,o)
    error =  Nx.sum(Nx.power(finalDerivative,2))
    correct = Nx.as_type(Nx.equal(Nx.argmax(o),Nx.argmax(target)),{:u, 32})
    newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    nextLayerD = Nx.dot(finalDerivative,Nx.transpose(weights_l))
    {[newWeights],nextLayerD,error,correct}
  end
  def runNet(input,[w|tl],target,lr) do
    o =  relu(Nx.dot(input,w))
    {net,wD,error,correct} = runNet(o,tl,target,lr)
    myDeriv = mult(wD,relu2deriv(o))
    #IO.inspect(relu2deriv(o))
    newWeights = genNewWeights(w,lr,input,myDeriv)
    #IO.inspect(Nx.sum(newWeights))
    #nextLayerD = Nx.dot(myDeriv,Nx.transpose(w))
    {[newWeights|net],0,error,correct}
  end
  defn genNewWeights(weights,lr,layer,der) do
    weights + (lr*Nx.dot(Nx.transpose(layer),der))
  end
  def trainNN(1, input, nn,target,lr) do
    input1 = input[0..0]
    target1 = target[0..0]
   # IO.puts "saida do train"
    #IO.inspect(input1)
    #IO.inspect(target1)
    {net,wD,error,correct} = runNet(input1,nn,target1,lr)
    {net,error,correct}
  end
  def trainNN(n,input,nn,target,lr) do
    input1 = input[0..0]
    target1 = target[0..0]
    {net,wD,newError,correct} = runNet(input1,nn,target1,lr)
    if(Nx.to_number(correct)==1)do
     # IO.puts(n)

      #raise "end"
    end
    tinput  = input[1..-1//1] # Drop the first "row"
    ttarget = target[1..-1//1] # Drop the first "row"
    {finalNet,errorsofar,correctsofar} = trainNN(n-1,tinput,net,ttarget,lr)

    myError = Nx.add(newError,errorsofar)
    myCorrect = Nx.add(correct,correctsofar)# correct+correctsofar
    IO.inspect(myCorrect)
    {finalNet,myError,myCorrect}
  end
  defn mult(wD,output) do
        wD*output
  end

  defn sub(output,target) do
     output - target
  end
  defn sum(t1,t2) do
    t1 + t2
  end
  def newDenseLayer(x,y,type) do
   fitWeights(Nx.random_uniform({x, y}, 0.0, 1.0, type: {:f, 64}))
  end

  defn fitWeights(w) do
    0.2*w-0.1
  end
  def loop(1,ntrain,input,nn,target,lr) do
    {newnet,error}=trainNN(ntrain,input,nn,target,lr)
    {newnet,error}
  end
  def loop(n,ntrain, input,nn,target,lr) do
    {newnet,error}=trainNN(ntrain,input,nn,target,lr)
   # IO.puts "Error"
    #IO.inspect error
    r = loop(n-1,ntrain,input,newnet,target,lr)
    r
  end
end


#inputSize = 3
#hiddenSize = 4
#outputSize = 1
alpha = 0.005
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


#nn = [weights_0_1,weights_1_2]

nn2_ = [w0_,w1_]

sl_input = Nx.tensor([  [ 1, 0, 1],
                        [ 0, 1, 1],
                        [ 0, 0, 1],
                        [ 1, 1, 1] ])

sl_target = Nx.transpose(Nx.tensor([[1, 1, 0, 0]]))



#input1 = sl_input[0..0]

#target1 = sl_target[0..0]

#{newNet,d,errorFinal} = NN.runNet(input1,nn,target1,alpha)


#{newNet,errorFinal} = NN.trainNN(4,sl_input,nn2_,sl_target,alpha)

#{newNet,errorFinal} = NN.loop(10,4, sl_input,nn2_,sl_target,alpha)


### MNIST

labels = Nx.from_numpy("labelsMNIST.npy")
images = Nx.from_numpy("imagesMNIST.npy")

input1 = images[0..0]

target1 = labels[0..0]

#labels = Nx.as_type(labels,{:u, 8})
#images = Nx.as_type(images,{:u, 8})
inputSize = 784 #pixels per image
hiddenSize = 40
outputSize = 10
alpha = 0.005

w01 = Nx.from_numpy("w01.npy")
w12 = Nx.from_numpy("w12.npy")

nn = [w01,w12]
#nn = [NN.newDenseLayer(inputSize,hiddenSize,:relu),
#      NN.newDenseLayer(hiddenSize,outputSize,:relu)]


[w1_|tail_] = nn

#IO.puts "weights"
#IO.inspect w1
#{newNet,d,errorFinal,correct} = NN.runNet(input1,nn,target1,alpha)

time1 = Time.utc_now()

{newNet,errorFinal,correct} = NN.trainNN(1000,images,nn,labels,alpha)

time2 = Time.utc_now()

timef = Time.diff(time2,time1)

IO.inspect timef

IO.puts "Correct"
correct_ = Nx.to_number(correct)
IO.puts correct_
IO.puts "Error final:"
IO.inspect errorFinal

[head|tail_] = newNet

#[tt_] = tail_

#IO.inspect head

#IO.inspect tt_

#labels = Nx.from_numpy("labels.npy")
#images = Nx.from_numpy("images.npy")
