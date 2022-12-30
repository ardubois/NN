Mix.install([
  {:nx, "~> 0.4.0"},
  {:torchx, "~> 0.4.0"},
  {:exla, "~> 0.4"},
  {:matrex, "~> 0.6"}
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
  def relu(x) do
    Matrex.apply(x,fn(n)-> if (n>0) do n else 0 end end)
  end
  def relu2deriv(output) do
    Matrex.apply(output,fn(n)-> if (n>0) do 1 else 0 end end)
    #Nx.greater(output,0)
  end
  def runNet(input,[weights_l],target,lr) do
    o = Matrex.dot(input,weights_l)
    finalDerivative = Matrex.subtract(target,o)
    error =  Matrex.sum(Matrex.apply(finalDerivative,fn(n)-> n**2 end))
    correct = if (Matrex.argmax(o)==Matrex.argmax(target)) do 1 else 0 end
    newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    nextLayerD = Matrex.dot(finalDerivative,Matrex.transpose(weights_l))
    {[newWeights],nextLayerD,error,correct}
  end
  def runNet(input,[w|tl],target,lr) do
    o =  relu(Matrex.dot(input,w))
    #IO.inspect o
    #raise "oi"
    {net,wD,error,correct} = runNet(o,tl,target,lr)
    myDeriv = Matrex.multiply(wD,relu2deriv(o))
    #IO.inspect(relu2deriv(o))
    newWeights = genNewWeights(w,lr,input,myDeriv)
    #IO.inspect(Nx.sum(newWeights))
    #nextLayerD = Nx.dot(myDeriv,Nx.transpose(w))
    {[newWeights|net],0,error,correct}
  end
  def genNewWeights(weights,lr,layer,der) do
    Matrex.add(weights,Matrex.multiply(lr,Matrex.dot(Matrex.transpose(layer),der)))
  end
  def trainNN(1, input, nn,target,lr) do
    input1 = Matrex.row(input,1)
    target1 = Matrex.row(target,1)
    {net,wD,error,correct} = runNet(input1,nn,target1,lr)
    {net,error,correct}
  end
  def trainNN(n,input,nn,target,lr) do
    input1 = Matrex.row(input,1)
    target1 = Matrex.row(target,1)
    {net,wD,newError,correct} = runNet(input1,nn,target1,lr)
    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    if (il == 2) do
        tinput  = Matrex.row(input,2) # Drop the first "row"
        ttarget = Matrex.row(target,2)
        {finalNet,errorsofar,correctsofar} = trainNN(n-1,tinput,net,ttarget,lr)
        myError = newError+errorsofar
        myCorrect = correct+correctsofar# correct+correctsofar
        {finalNet,myError,myCorrect}
    else
        tinput  = input[2..il] # Drop the first "row"
        ttarget = target[2..tl] # Drop the first "row"
        {finalNet,errorsofar,correctsofar} = trainNN(n-1,tinput,net,ttarget,lr)
        myError = newError+errorsofar
        myCorrect = correct+correctsofar# correct+correctsofar
        {finalNet,myError,myCorrect}
    end



  end

  def newDenseLayer(x,y,type) do
   fitWeights(Matrex.random(x, y))
  end

  def fitWeights(w) do
    Matrex.multiply(0.02,Matrex.subtract(w,0.1))
  end
  def loop(1,ntrain,input,nn,target,lr) do
    {newnet,error,correct}=trainNN(ntrain,input,nn,target,lr)
    IO.puts("I #{1} error: #{error/ntrain} Acc: #{correct/ntrain}")
    {newnet,error,correct}
  end
  def loop(n,ntrain, input,nn,target,lr) do
    {newnet,error,correct}=trainNN(ntrain,input,nn,target,lr)
   # IO.puts "Error"
    #IO.inspect error
    IO.puts("I #{n} error: #{Nx.to_number(error)/ntrain} Acc: #{correct/ntrain}")
    r = loop(n-1,ntrain,input,newnet,target,lr)
    r
  end
  def dotP(vet,matrix) do
    #{r_,c} = Nx.shape(matrix)
    {r_,c}=Matrex.size(matrix)
  #  #IO.inspect(vet)
  #  #IO.inspect(matrix)
  #  #raise "ok"
     list1 = parallelDot(c,vet,matrix)
  #  #raise "ok"
     list2 = Enum.map(list1,&Task.await/1)
     listf = Enum.map(list2,fn(n) ->  Matrex.at(n,1,1) end)
     Matrex.new([listf])
  end
  def parallelDot(1,vet,matrix) do
    col = Matrex.column(matrix,1)
    task = Task.async(fn -> Matrex.dot(vet,col) end)
    #raise "ok"
    [task]
  end
  def parallelDot(n,vet,matrix) do
    col = Matrex.column(matrix,1)
    {nl,nc}=Matrex.size(matrix)
    restMatrix = Matrex.submatrix(matrix,1..nl,2..nc)
    task = Task.async(fn -> Matrex.dot(vet,col) end)
    tasks = parallelDot(n-1,vet,restMatrix)
    [task|tasks]
  end
end


#inputSize = 3
#hiddenSize = 4
#outputSize = 1
alpha = 0.005
#weights_0_1 = Nx.tensor ( [[-0.16595599,  0.40763847, -0.99977125],
#                            [-0.39533485, -0.70648822, -0.81532281],
#                            [-0.62747958 ,-0.34188906 ,-0.20646505]]) #NN.newDenseLayer(inputSize,hiddenSize,:relu)
#

#weights_1_2 = Nx.tensor([[ 0.07763347],
#                          [-0.16161097],
#                          [ 0.370439  ]])#NN.newDenseLayer(hiddenSize,outputSize,:relu)


w0_ = Matrex.load("w01.csv")
w1_ = Matrex.load("w12.csv")
#nn = [w0_,w1_]

inputSize = 784 #pixels per image
hiddenSize = 40#180#40
outputSize = 10
alpha = 0.005
nn = [NN.newDenseLayer(inputSize,hiddenSize,:relu),
      NN.newDenseLayer(hiddenSize,outputSize,:relu)]

[wh|wt] =nn

IO.inspect(wh)

#nn = [weights_0_1,weights_1_2]


IO.inspect(w0_)
IO.inspect(w1_)


sl_input = Matrex.new([  [ 1, 0, 1],
                        [ 0, 1, 1],
                        [ 0, 0, 1],
                        [ 1, 1, 1] ])

sl_target = Matrex.transpose(Matrex.new([[1, 1, 0, 0]]))



images = Matrex.load("imgMNIST.csv")
labels = Matrex.load("tarMNIST.csv")

#IO.inspect(input1)
#IO.inspect(target1)
#raise "o"

#{newNet,d,errorFinal,correct} = NN.runNet(input1,nn2,target1,alpha)
#time1 = Time.utc_now()
#{newNet,errorFinal,correct} = NN.trainNN(1000,images,nn,labels,alpha)

#{newNet,errorFinal,correct} = NN.loop(100,1000,images,nn,labels,alpha)

{time,{newNet,errorFinal,correct} } = :timer.tc(&NN.loop/6,[100,1000,images,nn,labels,alpha])

IO.puts ("time: #{time/(1_000_000)}")

#timef = Time.diff(time2,time1)

#IO.inspect timef
#IO.puts("error")
#IO.inspect errorFinal
#IO.puts("Acc")
#IO.inspect(correct)
