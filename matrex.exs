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
  #import Nx.Defn
 # Nx.default_backend(Torchx)
  def relu(x) do
    Matrex.apply(x,fn(n)-> if (n>0) do n else 0 end end)
  end
  def relu2deriv(output) do
    Matrex.apply(output,fn(n)-> if (n>0) do 1 else 0 end end)
    #Nx.greater(output,0)
  end
  def runNet(input,[weights_l],target,lr) do
    #IO.inspect(weights_l)
    #raise "hell"
    #o = dotPSize(5,input,weights_l)
    o = Matrex.dot(input,weights_l)
    finalDerivative = Matrex.subtract(target,o)
    error =  Matrex.sum(Matrex.apply(finalDerivative,fn(n)-> n**2 end))
    correct = if (Matrex.argmax(o)==Matrex.argmax(target)) do 1 else 0 end
    newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    nextLayerD = Matrex.dot(finalDerivative,Matrex.transpose(weights_l))
    {[newWeights],nextLayerD,error,correct}
  end
  def runNet(input,[w|tl],target,lr) do
    #o =  relu(Matrex.dot(input,w))
    #IO.inspect(w)
    #raise "hell"
    o =  relu(dotPSize(1440,input,w))
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
  def dotPSize(size,vet,matrix) do
     list_ = parallelDotSize2(size,vet,matrix)
  #  #raise "ok"
     list__ = List.flatten(list_)
     #IO.inspect(list1)
     #raise "list_: #{length list_} list1: #{length list1}"
     list1 = Enum.map(list__,&Task.await/1)
     list2 = List.flatten(list1)
     #IO.inspect list2
     #raise "list2 size: #{length(list2)}"
     listf = Matrex.concat(list2)#Enum.map(list2,fn(n) ->  Matrex.at(n,1,1) end)
     listf
     #Matrex.new([listf])
  end
  def parallelDotSize2(n,vet,matrix) do
    {nl,nc}=Matrex.size(matrix)
    if (nc>n) do
      subMatrix = Matrex.submatrix(matrix,1..nl,1..n)
      #IO.inspect list
      #raise "hell"
      #if (length(list) != 5) do raise "size" end
      task = Task.async(fn -> Matrex.dot(vet,subMatrix) end)
      restMatrix = Matrex.submatrix(matrix,1..nl,(n+1)..nc)
      tasks = parallelDotSize2(n,vet,restMatrix)
      [ task, tasks]
    else
      #list = getColumns(matrix,nc)
      #raise "fuck"
      task = Task.async(fn -> Matrex.dot(vet,matrix) end)
      #restMatrix = Matrex.submatrix(matrix,1..nl,(n+1)..nc)
      #tasks = parallelDotSize(n,vet,restMatrix)
      [task]
    end
  end
  def parallelDotSize(n,vet,matrix) do
    {nl,nc}=Matrex.size(matrix)
    if (nc>n) do
      list = getColumns(matrix,n)
      #IO.inspect list
      #raise "hell"
      #if (length(list) != 5) do raise "size" end
      task = Task.async(fn -> Enum.map(list, fn(col) -> Matrex.dot(vet,col) end) end)
      restMatrix = Matrex.submatrix(matrix,1..nl,(n+1)..nc)
      tasks = parallelDotSize(n,vet,restMatrix)
      [ task, tasks]
    else
      list = getColumns(matrix,nc)
      #raise "fuck"
      task = Task.async(fn -> Enum.map(list, fn(col) -> Matrex.dot(vet,col) end) end)
      #restMatrix = Matrex.submatrix(matrix,1..nl,(n+1)..nc)
      #tasks = parallelDotSize(n,vet,restMatrix)
      [task]
    end
  end
  def getColumns(matrix,1)do
    col = Matrex.column(matrix,1)
    [col]
  end
  def getColumns(matrix,ncol)do
    col = Matrex.column(matrix,1)
    {nl,nc}=Matrex.size(matrix)
    restMatrix = Matrex.submatrix(matrix,1..nl,2..nc)
    cols =getColumns(restMatrix,ncol-1)
    [col|cols]
  end
  def run_batch(size,input,weights,target,lr) do
    tasks = slice_entries(size,input,target,weights,lr)
    #tasks = Enum.map(list, fn({i1,t1}) -> Task.async(fn -> NN.runNet(i1,weights,t1,lr)end) end)
    #results = Enum.map(tasks,&Task.await/1)
    [hr|tr] = tasks
    r1 = Task.await(hr)
    List.foldr(tr, r1, fn(task,{nn2,wd2,erro2,acc2}) -> {nn1,wd1,erro1,acc1} = Task.await task
                                                        {sumNNs(nn1,nn2),wd1,erro1+erro2,acc1+acc2} end)
  end
  def sumNNs([w1],[w2]) do
    w3 =Matrex.add(w1,w2)
    [w3]
  end
  def sumNNs([w1|t1],[w2|t2]) do
    w3 =Matrex.add(w1,w2)
    rest = sumNNs(t1,t2)
    [w3|rest]

  end
  def slice_entries(1,input,target,weights,lr)do
    input1 = Matrex.row(input,1)
    target1 = Matrex.row(target,1)
    task =Task.async(fn -> NN.runNet(input1,weights,target1,lr)end)
    [task]
  end
  def slice_entries(n,input,target,weights,lr)do
    input1 = Matrex.row(input,1)
    target1 = Matrex.row(target,1)
    task =Task.async(fn -> NN.runNet(input1,weights,target1,lr)end)
    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    restinput = Matrex.submatrix(input,2..il,1..ic)
    resttarget = Matrex.submatrix(target,2..tl,1..tc)
    slices = slice_entries(n-1,restinput,resttarget,weights,lr)
    [task|slices]
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


#w0_ = Matrex.load("w01.csv")
#w1_ = Matrex.load("w12.csv")
#nn = [w0_,w1_]

inputSize = 784 #pixels per image
hiddenSize = 180#40#720#360#40#360#180#40
outputSize = 10
alpha = 0.005
nn = [NN.newDenseLayer(inputSize,hiddenSize,:relu),
      NN.newDenseLayer(hiddenSize,outputSize,:relu)]

#[wh|wt] =nn

#IO.inspect(wh)

#nn = [weights_0_1,weights_1_2]


#IO.inspect(w0_)
#IO.inspect(w1_)


sl_input = Matrex.new([  [ 1, 0, 1],
                        [ 0, 1, 1],
                        [ 0, 0, 1],
                        [ 1, 1, 1] ])

sl_target = Matrex.transpose(Matrex.new([[1, 1, 0, 0]]))



images = Matrex.load("imgMNIST.csv")
labels = Matrex.load("tarMNIST.csv")

input1 = images[1]
target1 = labels[1]
#r = NN.dotPSize(5,input1,wh)

#IO.inspect input1
#IO.inspect wh
#IO.inspect r
#raise "ok"
#IO.inspect(input1)
#IO.inspect(target1)
#raise "o"

#{newNet,d,errorFinal,correct} = NN.runNet(input1,nn2,target1,alpha)
#time1 = Time.utc_now()
#{newNet,errorFinal,correct} = NN.trainNN(1000,images,nn,labels,alpha)

#time1 = Time.utc_now()
defmodule Bench do
  def execNN(1,input1,nn,target1,alpha) do
    #task1 = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    #task2 = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    #task3 = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    NN.runNet(input1,nn,target1,alpha)
    #Task.await(task1)
    #Task.await(task2)
    #Task.await(task3)
  end
  def execNN(n,input1,nn,target1,alpha) do
    NN.runNet(input1,nn,target1,alpha)
    execNN(n-1,input1,nn,target1,alpha)
  end
  def execNNp(1,l,input1,nn,target1,alpha) do
    task = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    Enum.map([task|l],&Task.await/1)
    #Task.await(task1)
    #Task.await(task2)
    #Task.await(task3)
  end
  def execNNp(n,l,input1,nn,target1,alpha) do
    task = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    execNNp(n-1,[task|l],input1,nn,target1,alpha)
  end
end

#{time,r} = :timer.tc(&Bench.execNNp/6,[100,[],input1,nn,target1,alpha])
#{time,r} = :timer.tc(&Bench.execNN/5,[100,input1,nn,target1,alpha])
#IO.puts ("time: #{(time)/(1_000_000)}")



#{time,{newNet,errorFinal,correct} } = :timer.tc(&NN.loop/6,[1,1000,images,nn,labels,alpha])


#{time,r}=:timer.tc(&NN.run_batch/5,[1000,images,nn,labels,alpha])

{time,{newnet,error,correct}}=:timer.tc(&NN.trainNN/5,[1000,images,nn,labels,alpha])
IO.puts ("time: #{time/(1_000_000)}")
#timef = Time.diff(time2,time1)






#IO.inspect timef
#IO.puts("error")
#IO.inspect errorFinal
#IO.puts("Acc")
#IO.inspect(correct)
