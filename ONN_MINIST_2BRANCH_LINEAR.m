clear;close all;clc
%% Data reading and preprocessing
% Read training set data
train_data_file = 'train-images.idx3-ubyte';

train_data = loadMNISTImages(train_data_file);

% Read training set labels
train_labels_file = 'train-labels.idx1-ubyte';

train_labels = loadMNISTLabels(train_labels_file);

% Read Test Set Data
test_data_file = 't10k-images.idx3-ubyte';

test_data = loadMNISTImages(test_data_file);

% Read test set labels
test_labels_file = 't10k-labels.idx1-ubyte';

test_labels = loadMNISTLabels(test_labels_file);

n_train = size(train_data,2); % The number of the training signal

n_test = size(test_data,2); % The number of the test signal

%One Hot encoded
encodedData_train = dummyvar(train_labels+1);

encodedData_test= dummyvar(test_labels+1);

%Apply maximum pooling
X_train=max_pooling_operation(train_data);%Final input training set data

X_test=max_pooling_operation(test_data);%Final input test set data

Y_train=encodedData_train';%Training Set Label

Y_test=encodedData_test';%Test Set Label


%% Set hyperparameters related to the network

num_hidden_layers = 10;%Number of hidden layers

hidden_layer_size = 49;%Number of hidden layer neurons

epoch=500;%Total number of training sets

landa=0.001;%regularization parameter

batch_train=2048;%batch_size of Small Batch Gradient Descent Algorithm

iteration=floor(n_train/batch_train);%Number of iterations

num_input=size(X_train,1);%Size of input data

num_output=size(Y_train,1);%Size of output data

%Initialize weights and create structures that store various parameter matrices
weights = cell(num_hidden_layers+1, 1);%Structure for storing weights for each layer

A=cell(num_hidden_layers+2, 1);%A,B represents the intermediate values of neurons in the neural network, where B is the linear synthesis of A, and A {i+1} is the nonlinear activation of B {i}. In the linear network, A {i+1}=B {i}

B=cell(num_hidden_layers+1, 1);

%Initialize the first and last fully connected layers
%The first layer is an identity matrix, which directly hands over the 49 eigenvalues inputted each time to the corresponding 49 neurons of the neural network
%The last layer is a computer-assisted fully connected layer. To demonstrate the fitting ability of the middle layer, the edges will not change after initialization
weights{1} = eye(hidden_layer_size);

weights{num_hidden_layers+1} = rand(num_output, hidden_layer_size);


%Initialize the hidden layer in the middle
for i=2:num_hidden_layers        
    weights{i} = zeros(hidden_layer_size, hidden_layer_size);
    for j=1:hidden_layer_size        
        if j== 1            
            inds=j; 
            weight_values = rand(1,1);
            weights{i}(j, inds) = weight_values;
        else           
            inds = j-1:j;
            weight_values =rand(1,1);
            weights{i}(j, inds) = weight_values;
        end
    end
end

weights_init=weights;%Save an initialized weight matrix，control variables during multiple training sessions to select a hyperparameter later

%Matrix for storing legends, distinguish when drawing for subsequent multiple training sessions
legend_str=strings(1,20);

line_mod={'-o','-+','-*','-.','-x','-s','-d','-^','-v','->','-<','-p','-h'};

%% Start multiple training and predictions to determine the best hyperparameter. When all hyperparameters are determined, change the number of cycles to 1
%model_num represents to the number of cycles
for model_num = 1 : 1 : 1

%%Random learning rate
%Ran=unifrnd(-6,-1,[1,1]);

%learning_rate = 10^Ran;

learning_rate = 8e-4;%Determine a good learning rate of 8e-4

acc=[];%An array storing the accuracy of the test set after each iteration

cost_train=[];%An array that stores the loss values of the training set after each iteration

cost_test=[];%An array that stores the loss values of the test set after each iteration

%Optimization algorithm with reduced learning rate
decayRate=0.001;%Learning rate decay rate

%Using the Adam optimizer,'belta_1','belta_2','epc' are all the classical parameter when applying Adam optimizer 
belta_1=0.9;

belta_2=0.999;

epc=1e-8;

vdW=cell(num_hidden_layers+1, 1);%Moving Index Weighted Average

vdW{1} = zeros(hidden_layer_size,hidden_layer_size);

vdW{num_hidden_layers+1} = zeros(num_output, hidden_layer_size);

sdW=cell(num_hidden_layers+1, 1);%Second-order moment estimation

sdW{1} = zeros(hidden_layer_size,hidden_layer_size);

sdW{num_hidden_layers+1} = zeros(num_output, hidden_layer_size);

for i=2:num_hidden_layers    
    vdW{i} = zeros(hidden_layer_size, hidden_layer_size);
end

for i=2:num_hidden_layers    
    sdW{i} = zeros(hidden_layer_size, hidden_layer_size);
end


%权重初始化

weights=weights_init;

weights{num_hidden_layers+1}=weights_final_init;

for p=1:1:epoch

    %vdW=0*vdW;%每个epoch初始化
    
    lr=learning_rate*(1/(1+decayRate*p));

for iter = 1 : 1 : iteration    
    %前向传播循环
    A{1} = X_train(:,1+batch_train*(iter-1):batch_train*iter);

    for i=1:num_hidden_layers+1
        if i == num_hidden_layers+1
            B{i} = (weights{i} * A{i});
            A{i+1} = softmax((B{i}));%最后一层变为概率
        else
            B{i} = (weights{i} * A{i});
            A{i+1} =((B{i}));
        end     
    end
    
    Y_train_now=Y_train(:,1+batch_train*(iter-1):batch_train*iter);%当前正在测试的标签子集

    cost_train(iteration*(p-1)+iter)=0;
    
    %计算损失，概率损失采用交叉熵
    for m=1:1:batch_train
        for n = 1:1:size(A{i+1},1)            
            cost_train(iteration*(p-1)+iter)=cost_train(iteration*(p-1)+iter)-Y_train_now(n,m)*log(A{i+1}(n,m));
        end   
    end
    
    cost_train(iteration*(p-1)+iter)=cost_train(iteration*(p-1)+iter)/batch_train;
    
    %cost_train(iter)=mse(Y_train,A{i+1});
    
    %开始反向传播误差
    dCdB=(A{i+1}-Y_train_now); %求导链式，损失函数以及softmax函数对加权输出的求导
    %dCdA=(A{i+1}-Y_train);
    
    dBdW=A{i}'; %线性组合对权重的求导
    
    dCdWfinal=dCdB*dBdW/batch_train; %损失函数对最后一层连接层的求导
    
    dCdW=cell(num_hidden_layers+1, 1); %创建结构体，损失函数对权重的导数结构体
   
    dCdW{num_hidden_layers+1}=dCdWfinal;
    
    dCda=weights{num_hidden_layers+1}'*(dCdB);%损失函数对倒数第二层的求导，这个变量是更新的，随着循环的进行一直往前走
    
    %误差反向传播，按照求导链式法则，误差逐渐向前传递
    for i=num_hidden_layers:-1:2
        
        %dCdW{i}=dCda.*relu_derivative(B{i})*A{i}';%对于权重的导数
        
        %更新环路中间的各种参数
        for j=1:hidden_layer_size        
            if j== 1            
                inds=j; %下标 
                dCdW{i}(j, inds)=dCda(j,:).*1*A{i}(j,:)'/batch_train;%对于权重的导数
                %weights{i}(j, inds) = cos(xitas{i}(j, inds))*exp(1i*phy_lefts{i}(j, inds));

            else           
                inds = j-1:j; %下标
                dCdW{i}(j, j-1)=dCda(j,:).*1*A{i}(j-1,:)'/batch_train;%对于权重的导数
                dCdW{i}(j, j)=dCda(j,:).*1*A{i}(j,:)'/batch_train;%对于权重的导
                %weights{i}(j, inds(1)) = 1i*sin(xitas{i}(j, inds(1)))*exp(1i*phy_rights{i}(j, inds(1)))*exp(gamas{i}(j, inds(1)));
                %weights{i}(j, inds(2)) = cos(xitas{i}(j, inds(2)))*exp(1i*phy_lefts{i}(j, inds(2)));
            end
        end

        %继续往前推一层，dCda产生变化
        dCda=weights{i}'*dCda.*1;

        %dCdW{i}=0;
    end
    
    %误差传递到第一0u 层
    %dCdW{1}=dCda.*relu_derivative(B{1})*A{1}'/batch_train;
    
    dCdW{1}=0;%第一层单一映射不动
    
    %权重进行更新，

    for i=2:num_hidden_layers

        vdW{i}=belta_1*vdW{i}+(1-belta_1)* dCdW{i};

        sdW{i}=belta_2*sdW{i}+(1-belta_2)* (dCdW{i}).^2;
        
        %weights{i} = weights{i} - learning_rate * dCdW{i}-landa*learning_rate*(weights{i});

        weights{i} = weights{i} - lr * vdW{i}./(sqrt(sdW{i})+epc)-landa*lr*(weights{i});
        
    end
    

    %% 测试集测试
    B_T = X_test;%当前正在测试的数据和标签,测试的时候所以数据一并测试
    L_T=Y_test;
    %B_T=X_test;
    %L_T=Y_test;
    for i=1:num_hidden_layers
        T = weights{i} * B_T;
        B_T =(T);
    end
    T_final = weights{num_hidden_layers+1} * B_T ;%+ biases{num_hidden_layers+1};
    
    y_predict = softmax(T_final);
    
    cost_test(iteration*(p-1)+iter)=0;
    
    %计算损失，概率损失采用交叉熵
    for m=1:1:size(L_T,2)
        for n = 1:1:size(y_predict,1)
            cost_test(iteration*(p-1)+iter)=cost_test(iteration*(p-1)+iter)-L_T(n,m)*log(y_predict(n,m));
        end
    end
    cost_test(iteration*(p-1)+iter)=cost_test(iteration*(p-1)+iter)/size(L_T,2);

    %cost_test(iter)=mse(Y_test,y_predict);
    
    %判决
    num_true=0;
    for i=1:1:size(y_predict,2)

        y_predict(:,i)=max(floor(y_predict(:,i)-max(y_predict(:,i))+1),0);


         if y_predict(:,i)==L_T(:,i)
             num_true=num_true+1;
         end
    end
   
    acc(iteration*(p-1)+iter)=num_true/size(y_predict,2);% Test set recognition accuracy

    fprintf('当前为第%d次迭代', iteration*(p-1)+iter);
    fprintf('当前cost_train为%d\n', cost_train(iteration*(p-1)+iter));
    fprintf('当前cost_test为%d\n', cost_test(iteration*(p-1)+iter));
    fprintf('当前准确率acc = %d\n', acc(iteration*(p-1)+iter));
end
end

co=rand(1,3);
legend_str{model_num}=['学习率',num2str(learning_rate)];
%legend_str{model_num}=['隐层为线性层'];

figure(1)
plot(cost_train,line_mod{model_num},'Color',co)
xlabel('Iterations')
ylabel('Training Set Loss Value')
ylim([0 5])
title('模型训练集损失值变化')
hold on

figure(2)
plot(cost_test,line_mod{model_num},'Color',co)
xlabel('Iterations')
ylabel('Test Set Loss Value')
ylim([0 5])
title('模型测试集损失值变化')
hold on

figure(3)
plot(acc,line_mod{model_num},'Color',co)
xlabel('Iterations')
ylabel('Test Set Accuracy')
ylim([0 1])
title('模型测试集准确率变化')
hold on
 
end

legend(legend_str,'Location','EastOutside');