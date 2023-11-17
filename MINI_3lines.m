% clear;close all;clc
% %% 设置训练集
% % 读取训练集数据
% train_data_file = 'train-images.idx3-ubyte';
% 
% train_data = loadMNISTImages(train_data_file);
% 
% % 读取训练集标签
% train_labels_file = 'train-labels.idx1-ubyte';
% 
% train_labels = loadMNISTLabels(train_labels_file);
% 
% %训练集池化操作
% train=zeros(7,7,size(train_data,2));
% 
% train_final=zeros(49,size(train_data,2));
% 
% sq=sqrt(size(train_data,1));
% 
% tr=zeros(sq,sq,size(train_data,2));
% 
% for p=1:1:size(train_data,2)
%     for q=1:1:sq
%         tr(:,q,p)=train_data((q-1)*sq+1:q*sq,p);
%     end
% 
%     train(:,:,p)=max_pooling(tr(:,:,p),4);
% 
% end
% 
% for p=1:1:size(train,3)
%     for q=1:1:size(train,1)
%         train_final((q-1)*size(train,1)+1:(q)*size(train,1),p)=train(q,:,p);
%     end
% end
% 
% 
% % 读取测试集数据
% test_data_file = 't10k-images.idx3-ubyte';
% 
% test_data = loadMNISTImages(test_data_file);
% 
% %测试集池化操作
% test=zeros(7,7,size(test_data,2));
% 
% test_final=zeros(49,size(test_data,2));
% 
% sq=sqrt(size(test_data,1));
% 
% te=zeros(sq,sq,size(test_data,2));
% 
% for p=1:1:size(test_data,2)
%     for q=1:1:sq
%         te(:,q,p)=test_data((q-1)*sq+1:q*sq,p);
%     end
% 
%     test(:,:,p)=max_pooling(te(:,:,p),4);
% 
% end
% 
% for p=1:1:size(test,3)
%     for q=1:1:size(test,1)
%         test_final((q-1)*size(test,1)+1:(q)*size(test,1),p)=test(q,:,p);
%     end
% end

%读取测试集标签
 test_labels_file = 't10k-labels.idx1-ubyte';
 
 test_labels = loadMNISTLabels(test_labels_file);

n_train = size(train_data,2); % 训练信号长度

n_test = size(test_data,2); % 测试信号长度

X_train=train_final;%变换为纵向一个码元

encodedData_train = dummyvar(train_labels+1);%独热编码

encodedData_test= dummyvar(test_labels+1);

%X_te=signal_test;%测试集的输入信号和标签

X_test=test_final;%

% for i=1:1:n_train
%     X_train(i,:)=X_tr(10*(i-1)+1:10*i);
%     X_test(i,:)=X_te(10*(i-1)+1:10*i);
% end

%X_train=X_train';

%X_test=X_test';

Y_train=encodedData_train';
Y_test=encodedData_test';


%Y_train=train_labels';
%Y_test=test_labels';

%设置一系列参数和超参数
num_hidden_layers = 20;%隐层数量

hidden_layer_size = 49;%隐层神经元个数

learning_rate = 3e-3;%学习率

%%
%信号按批划分
epoch=100;

landa=0.02; %正则化参数

batch_train=2048;%每一批的训练数

batch_test=256;%每一批的训练数

iteration=floor(n_train/batch_train);%迭代的次数

num_input=size(X_train,1);

num_output=size(Y_train,1);

%%

% 初始化权重,创建存储各个参数矩阵的结构体
weights = cell(num_hidden_layers+1, 1);

A=cell(num_hidden_layers+2, 1);%AB分别表示神经网络中的神经元中间值，其中B是A经过线性合成后的，而A{i+1}是B{i}经过非线性激活后的

B=cell(num_hidden_layers+1, 1);

% 初始化第一个和最后一个全连接层
weights{1} = eye(hidden_layer_size);

weights{num_hidden_layers+1} = rand(num_output, hidden_layer_size);

%X_train=reshape(X_train,28,28,60000);

% 初始化中间的隐藏层
for i=2:num_hidden_layers    
    weights{i} = zeros(hidden_layer_size, hidden_layer_size);
    
    for j=1:hidden_layer_size        
        if j== 1            
            inds=j; %下标 
            weight_values = rand(1,1);
            weights{i}(j, inds) = weight_values;
            
        else           
            inds = j-1:j; %下标
            weight_values =rand(1,1);
            weights{i}(j, inds) = weight_values;
        end
    end
end


acc=[];%存放测试集的准确率数组
cost_train=[];%存放训练集的损失值数组
cost_test=[];%存放测试集的损失值数组
%%
%训练过程

%采用学习率衰减的优化方式
decayRate=0;%学习率衰减速率

%采用Momentum梯度下降法
belta=0.9;%移动指数加权平均参数

vdW=cell(num_hidden_layers+1, 1);%移动指数加权平均

vdW{1} = zeros(hidden_layer_size,hidden_layer_size);

vdW{num_hidden_layers+1} = zeros(num_output, hidden_layer_size);

for i=2:num_hidden_layers
    
    vdW{i} = zeros(hidden_layer_size, hidden_layer_size);

end

for p=1:1:epoch

    %vdW=0*vdW;%每个epoch初始化
    
    learning_rate=learning_rate*(1/(1+decayRate*p));

for iter = 1 : 1 : iteration    
    %前向传播循环
    A{1} = X_train(:,1+batch_train*(iter-1):batch_train*iter);
    %A{1}=X_train;
    for i=1:num_hidden_layers+1
        if i == num_hidden_layers+1
            B{i} = (weights{i} * A{i});
            A{i+1} = softmax((B{i}));%最后一层变为概率
        else
            B{i} = (weights{i} * A{i});
            A{i+1} = relu((B{i}));
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
    
    %dAdB=relu_derivative(B{i}); %非线性激活函数前后的求导
    %dAdB=d_Sigmod(B{i}); %非线性激活函数前后的求导
    
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
                inds=j:j+1; %下标 
                dCdW{i}(j, j)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j,:)';%对于权重的导数
                dCdW{i}(j, j+1)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j+1,:)';%对于权重的导数
                %weights{i}(j, inds) = cos(xitas{i}(j, inds))*exp(1i*phy_lefts{i}(j, inds));
            elseif j== hidden_layer_size             
                inds=j-1:j; %下标 
                dCdW{i}(j, j)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j,:)';%对于权重的导数
                dCdW{i}(j, j-1)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j-1,:)';%对于权重的导数
                %weights{i}(j, inds) = cos(xitas{i}(j, inds))*exp(1i*phy_lefts{i}(j, inds));

            else           
                inds = j-1:j+1; %下标
                dCdW{i}(j, j-1)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j-1,:)';%对于权重的导数
                dCdW{i}(j, j)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j,:)';%对于权重的导
                dCdW{i}(j, j+1)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j+1,:)';%对于权重的导数
                %weights{i}(j, inds(1)) = 1i*sin(xitas{i}(j, inds(1)))*exp(1i*phy_rights{i}(j, inds(1)))*exp(gamas{i}(j, inds(1)));
                %weights{i}(j, inds(2)) = cos(xitas{i}(j, inds(2)))*exp(1i*phy_lefts{i}(j, inds(2)));
            end
        end

        %继续往前推一层，dCda产生变化
        dCda=weights{i}'*dCda.*relu_derivative(B{i});

        %dCdW{i}=0;
    end
    
    %误差传递到第一层
    %dCdW{1}=dCda.*relu_derivative(B{1})*A{1}'/batch_train;
    
    dCdW{1}=0;
    
    %权重进行更新，

    for i=2:num_hidden_layers+1

        vdW{i}=belta*vdW{i}+(1-belta)* dCdW{i};
        
        %weights{i} = weights{i} - learning_rate * dCdW{i}-landa*learning_rate*(weights{i});

        weights{i} = weights{i} - learning_rate * vdW{i}-landa*learning_rate*(weights{i});
        
    end
    

    %% 测试集测试
    B_T = X_test(:,1+batch_test*(iter-1):batch_test*iter);%当前正在测试的数据和标签
    L_T=Y_test(:,1+batch_test*(iter-1):batch_test*iter);
    %B_T=X_test;
    %L_T=Y_test;
    for i=1:num_hidden_layers
        T = weights{i} * B_T;
        B_T = relu(T);
    end
    T_final = weights{num_hidden_layers+1} * B_T ;%+ biases{num_hidden_layers+1};
    
    y_predict = softmax(T_final);
    
    cost_test(iteration*(p-1)+iter)=0;
    
    %计算损失，概率损失采用交叉熵
    for m=1:1:batch_test
        for n = 1:1:size(y_predict,1)
            cost_test(iteration*(p-1)+iter)=cost_test(iteration*(p-1)+iter)-L_T(n,m)*log(y_predict(n,m));
        end
    end
    cost_test(iteration*(p-1)+iter)=cost_test(iteration*(p-1)+iter)/batch_test;

    %cost_test(iter)=mse(Y_test,y_predict);
    
    %判决
    num_true=0;
    for i=1:1:size(y_predict,2)

        y_predict(:,i)=max(floor(y_predict(:,i)-max(y_predict(:,i))+1),0);


         if y_predict(:,i)==L_T(:,i)
             num_true=num_true+1;
         end
    end
   
    acc(iteration*(p-1)+iter)=num_true/size(y_predict,2);
    % 输出损失和进度
    fprintf('当前为第%d次迭代', iteration*(p-1)+iter);
    fprintf('当前cost_train为%d\n', cost_train(iteration*(p-1)+iter));
    fprintf('当前cost_test为%d\n', cost_test(iteration*(p-1)+iter));
    fprintf('当前准确率acc = %d\n', acc(iteration*(p-1)+iter));
end
end
figure (1);
plot(acc)
xlabel('Iterations')
ylabel('Test Set Accuracy')
%title('测试集准确率变化')

figure (2);
plot(cost_train)
xlabel('Iterations')
ylabel('Training Set Loss Value')
%title('训练集损失值变化')

figure (3);
plot(cost_test)
xlabel('Iterations')
ylabel('Test Set Loss Value')
%title('测试集损失值变化')