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

%进行学习率的对比训练

%读取测试集标签
test_labels_file = 't10k-labels.idx1-ubyte';
 
test_labels = loadMNISTLabels(test_labels_file);

n_train = size(train_data,2); % 训练信号长度

n_test = size(test_data,2); % 测试信号长度

X_train=train_final;%变换为纵向一个码元

encodedData_train = dummyvar(train_labels+1);%独热编码

encodedData_test= dummyvar(test_labels+1);

X_test=test_final;%

Y_train=encodedData_train';

Y_test=encodedData_test';


%设置跟网络有关的参数和超参数
num_hidden_layers = 25;%隐层数量

hidden_layer_size = 49;%隐层神经元个数

epoch=100;

batch_train=2048;%每一批的训练数

%batch_test=256;%每一批的训练数

iteration=floor(n_train/batch_train);%迭代的次数

num_input=size(X_train,1);

num_output=size(Y_train,1);

% 初始化权重,创建存储各个参数矩阵的结构体
weights = cell(num_hidden_layers+1, 1);

A=cell(num_hidden_layers+2, 1);%AB分别表示神经网络中的神经元中间值，其中B是A经过线性合成后的，而A{i+1}是B{i}经过非线性激活后的

B=cell(num_hidden_layers+1, 1);

% 初始化第一个和最后一个全连接层
weights{1} = eye(hidden_layer_size);

weights{num_hidden_layers+1} = rand(num_output, hidden_layer_size);


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

%weights_init=weights;%保存一个初始化的权重矩阵

%存放图例的矩阵
legend_str=strings(1,20);

line_mod={'-o','-+','-*','-.','-x','-s','-d','-^','-v','->','-<','-p','-h'};

landa_list=[1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001];

for model_num = 1 : 1 : 11


%随机取学习率
Ran=unifrnd(-4,0,[1,1]);

learning_rate = 5e-4;%学习率

landa=landa_list(model_num); %正则化参数

%learning_rate = 10^Ran;%学习率

%%
%信号按批划分
acc=[];%存放测试集的准确率数组
cost_train=[];%存放训练集的损失值数组
cost_test=[];%存放测试集的损失值数组
%%
%训练过程

%采用学习率衰减的优化方式
decayRate=0.001;%学习率衰减速率

%采用Adam梯度下降优化法
belta_1=0.9;%移动指数加权平均系数
belta_2=0.999;%方均根系数
epc=1e-8;%方均根分母中加入的系数

vdW=cell(num_hidden_layers+1, 1);%移动指数加权平均

vdW{1} = zeros(hidden_layer_size,hidden_layer_size);

vdW{num_hidden_layers+1} = zeros(num_output, hidden_layer_size);

sdW=cell(num_hidden_layers+1, 1);%方均根系数

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

for p=1:1:epoch

    %vdW=0*vdW;%每个epoch初始化
    
    lr=learning_rate*(1/(1+decayRate*p));

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
                dCdW{i}(j, inds)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j,:)'/batch_train;%对于权重的导数
                %weights{i}(j, inds) = cos(xitas{i}(j, inds))*exp(1i*phy_lefts{i}(j, inds));

            else           
                inds = j-1:j; %下标
                dCdW{i}(j, j-1)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j-1,:)'/batch_train;%对于权重的导数
                dCdW{i}(j, j)=dCda(j,:).*relu_derivative(B{i}(j))*A{i}(j,:)'/batch_train;%对于权重的导
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
    
    dCdW{1}=0;%第一层单一映射不动
    
    %权重进行更新，

    for i=2:num_hidden_layers

        vdW{i}=belta_1*vdW{i}+(1-belta_1)* dCdW{i};

        sdW{i}=belta_2*sdW{i}+(1-belta_2)* (dCdW{i}).^2;

        vdW{i}=vdW{i}/(1-belta_1^(p-1)*(interation)+iter);       %修正一阶矩

        sdW{i}=sdW{i}/(1-belta_1^(p-1)*(interation)+iter);       %修正二阶矩
        
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
        B_T = relu(T);
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
   
    acc(iteration*(p-1)+iter)=num_true/size(y_predict,2);
    % 输出损失和进度
    fprintf('当前为第%d次迭代', iteration*(p-1)+iter);
    fprintf('当前cost_train为%d\n', cost_train(iteration*(p-1)+iter));
    fprintf('当前cost_test为%d\n', cost_test(iteration*(p-1)+iter));
    fprintf('当前准确率acc = %d\n', acc(iteration*(p-1)+iter));
end
end

co=rand(1,3);
legend_str{model_num}=['正则化参数',num2str(landa)];

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