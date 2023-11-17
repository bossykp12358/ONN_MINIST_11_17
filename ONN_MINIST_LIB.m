% % 清空环境
% clear; close all; clc;
% 
% %创建输入和目标数据
% 
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
% train_1=zeros(7,7,size(train_data,2));
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
%     train_1(:,:,p)=max_pooling(tr(:,:,p),4);
% 
% end
% 
% for p=1:1:size(train_1,3)
%     for q=1:1:size(train_1,1)
%         train_final((q-1)*size(train_1,1)+1:(q)*size(train_1,1),p)=train_1(q,:,p);
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
% 
% % 读取测试集标签
%  test_labels_file = 't10k-labels.idx1-ubyte';
% 
%  test_labels = loadMNISTLabels(test_labels_file);

X_train=train_final(:,1:2000);%变换为纵向一个码元

encodedData_train = dummyvar(train_labels+1);%独热编码

encodedData_test= dummyvar(test_labels+1);

X_test=test_final;%

Y_train=(encodedData_train');
Y_train=Y_train(:,1:2000)

Y_test=encodedData_test';

% 创建神经网络
% 创建一个具有49个输入神经元、49个隐层神经元和10个输出神经元的神经网络
net = feedforwardnet(49);

% 设置训练参数
net.trainParam.epochs = 100; % 最大训练轮次
net.trainParam.goal = 1e-2; % 训练目标误差
net.trainParam.lr = 0.001; % 学习率
net.numLayers = 5; 
net.performFcn='mse'; 


% 训练神经网络
[trained_net, tr] = train(net, X_train, Y_train);

% 测试神经网络
test_data = X_test;
test_output = trained_net(test_data);

% 将输出结果转换为二进制
binary_output = round(test_output);

% 显示结果
disp("测试数据：");
disp(test_data');
disp("测试输出：");
disp(binary_output);

% 可视化神经网络结构
view(trained_net);

% 可视化训练过程中的性能
figure;
plotperform(tr);
title('训练过程中的性能');

% 可视化训练过程中的梯度
figure;
plot(tr.gradient);
xlabel('迭代次数');
ylabel('梯度');
title('训练过程中的梯度');