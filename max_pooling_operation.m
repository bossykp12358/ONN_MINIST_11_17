function [output_tensor] = max_pooling_operation(input_tensor)
%This function is an operation function that pools MINIST raw images to 7 * 7.
%This function is only applicable to this program.
%The reason for using silly loops to operate is that for some unknown reason, the built-in 'reshape' function reported an error.
%'input_tensor' is the tensor before pooling,'output_tensor' is the tensor
%after pooling. All other variables are intermediate auxiliary effects.

train=zeros(7,7,size(input_tensor,2));

output_tensor=zeros(49,size(input_tensor,2));

sq=sqrt(size(input_tensor,1));

tr=zeros(sq,sq,size(input_tensor,2));

for p=1:1:size(input_tensor,2)
    for q=1:1:sq
        tr(:,q,p)=input_tensor((q-1)*sq+1:q*sq,p);
    end
    train(:,:,p)=max_pooling(tr(:,:,p),4);
end

for p=1:1:size(train,3)
    for q=1:1:size(train,1)
        output_tensor((q-1)*size(train,1)+1:(q)*size(train,1),p)=train(q,:,p);
    end
end

end