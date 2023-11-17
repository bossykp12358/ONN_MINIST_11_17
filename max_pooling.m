function [dst_img] = max_pooling(img,win_size)
% Perform a global maximum pooling operation on the image 'img', with a
% default window of square and a side length of 'win_size'

% Returning pooled images
fun = @(block_struct) max(block_struct.data(:));
X=win_size; Y=win_size; %window sizes
dst_img = blockproc (img, [X Y], fun);
end
