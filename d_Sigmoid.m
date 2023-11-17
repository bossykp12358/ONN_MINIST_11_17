function y = d_Sigmod(x) %非线性激励relu函数
y = (exp(-x))./((1+exp(-x)).^2);
end