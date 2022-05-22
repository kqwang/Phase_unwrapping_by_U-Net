close all
clear all
clc

path1='.\Results_real\';  % path of the testing results
imgDir1  = dir([path1 '*.mat']);  % get dir of the results
n=length(imgDir1); % get size of the testing data

for j = 1:n     
load([path1 imgDir1(j).name]); % read the results
[size_xy size_xy]= size(input);
rmses(j)=sqrt(sum(sum((output-gt).^2))/(size_xy*size_xy)); % get RMSE for each result
j/n
end

rmse_mean=sum(rmses)/length(rmses); % get mean of RMSE

% get standard deviation of RMSE
sum_sd=0;
for ii =1 : length(rmses)
sum_sd  = sum_sd +  (rmses(ii)-rmse_mean)^2
end
% show scatter of RMSEs
xx = 1 : length(rmses);
figure
scatter(xx,rmses);

% show histogram of RMSEs
figure
histogram(rmses,20);