% This is the code demo for the random matrix enlargement (RME) phase dataset generation method
% Parameters can be adjusted according to your actual needs

% clear and close
clear all
close all
clc

% path for dataset saving
path1='.\train_in\'; % path for the training dataset input
path2='.\train_gt\'; % path for the training dataset ground truth
path3='.\test_in\'; % path for the testing dataset input
path4='.\test_gt\'; % path for the testing dataset ground truth

% path generation
if ~exist(path1,'dir')
    mkdir(path1)
end
if ~exist(path2,'dir')
    mkdir(path2)
end
if ~exist(path3,'dir')
    mkdir(path3)
end
if ~exist(path4,'dir')
    mkdir(path4)
end

% parameters
size_min = 2; % minimum size of initial matrix
size_max = 8; % maximum size of initial matrix
height_min = 10; % minimum value of height h
height_max = 40; % maximum value of height h
h_h = height_max - height_min;  
train_num = 20000; % data number of training dataset
test_num = 2000; % data number of testing dataset
phase_size = 128; % size of the wrapped and absolute phase
noise_max = 0; % maximum value of the standard deviation of the noise

% generate and save training dataset
for jj = 1:train_num
    fprintf(  ['progress of training dataset generation = ' num2str(jj/train_num,'%4.6f') '\n']); % progress of training dataset generation
    
    % initial absolute phase
    size_xy = randi([size_min,size_max]); % get size of initial matrix
    initial_matrix = rand(size_xy,size_xy); % get initial matrix
    phase_size_pre = phase_size * 1.25; % get size of pre-enlarged absolute phase
    pre_enlarged_ap = imresize(initial_matrix,...
        [phase_size_pre phase_size_pre]);  % get pre-enlarged absolute phase
    initial_ap = pre_enlarged_ap((phase_size_pre/2-phase_size/2+1):...
        (phase_size_pre/2+phase_size/2),(phase_size_pre/2-phase_size/2+1)...
        :(phase_size_pre/2+phase_size/2)); % get initial absolute phase
            
    % set height h
    if jj <= train_num*0.5
        h(jj) = unifrnd(height_min,(height_min+2/3*h_h), 1, 1);
    else if jj <= train_num*0.7
            h(jj) = unifrnd((height_min+2/3*h_h),(height_min+5/6*h_h), 1, 1);
        else
            h(jj) = unifrnd((height_min+5/6*h_h),(height_min+h_h), 1, 1);
        end
    end
    
    % normalize the absolute phase to [0,h]
    ymax=h(jj);ymin=0;
    xmax = max(max(initial_ap));
    xmin = min(min(initial_ap));
    ap = (ymax-ymin)*(initial_ap-xmin)/(xmax-xmin) + ymin;

    % add noise to absolute phase
    sd_noise(jj)=unifrnd (0, noise_max, 1, 1);
    Gauss_RP = randn(128,128)*sd_noise(jj);
    ap_N = ap + Gauss_RP;
    
    % get wrapped phase
    R = real(exp(1i*ap_N));
    I = imag(exp(1i*ap_N));
    wp = atan2(I,R);
   
    % save wrapped phase as input and absolute phase as ground truth
    input = single(wp);
    save([path1 num2str(jj,'%06d')  '.mat'], 'input' );
    gt = single(ap);
    save([path2 num2str(jj,'%06d')  '.mat'], 'gt');
end

% generate and save testing dataset
for jj = 1:test_num
    fprintf(  ['progress of testing dataset generation = ' num2str(jj/test_num,'%4.6f') '\n']); % progress of testing dataset generation
    
    % initial absolute phase
    size_xy = randi([size_min,size_max]); % get size of initial matrix
    initial_matrix = rand(size_xy,size_xy); % get initial matrix
    phase_size_pre = phase_size * 1.25; % get size of pre-enlarged absolute phase
    pre_enlarged_ap = imresize(initial_matrix,...
        [phase_size_pre phase_size_pre]);  % get pre-enlarged absolute phase
    initial_ap = pre_enlarged_ap((phase_size_pre/2-phase_size/2+1):(phase_size_pre/2+phase_size/2),...
        (phase_size_pre/2-phase_size/2+1):(phase_size_pre/2+phase_size/2)); % get initial absolute phase
    
    % set height h
    h(jj) = unifrnd(height_min, height_max, 1, 1);
    
    % normalize the absolute phase to [0,h]
    ymax=h(jj);ymin=0;
    xmax = max(max(initial_ap));
    xmin = min(min(initial_ap));
    ap = (ymax-ymin)*(initial_ap-xmin)/(xmax-xmin) + ymin;
    
    
    % add noise to absolute phase
    sd_noise(jj)=unifrnd (0, noise_max, 1, 1);
    Gauss_RP = randn(128,128)*sd_noise(jj);
    ap_N = ap + Gauss_RP;
    
    % get wrapped phase
    R_N = real(exp(1i*ap_N));
    I_N = imag(exp(1i*ap_N));
    wp = atan2(I_N,R_N);
    
    % save wrapped phase as input and absolute phase as ground truth
    input = single(wp);
    save([path3 num2str(jj,'%06d')  '.mat'], 'input' );
    gt = single(ap);
    save([path4 num2str(jj,'%06d')  '.mat'], 'gt');
end