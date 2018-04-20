clc;
clear;
close all;

% STEP 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [1 2; 2 0; 3 1; 2 3];
X = [ones(4, 1) x];
b = [2; 2; -2; -2];
figure('Name','Step 1','NumberTitle','off'),
axis([-1 5 -1 5]);
plot(x(1:2, 1), x(1:2, 2), 'r*');
hold on;
axis([-1 5 -1 5]);
plot(x(3:4, 1), x(3:4, 2), 'bo');
w = X\b;
hold on,
t = 1:0.1:3;
y = (w(1) + w(2)*t)./(-w(3));
plot(t, y, 'g');
g = X*w;
evu = histc(double([g(1:2)>0; g(3:4)<0]), unique(double([g(1:2)>0; g(3:4)<0])));
if size(evu, 1) == 2
    disp(['Step 1, (a): ', num2str(evu(1)), 'points have not been classified correctly.']);
else
    disp('Step 1, (a): All points have been classified correctly.');
end


b2 = [1; 1; -1; -1];
w2 = X\b2;
hold on,
t = 1:0.1:3;
y2 = (w2(1) + w2(2)*t)./(-w2(3));
plot(t, y2, 'k--');
g2 = X*w2;
evu2 = histc(double([g2(1:2)>0; g2(3:4)<0]), unique(double ...
            ([g2(1:2)>0; g2(3:4)<0])));
if size(evu2, 1) == 2
    disp(['Step 1, (b): ', num2str(evu2(1)), 'points have not been classified correctly.']);
else
    disp('Step 1, (b): All points have been classified correctly.');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEP 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load fisheriris;
xdata = meas(51:end, 3:4);
N = 100;
XDATA = [ones(N, 1) xdata];
b3 = [ones(50, 1)*2; ones(50, 1)*-2];
figure('Name','Step 2','NumberTitle','off'),
% The first 50 are class=1 and the last 50 are class=2
plot(xdata(1:50, 1), xdata(1:50, 2), 'b*')
hold on;
plot(xdata(51:end, 1), xdata(51:end, 2), 'ro')
w3 = XDATA\b3;
hold on,
t = 2:0.1:8;
y = (w3(1) + w3(2)*t)./(-w3(3));
plot(t, y, 'g');
g3 = XDATA*w3;
evu3 = histc(double([g3(1:N/2)>0; g3((N/2)+1:N)<0]), unique(double ...
            ([g3(1:N/2)>0; g3((N/2)+1:N)<0])));
if size(evu3, 1) == 2
    disp(['Step 2: ', num2str(evu3(1)), ' points have not been classified correctly.']);
else
    disp('Step 2: All points have been classified correctly.');
end
disp (['        Training data set accuracy = ', num2str((1 - (evu3(1))/N)*100), '%']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEP 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TrainSetSize = 1000;
mu0 = [0.7416 0.7416];  % Negative class mean
Sigma = [1 0; 0 1];  % Negative class covariance
R = chol(Sigma);
NTrain = repmat(mu0, TrainSetSize, 1) + randn(TrainSetSize, 2)*R;  % Negative training data set
mu1 = [0 0];  % Positive class mean
Sigma = [1 0; 0 1];  % Positive class covariance
R = chol(Sigma);
PTrain = repmat(mu1, TrainSetSize, 1) + randn(TrainSetSize, 2)*R;  % Positive training data set
figure('Name','Step 3, (a)','NumberTitle','off'),
plot(NTrain(:, 1), NTrain(:, 2), 'ro')
hold on
plot(PTrain(:, 1), PTrain(:, 2), 'b*')

xTrain = [PTrain; NTrain];
X4 = [ones(2*TrainSetSize, 1) xTrain];
b4 = [ones(TrainSetSize, 1)*2; ones(TrainSetSize, 1)*-2];
w4 = X4\b4;
hold on,
t = -5:0.1:5;
y = (w4(1) + w4(2)*t)./(-w4(3));
plot(t, y, 'k');
% Testing the classifier on the training data set
g4 = X4*w4;
evu4 = histc(double([g4(1:TrainSetSize)>0; g4(TrainSetSize+1:end)<0]), ...
             unique(double([g4(1:TrainSetSize)>0; g4(TrainSetSize+1:end)<0])));
if size(evu4, 1) == 2
    disp(['Step 3, (a): ', num2str(evu4(1)), ' points have not been classified correctly.']);
else
    disp('Step 3, (a): All points have been classified correctly.');
end
disp (['        Training data set accuracy = ', num2str((1 - (evu4(1))/(TrainSetSize*2))*100), '%']);


% Creating a new test data set
TestSetSize = 5000;
mu0 = [0.7416 0.7416];
Sigma = [1 0; 0 1];
R = chol(Sigma);
NTest = repmat(mu0, TestSetSize, 1) + randn(TestSetSize, 2)*R; 
mu1 = [0 0];
Sigma = [1 0; 0 1]; 
R = chol(Sigma);
PTest = repmat(mu1, TestSetSize, 1) + randn(TestSetSize, 2)*R;
figure('Name','Step 3, (b)','NumberTitle','off'),
plot(NTest(:, 1), NTest(:, 2), 'mo')
hold on
plot(PTest(:, 1), PTest(:, 2), 'c*')
hold on,
t = -5:0.1:5;
y = (w4(1) + w4(2)*t)./(-w4(3));
plot(t, y, 'k');

% Testing the classifier on the new test data set
g5 = [ones(2*TestSetSize, 1) [PTest; NTest]]*w4;
evu5 = histc(double([g5(1:TestSetSize)>0; g5(TestSetSize+1:end)<0]), ...
             unique(double([g5(1:TestSetSize)>0; g5(TestSetSize+1:end)<0])));
if size(evu4, 1) == 2
    disp(['Step 3, (b): ', num2str(evu5(1)), ' points have not been classified correctly.']);
else
    disp('Step 3, (b): All points have been classified correctly.');
end
disp (['        Test data set accuracy = ', num2str((1 - (evu5(1))/(TestSetSize*2))*100), '%']);

% Repeating the whole process of testing classifier on new test data sets
TestSetSize = 5000;
avgAccuracy = 0;
for i = 1:5
    mu0 = [0.7416 0.7416];
    Sigma = [1 0; 0 1];
    R = chol(Sigma);
    NTest = repmat(mu0, TestSetSize, 1) + randn(TestSetSize, 2)*R; 
    mu1 = [0 0];
    Sigma = [1 0; 0 1]; 
    R = chol(Sigma);
    PTest = repmat(mu1, TestSetSize, 1) + randn(TestSetSize, 2)*R;

    % Testing the classifier on the new test data set
    g6 = [ones(2*TestSetSize, 1) [PTest; NTest]]*w4;
    evu6 = histc(double([g5(1:TestSetSize)>0; g6(TestSetSize+1:end)<0]), ...
                 unique(double([g5(1:TestSetSize)>0; g6(TestSetSize+1:end)<0])));

    if i == 1
        disp(['Step 3, (c): Test data set (1) accuracy = ', ...
            num2str((1 - (evu6(1))/(TestSetSize*2))*100), '%']);
    else
        disp(['             Test data set (', num2str(i), ') accuracy = ', ...
            num2str((1 - (evu6(1))/(TestSetSize*2))*100), '%']);
    end
    avgAccuracy = avgAccuracy + (1 - (evu6(1))/(TestSetSize*2))*100;
    
end
avgAccuracy = avgAccuracy/5;
disp(['             Average accuracy = ', num2str(avgAccuracy), '%']);


% Repeating the whole process of testing classifier on new test data sets
% with small number of data points
TestSetSize = 5;
avgAccuracy2 = 0;
for i = 1:5
    mu0 = [0.7416 0.7416];
    Sigma = [1 0; 0 1];
    R = chol(Sigma);
    NTest = repmat(mu0, TestSetSize, 1) + randn(TestSetSize, 2)*R; 
    mu1 = [0 0];
    Sigma = [1 0; 0 1]; 
    R = chol(Sigma);
    PTest = repmat(mu1, TestSetSize, 1) + randn(TestSetSize, 2)*R;

    % Testing the classifier on the new test data set
    g6 = [ones(2*TestSetSize, 1) [PTest; NTest]]*w4;
    evu6 = histc(double([g5(1:TestSetSize)>0; g6(TestSetSize+1:end)<0]), ...
                 unique(double([g5(1:TestSetSize)>0; g6(TestSetSize+1:end)<0])));

    if i == 1
        disp(['Step 3, (d): Test data set (1) accuracy = ', ...
            num2str((1 - (evu6(1))/(TestSetSize*2))*100), '%']);
    else
        disp(['             Test data set (', num2str(i), ') accuracy = ', ...
            num2str((1 - (evu6(1))/(TestSetSize*2))*100), '%']);
    end
    avgAccuracy2 = avgAccuracy2 + (1 - (evu6(1))/(TestSetSize*2))*100;
    
end
avgAccuracy2 = avgAccuracy2/5;
disp(['             Average accuracy = ', num2str(avgAccuracy2), '%']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEP 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('BREASTDIAG_UCI_5foldValidationData_NEWFIXEDcorrupt1.mat');
% RareTrain1: positive training set
% MajorTrain1: negative training set
% RareTest1: positive test set
% MajorTest1: negative test set
TrainPSize = size(RareTrain1, 1);
TrainNSize = size(MajorTrain1 , 1);
TestPSize = size(RareTest1, 1);
TestNSize = size(MajorTest1 , 1);
x7 = [RareTrain1; MajorTrain1];
X7 = [ones(TrainPSize+TrainNSize, 1) x7];
b7 = [ones(TrainPSize, 1)*((TrainPSize + TrainNSize)/TrainPSize); ...
      ones(TrainNSize, 1)*-((TrainPSize + TrainNSize)/TrainNSize)];
w7 = X7\b7;
% Testing the classifier on the test data set
g7 = [ones(TestPSize + TestNSize, 1) [RareTest1; MajorTest1]]*w7;
evu7 = histc(double([g7(1:TestPSize)>0; g7(TestPSize+1:end)<0]), ...
             unique(double([g7(1:TestPSize)>0; g7(TestPSize+1:end)<0])));

disp(['Step 4: ', num2str(evu7(1)), ' points have not been classified correctly.']);
disp (['        Test data set accuracy = ', num2str((1 - (evu7(1))/(TestPSize + TestNSize))*100), '%']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
