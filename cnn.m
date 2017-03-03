load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xtest = standardizeCols(Xtest,mu,sigma);

% Choose network structure
nHidden = [144*2,100];

% Count number of parameters and initialize weights 'w'
% elements of convolution kernel
nParams = 5*5*nHidden(1)/144;

for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1)/20;

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)MLPclassificationLoss_cnn(w,X(i,:),yExpanded(i,:),nHidden,nLabels);

%For momentum
%lastweights = w;
%momentum_strength = 0.9;
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict_cnn(w,Xvalid,nHidden,nLabels);
        validation_error = sum(yhat~=yvalid)/t;
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,validation_error);
        %if validation_error < 0.3 && validation_error > 0.15
        %  stepSize = 0.7*1e-3;
        %elseif validation_error < 0.15
        %  stepSize = 0.5*1e-3;
        %end
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    %momentum_term = momentum_strength*(w-lastweights);
    %lastweights = w;
    w = w - stepSize*g;% + momentum_term;
end

% Evaluate test error
yhat = MLPclassificationPredict_cnn(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);

