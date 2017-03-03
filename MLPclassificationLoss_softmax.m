function [f,g] = MLPclassificationLoss_softmax(w,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
lambda = 0.2;
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end}*outputWeights;
    
    %softmax function 
    probability = exp(yhat)/sum(exp(yhat)); 
    true_label = find(y(i,:)==1);
    f = f - log(probability(true_label));
    
    if nargout > 1     
        % Output Weights
        %for c = 1:nLabels
        %    if c == true_label
        %        gOutput(:,c) = gOutput(:,c) + (exp(yhat(c))-sum(exp(yhat)))/sum(exp(yhat))*fp{end}';
        %    else 
        %        gOutput(:,c) = gOutput(:,c) + exp(yhat(c))/sum(exp(yhat))*fp{end}';
        %    end
        %end
        gOutput = gOutput + fp{end}'*exp(yhat)/sum(exp(yhat));
        gOutput(:,true_label) = gOutput(:,true_label) - fp{end}';
        
        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop 
            clear backprop2
            backprop = exp(yhat)*(repmat(sech(ip{end}).^2,[nLabels,1]).*outputWeights');
            backprop2 = (sech(ip{end}).^2.*outputWeights(:,true_label)')*sum(exp(yhat));
            gHidden{end} = gHidden{end} + (fp{end-1}'*backprop-fp{end-1}'*backprop2)/sum(exp(yhat));
            
            backprop = sum(backprop,1);     
            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                backprop2 = (backprop2*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + (fp{h}'*backprop-fp{h}'*backprop2)/sum(exp(yhat));
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            backprop2 = (backprop2*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + (X(i,:)'*backprop-X(i,:)'*backprop2)/sum(exp(yhat));
        else
           % Input Weights
           %gradient_all = 0;
           %gradient_true = 0;
           %for c = 1:nLabels
               %gradient_all = gradient_all + exp(yhat(c))*X(i,:)'*(sech(ip{end}).^2.*outputWeights(:,c)');
               %if c == true_label
               %    gradient_true = X(i,:)'*(sech(ip{end}).^2.*outputWeights(:,c)')*sum(exp(yhat));
               %end
           %end
           %gInput = gInput + (gradient_all-gradient_true)/sum(exp(yhat));
           gradient_all = X(i,:)'*exp(yhat)*(repmat(sech(ip{end}).^2,[nLabels,1]).*outputWeights');
           gradient_true = X(i,:)'*(sech(ip{end}).^2.*outputWeights(:,true_label)')*sum(exp(yhat));
           gInput = gInput + (gradient_all-gradient_true)/sum(exp(yhat));
        end 
    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end

