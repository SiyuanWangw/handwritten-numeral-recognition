function [f,g] = MLPclassificationLoss_cnn(w,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X);

% Form Weights
offset = 0;
for j=1:nHidden(1)/144
   kernel{j} = reshape(w(offset+1:offset+5*5),5,5); 
   offset = offset+5*5;
end

for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

f = 0;
if nargout > 1
    for j=1:nHidden(1)/144
        gConvolution{j} = zeros(size(kernel{j}));
    end
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances
    %using 2D convolutional layer
    image = reshape(X(i,:),16,16);
    for j=1:nHidden(1)/144
        convolution = conv2(image, kernel{j},'valid');
        if j==1
            ip{1} = reshape(convolution, 1, 144);
        else
            ip{1} = [ip{1} reshape(convolution, 1, 144)];
        end
    end
    fp{1} = tanh(ip{1});
    
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end}*outputWeights;
    
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2*relativeErr;

        % Output Weights
        gOutput = gOutput + fp{end}'*err;

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
       
            backprop = err*(repmat(sech(ip{end}).^2,[nLabels,1]).*outputWeights');
            gHidden{end} = gHidden{end} + fp{end-1}'*backprop;
            
            backprop = sum(backprop,1);
            
            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
            end

            % Convolutional Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            for j=1:nHidden(1)/144
                backprophelp = reshape(backprop((j-1)*144+1:j*144), 12, 12);
                gConvolution{j} = gConvolution{j} + rot90(conv2(image, rot90(backprophelp,2),'valid'),2); 
            end
        else
           % Convolutional Weights
            clear backprop2
            backprop2 = err*(repmat(sech(ip{end}).^2,[nLabels,1]).*outputWeights');
            for j=1:nHidden(1)/144
                backprop2help = reshape(backprop2((j-1)*144+1:j*144), 12, 12);
                gConvolution{j} = gConvolution{j} + rot90(conv2(image, rot90(backprop2help,2),'valid'),2); 
            end
        end
        
    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    offset = 0;
    for j=1:nHidden(1)/144
        g(offset+1:offset+5*5) = gConvolution{j};
        offset = offset+5*5;
    end
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
