function [y] = MLPclassificationPredict_cnn(w,X,nHidden,nLabels)
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

% Compute Output
for i = 1:nInstances
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
    y(i,:) = fp{end}*outputWeights;
end
[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
