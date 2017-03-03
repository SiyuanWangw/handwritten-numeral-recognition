function [ X_add ] = noise( X )
% add noise to transform the original images
  X_add = X + rand(size(X))*2+3;
 
end