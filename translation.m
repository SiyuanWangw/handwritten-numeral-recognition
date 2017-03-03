function [ X_add ] = translation( X )
% using translation to transform the original images
 X_add = X;
 for i=1:length(X)
     image = reshape(X(i,:),16,16);
     newImage = [image(:,3:16) image(:,1:2)];
     X_add(i,:) = reshape(newImage,1,256);
 end
 
end

