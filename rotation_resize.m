function [ X_add ] = rotation_resize( X )
% using rotation to transform the original images
  X_add = X;
  
  for i=1:length(X)
      image = reshape(X(i,:),16,16);
      angle = rand*40-20;
      newImage = imrotate(image,angle,'crop');
      X_add(i,:) = reshape(newImage,1,256);
  end  
 
end


