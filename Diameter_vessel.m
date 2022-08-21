clc;             % Clear the command window.
close all;       % Close all figures 
clear;           % Erase all existing variables
%workspace;       
format long g;
format compact;
fontSize = 20;

image = imread('VDIS/VDIS007.bmp');
a = image(:,:,2);
crop = imcrop(a);                            %crop image
image = uint8(255 * mat2gray(crop));
BI = double(image);
binaryImage = ~imbinarize(BI, 85);

% Display the original image. 
subplot(2, 2, 1);
imshow(crop, []);
title('Original Image');




%% Remove small objects from binary image
binaryImage = bwareaopen(binaryImage, 300);
subplot(2,2,2);
imshow(binaryImage, []);
title('Cropped binary image');

%% Get the Euclidean Distance Transform.
complement = ~binaryImage ; 
c_Image = bwdist(complement);
drawnow;

%% Display the Euclidean Distance Transform Image.
subplot(2, 2, 3);
imshow(c_Image, []);
title('Euclidean Distance Transform Image')

skeletonize_Image = bwmorph(binaryImage, 'skel', inf);
skeletonize_Image = bwmorph(skeletonize_Image, 'spur', 40);

%% There should be just one now.  Let's check
[labeledImage, numLines] = bwlabel(skeletonize_Image);
fprintf('Found %d lines\n', numLines);

%% Display the Skeleton image.
subplot(2, 2, 4);
imshow(skeletonize_Image, []);
title('Skeleton Image');

%% Measure the radius be looking along the skeleton of the distancetransform.
j=round(numLines/2)+1;
figure(2)
textFontSize=10;

for k=1:numLines
   Obj = (labeledImage == k); 
   subplot(2,j, k),imshow(Obj,[]); 
   
   mean_Radius = mean(c_Image(Obj));
   mean_diameter(k) = 2*mean(c_Image(Obj));
    std_diameter(k) = 2*std(c_Image(Obj));
     minn(k) = min(c_Image(Obj));
     maxx(k) = max(c_Image(Obj));
     kurtoss(k) = 2 * kurtosis(c_Image(Obj));
     skeww(k) = 2 * skewness(c_Image(Obj));
    caption = sprintf(' Vessel #%d has a \nDiameter = %f pixels\nstd = %f pixels', ...
			k, mean_diameter(k), std_diameter(k));
		title(caption, 'FontSize', textFontSize);
    fprintf('component number, mean diameter, standard deviation ,min , maxx , kurtoss ,skew  \n%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n',k, mean_diameter(k), std_diameter(k) ,minn(k) , maxx(k) , skeww(k) , kurtoss(k) );
    
end


