clc;
clear all;
close all;

%% Input and Display Image
% Read the input image
inputImage = imread('heart.jpg'); %face.jpg heart.jpg
figure(1);
imshow(inputImage);
title('Input Image');

% Display size of the input image
imageSize = size(inputImage);
disp(['Image size: ', num2str(imageSize)]);

%% Convert to Grayscale and Process Binary Image
% Convert the image to grayscale
grayImage = rgb2gray(inputImage);

% Convert to binary image and scale
binaryImage = 255 * imbinarize(grayImage);

% Normalize the grayscale image
normalizedGray = double(grayImage) / max(max(double(grayImage)));
averageIntensity = mean(normalizedGray, 'all');

% Enhance binary image
for p = 1:size(binaryImage, 1)
    for j = 1:size(binaryImage, 2)
        if binaryImage(p, j) >= 180
            binaryImage(p, j) = 0;
        else
            binaryImage(p, j) = 255;
        end
    end
end

% Display the processed binary image
figure(2);
imshow(binaryImage, []);
title('Binary Image');
colormap('gray');

%% Cost Function Optimization
% Initialize parameters
iterations = 100;
relativeThreshold = 1e-3; % Small value for stopping
previousRMSE = inf;
binaryImage = double(binaryImage);
unityMatrix = ones(size(binaryImage));
fourierTransform = fftshift(binaryImage);
inverseFourierTransform = ifft2(fourierTransform);
phaseShifted = fftshift(inverseFourierTransform);

% Display RMSE optimization process
figure(3);
hold on;
axis([0, iterations, 0, 1]);
xlabel('Number of Iterations');
ylabel('RMSE');
title('Cost Function Optimization');

for k = 1:iterations
    % Update phase in Fourier domain
    magnitude = abs(unityMatrix) .* exp(1i * angle(phaseShifted));
    updatedFourier = fftshift(fft2(fftshift(magnitude)));
    
    % Update intensity
    intensity = abs(binaryImage) .* exp(1i * angle(updatedFourier));
    phaseShifted = fftshift(ifft2(fftshift(intensity)));
    
    % Calculate RMSE
    adjustedAverage = mean(abs(phaseShifted), 'all');
    intensityNormalized = (phaseShifted / adjustedAverage) * averageIntensity;
    rmse = sqrt(mean((abs(intensityNormalized) - normalizedGray).^2, 'all'));
    plot(k, rmse, 'o');

    % Stop if RMSE threshold is met
    if abs(previousRMSE - rmse) / previousRMSE < relativeThreshold && k > 50
        break;
    end

    previousRMSE = rmse;

end
hold off;

%% Generate Phase and CGH Hologram
% Compute phase information
phase = angle(phaseShifted);
quantizedPhase = round((phase + pi) * 255 / (2 * pi));

% Save and display CGH hologram
figure(4);
imagesc(quantizedPhase);
colormap('gray');
title('CGH Hologram');
imwrite(uint8(quantizedPhase), gray(256), 'output.png');

%% Display Output Image
% Compute and display Fourier transform magnitude
outputImage = abs(fftshift(fft2(fftshift(exp(1i * phase)))));
figure(5);
imagesc(outputImage);
colormap('gray');
title('Output Image');
