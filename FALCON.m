clear all;
clc;

% Skripta za FALCON metod sa analizom deformacija i šuma u poređenju sa ZNCC

%% Faza 1: Učitavanje osnovne slike
disp('Izvršava se faza: Učitavanje osnovne slike...');
imagePath = 'C:\Users\Bato\Desktop\RADOVI 2025\FALCON\1-0004.jpg';
I1 = imread(imagePath);

% Pretvaranje slike u grayscale ako je RGB
if size(I1, 3) == 3
    I1 = rgb2gray(I1);
end

% Normalizacija slike na opseg [0, 1]
I1 = double(I1) / 255;

% Prikaz osnovne slike
figure(10);
imshow(I1, []);
title('Osnovna slika (Referentna slika)');
disp('Osnovna slika prikazana u Figure (10).');

%% Faza 2: Parametri za analizu i inicijalizacija
disp('Izvršava se faza: Inicijalizacija parametara...');
imageSize = size(I1);
windowSize = 32;
hanningWindow = hann(windowSize) * hann(windowSize)';

noiseLevels = linspace(0.01, 20, 5);  % Nivoi šuma
deformationLevels = linspace(1, 10, 5);  % Intenzitet deformacija

falconResults = zeros(length(noiseLevels), length(deformationLevels));
znccResults = zeros(length(noiseLevels), length(deformationLevels));
executionTimesFALCON = zeros(length(noiseLevels), length(deformationLevels));
executionTimesZNCC = zeros(length(noiseLevels), length(deformationLevels));
pearsonFalcon = zeros(length(noiseLevels), length(deformationLevels));
pearsonZncc = zeros(length(noiseLevels), length(deformationLevels));
ssimFalcon = zeros(length(noiseLevels), length(deformationLevels));
ssimZncc = zeros(length(noiseLevels), length(deformationLevels));
mseFalcon = zeros(length(noiseLevels), length(deformationLevels));
mseZncc = zeros(length(noiseLevels), length(deformationLevels));
psnrFalcon = zeros(length(noiseLevels), length(deformationLevels));
psnrZncc = zeros(length(noiseLevels), length(deformationLevels));
snrFalcon = zeros(length(noiseLevels), length(deformationLevels));
snrZncc = zeros(length(noiseLevels), length(deformationLevels));
savedImages = cell(length(noiseLevels), length(deformationLevels));

%% Faza 3: Petlja za analizu šuma i deformacija
disp('Izvršava se faza: Analiza sa različitim šumom i deformacijama...');
for i = 1:length(noiseLevels)
    noiseLevel = noiseLevels(i);
    
    for j = 1:length(deformationLevels)
        deformationLevel = deformationLevels(j);

        % Generisanje deformisane slike sa šumom
        I2 = circshift(I1, round([5, 5] * deformationLevel));
        I2 = I2 + lorentzNoise(imageSize, noiseLevel);
        I2 = imrotate(I2, deformationLevel * 5, 'bilinear', 'crop');
        I2 = imresize(I2, 1 + deformationLevel * 0.05);
        I2 = imresize(I2, imageSize);
        savedImages{i, j} = I2;

        % FALCON model
        tic;
        localFourierRef = computeLocalFourier(I1, hanningWindow, windowSize);
        localFourierDef = computeLocalFourier(I2, hanningWindow, windowSize);
        phaseCorr = abs(sum(localFourierRef .* conj(localFourierDef), 'all')) / ...
                    (sqrt(sum(abs(localFourierRef).^2, 'all')) * sqrt(sum(abs(localFourierDef).^2, 'all')));
        amplitudeCorr = sum(abs(localFourierRef) .* abs(localFourierDef), 'all');
        combinedCorr = 0.6 * phaseCorr + 0.4 * amplitudeCorr;
        executionTimesFALCON(i, j) = toc;
        falconResults(i, j) = combinedCorr;

        % ZNCC model
        tic;
        zncc = normxcorr2(I1, I2);
        executionTimesZNCC(i, j) = toc;
        znccResults(i, j) = max(zncc(:));

        % Pearson, SSIM, MSE, PSNR i SNR
        pearsonFalcon(i, j) = corr2(imresize(combinedCorr, size(I1)), I1);
        pearsonZncc(i, j) = corr2(imresize(zncc, size(I1)), I1);
        ssimFalcon(i, j) = ssim(imresize(combinedCorr, size(I1)), I1);
        ssimZncc(i, j) = ssim(imresize(zncc, size(I1)), I1);
        mseFalcon(i, j) = immse(imresize(combinedCorr, size(I1)), I1);
        mseZncc(i, j) = immse(imresize(zncc, size(I1)), I1);
        psnrFalcon(i, j) = 10 * log10(1 / mseFalcon(i, j));
        psnrZncc(i, j) = 10 * log10(1 / mseZncc(i, j));
        snrFalcon(i, j) = 10 * log10(sum(I1(:).^2) / mseFalcon(i, j));
        snrZncc(i, j) = 10 * log10(sum(I1(:).^2) / mseZncc(i, j));
    end
end
disp('Analiza završena.');

%% Faza 4: Prikaz deformacija i prostorne gustine šuma
disp('Izvršava se faza: Prikaz deformacija i šuma...');
figure(20);
k = 1;
for i = 1:3
    for j = 1:3
        subplot(3, 3, k);
        imshow(savedImages{i, j}, []);
        title(sprintf('Noise: %.2f, Def: %.2f', noiseLevels(i), deformationLevels(j)));
        k = k + 1;
    end
end
sgtitle('Deformacije i prostorna gustina šuma');
disp('Prikazane slike u Figure (20).');

%% Faza 5: Prikaz zavisnosti korelacije od šuma i deformacija
disp('Izvršava se faza: Prikaz zavisnosti korelacije...');

% Figure (80): Zavisnost korelacije za FALCON metod
figure(80);
for j = 1:length(deformationLevels)
    plot(noiseLevels, falconResults(:, j), '-o', 'DisplayName', ['Def: ' num2str(deformationLevels(j))]);
    hold on;
end
xlabel('Noise Level [%]');
ylabel('FALCON Correlation');
legend('show');
title('Dependence of Correlation on Noise and Deformation (FALCON)');
grid on;
disp('Prikazana Figure (80) za FALCON korelaciju.');

% Figure (85): Zavisnost korelacije za ZNCC metod
figure(85);
for j = 1:length(deformationLevels)
    plot(noiseLevels, znccResults(:, j), '-o', 'DisplayName', ['Def: ' num2str(deformationLevels(j))]);
    hold on;
end
xlabel('Noise Level [%]');
ylabel('ZNCC Correlation');
legend('show');
title('Dependence of Correlation on Noise and Deformation (ZNCC)');
grid on;
disp('Prikazana Figure (85) za ZNCC korelaciju.');


%% Faza 6: Prikaz SSIM vrednosti
disp('Izvršava se faza: Prikaz SSIM vrednosti...');
figure(90);
plot(noiseLevels, mean(ssimFalcon, 2), '-bo', noiseLevels, mean(ssimZncc, 2), '-rx');
xlabel('Noise Level [%]');
ylabel('SSIM');
legend({'FALCON', 'ZNCC'}, 'Location', 'Best');
title('SSIM Comparison for FALCON and ZNCC Models');
grid on;

%% Faza 7: Prikaz PSNR i SNR vrednosti
disp('Izvršava se faza: Prikaz PSNR i SNR...');

% Figure (100): Prikaz PSNR vrednosti
figure(100);
plot(noiseLevels, mean(psnrFalcon, 2), '-bo', noiseLevels, mean(psnrZncc, 2), '-rx');
xlabel('Noise Level [%]');
ylabel('PSNR [dB]');
legend({'FALCON', 'ZNCC'}, 'Location', 'Best');
title('PSNR Comparison');
grid on;
disp('Prikazana Figure (100) za PSNR.');

% Figure (105): Prikaz SNR vrednosti
figure(105);
plot(noiseLevels, mean(snrFalcon, 2), '-bo', noiseLevels, mean(snrZncc, 2), '-rx');
xlabel('Noise Level [%]');
ylabel('SNR [dB]');
legend({'FALCON', 'ZNCC'}, 'Location', 'Best');
title('SNR Comparison');
grid on;
disp('Prikazana Figure (105) za SNR.');


%% Faza 8: Prikaz vremena izvršavanja
disp('Izvršava se faza: Prikaz vremena izvršavanja...');
figure(110);
plot(noiseLevels, mean(executionTimesFALCON, 2), '-bo', noiseLevels, mean(executionTimesZNCC, 2), '-rx');
xlabel('Noise Level [%]');
ylabel('Execution Time (s)');
legend({'FALCON', 'ZNCC'}, 'Location', 'Best');
title('Execution Time of FALCON vs. ZNCC');
grid on;

%% Faza 9: Prikaz Pearsonovog koeficijenta
disp('Izvršava se faza: Prikaz Pearsonovog koeficijenta...');
figure(120);
plot(noiseLevels, mean(pearsonFalcon, 2), '-bo', noiseLevels, mean(pearsonZncc, 2), '-rx');
xlabel('Noise Level [%]');
ylabel('Pearson Correlation Coefficient');
legend({'FALCON', 'ZNCC'}, 'Location', 'Best');
title('Comparative Pearson Correlation Coefficient');
grid on;

disp('Sve faze izvršavanja završene.');

%% Funkcije
function localFourier = computeLocalFourier(image, window, windowSize)
    [rows, cols] = size(image);
    localFourier = zeros(rows, cols);
    for i = 1:windowSize:rows-windowSize
        for j = 1:windowSize:cols-windowSize
            subRegion = image(i:i+windowSize-1, j:j+windowSize-1) .* window;
            fftRegion = abs(fft2(subRegion));
            localFourier(i:i+windowSize-1, j:j+windowSize-1) = fftRegion(1:windowSize, 1:windowSize);
        end
    end
end

function noise = lorentzNoise(imageSize, noiseLevel)
    sigma = 10; rho = 28; beta = 8/3;
    dt = 0.01; numSteps = prod(imageSize);
    x = zeros(1, numSteps); y = zeros(1, numSteps); z = zeros(1, numSteps);
    x(1) = rand() * noiseLevel; y(1) = rand() * noiseLevel; z(1) = rand() * noiseLevel;
    for i = 2:numSteps
        dx = sigma * (y(i-1) - x(i-1)) * dt;
        dy = (x(i-1) * (rho - z(i-1)) - y(i-1)) * dt;
        dz = (x(i-1) * y(i-1) - beta * z(i-1)) * dt;
        x(i) = x(i-1) + dx; y(i) = y(i-1) + dy; z(i) = z(i-1) + dz;
    end
    noise = reshape(z, imageSize);
    noise = noise / max(abs(noise(:))) * noiseLevel;
end
