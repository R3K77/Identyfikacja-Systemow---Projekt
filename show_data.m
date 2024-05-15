% Load data from dryer.dat
data = load('Dane/dryer.dat');
sampling_time = 0.8;
time = (0:size(data, 1)-1) * sampling_time;
% Plot the data
figure;
subplot(2,1,1);
plot(data(:,1));


subplot(2,1,2);

plot(data(:,2), 'r');