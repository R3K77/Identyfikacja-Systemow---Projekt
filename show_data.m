% Load data from dryer.dat
data = load('Dane/dryer.dat');
sampling_time = 0.8;
time = (0:size(data, 1)-1) * sampling_time;
% Plot the data
figure;
plot(data(:,1));
hold on;
plot(data(:,2));