clc; clear; clear all;

% Wczytanie danych z pliku dryer.dat
data = load('Dane/dryer.dat');
Tp = 0.08;
time = (0:size(data, 1)-1) * Tp;

% Załadowanie do zmiennych:
input_data = data(:,1);    % Moc grzałki wyrażona w [W]
output_data = data(:,2);   % Temperatura wyrażona w [C]

% Tworzenie wykresów
% figure;

% % Wykres danych wejściowych
% subplot(2, 1, 1);
% plot(input_data);
% title('Input Data');
% xlabel('Sample');
% ylabel('Voltage');
% legend('Input Data');

% % Wykres danych wyjściowych
% subplot(2, 1, 2);
% plot(output_data);
% title('Output Data');
% xlabel('Sample');
% ylabel('Voltage');
% legend('Output Data');

% Parametry
M = 100; % liczba próbek odpowiedzi impulsowej do estymacji, dostosuj według potrzeb
Tp = 0.08; % okres próbkowania, dostosuj do swoich danych

% Oblicz funkcję korelacji
r_yu = xcorr(output_data, input_data, M-1, 'biased'); % korelacja wzajemna
r_uu = xcorr(input_data, input_data, M-1, 'biased'); % korelacja własna

% Utwórz macierz korelacji własnej
R_uu = toeplitz(r_uu(M:end));

% Oblicz estymator odpowiedzi impulsowej za pomocą wzoru
g_hat_M = (1/Tp) * ((R_uu' * R_uu) \ (R_uu' * r_yu(M:end)));

% Oblicz estymowaną odpowiedź skokową
h_hat_M = Tp * cumsum(g_hat_M);

% Tworzenie wykresu analizy korelacyjnej
figure;
subplot(2, 1, 1);
plot(1:M-1, g_hat_M(2:end));
% title('Estimated Impulse Response using Cross-Correlation Analysis');
xlabel('Time $[nTp]$', 'Interpreter', 'latex');
ylabel('Amplitude $g_{Mb}$', 'Interpreter', 'latex');
grid on;
set(gca, 'TickLabelInterpreter', 'latex');

% Tworzenie wykresu estymowanej odpowiedzi skokowej
subplot(2, 1, 2);
plot(1:M-1, h_hat_M(2:end));
% title('Estimated Step Response');
xlabel('Time $[nTp]$', 'Interpreter', 'latex');
ylabel('Amplitude $h_{Mb}$', 'Interpreter', 'latex');
grid on;
set(gca, 'TickLabelInterpreter', 'latex');