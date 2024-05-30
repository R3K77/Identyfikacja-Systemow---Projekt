clc; clear; clear all;

% Wczytanie danych z pliku dryer.dat
data = load('../Dane/dryer.dat');
Tp = 0.08;
time = (0:size(data, 1)-1) * Tp;

% Załadowanie do zmiennych:
input_data = data(:,1);    % Moc grzałki wyrażona w [W]
output_data = data(:,2);   % Temperatura wyrażona w [C]

% Podział danych na zestawy treningowe i testowe (50/50)
split_idx = floor(length(input_data) / 2);
input_train = input_data(1:split_idx);
output_train = output_data(1:split_idx);
input_test = input_data(split_idx+1:end);
output_test = output_data(split_idx+1:end);

% Parametry
N_train = length(input_train); % liczba próbek w zestawie treningowym
na = 2; % rząd modelu ARX dla wyjścia
nb = 2; % rząd modelu ARX dla wejścia
nk = 3; % opóźnienie (number of samples)


% Obliczenia rzeczywistej odpowiedzi impulsowej i skokowej za pomocą analizy korelacyjnej
M = 100; % liczba próbek odpowiedzi impulsowej do estymacji
r_yu = xcorr(output_data, input_data, M-1, 'biased'); % korelacja wzajemna
r_uu = xcorr(input_data, input_data, M-1, 'biased'); % korelacja własna

% Utwórz macierz korelacji własnej
R_uu = toeplitz(r_uu(M:end));

% Oblicz estymator odpowiedzi impulsowej za pomocą wzoru
g_hat_M = (1/Tp) * ((R_uu' * R_uu) \ (R_uu' * r_yu(M:end)));

% Oblicz estymowaną odpowiedź skokową
h_hat_M = Tp * cumsum(g_hat_M);

% Tworzenie wykresu odpowiedzi skokowej
figure;
subplot(2,1,1);
hold on;
plot((1:M-1)*Tp, h_hat_M(2:end), 'r');
xlabel('Time [$s$]', 'Interpreter', 'latex');
ylabel('$\hat{h}(k)$', 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex');
% title('Porównanie odpowiedzi skokowej');
xlim([0, 7.5])
grid on;

% Tworzenie wykresu odpowiedzi impulsowej
subplot(2,1,2);
hold on;
plot((1:M-1)*Tp, g_hat_M(2:end), 'r');
xlabel('Time [$s$]', 'Interpreter', 'latex');
ylabel('$\hat{h}(k)$', 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex');
% title('Porównanie odpowiedzi impulsowej');
xlim([0, 7.5])
grid on;
