clc;
clear;
close all;

% Wczytanie danych z pliku dryer.dat
data = load('Dane/dryer.dat');
Tp = 0.08;
time = (0:size(data, 1)-1) * Tp;

% Załadowanie do zmiennych
input_data = data(:, 1); % Moc grzałki wyrażona w [W]
output_data = data(:, 2); % Temperatura wyrażona w [C]

% Podział danych na zestawy treningowe i testowe (50/50)
split_idx = floor(length(input_data) / 2);
input_train = input_data(1:split_idx);
output_train = output_data(1:split_idx);
input_test = input_data(split_idx+1:end);
output_test = output_data(split_idx+1:end);

% Parametry modelu ARX
na = 2; % rząd modelu ARX dla wyjścia
nb = 2; % rząd modelu ARX dla wejścia
nk = 3; % opóźnienie (number of samples)

% Inicjalizacja zmiennych
N_train = length(input_train);
phi = zeros(N_train - max(na, nb+nk) + 1, na + nb);
y_train = output_train(max(na, nb+nk):N_train);

% Wypełnianie macierzy regresji phi
for n = max(na, nb+nk):N_train
    phi_row = [-output_train(n-1:-1:n-na)', input_train(n-nk:-1:n-nk-nb+1)'];
    phi(n - max(na, nb+nk) + 1, :) = phi_row;
end

% --- Metoda RLS ---
% Załaduj skrypt rls.m i wykonaj go
% run('rls.m');
theta_RLS = [-1.2715; 0.3865; 0.0665; 0.0459]; % Przypisanie parametrów z metody RLS

% --- Metoda IV ---
% Załaduj skrypt iv_2.m i wykonaj go
% run('iv_2.m');
theta_IV = [-1.3245; 0.4345; 0.0664; 0.0400]; % Przypisanie parametrów z metody IV

% --- Metoda LS ---
% Załaduj skrypt ls_2.m i wykonaj go
% run('ls_2.m');
theta_LS = [-1.2704; 0.3864; 0.0662; 0.0460]; % Przypisanie parametrów z metody LS

% --- Metoda RIV ---
% Załaduj skrypt riv.m i wykonaj go
% run('riv.m');
theta_RIV = [-1.2715; 0.3865; 0.0665; 0.0459]; % Przypisanie parametrów z metody RIV

% --- Symulacja odpowiedzi skokowych dla różnych metod ---
methods = {'RLS', 'IV', 'LS', 'RIV'};
theta_methods = {theta_RLS, theta_IV, theta_LS, theta_RIV};
colors = {'r', 'g', 'b', 'm'};

figure;
hold on;

for i = 1:length(methods)
    % Ekstrakcja parametrów modelu ARX
    theta = theta_methods{i};
    a = theta(1:na);
    b = theta(na+1:end);
    
    % Obliczenie transmitancji na podstawie parametrów ARX
    b_trans = [zeros(1, nk), b'];
    a_trans = [1, a'];
    sys = tf(b_trans, a_trans, Tp, 'Variable', 'z^-1');
    
    % Generowanie odpowiedzi skokowej
    [step_response, step_time] = step(sys, 0:Tp:7.5);
    
    % Wykres odpowiedzi skokowej
    plot(step_time, step_response, colors{i}, 'DisplayName', methods{i});
end

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

% Dodanie rzeczywistej odpowiedzi skokowej do wykresu
plot((1:M-1)*Tp, h_hat_M(2:end), 'k', 'DisplayName', 'Real');

legend('Location', 'best', 'Interpreter', 'latex');
xlabel('Time [$nT_p$]', 'Interpreter', 'latex');
% ylabel('Amplitude', 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex');
% title('Porównanie odpowiedzi skokowej');
grid on;
hold off;
