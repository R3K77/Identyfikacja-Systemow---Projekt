clc; clear; clear all;

% Wczytanie danych z pliku dryer.dat
data = load('Dane/dryer.dat');
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

% Inicjalizacja zmiennych
phi = zeros(N_train - max(na, nb+nk) + 1, na + nb); % macierz regresji
y_train = output_train(max(na, nb+nk):N_train); % wyjściowe dane treningowe

% Wypełnianie macierzy regresji phi
for n = max(na, nb+nk):N_train
    phi_row = [-output_train(n-1:-1:n-na)', input_train(n-nk:-1:n-nk-nb+1)'];
    phi(n - max(na, nb+nk) + 1, :) = phi_row;
end

% Estymacja parametrów metodą LS
theta_LS = (phi' * phi) \ (phi' * y_train);

% Predykcja na zbiorze testowym
N_test = length(input_test);
y_pred = zeros(N_test - max(na, nb+nk) + 1, 1);
phi_test = zeros(N_test - max(na, nb+nk) + 1, na + nb);

for n = max(na, nb+nk):N_test
    phi_row_test = [-output_test(n-1:-1:n-na)', input_test(n-nk:-1:n-nk-nb+1)'];
    phi_test(n - max(na, nb+nk) + 1, :) = phi_row_test;
    y_pred(n - max(na, nb+nk) + 1) = phi_row_test * theta_LS;
end

% Obliczanie Jfit
y_true = output_test(max(na, nb+nk):N_test);
Jfit = 100 * (1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2));

% Tworzenie wykresów estymat parametrów
figure;
for i = 1:na + nb
    subplot(na + nb, 1, i);
    plot(theta_LS(i) * ones(length(y_pred), 1));
    title(['Parameter \theta_', num2str(i)]);
    xlabel('Sample');
    ylabel(['\theta_', num2str(i)]);
    grid on;
end

% Tworzenie wykresu rzeczywistych i predykowanych wartości wyjściowych
figure;
plot(max(na, nb+nk):N_test, y_true, 'b', max(na, nb+nk):N_test, y_pred, 'r--');
title('True vs Predicted Output');
xlabel('Sample');
ylabel('Output');
legend('True Output', 'Predicted Output');
grid on;

% Wyświetlanie końcowych estymat parametrów i Jfit
disp('Estimated parameters:');
disp(theta_LS);

disp(['Jfit: ', num2str(Jfit)]);

% Obliczanie transmitancji końcowej
A = [1; theta_LS(1:na)];
B = [zeros(nk, 1); theta_LS(na+1:end)];

sys = tf(B', A', Tp);
disp('Final transfer function (tf object):');
disp(sys);

% Wyświetlanie czytelnej ułamkowej formy transmitancji
[num, den] = tfdata(sys, 'v');
disp('Final transfer function (fractional form):');
trans_func_str = ['(' num2str(num) ') / (' num2str(den) ')'];
disp(trans_func_str);
