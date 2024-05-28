clc; clear; clear all;

% Wczytanie danych z pliku dryer.dat
data = load('Dane/dryer.dat');
Tp = 0.08;
time = (0:size(data, 1)-1) * Tp;

% Załadowanie do zmiennych:
input_data = data(:,1);    % Moc grzałki wyrażona w [W]
output_data = data(:,2);   % Temperatura wyrażona w [C]


% Tworzenie wykresów
figure;


% Parametry
N = length(input_data); % liczba próbek
na = 2; % rząd modelu ARX dla wyjścia
nb = 2; % rząd modelu ARX dla wejścia
nk = 3; % opóźnienie (number of samples)

% Inicjalizacja zmiennych
theta = zeros(na + nb, 1); % estymowane parametry
P = 10 * eye(na + nb); % macierz kowariancji
phi = zeros(na + nb, 1); % wektor regresji

% Przechowywanie estymat parametrów
theta_history = zeros(N, na + nb);
y_pred = zeros(N, 1);

% Algorytm RLS
for n = max(na, nb+nk):N
    % Wektor regresji
    phi(1:na) = -output_data(n-1:-1:n-na);
    phi(na+1:end) = input_data(n-nk:-1:n-nk-nb+1);
    
    % Predykcja wyjścia
    y_hat = phi' * theta;
    y_pred(n) = y_hat;
    
    % Błąd predykcji
    e = output_data(n) - y_hat;
    
    % Wzmocnienie
    K = P * phi / (1 + phi' * P * phi);
    
    % Aktualizacja estymaty parametrów
    theta = theta + K * e;
    
    % Aktualizacja macierzy kowariancji
    P = (P - K * phi' * P);
    
    % Przechowywanie estymat parametrów
    theta_history(n, :) = theta';
end

% Obliczanie Jfit
y_true = output_data(max(na, nb+nk):N);
y_pred = y_pred(max(na, nb+nk):N);
Jfit = 100 * (1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2));

% Tworzenie wykresów estymat parametrów
figure;
for i = 1:na + nb
    subplot(na + nb, 1, i);
    plot(theta_history(:, i));
    title(['Parameter \theta_', num2str(i)]);
    xlabel('Sample');
    ylabel(['\theta_', num2str(i)]);
    grid on;
end

% Tworzenie wykresu rzeczywistych i predykowanych wartości wyjściowych
figure;
plot(max(na, nb+nk):N, y_true, 'b', max(na, nb+nk):N, y_pred, 'r--');
title('True vs Predicted Output');
xlabel('Sample');
ylabel('Output');
legend('True Output', 'Predicted Output');
grid on;

% Wyświetlanie końcowych estymat parametrów i Jfit
disp('Estimated parameters:');
disp(theta);



disp(['Jfit: ', num2str(Jfit)]);
