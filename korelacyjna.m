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

% Budowa macierzy regresji dla zestawu treningowego
Phi = zeros(N_train-max(na, nb+nk-1), na+nb);
Y = zeros(N_train-max(na, nb+nk-1), 1);

for i = max(na, nb+nk-1)+1:N_train
    % Wypełnianie macierzy regresji Phi
    Phi(i-max(na, nb+nk-1), 1:na) = -output_train(i-1:-1:i-na);
    Phi(i-max(na, nb+nk-1), na+1:na+nb) = input_train(i-nk:-1:i-nk-nb+1);
    % Wektor wyjść Y
    Y(i-max(na, nb+nk-1)) = output_train(i);
end

% Estymacja parametrów modelu ARX metodą najmniejszych kwadratów
theta = (Phi' * Phi) \ (Phi' * Y);

% Ekstrakcja parametrów modelu ARX
a = theta(1:na);
b = theta(na+1:end);

% Wyświetlenie wyników
disp('Parametry modelu ARX:');
disp('Współczynniki a:');
disp(a');
disp('Współczynniki b:');
disp(b');

% Symulacja odpowiedzi modelu ARX dla danych testowych
N_test = length(input_test);
Y_pred = zeros(N_test, 1);

for i = max(na, nb+nk-1)+1:N_test
    % Obliczanie wyjścia modelu ARX
    Y_pred(i) = -a' * output_test(i-1:-1:i-na) + b' * input_test(i-nk:-1:i-nk-nb+1);
end

% Obliczenie błędu średniokwadratowego (MSE) dla danych testowych
mse = mean((output_test(max(na, nb+nk-1)+1:end) - Y_pred(max(na, nb+nk-1)+1:end)).^2);
disp(['Błąd średniokwadratowy (MSE) na danych testowych: ', num2str(mse)]);

% Wykresy
figure;
subplot(2,1,1);
plot(time(split_idx+1:end), output_test, 'b', 'DisplayName', 'Pomiary rzeczywiste');
hold on;
plot(time(split_idx+1:end), Y_pred, 'r', 'DisplayName', 'Predykcja modelu ARX');
legend;
xlabel('Czas [s]');
ylabel('Temperatura [C]');
title('Porównanie pomiarów rzeczywistych z predykcją modelu ARX');
grid on;

subplot(2,1,2);
plot(time, input_data, 'k');
xlabel('Czas [s]');
ylabel('Moc grzałki [W]');
title('Sygnał wejściowy');
grid on;

% Symulacja odpowiedzi skokowej modelu ARX
step_input = [zeros(nk-1, 1); ones(N_test, 1)];
step_response = zeros(N_test + nk - 1, 1);

for i = max(na, nb+nk-1)+1:N_test+nk-1
    step_response(i) = -a' * step_response(i-1:-1:i-na) + b' * step_input(i-nk:-1:i-nk-nb+1);
end

% Symulacja odpowiedzi impulsowej modelu ARX
impulse_input = [zeros(nk-1, 1); 1; zeros(N_test-1, 1)];
impulse_response = zeros(N_test + nk - 1, 1);

for i = max(na, nb+nk-1)+1:N_test+nk-1
    impulse_response(i) = -a' * impulse_response(i-1:-1:i-na) + b' * impulse_input(i-nk:-1:i-nk-nb+1);
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

% Tworzenie wykresu odpowiedzi skokowej
figure;
subplot(2,1,1);
plot((0:length(step_response)-1)*Tp, step_response, 'b', 'DisplayName', 'Model ARX');
hold on;
plot((1:M-1)*Tp, h_hat_M(2:end), 'r', 'DisplayName', 'Rzeczywista');
legend;
xlabel('Czas [s]');
ylabel('Odpowiedź skokowa');
title('Porównanie odpowiedzi skokowej');
xlim([0, 7.5])
grid on;

% Tworzenie wykresu odpowiedzi impulsowej
subplot(2,1,2);
plot((0:length(impulse_response)-1)*Tp, impulse_response, 'b', 'DisplayName', 'Model ARX');
hold on;
plot((1:M-1)*Tp, g_hat_M(2:end), 'r', 'DisplayName', 'Rzeczywista');
legend;
xlabel('Czas [s]');
ylabel('Odpowiedź impulsowa');
title('Porównanie odpowiedzi impulsowej');
xlim([0, 7.5])
grid on;
