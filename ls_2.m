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

% Obliczenie transmitancji na podstawie parametrów ARX
b_trans = [zeros(1, nk), b'];
a_trans = [1, a'];

sys = tf(b_trans, a_trans, Tp, 'Variable', 'z^-1');

% Symulacja odpowiedzi modelu ARX dla danych testowych
Y_pred = lsim(sys, input_test, time(split_idx+1:end));

% Obliczenie błędu średniokwadratowego (MSE) dla danych testowych
mse = mean((output_test - Y_pred).^2);
disp(['Błąd średniokwadratowy (MSE) na danych testowych: ', num2str(mse)]);

% Obliczenie wskaźnika dopasowania (Jfit)
Jfit = 100 * (1 - sum((output_test - Y_pred).^2) / sum((output_test - mean(output_test)).^2));
disp(['Wskaźnik dopasowania (J_{fit}): ', num2str(Jfit), '%']);

% Wykresy
figure;
subplot(2,1,1);
plot(time(split_idx+1:end), output_test, 'b');
hold on;
plot(time(split_idx+1:end), Y_pred, 'r');
legend('$y$', '$y_m$', 'Interpreter', 'latex');
xlabel('Time [$nT_p$]', 'Interpreter', 'latex');
set(gca,'TickLabelInterpreter','latex');
% ylabel('Temperatura [C]');
% title('Porównanie pomiarów rzeczywistych z predykcją modelu ARX');
grid on;

subplot(2,1,2);
plot(time, input_data, 'k');
xlabel('Time [$nT_p$]', 'Interpreter', 'latex');
ylabel('Power [W]', 'Interpreter', 'latex');
set(gca,'TickLabelInterpreter','latex');
% title('Sygnał wejściowy');
grid on;

% Generowanie odpowiedzi skokowej
[step_response, step_time] = step(sys, 0:Tp:7.5);

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
plot(step_time, step_response, 'b');
hold on;
plot((1:M-1)*Tp, h_hat_M(2:end), 'r');
legend('$h_m$', '$h$', 'Interpreter', 'latex');
xlabel('Time [$s$]', 'Interpreter', 'latex');
set(gca,'TickLabelInterpreter','latex');
% ylabel('h', 'Interpreter', 'latex');
% title('Porównanie odpowiedzi skokowej');
xlim([0 7.5]);
grid on;
