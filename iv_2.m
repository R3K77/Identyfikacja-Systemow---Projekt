clc; clear; clear all;

% Wczytanie danych z pliku dryer.dat
data = load('Dane/dryer.dat');
Tp = 0.08;
time = (0:size(data, 1)-1) * Tp

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
nk =3; % opóźnienie (number of samples)

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
theta_LS = (Phi' * Phi) \ (Phi' * Y);

% Generowanie zmiennych instrumentalnych
x = zeros(size(output_train));
for i = max(na, nb+nk-1)+1:N_train
    x(i) = -theta_LS(1:na)' * output_train(i-1:-1:i-na) + theta_LS(na+1:end)' * input_train(i-nk:-1:i-nk-nb+1);
end

% Budowa macierzy zmiennych instrumentalnych
Z = zeros(N_train-max(na, nb+nk-1), na+nb);
for i = max(na, nb+nk-1)+1:N_train
    Z(i-max(na, nb+nk-1), 1:na) = -x(i-1:-1:i-na);
    Z(i-max(na, nb+nk-1), na+1:na+nb) = input_train(i-nk:-1:i-nk-nb+1);
end

% Estymacja parametrów modelu ARX metodą zmiennych instrumentalnych
theta_IV = (Z' * Phi) \ (Z' * Y);

% Wyświetlenie wyników
disp('Parametry modelu ARX (IV):');
disp('Współczynniki a:');
disp(theta_IV(1:na)');
disp('Współczynniki b:');
disp(theta_IV(na+1:end)');

% Obliczenie transmitancji na podstawie parametrów ARX
b = [zeros(1, nk), theta_IV(na+1:end)'];
a = [1, theta_IV(1:na)'];

sys = tf(b, a, Tp, 'Variable', 'z^-1');

% Symulacja odpowiedzi modelu ARX dla danych testowych
Y_pred = lsim(sys, input_test, time(split_idx+1:end));

% Obliczenie błędu średniokwadratowego (MSE) dla danych testowych
mse = mean((output_test - Y_pred).^2);
disp(['Błąd średniokwadratowy (MSE) na danych testowych: ', num2str(mse)]);

% Obliczenie wskaźnika dopasowania (Jfit)
Jfit = 100 * (1 - sum((output_test - Y_pred).^2) / sum((output_test - mean(output_test)).^2));
disp(['Wskaźnik dopasowania (J_{fit}): ', num2str(Jfit), '%']);

% Generowanie odpowiedzi skokowej
[step_response, step_time] = step(sys, 0:Tp:7.5);

% Tworzenie wykresu odpowiedzi skokowej
fig1 = figure;
fig1.Theme = "light";
plot(step_time, step_response, 'b');
hold on;

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

% Wykres rzeczywistej odpowiedzi skokowej
plot((1:M-1)*Tp, h_hat_M(2:end), 'r');
legend('$h_m$', '$h$', 'Interpreter', 'latex');
xlabel('Time [s]', 'Interpreter', 'Latex');
% ylabel('Odpowiedź skokowa', 'Interpreter', 'Latex');
% title('Porównanie odpowiedzi skokowej', 'Interpreter', 'Latex');
xlim([0 7.5]);
grid on;
set(gca, 'TickLabelInterpreter', 'latex');

% Wyświetlenie porównania danych rzeczywistych z predykcją modelu ARX (IV)
fig2 = figure;
fig2.Theme = "light";

subplot(2,1,1);
plot(time(split_idx+1:end), output_test, 'b');
hold on;
plot(time(split_idx+1:end), Y_pred, 'r');
legend('$y$', '$y_m$', 'Interpreter', 'latex');
xlabel('Time [s]', 'Interpreter', 'Latex');
% ylabel('Temperatura [C]', 'Interpreter', 'Latex');
% title('Porównanie pomiarów rzeczywistych z predykcją modelu ARX (IV)', 'Interpreter', 'Latex');
grid on;
set(gca, 'TickLabelInterpreter', 'latex');

subplot(2,1,2);
plot(time, input_data, 'k', 'LineWidth', 1);
xlabel('Time [s]', 'Interpreter', 'Latex');
ylabel('Power [W]', 'Interpreter', 'Latex');
% title('Sygnał wejściowy', 'Interpreter', 'Latex');
grid on;
set(gca, 'TickLabelInterpreter', 'latex');

