clc; clear; clear all;

% Wczytanie danych z pliku dryer.dat
data = load('../Dane/dryer.dat');
Tp = 0.08;
time = (0:size(data, 1)-1) * Tp;

% Załadowanie do zmiennych:
input_data = data(:,1);    % Moc grzałki wyrażona w [W]
output_data = data(:,2);   % Temperatura wyrażona w [C]

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

% Obliczanie Jfit dla predykcji
y_true = output_data(max(na, nb+nk):N);
y_pred = y_pred(max(na, nb+nk):N);
Jfit_pred = 100 * (1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2));

% Tworzenie wykresów estymat parametrów
figure;
for i = 1:na + nb
    subplot(na + nb, 1, i);
    plot(theta_history(:, i));
    ylabel(['$\theta_', num2str(i), '$'], 'Interpreter', 'latex');
    xlabel('Sample [n]', 'Interpreter', 'latex');
    set(gca,'TickLabelInterpreter','latex');
    % title(['$\theta_', num2str(i), '$'], 'Interpreter', 'latex');
    grid on;
end

% Obliczenie transmitancji na podstawie parametrów ARX
b = [zeros(1, nk), theta(na+1:end)'];
a = [1, theta(1:na)'];

sys = tf(b, a, Tp, 'Variable', 'z^-1');

% Symulacja odpowiedzi modelu ARX dla danych testowych
Y_m = lsim(sys, input_data, time);

% Ze względu na obliczanie wskaźników jakości modelu  na podstawie porówywania do odpowiedzi na nie zerowe warunki początkowe 
% naley pominać x próbek w celu wyeliminiowania tego błędu i normalizacji wyników
numSkipedSamples = 30;

% Obliczanie Jfit dla odpowiedzi modelu
Jfit_sys = 100 * (1 - sum((output_data(numSkipedSamples:end)  - Y_m(numSkipedSamples:end) ).^2) / sum((output_data(numSkipedSamples:end)  - mean(output_data(numSkipedSamples:end) )).^2));

% Tworzenie wykresu rzeczywistych i predykowanych wartości wyjściowych
figure;
plot(max(na, nb+nk):N, y_true, 'b', max(na, nb+nk):N, y_pred, 'r--');
hold on;
plot((time(numSkipedSamples:end))/Tp, Y_m(numSkipedSamples:end), '-.');
plot((time(1:numSkipedSamples-1))/Tp, Y_m(1:numSkipedSamples-1) , '-.k');
% title('True vs Predicted Output');
xlabel('Sample [n]', 'Interpreter', 'latex');
set(gca,'TickLabelInterpreter','latex');
legend('$y$', '$\hat{y}$', '$y_m$', 'Interpreter', 'latex');
grid on;

% Wyświetlanie końcowych estymat parametrów i Jfit
disp('Estimated parameters:');
disp(theta);

disp(['Jfit (Prediction): ', num2str(Jfit_pred)]);
disp(['Jfit (System Response): ', num2str(Jfit_sys)]);
% Obliczenie błędu średniokwadratowego (MSE) dla predyktora
mse_pred = mean((y_true - y_pred).^2);
disp(['Błąd średniokwadratowy (MSE) dla predyktora: ', num2str(mse_pred)]);

% Obliczenie błędu średniokwadratowego (MSE) dla modelu symulacyjnego
mse_sys = mean((output_data(numSkipedSamples:end) - Y_m(numSkipedSamples:end) ).^2);
disp(['Błąd średniokwadratowy (MSE) dla modelu symulacyjnego: ', num2str(mse_sys)]);

% Dodanie funkcjonalności porównania odpowiedzi skokowych

% Ekstrakcja parametrów modelu ARX
a = theta(1:na);
b = theta(na+1:end);

% Obliczenie transmitancji na podstawie parametrów ARX
b_trans = [zeros(1, nk), b'];
a_trans = [1, a'];

sys = tf(b_trans, a_trans, Tp, 'Variable', 'z^-1');

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
legend('$h$', '$h_m$', 'Interpreter', 'latex');
xlabel('Time [$nT_p$]','Interpreter','latex');
set(gca,'TickLabelInterpreter','latex');
% ylabel('');
% title('Porównanie odpowiedzi skokowej');
xlim([0 7.5]);
grid on;
