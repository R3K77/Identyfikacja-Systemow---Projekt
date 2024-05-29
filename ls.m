clc; clear; clear all;

% Wczytanie danych z pliku dryer.dat
data = load('Dane/dryer.dat');
Tp = 0.08;
time = (0:size(data, 1)-1) * Tp

% Załadowanie do zmiennych:
input_data = data(:,1);    % Moc grzałki wyrażona w [W]
output_data = data(:,2);   % Temperatura wyrażona w [C]

clc; clear; clear all;

t = (0:Tp:80-Tp);
N = 1000;
M = floor(N/2);
d = 2;

% Podział danych na zestawy treningowe i testowe (50/50)
input_train = input_data(1:M);
output_train = output_data(1:M);
input_test = input_data(M+1:N);
output_test = output_data(M+1:N);

t_estymowane = t(1:M);
t_weryfikacja = t(M+1:N);

%%Macierz regresji 
Phi = zeros(M,d);
for i=4:M
    Phi(i,:) = [output_train(i-1) input_train(i-3)]; %estymator
end
p_hat_LS = pinv(Phi)*output_train;
k_hat = p_hat_LS(2)/(1-p_hat_LS(1));
T_hat = -Tp/(log(p_hat_LS(1)));
wektor_hat = [k_hat T_hat];

disp("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
fprintf("Wartości do symulacji NIEZNANE \n")
fprintf("Otrzymane white wartości przez LS k = %.4f T %.4f)\n", wektor_hat(1), wektor_hat(2));
disp("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

%% Tworzenie predyktora jednokrokowego white
Phi_pred = zeros(N-M,d);
for i=2:(N-M)
    Phi_pred(i-1,:) = [output_test(i-1) input_test(i-1)];
end
y_hat_pred = Phi_pred*p_hat_LS;
Gsim = tf(k_hat,[T_hat 1]);
y_sim = lsim(Gsim,input_test,t_weryfikacja,'zoh');

%Plot
figure
hold on
plot(t_weryfikacja,y_hat_pred,'-o') % PREDYKTOR
plot(t_weryfikacja,output_test,'-',LineWidth=2) % ZMIERZONA ODPOWIEDZ
plot(t_weryfikacja,y_sim,'.') %SYMULOWANA
title("Wykresy")
legend('PREDYKTOR','ZMIERZONA ODPOWIEDZ','SYMULOWANA');
hold off

% Obliczenie wskaźnika dopasowania (Jfit)
Jfit = 100 * (1 - sum((output_test - y_hat_pred).^2) / sum((output_test - mean(output_test)).^2));
disp(['Wskaźnik dopasowania (J_{fit}): ', num2str(Jfit), '%']);