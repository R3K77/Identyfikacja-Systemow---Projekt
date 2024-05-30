function output = fun_wskazniki(na, nb, nk)
    % Opis funkcji
    

% Wczytanie danych z pliku dryer.dat
data = load('./Dane/dryer.dat');
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

% Ze względu na obliczanie wskaźników jakości modelu  na podstawie porówywania do odpowiedzi na nie zerowe warunki początkowe 
% naley pominać x próbek w celu wyeliminiowania tego błędu i normalizacji wyników
numSkipedSamples = 30;

% Obliczenie błędu średniokwadratowego (MSE) dla danych testowych
mse = mean((output_test(numSkipedSamples:end) - Y_pred(numSkipedSamples:end) ).^2);
disp(['Błąd średniokwadratowy (MSE) na danych testowych: ', num2str(mse)]);

% Obliczenie wskaźnika dopasowania (Jfit)
Jfit = 100 * (1 - sum((output_test(numSkipedSamples:end)  - Y_pred(numSkipedSamples:end) ).^2) / sum((output_test(numSkipedSamples:end)  - mean(output_test(numSkipedSamples:end) )).^2));
disp(['Wskaźnik dopasowania (J_{fit}): ', num2str(Jfit), '%']);

Aic = log(mse) + 4*max(numel(a),numel(b))/numel(output_test(numSkipedSamples:end));
    output = [na,nb,nk,na+nb+nk,mse,Jfit,Aic];
end