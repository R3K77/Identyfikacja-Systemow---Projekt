clc; clear; clear all;


%-----estymacja parametryczna LS i IV dla modelu dyskretnego (dane: (u,y))----
%----definicje wstępne:----
data = load('Dane/dryer.dat');
Tp = 0.8; %okres próbkowania w [s]
t = (0:size(data, 1)-1) * Tp;

% Załadowanie do zmiennych:
u = data(:,1);    % Moc grzałki wyrażona w [W]
y = data(:,2);   % Temperatura wyrażona w [C]

u = detrend(u);
y = detrend(y);

N = size(y,1); %liczba próbek z zbiorze danych pomiarowych
M = floor(N/2); %liczba próbek brana do estymacji
ue = u(1:M); %podwektor próbek wejścia 'u' do celów estymacji
ye = y(1:M); %podwektor próbek wyjścia 'y' do celów estymacji
te = t(1:M); %podwektor czasu do celów estymacji
uv = u(M+1:N); %podwektor próbek wejścia 'u' do celów weryfikacji
yv = y(M+1:N); %podwektor próbek wyjścia 'y' do celów weryfikacji
tv = t(M+1:N); %podwektor czasu do celów weryfikacji

% Tworzenie obiektu iddata
data_id = iddata(y, u, Tp);

% Estymacja modelu pierwszego rzędu z opóźnieniem czasowym (FOPTD)
model = tfest(data_id, 1, 0); % 1 - rząd liczby, 0 - rząd mianownika (bez zera)

% Wyświetlenie parametrów modelu
disp(model);

% porównanie odpowiedzi modelu z danymi
figure;
compare(data_id, model);
