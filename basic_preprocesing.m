clc; clear; clear all

% Load data from dryer.dat
data = load('Dane/dryer.dat');
sampling_time = 0.08; %sampling time powinnien być 0.08s
time = (0:size(data, 1)-1) * sampling_time;

% Załadowanie do zmiennych:
input_u = data(:,1);    % Moc grzałki wyrażona w [W]
output_y = data(:,2);   % Temperatura wyrażona w [C]

all_data = iddata(output_y, input_u, sampling_time);

%% Obrazowanie danych
% 
figure(1);
plot(time, input_u,'Color',"#D95319");
xlabel('Time [s]');
ylabel('Input Power [W]');
grid on;

figure(2);
plot(time, output_y);
xlabel('Time [s]');
ylabel('Temperature [C]');
grid on;


%% Podzielenie danych na zbiory uczący i testowy
u_train = input_u(1:length(input_u)/2);
y_train = output_y(1:length(output_y)/2);

u_test = input_u(length(input_u)/2+1:end);
y_test = output_y(length(output_y)/2+1:end);


%% Usunięcie trendów 

u_train = detrend(u_train);
y_train = detrend(y_train);

u_test = detrend(u_test);
y_test = detrend(y_test);

data = iddata(y_train, u_train, sampling_time);
data_test = iddata(y_test, u_test, sampling_time);


%% Określenie opóźnienia wejścia
figure;

FIRModel = impulseest(data);
clf
h = impulseplot(FIRModel);
setoptions(h, 'ConfidenceRegionNumber', 3);
% Prawym przyciskiem na wykresie "Charateristics" -> "Confidence region"
% Opóźnienie czasowe jest równe pierwszemu indeksowi, dla którego wartość
% jest większa od obszaru ufności (confidence region)
% Obszar ufności wynosi trzy wartości odchylenia standardowego.
% W naszym przypadku opóźnienie wynosi 2.4 [s] (trzy próbki).
% Odnośnie tego czasu to chyba trzy próbki to 0.24s, a nie 2.4s.
%bo Tp = 0.08s, a nie 0.8s.


%% Określenie stopnia modelu

% Wykorzystanie funkcji arxstruc do określenia stopnia modelu
% Sprawdzenie 100 modeli od 1 do 10 rzędów zarówno dla licznika jak i mianownika modelu
V = arxstruc(data, data_test, struc(1:10, 1:10, 3)); % 3 - liczba próbek odchylenia standardowego

% Wybór najlepszego modelu
nn = selstruc(V, 0)

% nn = 10, 4, 3 - najlepsze dopasowanie
% nn = 4, 4, 3 - najlepsze dopasowanie dla metody najmniejszych kwadratów

% nns = selstruc(V) % interaktywny wybór modelu
% Wybieramy model 4, 4, 3

%% Sprawdzenie biegunów i zer modelu

th4 = arx(data, [4 4 3]);
figure; % odkomentować
h = iopzplot(th4); % odkomentować
setoptions(h, 'ConfidenceRegionNumber', 3); % odkomentować
% Analizując wykres można zauważyć, że obszar ufności dwóch sprzężonych biegunów i zer nakłada się na siebie, 
% co ozancza, że prawdopodobnie się zniosą.
% Dlatego wybieramy model rzędu 2, 2, 3.

% Sprawdzenie biegunów i zer modelu rzędu 2, 2, 3. 
th2 = arx(data, [2 2 3]);
figure; % odkomentować
h = iopzplot(th2); % odkomentować
setoptions(h, 'ConfidenceRegionNumber', 3); % odkomentować

%% Sprawdzenie jakości modelu
% Sprawdzamy przy użyciu funkcji compare, jak oba modele radzą sobie z
% przewidywaniem danych testowych.
figure;
compare(data_test, th4, th2);

% Możemy odczytać, że zmniejszając rząd modelu (z 4, 4, 3 na 2, 2, 3) nie pogorszyliśmy znacząco jakości modelu.
print('-f1','\Zrzuty ekranu\1_input','-dpdf','-bestfit')
