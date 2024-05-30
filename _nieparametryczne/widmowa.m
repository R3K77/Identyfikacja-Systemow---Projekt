clc; clear; clear all;

% Wczytanie danych z pliku dryer.dat
data = load('../Dane/dryer.dat');
Tp = 0.08;
time = (0:size(data, 1)-1) * Tp;

% Załadowanie do zmiennych:
u = data(:,1);    % Moc grzałki wyrażona w [W]
y = data(:,2);   % Temperatura wyrażona w [C]

u = detrend(u);
y = detrend(y);

all_data = iddata(y, u, Tp);

%% Obrazowanie danych
% figure;
% subplot(2,1,1);
% plot(time, u);
% xlabel('Time [s]');
% ylabel('Input Power [W]');
% title('Input Power vs Time');
% 
% subplot(2,1,2);
% plot(time, y);
% xlabel('Time [s]');
% ylabel('Temperature [°C]');
% title('Temperature vs Time');

%% Podzielenie danych na zbiory uczący i testowy
u_train = u(1:floor(length(u)/2));
y_train = y(1:floor(length(y)/2));

u_test = u(floor(length(u)/2)+1:end);
y_test = y(floor(length(y)/2)+1:end);

%% Tworzenie obiektów iddata dla zbiorów uczącego i testowego
data_train = iddata(y_train, u_train, Tp);
data_test = iddata(y_test, u_test, Tp);

%% Analiza widmowa

TF = 2.5*Tp/(2*pi)
N = size(u, 1);
Mw = N/10;  % wybór długości okna

for i=0:N-1
    ruuP(i+1) = Covar([u u],i);
    if (i<=Mw)
        ruuP(i+1) = ruuP(i+1)*0.5*(1+cos(pi*i/Mw));    % okno Hanniga
    else
        ruuP(i+1) = ruuP(i+1)*0.0;
    end
end

Ruu = [ruuP(1:Mw+1) zeros(1,2*N-2*Mw-2) ruuP(Mw+1:-1:2)];    % długość 2N-1

for i=0:N-1
    ryyP(i+1) = Covar([y y],i);
    if (i<=Mw)
        ryyP(i+1) = ryyP(i+1)*0.5*(1+cos(pi*i/Mw));    % okno Hanniga
    else
        ryyP(i+1) = ryyP(i+1)*0.0;
    end
end
Ryy = [ryyP(1:Mw+1) zeros(1,2*N-2*Mw-2) ryyP(Mw+1:-1:2)];    % długość 2N-1

for i=0:N-1
    ryuP(i+1) = Covar([y u],i);
    if (i<=Mw)
        ryuP(i+1) = ryuP(i+1)*0.5*(1+cos(pi*i/Mw));    % okno Hanninga
    else
        ryuP(i+1) = ryuP(i+1)*0.0;
    end
end
for i=0:N-1
    j = i-(N-1);
    ryuN(i+1) = Covar([y u],j);
    if (abs(j)<=Mw)
        ryuN(i+1) = ryuN(i+1)*0.5*(1+cos(pi*j/Mw));    % okno Hanninga
    else
        ryuN(i+1) = ryuN(i+1)*0.0;
    end
end

Ryu = [ryuP(1:Mw+1) zeros(1,2*N-2*Mw-2) ryuN(N-Mw:N-1)];    % długość 2N-1

PHI_uu = Tp*fft(Ruu);        % estymata gęstości widmowej mocy sygnału u
PHI_yy = Tp*fft(Ryy);        % estymata gęstości widmowej mocy sygnału y
PHI_yu = Tp*fft(Ryu);        % estymata gęstości widmowej mocy sygnałów y i u
AmpPHI_yu = abs(PHI_yu);     % wyznaczenie modułów dla liczb zespolonych
YN = Tp*fft(y);              % transformata DFT sekwencji próbek {y(n)}
UN = Tp*fft(u);              % transformata DFT sekwencji próbek {u(n)}

ETFE = YN./UN;               % ilorazy transformat FFT sygnałów y i u
AmpETFE = abs(ETFE);         % wyznaczenie modułów dla ilorazów
LmETFE = 20*log10(AmpETFE);  % moduł logarytmiczny w [dB]
ArgETFE = unwrap(angle(ETFE))*180/pi;  % wyznaczenie kątów fazowych w [deg]

Nmm = size(AmpETFE,1);       % określenie liczby elementów
dOmegam = 2*pi/Nmm;          % bin pulsacji unormowanej w [rad]
k = (0:1:Nmm-1)';            % wektor indeksów pulsacji
Omegam = dOmegam*k;          % wektor pulsacji unormowanych w [rad]
omegam = Omegam/Tp;          % wektor pulsacji w [rad/s]
indm = floor(Nmm/2);         % wyznaczenie indeksu połowy pulsacji
omega2m = omegam(1:indm);    % wektor pulsacji z połowy zakresu (do wykresów)

hatGs = PHI_yu./PHI_uu;      % ilorazy gęstości widmowych mocy
AmphatGs = abs(hatGs);       % wyznaczenie modułów dla ilorazów
LmhatGs = 20*log10(AmphatGs); % moduł logarytmiczny w [dB]
ArghatGs = unwrap(angle(hatGs))*180/pi; % wyznaczenie kątów fazowych w [deg]

Nm = size(AmphatGs,2);       % określenie liczby elementów
dOmega = 2*pi/Nm;            % bin pulsacji unormowanej w [rad]
k = (0:1:Nm-1)';             % wektor indeksów pulsacji
Omega = dOmega*k;            % wektor pulsacji unormowanych w [rad]
omega = Omega/Tp;            % wektor pulsacji w [rad/s]
ind = floor(Nm/2);           % wyznaczenie indeksu połowy pulsacji
omega2 = omega(1:ind);       % wektor pulsacji z połowy zakresu (do wykresów)


% Wyświetlenie wyników analizy widmowej - charakterystyki Bodego
fig2 = figure;
fig2.Theme = "light";
subplot(2,1,1);
semilogx(omega2m, LmETFE(1:indm), 'r', 'LineWidth', 1);
hold on;
semilogx(omega2, LmhatGs(1:ind), 'b', 'LineWidth', 1);
xlabel('Frequency [rad/s]' , Interpreter='Latex');
ylabel('Magnitude [dB]' , Interpreter='Latex');
grid on;
set(gca, 'TickLabelInterpreter', 'latex');

% title('Bode Magnitude Estimate');
legend('ETFE', 'Smoothed ETFE', Interpreter='Latex');
subplot(2,1,2);
semilogx(omega2m, ArgETFE(1:indm), 'r', 'LineWidth', 1);
hold on;
grid on;
semilogx(omega2, ArghatGs(1:ind), 'b', 'LineWidth', 1);
xlabel('Frequency [rad/s]' , Interpreter='Latex');
ylabel('Phase [deg]' , Interpreter='Latex');
% title('Bode Phase Estimate, chyba bez sensu to wklejac do sprawka');
legend('ETFE', 'Smoothed ETFE', Interpreter='Latex');
set(gca, 'TickLabelInterpreter', 'latex');
