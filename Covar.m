function C=Covar(D,tau)

Y = D(:,1);
U = D(:,2);
N = size(Y,1);
Yp = zeros(N,1);
MU = (1/N)*sum(U);
MY = (1/N)*sum(Y);
Ud = U;
Yd = Y;

if (tau >= 0)
    Yp(1:(N-tau)) = Yd((1+tau):N);
else
    Yp((1-tau):N) = Yd(1:(N+tau));
end

CYU = (1/N)*(Ud'*Yp);
% CYU = (1/(N-abs(tau)))*(Ud'*Yp); % Estymator nieobciążony funkcji korelacji
C = CYU;