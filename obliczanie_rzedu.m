wyniki =[]; % tabela z wynikami, parametry kolejno: 
% na, nb, nk, na+nb+k, mse, Jfit , AIC
% otworzyc zmienna i w variables mozna posorotwac wedlug danego parametru
for na =1:6
    for nb = 1:6
            temp = fun_wskazniki(na,nb,3);
            wyniki = [wyniki;temp];
    end
end     
