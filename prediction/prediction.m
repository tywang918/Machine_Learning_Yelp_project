beta_0 = -1.707240;
beta(1, 1) = 0.047583; 
beta(2, 1) = -0.042175; 
beta(3, 1) = 0.022985; 
beta(4, 1) = -0.203906;
beta(5, 1) = 0.107342;
beta(6, 1) = 0.102000;
beta(7, 1) = -0.125309;
beta(8, 1) = 0.058917;
beta(9, 1) = -0.060631;

A = beta' * result + beta_0 * ones(1, 40);
A = A';

for i = 1 : 40
    A(i, 1) = exp(A(i, 1));
    A(i, 1) = A(i, 1)/(A(i, 1) + 1);
end

index = A;
xlswrite('index.csv', index);
