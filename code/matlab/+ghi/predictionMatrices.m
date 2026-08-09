function [Sx, Su] = predictionMatrices(plant, N)
%PREDICTIONMATRICES  X = Sx x0 + Su U, con X = [x_0; x_1; ...; x_N].
n = plant.n;  m = plant.m;
Sx = zeros(n * (N + 1), n);
Su = zeros(n * (N + 1), m * N);
Ak = eye(n);
for i = 0:N
    Sx(i*n + (1:n), :) = Ak;
    for j = 0:i-1
        Su(i*n + (1:n), j*m + (1:m)) = plant.A^(i - 1 - j) * plant.B;
    end
    Ak = plant.A * Ak;
end
end
