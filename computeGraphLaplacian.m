function L = computeGraphLaplacian(A)
D = diag(sum(A,1));
L = D-A;


