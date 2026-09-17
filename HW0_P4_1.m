n = 100;
x = linspace(0, 5, n)';
y = linspace(0, 5, n)';
h = x(2) - x(1);

u = @(X, Y) sin(X)*cos(Y);

rows = [];
cols = [];
vals = [];

F = zeros(n * n, 1);
true_U = zeros(n * n, 1);

for j = 1:n
    for i = 1:n 
        center = (j - 1)*n + i;

        if j == 1 || j == n || i == 1 || i == n
            rows = [rows; center];
            cols = [cols; center];
            vals = [vals; 1];
            F(center) = u(x(j), y(n - i + 1)) / h^2;
            true_U(center) = u(x(j), y(n - i + 1));
            continue
        end

        below = center + 1;
        above = center - 1;
        left = (j - 2)*n + i;
        right = j*n + i;

        rows = [rows; center];
        cols = [cols; center];
        vals = [vals; -4];

        rows = [rows; center];
        cols = [cols; below];
        vals = [vals; 1];

        rows = [rows; center];
        cols = [cols; above];
        vals = [vals; 1];

        rows = [rows; center];
        cols = [cols; left];
        vals = [vals; 1];

        rows = [rows; center];
        cols = [cols; right];
        vals = [vals; 1];

        F(center) = -2*u(x(j), y(n - i + 1));
        true_U(center) = u(x(j), y(n - i + 1));
    end
end

l = n*n;

M = sparse(rows, cols, vals, l, l);

U = reshape((M \ F) .* h^2, n, n);
true_U = reshape(true_U, n, n);


color_2 = "#009E73";
color_inf = "#D55E00";

[X, Y] = meshgrid(x, y);

figure;

subplot(1,2,1);
contourf(X, Y, U, 100);
xlabel('x'); ylabel('y');
title('Numeric approximation');
colorbar;
axis equal;

subplot(1,2,2);
contourf(X, Y, true_U, 100);
xlabel('x'); ylabel('y');
title('Solution');
colorbar;
axis equal;