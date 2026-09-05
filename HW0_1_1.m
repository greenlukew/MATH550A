n = 20
x = linspace(0, 2*pi, n)';
dx = x(1) - x(2)

f = exp(sin(x))


e = ones(n, 1);
M = spdiags([-e, zeros(n,1), e], [-1 0 1], n, n);

M(1, 1) = -1;
M(n, n) = 1;

df = M*f / dx

df(2:n-1) = df(2:n-1) / 2;

true_df = -cos(x) .*f

numeric_color = "#009E73"
exact_color = "#D55E00"

figure
theme(figure,"light")
plot(x, true_df, '-', 'Color', exact_color, 'LineWidth', 3)
hold on
plot(x, df, '--', 'Color', numeric_color, 'LineWidth', 3)
hold off
xlabel('X')
ylabel('Y')
legend('Exact solution', 'Numerical solution')
grid on