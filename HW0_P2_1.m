n = 50
x = linspace(0, 2*pi, n)';
dx = x(2) - x(1);
x = [x(1) - dx; x];
x = [x; x(n + 1) + dx]


f = exp(sin(x))


e = ones(n + 2, 1);
M = spdiags([e, -2*e, e], [0 1 2], n, n + 2);
full(M)
df = M*f / (dx^2)

true_df = (cos(x).*cos(x) - sin(x)) .*f

true_df = true_df(2:n + 1)
x = x(2:n + 1)

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