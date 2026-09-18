n = 100
x = linspace(0, 5, n)';
dx = x(2) - x(1);

max_entries = (n-2) * 3 + 2

rows = zeros(max_entries, 1);
cols = zeros(max_entries, 1);
vals = zeros(max_entries, 1);
k = 1;

for i = 1:n
    if i == 1 || i == n
        rows(k) = i
        cols(k) = i
        vals(k) = 1
        k = k + 1
        continue
    end

    rows(k) = i;
    cols(k) = i - 1;
    v = 2 - dx*sin(x(i));

    if v ~= 0
        vals(k) = v;
    end

    k = k + 1;

    rows(k) = i;
    cols(k) = i;
    v = 2*(dx^2) - 4;
    if v ~= 0
        vals(k) = v;
    end 

    k = k + 1;

    rows(k) = i;
    cols(k) = i + 1;
    v = 2 + dx*sin(x(i));
    if v ~= 0
        vals(k) = v;
    end 

    k = k + 1;
end

M = sparse(rows, cols, vals, n, n)

disp(full(M))

u = @(X) sin(X)
f = @(X) sin(X)*cos(X);

den = 2*dx^2

F = [u(x(1)) / den; sin(x(2:n-1)) .* cos(x(2:n-1)); u(x(n)) / den]

U = (M \ F) .* den

true_U = sin(x)

color_2 = "#009E73"
color_inf = "#D55E00"

figure
theme(figure,"light")
plot(x, U, '-', 'Color', color_2, 'LineWidth', 3)
hold on
plot(x, true_U, '--', 'Color', color_inf, 'LineWidth', 3)
hold off
xlabel('X', 'FontSize', 18)
ylabel('Y', 'FontSize', 18)
legend('U numerical', 'U actual', 'FontSize', 18)
grid on

set(gca, 'FontSize', 16)   % <-- enlarges the tick labels