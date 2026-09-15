n = 250
x = linspace(0, 5, n)';
dx = x(2) - x(1);

alpha = 5
beta = 2

maxEntries = 3*(n-2)
rows = zeros(maxEntries, 1);
cols = zeros(maxEntries, 1);
vals = zeros(maxEntries, 1);
k = 1;

for i = 1:(n-2)
    rows(k) = i;
    cols(k) = i;
    v = 2 - dx*sin(x(i));
    if v ~= 0
        vals(k) = v / (2*dx^2);
    end
    
    k = k + 1;

    rows(k) = i;
    cols(k) = i + 1;
    v = 2*(dx^2) - 4;
    if v ~= 0
        vals(k) = v / (2*dx^2);
    end 

    k = k + 1;

    rows(k) = i;
    cols(k) = i + 2;
    v = 2 + dx*sin(x(i));
    if v ~= 0
        vals(k) = v / (2*dx^2);
    end 

    k = k + 1;
end

U = sin(x);


M = sparse(rows, cols, vals, n-2, n);

F = M*U;

disp(F)

numeric_color = "#009E73"
exact_color = "#D55E00"

figure
theme(figure,"light")
plot(x, U, '-', 'Color', exact_color, 'LineWidth', 3)
hold on
plot(x(2:(n-1)), F, '--', 'Color', numeric_color, 'LineWidth', 3)
hold off
xlabel('X')
ylabel('Y')
legend('sin(x)', 'f(x)')
grid on