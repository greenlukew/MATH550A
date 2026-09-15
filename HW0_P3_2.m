function E = compute_error(n)
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

    F_actual = sin(x(2:(n-1))) .* cos(x(2:(n-1)))

    F_diff = F_actual - F;

    E = sqrt(dot(F_diff, F_diff)) / sqrt(dot(F, F));
end

nvec = 10:10:1000;
E = zeros(numel(nvec), 3);

for k = 1:numel(nvec)
    err = compute_error(nvec(k))
    E(k, :) = [log(nvec(k)), log(err), err]
end 

color_2 = "#009E73"
color_inf = "#D55E00"

X = E(:, 1)
Y2 = E(:, 2)

figure
theme(figure,"light")
plot(X, Y2, '-', 'Color', color_2, 'LineWidth', 3)
xlabel('Log(n)')
ylabel('Err (2 norm)')
grid on
