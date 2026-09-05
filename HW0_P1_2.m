function E = compute_error(n)
    x = linspace(0, 2*pi, n)';
    dx = x(2) - x(1);
    
    f = exp(sin(x));
    
    
    e = ones(n, 1);
    M = spdiags([-e, zeros(n,1), e], [-1 0 1], n, n);
    
    M(1, 1) = -1;
    M(n, n) = 1;
    
    df = M*f / dx;
    
    df(2:n-1) = df(2:n-1) / 2;
    
    true_df = cos(x) .*f;
    
    df_diff = true_df - df;
    
    err_2 = sqrt(dot(df_diff, df_diff)) / sqrt(dot(df, df));
    
    err_inf = max(abs(df_diff)) / max(abs(df));
    
    E = [err_2, err_inf];
end

nvec = 10:10:100;
E = zeros(numel(nvec), 3);

for k = 1:numel(nvec)
    err = compute_error(nvec(k))
    E(k, :) = [log(nvec(k)), log(err(1)), log(err(2))]
end 

color_2 = "#009E73"
color_inf = "#D55E00"

X = E(:, 1)
Y2 = E(:, 2)
Yinf = E(:, 3)

figure
theme(figure,"light")
plot(X, Y2, '-', 'Color', color_2, 'LineWidth', 3)
hold on
plot(X, Yinf, '--', 'Color', color_inf, 'LineWidth', 3)
hold off
xlabel('X')
ylabel('Y')
legend('2 norm', 'Inf norm')
grid on