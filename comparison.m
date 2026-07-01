% =========================================================================
% RPCA KAPSAMLI KARŞILAŞTIRMA (Otonom vs AccAltProj vs Candes)
% Not: Çalıştırmadan önce X, L_ex, S_ex, G (varsa) ve nvar workspace'te olmalı
% =========================================================================

% --- 1. Candès (ALM / Convex PCP) ---
fprintf('\n--- 1. Candes ALM (Convex) Calistiriliyor ---\n');
tic; 
[L_can, S_can, G_err_can, L_err_can, S_err_can] = candes_alm(X, its, L_ex, S_ex, G_ex); 
time_can = toc;
fprintf('Bitti. (%.2f sn)\n', time_can);

% --- 2. AccAltProj (Adaptif / Orijinal Teori ile c-bulan) ---
% fprintf('\n--- 2. AccAltProj (Adaptif / Orijinal) Calistiriliyor ---\n');
% mu_acc = 1.0;      
% gamma_acc = 0.5;   
% r_target = 50; % Rakipler icin hala r vermek zorundayiz
% tic; 
% [L_acc_orig, S_acc_orig, G_err_acc_orig, L_err_acc_orig, S_err_acc_orig] = accaltproj_original(X, r_target, mu_acc, gamma_acc, its, L_ex, S_ex, G_ex); 
% time_acc_orig = toc;
% fprintf('Bitti. (%.2f sn)\n', time_acc_orig);

% --- 3. AccAltProj (Sabit c) ---
fprintf('\n--- 3. AccAltProj (Sabit c) Calistiriliyor ---\n');
c_target = nnz(S_ex);
tic; 
[L_acc_fix, S_acc_fix, G_err_acc_fix, L_err_acc_fix, S_err_acc_fix] = accaltproj_fixed_c(X, r_true, c_true, its, L_ex, S_ex, G_ex); 
time_acc_fix = toc;
fprintf('Bitti. (%.2f sn)\n', time_acc_fix);

% =========================================================================
% --- Çizim (Plot) Bölümü ---
% =========================================================================
figure(2); clf;
tightfig = @(f) set(f,'Units','normalized'); 
tightfig(gcf);
set(gcf, 'Position', [0.1, 0.3, 0.8, 0.4]);
lw = 2; ms = 6;

% 1: Kalıntı Hatası (G)
subplot(1,3,1);
semilogy(0:length(G_err_rec)-1, sqrt(G_err_rec), 'g-o', 'LineWidth', 2.5, 'MarkerSize', ms); hold on;
semilogy(0:length(G_err_acc_fix)-1, G_err_acc_fix, 'c-s', 'LineWidth', lw, 'MarkerSize', ms);
% semilogy(0:length(G_err_acc_orig)-1, G_err_acc_orig, 'b-x', 'LineWidth', lw, 'MarkerSize', ms);
semilogy(0:length(G_err_can)-1, G_err_can, 'r-d', 'LineWidth', lw, 'MarkerSize', ms);
yline(1.0, 'r--', 'Optimal Gürültü Tabanı', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
title('Veri Uyumluluğu: Gürültü Tabanı');
xlabel('Yineleme'); ylabel('||A||_F / ||A_0||_F');
grid on;

% 2: Düşük Kerteli Matris (L) Hatası
subplot(1,3,2);
semilogy(0:length(L_err_rec)-1, L_err_rec, 'g-o', 'LineWidth', 2.5, 'MarkerSize', ms); hold on;
semilogy(0:length(L_err_acc_fix)-1, L_err_acc_fix, 'c-s', 'LineWidth', lw, 'MarkerSize', ms);
% semilogy(0:length(L_err_acc_orig)-1, L_err_acc_orig, 'b-x', 'LineWidth', lw, 'MarkerSize', ms);
semilogy(0:length(L_err_can)-1, L_err_can, 'r-d', 'LineWidth', lw, 'MarkerSize', ms);
title('Düşük Kerteli (K) Matris Hatası');
xlabel('Yineleme'); ylabel('||K_0 - K||_F / ||K_0||_F');
grid on;

% 3: Seyrek Matris (S) Hatası
subplot(1,3,3);
semilogy(0:length(S_err_rec)-1, S_err_rec, 'g-o', 'LineWidth', 2.5, 'MarkerSize', ms); hold on;
semilogy(0:length(S_err_acc_fix)-1, S_err_acc_fix, 'c-s', 'LineWidth', lw, 'MarkerSize', ms);
% semilogy(0:length(S_err_acc_orig)-1, S_err_acc_orig, 'b-x', 'LineWidth', lw, 'MarkerSize', ms);
semilogy(0:length(S_err_can)-1, S_err_can, 'r-d', 'LineWidth', lw, 'MarkerSize', ms);
title('Seyrek (S) Matris Hatası');
xlabel('Yineleme'); ylabel('||S_0 - S||_F / ||S_0||_F');
grid on;

legend('Önerilen (Otonom)', 'AccAltProj (Sabit r ve c)', 'Candes PCP', 'Location', 'best');


% =========================================================================
% YARDIMCI VE ÇEKİRDEK FONKSİYONLAR
% =========================================================================

function ret = new_S_rect(X, L_i, m, n, c)
    X_f = X - L_i;
    abs_Xf = abs(X_f(:));
    [~, idx] = maxk(abs_Xf, c);
    [xc, yc] = ind2sub([m, n], idx);
    ns = X_f(idx);
    ret = sparse(xc, yc, ns, m, n);
end

% -------------------------------------------------------------------------
% 1. Candès ALM (Dinamik Durdurma ve Hata Kayıtlı)
% -------------------------------------------------------------------------
function [L_can, S_can, G_err, L_err, S_err] = candes_alm(X, its, L_ex, S_ex, G_ex)
    [m, n] = size(X);
    lambda = 1 / sqrt(max(m, n));
    mu = (m * n) / (4 * sum(abs(X(:))));
    
    norm_X = norm(X, 'fro');
    norm_Lex = norm(L_ex, 'fro');
    norm_Sex = norm(S_ex, 'fro');
    
    if nargin > 5 && ~isempty(G)
        delta = max(1e-7, norm(G, 'fro') / norm_X);
    else
        delta = 1e-7;
    end
    
    S_can = zeros(m, n);
    Y = zeros(m, n);
    L_can = zeros(m, n);
    
    G_err = zeros(1, its + 1); L_err = zeros(1, its + 1); S_err = zeros(1, its + 1);
    
    G_err(1) = norm(X - L_can - S_can, 'fro')^2  / norm(G_ex, 'fro')^2;
    L_err(1) = norm(L_ex - L_can, 'fro') / norm_Lex;
    S_err(1) = norm(S_ex - S_can, 'fro') / norm_Sex;
    
    for k = 1:its
        target_L = X - S_can + (1/mu) * Y;
        [U, Sigma, V] = svd(target_L, 'econ');
        Sigma_shrink = sign(Sigma) .* max(abs(Sigma) - (1/mu), 0);
        L_can = U * Sigma_shrink * V';
        
        target_S = X - L_can + (1/mu) * Y;
        S_can = sign(target_S) .* max(abs(target_S) - (lambda/mu), 0);
        
        residual = X - L_can - S_can;
        Y = Y + mu * residual;
        
        G_err(k+1) = norm(residual, 'fro')^2 / norm(G_ex, 'fro')^2;
        L_err(k+1) = norm(L_ex - L_can, 'fro') / norm_Lex;
        S_err(k+1) = norm(S_ex - S_can, 'fro') / norm_Sex;
        
        % Durdurma kriteri sağlanırsa dizileri kırp ve çık
        % if (norm(residual, 'fro') / norm_X) < delta
        %     G_err = G_err(1:k+1); L_err = L_err(1:k+1); S_err = S_err(1:k+1);
        %     break;
        % end
    end
end

% -------------------------------------------------------------------------
% 2. AccAltProj (Orijinal Teori - Adaptif c bulucu)
% -------------------------------------------------------------------------
function [L_acc, S_acc, G_err, L_err, S_err] = accaltproj_original(X, r, mu, gamma, its, L_ex, S_ex, G_ex)
    [m, n] = size(X);
    beta_init = (1.1 * mu * r) / sqrt(m * n);
    beta = (1.1 * mu * r) / (2 * sqrt(m * n));
    
    sigma1_D = norm(X, 2);
    S_minus1 = X .* (abs(X) > (beta_init * sigma1_D));
    
    [U, Sigma, V] = svd(X - S_minus1, 'econ');
    U = U(:, 1:r); Sigma = Sigma(1:r, 1:r); V = V(:, 1:r);
    L_acc = U * Sigma * V';
    
    Temp = X - L_acc;
    S_acc = Temp .* (abs(Temp) > (beta * Sigma(1,1)));
    
    G_err = zeros(1, its + 1); L_err = zeros(1, its + 1); S_err = zeros(1, its + 1);
    G_err(1) = norm(X - L_acc - S_acc, 'fro')^2  / norm(G_ex, 'fro')^2;
    L_err(1) = norm(L_ex - L_acc, 'fro') / norm(L_ex, 'fro');
    S_err(1) = norm(S_ex - S_acc, 'fro') / norm(S_ex, 'fro');
    
    for k = 1:its
        X_S = X - S_acc;
        UXSV = U' * X_S * V;
        A = X_S * V - U * UXSV; 
        B = X_S' * U - V * UXSV'; 
        [Q1, R1] = qr(A, 0); [Q2, R2] = qr(B, 0);
        
        M_core = [UXSV, R2'; R1, zeros(r, r)];
        [U_c, Sig_c, V_c] = svd(M_core);
        
        sigma_1 = Sig_c(1, 1);
        sigma_r_plus_1 = Sig_c(r+1, r+1); 
        
        U = [U, Q1] * U_c(:, 1:r);
        V = [V, Q2] * V_c(:, 1:r);
        Sigma = Sig_c(1:r, 1:r);
        L_acc = U * Sigma * V';
        
        zeta = beta * (sigma_r_plus_1 + (gamma^k) * sigma_1);
        Temp = X - L_acc;
        S_acc = Temp .* (abs(Temp) > zeta);
        
        G_err(k+1) = norm(X - L_acc - S_acc, 'fro')^2 / norm(G_ex, 'fro')^2;
        L_err(k+1) = norm(L_ex - L_acc, 'fro') / norm(L_ex, 'fro');
        S_err(k+1) = norm(S_ex - S_acc, 'fro') / norm(S_ex, 'fro');
    end
end

% -------------------------------------------------------------------------
% 3. AccAltProj (Sabit c)
% -------------------------------------------------------------------------
function [L_acc, S_acc, G_err, L_err, S_err] = accaltproj_fixed_c(X, r, c, its, L_ex, S_ex, G_ex)
    [m, n] = size(X);
    [U, Sigma, V] = svd(X, 'econ');
    U = U(:, 1:r); Sigma = Sigma(1:r, 1:r); V = V(:, 1:r);
    L_acc = U * Sigma * V';
    S_acc = sparse(m, n);
    
    G_err = zeros(1, its + 1); L_err = zeros(1, its + 1); S_err = zeros(1, its + 1);
    G_err(1) = norm(X - L_acc - S_acc, 'fro')^2 / norm(G_ex, 'fro')^2;
    L_err(1) = norm(L_ex - L_acc, 'fro') / norm(L_ex, 'fro');
    S_err(1) = norm(S_ex - S_acc, 'fro') / norm(S_ex, 'fro');
    
    for k = 1:its
        X_S = X - S_acc;
        UXSV = U' * X_S * V;
        A = X_S * V - U * UXSV; 
        B = X_S' * U - V * UXSV'; 
        [Q1, R1] = qr(A, 0); [Q2, R2] = qr(B, 0);
        
        M_core = [UXSV, R2'; R1, zeros(r, r)];
        [U_c, Sig_c, V_c] = svd(M_core);
        
        U = [U, Q1] * U_c(:, 1:r);
        V = [V, Q2] * V_c(:, 1:r);
        Sigma = Sig_c(1:r, 1:r);
        L_acc = U * Sigma * V';
        
        S_acc = new_S_rect(X, L_acc, m, n, c);
        
        G_err(k+1) = norm(X - L_acc - S_acc, 'fro')^2 / norm(G_ex, 'fro')^2;
        L_err(k+1) = norm(L_ex - L_acc, 'fro') / norm(L_ex, 'fro');
        S_err(k+1) = norm(S_ex - S_acc, 'fro') / norm(S_ex, 'fro');
    end
end
