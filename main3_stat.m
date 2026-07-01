use_svd = false;
is_complex = false;
its = 150;

%% Initialization (Top-Down Overestimate)
tic;
r_init = 300; % Deliberate overestimate
r_cur = r_init; 
c_cur = 1; 
r_cost = 1.5;
c_cost = 1.5;

if use_svd
    [U_init, D_init, V_init] = svds(X, r_init);
else
    % Initialization (SVD-Free Randomized Power Method)
    
    % 1. Create a random projection matrix
    if is_complex
        Omega = randn(n, r_cur) + 1i*randn(n, r_cur);
    else
        Omega = randn(n, r_cur);
    end
    
    % 2. Power Iterations (Explicit q loop for guaranteed stability)
    q = 2; % q=1 or 2 is standard for robust PCA
    Y = X * Omega;
    [Q, ~] = qr(Y, 0);
    
    for iter = 1:q
        Y = X' * Q;
        [Q_v, ~] = qr(Y, 0);
        Y = X * Q_v;
        [Q, ~] = qr(Y, 0);
    end
    
    % 3. Extract the initial orthogonal bases and singular values
    U_init = Q;
    V_init = Q_v;
    
    % Project X onto the stabilized bases to get the small core matrix
    R_core = U_init' * X * V_init;
    
    % Do a tiny, instantaneous SVD on the r x r core
    [u_tiny, D_init, v_tiny] = svd(R_core);
    
    % Rotate the initial bases to align perfectly with the principal components
    U_init = U_init * u_tiny;
    V_init = V_init * v_tiny;
    
    U_cur = U_init; 
    D_cur = D_init; 
    V_cur = V_init; 
    L_cur = U_cur * D_cur * V_cur';
end

U_cur = U_init(:, 1:r_cur); 
D_cur = D_init(1:r_cur, 1:r_cur); 
V_cur = V_init(:, 1:r_cur); 
L_cur = U_cur * D_cur * V_cur';
S_cur = sparse(m, n);
G_cur = X - L_cur;


%% Start Tracking
r_rec = zeros(1, its);
c_rec = zeros(1, its);
norm_res_rec = zeros(1, its); 

% New Tracking Arrays for Relative Errors
L_err_rec = zeros(1, its);
S_err_rec = zeros(1, its);
G_err_rec = zeros(1, its);

dof_mult = 1;
if is_complex
    dof_mult = 2;
end

P_sparse = log(N_pix);    % BIC penalty aligns with Universal Outlier Threshold
hold_rank_pruning = 1;

%% Momentum parameters
beta = 0.5;                 % Nesterov coefficient (0 = vanilla, ~0.5-0.9 typical)
DirU_prev = [];             % Empty = no history yet
DirV_prev = [];

floor_nvar = 0;
floor_sigma = 0;
nvar = max(floor_nvar,nvar_true);
sigma_est = sqrt(nvar);

for it = 1:its
    % 0. Estimate nvar on the fly using MAD (exclude current S estimate)
    R_raw = X - L_cur;
    %sigma_est = median(abs(R_raw(:))) / 0.6745;
    %nvar = sigma_est^2;

    r_before = r_cur;

    % ==========================================================
    % 1. Single Manifold Step with Momentum
    % ==========================================================
    [U_t, D_t, V_t, DirU_new, DirV_new] = ...
        LS_step(X, U_cur, D_cur, V_cur, L_cur, S_cur, m, n, ...
                beta, DirU_prev, DirV_prev);
    
    if it > hold_rank_pruning
        % 1. Calculate the exact Marchenko-Pastur limit
        tau_MP = sqrt(nvar) * (sqrt(m) + sqrt(n)) * r_cost;
        
        % 2. Extract approximate singular values directly from D_t
        approx_sigmas = abs(diag(D_t));
        
        % 3. Count valid ranks directly against the threshold
        valid_ranks = sum(approx_sigmas > tau_MP);
        r_target = max(valid_ranks, 1); % Ensure it never drops to 0
        
        % 4. Execute direct pruning, sorted by singular value magnitude
        if r_target < r_cur
            [~, ord] = sort(approx_sigmas, 'descend');
            keep = ord(1:r_target);
            
            U_cur = U_t(:, keep);
            V_cur = V_t(:, keep);
            D_cur = D_t(keep, keep);
            r_cur = r_target;
            
            % Recalculate L_cur with the pruned bases
            L_cur = U_cur * D_cur * V_cur'; 
        else
            % Accept the stepped matrices immediately
            U_cur = U_t; D_cur = D_t; V_cur = V_t;
            L_cur = U_cur * D_cur * V_cur';
        end
    else
        % BURN-IN PHASE
        U_cur = U_t; D_cur = D_t; V_cur = V_t;
        L_cur = U_cur * D_cur * V_cur';
    end

    % Update momentum buffers (reset if rank changed)
    if r_cur == r_before
        DirU_prev = DirU_new;
        DirV_prev = DirV_new;
    else
        DirU_prev = [];
        DirV_prev = [];
    end

    % ==========================================================
    % 3. Absolute Hard Thresholding (S) Update
    % ==========================================================
    raw_error_matrix = X - L_cur;
    
    % --- STEP 1: Calculate Universal Outlier Threshold ---
    % The absolute maximum amplitude that Gaussian noise can reach.
    % c_cost acts as the relaxation weight for real-world spatial correlation.
    tau_S = c_cost * max(sigma_est, floor_sigma) * sqrt(2 * log(N_pix));
    
    % --- STEP 2: O(N) Logical Thresholding ---
    % Find all linear indices where the residual exceeds the noise floor
    idx_sparse = find(abs(raw_error_matrix) > tau_S);
    
    % Extract row and column subscripts
    [xc, yc] = ind2sub([m, n], idx_sparse);
    
    % --- STEP 3: Construct Sparse Matrix ---
    % Build the sparse matrix directly using the surviving pixels
    S_cur = sparse(xc, yc, raw_error_matrix(idx_sparse), m, n);
    
    % Update the final Gaussian noise residual
    G_cur = X - L_cur - S_cur;
    
    % Track the actual number of non-zero pixels kept
    c_cur = nnz(S_cur);

    % ==========================================================
    % 4. Record Metrics
    % ==========================================================
    r_rec(it) = r_cur;
    c_rec(it) = c_cur;
    norm_res_rec(it) = norm(G_cur, "fro")^2 / (N_pix * nvar_true); % Track against true noise

    % Relative Recovery Errors
    L_err_rec(it) = norm(L_ex - L_cur, 'fro') / norm(L_ex, 'fro');
    S_err_rec(it) = norm(S_ex - S_cur, 'fro') / norm(S_ex, 'fro');
    G_err_rec(it) = norm(G_cur, 'fro')^2 / norm(G_ex, 'fro')^2;
end
toc;

%% Plotting
figure(2); clf;
tightfig = @(f) set(f,'Units','normalized'); 
tightfig(gcf);

% 1) Normalize Kalıntı (Gürültü Tabanı Kontrolü)
% subplot(2,2,1);
% plot(1:its, norm_res_rec, 'c-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'c');
% hold on;
% yline(1.0, 'r--', 'Optimal Gürültü Tabanı (1.0)', 'LineWidth', 1.5);
% grid on; ylabel('||A||_F / ||A_i||_F'); 
% title('Veri Uyumluluğu (Gürültü Tabanı)');

% 2) Kerte ve Seyreklik Takibi
% subplot(2,2,3);
yyaxis left;
plot(1:its, r_rec, 'g-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'g');
yline(r_true, 'g--', 'Gerçek r', 'LineWidth', 1.5);
ylabel('Kerte (r)'); ylim([0, r_init+5]);

yyaxis right;
plot(1:its, c_rec/N_pix, 'm-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'm');
yline(c_true/N_pix, 'm--', 'Gerçek c', 'LineWidth', 1.5);
ylabel('Seyreklik (c/mn)');
title('Parametre Takibi');
xlabel('Yineleme'); grid on;

% 3) Göreceli Bileşen Hataları
% subplot(2,2,[2,4]);
% plot(1:its, L_err_rec, 'b-o', 'LineWidth', 2); hold on;
% plot(1:its, S_err_rec, 'r-s', 'LineWidth', 2);
% set(gca, 'YScale', 'log'); % Göreceli hata yakınsaması için logaritmik ölçek en iyisidir
% grid on;
% title('Göreceli Kestirim Hatası');
% xlabel('İterasyon'); ylabel('Göreceli Hata (Logaritmik Ölçek)');
% legend('L', 'S', 'Location', 'northeast');

fprintf('\nFinal U Orthogonality Error: %e\n', norm(U_cur' * U_cur - eye(r_cur), "fro"));
fprintf('Final V Orthogonality Error: %e\n', norm(V_cur' * V_cur - eye(r_cur), "fro"));
