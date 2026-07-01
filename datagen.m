% =========================================================================
% --- 1. Experimental Parameters ---
% =========================================================================
m = 1080;
n = 1920;
r_true = 100;
c_ratio = 0.1;
snr_dB = 20;  

N_pix = m * n;
c_true = round(N_pix * c_ratio);

% =========================================================================
% --- 2. Generate Low-Rank Background (L_ex) ---
% =========================================================================
% Create random orthogonal bases
[U_true, ~] = qr(randn(m, r_true), 0);
[V_true, ~] = qr(randn(n, r_true), 0);

% Generate singular values (Linearly decaying to mimic real-world data)
sigmas = linspace(100, 20, r_true)'; 
D_true = diag(sigmas);

% Form the exact low-rank matrix
L_ex = U_true * D_true * V_true';

% =========================================================================
% --- 3. Calculate Noise Variance (nvar) from Target SNR ---
% =========================================================================
% Signal power is the Mean Squared Value of the L matrix
power_L = norm(L_ex, 'fro')^2 / N_pix; 

% Calculate the required noise power (nvar) to hit the target SNR in dB
% Formula: SNR_dB = 10 * log10(Power_L / Power_G)
nvar_true = power_L / (10^(snr_dB / 10));
sigma_noise = sqrt(nvar_true);

% Generate Dense Gaussian Noise (G_ex)
G_ex = sigma_noise * randn(m, n);

% =========================================================================
% --- 4. Information-Theoretic Sanity Check ---
% =========================================================================
% Check if the requested SNR buries the signal under the Marchenko-Pastur limit.
MP_ubound = sigma_noise * (sqrt(m) + sqrt(n));

fprintf('--- Data Generation Stats ---\n');
fprintf('Target SNR:        %d dB\n', snr_dB);
fprintf('Noise Variance:    %.6f\n', nvar_true);
fprintf('Min Signal Sigma:  %.2f\n', min(sigmas));
fprintf('MP Noise Bound:    %.2f\n', MP_ubound);

if min(sigmas) < MP_ubound
    warning('SNR is too low! The weakest singular values are mathematically buried under the noise floor and cannot be recovered.');
end

% =========================================================================
% --- 5. Generate Sparse Foreground (S_ex) ---
% =========================================================================
% To make the sparse outliers realistic, we scale their magnitudes relative 
% to the actual dynamic range of the background matrix L.
max_L = max(abs(L_ex(:)));

% Randomly select c_true spatial locations
idx = randperm(N_pix, c_true);
[m_row, m_col] = ind2sub([m, n], idx);

% Generate values that are large enough to be outliers, but not "blinding lasers"
% Magnitudes between 0.5x and 1.5x the max background peak, with random signs.
magnitudes = (rand(c_true, 1) * 1.0 + 0.5) * max_L;
signs = sign(randn(c_true, 1));
vals = magnitudes .* signs;

S_ex = sparse(m_row, m_col, vals, m, n);

% =========================================================================
% --- 6. Form Final Observation Matrix ---
% =========================================================================
X = L_ex + S_ex + G_ex;

r_max = min(n,m);