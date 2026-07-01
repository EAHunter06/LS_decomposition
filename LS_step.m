function [U, D, V, DirU_out, DirV_out] = LS_step(X, U_cur, D_cur, V_cur, L_cur, S_cur, m, n, beta, DirU_prev, DirV_prev)
    % LS_step_momentum  Same as LS_step, plus Nesterov extrapolation.
    % Linear analytical step size, tangent-space projection, no retraction.
    %
    % Without momentum (nargin <= 8 or beta == 0), this reduces exactly to LS_step.

    r = size(U_cur, 2);

    use_momentum = (nargin >= 11) && (beta > 0) && ...
                   ~isempty(DirU_prev) && ~isempty(DirV_prev) && ...
                   size(DirU_prev, 2) == r;

    % --- 1. Lookahead point (Nesterov extrapolation, projected to tangent) ---
    if use_momentum
        MomU = DirU_prev - U_cur * (U_cur' * DirU_prev);
        MomV = DirV_prev - V_cur * (V_cur' * DirV_prev);
        U_look = U_cur + beta * MomU;
        V_look = V_cur + beta * MomV;
    else
        U_look = U_cur;
        V_look = V_cur;
    end

    % --- 2. Vectorized Gradient Calculation (at lookahead) ---
    M = (X - S_cur);
    GU = M * (V_look * D_cur');
    GV = M' * (U_look * D_cur);

    % --- 3. Tangent Space Projection ---
    DirU = GU - U_look * (GU' * U_look);
    DirV = GV - V_look * (GV' * V_look);

    % --- 4. Analytical Step Size (linear, O((m+n)*r^2)) ---
    Residual = X - S_cur - L_cur;

    RV    = Residual * V_look;
    RtU   = Residual' * U_look;
    DirUD = DirU * D_cur;

    numerator = real( sum(DirUD .* RV, 'all') + sum((RtU * D_cur) .* DirV, 'all') );

    DtDU  = DirUD' * DirUD;
    term1 = real(trace(DtDU));

    DtD   = D_cur' * D_cur;
    DvtDv = DirV' * DirV;
    term2 = real(trace(DtD * DvtDv));

    DirVtV = DirV' * V_look;
    DirUtU = DirU' * U_look;
    term3  = 2 * real(trace(D_cur * DirVtV * D_cur' * DirUtU));

    denominator = term1 + term2 + term3;

    t_opt = numerator / (denominator + 1e-12);

    % --- 5. Update (no retraction, matching LS_step.m) ---
    U = U_look + t_opt * DirU;
    V = V_look + t_opt * DirV;

    % --- 6. Update singular-value weights from UPDATED bases ---
    D_vec = real(diag(U' * (X - S_cur) * V));
    D = diag(D_vec);

    % --- 7. Export step for next iteration's momentum ---
    DirU_out = U - U_look;
    DirV_out = V - V_look;
end
