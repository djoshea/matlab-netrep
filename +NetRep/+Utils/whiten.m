function [X_whitened, Z] = whiten(X, args)
    % Return regularized whitening transform for a matrix X.
    % Parameters
    % ----------
    % X : ndarray
    %     Matrix with shape `(m, n)` holding `m` observations
    %     in `n`-dimensional feature space. Columns of `X` are
    %     expected to be mean-centered so that `X.T @ X` is
    %     the covariance matrix.
    % alpha : float
    %     Regularization parameter, `0 <= alpha <= 1`.
    % preserve_variance : bool
    %     If True, rescale the (partial) whitening matrix so
    %     that the total variance, trace(X.T @ X), is preserved.
    % eigval_tol : float
    %     Eigenvalues of covariance matrix are clipped to this
    %     minimum value.
    % Returns
    % -------
    % X_whitened : ndarray
    %     Transformed data matrix.
    % Z : ndarray
    %     Matrix implementing the whitening transformation.
    %     `X_whitened = X @ Z`.

    arguments
        X (:, :) double
        args.alpha (1, 1) double {mustBeInRange(args.alpha, 0, 1)} = 1.0;
        args.preserve_variance (1, 1) logical = true;
        args.eigval_tol (1, 1) double = 1e-7;
    end

    alpha = args.alpha;

    % Return early if regularization is maximal (no whitening).
    if alpha > (1 - args.eigval_tol)
        X_whitened = X;
        Z = eye(size(X, 2));
        return;
    end

    % Compute eigendecomposition of covariance matrix
    [V, D] = eig(X' * X);
    lam = max(diag(D), args.eigval_tol);
    
    % Compute diagonal of (partial) whitening matrix.
    % 
    % When (alpha == 1), then (d == ones).
    % When (alpha == 0), then (d == 1 / sqrt(lam)).
    d = alpha + (1 - alpha)./sqrt(lam);

    % Rescale the whitening matrix.
    if args.preserve_variance
        % Compute the variance of the transformed data.
        %
        % When (alpha == 1), then new_var = sum(lam)
        % When (alpha == 0), then new_var = len(lam)
        new_var = sum( alpha^2 * lam + 2*alpha*(1 - alpha)*sqrt(lam) + ...
            + (1 - alpha)^2 * ones(size(lam)) );

        % Now re-scale d so that the variance of (X @ Z)
        % will equal the original variance of X.
        d = d * sqrt(sum(lam) / new_var);
    end

    % Form (partial) whitening matrix.
    Z = (V .* d') * V';

    % An alternative regularization strategy would be:
    %
    % lam, V = np.linalg.eigh(X.T @ X)
    % d = lam ** (-(1 - alpha) / 2)
    % Z = (V * d[None, :]) @ V.T

    % Returned (partially) whitened data and whitening matrix.
    X_whitened = X * Z;
end
