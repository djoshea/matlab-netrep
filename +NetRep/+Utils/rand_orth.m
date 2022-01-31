function Q = rand_orth(m, n, args)
    % Creates a random matrix with orthogonal columns or rows.
    % Parameters
    % ----------
    % m : int
    %     First dimension
    % n : int
    %     Second dimension (if None, matrix is m x m)
    % random_state : int or np.random.RandomState
    %     Specifies the state of the random number generator.
    % Returns
    % -------
    % Q : ndarray
    %     An m x n random matrix. If m > n, the columns are orthonormal.
    %     If m < n, the rows are orthonormal. If m == n, the result is
    %     an orthogonal matrix.

    arguments
        m (1, 1) {mustBeInteger}
        n (1, 1) {mustBeInteger} = m
        args.random_state = [];
    end

    if ~isempty(args.random_state)
        if isa(args.random_state, "RandStream")
            rs = args.random_state;
        else
            rs = RandStream('mt19937ar', 'Seed', args.random_state);
        end
    else
        rs = RandStream.getGlobalStream();
    end

    Q = ortho_group(rs, max(m, n));
    if size(Q, 1) > m
        Q = Q(1:m, :);
    end
    if size(Q, 2) > n
        Q = Q(:, 1:n);
    end
end


function H = ortho_group(rs, dim)
    % https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/_multivariate.py#L3425
    H = eye(dim);
    for n = 1:dim
        x = randn(rs, dim-n+1, 1);
        norm2 = dot(x, x);
        x1 = x(1);
        % random sign, 50/50, but chosen carefully to avoid roundoff error
        if x1 == 0
            D = 1;
        else
            D = sign(x1);
        end
        x(1) = x(1) + D * sqrt(norm2);
        x = x / sqrt((norm2 - x1^2 + x(1)^2) / 2);
        
        % Householder transformation
        H(:, n:end) = -D * (H(:, n:end) - (H(:, n:end) * x) * x');
    end
end