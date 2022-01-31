function dist = angular_distance(X, Y)
    % Computes angular distance based on Frobenius inner product
    % between two matrices.
    %
    % Parameters
    % ----------
    % X : m x n matrix
    % Y : m x n matrix
    % Returns
    % -------
    % distance : float between zero and pi.
   
    arguments
        X (:, :)
        Y (:, :)
    end

    normalizer = norm(X(:)) * norm(Y(:));
    cor = dot(X(:), Y(:)) / normalizer;
    % numerical precision issues require us to clip inputs to arccos

    cor(cor < -1) = -1;
    cor(cor > 1) = 1;
    dist = acos(cor);
end
    