function [X, Y] = check_equal_shapes(X, Y, args)
    arguments
        X {mustBeNumeric}
        Y {mustBeNumeric}
        args.nd (1, 1) {mustBeInteger} = 2;
        args.zero_pad (1, 1) logical = false;
    end
    
    nd = args.nd;

    assert(ndims(X) == nd, 'X must have % dimensions', nd);
    assert(ndims(Y) == nd, 'Y must have % dimensions', nd);
    
    if ~isequal(size(X), size(Y))
        if args.zero_pad && size(X, 1) == size(Y, 1)
            % Number of padded zeros to add.
            n = max(size(X, nd), size(Y, nd));
            
            % Padding specifications for X and Y.
            X = pad_to_len_along_dim(X, nd, n);
            Y = pad_to_len_along_dim(Y, nd, n);
        else
            error('Expected arrays with equal dimensions with zero_pad false');
        end
    end
end

function X = pad_to_len_along_dim(X, dim, n)
    spx = size(X);
    spx(dim) = n - size(X, dim);
    px = zeros(spx, 'like', X);
    X = cat(dim, X, px);
end