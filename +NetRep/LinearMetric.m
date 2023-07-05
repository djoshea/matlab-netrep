classdef LinearMetric < handle

    properties
        % Regularization parameter between zero and one. When
        % (alpha == 1.0) the metric only allows for rotational
        % alignments. When (alpha == 0.0) the metric allows for
        % any invertible linear transformation.
        alpha (1, 1) double {mustBeInRange(alpha, 0, 1)} = 1.0

        % If true, learn a mean-centering operation in addition
        % to the linear/rotational alignment.
        center_columns (1, 1) logical = true

        % If False, an error is thrown if representations are
        % provided with different dimensions. If True, the smaller
        % matrix is zero-padded prior to allow for an alignment.
        % Some amount of regularization (alpha > 0) is required to
        % align zero-padded representations.
        zero_pad (1, 1) logical = true;
        
        % normalize the total variance across datasets by scaling the training data variance to 1
        normalize_total_variance (1, 1) logical = false;

        score_method (1, 1) string {mustBeMember(score_method, ["angular", "euclidean"])} = "angular"

        aligned_dim (1, 1) double {mustBePositive} = Inf;
    end

    properties % 
        % fitted parameters
        mx_ (1, :) {mustBeNumeric}
        my_ (1, :) {mustBeNumeric}
        Wx_ (:, :) {mustBeNumeric}
        Wy_ (:, :) {mustBeNumeric}
    end

    properties(Dependent)
        is_fitted
    end

    methods
        function met = LinearMetric(args)
            arguments
                args.alpha (1, 1) double {mustBeInRange(args.alpha, 0, 1)} = 1.0;
                args.center_columns (1, 1) logical = true;
                args.zero_pad (1, 1) logical = true;
                args.score_method (1, 1) string ="angular"; 
                args.normalize_total_variance (1, 1) logical = false;
                args.aligned_dim (1, 1) double = Inf;
            end

            met.alpha = args.alpha;
            met.center_columns = args.center_columns;
            met.normalize_total_variance = args.normalize_total_variance;
            met.zero_pad = args.zero_pad;
            met.score_method = args.score_method;
            met.aligned_dim = args.aligned_dim;
        end

        function tf = get.is_fitted(met)
            tf = ~isempty(met.Wy_) && ~isempty(met.Wx_);
        end

        function assert_is_fitted(met)
            assert(met.is_fitted, "LinearMetric has not been fit() yet");
        end

        function [mx, Xw, Zx] = partial_fit(met, X)
            % Computes partial whitening transformation
           % Parameters
            % ----------
            % X : (num_samples x num_neurons) matrix of activations.
            %
            % Returns
            % -------
            % mx : (1 x num_neurons) means used for centering
            % Xw : whitened data
            % Zx : whitening matrix, scaled by normalizing term if normalize_total_variance
            arguments
                met
                X (:, :) 
            end
           
            if met.center_columns
                mx = mean(X, 1, 'omitnan');
                X = X - mx;
            else
                mx = zeros(1, size(X, 2));
            end
            
            if met.normalize_total_variance
                normalizer = norm(X, 'fro');
                X = X ./ normalizer;
            else
                normalizer = 1;
            end
            
            [Xw, Zx] = NetRep.Utils.whiten(X, alpha=met.alpha, preserve_variance=true);
            
            Zx = Zx / normalizer;
        end

        function met = fit(met, X, Y)
            % Fits transformation matrices (Wx, Wy) and bias terms (mx, my)
            % to align a pair of neural activation matrices.
            % Parameters
            % ----------
            % X : ndarray
            % (num_samples x num_neurons) matrix of activations.
            % Y : ndarray
            % (num_samples x num_neurons) matrix of activations.
            
            arguments
                met
                X (:, :) 
                Y (:, :)
            end

            [X, Y] = NetRep.Utils.check_equal_shapes(X, Y, nd=2, zero_pad=met.zero_pad);
            [met.mx_, Xw, Zx] = met.partial_fit(X);
            [met.my_, Yw, Zy] = met.partial_fit(Y);
            
            [U, ~, V] = svd(Xw' * Yw, 'econ');
            num_neurons = size(U, 1);
            if isfinite(met.aligned_dim) && met.aligned_dim < num_neurons
                % truncate to aligned dim
                U = U(:, 1:met.aligned_dim);
                V = V(:, 1:met.aligned_dim);
            end
            
            met.Wx_ = Zx * U;
            met.Wy_ = Zy * V;
        end

        function [tX, tY] = transform(met, X, Y, args)
            % Applies linear alignment transformations to X and Y.
            % Parameters
            % ----------
            % X : (num_samples x num_neurons) matrix of activations.
            % Y : (num_samples x num_neurons) matrix of activations.
            % Returns
            % -------
            % tX : Transformed version of X.
            % tY : Transformed version of Y.

            arguments
                met
                X (:, :) 
                Y (:, :)
                args.aligned_dim_mask (:, 1) = [];
            end

            [X, Y] = NetRep.Utils.check_equal_shapes(X, Y, nd=2, zero_pad=met.zero_pad);
            tX = met.transform_X(X, aligned_dim_mask = args.aligned_dim_mask);
            tY = met.transform_Y(Y, aligned_dim_mask = args.aligned_dim_mask);
        end

        function score = fit_score(met, X, Y)
            % Fits alignment by calling `fit(X, Y)` and then evaluates
            % the distance by calling `score(X, Y)`.
            arguments
                met
                X (:, :) 
                Y (:, :)
            end

            score = met.fit(X, Y).score(X, Y);
        end

        function score = score(met, X, Y, args)
            % Computes the distance metric between X and Y in
            % the aligned space.
            %
            % Parameters
            % ----------
            % X : (num_samples x num_neurons) matrix of activations.
            % Y : (num_samples x num_neurons) matrix of activations.
            % Returns
            % -------
            % dist : float
            %     Angular distance between X and Y.

            arguments
                met
                X (:, :) 
                Y (:, :)
                args.aligned_dim_mask (:, 1) = [];
            end

            score = met.score_vs_time(X, Y, aligned_dim_mask = args.aligned_dim_mask);
        end

        function [score, score_vs_time] = score_vs_time(met, X, Y, args)
            % Computes the distance metric between X and Y in
            % the aligned space, as well as the contribution at each 
            % time point to that cost.
            %
            % Parameters
            % ----------
            % X : (num_samples x num_neurons) matrix of activations.
            % Y : (num_samples x num_neurons) matrix of activations.
            % Returns
            % -------
            % score : float
            %     Distance between X and Y.
            % score_vs_time: float
            %     Contribution to distance at each time point

            arguments
                met
                X (:, :) 
                Y (:, :)
                args.aligned_dim_mask (:, 1) = [];
            end

            
            [tX, tY] = met.transform(X, Y, aligned_dim_mask = args.aligned_dim_mask);

            switch met.score_method
                case "angular"
                    error('Angular distance not supported vs. time.')
%                     [score, score_vs_time] = NetRep.Utils.angular_distance(tX, tY);

                case "euclidean"
                    score_vs_time = vecnorm(tX - tY, 2, 2); % num_samples x 1
                    score = mean(score_vs_time, 1, 'omitnan');

                otherwise
                    error("Unknown score_method")
            end
        end


        function tX = transform_X(met, X, args)
            arguments
                met
                X
                args.aligned_dim_mask (:, 1) = [];
            end

            % Transform X into the aligned space.
            met.assert_is_fitted()
            if size(X, 2) ~= size(met.Wx_, 1)
                error('Array with wrong shape passed transform, expected %d columns', size(met.Wx_, 1));
            end

            % check whether aligned dim has changed and select specific aligned dimensions if requested (mostly for debuggging)
            Wx = met.Wx_;
            if isfinite(met.aligned_dim)
                if size(Wx, 2) > met.aligned_dim
                    Wx = Wx(:, 1:met.aligned_dim);
                end
            end
            if ~isempty(args.aligned_dim_mask)
                Wx = Wx(:, args.aligned_dim_mask);
            end

            if met.center_columns
                tX = (X - met.mx_) * Wx;
            else
                tX = X * Wx;
            end
            
            % normalizer sits inside Wx_
%             if met.normalize_total_variance
%                 tX = tX ./ norm(tX, 'fro');
%             end
        end

        function tY = transform_Y(met, Y, args)
            arguments
                met
                Y
                args.aligned_dim_mask (:, 1) = [];
            end

            % Transform X into the aligned space.
            met.assert_is_fitted()
            if size(Y, 2) ~= size(met.Wy_, 1)
                error('Array with wrong shape passed transform, expected %d columns', size(met.Wy_, 1));
            end

            % check whether aligned dim has changed and select specific aligned dimensions if requested (mostly for debuggging)
            Wy = met.Wy_;
            if isfinite(met.aligned_dim)
                if size(Wy, 2) > met.aligned_dim
                    Wy = Wy(:, 1:met.aligned_dim);
                end
            end
            if ~isempty(args.aligned_dim_mask)
                Wy = Wy(:, args.aligned_dim_mask);
            end

            if met.center_columns
                tY = (Y - met.my_) * Wy;
            else
                tY = Y * Wy;
            end
            
            % normalizer sits inside Wx_
%             if met.normalize_total_variance
%                 tY = tY ./ norm(tY, 'fro');
%             end
        end
    end

end