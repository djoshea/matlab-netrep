classdef LinearMetricUnbiased < handle
    % similar to LinearMetric, but uses single trial PSTH data to construct an unbiased estimator of the distance
    % also, because of the single trials, handles conditions separately than time. 

    properties
        % Regularization parameter between zero and one. When
        % (alpha == 1.0) the metric only allows for rotational
        % alignments. When (alpha == 0.0) the metric allows for
        % any invertible linear transformation.
        alpha (1, 1) double {mustBeInRange(alpha, 0, 1)} = 1.0

        % If true, learn a mean-centering operation in addition
        % to the linear/rotational alignment.
        center_columns (1, 1) logical = true

        % normalize the total variance across datasets by scaling the training data variance to 1
        normalize_total_variance (1, 1) logical = false;

        score_method (1, 1) string {mustBeMember(score_method, ["euclidean"])} = "euclidean"

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
        function met = LinearMetricUnbiased(args)
            arguments
                args.alpha (1, 1) double {mustBeInRange(args.alpha, 0, 1)} = 1.0;
                args.center_columns (1, 1) logical = true;
                %args.zero_pad (1, 1) logical = true;
                args.score_method (1, 1) string = "euclidean"; 
                args.normalize_total_variance (1, 1) logical = false;
                args.aligned_dim (1, 1) double = Inf;
            end

            met.alpha = args.alpha;
            met.center_columns = args.center_columns;
            met.normalize_total_variance = args.normalize_total_variance;

            assert(args.score_method == "euclidean", "Only euclidean scores supported");
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
            
            % include the normalization in the Zx weights
            Zx = Zx / normalizer;
        end

        function met = fit(met, X, Y)
            % Fits transformation matrices (Wx, Wy) and bias terms (mx, my)
            % to align a pair of neural activation matrices.
            % Parameters
            % ----------
            % X : ndarray
            % (time*conditions x num_neurons) matrix of activations.
            % Y : ndarray
            % (time*conditions x num_neurons) matrix of activations.
            
            arguments
                met
                X (:, :) 
                Y (:, :)
            end

            %[X, Y] = NetRep.Utils.check_equal_shapes(X, Y, nd=2, zero_pad=true);
            [met.mx_, Xw, Zx] = met.partial_fit(X);
            [met.my_, Yw, Zy] = met.partial_fit(Y);
            
            % Xw, Yw is samples x neurons. Xw'*Yw is neurons x neurons
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

        function [tX, tY] = transform(met, X, Y)
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
            end

            % [X, Y] = NetRep.Utils.check_equal_shapes(X, Y, nd=2, zero_pad=true);
            tX = met.transform_X(X);
            tY = met.transform_Y(Y);
        end

        % function score = fit_score(met, X, Y)
        %     % Fits alignment by calling `fit(X, Y)` and then evaluates
        %     % the distance by calling `score(X, Y)`.
        %     arguments
        %         met
        %         X (:, :) 
        %         Y (:, :)
        %     end
        % 
        %     score = met.fit(X, Y).score(X, Y);
        % end

        function [X, Y] = means_from_single_trials(~, single_trials, condition_lookup, Xlookup, Ylookup)
            % single_trials is a sessions { time x neurons x trials }  cell
            % condition_lookup is sessions { trials } list of conditions indices (or NaN / 0 where not included)
            % X/Ylookup are neurons x 2. the first column indexes which session each neuron 
            % is drawn from, and the second column indicates which column of that session the neuron is at
            % the conditions will be concatentated together in ascending order in time in the means, 
            % but we treat them separately because distinct sets of single trials are involved
            
            arguments
                ~
                single_trials (:, 1) cell
                condition_lookup (:, 1) cell
                Xlookup (:, 2) 
                Ylookup (:, 2)
            end
            
            Nx = size(Xlookup, 1);
            Ny = size(Ylookup, 1);
            S = size(single_trials, 1);
            T = size(single_trials{1}, 1);
            C = double(max(cat(1, condition_lookup{:}), [], 'omitnan'));

            S_ = max(cat(1, Xlookup(:, 1), Ylookup(:, 1)));
            assert(S_ <= S, 'Max sessions in X/Ylookup(:, 1)')

            X = nan(T*C, Nx, 'like', single_trials{1});
            Y = nan(T*C, Ny, 'like', single_trials{1});

            for iS = 1:S
                n_mask_x = Xlookup(:, 1) == iS;
                n_cols_x = Xlookup(n_mask_x, 2);
                n_mask_y = Ylookup(:, 1) == iS;
                n_cols_y = Ylookup(n_mask_y, 2);
                
                for iC = 1:C
                    time_idx = (1:T) + (iC-1)*T;
                    trial_mask = condition_lookup{iS} == iC;
                    X(time_idx, n_mask_x) = mean(single_trials{iS}(:, n_cols_x, trial_mask), 3, 'omitnan');
                    Y(time_idx, n_mask_y) = mean(single_trials{iS}(:, n_cols_y, trial_mask), 3, 'omitnan');
                end
            end
        end

        function [score, score_vs_time] = score_vs_time(met, single_trials, condition_lookup, Xlookup, Ylookup, args)
            % Computes the distance metric between X and Y in
            % the aligned space.
            %
            % Parameters
            % ----------
            % single_trials is a sessions { samples x neurons x trials }  cell
            % condition_lookup is sessions { trials } list of conditions indices (or NaN / 0 where not included)
            % X/Ylookup are neurons x 2. the first column indexes which session each neuron 
            % is drawn from, and the second column indicates which column of that session the neuron is at
            %
            % Returns
            % -------
            % score is the total distance over time and conditions
            % score_vs_time is time, conditions x 1

            arguments
                met
                single_trials (:, 1) cell
                condition_lookup (:, 1) cell
                Xlookup (:, 2) 
                Ylookup (:, 2)
                args.aligned_dim_mask (:, 1) = []; % typical for debugging, to slice into Wx
                args.center_each_fold (1, 1) logical = true;
            end

            %[X, Y] = met.means_from_single_trials(single_trials, condition_lookup, Xlookup, Ylookup);
            %[X, Y] = NetRep.Utils.check_equal_shapes(X, Y, nd=2, zero_pad=met.zero_pad);

            %NetRep.Utils.check_equal_shapes(Xlookup', Ylookup', nd=2, zero_pad=true);

            % biased estimate looks like:
            % [tX, tY] = met.transform(X, Y);
            % score = mean(vecnorm(tX - tY, 2, 2), 1, 'omitnan');

            %trialCounts = cellfun(@(x) size(x, 3), single_trials);
            %[bigFoldMatrices, smallFoldMatrices] = met.getSequentialFoldIndicatorMatrices(trialCounts); % S { nFolds x trials } logical indicator matrices
            S = numel(single_trials); % number of sessions
            T = size(single_trials{1}, 1); % num_samples
            C = double(max(cat(1, condition_lookup{:})));

            % assemble partial projection scalars
            [delta_bar_small, delta_bar_big] = deal(cell(S, 1));
            M = nan(T*C, S, S);

            % check whether aligned dim has changed and select specific aligned dimensions if requested (mostly for debuggging)
            Wx = met.Wx_;
            Wy = met.Wy_;
            if isfinite(met.aligned_dim)
                if size(Wx, 2) > met.aligned_dim
                    Wx = Wx(:, 1:met.aligned_dim);
                end
                if size(Wy, 2) > met.aligned_dim
                    Wy = Wy(:, 1:met.aligned_dim);
                end
            end
            if ~isempty(args.aligned_dim_mask)
                Wx = Wx(:, args.aligned_dim_mask);
                Wy = Wy(:, args.aligned_dim_mask);
            end
            for iS = 1:S
                cond_this = condition_lookup{iS};
                trial_mask_all_cond = cond_this >= 1; % this is the selection for all conditions
                cond_this_masked = cond_this(trial_mask_all_cond);

                %n_trials = trialCounts(iS);

                % extract the X neurons for this session
                x_mask = Xlookup(:, 1) == iS;
                x_cols = Xlookup(x_mask, 2);
                
                % neurons from X, projected through the appropriate columns of W
                % single trials is time x neurons x trials. subtract neuron means mx_, 
                % multiply (T x N x R) * (N x K x R) --> (T x K x R)
                x_proj_trials = pagemtimes((single_trials{iS}(:, x_cols, trial_mask_all_cond) - met.mx_(x_mask)), Wx(x_mask, :)); % T x N (xR) * (N x K)

                % extract the Y neurons for this session
                y_mask = Ylookup(:, 1) == iS;
                y_cols = Ylookup(y_mask, 2);
                
                % neurons from X, projected through the appropriate columns of W
                y_proj_trials = pagemtimes((single_trials{iS}(:, y_cols, trial_mask_all_cond) - met.my_(y_mask)), Wy(y_mask, :)); % T x N (xR) * (N x K)

                if T > 1 && args.center_each_fold
                    x_proj_trials = x_proj_trials - mean(x_proj_trials, 1, 'omitnan');
                    y_proj_trials = y_proj_trials - mean(y_proj_trials, 1, 'omitnan');
                end

                K = max(size(x_proj_trials, 2), size(y_proj_trials, 2));
                [delta_bar_small{iS}, delta_bar_small{iS}] = deal(nan(T*C, K));
                for iC = 1:C
                    % n.b.: cond_this_masked and x/y_proj_trials are already subselected by trial_mask_all_cond
                    cond_trial_mask = cond_this_masked == iC; 
                    n_trials = nnz(cond_trial_mask);
                    cond_time_idx = (1:T) + (iC-1)*T;

                    % small folds are just the single trials
                    % big folds are means of everything but the single trials
                    x_proj_small = x_proj_trials(:, :, cond_trial_mask);
                    x_proj_mean = mean(x_proj_small, 3, 'omitnan');
                    
                    % mean = (others + one_trial) / n 
                    % big = (mean * n - one_trial) / (n-1) = mean * n/(n-1) - one_trial / (n-1)
                    x_proj_big = x_proj_mean * n_trials / (n_trials-1) - x_proj_small / (n_trials-1);
    
                    % small folds are just the single trials
                    % big folds are means of everything but the single trials
                    y_proj_small = y_proj_trials(:, :, cond_trial_mask);
                    y_proj_mean = mean(y_proj_small, 3, 'omitnan');
    
                    % mean = (others + one_trial) / n 
                    % big = (mean * n - one_trial) / (n-1) = mean * n/(n-1) - one_trial / (n-1)
                    y_proj_big = y_proj_mean * n_trials / (n_trials-1) - y_proj_small / (n_trials-1);
    
                    % projected X - Y components from this session, folded. 
                    delta_small = x_proj_small - y_proj_small; % T x K x Folds
                    delta_big = x_proj_big - y_proj_big; % T x K x Folds
    
                    % fill in diagonal (dims 2 and 3) terms of M as T x K x 1 --> T x 1 
                    M(cond_time_idx, iS, iS) = sum(mean(delta_big .* delta_small, 3), 2);
    
                    delta_bar_small{iS}(cond_time_idx, :) = mean(delta_small, 3); % T x K
                    delta_bar_big{iS}(cond_time_idx, :) = mean(delta_big, 3); % T x K
                end
            end

            % fill in off-diagonal terms of M
            for iS = 1:S
                for jS = 1:S
                    if iS == jS, continue, end
                    M(:, iS, jS) = sum(delta_bar_big{iS} .* delta_bar_small{jS}, 2); % CT x K --> CT x 1
                end
            end

            sq_score_vs_time = sum(M, [2 3]); % CT x K x K --> CT x 1
            score_vs_time = sign(sq_score_vs_time) .* sqrt(abs(sq_score_vs_time));

            sq_score = sum(sq_score_vs_time);
            score = sign(sq_score) .* sqrt(abs(sq_score));
        end

        function tX = transform_X(met, X)
            % Transform X into the aligned space.
            met.assert_is_fitted()
            if size(X, 2) ~= size(met.Wx_, 1)
                error('Array with wrong shape passed transform, expected %d columns', size(met.Wx_, 1));
            end

            if met.center_columns
                tX = (X - met.mx_) * met.Wx_;
            else
                tX = X * met.Wx_;
            end
            
            % note: normalizer already sits inside Wx_ so we don't need to worry about it here
        end

        function tY = transform_Y(met, Y)
            % Transform X into the aligned space.
            met.assert_is_fitted()
            if size(Y, 2) ~= size(met.Wy_, 1)
                error('Array with wrong shape passed transform, expected %d columns', size(met.Wy_, 1));
            end

            if met.center_columns
                tY = (Y - met.my_) * met.Wy_;
            else
                tY = Y * met.Wy_;
            end
            
            % normalizer sits inside Wy_
        end
    end

    methods(Static)
        function [bigFoldMatrices, smallFoldMatrices] = getSequentialFoldIndicatorMatrices( trialCounts, args )
            % An internal function used to split data into folds for cross-validation for use with sequentially recorded data. 
            % trialCounts is nSessions x 1 vector of trialcounts.
            % big and small foldMatrices is nSessions { nFolds x nNeurons }. 
            % If normalizeForAveraging is false, these are logical indicating whether to include a given trial,
            % If normalizeForAveraging is true, then they are numeric weights that when left multiplied, will average the included trials
            
            arguments
                trialCounts (:, 1) { mustBeInteger, mustBePositive }
                args.normalize (1, 1) logical = false;
            end
            
            minTrials = min(trialCounts, [], 'all');
            assert(minTrials >= 2, 'Minimum of 2 trials per session required for corss-validation');
            
            S = numel(trialCounts);
            [bigFoldMatrices, smallFoldMatrices] = deal(cell(S, 1));

            for s = 1:S
                n = trialCounts(s);
                smallFoldMatrices{s} = eye(n, n);
                if args.normalize
                    bigFoldMatrices{s} = ~eye(n, n) / (n-1);
                else
                    bigFoldMatrices{s} = ~eye(n, n);
                end
            end 
        end
    end
end