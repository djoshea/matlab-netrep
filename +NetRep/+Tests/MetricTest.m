classdef MetricTest < matlab.unittest.TestCase
    % Run tests via runtests("NetRep.Tests.LinearMetricTest")

    properties(Constant)
        TOL = 1e-7;
    end

    properties (MethodSetupParameter)
        seed = {1, 2, 3};
    end

    properties (TestParameter)
        m = {100};
        n = {10};
        zero_pad = {false, true};

        alpha = {0, 0.5, 1};
        m_tri = {31};
        n_tri = {30};
        center_columns = {false, true};
    end

%     methods (TestClassSetup)
%         function classSetup(testCase,generator)
%             orig = rng;
%             testCase.addTeardown(@rng,orig)
%             rng(0,generator)
%         end
%     end
    
    methods (TestMethodSetup)
        function methodSetup(testCase,seed)
            orig = rng;
            testCase.addTeardown(@rng,orig)
            rng(seed)
        end
    end

    methods (Test)
        function test_uncentered_procrustes(testCase, m, n, zero_pad)
            import matlab.unittest.constraints.IsLessThan

            if zero_pad
                Q = NetRep.Utils.rand_orth(n, n + 10);
            else
                Q = NetRep.Utils.rand_orth(n, n);
            end

            % Create a pair of randomly rotated matrices.
            X = randn(m, n);
            Y = X * Q;

            % Fit model, assert distance == 0.
            metric = NetRep.LinearMetric(alpha=1.0, center_columns=false, zero_pad=zero_pad);
            metric.fit(X, Y);

            score = abs(metric.score(X, Y));
            testCase.verifyThat(score, IsLessThan(testCase.TOL));
        end

        function test_centered_procrustes(testCase, m, n)
            import matlab.unittest.constraints.IsLessThan

            Q = NetRep.Utils.rand_orth(n, n);
            v = randn(1, n);
            c = exprnd(1, 1);

            % Create a pair of randomly rotated matrices.
            X = randn(m, n);
            Y = c * X * Q + v;

            % Fit model, assert distance == 0.
            metric = NetRep.LinearMetric(alpha=1.0, center_columns=true);
            metric.fit(X, Y);

            score = abs(metric.score(X, Y));
            testCase.verifyThat(score, IsLessThan(testCase.TOL));
        end

        function test_uncentered_cca(testCase, m, n)
            import matlab.unittest.constraints.IsLessThan
            import matlab.unittest.constraints.IsGreaterThan

            W = randn(n, n);

            % Create a pair of randomly rotated matrices.
            X = randn(m, n);
            Y = X * W;

            % Fit CCA, assert distance == 0.
            metric = NetRep.LinearMetric(alpha=0.0, center_columns=false);
            metric.fit(X, Y);
            testCase.verifyThat(abs(metric.score(X, Y)), IsLessThan(testCase.TOL));

            % Fit Procrustes, assert distance is nonzero.
            metric = NetRep.LinearMetric(alpha=1.0, center_columns=false);
            metric.fit(X, Y);
            testCase.verifyThat(abs(metric.score(X, Y)), IsGreaterThan(testCase.TOL));
        end

        function test_centered_cca(testCase, m, n)
            import matlab.unittest.constraints.IsLessThan
            import matlab.unittest.constraints.IsGreaterThan

            W = randn(n, n);
            v = randn(1, n);

            % Create a pair of randomly rotated matrices.
            X = randn(m, n);
            Y = X * W + v;

            % Fit CCA, assert distance == 0.
            metric = NetRep.LinearMetric(alpha=0.0, center_columns=true);
            metric.fit(X, Y);
            testCase.verifyThat(abs(metric.score(X, Y)), IsLessThan(testCase.TOL));

            % Fit Procrustes, assert distance is nonzero.
            metric = NetRep.LinearMetric(alpha=1.0, center_columns=true);
            metric.fit(X, Y);
            testCase.verifyThat(abs(metric.score(X, Y)), IsGreaterThan(testCase.TOL));
        end

        function test_principal_angles(testCase, m, n)
            import matlab.unittest.constraints.IsLessThan
            W = randn(n, n);
            
            % Create a pair of randomly rotated matrices.
            X = NetRep.Utils.rand_orth(m, n);
            Y = NetRep.Utils.rand_orth(m, n);

            % Compute metric based on principal angles.
            cos_thetas = svd(X' * Y);
            dist_1 = acos(mean(cos_thetas));

            % Fit model, assert two approaches match.
            metric = NetRep.LinearMetric(alpha=1.0, center_columns=false);
            metric.fit(X, Y);
            testCase.verifyThat(abs(dist_1 - metric.score(X, Y)), IsLessThan(testCase.TOL));
        end

        function test_triangle_inequality_linear(testCase, alpha, m_tri, n_tri)
            import matlab.unittest.constraints.IsLessThan
            
            X = randn(m_tri, n_tri);
            Y = randn(m_tri, n_tri);
            M = randn(m_tri, n_tri);
            
            metric = NetRep.LinearMetric(alpha=alpha, center_columns=true);
            dXY = metric.fit(X, Y).score(X, Y);
            dXM = metric.fit(X, M).score(X, M);
            dMY = metric.fit(M, Y).score(M, Y);
            testCase.verifyThat(dXY - dXM - dMY, IsLessThan(testCase.TOL));
        end

        function test_permutation(testCase, center_columns, m, n)
            import matlab.unittest.constraints.IsLessThan
            W = randn(n, n);
            
            % Create a pair of randomly permuted matrices.
            X = randn(m, n);
            pidx = randperm(n);
            Y = X(:, pidx);

            % Fit model, assert distance == 0
            metric = NetRep.PermutationMetric(center_columns=center_columns);
            metric.fit(X, Y);
            testCase.verifyThat(abs(metric.score(X, Y)), IsLessThan(testCase.TOL));
        end
    end
end