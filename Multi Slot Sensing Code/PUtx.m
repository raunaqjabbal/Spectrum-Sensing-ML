function [X,S] = PUtx(samples,slots,txPower, Pr)

% Get the PU transmission
%
% PUtx(M,samples,txPower, Pr_1)
% M - Number of PUs
% samples - Number of transmission samples
% txPower - Average transmission power for each PU
% Pr - Active probability for each PU

S = zeros(1); % PU states
X = zeros(samples,slots); % Signal at PU transmitters

    p = rand(1);
    if (p <= Pr)
        S(:) = 1;
    end


if (S>0)
    X = normrnd(zeros(samples,slots),ones(samples,slots).*sqrt(txPower));
end
