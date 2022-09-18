function [Y,A,PU,n,Z,SNR] = MCS(scenario)

M = 1;                    % Number of PUs
N = size(scenario.SU,1);                 % Number of SUs
txPower = scenario.TXPower;       % Transmission power
noisePower = scenario.NoisePower*ones(N,1); % Gaussian noise power
a = 4;                                      % Path-loss exponent
slots=scenario.slots;
samples = round(2*scenario.T*scenario.w/slots);   % Number of samples
realiz = scenario.realiz;                   % Number of Monte-Carlo realizations
fading=scenario.fading;

%% Compute the Euclidean distance for each PU-SU pair
d = zeros(N,1);

    for i=1:N
        d(i) = norm(scenario.PU-scenario.SU(i,:));
        if (d(i) == 0)
            d(i) = 1;
        end
    end
%% Main
Y = zeros(realiz,slots,N);                          % Power estimated
S = zeros(realiz,1);                          % Channel availability
        %  i , t    ,   b , k
PU = zeros(N,samples,slots,realiz);  % PUs signal received at SU
n = zeros(N,samples,slots,realiz);   % Noise received at SU
Z = zeros(N,samples,slots,realiz);   % PU signal + noise received at SU
SNR = zeros(N,realiz);                        % PU SNR at the SU receiver
for k=1:realiz
        n(:,:,:,k) = gaussianNoise(samples,noisePower,slots);       % Get the noise at SU receivers
        H = channel(N,d,a,fading,scenario.variance);                      % Get the channel matrix             
        [X, S(k)] = PUtx(samples,slots,txPower, scenario.Pr); % Get the PU transmissions
        for i=1:N
            for b=1:slots
%                 sizeX=size(X)
%                 sizePU=size(PU)
%                 sizeN=size(n)
                for t=1:samples
                    PU(i,t,b,k)= PU(i,t,b,k)+H(i)*X(t,b);
                    Z(i,t,b,k) = PU(i,t,b,k) + n(i,t,b,k);
                end
                Y(k,b,i) = sum(abs(Z(i,:,b,k)).^2)/(noisePower(i)*samples); % Normalized by noise variance
            end
            SNR(i,k) = mean(mean(abs(PU(i,:,:,k).^2)))/noisePower(i);
            for b=1:slots
            end
        end
end
A=S(:,1); % Channel availability
