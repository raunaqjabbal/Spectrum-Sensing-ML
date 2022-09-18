function [n] = gaussianNoise(samples,noisePower, slots)

% Returns the Gaussian Noise with zero mean and unitary variance
% 
% gaussianNoise(N,samples,noisePower)
% N - Number of SUs
% samples - Number of noise samples 
% noisePower - Noise power for each SU

N = size(noisePower,1);
n = zeros(N,samples,slots);

for i=1:N
    n(i,:,:) = normrnd(0,real(sqrt(noisePower(i))),1,samples,slots);
end

end