% Simulation Scenario
scenario.PU = [0 0]*1e3; 					% PU cartesian position in meters
scenario.Pr = 0.5; 							% PU transmission probability
scenario.TXPower = 0.1; 					% PU transmission power in mW
scenario.T = 5e-6; 							% SU spectrum sensing period in seconds
scenario.w = 5e6; 							% SU spectrum sensing bandwidth in hertz
scenario.NoisePSD_dBm = -153; 				% Noise PSD in dBm/Hz
scenario.NoisePower = (10^(scenario.NoisePSD_dBm/10)*1e-3)*scenario.w;


SuNumber=3;
scenario.SU = [zeros(1,SuNumber); linspace(0.5,1,SuNumber)]'*1e3; % SU cartesian position in meters
scenario.fading = 'rician'; % Adds Rayleigh fading to the received signals
scenario.variance=1;
scenario.realiz = 50000; 						% MCS realization
scenario.slots=1;
trainingScenario = scenario;
trainingScenario.realiz = 250;



train = struct();
[test.X, test.Y ,~,~,~,SNR] = MCS(scenario);
[train.X,train.Y,~,~,~,~]   = MCS(trainingScenario);

% size(test.X)
% size(test.Y)
% size(train.X)
% size(train.Y)
slots=scenario.slots;

meanSNR = mean(SNR(:,test.Y==1),2);
meanSNRdB = 10*log10(meanSNR)
meanSNR=[slots;round(2*scenario.T*scenario.w/slots);meanSNR]
csvwrite("ClassificationDataSNR.csv",meanSNR);
csvwrite("ClassificationDataTrainX.csv",train.X);
csvwrite("ClassificationDataTrainY.csv",train.Y);
csvwrite("ClassificationDataTestX.csv",test.X);
csvwrite("ClassificationDataTestY.csv",test.Y);
