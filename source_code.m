clc;
clear variables;
stock_data = readtable('dataset.csv');
stock_data = table2cell(stock_data);
stock_data(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),stock_data)) = {''};
count=1;
Dates = cell(250, 1);
open_price = zeros(250, 1);
high_price = zeros(250, 1);
low_price = zeros(250, 1);
close_price = zeros(250, 1);
for i= 2:250
    Dates{i,1}=stock_data{i,1};
    open_price(i,1)=stock_data{i,2};
    high_price(i,1)=stock_data{i,3};
    low_price(i,1)=stock_data{i,4};
    close_price(i,1)=stock_data{i+1,5};
end
D=double([open_price low_price high_price close_price])/100000;
Inputs = D(30:end,:);
Targets =(double(open_price(14:end)))/100000;
N = size(Inputs,1);
PERM = randperm(N);
pTrain=0.85;
nTraiN=round(pTrain*N);
TrainInd=PERM(1:nTraiN);
TrainInputs=Inputs(TrainInd,:);
TrainTargets=Targets(TrainInd,:);
pTest=1-pTrain;
nTestData=N-nTraiN;
TestInd=PERM(nTraiN+1:end);
TestInputs=Inputs(TestInd,:);
TestTargets=Targets(TestInd,:);
nCluster=5;       
Exponent=2;        
MaxIt=200;
MinImprovment=1e-7;
DisplayInfo=1;
FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];
fis=genfis3(TrainInputs,TrainTargets,'sugeno',nCluster,FCMOptions);

MaxEpoch=200;               
ErrorGoal=0;            
InitialStepSize=0.01;       
StepSizeDecreaseRate=0.9;   
StepSizeIncreaseRate=1.1;    
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];
DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];
OptimizationMethod=0;
fis=anfis([TrainInputs TrainTargets],fis,TrainOptions,DisplayOptions,[],OptimizationMethod);
Outputs=evalfis(Inputs,fis);
TrainOutputs=Outputs(TrainInd,:);
TestOutputs=Outputs(TestInd,:);
figure('Name','Training Data');
PlotResults(TrainTargets,TrainOutputs,'Train Data');
figure('Name','Testing Data');
PlotResults(TestTargets,TestOutputs,'Test Data');
figure('Name','Output Data');
PlotResults(Targets(17:end), Outputs, 'All Data');
function PlotResults(targets, outputs, Name)
    errors=targets-outputs;
    MSE=mean(errors.^2);
    RMSE=sqrt(MSE);
    error_mean=mean(errors);
    error_std=std(errors);
    subplot(2,2,[1 2]);
    plot(targets,'k');
    hold on;
    plot(outputs,'r');
    legend('Target','Output');
    title(Name);
    xlabel('Sample Index');
    grid on;
    subplot(2,2,3);
    plot(errors);
    legend('Error');
    title(['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)]);
    grid on;
    subplot(2,2,4);
    histfit(errors, 50);
    title(['Error Mean = ' num2str(error_mean) ', Error St.D. = ' num2str(error_std)]);
end