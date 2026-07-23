%% A04_train_supervised_BiLSTM.m
% Supervised manoeuvre detector using a bidirectional LSTM classifier.
%
% The BiLSTM processes each past-only window in both temporal directions.
% No observation after the classified sequence endpoint is included.
% Run A03 first. Required files:
%   ml_feature_datasetReady.mat
%   ManoeuvreLabel_QZS3.txt
%
% Temporal protocol:
%   Train:      2018-2022
%   Validation: 2023
%   Test:       2024-2025 (not used for fitting or tuning)
%
% Each observation presented to the LSTM is a causal sliding window:
%   [numFeatures x sequenceLength]
%
% The label of a sequence is the label of its final epoch. The same
% official-event grouping, positive-epoch definition, event grouping and
% matching tolerance used by the boosted-tree model are retained.

clear; clc; close all;
rng(7);

%% 1) Configuration

dataFile = 'ml_feature_datasetReady.mat';
labelFileName = 'ManoeuvreLabel_QZS3.txt';

trainEnd = datetime(2022,12,31,23,59,59,'TimeZone','UTC');
validationStart = datetime(2023,1,1,0,0,0,'TimeZone','UTC');
validationEnd = datetime(2023,12,31,23,59,59,'TimeZone','UTC');
testStart = datetime(2024,1,1,0,0,0,'TimeZone','UTC');
testEnd = datetime(2025,12,31,23,59,59,'TimeZone','UTC');

% Same physical-event and post-processing settings as the tree model.
officialGroupingDays = 3.0;
eventGapDays = 3.0;
matchingToleranceDays = 5.0;
numPositiveEpochsPerEvent = 3; % 2

% A sequence is rejected if any adjacent TLE gap exceeds this value.
maxSequenceGapDays = 3.0;

% Validation threshold grid.
thresholdGrid = (0.10:0.02:0.90)';

% Compact architecture search suitable for this relatively small dataset.
sequenceLengthGrid = [6 12 20];
hiddenUnitsGrid = [8 16 32];
dropoutGrid = [0.1 0.2 0.3];

% Training settings.
maxEpochs = 60;
miniBatchSize = 32;
initialLearnRate = 1e-3;
gradientThreshold = 1.0;

% Training balancing only. Validation and test remain untouched.
% Retain at most this many normal sequences per positive sequence.
normalToPositiveRatio = 3;

networkArchitecture = "BiLSTM";

%% 2) Load A03 feature dataset

if ~isfile(dataFile)
    error('Dataset not found: %s',dataFile);
end

S = load(dataFile,'ml_features','selectedFeatureNames');

if ~isfield(S,'ml_features') || ~isfield(S,'selectedFeatureNames')
    error('%s must contain ml_features and selectedFeatureNames.',dataFile);
end

ml_features = S.ml_features;
usedFeatureNames = cellstr(string(S.selectedFeatureNames));

if ~ismember('epoch_datetime',ml_features.Properties.VariableNames)
    error('ml_features must contain epoch_datetime.');
end

missingFeatures = usedFeatureNames( ...
    ~ismember(usedFeatureNames,ml_features.Properties.VariableNames));

if ~isempty(missingFeatures)
    disp(missingFeatures(:));
    error('Some A03 selected features are missing from ml_features.');
end

dates = ml_features.epoch_datetime;
dates.TimeZone = 'UTC';

Xraw = ml_features{:,usedFeatureNames};

validRows = ~isnat(dates) & any(isfinite(Xraw),2);
dates = dates(validRows);
Xraw = Xraw(validRows,:);
ml_features = ml_features(validRows,:);

[dates,order] = sort(dates);
Xraw = Xraw(order,:);
ml_features = ml_features(order,:);

fprintf('\n============================================================\n');
fprintf(' SUPERVISED BIDIRECTIONAL LSTM MANOEUVRE DETECTOR\n');
fprintf('============================================================\n');
fprintf('Rows after filtering: %d\n',size(Xraw,1));
fprintf('A03 features loaded:  %d\n',size(Xraw,2));

%% 3) Official manoeuvre events and epoch labels

labelFile = locateFile(labelFileName);
fprintf('Official label file:\n%s\n',labelFile);

officialBurns = readOfficialManeuvers(labelFile);
officialEvents = groupOfficialManeuvers(officialBurns,officialGroupingDays);

if isempty(officialEvents)
    error('No official manoeuvre events were parsed.');
end

[yEpoch,eventIndexEpoch] = createEpochLabels( ...
    dates,officialEvents.event_date,numPositiveEpochsPerEvent);

%% 4) Chronological split

idxTrain = dates <= trainEnd;
idxValidation = dates >= validationStart & dates <= validationEnd;
idxTest = dates >= testStart & dates <= testEnd;

if ~any(idxTrain) || ~any(idxValidation) || ~any(idxTest)
    error('One or more temporal partitions are empty.');
end

fprintf('\nTemporal split:\n');
fprintf('  Train:      %s to %s | %d epochs\n', ...
    char(min(dates(idxTrain))),char(max(dates(idxTrain))),sum(idxTrain));
fprintf('  Validation: %s to %s | %d epochs\n', ...
    char(min(dates(idxValidation))),char(max(dates(idxValidation))),sum(idxValidation));
fprintf('  Test:       %s to %s | %d epochs\n', ...
    char(min(dates(idxTest))),char(max(dates(idxTest))),sum(idxTest));

%% 5) Train-only imputation and robust scaling

XtrainRaw = Xraw(idxTrain,:);

trainMedian = median(XtrainRaw,1,'omitnan');
trainMedian(~isfinite(trainMedian)) = 0;

Ximputed = Xraw;
for j = 1:size(Ximputed,2)
    bad = ~isfinite(Ximputed(:,j));
    Ximputed(bad,j) = trainMedian(j);
end

trainMAD = mad(Ximputed(idxTrain,:),1,1);
fallbackStd = std(Ximputed(idxTrain,:),0,1);

badScale = ~isfinite(trainMAD) | trainMAD < 1e-12;
trainMAD(badScale) = fallbackStd(badScale);
badScale = ~isfinite(trainMAD) | trainMAD < 1e-12;
trainMAD(badScale) = 1;

% Escalado 0-1
Xscaled = (Ximputed-trainMedian)./trainMAD;

trainStd = std(Xscaled(idxTrain,:),0,1);
badCols = ~isfinite(trainStd) | trainStd < 1e-12;

if any(badCols)
    fprintf('\nRemoving %d constant/invalid training features:\n',sum(badCols));
    disp(usedFeatureNames(badCols)');
end

Xscaled(:,badCols) = [];
trainMedian(badCols) = [];
trainMAD(badCols) = [];
usedFeatureNames = usedFeatureNames(~badCols);

if isempty(usedFeatureNames)
    error('No usable features remain.');
end

numFeatures = size(Xscaled,2);
fprintf('Features retained for LSTM: %d\n',numFeatures);

%% 6) Joint architecture and threshold selection on validation

numCandidates = numel(sequenceLengthGrid)* ...
    numel(hiddenUnitsGrid)*numel(dropoutGrid);

candidateRows = cell(numCandidates,1);
candidateNetworks = cell(numCandidates,1);
candidateThresholdTables = cell(numCandidates,1);
candidateSequenceData = cell(numCandidates,1);

candidateCounter = 0;

fprintf('\nCandidate networks: %d\n',numCandidates);

for iLength = 1:numel(sequenceLengthGrid)
    sequenceLength = sequenceLengthGrid(iLength);

    [XTrainSeq,YTrainSeq,trainSequenceDates,trainEndIndices] = ...
        buildCausalSequences( ...
        Xscaled(idxTrain,:),yEpoch(idxTrain),dates(idxTrain), ...
        sequenceLength,maxSequenceGapDays);

    [XValidationSeq,YValidationSeq,validationSequenceDates, ...
        validationEndIndices] = buildCausalSequences( ...
        Xscaled(idxValidation,:),yEpoch(idxValidation), ...
        dates(idxValidation),sequenceLength,maxSequenceGapDays);

    if isempty(XTrainSeq) || isempty(XValidationSeq)
        warning('No valid sequences for sequenceLength=%d.',sequenceLength);
        continue
    end

    [XTrainBalanced,YTrainBalanced] = balanceTrainingSequences( ...
        XTrainSeq,YTrainSeq,normalToPositiveRatio);

    fprintf('\nSequence length %d:\n',sequenceLength);
    fprintf('  Train sequences:      %d (%d positive)\n', ...
        numel(XTrainSeq),sum(YTrainSeq == 1));
    fprintf('  Balanced sequences:   %d (%d positive)\n', ...
        numel(XTrainBalanced),sum(YTrainBalanced == 1));
    fprintf('  Validation sequences: %d (%d positive)\n', ...
        numel(XValidationSeq),sum(YValidationSeq == 1));

    YTrainCategorical = categorical( ...
        YTrainBalanced,[0 1],{'normal','manoeuvre'});

    for iHidden = 1:numel(hiddenUnitsGrid)
        for iDropout = 1:numel(dropoutGrid)

            candidateCounter = candidateCounter+1;

            hiddenUnits = hiddenUnitsGrid(iHidden);
            dropoutProbability = dropoutGrid(iDropout);

            fprintf(['Training BiLSTM candidate %d/%d: ', ...
                     'L=%d, hidden/direction=%d, dropout=%.2f\n'], ...
                     candidateCounter,numCandidates, ...
                     sequenceLength,hiddenUnits,dropoutProbability);

            layers = [
                sequenceInputLayer( ...
                    numFeatures, ...
                    'Name','sequenceInput', ...
                    'Normalization','none')
            
                bilstmLayer( ...
                    hiddenUnits, ...
                    'OutputMode','last', ...
                    'Name','bilstm')
            
                dropoutLayer( ...
                    dropoutProbability, ...
                    'Name','dropout')
            
                fullyConnectedLayer( ...
                    2, ...
                    'Name','classifier')
            
                softmaxLayer( ...
                    'Name','softmax')
            
                classificationLayer( ...
                    'Name','classOutput')
            ];

        %     % Opcion con dos capas:
        %     layers = [
        %     sequenceInputLayer(numFeatures,'Name','sequenceInput')
        % 
        %     bilstmLayer( ...
        %         hiddenUnits1, ...
        %         'OutputMode','sequence', ...
        %         'Name','bilstm1')
        % 
        %     dropoutLayer(dropoutProbability,'Name','dropout1')
        % 
        %     bilstmLayer( ...
        %         hiddenUnits2, ...
        %         'OutputMode','last', ...
        %         'Name','bilstm2')
        % 
        %     dropoutLayer(dropoutProbability,'Name','dropout2')
        % 
        %     fullyConnectedLayer(2,'Name','classifier')
        %     softmaxLayer('Name','softmax')
        %     classificationLayer('Name','classOutput')
        % ];

            options = trainingOptions('adam', ...
                'MaxEpochs',maxEpochs, ...
                'MiniBatchSize',miniBatchSize, ...
                'InitialLearnRate',initialLearnRate, ...
                'GradientThreshold',gradientThreshold, ...
                'Shuffle','every-epoch', ...
                'ExecutionEnvironment','auto', ...
                'Verbose',false, ...
                'Plots','none');

            net = trainNetwork( ...
                XTrainBalanced,YTrainCategorical,layers,options);

            [~,validationScores] = classify( ...
                net,XValidationSeq, ...
                'MiniBatchSize',miniBatchSize);

            validationProbability = extractLSTMPositiveProbability( ...
                validationScores,net,'manoeuvre');

            officialValidationEvents = officialEvents( ...
                officialEvents.event_date >= validationStart & ...
                officialEvents.event_date <= validationEnd,:);

            thresholdResults = evaluateThresholds( ...
                validationSequenceDates,validationProbability, ...
                officialValidationEvents.event_date,thresholdGrid, ...
                eventGapDays,matchingToleranceDays, ...
                validationStart,validationEnd);

            bestThresholdRow = selectBestThreshold(thresholdResults);

            architectureName = categorical(networkArchitecture);
            candidateRows{candidateCounter} = table( ...
            architectureName, ...
            sequenceLength,hiddenUnits,dropoutProbability, ...
            bestThresholdRow.Threshold,bestThresholdRow.TP, ...
            bestThresholdRow.FP,bestThresholdRow.FN, ...
            bestThresholdRow.Precision,bestThresholdRow.Recall, ...
            bestThresholdRow.F1,bestThresholdRow.FalseAlarmsPerYear, ...
            numel(XTrainSeq),numel(XTrainBalanced), ...
            numel(XValidationSeq), ...
            'VariableNames',{ ...
            'Architecture', ...
            'SequenceLength','HiddenUnits','Dropout', ...
            'Threshold','TP','FP','FN','Precision','Recall','F1', ...
            'FalseAlarmsPerYear','NumTrainSequences', ...
            'NumBalancedTrainSequences','NumValidationSequences'});

            candidateNetworks{candidateCounter} = net;
            candidateThresholdTables{candidateCounter} = thresholdResults;
            candidateSequenceData{candidateCounter} = struct( ...
                'trainSequenceDates',trainSequenceDates, ...
                'trainEndIndices',trainEndIndices, ...
                'validationSequenceDates',validationSequenceDates, ...
                'validationEndIndices',validationEndIndices);

            fprintf('  Best validation threshold %.2f | F1 %.4f | FA/y %.2f\n', ...
                bestThresholdRow.Threshold,bestThresholdRow.F1, ...
                bestThresholdRow.FalseAlarmsPerYear);
        end
    end
end

candidateRows = candidateRows(1:candidateCounter);
candidateNetworks = candidateNetworks(1:candidateCounter);
candidateThresholdTables = candidateThresholdTables(1:candidateCounter);
candidateSequenceData = candidateSequenceData(1:candidateCounter);

if candidateCounter == 0
    error('No LSTM candidate was trained.');
end

hyperparameterResults = vertcat(candidateRows{:});

% Rank by validation event F1, then fewer false alarms/year,
% then greater recall, then smaller network.
rankTable = hyperparameterResults;
rankTable.NegativeF1 = -rankTable.F1;
rankTable.NegativeRecall = -rankTable.Recall;

[~,rankOrder] = sortrows( ...
    rankTable, ...
    {'NegativeF1','FalseAlarmsPerYear','NegativeRecall', ...
     'HiddenUnits','SequenceLength'}, ...
    {'ascend','ascend','ascend','ascend','ascend'});

bestCandidateIndex = rankOrder(1);
lstmModel = candidateNetworks{bestCandidateIndex};
thresholdResults = candidateThresholdTables{bestCandidateIndex};
bestSequenceData = candidateSequenceData{bestCandidateIndex};

bestParams = hyperparameterResults(bestCandidateIndex,:);
selectedThreshold = bestParams.Threshold;
selectedSequenceLength = bestParams.SequenceLength;
selectedHiddenUnits = bestParams.HiddenUnits;
selectedDropout = bestParams.Dropout;

fprintf('\n============================================================\n');
fprintf(' SELECTED LSTM CONFIGURATION\n');
fprintf('============================================================\n');
disp(bestParams);

%% 7) Rebuild selected train, validation and untouched test sequences

[XTrainSeq,YTrainSeq,trainSequenceDates,trainEndIndices] = ...
    buildCausalSequences( ...
    Xscaled(idxTrain,:),yEpoch(idxTrain),dates(idxTrain), ...
    selectedSequenceLength,maxSequenceGapDays);

[XValidationSeq,YValidationSeq,validationSequenceDates,validationEndIndices] = ...
    buildCausalSequences( ...
    Xscaled(idxValidation,:),yEpoch(idxValidation),dates(idxValidation), ...
    selectedSequenceLength,maxSequenceGapDays);

[XTestSeq,YTestSeq,testSequenceDates,testEndIndices] = ...
    buildCausalSequences( ...
    Xscaled(idxTest,:),yEpoch(idxTest),dates(idxTest), ...
    selectedSequenceLength,maxSequenceGapDays);

[~,trainScores] = classify( ...
    lstmModel,XTrainSeq,'MiniBatchSize',miniBatchSize);
[~,validationScores] = classify( ...
    lstmModel,XValidationSeq,'MiniBatchSize',miniBatchSize);
[~,testScores] = classify( ...
    lstmModel,XTestSeq,'MiniBatchSize',miniBatchSize);

trainProbability = extractLSTMPositiveProbability( ...
    trainScores,lstmModel,'manoeuvre');
validationProbability = extractLSTMPositiveProbability( ...
    validationScores,lstmModel,'manoeuvre');
testProbability = extractLSTMPositiveProbability( ...
    testScores,lstmModel,'manoeuvre');

% Full chronological probability vector. The first L-1 epochs of each
% partition and epochs following excessive data gaps remain NaN because no
% valid causal sequence ends there.
manoeuvreProbability = nan(numel(dates),1);

globalTrainIndices = find(idxTrain);
globalValidationIndices = find(idxValidation);
globalTestIndices = find(idxTest);

manoeuvreProbability(globalTrainIndices(trainEndIndices)) = trainProbability;
manoeuvreProbability(globalValidationIndices(validationEndIndices)) = ...
    validationProbability;
manoeuvreProbability(globalTestIndices(testEndIndices)) = testProbability;

isDetectedEpoch = isfinite(manoeuvreProbability) & ...
    manoeuvreProbability >= selectedThreshold;

%% 8) Build events

allDetectedEvents = buildProbabilityEvents( ...
    dates,manoeuvreProbability,selectedThreshold,eventGapDays);

validationDetectedEvents = allDetectedEvents( ...
    allDetectedEvents.PeakDate >= validationStart & ...
    allDetectedEvents.PeakDate <= validationEnd,:);

testDetectedEvents = allDetectedEvents( ...
    allDetectedEvents.PeakDate >= testStart & ...
    allDetectedEvents.PeakDate <= testEnd,:);

trainEpochMetrics = epochMetrics( ...
    YTrainSeq,trainProbability,selectedThreshold);
validationEpochMetrics = epochMetrics( ...
    YValidationSeq,validationProbability,selectedThreshold);

fprintf('\nEpoch-level diagnostics on valid sequence endpoints:\n');
fprintf('  Train F1:      %.4f\n',trainEpochMetrics.F1);
fprintf('  Validation F1: %.4f\n',validationEpochMetrics.F1);
fprintf('  Test sequence endpoints held untouched: %d\n',numel(XTestSeq));

%% 9) Plots

figure('Color','w','Name','BiLSTM manoeuvre probability');
plot(dates,manoeuvreProbability,'k','LineWidth',1);
hold on;
yline(selectedThreshold,'--','Selected threshold','LineWidth',1.1);
xline(validationStart,':','Validation start');
xline(testStart,':','Test start');
grid on; box on;
xlabel('Epoch');
ylabel('P(manoeuvre)');
title(sprintf( ...
    'BiLSTM manoeuvre probability — L=%d, hidden/direction=%d', ...
    selectedSequenceLength,selectedHiddenUnits));

figure('Color','w','Name','LSTM validation threshold selection');
plot(thresholdResults.Threshold,thresholdResults.F1,'-o','LineWidth',1);
hold on;
xline(selectedThreshold,'--','Selected');
grid on; box on;
xlabel('Probability threshold');
ylabel('Validation event F1');
title('BiLSTM threshold selected using 2023 validation only');

figure('Color','w','Name','LSTM hyperparameter tuning');
scatter(hyperparameterResults.SequenceLength, ...
    hyperparameterResults.F1, ...
    40+2*hyperparameterResults.HiddenUnits, ...
    hyperparameterResults.HiddenUnits,'filled');
grid on; box on;
xlabel('Sequence length [epochs]');
ylabel('Best validation event F1');
title('BiLSTM architecture selection');
cb = colorbar;
cb.Label.String = 'Hidden units';

%% 10) Save

save('supervised_LSTM_results.mat', ...
    'lstmModel','ml_features','dates','Xraw','Xscaled', ...
    'usedFeatureNames','trainMedian','trainMAD','badCols', ...
    'officialEvents','yEpoch','eventIndexEpoch', ...
    'idxTrain','idxValidation','idxTest', ...
    'trainEnd','validationStart','validationEnd','testStart','testEnd', ...
    'officialGroupingDays','eventGapDays','matchingToleranceDays', ...
    'numPositiveEpochsPerEvent','maxSequenceGapDays', ...
    'thresholdGrid','thresholdResults','selectedThreshold', ...
    'sequenceLengthGrid','hiddenUnitsGrid','dropoutGrid', ...
    'selectedSequenceLength','selectedHiddenUnits','selectedDropout', ...
    'hyperparameterResults','bestParams', ...
    'maxEpochs','miniBatchSize','initialLearnRate','gradientThreshold', ...
    'normalToPositiveRatio', ...
    'trainSequenceDates','validationSequenceDates','testSequenceDates', ...
    'trainEndIndices','validationEndIndices','testEndIndices', ...
    'YTrainSeq','YValidationSeq','YTestSeq', ...
    'trainProbability','validationProbability','testProbability', ...
    'manoeuvreProbability','isDetectedEpoch', ...
    'allDetectedEvents','validationDetectedEvents','testDetectedEvents', ...
    'trainEpochMetrics','validationEpochMetrics');

writetable(hyperparameterResults,'LSTM_hyperparameter_tuning.csv');
writetable(bestParams,'LSTM_selected_hyperparameters.csv');
writetable(thresholdResults,'LSTM_threshold_tuning.csv');
writetable(allDetectedEvents,'LSTM_all_detected_events.csv');
writetable(validationDetectedEvents,'LSTM_validation_detected_events.csv');
writetable(testDetectedEvents,'LSTM_test_detected_events.csv');

fprintf('\nSaved supervised_LSTM_results.mat\n');
fprintf('Run A05_validate_supervised_LSTM.m for untouched test metrics.\n');

%% =========================================================
% LOCAL FUNCTIONS
% ==========================================================

function [XSeq,YSeq,sequenceDates,endIndices] = buildCausalSequences( ...
        X,y,dates,sequenceLength,maxGapDays)

    n = size(X,1);
    XSeq = cell(0,1);
    YSeq = zeros(0,1);
    sequenceDates = NaT(0,1,'TimeZone','UTC');
    endIndices = zeros(0,1);

    if n < sequenceLength
        return
    end

    accepted = 0;

    for lastIdx = sequenceLength:n
        firstIdx = lastIdx-sequenceLength+1;
        localDates = dates(firstIdx:lastIdx);

        if any(days(diff(localDates)) > maxGapDays)
            continue
        end

        localX = X(firstIdx:lastIdx,:);

        if any(~isfinite(localX),'all')
            continue
        end

        accepted = accepted+1;
        XSeq{accepted,1} = localX'; %#ok<AGROW>
        YSeq(accepted,1) = y(lastIdx); %#ok<AGROW>
        sequenceDates(accepted,1) = dates(lastIdx); %#ok<AGROW>
        endIndices(accepted,1) = lastIdx; %#ok<AGROW>
    end
end

function [XBalanced,YBalanced] = balanceTrainingSequences( ...
        XSeq,YSeq,normalToPositiveRatio)

    positiveIdx = find(YSeq == 1);
    normalIdx = find(YSeq == 0);

    if isempty(positiveIdx)
        error('No positive LSTM training sequences were generated.');
    end

    maxNormal = min(numel(normalIdx), ...
        normalToPositiveRatio*numel(positiveIdx));

    normalIdx = normalIdx(randperm(numel(normalIdx),maxNormal));
    selected = [positiveIdx; normalIdx];
    selected = selected(randperm(numel(selected)));

    XBalanced = XSeq(selected);
    YBalanced = YSeq(selected);
end

function thresholdResults = evaluateThresholds( ...
        dates,p,officialDates,thresholdGrid,eventGapDays, ...
        toleranceDays,periodStart,periodEnd)

    n = numel(thresholdGrid);
    TP = zeros(n,1);
    FP = zeros(n,1);
    FN = zeros(n,1);
    Precision = zeros(n,1);
    Recall = zeros(n,1);
    F1 = zeros(n,1);
    FalseAlarmsPerYear = zeros(n,1);

    durationYears = max(years(periodEnd-periodStart),eps);

    for i = 1:n
        detected = buildProbabilityEvents( ...
            dates,p,thresholdGrid(i),eventGapDays);

        [matchedDetected,~,~] = matchManeuverEvents( ...
            detected.PeakDate,officialDates,toleranceDays);

        TP(i) = numel(matchedDetected);
        FP(i) = height(detected)-TP(i);
        FN(i) = numel(officialDates)-TP(i);

        Precision(i) = safeDivide(TP(i),TP(i)+FP(i));
        Recall(i) = safeDivide(TP(i),TP(i)+FN(i));
        F1(i) = safeDivide( ...
            2*Precision(i)*Recall(i),Precision(i)+Recall(i));
        FalseAlarmsPerYear(i) = FP(i)/durationYears;
    end

    thresholdResults = table( ...
        thresholdGrid,TP,FP,FN,Precision,Recall,F1,FalseAlarmsPerYear, ...
        'VariableNames',{'Threshold','TP','FP','FN','Precision', ...
        'Recall','F1','FalseAlarmsPerYear'});
end

function bestRow = selectBestThreshold(T)
    rankT = T;
    rankT.NegativeF1 = -rankT.F1;
    rankT.NegativeRecall = -rankT.Recall;

    [~,order] = sortrows(rankT, ...
        {'NegativeF1','FalseAlarmsPerYear','NegativeRecall','Threshold'}, ...
        {'ascend','ascend','ascend','descend'});

    bestRow = T(order(1),:);
end

function p = extractLSTMPositiveProbability(scores,net,positiveName)
    classNames = string(net.Layers(end).Classes);

    if isempty(classNames)
        error('Could not recover LSTM output class names.');
    end

    positiveColumn = find(classNames == positiveName,1);

    if isempty(positiveColumn)
        error('Positive class %s was not found.',positiveName);
    end

    p = scores(:,positiveColumn);
end

function filePath = locateFile(fileName)
    filePath = which(fileName);
    if isempty(filePath)
        matches = dir(fullfile(pwd,'**',fileName));
        if isempty(matches)
            error('File not found: %s',fileName);
        end
        filePath = fullfile(matches(1).folder,matches(1).name);
    end
end

function burns = readOfficialManeuvers(filePath)
    lines = readlines(filePath);
    startDate = NaT(0,1,'TimeZone','UTC');
    endDate = NaT(0,1,'TimeZone','UTC');
    manoeuvreType = strings(0,1);

    for i = 1:numel(lines)
        line = strtrim(lines(i));
        if strlength(line)==0 || startsWith(line,"#")
            continue
        end

        fields = split(line,",");
        if numel(fields) < 4
            continue
        end

        try
            t0 = datetime(strtrim(fields(1)), ...
                'InputFormat','yyyy-MM-dd HH:mm:ss','TimeZone','UTC');
            t1 = datetime(strtrim(fields(2)), ...
                'InputFormat','yyyy-MM-dd HH:mm:ss','TimeZone','UTC');
        catch
            continue
        end

        rawType = upper(strtrim(fields(4)));
        if startsWith(rawType,"EW")
            type = "EW";
        elseif startsWith(rawType,"NS")
            type = "NS";
        else
            continue
        end

        startDate(end+1,1) = t0; %#ok<AGROW>
        endDate(end+1,1) = t1; %#ok<AGROW>
        manoeuvreType(end+1,1) = type; %#ok<AGROW>
    end

    burns = table(startDate,endDate,manoeuvreType, ...
        'VariableNames',{'start_date','end_date','type'});
    burns = sortrows(burns,'start_date');
end

function events = groupOfficialManeuvers(burns,gapDays)
    if isempty(burns)
        events = table();
        return
    end

    burns = sortrows(burns,'start_date');
    eventID = zeros(height(burns),1);
    currentEvent = 1;
    eventID(1) = currentEvent;

    for i = 2:height(burns)
        temporalGapDays = days( ...
            burns.start_date(i)-burns.end_date(i-1));

        if temporalGapDays > gapDays
            currentEvent = currentEvent+1;
        end

        eventID(i) = currentEvent;
    end

    nEvents = max(eventID);
    EventID = (1:nEvents)';
    event_date = NaT(nEvents,1,'TimeZone','UTC');
    start_date = NaT(nEvents,1,'TimeZone','UTC');
    end_date = NaT(nEvents,1,'TimeZone','UTC');
    type = strings(nEvents,1);
    NumBurns = zeros(nEvents,1);
    NumEW = zeros(nEvents,1);
    NumNS = zeros(nEvents,1);

    for e = 1:nEvents
        idx = eventID == e;
        start_date(e) = min(burns.start_date(idx));
        end_date(e) = max(burns.end_date(idx));
        event_date(e) = start_date(e);
        NumBurns(e) = sum(idx);
        NumEW(e) = sum(burns.type(idx) == "EW");
        NumNS(e) = sum(burns.type(idx) == "NS");

        if NumEW(e)>0 && NumNS(e)>0
            type(e) = "EW+NS";
        elseif NumEW(e)>0
            type(e) = "EW";
        else
            type(e) = "NS";
        end
    end

    events = table(EventID,event_date,start_date,end_date,type, ...
        NumBurns,NumEW,NumNS);
end

function [labels,eventIndex] = createEpochLabels( ...
        dates,eventDates,numPositiveEpochs)

    nEpochs = numel(dates);
    labels = zeros(nEpochs,1);
    eventIndex = zeros(nEpochs,1);

    for e = 1:numel(eventDates)
        firstIdx = find(dates >= eventDates(e),1,'first');

        if isempty(firstIdx)
            continue
        end

        lastIdx = min(firstIdx+numPositiveEpochs-1,nEpochs);
        idx = firstIdx:lastIdx;
        free = eventIndex(idx) == 0;
        idx = idx(free);

        labels(idx) = 1;
        eventIndex(idx) = e;
    end
end

function eventTable = buildProbabilityEvents(dates,p,threshold,gapDays)
    valid = isfinite(p);
    idx = find(valid & p >= threshold);

    if isempty(idx)
        eventTable = table( ...
            zeros(0,1),NaT(0,1,'TimeZone','UTC'), ...
            NaT(0,1,'TimeZone','UTC'),NaT(0,1,'TimeZone','UTC'), ...
            zeros(0,1),zeros(0,1), ...
            'VariableNames',{'EventID','StartDate','EndDate','PeakDate', ...
            'NumPositiveEpochs','PeakProbability'});
        return
    end

    candidateDates = dates(idx);
    newEvent = [true; days(diff(candidateDates)) > gapDays];
    pointEventID = cumsum(newEvent);
    nEvents = max(pointEventID);

    EventID = (1:nEvents)';
    StartDate = NaT(nEvents,1,'TimeZone','UTC');
    EndDate = NaT(nEvents,1,'TimeZone','UTC');
    PeakDate = NaT(nEvents,1,'TimeZone','UTC');
    NumPositiveEpochs = zeros(nEvents,1);
    PeakProbability = zeros(nEvents,1);

    for e = 1:nEvents
        local = pointEventID == e;
        globalIdx = idx(local);
        localP = p(globalIdx);

        StartDate(e) = dates(globalIdx(1));
        EndDate(e) = dates(globalIdx(end));
        NumPositiveEpochs(e) = numel(globalIdx);
        [PeakProbability(e),k] = max(localP);
        PeakDate(e) = dates(globalIdx(k));
    end

    eventTable = table(EventID,StartDate,EndDate,PeakDate, ...
        NumPositiveEpochs,PeakProbability);
end

function [matchedDetected,matchedOfficial,timingErrors] = ...
        matchManeuverEvents(detectedDates,officialDates,toleranceDays)

    matchedDetected = zeros(0,1);
    matchedOfficial = zeros(0,1);
    timingErrors = zeros(0,1);

    if isempty(detectedDates) || isempty(officialDates)
        return
    end

    pairs = zeros(0,3);

    for d = 1:numel(detectedDates)
        delta = days(detectedDates(d)-officialDates);
        valid = find(abs(delta) <= toleranceDays);

        for j = 1:numel(valid)
            pairs(end+1,:) = [abs(delta(valid(j))),d,valid(j)]; %#ok<AGROW>
        end
    end

    if isempty(pairs)
        return
    end

    pairs = sortrows(pairs,1);
    usedD = false(numel(detectedDates),1);
    usedO = false(numel(officialDates),1);

    for i = 1:size(pairs,1)
        d = pairs(i,2);
        o = pairs(i,3);

        if ~usedD(d) && ~usedO(o)
            usedD(d) = true;
            usedO(o) = true;
            matchedDetected(end+1,1) = d; %#ok<AGROW>
            matchedOfficial(end+1,1) = o; %#ok<AGROW>
            timingErrors(end+1,1) = ...
                days(detectedDates(d)-officialDates(o)); %#ok<AGROW>
        end
    end
end

function metrics = epochMetrics(y,p,threshold)
    y = y(:);
    pred = p(:) >= threshold;

    TP = sum(pred & y==1);
    FP = sum(pred & y==0);
    FN = sum(~pred & y==1);
    TN = sum(~pred & y==0);

    precision = safeDivide(TP,TP+FP);
    recall = safeDivide(TP,TP+FN);
    F1 = safeDivide(2*precision*recall,precision+recall);
    balancedAccuracy = 0.5*( ...
        safeDivide(TP,TP+FN)+safeDivide(TN,TN+FP));

    metrics = table(TP,FP,FN,TN,precision,recall,F1,balancedAccuracy);
end

function value = safeDivide(a,b)
    if b == 0
        value = NaN;
    else
        value = a/b;
    end
end
