%read the data. Class labels have been replaced by 1,2,3
rawData = dlmread('raw.data');
disp('raw data:');
disp(rawData);

%shuffle data
shuffledData = rawData(randperm(size(rawData,1)),:);
disp('shuffled data:');
disp(shuffledData);

%output shuffled data
dlmwrite('shuffledData.data',shuffledData,'delimiter',' ','precision',16);