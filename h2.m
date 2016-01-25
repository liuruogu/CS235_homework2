%CS235 hw2
%Liuqing(Abbey) Yang, Ruogu Liu
%Dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/

clear;

rawData = dlmread('shuffledData.data');
% disp('shuffled raw data:');
% disp(rawData);

%z-normalize
for k=1:size(rawData,2)-1
    rawData(:,k)=(rawData(:,k)-mean(rawData(:,k)))/std(rawData(:,k));
end
% disp('normalized data:');
% disp(rawData);

%training data. The first half.
trainData = rawData(1:(size(rawData,1)/2),1:size(rawData,2)-1);
% disp('trainData:');
% disp(trainData);

%class labels (Group) of training data 
trainGrp = rawData(1:(size(rawData,1)/2),size(rawData,2));
% disp('train Group:');
% disp(trainGrp);

%test data is the second half.
%class labels of test data
testGrp=rawData(size(rawData,1)/2+1:size(rawData,1),size(rawData,2));
% disp('test Group: ');
% disp(testGrp);

%test data 
testData = rawData(size(rawData,1)/2+1:size(rawData,1),1:size(rawData,2)-1);
% disp('test data: ');
% disp(num2str(testData));

%default rate
%first find all the classes and the number of instances of each class
DataClasses=[];
ClassNames=[];
classLabels=rawData(:,size(rawData,2));
flag=0;
for j=1:size(classLabels,1)
    if size(DataClasses,1)==0
        DataClasses(1,1)=1;
        ClassNames(1,1)=classLabels(1,1);
    else
        for m=1:size(DataClasses,1)
            if(classLabels(j,1)==ClassNames(m,1))
                DataClasses(m,1)=DataClasses(m,1)+1;
                flag=1;
            end
        end
        if flag==0;
            DataClasses(size(DataClasses,1)+1,1)=1;
            ClassNames(size(ClassNames,1)+1,1)=classLabels(j,1);
        end
    end
    flag=0;
end
DominantClass=ClassNames(1,1);
DominantNum=0;
for n=1:size(DataClasses,1)
    if(DataClasses(n,1)>DominantNum)
        DominantClass=ClassNames(n,1);
        DominantNum=DataClasses(n,1);
    end
end
defaultRate=DominantNum/size(rawData,1);
disp('default rate is:');
disp(defaultRate);

% Run the KNN classifer, and predict the class labels for the test data. 
class = knnclassify(testData, trainData, trainGrp, 7);

% The accuracy of knn
equal=0;
for i = 1:length(testGrp)
    if(class(i)==testGrp(i))
    equal = equal + 1;
    %disp('plus 1');
    end    
end
accuracy = equal/length(testGrp);
disp(['The accuracy is: ',num2str(accuracy)]);

%Try removing some features
for t=1:size(rawData,2)-1
    testD=testData;
    trainD=trainData;
    testD(:,t)=[];
    trainD(:,t)=[];
    % Run the KNN classifer, and predict the class labels for the test data. 
    class = knnclassify(testD, trainD, trainGrp, 7);

    % The accuracy of knn
    equal=0;
    for i = 1:length(testGrp)
        if(class(i)==testGrp(i))
        equal = equal + 1;
        %disp('plus 1');
        end    
    end
    accuracy = equal/length(testGrp);
    disp(['The accuracy of removing the ',num2str(t),'th feature is: ',num2str(accuracy)]);
end

%add one column of random number
Add1=randn(size(rawData,1),1);
% disp('add1:');
% disp(Add1);
rawD1=[rawData(:,(1:size(rawData,2)-1)) Add1 rawData(:,size(rawData,2))];
trainD = rawD1(1:(size(rawD1,1)/2),1:size(rawD1,2)-1);
testD = rawD1(size(rawD1,1)/2+1:size(rawD1,1),1:size(rawD1,2)-1);
% Run the KNN classifer, and predict the class labels for the test data. 
class = knnclassify(testD, trainD, trainGrp, 7);

% The accuracy of knn
equal=0;
for i = 1:length(testGrp)
    if(class(i)==testGrp(i))
    equal = equal + 1;
    %disp('plus 1');
    end    
end
accuracy = equal/length(testGrp);
disp(['The accuracy of adding one random column is: ',num2str(accuracy)]);

%add two column
Add2=[Add1 rand(size(rawData,1),1)];
% disp('add2:');
% disp(Add2);
rawD2=[rawData(:,(1:size(rawData,2)-1)) Add2 rawData(:,size(rawData,2))];
trainD = rawD2(1:(size(rawD2,1)/2),1:size(rawD2,2)-1);
testD = rawD2(size(rawD2,1)/2+1:size(rawD2,1),1:size(rawD2,2)-1);
% Run the KNN classifer, and predict the class labels for the test data. 
class = knnclassify(testD, trainD, trainGrp, 7);

% The accuracy of knn
equal=0;
for i = 1:length(testGrp)
    if(class(i)==testGrp(i))
    equal = equal + 1;
    %disp('plus 1');
    end    
end
accuracy = equal/length(testGrp);
disp(['The accuracy of adding two random column is: ',num2str(accuracy)]);

%add four column
Add3=[Add2 rand(size(rawData,1),1) rand(size(rawData,1),1)]; 
% disp('add3');
% disp(Add3);
rawD3=[rawData(:,(1:size(rawData,2)-1)) Add3 rawData(:,size(rawData,2))];
trainD = rawD3(1:(size(rawD3,1)/2),1:size(rawD3,2)-1);
testD = rawD3(size(rawD3,1)/2+1:size(rawD3,1),1:size(rawD3,2)-1);
% Run the KNN classifer, and predict the class labels for the test data. 
class = knnclassify(testD, trainD, trainGrp, 7);

% The accuracy of knn
equal=0;
for i = 1:length(testGrp)
    if(class(i)==testGrp(i))
    equal = equal + 1;
    %disp('plus 1');
    end    
end
accuracy = equal/length(testGrp);
disp(['The accuracy of adding four random column is: ',num2str(accuracy)]);

%add until decrease down to default rate
Addrand=Add3;
count=4;
while accuracy>defaultRate
    %add one more column
Addrand=[Addrand rand(size(rawData,1),1)]; 
% disp('add rand');
% disp(Addrand);
rawDa=[rawData(:,(1:size(rawData,2)-1)) Addrand rawData(:,size(rawData,2))];
trainD = rawDa(1:(size(rawDa,1)/2),1:size(rawDa,2)-1);
testD = rawDa(size(rawDa,1)/2+1:size(rawDa,1),1:size(rawDa,2)-1);
% Run the KNN classifer, and predict the class labels for the test data. 
class = knnclassify(testD, trainD, trainGrp, 7);

% The accuracy of knn
equal=0;
for i = 1:length(testGrp)
    if(class(i)==testGrp(i))
    equal = equal + 1;
    %disp('plus 1');
    end    
end
count=count+1;
accuracy = equal/length(testGrp);
disp(['The accuracy of adding ',num2str(count),' random column(s) is: ',num2str(accuracy)]);
end
