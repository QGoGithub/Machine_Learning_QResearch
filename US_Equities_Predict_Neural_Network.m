% We apply neural network to predict US Equity Index %

% Index = Nasdaq

clear all

filename1 = 'nasdaqinputdata.xlsx'; 
filename2 = 'nasdaqtestdata.xlsx'; 
Xtotal = xlsread(filename1);
ytotal = xlsread(filename1);
X = Xtotal(:,7:11);
y = ytotal(:,12);


inputsize = 5;
outputsize = 1;
hiddensize = 5;

yhat = zeros (length(X), 1);
yhat_error = zeros (length(X), 1);

%Weight ans bias matrices

w1 = 2 * rand(inputsize, hiddensize) - 1;
w2 = 2 * rand(hiddensize, outputsize) - 1;
b1 = 2 * rand(1, hiddensize) - 1;
b2 = 2 * rand(1, outputsize) - 1;

%Training the Neural Net using back propagation
epoch = 10000;
for j = 1:epoch
    i = randi([1,length(X)],[1,1]);
    %Feed Forward NN
    z2 = (X(i,:) * w1) + b1;
    a2 = sigmoid(z2);
    z3= (a2 * w2) + b2;
    yhat(i) = sigmoid(z3);

    %cost function
    yhat_error(i) = y(i) - yhat(i); 
    J(j) = 0.5 * sum(yhat_error(i) ^ 2);
    
    del3 = yhat_error(i) .* sigmoidprime(yhat(i));
    dJdW2 = a2' * del3;
    dJdb2 = del3;
    
    a2_error = del3 * w2';
    del2 = a2_error .* sigmoidprime(a2);
    dJdW1 = X(i,:)' * del2;
    dJdb1 = del2;
    
    w2 = w2 + dJdW2;
    w1 = w1 + dJdW1;
    b2 = b2 + dJdb2;
    b1 = b1 + dJdb1;
    
end
experimental = (yhat * 952.020019)+ 4266.839844;
actual = ytotal(:,1);
percent_error = abs(((actual - experimental)./actual) * 100);
mean_error = mean(percent_error)
mean_square_error = mean((yhat_error).^2)

%Testing the neural network

testdata = xlsread(filename2);
Xtest = testdata(:,7:11);
ytest_actual = testdata(:,6);
ytest = zeros (length(Xtest), 1);

for k = 1:52
    z2 = (Xtest(k,:) * w1) + b1;
    a2 = sigmoid(z2);
    z3= (a2 * w2) + b2;
    ytest(k) = sigmoid(z3);
end
ytest = (ytest * 693.180176)+ 4266.839844;
test_percent_error = abs(((ytest_actual - ytest)./ytest_actual) * 100);
mean_test_error = mean(test_percent_error);
