%% Phase 1 
% Hamed Ajorlou 97101167
load('All_data')
%% extracting features 
stat_Features = [] ;
for i=1:316
    input = x_train(:,:,i);
    for j=1:28
        stat_Features(j,i) = var(input(:,j));
        for k=1:28
            stat_Features(k+28+28*(j-1),i) = corr(input(:,j),input(:,k)) ;
        end
        stat_Features(812+j,i)=sqrt(var(diff(diff(input(:,j)))))*sqrt(var(input(:,j)))/var(diff(input(:,j)));
    end    
end
freq_Features = [] ;
for i=1:316   
    Fs=1000;
    input = x_train(:,:,i) ;
    N = length(input);
    xdft = fft(input);
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/length(input):Fs/2;
    for j=1:28
        N = length(input(:,j));
        xdft = fft(input(:,j));
        xdft = xdft(1:N/2+1);
        psdx = (1/(Fs*N)) * abs(xdft).^2    ;
        psdx(2:end-1) = 2*psdx(2:end-1)     ;
        freq = 0:Fs/length(input(:,j)):Fs/2;
        [a1,b1] = max(psdx);
        freq_Features(j,i)=freq(b1);
        sum1 = 0 ;
        sum2 = 0 ;
        for k=1:26
            sum1 = sum1 + freq(k)*psdx(k) ; 
            sum2 = sum2 + psdx(k) ;
        end
        freq_Features(j+28,i)=sum1/sum2 ;
    end    
end

features=[stat_Features;freq_Features];
% Final extracted features
features=features(:,1:315);
[Normalized,xPS] = mapminmax(features) ;
oneindices = find(y_train(1:315)==1) ;
zeroindices = find(y_train(1:315)==0) ;

%% Fischer
for i=1:896
    u1 = mean(Normalized(i,oneindices)) ;
    S1 = (Normalized(i,oneindices)-u1)*(Normalized(i,oneindices)-u1)' ; 
    u2 = mean(Normalized(i,zeroindices)) ;
    S2 = (Normalized(i,zeroindices)-u2)*(Normalized(i,zeroindices)-u2)' ; 
    Sw = S1+S2 ;
    u0 = mean(Normalized(i,:)) ; 
    Sb = (u1-u0)^2 + (u2-u0)^2 ;
    J(i) = Sb/Sw ;
end
%%
eff_i = find(J > 0.0001690);
eff=Normalized(eff_i,:);
%% MLP
for N=1:20
    for M=1:20
        Accuracy = 0 ;
        for k=1:5
            train_indices = [1:(k-1)*63,k*63+1:315] ;
            valid_indices = (k-1)*63+1:k*63 ;
            Train_X = eff(:,train_indices) ;
            Validation_X = eff(:,valid_indices) ;
            Train_Y = y_train(train_indices) ;
            Validation_Y = y_train(valid_indices) ;
            net = feedforwardnet([M,N]);
            net = train(net,Train_X,Train_Y);
            predicted = net(Validation_X);
            Threshold = 0.5 ;
            predicted = predicted >= Threshold ;
            Accuracy = Accuracy + length(find(predicted==Validation_Y)) ;
        end
        AccMATRIX(N,M)= Accuracy/315;
    end
end
%%
for i=1:20
    for j=1:20
        if(AccMATRIX(i,j)==max(max(AccMATRIX)))
            MM=i
            NN=j
        end
    end
end
%%
stat_Features_t = [] ;
for i=1:100
    input = x_test(:,:,i);
    for j=1:28
        stat_Features_t(j,i) = var(input(:,j));
        for k=1:28
            stat_Features_t(k+28+28*(j-1),i) = corr(input(:,j),input(:,k)) ;
        end
        stat_Features_t(812+j,i)=sqrt(var(diff(diff(input(:,j)))))*sqrt(var(input(:,j)))/var(diff(input(:,j)));
    end    
end
%

freq_Features_t = [] ;
for i=1:100   
    Fs=1000;
    input = x_test(:,:,i) ;
    N = length(input);
    xdft = fft(input);
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/length(input):Fs/2;
    for j=1:28
        N = length(input(:,j));
        xdft = fft(input(:,j));
        xdft = xdft(1:N/2+1);
        psdx = (1/(Fs*N)) * abs(xdft).^2    ;
        psdx(2:end-1) = 2*psdx(2:end-1)     ;
        freq = 0:Fs/length(input(:,j)):Fs/2;
        [a1,b1] = max(psdx);
        freq_Features_t(j,i)=freq(b1);
        sum1 = 0 ;
        sum2 = 0 ;
        for k=1:26
            sum1 = sum1 + freq(k)*psdx(k) ; 
            sum2 = sum2 + psdx(k) ;
        end
        freq_Features_t(j+28,i)=sum1/sum2 ;
    end    
end

features_t=[stat_Features_t;freq_Features_t];
[Normalized_Test,xPS] = mapminmax(features_t) ;
eff_t=Normalized_Test(eff_i,:);

%%
Train_X = eff ;
Train_Y = y_train(1:315) ;
Test_X = eff_t ;

net = feedforwardnet([MM,NN]);
net = train(net,Train_X,Train_Y);
view(net)
predicted = net(Test_X);
Threshold = 0.5 ;
% Best labels based on MLP network
predicted = predicted >= Threshold ;

%% RBF
y_train=y_train(1:315);
spq = [.1,.5,.9,1.5,2] ;
P = [5,10,15,20,25] ;
for s = 1:5
    spread = spq(s) ;
    for n = 1:5 
        Maxnumber = P(n) ;
        Accuracy = 0 ;
        for k=1:5
            train_indices = [1:(k-1)*63,k*63+1:315] ;
            valid_indices = (k-1)*63+1:k*63 ;
            Train_X = eff(:,train_indices) ;
            Validation_X = eff(:,valid_indices) ;
            Train_Y = y_train(train_indices) ;
            Validation_Y = y_train(valid_indices) ;
            net = newrb(Train_X,Train_Y,10^-5,spread,Maxnumber) ;
            predicted = net(Validation_X);
            Threshold = 0.5 ;
            predicted = predicted >= Threshold ;
            Accuracy = Accuracy + length(find(predicted==Validation_Y)) ;
        end
        ACCMatrix(s,n) = Accuracy/315 ;
    end
end
%% Test
for i=1:5
    for j=1:5
        if(ACCMatrix(i,j)==max(max(ACCMatrix)))
            MM=i
            NN=j
        end
    end
end%%
%%
spread = spq(MM) ; 
Maxnumber = P(NN) ; 

Train_X = eff ;
Train_Y = y_train(1:315) ;
Test_X = eff_t ;

net = newrb(Train_X,Train_Y,10^-5,spread,Maxnumber) ;
predicted = net(Test_X);
Threshold = 0.5 ;
% Best labels based on RBF network
predicted = predicted >= Threshold ;

