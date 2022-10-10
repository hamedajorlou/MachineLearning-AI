
load('All_data')
%% features obtained using GA
final=[0.0 1.0 0.0	1.0	0.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	0.0	1.0	1.0	0.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	0.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0]

featt=find(final==1);
eff=Normalized(featt,:);
%%
for N=1:20
    for M=1:20
        Accuracy = 0 ;
        for k=1:5
            train_indices = [1:(k-1)*63,k*63+1:315] ;
            validation_indices = (k-1)*63+1:k*63 ;
            Train_X = eff(:,train_indices) ;
            Validation_X = eff(:,validation_indices) ;
            Train_Y = y_train(train_indices) ;
            Validation_Y = y_train(validation_indices) ;
            net = feedforwardnet([M,N]);
            net = train(net,Train_X,Train_Y);
            predicted = net(Validation_X);
            Threshold = 0.5 ;
            predicted = predicted >= Threshold ;
            Accuracy = Accuracy + length(find(predicted==Validation_Y)) ;
        end
        ACCMatrix(N,M)= Accuracy/315;
    end
end
%%
for i=1:20
    for j=1:20
        if(ACCMatrix(i,j)==max(max(ACCMatrix)))
            MM=i
            NN=j
        end
    end
end
%%
stat_Features_t = [] ;
for i=1:100
    NewSig = x_test(:,:,i);
    for j=1:28
        stat_Features_t(j,i) = var(NewSig(:,j));
        for k=1:28
            stat_Features_t(k+28+28*(j-1),i) = corr(NewSig(:,j),NewSig(:,k)) ;
        end
        stat_Features_t(812+j,i)=sqrt(var(diff(diff(NewSig(:,j)))))*sqrt(var(NewSig(:,j)))/var(diff(NewSig(:,j)));
    end    
end
%

freq_Features_t = [] ;
for i=1:100   
    Fs=1000;
    NewSig = x_test(:,:,i) ;
    N = length(NewSig);
    xdft = fft(NewSig);
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/length(NewSig):Fs/2;
    for j=1:28
        N = length(NewSig(:,j));
        xdft = fft(NewSig(:,j));
        xdft = xdft(1:N/2+1);
        psdx = (1/(Fs*N)) * abs(xdft).^2    ;
        psdx(2:end-1) = 2*psdx(2:end-1)     ;
        freq = 0:Fs/length(NewSig(:,j)):Fs/2;
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
eff_t=Normalized_Test(featt,:);

%% MLP
Train_X = eff ;
Train_Y = y_train(1:315) ;
Test_X = eff_t ;

net = feedforwardnet([MM,NN]);
net = train(net,Train_X,Train_Y);
view(net)
predicted = net(Test_X);
Threshold = 0.5 ;
predicted = predicted >= Threshold ;


%% RBF
ACCMatrix=[];
y_train=y_train(1:315);
spq = [.1,.5,.9,1,2,5,7,10] ;
P = [5,10,15,20,25,30,40,100] ;
for s = 1:8
    spread = spq(s) ;
    for n = 1:8 
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
for i=1:8
    for j=1:8
        if(ACCMatrix(i,j)==max(max(ACCMatrix)))
            MM = i
            NN = j
        end
    end
end

%%
spread = spq(MM) ; 
Maxnumber = P(NN) ; 

Train_X = eff ;
Train_Y = y_train(1:315) ;
Test_X = eff_t ;

net = newrb(Train_X,Train_Y,10^-5,spread,Maxnumber) ;
predicted = net(Test_X);
view(net)
Threshold = 0.5 ;
% Best labels based on RBF network
predicted = predicted >= Threshold ;


