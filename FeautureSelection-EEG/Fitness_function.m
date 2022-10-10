load('All_data')
%%
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
[t,w]=sort(J);

best_fifty_index=w(847:896);
best_fifty=Normalized(best_fifty_index,:)
save('best_fifty')
y_t=y_train(1:315);
save('y_t')
%%
fitness=@(x) fisher(x);
function J2=fisher(x)
    load('best_fifty');
    load('y_t');
    oneee=find(x==1);
    feat=[];
        for i=oneee
            feat=[feat;best_fifty(:,i)];
        end
    Sw=0;
    sb=0;
    S1=0;
    S2=0;
    feat=feat';
    oneindices = find(y_t==1) ;
    zeroindices = find(y_t==0) ;
    u1 = sum(feat(:,oneindices))/length(oneindices) 
    for i=oneindices
            S1=S1+transpose(feat(:,i)-u1)*(feat(:,i)-u1);
    end
    u2 = sum(feat(:,zeroindices))/length(zeroindices)
    for i=zeroindices
            S2=S2+transpose(feat(:,i)-u2)*(feat(:,i)-u2);
    end
    S1=S1/length(oneindices);
    S2=S2/length(zeroindices);
    Sw = S1+S2 ;
    u0 = sum(feat)/(length(oneindices)+length(zeroindices));  
    sb = transpose(u1-u0)*(u1-u0)+transpose(u2-u0)*(u2-u0);
    J2 = -(trace(sb)/trace(Sw)) ;
end
