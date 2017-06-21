
% this version has no stopping criterion
clear all

% Y is the observable states, generated randomly.
load('Y_obs.mat');
Y=Y_obs;

plot(Y);


size_data=size(Y);
p=size_data(2);
N=size_data(1);



iterate=100; 
N_states=4;
T=200; 


N=N/T;

c=diag(diag(cov(Y)));

mean=randn(N_states,p)*sqrtm(c)+ones(N_states,1)*mean(Y);

% prior, Pri & transition prob matrix, trs_P
% https://www.youtube.com/watch?v=9yl4XGp5OEg
% 6:44
Pri=rand(1,N_states);
Pri=Pri/sum(Pri);

trs_P=rand(N_states);

tZ_=zeros(size(trs_P(:,1)));

        for i=1:length(trs_P(1,:))
          tZ_=tZ_+trs_P(:,i);
        end
        
tZ=zeros(size(trs_P));

        for i=1:length(trs_P(1,:))
          tZ(:,i)=trs_P(:,i)./tZ_;
        end
        trs_P = tZ;

coll_likelihood=[];
likelihood=0;




B=zeros(T,N_states);
k1=(2*pi)^(-p/2);
alpha = zeros(T,N_states);
beta = alpha;
gamma = beta;
for cycle=1:iterate
  
  % forward backword portion
  % in the bigger scheme of things of EM, it is E/Estimation portion
  % https://www.youtube.com/watch?v=9yl4XGp5OEg
  % 26:36
  
  Fuse=[];
  Fusesum=zeros(1,N_states);
  s_norm=zeros(T,1);
  
  % X_hid, hidden states
  X_hid=zeros(T-1,N_states*N_states);
  
  for n=1:N
    
    norm=zeros(T,1);
    iCov=inv(c);
    k2=k1/sqrt(det(c));
    
    % emission probabilities, emm_P
    
    for i=1:T
      for l=1:N_states
        d=mean(l,:)-Y((n-1)*T+i,:);
        emm_P(i,l)=k2*exp(-0.5*d*iCov*d');
      end;
    end;
    
    % Forward Inference
    % https://www.youtube.com/watch?v=9yl4XGp5OEg
    % 13:01
    
    % alpha_t+1(X_hid_t+1) = SUM_X_hid_t[alpha_t(X_hid_t)*trs_P*emm_P]
    
    for i=1:T
       
        %initial calculation for forward-alpha
      if(i==1)
          alpha(i,:)=Pri.*emm_P(i,:);
      end  
      
        %next calculation for forward-alpha
      if(i~=1)
      alpha(i,:)=(alpha(i-1,:)*trs_P).*emm_P(i,:);
      end
      
      norm(i)=sum(alpha(i,:));
      alpha(i,:)=alpha(i,:)/norm(i);
      
    end;
    
    % Forward Inference
    % https://www.youtube.com/watch?v=9yl4XGp5OEg
    % 16:40
    beta(T,:)=ones(1,N_states)/norm(T);
    for i=T-1:-1:1
      beta(i,:)=(beta(i+1,:).*emm_P(i+1,:))*(trs_P')/norm(i); 
    end;
    
    % Fusing forward and backword
    % https://www.youtube.com/watch?v=9yl4XGp5OEg
    % 20:30
    alpha_beta=(alpha.*beta); 
    
    tZ_=zeros(size(alpha_beta(:,1)));

        for i=1:length(alpha_beta(1,:))
          tZ_=tZ_+alpha_beta(:,i);
        end

    
    tZ=zeros(size(alpha_beta));

        for i=1:length(alpha_beta(1,:))
          tZ(:,i)=alpha_beta(:,i)./tZ_;
        end
        alpha_beta = tZ;
    
    alpha_betasum=sum(alpha_beta);
    
    x_temp=zeros(T-1,N_states*N_states);
    for i=1:T-1
      t=trs_P.*( alpha(i,:)' * (beta(i+1,:).*emm_P(i+1,:)));
      x_temp(i,:)=t(:)'/sum(t(:));
    end;
    
    s_norm=s_norm+log(norm);
    Fuse=[Fuse; alpha_beta];
    Fusesum=Fusesum+alpha_betasum;
    X_hid=X_hid+x_temp;

  end;
  
  % in the bigger scheme of things of EM, it is M/Maximation portion
  % https://www.youtube.com/watch?v=9yl4XGp5OEg
  % 26:36
  
  % outputs
  mean=zeros(N_states,p);
  mean=Fuse'*Y;

  FusesumT=Fusesum';
  
  tZ_=zeros(size(FusesumT(:,1)));

        for i=1:length(FusesumT(1,:))
          tZ_=tZ_+FusesumT(:,i);
        end

    
  tZ=zeros(size(mean));

        for i=1:length(mean(1,:))
          tZ(:,i)=mean(:,i)./tZ_;
        end
        mean = tZ;
  
  % Transition Probability Matrix, trs_P
 
  
   X_hidT=X_hid';
   tZ_=zeros(size(X_hidT(:,1)));

        for i=1:length(X_hidT(1,:))
          tZ_=tZ_+X_hidT(:,i);
        end
      sxi=  tZ_;
        
  sxi=reshape(sxi,N_states,N_states);
  
   tZ_=zeros(size(sxi(:,1)));

        for i=1:length(sxi(1,:))
          tZ_=tZ_+sxi(:,i);
        end

    
    tZ=zeros(size(sxi));

        for i=1:length(sxi(1,:))
          tZ(:,i)=sxi(:,i)./tZ_;
        end
        trs_P = tZ;
  
  % Priors, Pri
  Pri=zeros(1,N_states);
  for i=1:N
    Pri=Pri+Fuse((i-1)*T+1,:);
  end
  Pri=Pri/N;
  
  
  c=zeros(p,p);
  for l=1:N_states
    d=(Y-ones(T*N,1)*mean(l,:));
    Z=zeros(size(d));

    for i=1:length(d(1,:))
       Z(:,i)=d(:,i).*Fuse(:,l);
    end
    res=Z';
    c=c+res*d;
  end;
  c=c/(sum(Fusesum));
  
  temp_likelihood=likelihood;
  likelihood=sum(s_norm);
  coll_likelihood=[coll_likelihood likelihood];
  fprintf('likelihood = %f \n',likelihood);  
  
end
