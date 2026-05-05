function [Result] = Laurens_Angelaki_2017_Kalman_Model(t, Omega, F, Omega_u, A_u, dt, param, Omega_vision)

% Inputs
% t: time (nx1 vector)
% Omega: real head angular velocity (nx1 vector)
% F: real GIA (nx1 vector). Fill with NaN to simulate rotations around an
% earth-vertical axis.
% Omega_u: angular head velocity motor command (nx1 vector)
% Acc_u: linear head acceleration motor command (nx1 vector)
% dt: time step
% param: matrix of parameters (see below)
% Vis: visual head velocity signals (optional) (nx1 vector); omit it or set
% it to NaN to simulate rotations in darkness

% Parameters
% dt=0.01;
% param.so = 40*pi/180;
% param.sa=0.3;
% param.sv=10*pi/180;
% param.tau=4;
% param.sf=0.002;
% param.svis = 7*pi/180;
% param.sensory_noise = 0 ;

if ~exist('Omega_vision','var'), Omega_vision = [] ; end

k = 1-dt/param.tau ;
k1 = param.tau/(param.tau+dt) ;
k2 = dt/(param.tau+dt) ;

if isempty(Omega_vision)
    H = [1 -1 0 0;0 0 1 1] ;
    R = [param.sv^2 0;0 param.sf^2] ;
elseif isnan(Omega_vision(1))
    H = [1 -1 0 0;0 0 1 1;1 0 0 0] ;
    R = [param.sv^2 0 0;0 param.sf^2 0;0 0 10000^2] ;
else
    H = [1 -1 0 0;0 0 1 1;1 0 0 0] ;
    R = [param.sv^2 0 0;0 param.sf^2 0;0 0 param.svis^2] ;
end

x = zeros(4,1) ; 
v = 0 ; ht_1 = 0 ; g= 0; % This will be used to simulate the motion of the head based on the model's inputs

P = zeros(4,4) ;
[m,n] = size(t) ; n = max([n m]) ;
if isnan(F(1)), sw = 0;F(1)=0;else sw = 1;end

D = [0 0 0 0;...
    0 k1 0 0;...
    0 0 1 0;...
    0 0 0 0] ;

M = [1     0;...
    k2     0;...
    sw*dt  0;...
    0      1];
E=M ;

Q = E*[param.so^2 0;0 param.sa^2]*E' ;

% This is to initialize the Kalman filter matrix K at its steady-state value
for i = 1:500
    Pm = D*P*(D') + Q ;
    K = Pm*(H')*((H*Pm*(H')+R)^(-1)) ;
    P = (eye(size(K*H)) - K*H)*Pm ;
end


for i = 1:n
    
    if isnan(F(i)), sw = 0;F(i)=0;else sw = 1;end
    D = [0 0 0 0;...
        0 k1 0 0;...
        0 0 1 0;...
        0 0 0 0] ;
    
    M = [1     0;...
        k2     0;...
        sw*dt  0;...
        0      1];

    Xu = [Omega_u(i); A_u(i)] ;
    
    % Compute the real motion of the head
    d_omega = (Omega(i)-ht_1)/dt ; ht_1 = Omega(i) ; v = (1-dt/param.tau)*v + (param.tau/(param.tau+dt))*d_omega*dt ;
    g = g+sw*Omega(i)*dt ; a = F(i)-g ;
    
    if isempty(Omega_vision)
        z = [v;F(i)] ;
    else
        if isnan(Omega_vision(i))
            z = [v;F(i);0] ;
            R = [param.sv^2 0 0;0 param.sf^2 0;0 0 10000^2] ;
        else
            z = [v;F(i);Omega_vision(i)] ;
            R = [param.sv^2 0 0;0 param.sf^2 0;0 0 param.svis^2] ;
        end
    end
    
    if param.sensory_noise
        z(1)=z(1)+randn()*param.sv ;
        z(2)=z(2)+randn()*param.sf ;
        if size(z,1)>2
            z(3)=z(3)+randn()*param.svis ;
        end
    end
    
    % This is to update Q in case the switch sw is changed during the
    % simulation
    M = [1     0;...
         k2     0;...
         sw*dt  0;...
         0      1];
    E=M ;
    Q = E*[param.so^2 0;0 param.sa^2]*E' ;

    
    xm = D*x + M*Xu ;
    Pm = D*P*(D') + Q ;
    K = Pm*(H')*((H*Pm*(H')+R)^(-1)) ;
    P = (eye(size(K*H)) - K*H)*Pm ;    
    x = xm + K*(z-H*xm) ;
    
    Result(i).time = t(i) ;  
    Result(i).X = [Omega(i);Omega(i)-v;g;a] ;
    Result(i).Xf = x ;
    Result(i).Xm = xm ;
    Result(i).K = K ;
    Result(i).Zp = H*xm;
    Result(i).Z = z ;    
    Result(i).delta_S = (z-H*xm) ;
    
    scaled_K = K ; scaled_K([2 3  6 7])=scaled_K([2 3  6 7])/dt ; % See Suppl Methods: Kalman Feedback Gains
    Result(i).Vestibular_Feedback = scaled_K(:,1)*Result(i).delta_S(1) ;
    Result(i).Otolith_Feedback = scaled_K(:,2)*Result(i).delta_S(2) ;
    if size(scaled_K,2)==3
        Result(i).Visual_Feedback = scaled_K(:,3)*Result(i).delta_S(3) ;
    else
        Result(i).Visual_Feedback = scaled_K(:,2)*NaN ;
    end
end