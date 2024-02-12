function ks_etdrk4(K,Llx,nummodes)

KT = 2*K;
Xmesh = linspace(-Llx,Llx,KT+1)';

Dx = pi/Llx*[0:K-1 0 -K+1:-1]';
Dxt = -.5*1j*Dx;
Lop = Dx.^2-Dx.^4;
dt = .25;

un = fft(cos(2*pi*Xmesh(1:KT)/Llx) + sin(3*pi*Xmesh(1:KT)/Llx) + sin(pi*Xmesh(1:KT)/Llx));

burn_in_time = 10;
tf = burn_in_time + (Llx/pi)^4;
disp(tf)
nmax = round(tf/dt);
ustore = zeros(nmax, KT);
tvals = [0.];
ustore(1,:) = real(ifft(un));

% Fourier multipliers
E2 = exp(dt*Lop/2);
E = exp(dt*Lop);
M = 16;

% no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity

% construct things on the
LR = dt*Lop(:,ones(M,1)) + r(ones(KT,1),:);

Q = dt*real(mean( (exp(LR/2)-1)./LR ,2) );
f1 = dt*real(mean( (-4 - LR + exp(LR).*(4 - 3*LR + LR.^2))./LR.^3 ,2));
f2 = dt*real(mean( (2 + LR + exp(LR).*(-2 + LR))./LR.^3 ,2));
f3 = dt*real(mean( (-4 - 3*LR - LR.^2 + exp(LR).*(4 - LR))./LR.^3 ,2));
tic 

for jj=1:nmax
    
    Nu = Dxt.*fft( real(ifft(un)).^2 );
    
    a = E2.*un + Q.*Nu;
    Na = Dxt.*fft( real(ifft(a)).^2 );
    
    b = E2.*un + Q.*Na;
    Nb = Dxt.*fft( real(ifft(b)).^2 );
    
    c = E2.*a + Q.*(2*Nb - Nu);
    Nc = Dxt.*fft( real(ifft(c)).^2 );
    
    un = E.*un + Nu.*f1 + 2*(Na + Nb).*f2 + Nc.*f3;   
    tcurr = dt *jj;
    tvals = [tvals tcurr];
    ustore(jj+1, :) = real(ifft(un));
end

[u, s, v] = svd(ustore(burn_in_time:end, :)');
vt = v';        
smax = max(diag(s(1:nummodes,1:nummodes)));
svals = diag(s(1:nummodes,1:nummodes))/smax;            
time_coeffs = s(1:nummodes,1:nummodes)*vt(1:nummodes,:);   
modes = u(:,1:nummodes);            
energy = trace(s(1:nummodes,1:nummodes))/sum(diag(s));

disp(energy)

toc
%save('time_coeffs.mat','time_coeffs','-v7.3')
save('time_coeffs_small.mat','time_coeffs')
save('modes_small.mat','modes')
%save('ksdata_small.mat', ustore)

%[acttsteps, ~] = size(ufin);
%tsteps = min([acttsteps,length(tvals)]);

skp = 1;

figure(1)
surf(Xmesh(1:KT),tvals(1:skp:nmax),ustore(1:skp:nmax,:), 'LineStyle', 'none')

figure(5)
plot(log10(svals))
