function [modes, time_coeffs] = pod_decomp(data,nummodes)

    % note, for data it's time x space.  We take a transpose so we don't
    % have to work in such a weird representation.  
    [numbatches, ntsteps, KT] = size(data);
    time_coeffs = zeros(numbatches, nummodes, ntsteps);
    modes = zeros(numbatches, KT, nummodes);

    energies = zeros(numbatches);
    
    for ll = 1:numbatches
        cdata = (squeeze(data(ll,:,:)))'; 
        cdata = cdata - mean(cdata,2);
        [u, s, v] = svd(cdata);
        vt = v';        
        modes(ll,:,:) = u(:, 1:nummodes);
        smax = max(diag(s(1:nummodes,1:nummodes)));
        time_coeffs(ll,:,:) = 1/smax*s(1:nummodes,1:nummodes)*vt(1:nummodes,:);
        energies(ll) = trace(s(1:nummodes,1:nummodes))/sum(diag(s));
    end 

    figure(3)
    plot((energies(1:numbatches-1)))
    