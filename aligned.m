function [S,T] = aligned(Z,c,target_view)
numview = length(Z);
S = Z{target_view};
T{target_view} = eye(size(S,1));
for nv = 1:numview
    if nv ~= target_view
        K = Z{target_view}*Z{nv}';  %%% Ò»½×ÏàËÆ¶È
        S1 = Z{target_view}* Z{target_view}';
        S2 = Z{nv}* Z{nv}';
        T{nv} = DSPFP(S1,S2,K,c);
        S = S+T{nv}*Z{nv};
    end
end
S = S/nv;