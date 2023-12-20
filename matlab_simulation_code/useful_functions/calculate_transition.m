function p=calculate_transition(x,S)
m = S;
n = numel(x);
y = zeros(m,1);
p = zeros(m,m);
for k=1:n-1
    y(x(k)) = y(x(k)) + 1;
    p(x(k),x(k+1)) = p(x(k),x(k+1)) + 1;
end
p = bsxfun(@rdivide,p,y); p(isnan(p)) = 0;