function [y1,y2] = governingEqn(x1,x2)
y1 = 1./(2+x1.*x2.^2).*x2^2.*2*sin(x1)
y2 = 1./(1+x1.*x2.^2).*sin(x1)