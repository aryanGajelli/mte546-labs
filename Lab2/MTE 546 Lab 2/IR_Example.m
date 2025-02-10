clc; clear; close all;


x = 0:0.1:10;
y = 0:0.1:10;

[X,Y] = meshgrid(x,y);

gaussian_x = gaussmf(X,[1 4]);

gaussian_y = gaussmf(Y,[1 6]);

gaussian_xy = gaussian_x .* gaussian_y;

subplot(3,1,1)
surf(X,Y,gaussian_x/sum(gaussian_x,'all'))
xlabel('X'); ylabel('Y'); title('IR Sensor 1: $\bar{X}=4$','Interpreter','Latex')

subplot(3,1,2)
surf(X,Y,gaussian_y/sum(gaussian_y,'all'))
xlabel('X'); ylabel('Y'); title('IR Sensor 2: $\bar{Y}=6$','Interpreter','Latex')

subplot(3,1,3)
surf(X,Y,gaussian_xy/sum(gaussian_xy,'all'))
xlabel('X'); ylabel('Y'); title('Fused')