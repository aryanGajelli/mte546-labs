clc; clear; close all;


x = 0:0.1:10;
y = 0:0.1:10;

[X,Y] = meshgrid(x,y);

sensor_x = 1;
sensor_y = 1;
distance_to_sensor = sqrt((X-sensor_x).^2 + (Y-sensor_y).^2);

gaussian_xy = gaussmf(distance_to_sensor,[1 5]);

surf(X,Y,gaussian_xy/sum(gaussian_xy,'all'))
xlabel('X'); ylabel('Y'); title('sensor pos=(1,1), distance$=5$','Interpreter','Latex')
