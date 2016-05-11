clear all
close all
depth2cam1 = load('export/depth2cam1.txt');
depth2cam2 = load('export/depth2cam2.txt');
depth2cam3 = load('export/depth2cam3.txt');
rawDepth1 = load('export/depths1.txt');
rawDepth2 = load('export/depths2.txt');
rawDepth3 = load('export/depths3.txt');

% scatter3(depths1(:,1),depths1(:,2),depths1(:,3),5,'o','b')
% hold on
% scatter3(depths2(:,1),depths2(:,2),depths2(:,3),5,'o','r')
% scatter3(depths3(:,1),depths3(:,2),depths3(:,3),5,'o','g')


% scatter3(depth2cam1(:,1),depth2cam1(:,2),depth2cam1(:,3),5,'o','b')
% hold on
% scatter3(depth2cam2(:,1),depth2cam2(:,2),depth2cam2(:,3),5,'o','r')
% scatter3(depth2cam3(:,1),depth2cam3(:,2),depth2cam3(:,3),5,'o','g')

%diff13 = depth2cam1(1:5091,2)-depth2cam2(:,2);

figure(1)
subplot(3,1,1)
plot(rawDepth1(:,2))

subplot(3,1,2)
plot(rawDepth2(:,2))

subplot(3,1,3)
plot(rawDepth3(:,2))

figure(2)

subplot(3,1,1)
plot(depth2cam1(:,2))

subplot(3,1,2)
plot(depth2cam2(:,2))

subplot(3,1,3)
plot(depth2cam3(:,2))


