CSVPATH = '/Users/gakki/Dropbox/thesis/surface_flow_sort.csv';


T = readtable(CSVPATH);
scatter(T.x_coord(1:end/2),T.y_coord(1:end/2),'.r');
hold on;

scatter(T.x_coord(end/2:end),T.y_coord(end/2:end),'.g');
s
axis equal