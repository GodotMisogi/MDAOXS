CSVPATH = '/Users/gakki/Dropbox/thesis/surface_flow_sort.csv';


T = readtable(CSVPATH);
B = sortrows(T,'x_coord');
[nrow,ncol] = size(T);
newID = [1:nrow]';
B.nodeID = newID;
C = table2array(B,'Format','%f32');
E = 1.;
L = zeros(nrow,1);
I = 0.1;
K = zeros(2*nrow,2*nrow);
for element_ID = 1:nrow-1
    i = element_ID;
    j = element_ID+1;
    x1 = C(element_ID,2);
    y1 = C(element_ID,3);
    x2 = C(element_ID+1,2);
    y2 = C(element_ID+1,3);
    L(element_ID) = abs(x2-x1);
    k = BeamElementStiffness(E,I,L(element_ID));
    K = BeamAssemble(K, k, i, j);
end
x1 = C(nrow,2);
x2 = C(1,2);
L(nrow) = abs(x2-x1);
i = nrow;
j = 1;
k = BeamElementStiffness(E,I,L(nrow));
K = BeamAssemble(K, k, i, j);

%%
CSVPATH = '/Users/gakki/Dropbox/thesis/surface_flow_sort.csv';


T = readtable(CSVPATH);
B = sortrows(T,'x_coord');
[nrow,ncol] = size(T);
newID = [1:nrow]';
B.nodeID = newID;
C = table2array(B,'Format','%f32');
E = 1.;
L = zeros(nrow,1);
I = 0.1;
K = zeros(2*nrow,2*nrow);
for element_ID = 1:nrow-1
    i = element_ID;
    j = element_ID+1;
    x1 = C(element_ID,2);
    y1 = C(element_ID,3);
    x2 = C(element_ID+1,2);
    y2 = C(element_ID+1,3);
    L(element_ID) = abs(x2-x1);
    k = BeamElementStiffness(E,I,L(element_ID));
    K = BeamAssemble(K, k, i, j);
end
x1 = C(nrow,2);
x2 = C(1,2);
L(nrow) = abs(x2-x1);
i = nrow;
j = 1;
k = BeamElementStiffness(E,I,L(nrow));
K = BeamAssemble(K, k, i, j);

%%
newK = K(3:end-2,3:end-2);
