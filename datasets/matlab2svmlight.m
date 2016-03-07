%%
%% Convert data files from MATLAB format to SVMlight format
%%

File = cell(10, 1);
File{1} = 'datasets/ionosphere_train';
File{2} = 'datasets/ionosphere_test';
File{3} = 'datasets/isolet_train';
File{4} = 'datasets/isolet_test';
File{5} = 'datasets/liver_train';
File{6} = 'datasets/liver_test';
File{7} = 'datasets/mnist_train';
File{8} = 'datasets/mnist_test';
File{9} = 'datasets/mushroom_train';
File{10} = 'datasets/mushroom_test';

for fid = 1:10
  load(File{fid});
  X = full(X);
  fout = fopen(strcat(File{fid}, '.dat'), 'w');
  for i = 1:size(X, 1)
    fprintf(fout, '%i', Y(i) * 2 - 1);
    for j = 1:size(X, 2)
      fprintf(fout, ' %i:%f', j, X(i, j));
    end
    fprintf(fout, '\n');
  end
  fclose(fout);
end
