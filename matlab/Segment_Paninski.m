%{ 

Run Paninski Pipeline 
- penalized matrix decomposition (PMD) performs denoising and compression, decomposing movie into the product of two matrices: block-sparse U (whole-field spatial components) and dense V (temporal components).
  - input: movie
  - output: U, V
- NMF with superpixel initialization demixes the movie into spatial and temporal components belonging to distinct cells
  - input: U, V, Yd (Yd is just the product of U and V)
  - output: A (spatial masks of cells), C (time traces of cells), B (background component, basically ignored here)
- To proceed with downstream steps, we use the spatial masks to extract raw traces from the original movie
  - TODO - if the raw movie is not available to be passed in, we might be able to just pass back the spatial masks and let the pipeline construct the traces using these...


SEE ALSO:
paninski_repos/README.md

Example pipeline run command: 

```MATLAB
movie = LoadMovie(...)
analysis = AnalyzeMovie( movie, 'segmentMethod', 'Paninski', 'chopPixels', false, 'maxRegionPixels', 64000 );
```

The basic procedure here is:
1. write input data from MATLAB memory to disk. 
  - NOTE that we operate on the whole movie here, because parallel processing will be handled in Python.
2. run bash script to launch python scripts in Docker containers.
3. python code saves results to disk and returns.
4. load from disk into MATLAB memory
5. return only the spatial masks to SegmentMovie.m. Surrounding code will use this to extract traces.

%}

function source = Segment_Paninski(source, movie2D, options)
  movie3D = reshape(movie2D', 80, 800, []);
  
  % quick-and-dirty flattening of the movie in time dimension
  movie_trace = squeeze(mean(mean(movie3D, 1), 2));
  sigma = std(movie_trace);
  med = median(movie_trace);
  movie3D = movie3D(:,:, abs(movie_trace - med) < 1.5 * sigma);
  figure();
  plot(movie_trace);
  hold on;
  plot(squeeze(mean(mean(movie3D, 1), 2)));

  % TODO - should use a system tmp folder, and create it if not found
  % dockerDir = 'D:\Documents\Data\niklas\matlab_docker_storage';
  dockerDir = 'C:\Users\mtorkashvand\workspace\trefide\data\docker';
  if ~exist(dockerDir, 'dir')
       mkdir(dockerDir);
  end
  pipelineImageName = 'trefide';

  inputFile = 'python_input.mat';

  % TODO - SaveFast fails here but would be nice
  % SaveFast(fullfile(dockerDir, inputFile), 'movie3D');
  save(fullfile(dockerDir, inputFile), 'movie3D', '-v7.3');
  % TODO - change windows dir string to bash style (otherwise get
  % "DDocumentsData...")
  %fakedir = '/d/Documents/Data/niklas/matlab_docker_storage';
  fakedir = '/mnt/c/Users/mtorkashvand/workspace/trefide/data/docker';
  % NOTE - example of providing different values for the Paninski values of interest
  % cut_off_point = '"0.975 0.925"'; % NOTE - spacing and double quotes is important
  % length_cut = '"18 13"';
  % run_command = ['bash run_pipeline.sh -abfko -x ', cut_off_point, ' -y ', length_cut,' -i ', inputFile, ' -d ', fakedir, ' -n dockerTmp -m ', pipelineImageName];
  run_command = ['bash -x run_pipeline.sh -abfko -i ', inputFile, ' -d ', fakedir, ' -n dockerTmp -m ', pipelineImageName];
  disp(run_command);

  [status, cmdout] = system(run_command);
  if status ~= 0
      error('error running docker');
      disp(cmdout);
  end

  A = h5read(fullfile(dockerDir, 'dockerTmp', 'demixing', 'main_results', 'A.h5'), '/A');
  % NOTE - since the input movie here is NOT the "raw" movie, the "raw_traces" that we get back will actually be flattened traces
  % Therefore, instead we can return only masks
  %raw_traces = h5read(fullfile(dockerDir, 'dockerTmp', 'demixing', 'main_results', 'raw_traces.h5'), '/raw_traces');
  source.sourceImage = A;
end
