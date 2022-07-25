% Example script for running Paninski segmentation and generating figures
% Add definition for ConcatenateSavefasts

% addpath('C:\Users\qstate.analytics\Documents\DataPipeline\UsefulFunctions', '-end');
addpath('C:\Users\mtorkashvand\Documents\MATLAB\Analysis\UsefulMatlabFunctions', '-end');

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Paths of interest for "new" results
basePath = 'C:\Users\mtorkashvand\workspace\Mahdi\trefide\results\';
qsmPath = fullfile(basePath, 'qsm'); 
% NOTE - need to already have the "PCA/ICA" savefasts downloaded at this
% location.
% NOTE - be aware of the naming, so that the previous savefast file names
% do not collide. The "AnalyzeMovie" loop below adds a suffix of
% ".paninski.savefast" which should probably be fine.
savefastPath = fullfile(basePath, 'savefast');
figuresPath = fullfile(basePath, 'figures');
masksPath = fullfile(basePath, 'cell_masks');
featureTablePath = fullfile(basePath, 'feature_table');

mkdir(basePath);
mkdir(qsmPath);
mkdir(savefastPath);
mkdir(figuresPath);
mkdir(masksPath);
mkdir(featureTablePath);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Make results using "new" analysis
% analyzeMovies( qsmPath , savefastPath ); % TODO - turn this on if
% desired!
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Make figures using "new" analysis
% First, save the colorized source masks
savefasts = FindFiles( savefastPath, '.savefast' );
close all
for k = 1:numel(savefasts)
    savefastFullPath = char(savefasts(k));
    LoadFast( savefastFullPath ); % makes `analysis` available
    [ ~, savefastName, ~ ] = fileparts( savefastFullPath );
    [ im, hitTestIm, cMap ] = VisualizeSourceImages( analysis, 'colorSaturation', 5 );
    imshow(im, 'InitialMagnification', 'fit');
    savefig(fullfile(masksPath, [savefastName, '.masks.fig']));
end

% Next, run normal figure generation
ConcatenateSavefasts( savefastPath );
LoadFast( fullfile( savefastPath, 'Analyses') );
conditionLocations = { ...
    {'metadata', 'experiment', 'genotype'}, ...
    {'options', 'segmentMethod'} ...
};
conditionTypes = { ...
    'independent', ...
    'independent' ...
};
    
[~, ~, ~, ~, options, featureTable] = AnalyzeProject( ...
            'projectName', 'UCB', ...
            'analysisObject', analyses, ...
            'figuresPath', figuresPath, ...
            'makeSuccinctFigures', false, ...
            'doSignificanceTesting', false, ...
            'conditionLocations', conditionLocations, ...
            'conditionTypes', conditionTypes); %, ...
        %    'fixAnalysesFunc', @UcbTsc2Hts_FixFunc, ...
        %    'colorMap', @UcbTsc2Hts_ColorMap, ...
        %    'orderRule', @UcbTsc2Hts_OrderRule );

% TODO - currently saves nothing?
SaveFast(featureTablePath, 'featureTable');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% Loop over the movies and analyze each one
function analyzeMovies(qsmDir, outputDir)
    movies = FindFiles(qsmDir, '.qsm');
    for k = 1:numel(movies)
        movieFullPath = char(movies(k));
        [~, movieName, ~] = fileparts( movieFullPath );
        movie = LoadMovie( movieFullPath );
        tic
        analysis = AnalyzeMovie( movie, 'segmentMethod', 'Paninski', 'chopPixels', false, 'maxRegionPixels', 64000 );
        analysis.metadata.analysisDuration = toc;
        SaveFast(fullfile(outputDir, [movieName, '.paninski', '.savefast']), 'analysis');
    end
end
