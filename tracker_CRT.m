
%error('Tracker not configured! Please edit the tracker_CRT.m file.'); % Remove this line after proper configuration

% The human readable label for the tracker, used to identify the tracker in reports
% If not set, it will be set to the same value as the identifier.
% It does not have to be unique, but it is best that it is.
tracker_label = 'CRT';

% For Python implementations we have created a handy function that generates the appropritate
% command that will run the python executable and execute the given script that includes your
% tracker implementation.
%
% Please customize the line below by substituting the first argument with the name of the
% script of your tracker (not the .py file but just the name of the script) and also provide the 
% path (or multiple paths) where the tracker sources % are found as the elements of the cell 
% array (second argument).
tracker_command = generate_python_command('vot_run_CRT', {'/home/chkap/workspace/pycharm/conv_reg_vot'});

tracker_interpreter = '/home/chkap/my_programs/anaconda3/envs/tf_work/bin/python';

tracker_linkpath = {'/usr/local/cuda/lib64'}; % A cell array of custom library directories used by the tracker executable (optional)

% tracker_trax = false; % Uncomment to manually disable TraX protocol testing
