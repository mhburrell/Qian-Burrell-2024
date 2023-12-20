function out = quickConcat(varargin)
%QUICKCONCAT Concatenate arrays, transposing inputs if necessary.
% Check inputs
if nargin < 2
    error('quickConcat:NotEnoughInputs', 'Not enough inputs.');
end

% Identify which inputs are row vectors
isRow = cellfun(@(x) isrow(x) || isempty(x), varargin);

% Check that all inputs are either row or column vectors
if ~all(isRow | ~isRow)
    error('quickConcat:InvalidInput', 'All inputs must be either row or column vectors.');
end

% transpose inputs if necessary
if any(~isRow)
    varargin(~isRow) = cellfun(@transpose, varargin(~isRow), 'UniformOutput', false);
end

% concatenate inputs
out = cat(2, varargin{:});

end
