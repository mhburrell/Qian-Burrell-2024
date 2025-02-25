function robust_parquetwrite(filename, data, varargin)
    maxAttempts = 10;
    for attempt = 1:maxAttempts
        try
            parquetwrite(filename, data, varargin{:});
            return;
        catch ME
            if attempt == maxAttempts
                fprintf('Failed to write to file %s after %d attempts. Error: %s\n', filename, maxAttempts, ME.message);
                rethrow(ME);
            else
                fprintf('Attempt %d to write to file %s failed. Retrying...\n', attempt, filename);
                pause(10); % Optional: wait for 1 second before retrying
            end
        end
    end
end