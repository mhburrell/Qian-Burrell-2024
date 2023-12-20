function countMatrix = calculate_observation(events,states,S,E)
nEvent = E;
nStates = S;
countMatrix = zeros([nStates nStates nEvent]);
for i = 1:numel(states)-1
    currentEvent = events(i+1);
    currentState = states(i);
    nextState = states(i+1);
    countMatrix(currentState,nextState,currentEvent) = countMatrix(currentState,nextState,currentEvent)+1;
end

sumEvents = sum(countMatrix,3);
for j = 1:nEvent
    countMatrix(:,:,j)=countMatrix(:,:,j)./sumEvents;
end
countMatrix(isnan(countMatrix))=0;