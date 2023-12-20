function outTable  = addTableVariable(inTable,colName,addVal)
[l,~]=size(inTable);
if isnumeric(addVal)
    inTable.(colName)=repmat(addVal,[l,1]);
else
    temp_cells= cell(l,1);
    temp_cells(:) = {addVal};
    inTable.(colName) = temp_cells;
end
outTable = inTable;